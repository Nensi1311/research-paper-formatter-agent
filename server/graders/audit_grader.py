"""
server/graders/audit_grader.py — Task 3: claim_evidence_audit grader.

FIXES vs original:
  1. TYPE_ALIASES — normalises "mismatch", "value_mismatch" etc → "table_text_mismatch"
     so agent never gets silently zero-scored for using a synonym
  2. thefuzz.fuzz.partial_ratio replaces difflib — handles partial matches better
  3. PARTIAL CREDIT reward — even wrong submissions get gradient signal based on
     numerical proximity + section location. Prevents GRPO zero-advantage collapse.
  4. Bipartite matching unchanged (correct) — but now applied AFTER type normalisation

References: F-beta (Ng 1999), PBRS (Ng et al. 1999), GRPO (arxiv 2512.07478)
Authors: Nensi Pansuriya, Krushna Parmar, Ishita Bhojani
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...corpus import Paper
    from ..reward_shaper import NavigationState

# ── HackNU gem: thefuzz Levenshtein (with difflib fallback) ──────────────────
try:
    from thefuzz import fuzz as _fuzz
    def _text_sim(a: str, b: str) -> float:
        return _fuzz.partial_ratio(str(a).lower(), str(b).lower()) / 100.0
except ImportError:
    import difflib
    def _text_sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

# ── Type normalisation: the silent-zero bug fix ───────────────────────────────
TYPE_ALIASES: dict[str, str] = {
    "mismatch":              "table_text_mismatch",
    "value_mismatch":        "table_text_mismatch",
    "number_mismatch":       "table_text_mismatch",
    "numerical_discrepancy": "table_text_mismatch",
    "table_mismatch":        "table_text_mismatch",
    "claim_mismatch":        "table_text_mismatch",
    "data_mismatch":         "table_text_mismatch",
    "discrepancy":           "table_text_mismatch",
    "inconsistency":         "number_mismatch",
    "contradiction":         "number_mismatch",
    "conflict":              "number_mismatch",
    "missing":               "missing_reference",
    "missing_ref":           "missing_reference",
    "ghost_ref":             "missing_reference",
    "undefined_reference":   "missing_reference",
}

def _norm_type(t: str) -> str:
    """Normalise type string to canonical form."""
    t = str(t).lower().strip().replace(" ", "_").replace("-", "_")
    return TYPE_ALIASES.get(t, t)

BETA_AUDIT      = 1.5   # was 0.5 — changed: β<1 created under-report Nash equilibrium
SIM_THRESHOLD   = 0.35    # lowered from 0.40 now that we have thefuzz
F_WEIGHT        = 0.70
SPEC_WEIGHT     = 0.20
COVERAGE_WEIGHT = 0.10

# ── G3: 4-class SemanticCite reward (arXiv 2511.16198) ───────────────────────
# Replaces binary found/not-found with nuanced match quality.
# Supported(0.90) > PartiallySupported(0.65) > Uncertain(0.40) > Unsupported(0.20)
# This gives GRPO finer gradient signal than binary 0/1.
FOUR_CLASS_REWARDS = {
    "exact":       0.90,   # type + location + claim + table all match
    "partial":     0.65,   # type + location match, claim numerically close
    "wrong_table": 0.30,   # right discrepancy, wrong table cited
    "hallucinated": -0.20, # G7: valid JSON but completely fabricated — composite reward penalty
}

# ── G7: Hallucination detection thresholds ───────────────────────────────────
# Agent submits valid-looking JSON but the numbers don't appear anywhere in paper.
# Penalise to discourage fabrication over abstention.
# Based on arXiv 2509.15557 (Composite Reward anti-hacking) and
# arXiv 2512.19920 (Behavioral Calibration — abstention > confabulation).
#
# v6.2: hallucination threshold raised 0.10 → 0.30 because the smoke v6.1 worst
# rollouts ("abstract claims X, table shows X — mismatch") were scoring just
# above the old floor and getting rewarded for partial credit.  G8 grounding
# already drops their similarity by 0.30×; the threshold raise makes sure they
# can't sneak through.
HALLUCINATION_PENALTY = -0.20
HALLUCINATION_MIN_SIM = 0.30   # was 0.10 (smoke v6.1 false negatives)


def _extract_number(text: str) -> float | None:
    """Extract first numeric value from text."""
    m = re.search(r'(\d+\.?\d*)', str(text))
    return float(m.group(1)) if m else None


# ── G8: Substring grounding (anti-template-copy + anti-self-contradiction) ───
# Smoke v6.1 surfaced two pathologies the matched-bipartite reward couldn't
# catch:
#   (a) self-contradicting submissions:  claim="91.85"  table_value="91.85"
#       → model invents a "mismatch" between two equal numbers
#   (b) hallucinated table values:       table_value not present in any table
#       → model token-substitutes from the few-shot example
# Both look like valid JSON to the bipartite matcher and earned partial credit.
# The grounding penalty multiplies similarity by GROUNDING_PENALTY for any
# submission that fails either check, dropping it below SIM_THRESHOLD.
GROUNDING_PENALTY     = 0.30   # surviving similarity is too low to match
SELF_CONTRADICT_PENALTY = 0.0  # claim_num == table_value_num → fully drop


def _paper_text_blob(paper) -> str:
    """All raw text the agent could legally quote from — sections + table dumps.

    Used by the grounding check.  Cached on the Paper object so repeated
    submissions in the same group don't re-stringify.
    """
    cached = getattr(paper, "_audit_grounding_blob", None)
    if cached is not None:
        return cached
    parts = []
    sections = getattr(paper, "sections", {}) or {}
    for v in sections.values():
        if v:
            parts.append(str(v))
    tables = getattr(paper, "tables", {}) or {}
    for tname, tdata in tables.items():
        try:
            import json as _j
            parts.append(f"=== {tname} ===\n" + _j.dumps(tdata, default=str))
        except Exception:
            parts.append(f"=== {tname} ===\n{tdata}")
    blob = "\n".join(parts).lower()
    try:
        paper._audit_grounding_blob = blob   # type: ignore[attr-defined]
    except Exception:
        pass
    return blob


def _grounding_score(sub: dict, paper) -> float:
    """Return a multiplier in [0, 1] reflecting how grounded the submission is.

    1.0  → claim's number AND table_value's number both appear in paper text,
           AND they differ from each other (real discrepancy)
    0.30 → one of the numbers can't be found (hallucinated half) → GROUNDING_PENALTY
    0.0  → claim_num == table_value_num (self-contradicting "mismatch")
           → SELF_CONTRADICT_PENALTY
    """
    blob = _paper_text_blob(paper)
    if not blob:
        return 1.0   # can't verify → don't penalise
    claim_str = _safe_str(sub.get("claim") or sub.get("text_claim"))
    tv_str    = _safe_str(sub.get("table_value"))
    cn = _extract_number(claim_str)
    tn = _extract_number(tv_str)
    # Self-contradicting submission: same number on both sides.
    if cn is not None and tn is not None and abs(cn - tn) < 1e-6:
        return SELF_CONTRADICT_PENALTY
    # Substring presence (cheap; works because the paper text is already lower).
    claim_ok = (not claim_str) or any(
        tok and tok in blob for tok in [
            claim_str.lower().strip(),
            (str(cn) if cn is not None else ""),
        ]
    )
    tv_ok = (not tv_str) or (tv_str.lower().strip() in blob) or (
        tn is not None and (str(tn) in blob or f"{tn:g}" in blob)
    )
    if claim_ok and tv_ok:
        return 1.0
    return GROUNDING_PENALTY


def _safe_str(v) -> str:
    """Coerce ANY model-supplied JSON value to a string before .strip() / .lower().

    The agent occasionally emits nested objects like
        {"table_value": {"GLUE": 90.95}}
    instead of strings.  Calling .strip() on that explodes — see the v6 Colab
    smoke crash at audit_grader.py L300.  This helper replaces every direct
    .strip()/.lower() on agent-supplied dict values throughout the grader.
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, dict):
        # Flatten to "k1=v1; k2=v2" so downstream fuzzy match still has signal.
        try:
            return "; ".join(f"{k}={v[k]}" for k in v)
        except Exception:
            return str(v)
    if isinstance(v, (list, tuple)):
        return ", ".join(_safe_str(x) for x in v)
    return str(v)


@dataclass
class AuditGradeResult:
    score:                float
    precision:            float
    recall:               float
    f_beta:               float
    evidence_specificity: float
    coverage_bonus:       float
    rule_results:         dict[str, bool]
    missed_ids:           list[str]

    def hint(self) -> str:
        if self.recall < 0.3:
            return (
                "Low recall — query more sections, use extract_claims, check tables "
                "carefully. Use type='table_text_mismatch' for discrepancies."
            )
        if self.precision < 0.5:
            return (
                "Low precision — submit only discrepancies confirmed against a table. "
                "Include 'table_id' and 'table_value' in each finding."
            )
        if self.evidence_specificity < 0.4:
            return "Add 'table_id' and 'table_value' to each finding for full credit."
        return f"Good. F-beta={self.f_beta:.2f}. Coverage bonus={self.coverage_bonus:.3f}."


class ClaimExtractor:
    """
    Regex-based claim extractor.
    Identifies sentences with numerical values + metric words.
    """
    _NUM = re.compile(r'\b\d+(?:\.\d+)?(?:\s*%|\s*percent)?\b')
    _TBL = re.compile(r'\b(?:Table|Tab\.)\s+(\d+[A-Za-z]?)\b', re.I)
    _FIG = re.compile(r'\b(?:Figure|Fig\.)\s+(\d+[A-Za-z]?)\b', re.I)
    _MET = re.compile(
        r'\b(?:accuracy|f1|precision|recall|score|performance|improvement|'
        r'reduction|increase|decrease|ap|map|bleu|rouge|auc|sensitivity|'
        r'specificity|throughput|latency|speedup|sharpe|drawdown|return|'
        r'perplexity|psnr|ssim|miou|parameter|epoch|batch)\b', re.I
    )

    def extract(self, text: str, section_name: str = "") -> list[dict]:
        claims = []
        for sent in re.split(r'(?<=[.!?])\s+', text):
            if not self._NUM.search(sent) or not self._MET.search(sent):
                continue
            claims.append({
                "sentence":    sent.strip()[:300],
                "values":      self._NUM.findall(sent)[:5],
                "table_refs":  [f"Table {r}" for r in self._TBL.findall(sent)],
                "figure_refs": [f"Figure {r}" for r in self._FIG.findall(sent)],
                "section":     section_name,
            })
        return claims


class AuditGrader:
    """Grades Task 3 submissions. Includes partial credit for RL gradient."""

    def __init__(self) -> None:
        self._extractor = ClaimExtractor()

    def grade(
        self,
        submissions: list[dict],
        paper: "Paper",
        nav_state: "NavigationState | None" = None,
    ) -> AuditGradeResult:
        ground_truth = paper.ground_truth.get("task3_discrepancies", [])

        if not ground_truth:
            ok    = len(submissions) == 0
            score = 1.0 if ok else max(0.0, 1.0 - len(submissions) * 0.1)
            return AuditGradeResult(score=score, precision=score, recall=1.0,
                                    f_beta=score, evidence_specificity=0.0,
                                    coverage_bonus=0.0, rule_results={}, missed_ids=[])

        if not submissions:
            # Partial credit from navigation even with no findings
            cov = self._coverage_bonus(nav_state)
            partial = cov * 0.15
            return AuditGradeResult(
                score=max(0.0001, partial), precision=0.0, recall=0.0,
                f_beta=0.0, evidence_specificity=0.0, coverage_bonus=cov,
                rule_results={d["id"]: False for d in ground_truth},
                missed_ids=[d["id"] for d in ground_truth],
            )

        # ── Bipartite matching (type-normalised + grounding-penalised) ───────
        # G8: each pair similarity is multiplied by the submission's grounding
        # score, so self-contradicting / hallucinated submissions can't match.
        sim_matrix: list[tuple[float, int, int]] = []
        for i, sub in enumerate(submissions):
            g_mult = _grounding_score(sub, paper)
            if g_mult <= 0.0:
                continue   # self-contradicting → never matches anything
            for j, gt in enumerate(ground_truth):
                s = self._similarity(sub, gt) * g_mult
                if s > 0:
                    sim_matrix.append((s, i, j))
        sim_matrix.sort(reverse=True)

        matched_gt:  set[int] = set()
        matched_sub: set[int] = set()
        for s, i, j in sim_matrix:
            if i not in matched_sub and j not in matched_gt and s >= SIM_THRESHOLD:
                matched_sub.add(i); matched_gt.add(j)

        tp        = len(matched_gt)
        precision = tp / len(submissions)
        recall    = tp / len(ground_truth)

        f_beta = 0.0
        if precision + recall > 0:
            f_beta = (1 + BETA_AUDIT**2) * precision * recall / (BETA_AUDIT**2 * precision + recall)

        # ── G7: Detect hallucinated findings (composite reward penalty) ───────────
        # Any submission with max similarity to ALL ground truth < threshold
        # is considered hallucinated — penalise to discourage fabrication.
        hallucination_count = 0
        for i, sub in enumerate(submissions):
            if i in matched_sub:
                continue  # already matched — not hallucinated
            max_sim_to_any_gt = max(
                (self._similarity(sub, gt) for gt in ground_truth),
                default=0.0
            )
            if max_sim_to_any_gt < HALLUCINATION_MIN_SIM:
                hallucination_count += 1

        # Hallucination penalty: −0.05 per hallucinated finding (capped at −0.20)
        hallucination_penalty = max(HALLUCINATION_PENALTY,
                                    -0.05 * hallucination_count)

        # Partial credit: reward unmatched submissions for being close.
        # G8: passes paper through so partial credit also grounds against it.
        partial_credit = self._partial_credit(submissions, ground_truth,
                                               matched_sub, paper)

        spec      = self._evidence_specificity(submissions, matched_sub)
        cov_bonus = self._coverage_bonus(nav_state)

        total = round(min(0.9999, max(0.0001,
            F_WEIGHT * f_beta
            + SPEC_WEIGHT * spec
            + COVERAGE_WEIGHT * cov_bonus
            + 0.05 * partial_credit     # near-miss gradient signal
            + hallucination_penalty     # G7: fabricated findings cost
        )), 4)

        return AuditGradeResult(
            score=total, precision=round(precision, 4),
            recall=round(recall, 4), f_beta=round(f_beta, 4),
            evidence_specificity=round(spec, 4),
            coverage_bonus=round(cov_bonus, 4),
            rule_results={d["id"]: (j in matched_gt) for j, d in enumerate(ground_truth)},
            missed_ids=[d["id"] for j, d in enumerate(ground_truth) if j not in matched_gt],
        )

    def _similarity(self, sub: dict, gt: dict) -> float:
        """Multi-criterion similarity with type normalisation.

        v6: every agent-supplied field goes through _safe_str so dicts /
        lists / None never reach .lower() / .strip().
        """
        sub_type = _norm_type(_safe_str(sub.get("type")))
        gt_type  = _norm_type(_safe_str(gt.get("type")))
        if sub_type != gt_type:
            return 0.0

        loc_sim   = _text_sim(
            _safe_str(sub.get("location") or sub.get("text_location")),
            _safe_str(gt.get("text_location")),
        )
        claim_sim = _text_sim(
            _safe_str(sub.get("claim") or sub.get("text_claim")),
            _safe_str(gt.get("text_claim")),
        )
        tbl_match = float(
            _safe_str(sub.get("table_id")).lower()
            == _safe_str(gt.get("table_id")).lower()
        ) if gt.get("table_id") else 1.0

        return 0.25 * loc_sim + 0.55 * claim_sim + 0.20 * tbl_match

    def _partial_credit(
        self, submissions: list[dict], ground_truth: list[dict],
        matched_sub: set[int], paper=None,
    ) -> float:
        """
        Partial credit for unmatched submissions that are numerically close.
        This gives GRPO a gradient even when the agent hasn't found exact match.

        G8: each unmatched submission's near-miss credit is multiplied by its
        grounding score, so a hallucinated near-miss earns ~0 partial credit.
        """
        unmatched_subs = [(i, s) for i, s in enumerate(submissions)
                          if i not in matched_sub]
        if not unmatched_subs:
            return 0.0

        best_partial = 0.0
        for _i, sub in unmatched_subs:
            g_mult = _grounding_score(sub, paper) if paper is not None else 1.0
            if g_mult <= 0.0:
                continue
            sub_num = _extract_number(_safe_str(sub.get("claim")))
            sub_loc = _safe_str(sub.get("location")).lower()
            for gt in ground_truth:
                gt_num = _extract_number(_safe_str(gt.get("text_claim")))
                gt_loc = _safe_str(gt.get("text_location")).lower()
                num_credit = 0.0
                if sub_num and gt_num:
                    rel_err = abs(sub_num - gt_num) / max(abs(gt_num), 1e-6)
                    num_credit = max(0.0, 1.0 - rel_err * 3)
                loc_credit = 0.3 if sub_loc == gt_loc else 0.0
                partial = (0.7 * num_credit + 0.3 * loc_credit) * g_mult
                best_partial = max(best_partial, partial)

        return best_partial

    def _evidence_specificity(self, submissions: list[dict], matched_sub: set[int]) -> float:
        if not matched_sub:
            return 0.0
        specific = sum(
            1 for i in matched_sub
            if bool(_safe_str(submissions[i].get("table_id")).strip())
            and bool(_safe_str(submissions[i].get("table_value")).strip())
        )
        return specific / len(matched_sub)

    def _coverage_bonus(self, nav_state) -> float:
        if nav_state is None:
            return 0.0
        try:
            from ..reward_shaper import PotentialBasedShaper
            return PotentialBasedShaper(nav_state).final_coverage_bonus()
        except Exception:
            return 0.0
