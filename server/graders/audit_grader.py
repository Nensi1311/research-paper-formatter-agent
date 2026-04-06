"""
server/graders/audit_grader.py — Task 3: claim_evidence_audit grader.

The hardest task: find discrepancies where text claims don't match tables/figures.

Reward composition:
  score = 0.70 × F-beta(0.5)
        + 0.20 × evidence_specificity
        + 0.10 × coverage_bonus   (from PBRS NavigationState)

Evidence specificity metric:
  For each correctly-matched discrepancy, check whether the agent's finding
  includes a specific table_id AND a table_value reference. If so, it counted as
  specific evidence (not vague). Score = fraction of matched findings with specificity.

Coverage bonus:
  PotentialBasedShaper.final_coverage_bonus() — rewards agents that read a
  large fraction of the document before submitting.

Why this matters for RL training:
  The optimal traversal strategy (which sections to read, in which order)
  CANNOT be reduced to a prompt.  It varies by paper structure.  RL discovers
  this strategy by exploring.  The coverage bonus provides signal that encourages
  exploration — agents that read more sections will discover more discrepancies.

  Expected GPT-4o score (no RL): 0.20-0.45
  Trained agent target:          0.55-0.75
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...corpus import Paper
    from ..reward_shaper import NavigationState

BETA_AUDIT = 0.5    # F-beta weight (precision-biased, same as Task 2)
SIM_THRESHOLD = 0.40  # lower than Task 2 — discrepancies are harder to phrase

F_WEIGHT       = 0.70
SPEC_WEIGHT    = 0.20
COVERAGE_WEIGHT = 0.10
MAX_SPEC_BONUS  = 0.20   # ceiling for this component (before weighting)


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
                "Low recall — query more sections and use extract_claims to "
                "find numerical claims before submitting. Check tables carefully."
            )
        if self.precision < 0.5:
            return (
                "Low precision — many findings are incorrect. "
                "Only submit discrepancies where you can cite a specific table_id "
                "and observed table_value."
            )
        if self.evidence_specificity < 0.4:
            return (
                "Include 'table_id' and 'table_value' in each finding dict "
                "to improve the evidence specificity score."
            )
        return f"Good work. F-beta={self.f_beta:.2f}. Coverage bonus={self.coverage_bonus:.3f}."


class ClaimExtractor:
    """
    Lightweight regex-based claim extractor.

    Identifies sentences containing numerical values and comparison words,
    which are candidates for cross-referencing against tables.

    Returns structured dicts rather than strings — consumable by the agent
    via the extract_claims action.
    """

    # Patterns that suggest a numerical claim
    _NUM_PATTERN    = re.compile(r'\b\d+(?:\.\d+)?(?:\s*%|\s*percent)?\b')
    _TABLE_REF      = re.compile(r'\b(?:Table|Tab\.)\s+(\d+[A-Za-z]?)\b', re.I)
    _FIGURE_REF     = re.compile(r'\b(?:Figure|Fig\.)\s+(\d+[A-Za-z]?)\b', re.I)
    _METRIC_WORDS   = re.compile(
        r'\b(?:accuracy|f1|precision|recall|score|performance|improvement|'
        r'reduction|increase|decrease|aps|map|bleu|rouge|psnr|ssim|perplexity|'
        r'throughput|latency|speedup|parameter|weight|layer|epoch|batch)\b',
        re.I,
    )

    def extract(self, text: str, section_name: str = "") -> list[dict]:
        """
        Return a list of claim dicts from the text.
        Each dict: {sentence, value, table_refs, figure_refs, section}.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []
        for sent in sentences:
            nums    = self._NUM_PATTERN.findall(sent)
            metrics = self._METRIC_WORDS.search(sent)
            if not nums or not metrics:
                continue
            table_refs  = self._TABLE_REF.findall(sent)
            figure_refs = self._FIGURE_REF.findall(sent)
            claims.append({
                "sentence":    sent.strip()[:300],
                "values":      nums[:5],
                "table_refs":  [f"Table {r}" for r in table_refs],
                "figure_refs": [f"Figure {r}" for r in figure_refs],
                "section":     section_name,
            })
        return claims


class AuditGrader:
    """Grades Task 3 (claim_evidence_audit) submissions against ground truth."""

    def __init__(self) -> None:
        self._extractor = ClaimExtractor()

    def grade(
        self,
        submissions:  list[dict],
        paper:        "Paper",
        nav_state:    "NavigationState | None" = None,
    ) -> AuditGradeResult:
        ground_truth = paper.ground_truth.get("task3_discrepancies", [])

        if not ground_truth:
            empty_correct = len(submissions) == 0
            score = 1.0 if empty_correct else max(0.0, 1.0 - len(submissions) * 0.1)
            return AuditGradeResult(
                score=score, precision=score, recall=1.0,
                f_beta=score, evidence_specificity=0.0, coverage_bonus=0.0,
                rule_results={}, missed_ids=[],
            )

        if not submissions:
            rule_results = {d["id"]: False for d in ground_truth}
            return AuditGradeResult(
                score=0.0, precision=0.0, recall=0.0,
                f_beta=0.0, evidence_specificity=0.0, coverage_bonus=0.0,
                rule_results=rule_results,
                missed_ids=[d["id"] for d in ground_truth],
            )

        # ── Greedy bipartite matching ────────────────────────────────────────
        sim_matrix: list[tuple[float, int, int]] = []
        for i, sub in enumerate(submissions):
            for j, gt in enumerate(ground_truth):
                sim = self._similarity(sub, gt)
                if sim > 0:
                    sim_matrix.append((sim, i, j))
        sim_matrix.sort(reverse=True)

        matched_gt:  set[int] = set()
        matched_sub: set[int] = set()
        for sim, i, j in sim_matrix:
            if i not in matched_sub and j not in matched_gt and sim >= SIM_THRESHOLD:
                matched_sub.add(i)
                matched_gt.add(j)

        tp        = len(matched_gt)
        precision = tp / len(submissions)
        recall    = tp / len(ground_truth)

        if precision + recall > 0:
            f_beta = (
                (1 + BETA_AUDIT**2) * precision * recall
                / (BETA_AUDIT**2 * precision + recall)
            )
        else:
            f_beta = 0.0

        # Evidence specificity
        spec = self._evidence_specificity(submissions, matched_sub)

        # Coverage bonus (PBRS terminal)
        cov_bonus = 0.0
        if nav_state is not None:
            from ..reward_shaper import PotentialBasedShaper
            cov_bonus = PotentialBasedShaper(nav_state).final_coverage_bonus()

        # Composite score
        total = round(
            min(1.0,
                F_WEIGHT * f_beta
                + SPEC_WEIGHT * spec
                + COVERAGE_WEIGHT * cov_bonus),
            4,
        )

        rule_results = {
            d["id"]: (j in matched_gt)
            for j, d in enumerate(ground_truth)
        }
        missed_ids = [d["id"] for j, d in enumerate(ground_truth)
                      if j not in matched_gt]

        return AuditGradeResult(
            score=total,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f_beta=round(f_beta, 4),
            evidence_specificity=round(spec, 4),
            coverage_bonus=round(cov_bonus, 4),
            rule_results=rule_results,
            missed_ids=missed_ids,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _similarity(self, sub: dict, gt: dict) -> float:
        """
        Multi-criterion similarity for discrepancy matching.
        Returns 0.0 on type mismatch.
        """
        if sub.get("type") != gt.get("type"):
            return 0.0

        # Text location match
        text_loc_sim = difflib.SequenceMatcher(
            None,
            str(sub.get("location", sub.get("text_location", ""))).lower(),
            str(gt.get("text_location", "")).lower(),
        ).ratio()

        # Claimed value match
        claim_sim = difflib.SequenceMatcher(
            None,
            str(sub.get("claim", sub.get("text_claim", ""))).lower(),
            str(gt.get("text_claim", "")).lower(),
        ).ratio()

        # Table ID match (binary)
        table_match = (
            str(sub.get("table_id", "")).lower() ==
            str(gt.get("table_id", "")).lower()
        ) if gt.get("table_id") else True

        return (
            0.30 * text_loc_sim
            + 0.50 * claim_sim
            + 0.20 * float(table_match)
        )

    def _evidence_specificity(
        self, submissions: list[dict], matched_sub: set[int]
    ) -> float:
        """
        Fraction of correctly-matched findings that include both:
          - a table_id reference
          - a table_value reference
        Normalised to [0, MAX_SPEC_BONUS] × SPEC_WEIGHT.
        """
        if not matched_sub:
            return 0.0
        specific = 0
        for i in matched_sub:
            sub = submissions[i]
            has_table = bool(sub.get("table_id", "").strip())
            has_value = bool(sub.get("table_value", "").strip())
            if has_table and has_value:
                specific += 1
        return specific / len(matched_sub)
