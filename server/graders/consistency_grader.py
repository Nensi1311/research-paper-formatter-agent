"""
server/graders/consistency_grader.py — Task 2: internal_consistency grader.
Upgraded: thefuzz matching + type aliases (from audit_grader pattern).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...corpus import Paper

# thefuzz with difflib fallback
try:
    from thefuzz import fuzz as _fuzz
    def _sim(a: str, b: str) -> float:
        return _fuzz.partial_ratio(str(a).lower(), str(b).lower()) / 100.0
except ImportError:
    import difflib
    def _sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

TYPE_ALIASES = {
    "mismatch": "number_mismatch", "value_mismatch": "number_mismatch",
    "numerical_discrepancy": "number_mismatch", "contradiction": "number_mismatch",
    "inconsistency": "number_mismatch", "conflict": "number_mismatch",
    "missing": "missing_reference", "ghost_ref": "missing_reference",
}
def _norm_type(t: str) -> str:
    t = str(t).lower().strip().replace(" ", "_").replace("-", "_")
    return TYPE_ALIASES.get(t, t)

BETA          = 0.5
SIM_THRESHOLD = 0.40
MAX_SPEC_BONUS = 0.05


@dataclass
class ConsistencyGradeResult:
    score: float; precision: float; recall: float; f_beta: float
    evidence_specificity: float; rule_results: dict; missed_ids: list; tier_breakdown: dict

    def hint(self) -> str:
        if self.recall < 0.4:
            return "Low recall — query more sections and look for number mismatches."
        if self.precision < 0.5:
            return "Low precision — only report inconsistencies you can specifically locate."
        return f"F-beta={self.f_beta:.2f}. Cite exact section names for evidence bonus."


class ConsistencyGrader:
    def grade(self, submissions: list[dict], paper: "Paper", step_count: int = 1) -> ConsistencyGradeResult:
        gt = paper.ground_truth.get("task2_inconsistencies", [])
        if not gt:
            ok    = len(submissions) == 0
            score = 1.0 if ok else max(0.0, 1.0 - len(submissions) * 0.1)
            return ConsistencyGradeResult(score=score, precision=score, recall=1.0,
                                          f_beta=score, evidence_specificity=0.0,
                                          rule_results={}, missed_ids=[], tier_breakdown={})
        if not submissions:
            return ConsistencyGradeResult(score=0.0, precision=0.0, recall=0.0,
                                          f_beta=0.0, evidence_specificity=0.0,
                                          rule_results={g["id"]: False for g in gt},
                                          missed_ids=[g["id"] for g in gt], tier_breakdown={})

        sims: list[tuple[float, int, int]] = []
        for i, sub in enumerate(submissions):
            for j, g in enumerate(gt):
                s = self._similarity(sub, g)
                if s > 0:
                    sims.append((s, i, j))
        sims.sort(reverse=True)
        mg: set[int] = set(); ms: set[int] = set()
        for s, i, j in sims:
            if i not in ms and j not in mg and s >= SIM_THRESHOLD:
                mg.add(j); ms.add(i)

        tp = len(mg); p = tp / len(submissions); r = tp / len(gt)
        f  = (1 + BETA**2) * p * r / (BETA**2 * p + r) if p + r > 0 else 0.0
        spec_bonus = self._spec_bonus(submissions, ms)
        total = round(min(1.0, f + spec_bonus), 4)

        return ConsistencyGradeResult(
            score=total, precision=round(p, 4), recall=round(r, 4), f_beta=round(f, 4),
            evidence_specificity=round(spec_bonus, 4),
            rule_results={g["id"]: (j in mg) for j, g in enumerate(gt)},
            missed_ids=[g["id"] for j, g in enumerate(gt) if j not in mg],
            tier_breakdown=self._tier_breakdown(gt, mg),
        )

    def _similarity(self, sub: dict, gt: dict) -> float:
        if _norm_type(sub.get("type", "")) != _norm_type(gt.get("type", "")):
            return 0.0
        loc   = _sim(sub.get("location", ""), gt.get("location_a", ""))
        claim = _sim(sub.get("claim", ""), gt.get("claim_a", ""))
        contra = _sim(sub.get("contradicts", ""), gt.get("claim_b", ""))
        return 0.25 * loc + 0.45 * claim + 0.30 * contra

    def _spec_bonus(self, submissions: list[dict], matched: set[int]) -> float:
        if not matched:
            return 0.0
        tokens = ("table", "figure", "fig.", "section", "abstract",
                  "introduction", "results", "methods", "discussion")
        spec = sum(1 for i in matched
                   if any(t in str(submissions[i].get("location", "")).lower()
                          for t in tokens))
        return MAX_SPEC_BONUS * (spec / len(matched))

    def _tier_breakdown(self, gt: list, matched: set[int]) -> dict:
        inj = [(j, g) for j, g in enumerate(gt) if g.get("injected")]
        nat = [(j, g) for j, g in enumerate(gt) if not g.get("injected")]
        def rec(items): return round(sum(1 for j, _ in items if j in matched) / max(len(items), 1), 4)
        return {"injected_recall": rec(inj), "natural_recall": rec(nat)}
