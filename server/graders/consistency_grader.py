"""
server/graders/consistency_grader.py — Task 2: internal_consistency grader.

Reward design principle (from ScholarEnv master guide):
  Use F-beta (beta=0.5) rather than plain F1.
  beta < 1 weights PRECISION more than RECALL.

  Rationale: an agent that reports many spurious inconsistencies (recall
  inflation / hallucination) should be penalised more than one that misses
  a few real ones.  This is critical for GRPO training — if the agent can
  score well by submitting long lists of guesses, RL will discover that
  exploit immediately.  F-beta with beta=0.5 closes that loophole.

Matching algorithm:
  For each submitted finding, find the best-matching ground truth issue
  using a 3-criterion fuzzy match (type, location, claim text).
  Matching is exclusive: each GT issue can only be matched once.
  Ties broken by similarity score.

Evidence specificity bonus:
  Small bonus (max 0.05) for findings that include specific location info
  (section name or figure/table ID).  Encourages the agent to cite evidence
  rather than make vague claims.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...corpus import Paper

BETA:              float = 0.5   # F-beta weight (precision-biased)
MAX_SPEC_BONUS:    float = 0.05  # evidence specificity bonus ceiling
SIM_THRESHOLD:     float = 0.45  # minimum similarity for claim match


@dataclass
class ConsistencyGradeResult:
    score:               float
    precision:           float
    recall:              float
    f_beta:              float
    evidence_specificity: float
    rule_results:        dict[str, bool]
    missed_ids:          list[str]
    tier_breakdown:      dict[str, float]

    def hint(self) -> str:
        if self.recall < 0.4:
            return (
                "Low recall — you are missing most inconsistencies. "
                "Query more sections and look for number mismatches."
            )
        if self.precision < 0.5:
            return (
                "Low precision — too many spurious findings. "
                "Only report inconsistencies you can specifically locate."
            )
        return f"F-beta={self.f_beta:.2f}. Improve by citing exact section names."


class ConsistencyGrader:
    """Grades Task 2 (internal_consistency) submissions against ground truth."""

    def grade(
        self,
        submissions: list[dict],
        paper: "Paper",
        step_count: int = 1,
    ) -> ConsistencyGradeResult:
        ground_truth = paper.ground_truth.get("task2_inconsistencies", [])

        if not ground_truth:
            # No issues in this paper — correct answer is empty list
            empty_correct = len(submissions) == 0
            score = 1.0 if empty_correct else max(0.0, 1.0 - len(submissions) * 0.1)
            return ConsistencyGradeResult(
                score=score, precision=score, recall=1.0,
                f_beta=score, evidence_specificity=0.0,
                rule_results={}, missed_ids=[], tier_breakdown={},
            )

        if not submissions:
            rule_results = {gt["id"]: False for gt in ground_truth}
            return ConsistencyGradeResult(
                score=0.0, precision=0.0, recall=0.0,
                f_beta=0.0, evidence_specificity=0.0,
                rule_results=rule_results, missed_ids=[gt["id"] for gt in ground_truth],
                tier_breakdown={},
            )

        # ── Greedy bipartite matching ────────────────────────────────────────
        matched_gt:  set[int] = set()
        matched_sub: set[int] = set()

        # Build similarity matrix first, then greedily match highest-sim pairs
        sim_matrix: list[tuple[float, int, int]] = []
        for i, sub in enumerate(submissions):
            for j, gt in enumerate(ground_truth):
                sim = self._similarity(sub, gt)
                if sim > 0:
                    sim_matrix.append((sim, i, j))
        sim_matrix.sort(reverse=True)

        for sim, i, j in sim_matrix:
            if i not in matched_sub and j not in matched_gt:
                if sim >= SIM_THRESHOLD:
                    matched_sub.add(i)
                    matched_gt.add(j)

        tp = len(matched_gt)
        precision = tp / len(submissions) if submissions else 0.0
        recall    = tp / len(ground_truth)

        # F-beta score (precision-weighted)
        if precision + recall > 0:
            f_beta = (
                (1 + BETA**2) * precision * recall
                / (BETA**2 * precision + recall)
            )
        else:
            f_beta = 0.0

        # Evidence specificity bonus
        spec_bonus = self._evidence_specificity_bonus(submissions, matched_sub)

        # Final score: clamp to [0, 1]
        total = round(min(1.0, f_beta + spec_bonus), 4)

        # Per-GT-issue breakdown (for curriculum tracking)
        rule_results = {
            gt["id"]: (j in matched_gt)
            for j, gt in enumerate(ground_truth)
        }
        missed_ids = [gt["id"] for j, gt in enumerate(ground_truth)
                      if j not in matched_gt]

        # Tier breakdown: injected vs natural
        tier_breakdown = self._tier_breakdown(ground_truth, matched_gt)

        return ConsistencyGradeResult(
            score=total,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f_beta=round(f_beta, 4),
            evidence_specificity=round(spec_bonus, 4),
            rule_results=rule_results,
            missed_ids=missed_ids,
            tier_breakdown=tier_breakdown,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _similarity(self, sub: dict, gt: dict) -> float:
        """
        3-criterion fuzzy similarity for one (submission, ground-truth) pair.
        Returns 0.0 if type does not match.
        """
        if sub.get("type") != gt.get("type"):
            return 0.0

        # Location similarity
        loc_sim = difflib.SequenceMatcher(
            None,
            str(sub.get("location", "")).lower(),
            str(gt.get("location_a", "")).lower(),
        ).ratio()

        # Claim text similarity
        claim_sim = difflib.SequenceMatcher(
            None,
            str(sub.get("claim", "")).lower(),
            str(gt.get("claim_a", "")).lower(),
        ).ratio()

        # "contradicts" / second claim similarity
        contra_sim = difflib.SequenceMatcher(
            None,
            str(sub.get("contradicts", "")).lower(),
            str(gt.get("claim_b", "")).lower(),
        ).ratio()

        # Weighted: location carries less weight than claim text
        return 0.25 * loc_sim + 0.45 * claim_sim + 0.30 * contra_sim

    def _evidence_specificity_bonus(
        self, submissions: list[dict], matched_sub: set[int]
    ) -> float:
        """
        Small bonus for submissions that cite specific evidence locations.
        Only applied to correctly-matched findings (no reward for wrong guesses).
        """
        if not matched_sub:
            return 0.0
        spec_count = 0
        for i in matched_sub:
            sub = submissions[i]
            location = str(sub.get("location", ""))
            # Counts as specific if it names a section or table/figure
            if any(tok in location.lower()
                   for tok in ("table", "figure", "fig.", "section",
                               "abstract", "introduction", "results", "methods")):
                spec_count += 1
        ratio = spec_count / len(matched_sub)
        return MAX_SPEC_BONUS * ratio

    def _tier_breakdown(
        self, ground_truth: list[dict], matched_gt: set[int]
    ) -> dict[str, float]:
        """Split recall into injected vs naturally-occurring inconsistencies."""
        injected   = [(j, gt) for j, gt in enumerate(ground_truth) if gt.get("injected")]
        natural    = [(j, gt) for j, gt in enumerate(ground_truth) if not gt.get("injected")]

        def recall_for(items: list) -> float:
            if not items:
                return 1.0
            found = sum(1 for j, _ in items if j in matched_gt)
            return round(found / len(items), 4)

        return {
            "injected_recall": recall_for(injected),
            "natural_recall":  recall_for(natural),
        }
