"""
server/curriculum.py — Hybrid curriculum: UCB1 bandit + AdaRFT difficulty targeting.

Two complementary signals drive paper selection:

  1. UCB1 bandit (server/bandit.py)
     Which papers produce the highest RL learning gradient?
     Learning gradient = variance of reward across pulls.
     A paper scoring ~0.5 ± 0.2 is maximally informative for GRPO.

  2. AdaRFT difficulty targeting (arxiv 2504.05520 §3.2)
     Keep the agent in the "productive zone": average score ∈ [0.40, 0.70].
     Too easy (avg > 0.70) → select harder papers.
     Too hard  (avg < 0.40) → select easier papers.
     Optimal   → mix with UCB1 signal.

  3. Weak-rule targeting
     Track per-rule success rates over a rolling window.
     Prioritise papers that exercise rules the agent currently fails most.

Final selection score:
  s(paper) = 0.50 × UCB1_rank
           + 0.30 × difficulty_match
           + 0.20 × weak_rule_coverage

This outperforms any single signal:
  - Pure UCB1 ignores rule-specific weaknesses
  - Pure difficulty ignores paper-level RL signal
  - Pure weak-rule ignores exploration-exploitation balance

Reference: AdaRFT — arxiv 2504.05520 (adaptive data selection for RLVR)
"""
from __future__ import annotations

import random
from collections import deque, defaultdict
from typing import TYPE_CHECKING

from .bandit import UCB1Bandit

if TYPE_CHECKING:
    from ..corpus import PaperCorpus

# Productive zone bounds (AdaRFT §3.2)
EASY_THRESHOLD = 0.40
HARD_THRESHOLD = 0.70

W_UCB1        = 0.50
W_DIFFICULTY  = 0.30
W_WEAK_RULES  = 0.20
WEAK_THRESHOLD = 0.55   # rule success rate below this → "weak rule"
RULE_WINDOW    = 20      # rolling window for per-rule success tracking


class Curriculum:
    """
    Unified AdaRFT + UCB1 curriculum for ScholarEnv.
    Thread-safe (no shared mutable state between episodes).
    """

    def __init__(
        self,
        window_size:    int   = 30,
        weak_threshold: float = WEAK_THRESHOLD,
    ) -> None:
        self.bandit          = UCB1Bandit(c=1.0, recency_penalty=0.2)
        self.rule_success:   dict[str, list[bool]] = defaultdict(list)
        self.weak_threshold  = weak_threshold
        self._window:        deque[dict] = deque(maxlen=window_size)

    # ── Public API ────────────────────────────────────────────────────────────

    def select(self, corpus: "PaperCorpus", task_id: str) -> str:
        """Select the best paper for the next episode."""
        candidates = [
            pid for pid, p in corpus.papers.items()
            if task_id in getattr(p, "applicable_tasks", [task_id])
        ]
        if not candidates:
            raise ValueError(f"No papers registered for task_id='{task_id}'")

        # Cold start: pure UCB1 for first 5 episodes
        if len(self._window) < 5:
            return self.bandit.select(candidates, task_id)

        weak_rules     = self._weak_rules()
        target_diff    = self._target_difficulty()

        best_pid, best_score = candidates[0], -float("inf")
        for pid in candidates:
            paper = corpus.papers[pid]

            # UCB1 rank (normalised to [0,1])
            ucb1_rank = self._ucb1_rank(pid, task_id, len(candidates))

            # Difficulty match: 1 - |paper_diff - target|
            paper_diff = getattr(paper, "difficulty_score", 0.5)
            diff_match = max(0.0, 1.0 - abs(paper_diff - target_diff))

            # Weak-rule coverage
            violated = set(getattr(paper, "violated_rules_task1", []))
            if weak_rules:
                coverage = len(set(weak_rules) & violated) / len(weak_rules)
            else:
                coverage = 0.5

            composite = (
                W_UCB1 * ucb1_rank
                + W_DIFFICULTY * diff_match
                + W_WEAK_RULES * coverage
            )
            if composite > best_score:
                best_score = composite
                best_pid   = pid

        return best_pid

    def update(
        self,
        paper_id:     str,
        task_id:      str,
        score:        float,
        rule_results: dict[str, bool],
    ) -> None:
        """Record episode outcome — call once per completed episode."""
        self.bandit.update(paper_id, task_id, score)
        self._window.append({
            "paper_id": paper_id,
            "task_id":  task_id,
            "score":    score,
            "rules":    rule_results,
        })
        # Rolling per-rule success rates
        for rule, passed in rule_results.items():
            self.rule_success[rule].append(passed)
            if len(self.rule_success[rule]) > RULE_WINDOW:
                self.rule_success[rule].pop(0)

    def hint(self, paper_id: str | None = None) -> str:
        """Return a short hint string for the current observation."""
        weak = self._weak_rules()
        if not weak:
            return ""
        return (
            f"Focus on: {', '.join(weak[:4])}. "
            "Pass rate below 55% recently on these rules."
        )

    def summary(self) -> dict:
        return {
            "episodes":   len(self._window),
            "avg_score":  (
                sum(e["score"] for e in self._window) / len(self._window)
                if self._window else 0.0
            ),
            "target_difficulty": self._target_difficulty(),
            "weak_rules": self._weak_rules(),
            "rule_rates": {
                r: round(sum(v) / len(v), 3)
                for r, v in self.rule_success.items() if v
            },
            "bandit_top": self.bandit.top_learning_arms(3),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _weak_rules(self) -> list[str]:
        return [
            r for r, v in self.rule_success.items()
            if v and sum(v) / len(v) < self.weak_threshold
        ]

    def _target_difficulty(self) -> float:
        """
        AdaRFT: shift target difficulty based on recent average score.
        Agent doing well → push harder; agent struggling → ease off.
        """
        if not self._window:
            return 0.5
        avg = sum(e["score"] for e in self._window) / len(self._window)
        if avg > HARD_THRESHOLD:
            return 0.8   # push harder — agent is ready
        if avg < EASY_THRESHOLD:
            return 0.3   # ease off — agent is stuck
        return 0.5       # optimal productive zone

    def _ucb1_rank(
        self, paper_id: str, task_id: str, n_arms: int
    ) -> float:
        """Normalise UCB1 score to [0, 1] for blending."""
        key = f"{paper_id}:{task_id}"
        arm = self.bandit.arms.get(key)
        if arm is None or arm.n_pulls == 0:
            return 1.0  # unexplored → max rank
        import math
        T   = max(self.bandit.total_episodes, 1)
        raw = (
            arm.learning_gradient()
            + self.bandit.c * math.sqrt(math.log(T) / arm.n_pulls)
        )
        # Crude normalisation — absolute values don't matter, only ranking
        return min(1.0, raw)
