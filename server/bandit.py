"""
server/bandit.py — UCB1 bandit for paper selection.

Multi-armed bandit where each arm = (paper_id, task_id) pair.
UCB1 balances exploration (trying underused papers) with exploitation
(using papers that have historically produced the highest learning gradient).

Learning gradient proxy = score variance, not mean score.
  A paper that always produces score 0.9 teaches the agent nothing.
  A paper that produces scores in [0.3, 0.7] is in the "productive zone"
  and should be selected more frequently.

Recency penalty (optional):
  Arms that were pulled very recently get a small penalty to encourage
  variety in the training distribution.

Reference: Auer, Cesa-Bianchi & Fischer (2002) — UCB1 regret bound O(√(KT log T)).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ArmStats:
    """Per-(paper_id, task_id) statistics."""
    n_pulls:       int   = 0
    sum_reward:    float = 0.0
    sum_sq_reward: float = 0.0
    last_pull:     int   = 0     # episode number of most recent pull

    @property
    def mean_reward(self) -> float:
        return self.sum_reward / self.n_pulls if self.n_pulls > 0 else 0.0

    @property
    def variance(self) -> float:
        if self.n_pulls < 2:
            return 1.0   # high uncertainty for underexplored arms
        mean = self.mean_reward
        return max(0.0,
                   self.sum_sq_reward / self.n_pulls - mean ** 2)

    def learning_gradient(self) -> float:
        """
        Proxy for how much RL signal this paper generates.
        Peaks at variance ~ 0.04 (scores spread ± 0.2 around 0.5).
        """
        v = self.variance
        # Gaussian-shaped reward: max at v=0.04, tails to 0
        return math.exp(-(v - 0.04) ** 2 / (2 * 0.02 ** 2))


class UCB1Bandit:
    """
    UCB1 with learning-gradient reward instead of mean reward.

    UCB score = learning_gradient + c × sqrt(log(T) / n_pulls)

    c=1.0 matches standard UCB1.  Increase for more exploration.
    """

    def __init__(self, c: float = 1.0, recency_penalty: float = 0.2) -> None:
        self.c               = c
        self.recency_penalty = recency_penalty
        self.arms:           dict[str, ArmStats] = defaultdict(ArmStats)
        self.total_episodes: int = 0

    def _arm_key(self, paper_id: str, task_id: str) -> str:
        return f"{paper_id}:{task_id}"

    def select(self, paper_ids: list[str], task_id: str) -> str:
        """Select the arm with the highest UCB1 score."""
        self.total_episodes += 1
        T = max(self.total_episodes, 1)

        best_id, best_score = paper_ids[0], -float("inf")
        for pid in paper_ids:
            arm = self.arms[self._arm_key(pid, task_id)]
            if arm.n_pulls == 0:
                # Unvisited arm — UCB is infinite; force exploration
                return pid
            recency = (
                self.recency_penalty
                if (T - arm.last_pull) < 3
                else 0.0
            )
            ucb = (
                arm.learning_gradient()
                + self.c * math.sqrt(math.log(T) / arm.n_pulls)
                - recency
            )
            if ucb > best_score:
                best_score = ucb
                best_id    = pid
        return best_id

    def update(self, paper_id: str, task_id: str, score: float) -> None:
        arm = self.arms[self._arm_key(paper_id, task_id)]
        arm.n_pulls      += 1
        arm.sum_reward   += score
        arm.sum_sq_reward += score ** 2
        arm.last_pull    = self.total_episodes

    def top_learning_arms(self, n: int = 3) -> list[dict]:
        """Return top-n arms by learning gradient (for logging)."""
        sorted_arms = sorted(
            self.arms.items(),
            key=lambda kv: kv[1].learning_gradient(),
            reverse=True,
        )
        return [
            {"arm": k, "gradient": round(v.learning_gradient(), 4),
             "mean": round(v.mean_reward, 4), "pulls": v.n_pulls}
            for k, v in sorted_arms[:n]
        ]
