"""
server/reward_shaper.py — Potential-Based Reward Shaping (PBRS) for
multi-step navigation tasks (Tasks 2 & 3).

Theory (Ng, Harada & Russell, ICML 1999):
  Shaping bonus  F(s, s') = γ·Φ(s') − Φ(s)
  is policy-invariant IFF Φ is a real-valued potential function.

Why ScholarEnv needs this:
  Tasks 2 & 3 have SPARSE terminal rewards — the agent receives 0.0
  for every navigation step and a final score only on submit_findings.
  Sparse rewards cause zero-advantage samples in GRPO, collapsing
  gradient updates (arxiv 2512.07478 §2.2, GTPO arxiv 2508.04349).

  PBRS adds dense intermediate rewards that:
    1. Are guaranteed not to change the optimal policy (Ng et al. 1999)
    2. Reflect genuine evidence-gathering progress
    3. Prevent zero-advantage collapse during GRPO rollout batches

Potential function Φ (∈ [0, 1]):
  Φ(s) = w_sec × (sections_read / total_sections)
        + w_tbl × (tables_checked / total_tables)
        + w_clm × min(1, claims_extracted / est_total_claims)

  Weights sum to 1; claim weight is highest because extracting structured
  claims is the most discriminative evidence-gathering action.

Coverage bonus (terminal):
  Sigmoid-shaped bonus for agents that read ≥ 60% of the document.
  Encourages systematic auditing over cherry-picking known sections.
"""
from __future__ import annotations

from dataclasses import dataclass, field

GAMMA: float = 0.99          # discount factor — must match training config
MAX_SHAPING_BONUS: float = 0.15   # clamp to prevent over-rewarding navigation


@dataclass
class NavigationState:
    """
    Tracks evidence-gathering actions taken in the current episode.

    Accumulated by ScholarEnvironment._step_navigate() and read by
    PotentialBasedShaper.potential().
    """
    total_sections:         int       = 0
    total_tables:           int       = 0
    sections_read:          set[str]  = field(default_factory=set)
    tables_checked:         set[str]  = field(default_factory=set)
    claims_extracted:       int       = 0
    estimated_total_claims: int       = 10  # refined on first extract_claims

    def record_section(self, section_name: str) -> None:
        self.sections_read.add(section_name.lower())

    def record_table(self, table_id: str) -> None:
        self.tables_checked.add(table_id.lower())

    def record_claims(self, n_claims: int) -> None:
        self.claims_extracted += n_claims
        # Refine total estimate (running max heuristic)
        if n_claims > 0:
            self.estimated_total_claims = max(
                self.estimated_total_claims,
                n_claims * max(1, self.total_sections)
            )

    @property
    def section_coverage(self) -> float:
        if self.total_sections == 0:
            return 0.0
        return len(self.sections_read) / self.total_sections

    @property
    def table_coverage(self) -> float:
        if self.total_tables == 0:
            return 0.0
        return len(self.tables_checked) / self.total_tables


class PotentialBasedShaper:
    """
    Implements PBRS for ScholarEnv navigation steps.

    Usage pattern in environment._step_navigate():
        phi_before = PotentialBasedShaper(ep.nav_state).potential()
        # ... apply action, update nav_state ...
        phi_after  = PotentialBasedShaper(ep.nav_state).potential()
        bonus      = PotentialBasedShaper(ep.nav_state).shaping_bonus(
                         phi_before, phi_after
                     )
    """

    def __init__(
        self,
        state:          NavigationState,
        section_weight: float = 0.30,
        table_weight:   float = 0.30,
        claim_weight:   float = 0.40,
        max_bonus:      float = MAX_SHAPING_BONUS,
    ) -> None:
        self._s        = state
        self.w_sec     = section_weight
        self.w_tbl     = table_weight
        self.w_clm     = claim_weight
        self.max_bonus = max_bonus

    def potential(self) -> float:
        """Φ(state) ∈ [0, 1]."""
        s = self._s
        clm_ratio = (
            min(1.0, s.claims_extracted / s.estimated_total_claims)
            if s.estimated_total_claims > 0 else 0.0
        )
        return (
            self.w_sec * s.section_coverage
            + self.w_tbl * s.table_coverage
            + self.w_clm * clm_ratio
        )

    def shaping_bonus(self, phi_before: float, phi_after: float) -> float:
        """
        F(s, s') = γ·Φ(s') − Φ(s)

        Clipped to [0, max_bonus] — negative shaping would penalise
        any navigation (we want to reward progress, not direction).
        """
        bonus = GAMMA * phi_after - phi_before
        return round(max(0.0, min(bonus, self.max_bonus)), 6)

    def final_coverage_bonus(self) -> float:
        """
        Terminal bonus for comprehensive document coverage.
        Sigmoid-shaped: activates above 60% section coverage.
        Max contribution: 0.05 (does not dominate the F-beta score).
        """
        cov = self._s.section_coverage
        if cov < 0.6:
            return 0.0
        return round(0.05 * (cov - 0.6) / 0.4, 6)
