"""
models.py — Canonical Pydantic models for ScholarEnv v0.3.

Design principles (Google DeepMind / Meta ML engineer standard):
  1. AnyAction uses Annotated discriminated union — FastAPI/Pydantic deserialises
     FormattingAction vs ScholarAction from a single 'task' field.
  2. EpisodeStatus enum — no magic strings.
  3. All Optional fields carry explicit defaults.
  4. ScholarReward mirrors the grader internals for full auditability.
  5. No circular imports — models.py imports nothing from server/.

References:
  PRS  — arxiv 2512.07478 (stage_N_score fields)
  PBRS — Ng, Harada & Russell 1999 (shaping_bonus)
  AdaRFT — arxiv 2504.05520 (curriculum_hint)
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class EpisodeStatus(str, Enum):
    ACTIVE = "active"
    DONE   = "done"


# ── Actions ───────────────────────────────────────────────────────────────────

class FormattingAction(BaseModel):
    """
    Task 1 action: submit the fully reformatted manuscript.

    The entire manuscript is submitted as a single string.
    The grader computes a Progressive Reward Shaping score across 3 stages.
    """
    task: Literal["formatting_compliance"] = "formatting_compliance"
    formatted_text: str = Field(
        description="Complete reformatted manuscript as a single string."
    )


class ScholarAction(BaseModel):
    """
    Tasks 2 & 3 actions: navigate the paper or submit findings.

    Navigation actions (query_section, check_table, extract_claims) return
    intermediate content with a PBRS shaping bonus.
    submit_findings ends the episode and triggers the F-beta grader.
    """
    task: Literal["internal_consistency", "claim_evidence_audit"]
    action_type: Literal[
        "query_section",    # → current_section_content in obs
        "check_table",      # → current_table_content in obs
        "extract_claims",   # → extracted_claims (structured) in obs
        "submit_findings",  # → final F-beta score, done=True
    ]
    section_name: Optional[str] = Field(
        default=None,
        description="Required for query_section and extract_claims."
    )
    table_id: Optional[str] = Field(
        default=None,
        description="Required for check_table. E.g. 'Table 1', 'Table 2A'."
    )
    findings: Optional[list[dict]] = Field(
        default=None,
        description=(
            "Required for submit_findings. Each dict must contain: "
            "type (str), location (str), claim (str), contradicts (str). "
            "For Task 3, also include: table_id (str), table_value (str)."
        ),
    )


# Discriminated union — FastAPI deserialises on the 'task' field
AnyAction = Annotated[
    Union[FormattingAction, ScholarAction],
    Field(discriminator="task"),
]


# ── Observation ───────────────────────────────────────────────────────────────

class ScholarObservation(BaseModel):
    """Unified observation returned by reset() and every step()."""

    # ── Always present ──────────────────────────────────────────────────────
    task_id:          str
    task_description: str
    paper_id:         str
    step_count:       int   = 0
    max_steps:        int   = 3
    cumulative_score: float = 0.0
    feedback:         str   = ""
    hint:             str   = ""

    # ── Task 1 only ─────────────────────────────────────────────────────────
    manuscript_text: Optional[str] = Field(
        default=None,
        description="Badly-formatted manuscript (Task 1 initial observation)."
    )
    style_guide: Optional[dict] = Field(
        default=None,
        description="IEEE style rule config — keys mirror STAGE_CONFIG rules."
    )

    # ── Tasks 2 & 3 — navigation ────────────────────────────────────────────
    available_sections:      list[str]          = Field(default_factory=list)
    available_tables:        list[str]          = Field(default_factory=list)
    current_section_content: Optional[str]      = None
    current_table_content:   Optional[dict]     = None
    extracted_claims:        Optional[list[dict]] = None
    findings_so_far:         list[dict]          = Field(default_factory=list)


# ── Reward (logging / documentation only — not returned in step response) ─────

class ScholarReward(BaseModel):
    """Full reward breakdown — logged in step info dict."""

    total: float = Field(ge=0.0, le=1.0)

    # Task 1 — PRS stages (arxiv 2512.07478)
    stage_1_score: float = 0.0
    stage_2_score: float = 0.0
    stage_3_score: float = 0.0

    # Tasks 2 & 3 — F-beta components
    f_beta:               float = 0.0
    precision:            float = 0.0
    recall:               float = 0.0
    evidence_specificity: float = 0.0  # did agent cite table_id / location?
    coverage_bonus:       float = 0.0  # PBRS coverage at submission

    # Shaping (PBRS intermediate steps)
    shaping_bonus: float = 0.0

    rule_breakdown: dict[str, float] = Field(default_factory=dict)
