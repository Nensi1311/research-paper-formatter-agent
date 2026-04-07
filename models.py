"""
models.py — Canonical Pydantic models for ScholarEnv v0.4.

Authors: Nensi Pansuriya, Krushna Parmar, Ishita Bhojani

Design:
  1. AnyAction uses Annotated discriminated union on 'task' field.
  2. ScholarObservation covers all 4 tasks with Optional fields.
  3. CitationAction supports Task 4 (citation_verification).
  4. No circular imports — models.py imports nothing from server/.

References:
  PRS  — arxiv 2512.07478
  PBRS — Ng, Harada & Russell 1999
  AdaRFT — arxiv 2504.05520
  Veri-R1 — arxiv 2510.01932 (Task 4 design)
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
    """Task 1: submit the fully reformatted manuscript."""
    task: Literal["formatting_compliance"] = "formatting_compliance"
    formatted_text: str = Field(
        description="Complete reformatted manuscript as a single string."
    )


class ScholarAction(BaseModel):
    """Tasks 2 & 3: navigate the paper or submit findings."""
    task: Literal["internal_consistency", "claim_evidence_audit"]
    action_type: Literal[
        "query_section",
        "check_table",
        "extract_claims",
        "submit_findings",
    ]
    section_name: Optional[str] = Field(default=None)
    table_id:     Optional[str] = Field(default=None)
    findings:     Optional[list[dict]] = Field(default=None)


class CitationAction(BaseModel):
    """Task 4: verify citations in paper reference list."""
    task: Literal["citation_verification"] = "citation_verification"
    action_type: Literal[
        "check_citation",   # → returns citation_data in obs
        "submit_verdicts",  # → final grade, done=True
    ]
    citation_id: Optional[str] = Field(
        default=None,
        description="Reference ID for check_citation, e.g. 'ref_1'"
    )
    verdicts: Optional[list[dict]] = Field(
        default=None,
        description=(
            "For submit_verdicts. Each dict: "
            "citation_id, status (valid|ghost|misattributed), issue, confidence."
        ),
    )


# Discriminated union — FastAPI deserialises on the 'task' field
AnyAction = Annotated[
    Union[FormattingAction, ScholarAction, CitationAction],
    Field(discriminator="task"),
]


# ── Observation ───────────────────────────────────────────────────────────────

class ScholarObservation(BaseModel):
    """Unified observation returned by reset() and every step()."""

    # Always present
    task_id:          str
    task_description: str
    paper_id:         str
    step_count:       int   = 0
    max_steps:        int   = 3
    cumulative_score: float = 0.0
    feedback:         str   = ""
    hint:             str   = ""

    # Task 1 only
    manuscript_text: Optional[str] = Field(
        default=None,
        description="Badly-formatted manuscript (Task 1 initial observation)."
    )
    style_guide: Optional[dict] = Field(
        default=None,
        description="IEEE style rule config."
    )

    # Tasks 2 & 3 — navigation
    available_sections:      list[str]          = Field(default_factory=list)
    available_tables:        list[str]          = Field(default_factory=list)
    current_section_content: Optional[str]      = None
    current_table_content:   Optional[dict]     = None
    extracted_claims:        Optional[list[dict]] = None
    findings_so_far:         list[dict]          = Field(default_factory=list)

    # Task 4 — citation verification
    available_references: list[dict] = Field(
        default_factory=list,
        description="Task 4: list of {id, citation_number, raw} dicts."
    )
    citation_data: Optional[dict] = Field(
        default=None,
        description="Task 4: returned after check_citation action."
    )


# ── Reward (logging / documentation only) ────────────────────────────────────

class ScholarReward(BaseModel):
    """Full reward breakdown — logged in step info dict."""
    total: float = Field(ge=0.0, le=1.0)
    # Task 1 — PRS stages
    stage_1_score: float = 0.0
    stage_2_score: float = 0.0
    stage_3_score: float = 0.0
    # Tasks 2 & 3 — F-beta
    f_beta:               float = 0.0
    precision:            float = 0.0
    recall:               float = 0.0
    evidence_specificity: float = 0.0
    coverage_bonus:       float = 0.0
    shaping_bonus:        float = 0.0
    # Task 4 — citation
    precision_valid:      float = 0.0
    recall_ghost:         float = 0.0
    rule_breakdown: dict[str, float] = Field(default_factory=dict)
