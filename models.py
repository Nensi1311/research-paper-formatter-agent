"""
Typed Pydantic models for the Research Paper Formatter OpenEnv environment.
Defines Observation, Action, Reward, and State models per OpenEnv spec.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class ConferenceFormat(str, enum.Enum):
    IEEE = "IEEE"
    ACM = "ACM"
    NeurIPS = "NeurIPS"
    ICML = "ICML"
    AAAI = "AAAI"
    ARXIV = "arXiv"
    SPRINGER = "Springer"
    ELSEVIER = "Elsevier"


class SectionType(str, enum.Enum):
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    ACKNOWLEDGMENTS = "acknowledgments"
    APPENDIX = "appendix"


class ActionType(str, enum.Enum):
    SET_FORMAT = "set_format"
    RENAME_SECTION = "rename_section"
    REORDER_SECTIONS = "reorder_sections"
    FORMAT_REFERENCES = "format_references"
    SET_TITLE_CASE = "set_title_case"
    SET_ABSTRACT_WORD_LIMIT = "set_abstract_word_limit"
    REMOVE_SECTION = "remove_section"
    ADD_SECTION = "add_section"
    FORMAT_AUTHOR_LIST = "format_author_list"
    SET_COLUMN_LAYOUT = "set_column_layout"
    FORMAT_CITATIONS = "format_citations"
    SUBMIT = "submit"


# ──────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────

class Section(BaseModel):
    name: str
    section_type: SectionType
    word_count: int = 0
    content_snippet: str = ""  # first 200 chars
    has_figures: bool = False
    has_tables: bool = False
    has_equations: bool = False


class Reference(BaseModel):
    index: int
    authors: List[str]
    title: str
    venue: str
    year: int
    style: str = "unknown"  # e.g. "IEEE", "APA", "MLA"


class AuthorInfo(BaseModel):
    name: str
    affiliation: str
    email: Optional[str] = None
    order: int = 0


class FormatSpec(BaseModel):
    """Describes what a valid submission looks like for a given conference."""
    conference: ConferenceFormat
    required_sections: List[SectionType]
    forbidden_sections: List[SectionType] = Field(default_factory=list)
    section_order: List[SectionType]
    max_abstract_words: int
    reference_style: str          # e.g. "IEEE", "ACL", "APA"
    author_format: str            # e.g. "First Last", "F. Last"
    columns: int                  # 1 or 2
    title_case: str               # "title_case" | "sentence_case" | "upper"
    max_pages: Optional[int] = None
    citation_style: str = "numeric"  # "numeric" | "author_year"


# ──────────────────────────────────────────────
# Core OpenEnv models
# ──────────────────────────────────────────────

class PaperObservation(BaseModel):
    """What the agent sees at each step."""

    # Paper identity
    paper_id: str
    paper_title: str

    # Current state of the paper
    current_format: Optional[ConferenceFormat]
    target_format: ConferenceFormat
    target_spec: FormatSpec

    # Structural state
    sections: List[Section]
    section_order: List[str]          # current section names in order
    references: List[Reference]
    authors: List[AuthorInfo]
    abstract_word_count: int
    column_layout: int                # 1 or 2
    title_case_style: str
    citation_style: str

    # Progress signals
    compliance_score: float           # 0.0–1.0 running compliance
    issues: List[str]                 # list of current format violations
    fixed_issues: List[str]           # issues resolved so far
    steps_taken: int
    max_steps: int

    # Episode info
    task_id: str
    done: bool = False


class PaperAction(BaseModel):
    """Actions the agent can take to format the paper."""

    action_type: ActionType
    parameters: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "examples": [
                {"action_type": "set_format", "parameters": {"format": "IEEE"}},
                {"action_type": "rename_section", "parameters": {"old_name": "Experiments", "new_name": "Evaluation"}},
                {"action_type": "reorder_sections", "parameters": {"order": ["introduction", "related_work", "methodology"]}},
                {"action_type": "format_references", "parameters": {"style": "IEEE"}},
                {"action_type": "submit", "parameters": {}},
            ]
        }


class PaperReward(BaseModel):
    """Reward with decomposed partial-credit signals."""

    total: float = Field(ge=0.0, le=1.0)

    # Decomposed components (each 0.0–1.0)
    section_structure_score: float = 0.0
    reference_format_score: float = 0.0
    abstract_compliance_score: float = 0.0
    author_format_score: float = 0.0
    layout_score: float = 0.0
    citation_style_score: float = 0.0

    # Step penalty (discourages wasted actions)
    step_penalty: float = 0.0

    # Bonus for clean single-step fix
    efficiency_bonus: float = 0.0

    info: str = ""


class EpisodeState(BaseModel):
    """Full internal state — returned by state()."""

    paper_id: str
    task_id: str
    current_format: Optional[ConferenceFormat]
    target_format: ConferenceFormat
    target_spec: FormatSpec
    sections: List[Section]
    section_order: List[str]
    references: List[Reference]
    authors: List[AuthorInfo]
    abstract_word_count: int
    column_layout: int
    title_case_style: str
    citation_style: str
    steps_taken: int
    max_steps: int
    cumulative_reward: float
    action_history: List[Dict[str, Any]]
    issue_history: List[List[str]]
    done: bool


class StepResult(BaseModel):
    """Return value of step()."""

    observation: PaperObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
