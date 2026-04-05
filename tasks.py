"""
Task definitions for the Research Paper Formatter environment.

Three tasks with increasing difficulty:
  Task 1 (EASY):   NeurIPS → IEEE — fix abstract length, column layout, citation style
  Task 2 (MEDIUM): ACM → NeurIPS — fix author format, reference style, section names, citation style
  Task 3 (HARD):   IEEE → ICML — fix title case, references, author format, citation style,
                                  section names, section order, plus multiple distractors
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from models import ConferenceFormat


@dataclass
class TaskSpec:
    task_id: str
    name: str
    description: str
    difficulty: str          # "easy" | "medium" | "hard"
    paper_id: str
    source_format: ConferenceFormat
    target_format: ConferenceFormat
    max_steps: int
    issues_to_fix: List[str]
    success_threshold: float  # score >= this is considered success
    hints: List[str] = field(default_factory=list)


TASKS: Dict[str, TaskSpec] = {

    "task_easy": TaskSpec(
        task_id="task_easy",
        name="NeurIPS to IEEE Conversion",
        description=(
            "A paper originally formatted for NeurIPS must be reformatted for IEEE submission. "
            "The agent must: (1) trim the abstract to ≤150 words, "
            "(2) switch from 1-column to 2-column layout, "
            "(3) convert author-year citations to numeric [1] style, "
            "and (4) ensure references use IEEE citation style."
        ),
        difficulty="easy",
        paper_id="paper_neurips_2024",
        source_format=ConferenceFormat.NeurIPS,
        target_format=ConferenceFormat.IEEE,
        max_steps=8,
        issues_to_fix=[
            "Abstract exceeds IEEE 150-word limit (current: 220 words)",
            "Column layout is 1 (IEEE requires 2)",
            "Citation style is 'author_year' (IEEE requires 'numeric')",
            "References use APA style (IEEE requires IEEE style)",
        ],
        success_threshold=0.85,
        hints=[
            "Use set_abstract_word_limit to trim the abstract",
            "Use set_column_layout with columns=2",
            "Use format_citations to switch to numeric style",
            "Use format_references with style='IEEE'",
        ],
    ),

    "task_medium": TaskSpec(
        task_id="task_medium",
        name="ACM Systems to NeurIPS Conversion",
        description=(
            "A systems paper from ACM must be converted to NeurIPS format. "
            "Issues include: numbered section names (must be removed), "
            "two 'Related Work' sections (one labelled 'Background'), "
            "author names using abbreviated initials (NeurIPS needs full first names), "
            "and numeric citations (NeurIPS uses author-year). "
            "The agent must detect and fix all issues across 6+ action types."
        ),
        difficulty="medium",
        paper_id="paper_acm_systems",
        source_format=ConferenceFormat.ACM,
        target_format=ConferenceFormat.NeurIPS,
        max_steps=12,
        issues_to_fix=[
            "Author names use 'F. Last' format (NeurIPS requires full first name)",
            "Section names have numeric prefixes ('1. Introduction' → 'Introduction')",
            "Duplicate Related Work sections ('Background' should be merged/renamed)",
            "Citation style is 'numeric' (NeurIPS requires 'author_year')",
            "References use ACM style (NeurIPS requires APA)",
            "Column layout is 2 (NeurIPS requires 1-column)",
        ],
        success_threshold=0.80,
        hints=[
            "Rename sections to remove numeric prefixes",
            "Use format_author_list to expand abbreviated names",
            "NeurIPS is single-column — use set_column_layout",
            "Switch citation style to author_year",
        ],
    ),

    "task_hard": TaskSpec(
        task_id="task_hard",
        name="IEEE to ICML Full Reformat",
        description=(
            "A paper formatted for IEEE must be completely overhauled for ICML submission. "
            "Problems: ALL-CAPS title (must be title case), IEEE section naming conventions "
            "('I. Introduction' → 'Introduction'), wrong citation style, wrong reference style, "
            "author name format (IEEE 'F. Last' → ICML 'First Last'), and the section order "
            "has 'Discussion' before 'Conclusion' which is non-standard for ICML. "
            "The agent must correctly sequence 8+ distinct formatting operations without "
            "introducing new violations."
        ),
        difficulty="hard",
        paper_id="paper_ieee_ml",
        source_format=ConferenceFormat.IEEE,
        target_format=ConferenceFormat.ICML,
        max_steps=15,
        issues_to_fix=[
            "Title is ALL CAPS (ICML requires standard title case)",
            "Section names use IEEE roman numeral convention ('I. Introduction')",
            "Author names are abbreviated ('H. Liu' → 'Hanxiao Liu')",
            "Citation style is 'numeric' (ICML requires 'author_year')",
            "References use IEEE style (ICML requires APA)",
            "Section 'Discussion' should be renamed to 'Results' and placed before 'Conclusion'",
            "Abstract is 145 words — within IEEE limit but within ICML limit too ✓",
        ],
        success_threshold=0.75,
        hints=[
            "Start with set_title_case to fix the ALL CAPS title",
            "Rename each section to remove roman numeral prefixes",
            "Expand author names using format_author_list",
            "Fix citations: ICML uses (Liu et al., 2019) style",
            "Reorder sections: 'Discussion' → 'Results', placed after 'Experiments'",
        ],
    ),
}


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> List[str]:
    return list(TASKS.keys())
