"""
Deterministic graders for each task.
Each grader scores the paper's current state against the target format spec.
Returns decomposed partial-credit scores (0.0–1.0 per dimension).
"""

from __future__ import annotations
from typing import List, Tuple
from models import EpisodeState, PaperReward, SectionType, ConferenceFormat
from conference_specs import get_spec


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _section_types_present(state: EpisodeState) -> List[SectionType]:
    return [s.section_type for s in state.sections]


def _section_names_in_order(state: EpisodeState) -> List[str]:
    return list(state.section_order)


# ──────────────────────────────────────────────
# Individual dimension graders
# ──────────────────────────────────────────────

def score_section_structure(state: EpisodeState) -> Tuple[float, List[str]]:
    """
    Checks:
    1. All required sections present
    2. No forbidden sections present
    3. Section order matches spec (Kendall tau style)
    """
    spec = get_spec(state.target_format)
    issues = []
    present_types = _section_types_present(state)

    # Required sections
    missing = [s for s in spec.required_sections if s not in present_types]
    if missing:
        issues.append(f"Missing required sections: {[s.value for s in missing]}")

    # Forbidden sections
    forbidden_found = [s for s in spec.forbidden_sections if s in present_types]
    if forbidden_found:
        issues.append(f"Forbidden sections present: {[s.value for s in forbidden_found]}")

    # Section order — check that present required sections appear in spec order
    spec_order_types = spec.section_order
    present_in_spec_order = [st for st in spec_order_types if st in present_types]
    actual_order = [s.section_type for s in state.sections]
    actual_filtered = [t for t in actual_order if t in present_in_spec_order]

    order_correct = (actual_filtered == present_in_spec_order)
    if not order_correct:
        issues.append("Section ordering does not match spec")

    # Score
    total_checks = len(spec.required_sections) + len(spec.forbidden_sections) + 1
    penalties = len(missing) + len(forbidden_found) + (0 if order_correct else 1)
    score = max(0.0, 1.0 - penalties / total_checks)
    return round(score, 4), issues


def score_reference_format(state: EpisodeState) -> Tuple[float, List[str]]:
    """Checks whether all references use the target reference style."""
    spec = get_spec(state.target_format)
    issues = []
    if not state.references:
        return 1.0, []

    correctly_styled = sum(1 for r in state.references if r.style == spec.reference_style)
    score = correctly_styled / len(state.references)
    if score < 1.0:
        issues.append(f"References use wrong style. Expected {spec.reference_style}, found mixed styles.")
    return round(score, 4), issues


def score_abstract_compliance(state: EpisodeState) -> Tuple[float, List[str]]:
    """Checks abstract word count against spec limit."""
    spec = get_spec(state.target_format)
    issues = []
    limit = spec.max_abstract_words
    count = state.abstract_word_count

    if count <= limit:
        return 1.0, []

    # Partial credit: penalize proportionally to overage
    overage = count - limit
    penalty = min(overage / limit, 1.0)
    score = 1.0 - penalty
    issues.append(f"Abstract has {count} words; limit is {limit}")
    return round(score, 4), issues


def score_author_format(state: EpisodeState) -> Tuple[float, List[str]]:
    """Checks whether author names follow the spec format."""
    spec = get_spec(state.target_format)
    issues = []
    if not state.authors:
        return 1.0, []

    def _matches_format(name: str, fmt: str) -> bool:
        parts = name.strip().split()
        if fmt == "F. Last":
            # Expect single initial + period + last name, e.g. "A. Johnson"
            return (len(parts) == 2 and len(parts[0]) == 2 and parts[0].endswith("."))
        elif fmt == "First Last":
            # Expect full first name (>1 char, no period) + last name
            return (len(parts) >= 2 and len(parts[0]) > 1 and not parts[0].endswith("."))
        return True  # unknown format → pass

    correct = sum(1 for a in state.authors if _matches_format(a.name, spec.author_format))
    score = correct / len(state.authors)
    if score < 1.0:
        issues.append(f"Author names should follow '{spec.author_format}' format")
    return round(score, 4), issues


def score_layout(state: EpisodeState) -> Tuple[float, List[str]]:
    """Checks column layout."""
    spec = get_spec(state.target_format)
    issues = []
    if state.column_layout == spec.columns:
        return 1.0, []
    issues.append(f"Column layout is {state.column_layout}; spec requires {spec.columns}")
    return 0.0, issues


def score_citation_style(state: EpisodeState) -> Tuple[float, List[str]]:
    """Checks in-text citation style (numeric vs author_year)."""
    spec = get_spec(state.target_format)
    issues = []
    if state.citation_style == spec.citation_style:
        return 1.0, []
    issues.append(f"Citation style is '{state.citation_style}'; spec requires '{spec.citation_style}'")
    return 0.0, issues


def score_title_case(state: EpisodeState) -> Tuple[float, List[str]]:
    """Checks title case style of the paper."""
    spec = get_spec(state.target_format)
    issues = []
    if state.title_case_style == spec.title_case:
        return 1.0, []
    issues.append(f"Title case is '{state.title_case_style}'; spec requires '{spec.title_case}'")
    return 0.0, issues


# ──────────────────────────────────────────────
# Composite grader
# ──────────────────────────────────────────────

# Weights for the composite score (must sum to 1.0)
WEIGHTS = {
    "section_structure": 0.30,
    "reference_format": 0.20,
    "abstract_compliance": 0.15,
    "author_format": 0.15,
    "layout": 0.10,
    "citation_style": 0.10,
}


def compute_reward(state: EpisodeState, step_penalty_coeff: float = 0.002) -> PaperReward:
    """
    Compute the full reward for the current state.

    step_penalty_coeff: small penalty per step to encourage efficiency.
    """
    sec_score, sec_issues = score_section_structure(state)
    ref_score, ref_issues = score_reference_format(state)
    abs_score, abs_issues = score_abstract_compliance(state)
    auth_score, auth_issues = score_author_format(state)
    lay_score, lay_issues = score_layout(state)
    cit_score, cit_issues = score_citation_style(state)

    # Step penalty (small, linear)
    step_penalty = min(step_penalty_coeff * state.steps_taken, 0.10)

    # Weighted composite
    raw = (
        WEIGHTS["section_structure"] * sec_score
        + WEIGHTS["reference_format"] * ref_score
        + WEIGHTS["abstract_compliance"] * abs_score
        + WEIGHTS["author_format"] * auth_score
        + WEIGHTS["layout"] * lay_score
        + WEIGHTS["citation_style"] * cit_score
    )

    total = max(0.0, min(raw - step_penalty, 1.0))

    all_issues = sec_issues + ref_issues + abs_issues + auth_issues + lay_issues + cit_issues
    info = f"Issues: {all_issues}" if all_issues else "All checks passed"

    return PaperReward(
        total=round(total, 4),
        section_structure_score=sec_score,
        reference_format_score=ref_score,
        abstract_compliance_score=abs_score,
        author_format_score=auth_score,
        layout_score=lay_score,
        citation_style_score=cit_score,
        step_penalty=round(step_penalty, 4),
        info=info,
    )


def get_issues(state: EpisodeState) -> List[str]:
    """Return all current formatting violations."""
    _, si = score_section_structure(state)
    _, ri = score_reference_format(state)
    _, ai = score_abstract_compliance(state)
    _, athi = score_author_format(state)
    _, li = score_layout(state)
    _, ci = score_citation_style(state)
    _, ti = score_title_case(state)
    return si + ri + ai + athi + li + ci + ti
