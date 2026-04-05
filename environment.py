"""
Research Paper Formatter OpenEnv Environment

Implements the OpenEnv interface:
  - reset() → PaperObservation
  - step(action) → StepResult
  - state() → EpisodeState

Domain: Reformatting academic papers between conference/journal formats.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from models import (
    ActionType,
    AuthorInfo,
    ConferenceFormat,
    EpisodeState,
    PaperAction,
    PaperObservation,
    Reference,
    Section,
    SectionType,
    StepResult,
)
from conference_specs import get_spec
from grader import compute_reward, get_issues
from paper_data import get_paper
from tasks import TaskSpec, get_task


class PaperFormatterEnv:
    """
    OpenEnv-compliant environment for research paper formatting.

    The agent receives an observation describing the paper's current state
    relative to a target conference format, and must issue formatting actions
    to bring it into compliance.
    """

    VERSION = "1.0.0"
    ENV_ID = "paper-formatter-v1"

    def __init__(self, task_id: str = "task_easy"):
        self.task_spec: TaskSpec = get_task(task_id)
        self._state: Optional[EpisodeState] = None

    # ──────────────────────────────────────────────
    # OpenEnv API
    # ──────────────────────────────────────────────

    def reset(self) -> PaperObservation:
        """Reset the environment to the initial paper state."""
        paper = get_paper(self.task_spec.paper_id)
        spec = get_spec(self.task_spec.target_format)

        self._state = EpisodeState(
            paper_id=self.task_spec.paper_id,
            task_id=self.task_spec.task_id,
            current_format=paper["current_format"],
            target_format=self.task_spec.target_format,
            target_spec=spec,
            sections=paper["sections"],
            section_order=paper["section_order"],
            references=paper["references"],
            authors=paper["authors"],
            abstract_word_count=paper["abstract_word_count"],
            column_layout=paper["column_layout"],
            title_case_style=paper["title_case_style"],
            citation_style=paper["citation_style"],
            steps_taken=0,
            max_steps=self.task_spec.max_steps,
            cumulative_reward=0.0,
            action_history=[],
            issue_history=[],
            done=False,
        )

        initial_issues = get_issues(self._state)
        self._state.issue_history.append(initial_issues)

        return self._build_observation()

    def step(self, action: PaperAction) -> StepResult:
        """Apply an action and return the new observation, reward, done, info."""
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Record pre-step issues for delta calculation
        pre_issues = get_issues(self._state)

        # Apply the action
        error_msg: Optional[str] = None
        try:
            self._apply_action(action)
        except ValueError as e:
            error_msg = str(e)

        self._state.steps_taken += 1
        self._state.action_history.append({
            "step": self._state.steps_taken,
            "action_type": action.action_type.value,
            "parameters": action.parameters,
            "error": error_msg,
        })

        # Compute reward
        reward_obj = compute_reward(self._state)
        reward = reward_obj.total
        self._state.cumulative_reward += reward

        # Check done conditions
        post_issues = get_issues(self._state)
        self._state.issue_history.append(post_issues)

        done = (
            action.action_type == ActionType.SUBMIT
            or self._state.steps_taken >= self._state.max_steps
            or (len(post_issues) == 0)
        )
        self._state.done = done

        obs = self._build_observation()

        info: Dict[str, Any] = {
            "reward_breakdown": reward_obj.model_dump(),
            "issues_before": pre_issues,
            "issues_after": post_issues,
            "fixed_this_step": [i for i in pre_issues if i not in post_issues],
            "new_this_step": [i for i in post_issues if i not in pre_issues],
            "error": error_msg,
            "steps_taken": self._state.steps_taken,
            "max_steps": self._state.max_steps,
        }
        if done:
            info["final_score"] = reward
            info["success"] = reward >= self.task_spec.success_threshold

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> EpisodeState:
        """Return the full internal episode state (for debugging / logging)."""
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return copy.deepcopy(self._state)

    def close(self) -> None:
        """Clean up resources."""
        self._state = None

    # ──────────────────────────────────────────────
    # Action dispatch
    # ──────────────────────────────────────────────

    def _apply_action(self, action: PaperAction) -> None:
        dispatch = {
            ActionType.SET_FORMAT: self._act_set_format,
            ActionType.RENAME_SECTION: self._act_rename_section,
            ActionType.REORDER_SECTIONS: self._act_reorder_sections,
            ActionType.FORMAT_REFERENCES: self._act_format_references,
            ActionType.SET_TITLE_CASE: self._act_set_title_case,
            ActionType.SET_ABSTRACT_WORD_LIMIT: self._act_set_abstract_word_limit,
            ActionType.REMOVE_SECTION: self._act_remove_section,
            ActionType.ADD_SECTION: self._act_add_section,
            ActionType.FORMAT_AUTHOR_LIST: self._act_format_author_list,
            ActionType.SET_COLUMN_LAYOUT: self._act_set_column_layout,
            ActionType.FORMAT_CITATIONS: self._act_format_citations,
            ActionType.SUBMIT: self._act_submit,
        }
        handler = dispatch.get(action.action_type)
        if handler is None:
            raise ValueError(f"Unknown action type: {action.action_type}")
        handler(action.parameters)

    # ── Individual action handlers ──

    def _act_set_format(self, params: Dict[str, Any]) -> None:
        """Declare the target format (auto-applies many defaults)."""
        fmt_str = params.get("format", "")
        try:
            fmt = ConferenceFormat(fmt_str)
        except ValueError:
            raise ValueError(f"Unknown format: {fmt_str}. Valid: {[f.value for f in ConferenceFormat]}")
        self._state.target_format = fmt
        self._state.target_spec = get_spec(fmt)

    def _act_rename_section(self, params: Dict[str, Any]) -> None:
        """Rename a section by its current name."""
        old_name = params.get("old_name", "")
        new_name = params.get("new_name", "")
        if not old_name or not new_name:
            raise ValueError("rename_section requires 'old_name' and 'new_name'")

        found = False
        for sec in self._state.sections:
            if sec.name == old_name:
                sec.name = new_name
                found = True
                break
        if not found:
            raise ValueError(f"Section '{old_name}' not found")

        # Update order list
        self._state.section_order = [
            new_name if n == old_name else n
            for n in self._state.section_order
        ]

    def _act_reorder_sections(self, params: Dict[str, Any]) -> None:
        """Reorder sections by providing a new ordered list of section names."""
        new_order = params.get("order", [])
        if not new_order:
            raise ValueError("reorder_sections requires 'order' list")

        current_names = {s.name for s in self._state.sections}
        for name in new_order:
            if name not in current_names:
                raise ValueError(f"Section '{name}' not found in paper")

        # Reorder sections list
        name_to_section = {s.name: s for s in self._state.sections}
        # Keep sections not in new_order at the end
        extra = [s for s in self._state.sections if s.name not in new_order]
        self._state.sections = [name_to_section[n] for n in new_order] + extra
        self._state.section_order = new_order + [s.name for s in extra]

    def _act_format_references(self, params: Dict[str, Any]) -> None:
        """Change the style of all references."""
        style = params.get("style", "")
        if not style:
            raise ValueError("format_references requires 'style'")
        for ref in self._state.references:
            ref.style = style

    def _act_set_title_case(self, params: Dict[str, Any]) -> None:
        """Set the title case style."""
        style = params.get("style", "title_case")
        valid = {"title_case", "sentence_case", "upper", "lower"}
        if style not in valid:
            raise ValueError(f"Invalid title case style '{style}'. Valid: {valid}")
        self._state.title_case_style = style

    def _act_set_abstract_word_limit(self, params: Dict[str, Any]) -> None:
        """Trim abstract to a specified word count."""
        limit = params.get("limit")
        if limit is None:
            raise ValueError("set_abstract_word_limit requires 'limit'")
        limit = int(limit)
        if limit <= 0:
            raise ValueError("limit must be positive")
        # Cap the abstract word count (simulates actual trimming)
        self._state.abstract_word_count = min(self._state.abstract_word_count, limit)

    def _act_remove_section(self, params: Dict[str, Any]) -> None:
        """Remove a section by name."""
        name = params.get("name", "")
        if not name:
            raise ValueError("remove_section requires 'name'")
        original_len = len(self._state.sections)
        self._state.sections = [s for s in self._state.sections if s.name != name]
        self._state.section_order = [n for n in self._state.section_order if n != name]
        if len(self._state.sections) == original_len:
            raise ValueError(f"Section '{name}' not found")

    def _act_add_section(self, params: Dict[str, Any]) -> None:
        """Add a new section."""
        name = params.get("name", "")
        section_type_str = params.get("section_type", "")
        position = params.get("position", len(self._state.sections))

        if not name or not section_type_str:
            raise ValueError("add_section requires 'name' and 'section_type'")
        try:
            section_type = SectionType(section_type_str)
        except ValueError:
            raise ValueError(f"Unknown section type: {section_type_str}")

        new_section = Section(name=name, section_type=section_type)
        self._state.sections.insert(int(position), new_section)
        self._state.section_order.insert(int(position), name)

    def _act_format_author_list(self, params: Dict[str, Any]) -> None:
        """
        Change author name format.
        style: "F. Last" → abbreviated, "First Last" → full names
        For simulation, we convert between formats using heuristics.
        """
        style = params.get("style", "First Last")
        for author in self._state.authors:
            parts = author.name.strip().split()
            if style == "First Last":
                # Expand abbreviated names: "A. Johnson" → "Alice Johnson" (simulated)
                if len(parts) == 2 and len(parts[0]) == 2 and parts[0].endswith("."):
                    # Expand: use the initial as a stand-in for a full name
                    initial = parts[0][0]
                    last = parts[1]
                    # Simulated expansion: Initial + last name
                    author.name = f"{initial}name {last}"
            elif style == "F. Last":
                # Abbreviate: "Alice Johnson" → "A. Johnson"
                if len(parts) >= 2 and (len(parts[0]) > 1 or not parts[0].endswith(".")):
                    author.name = f"{parts[0][0]}. {parts[-1]}"

    def _act_set_column_layout(self, params: Dict[str, Any]) -> None:
        """Set column layout (1 or 2)."""
        columns = params.get("columns")
        if columns is None:
            raise ValueError("set_column_layout requires 'columns'")
        columns = int(columns)
        if columns not in (1, 2):
            raise ValueError("columns must be 1 or 2")
        self._state.column_layout = columns

    def _act_format_citations(self, params: Dict[str, Any]) -> None:
        """Switch in-text citation style."""
        style = params.get("style", "numeric")
        valid = {"numeric", "author_year"}
        if style not in valid:
            raise ValueError(f"citation style must be one of {valid}")
        self._state.citation_style = style

    def _act_submit(self, params: Dict[str, Any]) -> None:
        """Agent declares the paper ready. Triggers episode end."""
        pass  # done flag is set in step()

    # ──────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────

    def _build_observation(self) -> PaperObservation:
        s = self._state
        current_issues = get_issues(s)
        fixed = []
        if len(s.issue_history) >= 2:
            initial = s.issue_history[0]
            fixed = [i for i in initial if i not in current_issues]

        reward_obj = compute_reward(s)

        return PaperObservation(
            paper_id=s.paper_id,
            paper_title=get_paper(s.paper_id)["title"],
            current_format=s.current_format,
            target_format=s.target_format,
            target_spec=s.target_spec,
            sections=s.sections,
            section_order=s.section_order,
            references=s.references,
            authors=s.authors,
            abstract_word_count=s.abstract_word_count,
            column_layout=s.column_layout,
            title_case_style=s.title_case_style,
            citation_style=s.citation_style,
            compliance_score=reward_obj.total,
            issues=current_issues,
            fixed_issues=fixed,
            steps_taken=s.steps_taken,
            max_steps=s.max_steps,
            task_id=s.task_id,
            done=s.done,
        )
