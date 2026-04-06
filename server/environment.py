"""
server/environment.py — ScholarEnvironment: production-grade OpenEnv environment.

Architecture decisions:

  1. Pure Python class — no FastAPI imports here.
     This makes it fully testable in isolation and keeps concerns separated.
     FastAPI wiring lives in server/app.py.

  2. Explicit REST endpoints (POST /reset, POST /step, GET /state, GET /health)
     rather than a magic base class.  More portable, easier to debug.

  3. Task 2 is MULTI-STEP (max 4 steps): agent can query_section before
     submit_findings.  Single-shot Task 2 would have been unresolvable —
     the agent needs to actually read sections to find contradictions.

  4. PBRS (Potential-Based Reward Shaping) provides dense intermediate rewards
     for Tasks 2 & 3 navigation steps.  Terminal steps use the F-beta graders.

  5. UCB1 + AdaRFT curriculum selects papers.  Not random sampling.

  6. All state is in EpisodeState — reset() always produces a clean slate.
     No global mutable state beyond the curriculum (which is intentional).

Episode state machine:
    IDLE → reset() → ACTIVE → step() × N → DONE → reset() → ACTIVE
    Calling step() in IDLE or after DONE returns an error dict.
"""
from __future__ import annotations

import sys
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Allow running from the root directory without installing
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import (
    EpisodeStatus, FormattingAction, ScholarAction, ScholarObservation,
)
from corpus import PaperCorpus, Paper
from server.curriculum import Curriculum
from server.reward_shaper import NavigationState, PotentialBasedShaper
from server.graders import FormattingGrader, ConsistencyGrader, AuditGrader


# ── Task configuration ────────────────────────────────────────────────────────

TASK_CONFIG: dict[str, dict] = {
    "formatting_compliance": {
        "max_steps":         3,
        "allows_navigation": False,
        "description": (
            "Reformat the manuscript to comply with IEEE style. Fix: "
            "title (≤15 words), abstract (150–250 words, no citations), "
            "sections in required order (Abstract → Introduction → Methods → "
            "Results → Discussion → References), figure captions (Fig. N. format), "
            "table captions (Table N: format), in-text citations ([N] format), "
            "keywords section, and author block."
        ),
    },
    "internal_consistency": {
        "max_steps":         4,
        "allows_navigation": True,
        "description": (
            "Find all internal contradictions in this paper — places where it "
            "contradicts itself without external knowledge. Look for: number "
            "mismatches between sections (e.g. abstract says 94.3%, Table 2 "
            "says 91.7%), references to nonexistent figures/tables, inconsistent "
            "contribution counts, unresolved placeholder text. "
            "Use query_section to read sections, then submit_findings with your "
            "complete list. F-beta (β=0.5) rewards precision: only report "
            "inconsistencies you can specifically locate."
        ),
    },
    "claim_evidence_audit": {
        "max_steps":         6,
        "allows_navigation": True,
        "description": (
            "Audit whether numerical claims in the paper text match the tables "
            "and figures they cite. Some discrepancies are deliberately injected "
            "— find them all. Navigate strategically: query_section to read "
            "sections, check_table to inspect table data, extract_claims to get "
            "structured numerical claims. Submit all confirmed discrepancies via "
            "submit_findings. Include 'table_id' and 'table_value' in each finding "
            "for full evidence specificity credit."
        ),
    },
}


# ── Episode state ─────────────────────────────────────────────────────────────

@dataclass
class EpisodeState:
    episode_id:    str           = field(default_factory=lambda: str(uuid.uuid4()))
    task_id:       str           = ""
    paper_id:      str           = ""
    status:        EpisodeStatus = EpisodeStatus.ACTIVE
    step_count:    int           = 0
    max_steps:     int           = 3
    nav_state:     NavigationState = field(default_factory=NavigationState)
    findings:      list[dict]    = field(default_factory=list)
    prev_phi:      float         = 0.0
    score_history: list[float]   = field(default_factory=list)
    started_at:    float         = field(default_factory=time.time)

    def tick(self) -> None:
        self.step_count += 1

    def is_done(self) -> bool:
        return self.status == EpisodeStatus.DONE


# ── Main environment class ────────────────────────────────────────────────────

class ScholarEnvironment:
    """
    Production-grade OpenEnv environment for scholarly integrity verification.

    Exposed via FastAPI in server/app.py.
    This class is pure Python — no web framework dependencies.
    """

    DATA_DIR = "data"

    def __init__(self, data_dir: str | None = None) -> None:
        d = data_dir or self.DATA_DIR
        self.corpus     = PaperCorpus.load(f"{d}/papers")
        self.curriculum = Curriculum()
        self.graders = {
            "formatting_compliance": FormattingGrader(f"{d}/styles/ieee.yaml"),
            "internal_consistency":  ConsistencyGrader(),
            "claim_evidence_audit":  AuditGrader(),
        }
        self._episode: EpisodeState | None = None

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, task_id: str = "formatting_compliance") -> dict:
        if task_id not in TASK_CONFIG:
            return {
                "error": (
                    f"Unknown task_id '{task_id}'. "
                    f"Valid: {list(TASK_CONFIG.keys())}"
                )
            }
        cfg      = TASK_CONFIG[task_id]
        paper_id = self.curriculum.select(self.corpus, task_id)
        paper    = self.corpus.papers[paper_id]

        nav = NavigationState(
            total_sections=len(paper.sections),
            total_tables=len(paper.tables),
        )
        self._episode = EpisodeState(
            task_id=task_id,
            paper_id=paper_id,
            max_steps=cfg["max_steps"],
            nav_state=nav,
        )
        obs = self._initial_obs(paper, task_id, cfg)
        return {
            "observation": obs.model_dump(),
            "info": {
                "episode_id": self._episode.episode_id,
                "task_id":    task_id,
                "paper_id":   paper_id,
                "curriculum": self.curriculum.summary(),
            },
        }

    def step(self, action_dict: dict) -> dict:
        if self._episode is None or self._episode.is_done():
            return {"error": "Call /reset before /step, or episode is already done."}

        ep    = self._episode
        ep.tick()
        paper = self.corpus.papers[ep.paper_id]
        task  = action_dict.get("task", "")

        # Route by task / action_type
        if task == "formatting_compliance":
            try:
                action = FormattingAction(**action_dict)
            except Exception as e:
                return {"error": f"Invalid FormattingAction: {e}"}
            return self._step_formatting(action, paper, ep)

        action_type = action_dict.get("action_type", "")

        if action_type in ("query_section", "check_table", "extract_claims"):
            try:
                action = ScholarAction(**action_dict)
            except Exception as e:
                return {"error": f"Invalid ScholarAction: {e}"}
            return self._step_navigate(action, paper, ep)

        if action_type == "submit_findings":
            try:
                action = ScholarAction(**action_dict)
            except Exception as e:
                return {"error": f"Invalid ScholarAction: {e}"}
            return self._step_submit(action, paper, ep)

        return {"error": f"Unknown action. task='{task}' action_type='{action_type}'"}

    def state(self) -> dict:
        if self._episode is None:
            return {"status": "idle", "episode_id": None}
        ep = self._episode
        return {
            "episode_id": ep.episode_id,
            "task_id":    ep.task_id,
            "paper_id":   ep.paper_id,
            "step_count": ep.step_count,
            "max_steps":  ep.max_steps,
            "status":     ep.status.value,
            "curriculum": self.curriculum.summary(),
            "nav_coverage": {
                "sections_read":  list(ep.nav_state.sections_read),
                "tables_checked": list(ep.nav_state.tables_checked),
            },
        }

    # ── Task 1: formatting compliance ─────────────────────────────────────────

    def _step_formatting(
        self, action: FormattingAction, paper: Paper, ep: EpisodeState
    ) -> dict:
        result = self.graders["formatting_compliance"].grade(
            action.formatted_text, paper
        )
        ep.score_history.append(result.score)

        done = ep.step_count >= ep.max_steps or result.score >= 0.95
        if done:
            ep.status = EpisodeStatus.DONE
            self.curriculum.update(
                ep.paper_id, ep.task_id, result.score, result.rule_results
            )

        obs = ScholarObservation(
            task_id=ep.task_id,
            task_description=TASK_CONFIG[ep.task_id]["description"],
            paper_id=paper.id,
            manuscript_text=action.formatted_text if not done else None,
            style_guide=self.graders["formatting_compliance"].style_config,
            step_count=ep.step_count,
            max_steps=ep.max_steps,
            feedback=result.hint(),
            hint=self.curriculum.hint(ep.paper_id),
            cumulative_score=result.score,
        )
        return {
            "observation": obs.model_dump(),
            "reward":      result.score,
            "done":        done,
            "info": {
                "stage_1":        result.stage_1_score,
                "stage_2":        result.stage_2_score,
                "stage_3":        result.stage_3_score,
                "failed_rules":   result.failed_rules,
                "rule_breakdown": result.rule_results,
            },
        }

    # ── Navigation (Tasks 2 & 3) ──────────────────────────────────────────────

    def _step_navigate(
        self, action: ScholarAction, paper: Paper, ep: EpisodeState
    ) -> dict:
        phi_before       = ep.prev_phi
        content          = None
        table_data       = None
        extracted_claims = None
        extra_info: dict[str, Any] = {}

        if action.action_type == "query_section":
            sec = action.section_name or ""
            content = paper.get_section(sec)
            if content:
                ep.nav_state.record_section(sec)
                extra_info["section"] = sec
            else:
                content = (
                    f"Section '{sec}' not found. "
                    f"Available: {paper.section_names}"
                )

        elif action.action_type == "check_table":
            tid = action.table_id or ""
            table_data = paper.get_table(tid)
            if table_data:
                ep.nav_state.record_table(tid)
                extra_info["table_id"] = tid
            else:
                table_data = {
                    "error": (
                        f"Table '{tid}' not found. "
                        f"Available: {paper.table_names}"
                    )
                }

        elif action.action_type == "extract_claims":
            sec  = action.section_name or ""
            text = paper.get_section(sec) or ""
            if text:
                from server.graders.audit_grader import ClaimExtractor
                claims = ClaimExtractor().extract(text, section_name=sec)
                extracted_claims = claims
                ep.nav_state.record_claims(len(claims))
                extra_info["n_claims"] = len(claims)
            else:
                extracted_claims = []

        # PBRS intermediate reward
        shaper     = PotentialBasedShaper(ep.nav_state)
        phi_after  = shaper.potential()
        shaping_bonus = shaper.shaping_bonus(phi_before, phi_after)
        ep.prev_phi   = phi_after

        done = ep.step_count >= ep.max_steps
        if done:
            ep.status = EpisodeStatus.DONE

        obs = ScholarObservation(
            task_id=ep.task_id,
            task_description=TASK_CONFIG[ep.task_id]["description"],
            paper_id=paper.id,
            available_sections=paper.section_names,
            available_tables=paper.table_names,
            current_section_content=content,
            current_table_content=table_data,
            extracted_claims=extracted_claims,
            step_count=ep.step_count,
            max_steps=ep.max_steps,
            findings_so_far=ep.findings,
            hint=self.curriculum.hint(ep.paper_id),
        )
        return {
            "observation": obs.model_dump(),
            "reward":      shaping_bonus,
            "done":        done,
            "info": {
                "action_type":   action.action_type,
                "shaping_bonus": shaping_bonus,
                "phi":           phi_after,
                **extra_info,
            },
        }

    # ── Submission (Tasks 2 & 3) ──────────────────────────────────────────────

    def _step_submit(
        self, action: ScholarAction, paper: Paper, ep: EpisodeState
    ) -> dict:
        findings    = action.findings or []
        ep.findings = findings
        ep.status   = EpisodeStatus.DONE

        if action.task == "internal_consistency":
            result = self.graders["internal_consistency"].grade(
                findings, paper, ep.step_count
            )
            info = {
                "f_beta":          result.f_beta,
                "precision":       result.precision,
                "recall":          result.recall,
                "tier_breakdown":  result.tier_breakdown,
                "missed":          result.missed_ids,
                "rule_breakdown":  result.rule_results,
            }
        else:
            result = self.graders["claim_evidence_audit"].grade(
                findings, paper, ep.nav_state
            )
            info = {
                "f_beta":                result.f_beta,
                "precision":             result.precision,
                "recall":                result.recall,
                "evidence_specificity":  result.evidence_specificity,
                "coverage_bonus":        result.coverage_bonus,
                "missed":                result.missed_ids,
                "rule_breakdown":        result.rule_results,
            }

        self.curriculum.update(
            ep.paper_id, ep.task_id, result.score, result.rule_results
        )

        obs = ScholarObservation(
            task_id=ep.task_id,
            task_description=TASK_CONFIG[ep.task_id]["description"],
            paper_id=paper.id,
            step_count=ep.step_count,
            max_steps=ep.max_steps,
            findings_so_far=findings,
            feedback=result.hint(),
            cumulative_score=result.score,
        )
        return {
            "observation": obs.model_dump(),
            "reward":      result.score,
            "done":        True,
            "info":        info,
        }

    # ── Initial observation builder ───────────────────────────────────────────

    def _initial_obs(
        self, paper: Paper, task_id: str, cfg: dict
    ) -> ScholarObservation:
        if task_id == "formatting_compliance":
            ms_text = paper.badly_formatted_text or self._rebuild_badly_formatted(paper)
            return ScholarObservation(
                task_id=task_id,
                task_description=cfg["description"],
                paper_id=paper.id,
                manuscript_text=ms_text,
                style_guide=self.graders["formatting_compliance"].style_config,
                step_count=0,
                max_steps=cfg["max_steps"],
                hint=self.curriculum.hint(paper.id),
            )
        return ScholarObservation(
            task_id=task_id,
            task_description=cfg["description"],
            paper_id=paper.id,
            available_sections=paper.section_names,
            available_tables=paper.table_names,
            step_count=0,
            max_steps=cfg["max_steps"],
            hint=self.curriculum.hint(paper.id),
        )

    @staticmethod
    def _rebuild_badly_formatted(paper: Paper) -> str:
        """
        Synthesise a badly-formatted manuscript from a well-structured paper.
        Applies common violations: wrong citation style, section order reversed,
        abstract too long (padded), missing keywords section.
        """
        import re
        parts = [paper.title, ""]

        # Reverse section order to violate ordering rule
        sections = list(paper.sections.items())
        sections_reordered = sections[::-1] if len(sections) > 2 else sections

        for name, content in sections_reordered:
            parts.append(name.upper())
            # Convert [N] citations to (Author, Year) style — violates IEEE
            corrupted = re.sub(
                r'\[(\d+)\]',
                lambda m: f'(Author, 200{m.group(1)[-1]})',
                content,
            )
            parts.append(corrupted)
            parts.append("")

        # Pad the abstract if it exists — violate word count
        result = "\n".join(parts)
        abstract_match = re.search(r'ABSTRACT\n(.*?)(?=\n[A-Z]+\n)', result, re.S)
        if abstract_match:
            abstract_text = abstract_match.group(1)
            padding = (
                " This study contributes to the broader understanding of the field "
                "and opens avenues for future research directions. The implications "
                "are significant and far-reaching across multiple domains."
            ) * 3
            result = result.replace(abstract_text, abstract_text + padding)

        return result
