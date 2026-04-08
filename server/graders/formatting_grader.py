"""
server/graders/formatting_grader.py — Task 1: formatting_compliance grader.

Implements Progressive Reward Shaping (PRS) from arxiv 2512.07478:
  Stage N rules only contribute reward when Stage N-1 score ≥ threshold.
  This prevents agents gaming easy rules while ignoring foundational structure,
  and produces a genuine gradient surface for GRPO training.

All checks are deterministic: regex, word count, or string comparison.
No LLM judge. Every check returns bool → float 0.0 or 1.0.

Stage design rationale:
  Stage 1 — basic structure (always active, weight 0.40)
    Checks a rule-following agent can parse section structure.
    Expected GPT-4o score: 0.85-1.0 (easy baseline sanity check).

  Stage 2 — section compliance (active when Stage 1 ≥ 0.60, weight 0.35)
    Ordering, per-section word limits, figure/table captions.
    Requires structural understanding, not just text manipulation.

  Stage 3 — IEEE style details (active when Stage 2 ≥ 0.70, weight 0.25)
    Citation format, author block, keywords, no citations in abstract.
    Genuine challenge — GPT-4o scores 0.30-0.60 on a first pass.
"""
from __future__ import annotations

import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...corpus import Paper


@dataclass
class FormattingGradeResult:
    score:         float
    stage_1_score: float
    stage_2_score: float
    stage_3_score: float
    rule_results:  dict[str, bool]
    failed_rules:  list[str]
    details:       dict

    def hint(self) -> str:
        if not self.failed_rules:
            return "All formatting checks passed. Excellent work."
        rules = ", ".join(self.failed_rules[:5])
        return (
            f"Still failing: {rules}. "
            "Fix Stage 1 rules first before Stage 2/3 will contribute to score."
        )


class FormattingGrader:
    """
    Progressive reward shaper for Task 1 (formatting_compliance).

    Stage thresholds are deliberately asymmetric:
      Stage 1 → 2: 0.60  (must get majority of basics right)
      Stage 2 → 3: 0.70  (must get section structure mostly right)
    This creates a curriculum effect even within a single episode.
    """

    STAGE_CONFIG: dict[int, dict] = {
        1: {
            "weight":    0.40,
            "threshold": 0.0,   # always active
            "rules": [
                "title_exists",
                "title_word_count",
                "abstract_exists",
                "abstract_min_words",
                "abstract_max_words",
                "has_introduction",
                "has_methods",
                "has_results",
                "has_discussion",
                "has_references",
            ],
        },
        2: {
            "weight":    0.35,
            "threshold": 0.60,
            "rules": [
                "section_order_correct",
                "intro_word_limit",
                "methods_word_limit",
                "results_word_limit",
                "figure_caption_format",
                "table_caption_format",
            ],
        },
        3: {
            "weight":    0.25,
            "threshold": 0.70,
            "rules": [
                "citation_format_ieee",
                "no_citations_in_abstract",
                "author_block_present",
                "keywords_section_present",
                "references_numbered",
            ],
        },
    }

    REQUIRED_SECTION_ORDER = [
        "abstract", "introduction", "methods", "results", "discussion", "references",
    ]

    WORD_LIMITS = {
        "introduction": 800,
        "methods":      1200,
        "results":      1000,
        "discussion":   800,
    }

    def __init__(self, style_path: str = "data/styles/ieee.yaml") -> None:
        self._style_path = Path(style_path)
        self.style_config: dict = {}
        if self._style_path.exists():
            with self._style_path.open() as f:
                self.style_config = yaml.safe_load(f) or {}

    # ── Public API ────────────────────────────────────────────────────────────

    def grade(self, text: str, paper: "Paper") -> FormattingGradeResult:
        all_results = self._run_all_checks(text)
        stage_scores = self._compute_staged(all_results)

        total = round(
            sum(stage_scores[n] * cfg["weight"]
                for n, cfg in self.STAGE_CONFIG.items()),
            4,
        )
        failed = [r for r, v in all_results.items() if not v]

        return FormattingGradeResult(
            score=total,
            stage_1_score=stage_scores[1],
            stage_2_score=stage_scores[2],
            stage_3_score=stage_scores[3],
            rule_results=all_results,
            failed_rules=failed,
            details={
                "title":               self._extract_title(text)[:80],
                "abstract_word_count": self._abstract_word_count(text),
                "stage_scores":        stage_scores,
            },
        )

    # ── Stage computation ─────────────────────────────────────────────────────

    def _compute_staged(self, results: dict[str, bool]) -> dict[int, float]:
        stage_scores: dict[int, float] = {}
        prev_score = 1.0   # Stage 1 always unlocked

        for stage_num, cfg in self.STAGE_CONFIG.items():
            if prev_score >= cfg["threshold"]:
                relevant = {k: v for k, v in results.items()
                            if k in cfg["rules"]}
                score = (sum(relevant.values()) / len(relevant)
                         if relevant else 0.0)
            else:
                score = 0.0  # locked — fix previous stage first

            stage_scores[stage_num] = round(score, 4)
            prev_score = score

        return stage_scores

    # ── All individual checks ─────────────────────────────────────────────────

    def _run_all_checks(self, text: str) -> dict[str, bool]:
        title    = self._extract_title(text)
        abstract = self._extract_abstract(text)
        sections = self._extract_sections_ordered(text)

        results: dict[str, bool] = {}

        # ── Stage 1: basic structure ─────────────────────────────────────────
        results["title_exists"]      = bool(title.strip())
        results["title_word_count"]  = len(title.split()) <= 15

        results["abstract_exists"]   = bool(abstract.strip())
        abs_wc = len(abstract.split())
        results["abstract_min_words"] = abs_wc >= 150
        results["abstract_max_words"] = abs_wc <= 250

        sec_lower = [s.lower() for s in sections]
        results["has_introduction"] = any("introduction" in s for s in sec_lower)
        results["has_methods"]      = any(
            s in ("methods", "methodology", "method") for s in sec_lower)
        results["has_results"]      = any("result" in s for s in sec_lower)
        results["has_discussion"]   = any(
            s in ("discussion", "conclusion", "conclusions") for s in sec_lower)
        results["has_references"]   = any("reference" in s for s in sec_lower)

        # ── Stage 2: section compliance ──────────────────────────────────────
        results["section_order_correct"] = self._check_section_order(sec_lower)

        sec_contents = self._extract_sections_with_content(text)
        for sec, limit in self.WORD_LIMITS.items():
            content = ""
            for k, v in sec_contents.items():
                if sec in k.lower():
                    content = v
                    break
            wc = len(content.split())
            results[f"{sec}_word_limit"] = (wc == 0 or wc <= limit)

        results["figure_caption_format"] = self._check_figure_captions(text)
        results["table_caption_format"]  = self._check_table_captions(text)

        # ── Stage 3: IEEE style details ──────────────────────────────────────
        results["citation_format_ieee"]       = self._check_citation_format(text)
        results["no_citations_in_abstract"]   = self._check_no_citations_in_abstract(abstract)
        results["author_block_present"]       = self._check_author_block(text)
        results["keywords_section_present"]   = bool(
            re.search(r'^\s*(?:index terms|keywords?)\s*[:\-—]',
                      text, re.I | re.M))
        results["references_numbered"]        = self._check_references_numbered(text)

        return results

    # ── Extraction helpers ────────────────────────────────────────────────────

    def _extract_title(self, text: str) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return lines[0] if lines else ""

    def _extract_abstract(self, text: str) -> str:
        # Try tight extraction between abstract heading and next section
        m = re.search(
            r'(?:^abstract\s*$\n)(.*?)'
            r'(?=^\s*(?:index terms|keywords?|1\.?\s+introduction|\d+\.?\s+\w+)\s*$)',
            text, re.I | re.M | re.S,
        )
        if m:
            return m.group(1).strip()
        # Fallback: content after "Abstract" heading, up to 400 words
        m2 = re.search(r'abstract\s*[\n\r](.*)', text, re.I | re.S)
        if m2:
            return " ".join(m2.group(1).split()[:400])
        return ""

    def _abstract_word_count(self, text: str) -> int:
        return len(self._extract_abstract(text).split())

    def _extract_sections_ordered(self, text: str) -> list[str]:
        """Return section headings in document order."""
        pattern = re.compile(
            r'^(?:\d+\.?\s+)?([A-Z][A-Za-z\s&]+?)\s*$', re.M
        )
        return [m.group(1).strip() for m in pattern.finditer(text)]

    def _extract_sections_with_content(self, text: str) -> dict[str, str]:
        """Return ordered {heading: content} dict."""
        pattern = re.compile(
            r'^(?:\d+\.?\s+)?([A-Z][A-Za-z\s&]+?)\s*$', re.M
        )
        matches = list(pattern.finditer(text))
        sections: dict[str, str] = {}
        for i, m in enumerate(matches):
            name  = m.group(1).strip().lower()
            start = m.end()
            end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections[name] = text[start:end].strip()
        return sections

    # ── Individual rule checks ────────────────────────────────────────────────

    def _check_section_order(self, section_names: list[str]) -> bool:
        required = self.REQUIRED_SECTION_ORDER
        present  = [s for s in required
                    if any(s in n for n in section_names)]
        if len(present) < 3:
            return False
        positions = []
        for sec in present:
            for i, name in enumerate(section_names):
                if sec in name:
                    positions.append(i)
                    break
        return positions == sorted(positions)

    def _check_figure_captions(self, text: str) -> bool:
        figs = re.findall(r'(?:Fig\.|Figure)\s+\d+', text, re.I)
        if not figs:
            return True
        correct = re.findall(r'Fig\.\s+\d+\.', text)
        return len(correct) >= len(figs) * 0.8

    def _check_table_captions(self, text: str) -> bool:
        tables = re.findall(r'Table\s+\d+', text, re.I)
        if not tables:
            return True
        correct = re.findall(r'Table\s+\d+[.:]', text)
        return len(correct) >= len(tables) * 0.8

    def _check_citation_format(self, text: str) -> bool:
        """IEEE requires [N] or [N1, N2]. Fail if (Author, Year) style found."""
        author_year = re.findall(
            r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s*\d{4}\)', text)
        return len(author_year) == 0

    def _check_no_citations_in_abstract(self, abstract: str) -> bool:
        return not bool(re.search(r'\[\d+\]', abstract))

    def _check_author_block(self, text: str) -> bool:
        lines = text.split("\n")[:40]  # author block typically near top
        for line in lines:
            if re.search(
                r'[A-Z][a-z]+\s+[A-Z][a-z]+.*(?:university|institute|dept|lab)',
                line, re.I,
            ):
                return True
        return False

    def _check_references_numbered(self, text: str) -> bool:
        matches = re.findall(r'^\s*\[\d+\]\s+\w+', text, re.M)
        return len(matches) >= 3
