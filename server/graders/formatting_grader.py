"""
server/graders/formatting_grader.py — Task 1: formatting_compliance grader.

Upgraded with HackNU ValidatorAgent 8-category weighting applied to PRS stages.
8 categories → 3 PRS stages → richer gradient signal for GRPO.

Categories (from HackNU DocLing ValidatorAgent):
  page_layout, typography, headings, title_page, abstract,
  citations, references, tables_figures

Mapped to PRS stages:
  Stage 1 (basic structure, weight 0.40):  title_page + abstract + headings
  Stage 2 (section compliance, weight 0.35): page_layout + typography + tables_figures
  Stage 3 (IEEE details, weight 0.25):     citations + references

Authors: Nensi Pansuriya, Krushna Parmar, Ishita Bhojani
"""
from __future__ import annotations

import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...corpus import Paper

# ── HackNU category weights ───────────────────────────────────────────────────
# From DocLing ValidatorAgent.CATEGORY_WEIGHTS
HACKNU_WEIGHTS = {
    "page_layout":    0.15,
    "typography":     0.15,
    "headings":       0.15,
    "title_page":     0.10,
    "abstract":       0.10,
    "citations":      0.15,
    "references":     0.15,
    "tables_figures": 0.05,
}

# PRS stage → HackNU categories mapping
STAGE_CATEGORIES = {
    1: ["title_page", "abstract", "headings"],        # basic structure
    2: ["page_layout", "typography", "tables_figures"], # section compliance
    3: ["citations", "references"],                    # IEEE style
}


@dataclass
class FormattingGradeResult:
    score: float; stage_1_score: float; stage_2_score: float; stage_3_score: float
    rule_results: dict[str, bool]; failed_rules: list[str]; details: dict
    category_scores: dict[str, float] = field(default_factory=dict)

    def hint(self) -> str:
        if not self.failed_rules:
            return "All formatting checks passed."
        rules = ", ".join(self.failed_rules[:5])
        return f"Still failing: {rules}. Fix Stage 1 before Stage 2/3 activate."


class FormattingGrader:
    """
    Progressive Reward Shaping (PRS) grader with HackNU 8-category weights.
    Stage thresholds: Stage1→2: 0.60, Stage2→3: 0.70
    """
    STAGE_CONFIG = {
        1: {"weight": 0.40, "threshold": 0.0,  "rules": [
                "title_exists", "title_word_count", "abstract_exists",
                "abstract_min_words", "abstract_max_words",
                "has_introduction", "has_methods", "has_results",
                "has_discussion", "has_references",
            ]},
        2: {"weight": 0.35, "threshold": 0.60, "rules": [
                "section_order_correct", "intro_word_limit", "methods_word_limit",
                "results_word_limit", "figure_caption_format", "table_caption_format",
            ]},
        3: {"weight": 0.25, "threshold": 0.70, "rules": [
                "citation_format_ieee", "no_citations_in_abstract",
                "author_block_present", "keywords_section_present",
                "references_numbered",
            ]},
    }

    # Rule → HackNU category mapping (for per-category scoring)
    RULE_CATEGORY = {
        "title_exists": "title_page", "title_word_count": "title_page",
        "author_block_present": "title_page",
        "abstract_exists": "abstract", "abstract_min_words": "abstract",
        "abstract_max_words": "abstract", "no_citations_in_abstract": "abstract",
        "has_introduction": "headings", "has_methods": "headings",
        "has_results": "headings", "has_discussion": "headings",
        "has_references": "headings", "keywords_section_present": "headings",
        "section_order_correct": "page_layout",
        "intro_word_limit": "typography", "methods_word_limit": "typography",
        "results_word_limit": "typography",
        "figure_caption_format": "tables_figures", "table_caption_format": "tables_figures",
        "citation_format_ieee": "citations",
        "references_numbered": "references",
    }

    REQUIRED_ORDER = ["abstract", "introduction", "methods", "results",
                      "discussion", "references"]
    WORD_LIMITS    = {"introduction": 800, "methods": 1200, "results": 1000, "discussion": 800}

    # DocLing SECTION_KEYWORDS — comprehensive section name list (from HackNU DocLing AI)
    # Much more robust than our original 5-word check
    SECTION_ALIASES = {
        # Introduction variants
        "introduction": "introduction", "background": "introduction",
        "related work": "introduction", "literature review": "introduction",
        # Methods variants
        "methods": "methods", "methodology": "methods",
        "materials and methods": "methods", "experimental methods": "methods",
        "experimental setup": "methods", "approach": "methods",
        # Results variants
        "results": "results", "findings": "results",
        "results and discussion": "results", "experiments": "results",
        "evaluation": "results", "performance": "results",
        # Discussion variants
        "discussion": "discussion", "conclusion": "discussion",
        "conclusions": "discussion", "summary": "discussion",
        "analysis": "discussion",
        # References variants
        "references": "references", "bibliography": "references",
        "works cited": "references",
    }

    def __init__(self, style_path: str = "data/styles/ieee.yaml") -> None:
        self._style_path = Path(style_path)
        self.style_config: dict = {}
        if self._style_path.exists():
            with self._style_path.open() as f:
                self.style_config = yaml.safe_load(f) or {}

    def grade(self, text: str, paper: "Paper") -> FormattingGradeResult:
        results = self._run_checks(text)
        stages  = self._staged_scores(results)
        total   = round(sum(stages[n] * cfg["weight"]
                            for n, cfg in self.STAGE_CONFIG.items()), 4)

        # Per-HackNU-category scores
        cat_scores: dict[str, list[float]] = {c: [] for c in HACKNU_WEIGHTS}
        for rule, passed in results.items():
            cat = self.RULE_CATEGORY.get(rule)
            if cat:
                cat_scores[cat].append(float(passed))
        category_scores = {
            c: round(sum(v) / len(v), 4) if v else 0.0
            for c, v in cat_scores.items()
        }

        return FormattingGradeResult(
            score=total, stage_1_score=stages[1],
            stage_2_score=stages[2], stage_3_score=stages[3],
            rule_results=results,
            failed_rules=[r for r, v in results.items() if not v],
            details={"title": self._title(text)[:80],
                     "abstract_wc": len(self._abstract(text).split()),
                     "stage_scores": stages,
                     "category_scores": category_scores},
            category_scores=category_scores,
        )

    def _staged_scores(self, results: dict) -> dict:
        scores = {}; prev = 1.0
        for n, cfg in self.STAGE_CONFIG.items():
            if prev >= cfg["threshold"]:
                rel = {k: v for k, v in results.items() if k in cfg["rules"]}
                score = sum(rel.values()) / len(rel) if rel else 0.0
            else:
                score = 0.0
            scores[n] = round(score, 4); prev = score
        return scores

    def _run_checks(self, text: str) -> dict:
        title    = self._title(text)
        abstract = self._abstract(text)
        secs_ord = self._sections_ordered(text)
        secs_cnt = self._sections_content(text)
        sl       = [s.lower() for s in secs_ord]
        r: dict[str, bool] = {}

        # Stage 1
        r["title_exists"]       = bool(title.strip())
        r["title_word_count"]   = len(title.split()) <= 15
        r["abstract_exists"]    = bool(abstract.strip())
        wc = len(abstract.split())
        r["abstract_min_words"] = wc >= 150
        r["abstract_max_words"] = wc <= 250
        r["has_introduction"]   = any("introduction" in s for s in sl)
        r["has_methods"]        = any(s in ("methods","methodology","method") for s in sl)
        r["has_results"]        = any("result" in s for s in sl)
        r["has_discussion"]     = any(s in ("discussion","conclusion","conclusions") for s in sl)
        r["has_references"]     = any("reference" in s for s in sl)

        # Stage 2
        r["section_order_correct"] = self._check_order(sl)
        for sec, lim in self.WORD_LIMITS.items():
            cont = next((v for k, v in secs_cnt.items() if sec in k.lower()), "")
            r[f"{sec}_word_limit"] = (len(cont.split()) == 0 or len(cont.split()) <= lim)
        r["figure_caption_format"] = self._check_fig_caps(text)
        r["table_caption_format"]  = self._check_tbl_caps(text)

        # Stage 3
        r["citation_format_ieee"]     = self._check_citations(text)
        r["no_citations_in_abstract"] = not bool(re.search(r'\[\d+\]', abstract))
        r["author_block_present"]     = self._check_author_block(text)
        r["keywords_section_present"] = bool(re.search(
            r'^\s*(?:index terms|keywords?)\s*[:\-—]', text, re.I | re.M))
        r["references_numbered"]      = len(re.findall(r'^\s*\[\d+\]\s+\w+', text, re.M)) >= 3
        return r

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _title(self, t: str) -> str:
        lines = [l.strip() for l in t.split("\n") if l.strip()]
        return lines[0] if lines else ""

    def _abstract(self, t: str) -> str:
        m = re.search(
            r'(?:^abstract\s*$\n)(.*?)(?=^\s*(?:index terms|keywords?|1\.?\s+introduction|\d+\.?\s+\w+)\s*$)',
            t, re.I | re.M | re.S)
        if m: return m.group(1).strip()
        m2 = re.search(r'abstract\s*[\n\r](.*)', t, re.I | re.S)
        if m2: return " ".join(m2.group(1).split()[:400])
        return ""

    def _sections_ordered(self, t: str) -> list:
        return [m.group(1).strip() for m in
                re.compile(r'^(?:\d+\.?\s+)?([A-Z][A-Za-z\s&]+?)\s*$', re.M).finditer(t)]

    def _sections_content(self, t: str) -> dict:
        patt = re.compile(r'^(?:\d+\.?\s+)?([A-Z][A-Za-z\s&]+?)\s*$', re.M)
        ms   = list(patt.finditer(t)); res = {}
        for i, m in enumerate(ms):
            end = ms[i+1].start() if i+1 < len(ms) else len(t)
            res[m.group(1).strip().lower()] = t[m.end():end].strip()
        return res

    def _check_order(self, sl: list) -> bool:
        """Check section order using DocLing-style alias mapping."""
        # Normalise all detected sections to canonical names
        canonical: list[str] = []
        for name in sl:
            n_low = name.lower().strip()
            mapped = self.SECTION_ALIASES.get(n_low)
            if not mapped:
                # Partial match
                for alias, canon in self.SECTION_ALIASES.items():
                    if alias in n_low or n_low in alias:
                        mapped = canon; break
            if mapped and mapped not in canonical:
                canonical.append(mapped)
        
        present = [s for s in self.REQUIRED_ORDER if s in canonical]
        if len(present) < 3: return False
        positions = [canonical.index(s) for s in present if s in canonical]
        return positions == sorted(positions)

    def _check_fig_caps(self, t: str) -> bool:
        figs = re.findall(r'(?:Fig\.|Figure)\s+\d+', t, re.I)
        if not figs: return True
        return len(re.findall(r'Fig\.\s+\d+\.', t)) >= len(figs) * 0.8

    def _check_tbl_caps(self, t: str) -> bool:
        tbls = re.findall(r'Table\s+\d+', t, re.I)
        if not tbls: return True
        return len(re.findall(r'Table\s+\d+[.:]', t)) >= len(tbls) * 0.8

    def _check_citations(self, t: str) -> bool:
        return len(re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s*\d{4}\)', t)) == 0

    def _check_author_block(self, t: str) -> bool:
        for line in t.split("\n")[:40]:
            if re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+.*(?:university|institute|dept|lab)',
                         line, re.I):
                return True
        return False
