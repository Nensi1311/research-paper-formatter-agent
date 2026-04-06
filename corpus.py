"""
corpus.py — PaperCorpus: loads and indexes annotated research paper JSONs.

Each paper JSON must conform to the schema in data/papers/paper_001.json.
Papers are lazy-loaded and cached.  Section and table access is O(1) after
first load.

Paper selection strategy (for test / annotation):
  - CC-BY papers from arXiv, 8-12 pages
  - NLP benchmark or CV survey domain (no domain knowledge required)
  - 2022-2024 vintage (stable, citable)
  - ≥ 3 tables (more cross-reference opportunities for Task 3)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Paper:
    """
    Hydrated view of one annotated paper JSON.

    Field layout mirrors the JSON schema from the master guide.
    Additional derived fields (section_names, table_names) are computed
    once at load time to avoid repeated dict key lookups.
    """
    id:                  str
    title:               str
    source:              str
    license:             str
    sections:            dict[str, str]   # name → full text
    tables:              dict[str, dict]  # "Table N" → {caption, data}
    figures:             dict[str, dict]  # "Figure N" → {caption, type}
    ground_truth:        dict             # task{1,2,3}_violations / inconsistencies / discrepancies
    difficulty_score:    float            = 0.5
    badly_formatted_text: str | None      = None

    # Derived — filled by __post_init__
    section_names:     list[str] = field(default_factory=list, init=False)
    table_names:       list[str] = field(default_factory=list, init=False)
    applicable_tasks:  list[str] = field(default_factory=list, init=False)
    violated_rules_task1: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.section_names    = list(self.sections.keys())
        self.table_names      = list(self.tables.keys())
        self.applicable_tasks = ["formatting_compliance",
                                  "internal_consistency",
                                  "claim_evidence_audit"]
        self.violated_rules_task1 = [
            v.get("rule", "") for v in
            self.ground_truth.get("task1_violations", [])
        ]

    # ── Section / table access ─────────────────────────────────────────────

    def get_section(self, name: str) -> str | None:
        """Case-insensitive, partial-match section lookup."""
        # Exact match first
        if name in self.sections:
            return self.sections[name]
        # Case-insensitive
        low = name.lower()
        for k, v in self.sections.items():
            if k.lower() == low:
                return v
        # Partial match
        for k, v in self.sections.items():
            if low in k.lower() or k.lower() in low:
                return v
        return None

    def get_table(self, table_id: str) -> dict | None:
        """Case-insensitive table lookup."""
        if table_id in self.tables:
            return self.tables[table_id]
        low = table_id.lower()
        for k, v in self.tables.items():
            if k.lower() == low:
                return v
        return None

    @property
    def full_text(self) -> str:
        """Concatenate all sections into a single string for regex graders."""
        parts = [self.title, ""]
        for name, content in self.sections.items():
            parts.append(name.upper())
            parts.append(content)
            parts.append("")
        return "\n".join(parts)


class PaperCorpus:
    """Loads all paper JSONs from a directory and provides dict-like access."""

    def __init__(self, papers: dict[str, Paper]):
        self.papers = papers

    @classmethod
    def load(cls, papers_dir: str = "data/papers") -> "PaperCorpus":
        path = Path(papers_dir)
        if not path.exists():
            raise FileNotFoundError(f"Papers directory not found: {path.resolve()}")

        papers: dict[str, Paper] = {}
        for json_file in sorted(path.glob("*.json")):
            try:
                with json_file.open(encoding="utf-8") as f:
                    raw = json.load(f)
                paper = Paper(
                    id=raw["id"],
                    title=raw["title"],
                    source=raw.get("source", ""),
                    license=raw.get("license", "CC-BY 4.0"),
                    sections=raw.get("sections", {}),
                    tables=raw.get("tables", {}),
                    figures=raw.get("figures", {}),
                    ground_truth=raw.get("ground_truth", {}),
                    difficulty_score=raw.get("difficulty_score", 0.5),
                    badly_formatted_text=raw.get("badly_formatted_text"),
                )
                papers[paper.id] = paper
            except (KeyError, json.JSONDecodeError) as exc:
                print(f"[corpus] Warning: skipping {json_file.name}: {exc}")

        if not papers:
            raise RuntimeError(
                f"No valid paper JSONs found in {path.resolve()}. "
                "Run scripts/generate_corpus.py to create the corpus."
            )
        return cls(papers)

    def __len__(self) -> int:
        return len(self.papers)

    def __repr__(self) -> str:
        return f"PaperCorpus({len(self.papers)} papers)"
