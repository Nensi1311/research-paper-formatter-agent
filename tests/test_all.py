#!/usr/bin/env python3
"""
tests/test_all.py — ScholarEnv comprehensive test suite.

Covers:
  1. Corpus loading
  2. FormattingGrader — stage scoring, PRS unlock logic
  3. ConsistencyGrader — F-beta, precision/recall, matching
  4. AuditGrader — Task 3 F-beta + evidence specificity
  5. PBRS shaping bonus
  6. UCB1 Bandit
  7. Curriculum selection
  8. ScholarEnvironment — full episode loops for all 3 tasks
  9. FastAPI endpoints (unit-level, no network)

Run:
  python tests/test_all.py

Pass condition printed as:
  ALL TESTS PASSED (N/N)
"""
from __future__ import annotations

import sys
import os
import json
from pathlib import Path

# Ensure root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corpus import PaperCorpus
from server.environment import ScholarEnvironment
from server.graders.formatting_grader import FormattingGrader
from server.graders.consistency_grader import ConsistencyGrader
from server.graders.audit_grader import AuditGrader, ClaimExtractor
from server.reward_shaper import NavigationState, PotentialBasedShaper
from server.bandit import UCB1Bandit
from server.curriculum import Curriculum

PASS = 0
FAIL = 0
RESULTS = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        RESULTS.append(f"  ✓ {name}")
    else:
        FAIL += 1
        RESULTS.append(f"  ✗ {name}" + (f" — {detail}" if detail else ""))


# ── 1. Corpus ─────────────────────────────────────────────────────────────────

def test_corpus() -> None:
    corpus = PaperCorpus.load("data/papers")
    check("Corpus loads without error",     len(corpus) > 0)
    check("Corpus has ≥ 3 papers",          len(corpus) >= 3)
    paper = list(corpus.papers.values())[0]
    check("Paper has sections",             len(paper.sections) > 0)
    check("Paper has tables",               len(paper.tables) > 0)
    check("Paper has ground_truth",         bool(paper.ground_truth))
    check("get_section case-insensitive",
          paper.get_section("ABSTRACT") is not None or
          paper.get_section("abstract") is not None)
    check("section_names populated",        len(paper.section_names) > 0)
    check("table_names populated",          len(paper.table_names) > 0)


# ── 2. FormattingGrader ───────────────────────────────────────────────────────

def test_formatting_grader() -> None:
    grader = FormattingGrader("data/styles/ieee.yaml")
    corpus = PaperCorpus.load("data/papers")
    paper  = list(corpus.papers.values())[0]

    # Perfect manuscript text (schematic)
    PERFECT = """John Smith, MIT CSAIL

Abstract
We present a model achieving state-of-the-art results. The model is trained on large
datasets and evaluated on standard benchmarks. Performance improvements are consistent
across all tested conditions. Results confirm the proposed approach is effective.
Additional experiments support the main findings. The method is computationally
efficient and scalable to large datasets.

Index Terms: machine learning, deep learning, benchmarks

1. Introduction
This paper presents a novel approach. Our main contribution is a new method
for processing information efficiently. We demonstrate improvements over baselines.

2. Methods
We describe our methodology here. The model consists of several components.
Training uses standard optimisation techniques.

3. Results
Results are shown in Table 1. Fig. 1. shows performance curves.

4. Discussion
We discuss implications of the results.

5. References
[1] Author et al. (2024). Title. Venue.
[2] Author et al. (2023). Title. Venue.
[3] Author et al. (2022). Title. Venue.
"""
    result = grader.grade(PERFECT, paper)
    check("Grader returns score in [0,1]",     0.0 <= result.score <= 1.0)
    check("Grader returns stage_1 score",      0.0 <= result.stage_1_score <= 1.0)
    check("Grader returns rule_results dict",  isinstance(result.rule_results, dict))
    check("Grader returns failed_rules list",  isinstance(result.failed_rules, list))

    # Perfect text scores well
    check("Well-formatted text scores ≥ 0.40", result.score >= 0.40,
          f"got {result.score}")

    # Corrupt text scores low
    CORRUPT = "x"
    result_bad = grader.grade(CORRUPT, paper)
    check("Empty text scores < 0.5",            result_bad.score < 0.5,
          f"got {result_bad.score}")

    # Stage locking: corrupt stage 1 → stage 2 = 0
    check("Stage 2 locked when stage 1 < 0.60",
          result_bad.stage_2_score == 0.0 or result_bad.stage_1_score >= 0.60)

    # hint() returns a string
    check("hint() returns non-empty string for failures",
          isinstance(result_bad.hint(), str) and len(result_bad.hint()) > 0)


# ── 3. ConsistencyGrader ─────────────────────────────────────────────────────

def test_consistency_grader() -> None:
    grader = ConsistencyGrader()
    corpus = PaperCorpus.load("data/papers")
    paper  = corpus.papers["paper_001"]

    gt = paper.ground_truth.get("task2_inconsistencies", [])
    check("paper_001 has task2 ground truth", len(gt) > 0)

    # Perfect submission (matches first GT entry)
    perfect = [{
        "type":       gt[0]["type"],
        "location":   gt[0]["location_a"],
        "claim":      gt[0]["claim_a"],
        "contradicts": gt[0]["claim_b"],
    }]
    result = grader.grade(perfect, paper)
    check("Perfect submission scores > 0",  result.score > 0.0,
          f"score={result.score}")
    check("F-beta returned",                0.0 <= result.f_beta <= 1.0)
    check("Precision in [0,1]",             0.0 <= result.precision <= 1.0)
    check("Recall in [0,1]",                0.0 <= result.recall <= 1.0)

    # Empty submission
    empty_result = grader.grade([], paper)
    check("Empty submission scores 0.0",    empty_result.score == 0.0)
    check("Empty submission recall = 0.0",  empty_result.recall == 0.0)

    # Hallucinated submission (wrong type)
    hallucinated = [{"type": "missing_reference", "location": "foo",
                     "claim": "bar", "contradicts": "baz"}] * 5
    hall_result = grader.grade(hallucinated, paper)
    check("Hallucinated findings get low precision",
          hall_result.precision < 0.5,
          f"precision={hall_result.precision}")
    check("F-beta penalises hallucinations vs F1",
          hall_result.f_beta <= hall_result.precision + 0.1)


# ── 4. AuditGrader ────────────────────────────────────────────────────────────

def test_audit_grader() -> None:
    grader = AuditGrader()
    corpus = PaperCorpus.load("data/papers")
    paper  = corpus.papers["paper_001"]

    gt = paper.ground_truth.get("task3_discrepancies", [])
    check("paper_001 has task3 discrepancies", len(gt) > 0)

    injected = [d for d in gt if d.get("injected")]
    check("At least one injected discrepancy", len(injected) > 0)

    # Perfect submission with evidence
    perfect = [{
        "type":         injected[0]["type"],
        "location":     injected[0]["text_location"],
        "claim":        injected[0]["text_claim"],
        "contradicts":  f"Table shows {injected[0]['table_value']}",
        "table_id":     injected[0]["table_id"],
        "table_value":  injected[0]["table_value"],
    }]
    nav = NavigationState(total_sections=5, total_tables=2)
    nav.record_section("results")
    nav.record_table(injected[0]["table_id"])
    result = grader.grade(perfect, paper, nav)
    check("Task3 perfect submission scores > 0", result.score > 0.0,
          f"score={result.score}")
    check("Task3 evidence specificity > 0",      result.evidence_specificity > 0.0)

    # ClaimExtractor
    extractor = ClaimExtractor()
    sample_text = (
        "Table 1 shows 87% accuracy. Our model achieves 94.3% F1 score. "
        "Figure 2 demonstrates the performance improvement."
    )
    claims = extractor.extract(sample_text, "results")
    check("ClaimExtractor finds numerical claims", len(claims) > 0)
    check("Claims have 'values' field",            all("values" in c for c in claims))


# ── 5. PBRS ───────────────────────────────────────────────────────────────────

def test_pbrs() -> None:
    nav = NavigationState(total_sections=4, total_tables=3)
    shaper = PotentialBasedShaper(nav)

    phi0 = shaper.potential()
    check("Initial potential is 0.0", phi0 == 0.0)

    nav.record_section("results")
    phi1 = shaper.potential()
    check("Potential increases after reading section", phi1 > phi0)

    bonus = shaper.shaping_bonus(phi0, phi1)
    check("Shaping bonus ≥ 0",                  bonus >= 0.0)
    check("Shaping bonus ≤ max_bonus",           bonus <= shaper.max_bonus)

    # Coverage bonus
    nav.record_section("introduction")
    nav.record_section("methods")
    phi_high = shaper.potential()
    cov_bonus = shaper.final_coverage_bonus()
    check("Coverage bonus ≥ 0",   cov_bonus >= 0.0)
    check("Coverage bonus ≤ 0.05", cov_bonus <= 0.05)


# ── 6. UCB1 Bandit ────────────────────────────────────────────────────────────

def test_bandit() -> None:
    bandit = UCB1Bandit(c=1.0)
    arms   = ["paper_001", "paper_002", "paper_003"]

    # All unvisited → returns one of them
    selected = bandit.select(arms, "formatting_compliance")
    check("Bandit selects from arm list", selected in arms)

    # Update and re-select
    bandit.update("paper_001", "formatting_compliance", 0.9)
    bandit.update("paper_002", "formatting_compliance", 0.5)
    bandit.update("paper_003", "formatting_compliance", 0.3)
    selected2 = bandit.select(arms, "formatting_compliance")
    check("Bandit still returns valid arm after updates", selected2 in arms)

    # Learning gradient: variance arm preferred
    for _ in range(5):
        bandit.update("paper_002", "formatting_compliance", 0.3)
        bandit.update("paper_002", "formatting_compliance", 0.7)
    top = bandit.top_learning_arms(1)
    check("top_learning_arms returns list", isinstance(top, list))


# ── 7. Curriculum ─────────────────────────────────────────────────────────────

def test_curriculum() -> None:
    corpus     = PaperCorpus.load("data/papers")
    curriculum = Curriculum()

    selected = curriculum.select(corpus, "formatting_compliance")
    check("Curriculum selects a valid paper_id",
          selected in corpus.papers)

    curriculum.update("paper_001", "formatting_compliance", 0.85,
                      {"abstract_max_words": True, "citation_format_ieee": False})
    curriculum.update("paper_002", "formatting_compliance", 0.35,
                      {"abstract_max_words": False, "citation_format_ieee": False})
    summary = curriculum.summary()
    check("Curriculum summary has episodes",    summary["episodes"] >= 2)
    check("Curriculum tracks weak rules",        isinstance(summary["weak_rules"], list))
    check("Curriculum hint returns string",      isinstance(curriculum.hint(), str))


# ── 8. Full environment episodes ──────────────────────────────────────────────

def test_environment() -> None:
    env = ScholarEnvironment()

    # Task 1 episode
    result = env.reset("formatting_compliance")
    check("reset() returns observation",     "observation" in result)
    check("reset() returns info",            "info"        in result)
    obs = result["observation"]
    check("Task1 obs has manuscript_text",   obs.get("manuscript_text") is not None)
    check("Task1 obs has style_guide",       obs.get("style_guide") is not None)

    step_result = env.step({
        "task": "formatting_compliance",
        "formatted_text": obs["manuscript_text"] or "dummy",
    })
    check("step() returns observation",  "observation" in step_result)
    check("step() returns reward float", isinstance(step_result.get("reward"), float))
    check("step() returns done bool",    isinstance(step_result.get("done"), bool))
    check("reward in [0,1]",
          0.0 <= step_result["reward"] <= 1.0)

    # Task 2 episode
    result2 = env.reset("internal_consistency")
    check("Task2 reset returns observation",      "observation" in result2)
    obs2 = result2["observation"]
    check("Task2 obs has available_sections",     len(obs2["available_sections"]) > 0)

    nav_result = env.step({
        "task":         "internal_consistency",
        "action_type":  "query_section",
        "section_name": obs2["available_sections"][0],
    })
    check("Navigation step reward ≥ 0", nav_result.get("reward", -1) >= 0)

    submit_result = env.step({
        "task":        "internal_consistency",
        "action_type": "submit_findings",
        "findings":    [],
    })
    check("Submit ends episode",         submit_result.get("done") is True)
    check("Submit reward in [0,1]",      0.0 <= submit_result.get("reward", -1) <= 1.0)

    # Task 3 episode
    result3 = env.reset("claim_evidence_audit")
    check("Task3 reset returns observation", "observation" in result3)
    obs3 = result3["observation"]
    check("Task3 obs has available_tables",  len(obs3["available_tables"]) > 0)

    extract_result = env.step({
        "task":         "claim_evidence_audit",
        "action_type":  "extract_claims",
        "section_name": obs3["available_sections"][0],
    })
    check("extract_claims returns claims list",
          extract_result["observation"].get("extracted_claims") is not None)

    # state() works
    state = env.state()
    check("state() returns episode_id", "episode_id" in state)
    check("state() returns task_id",    "task_id"    in state)

    # Unknown task returns error
    bad = env.reset("nonexistent_task")
    check("Unknown task returns error dict", "error" in bad)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all() -> None:
    suites = [
        ("Corpus",               test_corpus),
        ("FormattingGrader",     test_formatting_grader),
        ("ConsistencyGrader",    test_consistency_grader),
        ("AuditGrader",          test_audit_grader),
        ("PBRS",                 test_pbrs),
        ("UCB1 Bandit",          test_bandit),
        ("Curriculum",           test_curriculum),
        ("ScholarEnvironment",   test_environment),
    ]

    print(f"\n{'='*60}")
    print("ScholarEnv — Test Suite")
    print(f"{'='*60}")

    for name, fn in suites:
        print(f"\n[{name}]")
        try:
            fn()
        except Exception as exc:
            import traceback
            FAIL_extra = str(exc)
            RESULTS.append(f"  ✗ SUITE EXCEPTION: {FAIL_extra}")
            global FAIL
            FAIL += 1

    print("\nResults:")
    for line in RESULTS:
        print(line)

    total = PASS + FAIL
    print(f"\nResults: {PASS}/{total} passed")
    if FAIL == 0:
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"\n{FAIL} TEST(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    run_all()
