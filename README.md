---
title: Research Paper Formatter — OpenEnv
emoji: 📄
colorFrom: orange
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - paper-formatting
  - rl-environment
  - academic-publishing
  - nlp
license: apache-2.0
---

# 📄 Research Paper Formatter — OpenEnv

> An OpenEnv-compliant RL environment where AI agents learn to reformat academic research papers between conference and journal submission styles.

## Motivation

Every researcher who has submitted papers across multiple venues knows the pain: a paper formatted for NeurIPS needs 15+ changes before it can be submitted to IEEE. The column count changes, the reference style changes, the abstract length limit changes, the author name format changes, the citation style changes... and it's all manual.

This environment teaches agents to:

1. **Identify** formatting violations across 6+ dimensions
2. **Plan** an efficient sequence of formatting actions
3. **Execute** targeted fixes with the right parameters
4. **Submit** only when the paper is fully compliant

It fills a genuine gap — formatting-aware document agents are underexplored in the RL community, despite being a high-value real-world task.

---

## Environment Description

The environment simulates reformatting a real academic paper from one conference style to another. The paper is represented as a structured object (sections, references, authors, metadata) and the agent applies formatting actions one at a time.

**Supported conference formats:** IEEE, ACM, NeurIPS, ICML, AAAI, arXiv

Each conference has a `FormatSpec` defining:

- Required and forbidden sections
- Section ordering
- Abstract word limit
- Reference citation style (IEEE, ACM, APA, AAAI)
- Author name format ("F. Last" vs "First Last")
- Column layout (1 or 2)
- In-text citation style (numeric vs author-year)

---

## Observation Space

The observation is a `PaperObservation` JSON object:

| Field                 | Type             | Description                       |
| --------------------- | ---------------- | --------------------------------- |
| `paper_id`            | string           | Unique paper identifier           |
| `paper_title`         | string           | Paper title                       |
| `current_format`      | enum             | Source conference format          |
| `target_format`       | enum             | Target conference format          |
| `target_spec`         | FormatSpec       | Full spec for target conference   |
| `sections`            | list[Section]    | Paper sections with metadata      |
| `section_order`       | list[string]     | Current section ordering          |
| `references`          | list[Reference]  | Reference list with style info    |
| `authors`             | list[AuthorInfo] | Author names and affiliations     |
| `abstract_word_count` | int              | Current abstract length           |
| `column_layout`       | int              | Current column count (1 or 2)     |
| `title_case_style`    | string           | Current title casing              |
| `citation_style`      | string           | `numeric` or `author_year`        |
| `compliance_score`    | float            | Running 0.0–1.0 compliance score  |
| `issues`              | list[string]     | All current formatting violations |
| `fixed_issues`        | list[string]     | Issues resolved so far            |
| `steps_taken`         | int              | Steps used so far                 |
| `max_steps`           | int              | Maximum allowed steps             |
| `done`                | bool             | Episode completion flag           |

---

## Action Space

Actions are JSON objects with `action_type` and `parameters`:

| Action                    | Parameters                | Description                 |
| ------------------------- | ------------------------- | --------------------------- |
| `set_format`              | `format: str`             | Declare target format       |
| `rename_section`          | `old_name, new_name: str` | Rename a section            |
| `reorder_sections`        | `order: list[str]`        | Reorder sections            |
| `format_references`       | `style: str`              | Change reference style      |
| `set_title_case`          | `style: str`              | Change title casing         |
| `set_abstract_word_limit` | `limit: int`              | Trim abstract               |
| `remove_section`          | `name: str`               | Remove a section            |
| `add_section`             | `name, section_type: str` | Add a section               |
| `format_author_list`      | `style: str`              | Change author format        |
| `set_column_layout`       | `columns: int`            | Set 1 or 2 columns          |
| `format_citations`        | `style: str`              | Switch citation style       |
| `submit`                  | `{}`                      | Submit paper (ends episode) |

---

## Tasks

### Task 1: NeurIPS → IEEE (Easy)

**Max steps:** 8 | **Success threshold:** 0.85

Convert a transformer paper from NeurIPS to IEEE format. 4 clear issues:

- Abstract exceeds 150-word limit (current: 220 words)
- Column layout is 1 (IEEE requires 2)
- Citation style is `author_year` (IEEE requires `numeric`)
- References use APA style (IEEE requires IEEE style)

### Task 2: ACM → NeurIPS (Medium)

**Max steps:** 12 | **Success threshold:** 0.80

Convert an LLM systems paper from ACM to NeurIPS. 6 issues requiring careful sequencing:

- Author names use abbreviated format (`D. Zhang` → full first name)
- Section names have numeric prefixes (`1. Introduction`)
- Duplicate Related Work sections
- Citation style is `numeric` (NeurIPS requires `author_year`)
- References use ACM style (NeurIPS requires APA)
- Column layout is 2 (NeurIPS requires 1)

### Task 3: IEEE → ICML (Hard)

**Max steps:** 15 | **Success threshold:** 0.75

Full reformat of a NAS paper from IEEE to ICML. 7+ issues including traps:

- Title is ALL CAPS (must switch to title case)
- Section names use IEEE roman numeral convention (`I. Introduction`)
- Author names abbreviated (`H. Liu` → expanded)
- Citation style wrong (`numeric` → `author_year`)
- Reference style wrong (`IEEE` → `APA`)
- Section `V. Discussion` needs renaming and reordering
- Multiple sections need clean renaming without double-fixing

---

## Reward Function

The reward is a **weighted composite** of 6 dimension scores, all in [0.0, 1.0]:

| Dimension           | Weight | Description                                            |
| ------------------- | ------ | ------------------------------------------------------ |
| Section structure   | 30%    | Required sections present, no forbidden, correct order |
| Reference format    | 20%    | All references use target citation style               |
| Abstract compliance | 15%    | Abstract within word limit                             |
| Author format       | 15%    | Names follow target format pattern                     |
| Layout              | 10%    | Correct column count                                   |
| Citation style      | 10%    | In-text citations use target style                     |

**Step penalty:** −0.005 per step (max −0.10) — rewards efficient agents.

Reward is provided **every step**, not just on submission, giving dense learning signal.

---

## API Endpoints

```
POST /reset          {"task_id": "task_easy"}         → PaperObservation
POST /step           {"action_type": ..., "parameters": {...}} → StepResult
GET  /state          → EpisodeState (full internal state)
GET  /health         → {"status": "ok"}
GET  /tasks          → list of task metadata
GET  /action_space   → action schema documentation
GET  /docs           → Swagger UI
```

---

## Setup & Usage

### Local

```bash
git clone https://huggingface.co/spaces/your-username/paper-formatter-openenv
pip install -r requirements.txt
python server.py
```

### Docker

```bash
docker build -t paper-formatter-env .
docker run -p 7860:7860 paper-formatter-env
```

### Inference / Baseline

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` at temperature 0.2:

| Task                      | Score     | Success | Steps |
| ------------------------- | --------- | ------- | ----- |
| task_easy (NeurIPS→IEEE)  | ~0.88     | ✓       | 4–5   |
| task_medium (ACM→NeurIPS) | ~0.72     | ✓       | 8–9   |
| task_hard (IEEE→ICML)     | ~0.61     | ✗       | 12–13 |
| **Average**               | **~0.74** |         |       |

These scores represent a strong baseline — the hard task is genuinely challenging for frontier models.

---

## Project Structure

```
paper-formatter-openenv/
├── server.py           # FastAPI HTTP server (OpenEnv endpoints)
├── environment.py      # Core env: reset(), step(), state()
├── models.py           # Pydantic typed models (Observation, Action, Reward, State)
├── grader.py           # Deterministic multi-dimension grader
├── tasks.py            # Task definitions (easy/medium/hard)
├── paper_data.py       # Synthetic paper dataset
├── conference_specs.py # Conference format specifications
├── inference.py        # Baseline LLM agent script
├── openenv.yaml        # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Citation

```bibtex
@misc{paper-formatter-openenv,
  title={Research Paper Formatter: An OpenEnv Environment for Academic Document Reformatting},
  year={2025},
  publisher={HuggingFace Spaces},
  url={https://huggingface.co/spaces/your-username/paper-formatter-openenv}
}
```
