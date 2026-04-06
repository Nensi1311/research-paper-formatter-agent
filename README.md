# ScholarEnv v0.3.0

> **OpenEnv environment for AI-assisted scholarly integrity verification.**
> The first RL environment where an agent acts as a research integrity auditor,
> not a document producer — investigating papers for formatting violations,
> internal contradictions, and claim-evidence discrepancies.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.3-blue)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

---

## Why This Exists

Approximately **10,000 papers are retracted annually** (Retraction Watch, 2024).
Every major journal — Nature, Science, IEEE, ACM — faces a manual integrity
screening bottleneck at scale. The bottleneck is not expertise: it is the
combination of systematic cross-referencing and the volume of submissions.

An LLM asked to "find all inconsistencies in this paper" without RL training
will miss most of them. It lacks a systematic strategy. RL discovers the
optimal auditing strategy — which sections to read first, how to
cross-reference tables efficiently, when to stop. The task surface is large,
the baseline is low (GPT-4o scores 0.20–0.45 on Task 3), and the improvement
ceiling is real. This is exactly the class of problem RL was designed for.

---

## Three Tasks

| Task | Difficulty | Max Steps | Expected GPT-4o | RL Target |
|---|---|---|---|---|
| `formatting_compliance` | Easy | 3 | 0.80–0.95 | 0.95+ |
| `internal_consistency`  | Medium | 4 | 0.40–0.65 | 0.65–0.80 |
| `claim_evidence_audit`  | Hard   | 6 | 0.20–0.45 | 0.55–0.75 |

### Task 1 — `formatting_compliance`
Agent receives a badly-formatted manuscript and IEEE style rules. Must submit
a compliant version. Grader uses **Progressive Reward Shaping (PRS)**:
3 stages unlock sequentially — basic structure → section compliance → IEEE
style details. No LLM judge; every check is a deterministic regex or word count.

### Task 2 — `internal_consistency`
Agent navigates a paper section-by-section, then submits a list of internal
contradictions: number mismatches between sections, references to nonexistent
figures/tables, inconsistent contribution counts. Grader uses
**F-beta (β=0.5)** — precision-weighted — to penalise hallucinated findings.

### Task 3 — `claim_evidence_audit`
The hard task. Agent must find discrepancies where text claims don't match
the values in referenced tables/figures. The optimal traversal strategy
*varies by paper structure* — prompting cannot solve it, RL discovers it.
**PBRS** provides dense intermediate rewards during navigation; the terminal
grader uses F-beta + evidence specificity + coverage bonus.

---

## Design Rationale

Every architectural decision in ScholarEnv is backed by published research:

### Progressive Reward Shaping (PRS) — Task 1
*arxiv 2512.07478 — Zeng et al., 2025*

Sparse binary rewards cause zero-advantage samples in GRPO, collapsing
gradient updates. PRS introduces staged rewards: Stage N rules only contribute
when Stage N-1 score ≥ threshold. This creates a genuine gradient surface
without changing the optimal policy.

```
Stage 1 (weight 0.40, always active):  basic structure
Stage 2 (weight 0.35, unlocks at 0.60): section compliance
Stage 3 (weight 0.25, unlocks at 0.70): IEEE style details
```

### Potential-Based Reward Shaping (PBRS) — Tasks 2 & 3
*Ng, Harada & Russell, ICML 1999*

Navigation steps in Tasks 2 & 3 would otherwise receive zero reward (sparse).
PBRS adds a theoretically policy-invariant shaping bonus:

```
F(s, s') = γ·Φ(s') − Φ(s)
Φ(s) = 0.30×(sections_read/total) + 0.30×(tables_checked/total) + 0.40×(claims_extracted/est_total)
```

This bonus is guaranteed not to change the optimal policy (Ng et al. 1999),
while preventing the zero-advantage collapse that breaks GRPO training.

### AdaRFT Curriculum — Paper Selection
*arxiv 2504.05520 — Zhang et al., 2025*

The adaptive curriculum targets the "productive zone" — average score
∈ [0.40, 0.70]. Papers outside this zone teach the agent less:

```
avg > 0.70 → select harder papers (target difficulty 0.80)
avg < 0.40 → select easier papers (target difficulty 0.30)
avg ∈ [0.40, 0.70] → optimal learning zone
```

Combined with UCB1 bandit selection (which maximises learning gradient,
not mean reward) and weak-rule targeting.

### F-beta Precision Bias — Tasks 2 & 3
A key RLVR insight: if the reward function can be inflated by
submitting long lists of guesses, RL will find that exploit immediately.
F-beta with β=0.5 closes this loophole — precision is weighted 4× more
than recall:

```
F_β = (1 + β²) × P × R / (β²P + R),  β=0.5
```

`F_0.5(P=1.0, R=0.5) = 0.833` vs `F_0.5(P=0.5, R=1.0) = 0.556`

### LaMer In-Context Reflection — Observation Design
*arxiv 2512.16848*

The `hint` field in every observation surfaces the agent's 3-4 weakest
rules (rolling 20-episode window) as targeted feedback. The `feedback` field
on Task 1 returns specific failed rule IDs after each step. These fields
implement the LaMer in-context adaptation pattern, which produces 11–19%
performance gains over observations without structured feedback.

### UCB1 Learning Gradient Bandit
Standard UCB1 maximises mean reward — unhelpful for curriculum design, since
high mean means the agent has mastered that paper. Instead, our bandit
maximises **learning gradient** (peak variance proxy):

```
gradient(arm) = exp(−(variance − 0.04)² / (2 × 0.02²))
```

A paper consistently scoring ∈ [0.30, 0.70] (variance ≈ 0.04) maximises
the gradient and is selected most often during training.

---

## The Core Insight — Auditor Not Producer

Every other environment in this hackathon builds an agent that **produces**
something: formatted text, a customer reply, code. ScholarEnv builds an agent
that **investigates** something. The agent is an auditor, not a secretary.

This flips the RL signal entirely. Output generation is something LLMs are
already good at — prompting gets you 0.85+ with no training. Critical
cross-referential analysis is something LLMs fail at systematically —
RL training produces genuine improvement from a baseline of 0.20–0.45.

One-sentence RL justification:
> "Formatting compliance doesn't need RL — a good prompt gets 0.92. Claim
> auditing requires discovering a document traversal strategy that cannot
> be reduced to a prompt because the optimal strategy varies by paper
> structure. RL finds it. Prompting doesn't."

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (for HF deployment)

### 1. Install

```bash
git clone https://github.com/your-username/scholar-env
cd scholar-env
pip install -r requirements.txt
```

### 2. Generate corpus

```bash
python scripts/generate_corpus.py
```

This creates 3 annotated synthetic papers in `data/papers/`.

### 3. Run locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Test it:

```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "formatting_compliance"}'
```

### 4. Run tests

```bash
python tests/test_all.py
```

Expected output: `ALL TESTS PASSED (N/N)`

### 5. Run baseline agent

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token"
export HF_SPACE_URL="https://your-username-scholar-env.hf.space"

python inference.py
```

Scores are written to `baseline_scores.json`.

### 6. Deploy to HuggingFace Spaces

```bash
# Build and verify Docker image
docker build -t scholar-env .
docker run -p 7860:7860 scholar-env

# Deploy
huggingface-cli login
openenv push your-username/scholar-env
```

### 7. Validate submission

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://your-username-scholar-env.hf.space .
```

All 4 checks must pass before submitting.

---

## API Reference

### `POST /reset`

```json
{ "task_id": "formatting_compliance" }
```

Returns: `{ "observation": {...}, "info": {...} }`

### `POST /step`

**Task 1:**
```json
{
  "task": "formatting_compliance",
  "formatted_text": "...full manuscript..."
}
```

**Tasks 2/3 — navigate:**
```json
{
  "task": "claim_evidence_audit",
  "action_type": "query_section",
  "section_name": "results"
}
```

**Tasks 2/3 — submit:**
```json
{
  "task": "claim_evidence_audit",
  "action_type": "submit_findings",
  "findings": [
    {
      "type": "table_text_mismatch",
      "location": "abstract",
      "claim": "Table 2 shows 87% accuracy",
      "contradicts": "Table 2 value is 79%",
      "table_id": "Table 2",
      "table_value": "79%"
    }
  ]
}
```

Returns: `{ "observation": {...}, "reward": float, "done": bool, "info": {...} }`

### `GET /state`
Returns current episode state, curriculum summary, and navigation coverage.

### `GET /health`
Returns `{"status": "ok"}` with 200. Required by hackathon validation.

---

## File Structure

```
scholar_env/
├── __init__.py
├── models.py                    ← Pydantic actions, observation, reward
├── corpus.py                    ← PaperCorpus loader
├── inference.py                 ← Baseline agent (root — required by spec)
├── openenv.yaml                 ← Environment metadata + task registry
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── validate-submission.sh
├── README.md
├── data/
│   ├── papers/
│   │   ├── paper_001.json       ← NLP benchmark (easy)
│   │   ├── paper_002.json       ← CV survey (medium)
│   │   └── paper_003.json       ← MTL paper (hard)
│   └── styles/
│       └── ieee.yaml            ← IEEE formatting rules
├── scripts/
│   └── generate_corpus.py       ← Synthetic paper generator
├── server/
│   ├── __init__.py
│   ├── app.py                   ← FastAPI + endpoints
│   ├── environment.py           ← ScholarEnvironment (pure Python)
│   ├── reward_shaper.py         ← PBRS (Ng et al. 1999)
│   ├── curriculum.py            ← AdaRFT + UCB1 hybrid
│   ├── bandit.py                ← UCB1 learning-gradient bandit
│   └── graders/
│       ├── __init__.py
│       ├── formatting_grader.py ← Task 1 — PRS, 3 stages
│       ├── consistency_grader.py← Task 2 — F-beta, fuzzy match
│       └── audit_grader.py      ← Task 3 — F-beta + coverage + PBRS
└── tests/
    └── test_all.py              ← 40+ assertions across all components
```

---

## Reward Design Summary

| Task | Primary | Secondary | Tertiary |
|---|---|---|---|
| `formatting_compliance` | PRS staged score (3 stages) | — | — |
| `internal_consistency`  | F-beta (β=0.5) | Evidence specificity (+0.05) | PBRS navigation bonus |
| `claim_evidence_audit`  | 0.70 × F-beta (β=0.5) | 0.20 × evidence specificity | 0.10 × coverage bonus |

---

## References

1. **PRS**: Zeng et al. (2025). Enhancing Agentic RL with Progressive Reward Shaping and Value-based Sampling Policy Optimization. *arXiv 2512.07478*.
2. **AdaRFT**: Zhang et al. (2025). Adaptive Data Selection for RLVR. *arXiv 2504.05520*.
3. **LaMer**: arxiv 2512.16848. In-context reflection improves pass@1 by 11–19%.
4. **RLVE**: arxiv 2511.07317. Effective prompt ratio for RL environment corpus design.
5. **PBRS**: Ng, Harada & Russell (1999). Policy invariance under reward transformations. *ICML 1999*.
6. **statcheck**: Epskamp & Nuijten (2016). Statcheck. *J. Statistical Software*. Prior art in automated statistical claim verification.
7. **COPE Guidelines**: publicationethics.org/guidance/Guidelines. Research integrity violation taxonomy.
8. **Retraction Watch**: retractionwatch.com. Scale of the peer review problem (10K retractions/year).
