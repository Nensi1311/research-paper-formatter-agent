<div align="center">

<img src="https://img.shields.io/badge/OpenEnv-v0.4.0-blue?style=for-the-badge&logo=pytorch" alt="OpenEnv">
<img src="https://img.shields.io/badge/Python-3.10%2B-brightgreen?style=for-the-badge&logo=python" alt="Python">
<img src="https://img.shields.io/badge/License-Apache_2.0-orange?style=for-the-badge" alt="License">
<img src="https://img.shields.io/badge/Tasks-4-purple?style=for-the-badge" alt="Tasks">
<img src="https://img.shields.io/badge/Tests-63%2F63-success?style=for-the-badge" alt="Tests">

<br><br>

# ScholarEnv

### The first RL environment for AI-assisted peer review and scholarly integrity verification.

*An AI agent that investigates papers — not one that produces them.*

<br>

[![Open in HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-HuggingFace%20Space-blue?style=flat-square)](https://huggingface.co/spaces/nensi1311/research-paper-formatter-agent)
[![API Reference](https://img.shields.io/badge/API-Reference-lightgrey?style=flat-square)](#api-reference)
[![Tests](https://img.shields.io/badge/Tests-63%2F63%20passing-brightgreen?style=flat-square)](#testing)
[![arXiv Papers](https://img.shields.io/badge/arXiv-8%20papers%20cited-red?style=flat-square)](#research-foundation)

<br>

**Nensi Pansuriya · Krushna Parmar · Ishita Bhojani**

*Meta × PyTorch OpenEnv Hackathon · Round 1 · April 2026*

</div>

---

## The Core Insight

Every other RL environment builds an agent that **produces** something. ScholarEnv builds an agent that **investigates** something. The agent is an auditor, not a secretary.

| | Prompted LLM | RL-trained via ScholarEnv |
|---|---|---|
| Task 3 score (zero-shot) | 0.20–0.45 | → 0.55–0.75 after training |
| Consistency | Non-deterministic | Learned, reproducible strategy |
| Cost at scale | ~$0.15/paper | ~$0.001/paper (7B model) |
| False positive rate | Unknown | Measurable per episode |

> **One-sentence RL justification:** Formatting compliance doesn't need RL — a good prompt gets 0.92. Claim auditing requires discovering a document traversal strategy that *varies by paper structure* and **cannot be reduced to a fixed prompt**. RL finds it. Prompting doesn't.

---

## Why This Exists

**~10,000 papers retracted annually.** Every major journal — Nature, Science, IEEE — has a manual integrity screening bottleneck. [StatCheck](https://link.springer.com/article/10.3758/s13428-015-0664-2) found errors in ~50% of psychology papers in top journals. ScholarEnv is the first RL training environment for systematic paper auditing.

---

## Four Tasks

```
Formatting → Consistency → Claim Audit → Citation Check
   Easy          Medium         Hard          Medium
  (rules)      (self-ref)    (cross-ref)    (provenance)
```

### Task 1 — `formatting_compliance` · Easy · Max 3 steps

Agent receives a badly-formatted IEEE manuscript. Must submit a fully compliant version.

- **Grader:** Progressive Reward Shaping — 3 stages unlock sequentially
- **Checks:** Title (≤15 words), abstract (150–250 words, no citations), section order, figure/table captions, IEEE citation format, author block, keywords
- **Frontier baseline:** 0.80–0.95

### Task 2 — `internal_consistency` · Medium · Max 4 steps

Agent navigates section by section and identifies internal contradictions.

- **Grader:** F-beta (β=0.5) — precision-biased, penalises hallucination
- **Error types:** Number mismatches, nonexistent figure references, inconsistent contribution counts
- **Frontier baseline:** 0.40–0.65

### Task 3 — `claim_evidence_audit` · Hard · Max 6 steps

Agent finds places where numerical claims in the text don't match referenced table values. Optimal traversal strategy varies by paper — RL discovers it.

- **Grader:** 0.70 × F-beta + 0.20 × evidence specificity + 0.10 × coverage bonus
- **Intermediate rewards:** Potential-Based Reward Shaping (PBRS) on every navigation step
- **Frontier baseline:** 0.20–0.45 ← genuine RL training headroom

### Task 4 — `citation_verification` · Medium · Max 8 steps

Agent inspects the reference list and identifies ghost citations (fabricated) and misattributed ones. SQLite cache stores verified citations across episodes.

- **Grader:** precision(valid) + recall(ghost/misattributed) + evidence score
- **Frontier baseline:** 0.35–0.60

---

## Reward Design

### Task 1 — Progressive Reward Shaping

Three stages unlock when the previous stage meets its threshold. Prevents GRPO gradient collapse.

```
Stage 1 │ weight 0.40 │ unlock at 0.00 │ Basic structure
Stage 2 │ weight 0.35 │ unlock at 0.60 │ Section compliance
Stage 3 │ weight 0.25 │ unlock at 0.70 │ IEEE style details
```

> arxiv 2512.07478 — PRS for Agentic RL

### Tasks 2 & 3 — F-beta + PBRS

F-beta (β=0.5) makes precision 4× more important than recall:

```
F_β(precision=1.0, recall=0.5) = 0.833   ← correct and precise
F_β(precision=0.2, recall=1.0) = 0.227   ← spamming guesses
```

PBRS (Ng, Harada & Russell, ICML 1999) provides dense intermediate rewards:

```
Φ(s) = 0.30 × sections_read/total
      + 0.30 × tables_checked/total
      + 0.40 × claims_extracted/estimated

F(s,s') = γ·Φ(s') − Φ(s)     ← policy-invariant shaping bonus
```

### Curriculum — AdaRFT + UCB1

Keeps the agent in the productive zone (avg score 0.40–0.70). UCB1 bandit maximises **learning gradient** (reward variance), not mean reward.

```
avg > 0.70  →  select harder papers
avg < 0.40  →  select easier papers
avg in zone →  UCB1 balances exploration-exploitation
```

> arxiv 2504.05520 — AdaRFT Adaptive Data Selection

---

## Quick Start

### Install

```bash
git clone https://github.com/Nensi1311/research-paper-formatter-agent
cd research-paper-formatter-agent
pip install -r requirements.txt
```

### Generate corpus

```bash
python scripts/generate_corpus.py
# Safe to re-run. Use --force to regenerate.
```

### Run tests

```bash
python tests/test_all.py
# Expected: ALL TESTS PASSED (63/63)
```

### Start server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Test all 4 tasks — Linux/macOS

```bash
for task in formatting_compliance internal_consistency claim_evidence_audit citation_verification; do
  echo -n "$task: "
  curl -s -X POST localhost:7860/reset \
    -H "Content-Type: application/json" \
    -d "{\"task_id\":\"$task\"}" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print('OK' if 'observation' in d else 'FAIL')"
done
```

### Test all 4 tasks — Windows PowerShell

```powershell
foreach ($task in @("formatting_compliance","internal_consistency","claim_evidence_audit","citation_verification")) {
    $body = '{"task_id":"' + $task + '"}'
    $r = Invoke-RestMethod -Uri "http://localhost:7860/reset" -Method POST -ContentType "application/json" -Body $body
    Write-Host "$task : OK"
}
```

### Run baseline agent

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token"
export HF_SPACE_URL="https://nensi1311-research-paper-formatter-agent.hf.space"

python inference.py
# Writes: baseline_scores.json
```

### Docker

```bash
docker build -t scholar-env .
docker run -p 7860:7860 scholar-env
curl http://localhost:7860/health
```

---

## API Reference

### `POST /reset`

```json
{ "task_id": "formatting_compliance" }
```

Returns observation with `manuscript_text`, `style_guide`, `step_count`, `max_steps`, `hint`.

---

### `POST /step`

**Task 1:**
```json
{ "task": "formatting_compliance", "formatted_text": "...full manuscript..." }
```

**Tasks 2/3 — navigate:**
```json
{ "task": "claim_evidence_audit", "action_type": "query_section", "section_name": "results" }
{ "task": "claim_evidence_audit", "action_type": "check_table", "table_id": "Table 1" }
{ "task": "claim_evidence_audit", "action_type": "extract_claims", "section_name": "results" }
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

**Task 4 — navigate:**
```json
{ "task": "citation_verification", "action_type": "check_citation", "citation_id": "ref_3" }
```

**Task 4 — submit:**
```json
{
  "task": "citation_verification",
  "action_type": "submit_verdicts",
  "verdicts": [
    { "citation_id": "ref_3", "status": "ghost", "issue": "Implausible claim in title", "confidence": 0.9 }
  ]
}
```

**Response:**
```json
{ "observation": {...}, "reward": 0.7341, "done": false, "info": {"f_beta": 0.73, "precision": 0.8, "recall": 0.67} }
```

---

### Other Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Liveness probe — `{"status":"ok","version":"0.4.0"}` |
| `/state` | `GET` | Episode state, curriculum summary, navigation coverage |
| `/tasks` | `GET` | All 4 task descriptions and config |
| `/action_space` | `GET` | Full action schema with examples |

---

## Project Structure

```
├── inference.py                  ← Baseline agent (required in root)
├── models.py                     ← Pydantic: FormattingAction, ScholarAction,
│                                    CitationAction, ScholarObservation, AnyAction
├── corpus.py                     ← PaperCorpus loader
├── openenv.yaml                  ← 4 tasks, endpoints, authors, baseline_script
├── Dockerfile                    ← HuggingFace Docker deployment
├── requirements.txt
├── validate-submission.sh        ← Official 3-step pre-submission validator
│
├── data/
│   ├── papers/
│   │   ├── paper_001.json        ← NLP benchmark (easy) — 5 refs, 1 ghost
│   │   ├── paper_002.json        ← CV survey (medium)  — 4 refs, 1 ghost
│   │   └── paper_003.json        ← MTL paper (hard)    — 5 refs, 1 ghost
│   └── styles/ieee.yaml          ← IEEE formatting rules
│
├── server/
│   ├── app.py                    ← FastAPI application
│   ├── environment.py            ← 4-task state machine
│   ├── reward_shaper.py          ← PBRS (Ng et al. 1999)
│   ├── curriculum.py             ← AdaRFT + UCB1 hybrid
│   ├── bandit.py                 ← Learning-gradient UCB1
│   ├── citation_verifier.py      ← Citation parser + SQLite cache
│   └── graders/
│       ├── formatting_grader.py  ← PRS 3-stage (Task 1)
│       ├── consistency_grader.py ← F-beta fuzzy-match (Task 2)
│       └── audit_grader.py       ← F-beta + PBRS coverage (Task 3)
│
├── scripts/generate_corpus.py
└── tests/test_all.py             ← 63 assertions
```

---

## Testing

```bash
python tests/test_all.py
```

```
[Corpus]             8/8  ✓
[FormattingGrader]   8/8  ✓  PRS stage locking, hint generation
[ConsistencyGrader]  9/9  ✓  F-beta, fuzzy matching, hallucination penalty
[AuditGrader]        6/6  ✓  Evidence specificity, coverage bonus
[PBRS]               6/6  ✓  Potential monotonicity, bonus bounds
[UCB1 Bandit]        3/3  ✓  Arm selection, learning gradient
[Curriculum]         4/4  ✓  AdaRFT targeting, weak rule tracking
[ScholarEnvironment] 19/19 ✓ Full episode loops, all 4 tasks

Results: 63/63 passed — ALL TESTS PASSED
```

---

## Deployment

```bash
# Push to HuggingFace Spaces
git remote add huggingface https://huggingface.co/spaces/nensi1311/research-paper-formatter-agent
git push huggingface main --force
```

**Live:** https://nensi1311-research-paper-formatter-agent.hf.space

**Pre-submission validation:**
```bash
./validate-submission.sh https://nensi1311-research-paper-formatter-agent.hf.space .
```

---

## Research Foundation

| Paper | Role in ScholarEnv |
|---|---|
| [PRS · arXiv 2512.07478](https://arxiv.org/abs/2512.07478) | Task 1 progressive stage unlock prevents GRPO gradient collapse |
| [PBRS · Ng, Harada & Russell, ICML 1999](http://www.cs.utexas.edu/~ai-lab/pubs/ICML99-shaping.pdf) | Policy-invariant dense intermediate rewards for navigation |
| [AdaRFT · arXiv 2504.05520](https://arxiv.org/abs/2504.05520) | Curriculum targeting [0.40, 0.70] productive learning zone |
| [RLVE · arXiv 2511.07317](https://arxiv.org/abs/2511.07317) | Adaptive difficulty — UCB1 maximises learning gradient |
| [Veri-R1 · arXiv 2510.01932](https://arxiv.org/abs/2510.01932) | Online RL for claim verification; validates Task 3 approach |
| [LaMer · arXiv 2512.16848](https://arxiv.org/abs/2512.16848) | Structured feedback fields improve agent by 11–19% |
| [StatCheck · Epskamp 2016](https://link.springer.com/article/10.3758/s13428-015-0664-2) | Prior art; ~50% of papers have errors — scale motivation |
| [GROBID · Lopez 2008–2025](https://github.com/kermitt2/grobid) | Prior art in citation parsing (CitationVerifier is our RL-native alternative) |

---

## Authors

**Nensi Pansuriya · Krushna Parmar · Ishita Bhojani**

*Meta × PyTorch OpenEnv Hackathon · Round 1 · April 2026*

---

## License

[Apache 2.0](LICENSE)

---

<div align="center">

*The future of AI isn't just models that generate — it's models that verify.*

[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-HuggingFace-blue?style=for-the-badge)](https://huggingface.co/spaces/nensi1311/research-paper-formatter-agent)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Nensi1311/research-paper-formatter-agent)

</div>
