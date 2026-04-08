<div align="center">

# 🔬 ScholarEnv

### The first RL environment for AI-assisted peer review and scholarly integrity verification

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.4.0-blue?style=flat-square)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-orange?style=flat-square)](LICENSE)
[![Tasks](https://img.shields.io/badge/Tasks-4-purple?style=flat-square)](#four-tasks)
[![Tests](https://img.shields.io/badge/Tests-63%2F63-success?style=flat-square)](#testing)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Live-yellow?style=flat-square)](https://huggingface.co/spaces/nensi1311/research-paper-formatter-agent)

**An AI agent that investigates papers — not one that produces them.**

[Live Demo](https://huggingface.co/spaces/nensi1311/research-paper-formatter-agent) · [API Reference](#api-reference) · [Quick Start](#quick-start) · [Research](#research-foundation)

---

**Nensi Pansuriya · Krushna Parmar · Ishita Bhojani**

*Meta × PyTorch OpenEnv Hackathon · Round 1 · April 2026*

</div>

---

## Why This Exists

**~10,000 papers are retracted every year.** Every major journal — Nature, Science, IEEE, ACM — has a manual integrity screening bottleneck at scale. [StatCheck](https://link.springer.com/article/10.3758/s13428-015-0664-2) found errors in ~50% of psychology papers in top journals.

The key insight: **LLMs are already good at formatting. They fail at auditing.**

Ask GPT-4o to format a manuscript → scores ~0.92 with no training.  
Ask GPT-4o to find all numerical claim mismatches in a paper → scores **0.20–0.45**.

That gap is exactly where RL adds value. The agent must discover a document traversal strategy — which sections to read first, which tables to cross-reference — that **varies by paper structure and cannot be reduced to a fixed prompt**. RL finds this strategy. Prompting cannot.

---

## Four Tasks

```
Formatting → Consistency → Claim Audit → Citation Check
   Easy          Medium         Hard          Medium
```

| Task | What the agent does | Frontier baseline | RL target |
|------|-------------------|-------------------|-----------|
| `formatting_compliance` | Fix IEEE formatting violations in a manuscript | 0.80–0.95 | 0.95+ |
| `internal_consistency` | Find where the paper contradicts itself | 0.40–0.65 | 0.65–0.80 |
| `claim_evidence_audit` | Find where text claims don't match table values | **0.20–0.45** | **0.55–0.75** |
| `citation_verification` | Identify ghost and misattributed references | 0.35–0.60 | 0.65–0.80 |

Task 3's low baseline is the core RL contribution — it proves genuine training headroom exists.

---

## Reward Design

### Task 1 — Progressive Reward Shaping (PRS)
Three stages unlock sequentially. Stage N only contributes when Stage N-1 ≥ threshold. Prevents GRPO gradient collapse.

```
Stage 1 │ weight 0.40 │ threshold 0.00 │ Title, abstract, section headings
Stage 2 │ weight 0.35 │ threshold 0.60 │ Section order, word limits, captions
Stage 3 │ weight 0.25 │ threshold 0.70 │ IEEE citations, author block, keywords
```

### Tasks 2 & 3 — F-beta + Potential-Based Reward Shaping
**F-beta (β=0.5)** weights precision 4× over recall — prevents hallucination gaming:
```
F_β(P=1.0, R=0.5) = 0.833   ← correct and precise ✓
F_β(P=0.2, R=1.0) = 0.227   ← spamming guesses   ✗
```

**PBRS** (Ng et al., ICML 1999) gives dense intermediate rewards per navigation step:
```
Φ(s) = 0.30 × sections_read/total + 0.30 × tables_checked/total + 0.40 × claims_extracted/est
F(s,s') = γ·Φ(s') − Φ(s)     ← policy-invariant, guaranteed by theory
```

### Curriculum — AdaRFT + UCB1
Keeps the agent in the productive zone (avg score 0.40–0.70). UCB1 maximises **learning gradient** (reward variance), not mean reward — a paper always scoring 0.95 teaches nothing.

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
```

### Run tests
```bash
python tests/test_all.py
# → ALL TESTS PASSED (63/63)
```

### Start server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Test endpoints — Linux/macOS
```bash
curl http://localhost:7860/health

for task in formatting_compliance internal_consistency claim_evidence_audit citation_verification; do
  curl -s -X POST localhost:7860/reset \
    -H "Content-Type: application/json" \
    -d "{\"task_id\":\"$task\"}" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print('$task: OK' if 'observation' in d else '$task: FAIL')"
done
```

### Test endpoints — Windows PowerShell
```powershell
Invoke-RestMethod -Uri "http://localhost:7860/health"

foreach ($task in @("formatting_compliance","internal_consistency","claim_evidence_audit","citation_verification")) {
    $body = '{"task_id":"' + $task + '"}'
    $r = Invoke-RestMethod -Uri "http://localhost:7860/reset" -Method POST -ContentType "application/json" -Body $body
    Write-Host "$task : OK"
}
```

### Docker
```bash
docker build -t scholar-env .
docker run -p 7860:7860 scholar-env
curl http://localhost:7860/health
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

---

## API Reference

### `POST /reset`
```json
{"task_id": "formatting_compliance"}
```
Returns `observation` with `manuscript_text`, `style_guide`, `step_count`, `max_steps`, `hint`.

### `POST /step`

**Task 1:**
```json
{"task": "formatting_compliance", "formatted_text": "...full reformatted manuscript..."}
```

**Tasks 2/3 — navigate:**
```json
{"task": "claim_evidence_audit", "action_type": "query_section", "section_name": "results"}
{"task": "claim_evidence_audit", "action_type": "check_table", "table_id": "Table 1"}
{"task": "claim_evidence_audit", "action_type": "extract_claims", "section_name": "results"}
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
{"task": "citation_verification", "action_type": "check_citation", "citation_id": "ref_3"}
```

**Task 4 — submit:**
```json
{
  "task": "citation_verification",
  "action_type": "submit_verdicts",
  "verdicts": [
    {"citation_id": "ref_3", "status": "ghost", "issue": "Implausible title claim", "confidence": 0.9}
  ]
}
```

**Response:**
```json
{"observation": {...}, "reward": 0.7341, "done": false, "info": {"f_beta": 0.73, "precision": 0.8, "recall": 0.67}}
```

### Other endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | `{"status":"ok","version":"0.4.0"}` |
| `/state` | GET | Episode state, curriculum summary, nav coverage |
| `/tasks` | GET | All 4 task descriptions |
| `/action_space` | GET | Full action schema |

---

## Project Structure

```
├── inference.py                 ← Baseline agent (root — required by spec)
├── models.py                    ← FormattingAction, ScholarAction, CitationAction,
│                                   ScholarObservation, AnyAction (discriminated union)
├── corpus.py                    ← PaperCorpus loader
├── openenv.yaml                 ← 4 tasks, endpoints, authors, baseline_script
├── Dockerfile
├── requirements.txt
├── validate-submission.sh       ← Official 3-step pre-submission validator
│
├── data/
│   ├── papers/
│   │   ├── paper_001.json       ← NLP benchmark (easy)   — 5 refs, 1 ghost
│   │   ├── paper_002.json       ← CV survey (medium)     — 4 refs, 1 ghost
│   │   └── paper_003.json       ← MTL paper (hard)       — 5 refs, 1 ghost
│   └── styles/ieee.yaml
│
├── server/
│   ├── app.py                   ← FastAPI: /reset /step /state /health /tasks
│   ├── environment.py           ← 4-task state machine
│   ├── reward_shaper.py         ← PBRS (Ng et al. 1999)
│   ├── curriculum.py            ← AdaRFT + UCB1
│   ├── bandit.py                ← Learning-gradient UCB1
│   ├── citation_verifier.py     ← Citation parser + SQLite cache
│   └── graders/
│       ├── formatting_grader.py ← PRS 3-stage (Task 1)
│       ├── consistency_grader.py← F-beta fuzzy-match (Task 2)
│       └── audit_grader.py      ← F-beta + PBRS coverage (Task 3)
│
├── scripts/generate_corpus.py
└── tests/test_all.py            ← 63 assertions
```

---

## Testing

```
[Corpus]              8/8  ✓
[FormattingGrader]    8/8  ✓  PRS stage locking verified
[ConsistencyGrader]   9/9  ✓  F-beta, hallucination penalty
[AuditGrader]         6/6  ✓  Evidence specificity, coverage bonus
[PBRS]                6/6  ✓  Potential monotonicity, bonus bounds
[UCB1 Bandit]         3/3  ✓  Learning gradient maximisation
[Curriculum]          4/4  ✓  AdaRFT productive-zone targeting
[ScholarEnvironment] 19/19 ✓  Full episode loops, all 4 tasks

Results: 63/63 passed — ALL TESTS PASSED
```

---

## Research Foundation

| Paper | What it justifies |
|---|---|
| [PRS · arXiv 2512.07478](https://arxiv.org/abs/2512.07478) | Task 1 progressive staging prevents GRPO gradient collapse |
| [PBRS · Ng, Harada & Russell, ICML 1999](http://www.cs.utexas.edu/~ai-lab/pubs/ICML99-shaping.pdf) | Policy-invariant dense intermediate rewards |
| [AdaRFT · arXiv 2504.05520](https://arxiv.org/abs/2504.05520) | Curriculum targeting [0.40, 0.70] productive zone |
| [RLVE · arXiv 2511.07317](https://arxiv.org/abs/2511.07317) | Adaptive difficulty — why UCB1 maximises variance |
| [Veri-R1 · arXiv 2510.01932](https://arxiv.org/abs/2510.01932) | Online RL for claim verification is current SOTA |
| [LaMer · arXiv 2512.16848](https://arxiv.org/abs/2512.16848) | Structured feedback fields improve agent 11–19% |
| [StatCheck · Epskamp 2016](https://link.springer.com/article/10.3758/s13428-015-0664-2) | ~50% of papers have errors — scale motivation |
| [GROBID · Lopez 2008–2025](https://github.com/kermitt2/grobid) | Prior art; CitationVerifier is our RL-native alternative |

---

## Baseline Scores

| Task | Score | Notes |
|---|---|---|
| `formatting_compliance` | ~0.82 | Strong baseline, room to perfect |
| `internal_consistency` | ~0.51 | F-beta precision-biased |
| `claim_evidence_audit` | ~0.31 | **Core RL gap — biggest training value** |
| `citation_verification` | ~0.47 | Ghost detection improving with SQLite cache |

---

## License

[Apache 2.0](LICENSE)

---

<div align="center">

*The future of AI isn't just models that generate — it's models that verify.*

[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-HuggingFace-blue?style=for-the-badge)](https://huggingface.co/spaces/nensi1311/research-paper-formatter-agent)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Nensi1311/research-paper-formatter-agent)

</div>
