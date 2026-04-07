# ScholarEnv — Submission Guide
### Meta × PyTorch OpenEnv Hackathon · Round 1 · April 8, 2026 11:59 PM IST

**Authors:** Nensi Pansuriya · Krushna Parmar · Ishita Bhojani

---

## Why This Idea Wins

ScholarEnv is the **only environment in this hackathon where the agent investigates
rather than produces**. Every other submission builds a secretary. Ours builds an auditor.

| Signal | Evidence |
|---|---|
| Task 3 GPT-4o baseline: **0.20–0.45** | Proves genuine RL training value — not a benchmark |
| Veri-R1 (arxiv 2510.01932) validates Task 3 | Online RL for claim verification is current SOTA |
| RLVE (arxiv 2511.07317) validates curriculum | Adaptive difficulty is the right approach |
| 10,000 retractions/year | Real problem, immediate deployment value |
| Zero prior OpenEnv entries in this domain | Novel — judges will remember it |

One-sentence RL justification:
> *"Formatting compliance doesn't need RL — a good prompt gets 0.92. Claim auditing
> requires discovering a traversal strategy that varies by paper structure. RL finds it.
> Prompting doesn't."*

---

## Pre-Submission Checklist

Run through every item before pasting your HF Space URL.

### Step 1 — Local setup

```bash
# Clone / unzip your submission
cd scholar_env_v3

# Install dependencies
pip install -r requirements.txt

# Generate corpus (safe — skips if already present)
python scripts/generate_corpus.py

# Verify corpus
ls data/papers/        # should show paper_001.json, paper_002.json, paper_003.json
```

### Step 2 — Run tests locally

```bash
python tests/test_all.py
```

**Expected output:**
```
[FormattingGrader]
  ✓ Grader returns score in [0,1]
  ...
[ScholarEnvironment]
  ✓ reset() returns observation
  ...
ALL TESTS PASSED (N/N)
```

If any test fails, check the output line — each failure is annotated with the actual value.

### Step 3 — Run server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

In a second terminal, test the endpoints:

```bash
# Health
curl http://localhost:7860/health
# Expected: {"status":"ok","version":"0.4.0","corpus_size":3,...}

# Reset Task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"formatting_compliance"}'
# Expected: {"observation":{...},"info":{...}}

# Reset Task 3 (hardest)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"claim_evidence_audit"}'
# Expected: {"observation":{"available_sections":[...],...},...}

# All 4 tasks
for task in formatting_compliance internal_consistency claim_evidence_audit citation_verification; do
  echo "--- $task ---"
  curl -s -X POST http://localhost:7860/reset \
    -H "Content-Type: application/json" \
    -d "{\"task_id\":\"$task\"}" | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK' if 'observation' in d else 'FAIL:', list(d.keys()))"
done
```

### Step 4 — Docker build

```bash
docker build -t scholar-env .
docker run -p 7860:7860 scholar-env
# Wait for "Application startup complete" then test:
curl http://localhost:7860/health
```

If Docker build fails:
- Check Python version: `python3.11-slim` is required (in Dockerfile)
- Check `requirements.txt` — all packages must be on PyPI

### Step 5 — Deploy to HuggingFace Spaces

```bash
# Login
huggingface-cli login

# Create a new Space (Docker SDK)
# Go to: https://huggingface.co/new-space
# SDK: Docker
# Visibility: Public
# Name: scholar-env

# Push
git init
git add .
git commit -m "ScholarEnv v0.4.0"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/scholar-env
git push -u origin main
```

Wait ~3 minutes for the Space to build, then test:

```bash
curl https://YOUR_USERNAME-scholar-env.hf.space/health
# Expected: 200 {"status":"ok",...}
```

### Step 6 — Set environment variables

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"

# REQUIRED — must be your actual deployed Space URL
export HF_SPACE_URL="https://YOUR_USERNAME-scholar-env.hf.space"
```

**Note:** `HF_SPACE_URL` is required (not optional). The inference script will
raise `RuntimeError` immediately if it is not set.

### Step 7 — Run official pre-submission validator

```bash
chmod +x validate-submission.sh
./validate-submission.sh $HF_SPACE_URL .
```

All 3 checks must pass:
```
PASSED — /reset returns 200
PASSED — Docker build succeeded
PASSED — openenv validate passed
All 3/3 checks passed! Ready to submit.
```

### Step 8 — Run baseline inference

```bash
python inference.py 2>&1 | tee inference_log.txt
```

Expected log format (required for Phase 2 auto-scoring):
```
[START] task=formatting_compliance  env=scholar-env  model=meta-llama/...
[STEP]  step=1  action=submit_formatted_text len=...  reward=0.XXXX  done=False  error=None
[END]   success=True  steps=2  score=0.XXXX  rewards=[0.XXXX, 0.XXXX]
[START] task=internal_consistency  env=scholar-env  model=...
...
```

Verify `baseline_scores.json` is written:
```bash
cat baseline_scores.json
# Expected: {"scores":{"formatting_compliance":...,"internal_consistency":...,...},"average":...}
```

### Step 9 — Final submission

Paste your HuggingFace Space URL into the submission form before **April 8, 11:59 PM IST**.

---

## Common Issues & Fixes

| Issue | Fix |
|---|---|
| `RuntimeError: HF_SPACE_URL must be set` | `export HF_SPACE_URL=https://...` before running inference |
| `openenv validate` command not found | `pip install openenv-core` |
| Docker build fails on `pyyaml` | Check `requirements.txt` has `pyyaml>=6.0.1` |
| `/reset` returns 400 with unknown task | Task 4 requires `"task_id":"citation_verification"` |
| `[STEP]` format looks wrong | Check your log — fields must be double-space separated |
| Corpus not found error | Run `python scripts/generate_corpus.py` first |
| All graders return same score | Run variance check: `python tests/test_all.py` |

---

## Reward Variance Check (Phase 2 DQ risk)

The hackathon auto-evaluator checks that graders return varied scores.
Run this before submitting:

```python
# Quick variance check
from server.environment import ScholarEnvironment
env = ScholarEnvironment()

import random
random.seed(42)
scores = {"formatting_compliance": [], "internal_consistency": [], "claim_evidence_audit": []}

for task_id in scores:
    for _ in range(5):
        r = env.reset(task_id=task_id)
        # Submit random/empty action
        if task_id == "formatting_compliance":
            result = env.step({"task": task_id, "formatted_text": "hello"})
        else:
            result = env.step({"task": task_id, "action_type": "submit_findings",
                                "findings": []})
        scores[task_id].append(result["reward"])

for task_id, sc in scores.items():
    variance = sum((x - sum(sc)/len(sc))**2 for x in sc) / len(sc)
    print(f"{task_id}: scores={[round(s,3) for s in sc]} variance={variance:.4f}")
    assert variance > 0, f"GRADER BROKEN — all scores identical for {task_id}"
print("Variance check PASSED")
```

All 3 tasks must show `variance > 0` (tasks 2/3 will since F-beta varies by paper).

---

## Research References (cite in Phase 3 discussion)

| Paper | Why relevant |
|---|---|
| PRS — arxiv 2512.07478 | Task 1 progressive staging prevents GRPO gradient collapse |
| PBRS — Ng et al. ICML 1999 | Task 3 dense intermediate rewards, policy-invariant |
| AdaRFT — arxiv 2504.05520 | Curriculum targeting [0.4, 0.7] productive zone |
| RLVE — arxiv 2511.07317 | Validates adaptive difficulty approach |
| Veri-R1 — arxiv 2510.01932 | Online RL for claim verification is SOTA in 2025 |
| statcheck — Epskamp 2016 | Prior art; ~50% of papers have statistical errors |
| Retraction Watch | Scale motivation: 10K retractions/year |
| GROBID — Lopez 2008-2025 | Prior art in citation parsing (our CitationVerifier is lightweight alternative) |

---

## Architecture One-Pager (for Phase 3 presentation)

```
ScholarEnv v0.4.0
──────────────────────────────────────────────────────────────
TASK 1: formatting_compliance  (easy, max 3 steps)
  Grader: ProgressiveRewardShaper — 3 stages unlock sequentially
  Signal: dense gradient from stage unlock thresholds (0.60, 0.70)
  RL value: fast convergence baseline; proves env is working

TASK 2: internal_consistency  (medium, max 4 steps)
  Grader: F-beta(β=0.5) — precision-biased, prevents hallucination gaming
  Signal: PBRS navigation bonus + F-beta terminal
  RL value: forces systematic section reading strategy

TASK 3: claim_evidence_audit  (hard, max 6 steps)
  Grader: 0.70×F-beta + 0.20×evidence_specificity + 0.10×coverage_bonus
  Signal: PBRS shaping (Ng 1999) throughout navigation
  RL value: traversal order is learned — prompting baseline 0.20-0.45

TASK 4: citation_verification  (medium, max 8 steps)
  Grader: precision_valid + recall_invalid + evidence_score
  Signal: check_citation returns API lookup data; agent learns query strategy
  RL value: ghost citation detection requires learned trust calibration

CURRICULUM: AdaRFT + UCB1 learning-gradient bandit
  Tracks per-rule success rates; targets [0.40, 0.70] productive zone
  UCB1 uses variance as reward (not mean) — maximises learning gradient

CORPUS: 3 annotated papers (easy/medium/hard) + SQLite citation cache
```
