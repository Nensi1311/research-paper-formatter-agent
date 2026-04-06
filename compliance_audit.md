# ScholarEnv — OpenEnv Hackathon Compliance Audit

> Audited against the Round 1 Problem Statement (deadline: April 8, 2026 11:59 PM IST)

---

## ✅ PASS / ❌ FAIL Summary

| # | Requirement | Status | Notes |
|---|---|---|---|
| 1 | Real-world task (not games/toys) | ✅ PASS | Research integrity auditing |
| 2 | Typed Pydantic models | ✅ PASS | `models.py` — full |
| 3 | `step()` / `reset()` / `state()` | ✅ PASS | All 3 implemented |
| 4 | `openenv.yaml` with metadata | ✅ PASS | Present and complete |
| 5 | ≥ 3 tasks with graders (easy→hard) | ✅ PASS | 3 tasks, correct progression |
| 6 | Rewards 0.0–1.0 | ✅ PASS | All graders clamp to [0,1] |
| 7 | Meaningful reward (partial progress) | ✅ PASS | PRS + PBRS shaping |
| 8 | Baseline inference script | ⚠️ PARTIAL | See issues below |
| 9 | Uses OpenAI client for LLM calls | ✅ PASS | `openai.OpenAI` used |
| 10 | Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` | ✅ PASS | Line 30–32 |
| 11 | `inference.py` in root directory | ✅ PASS | Present |
| 12 | `[START]` / `[STEP]` / `[END]` log format | ⚠️ PARTIAL | See critical issue below |
| 13 | Dockerfile builds & runs | ✅ PASS | `python:3.11-slim`, port 7860 |
| 14 | Deploys to HuggingFace Spaces | 🔲 UNVERIFIED | No HF Space URL in repo |
| 15 | `openenv validate` passes | 🔲 UNVERIFIED | Needs CLI run |
| 16 | README with all required sections | ✅ PASS | Well documented |
| 17 | `baseline_scores.json` produced | ✅ PASS | Written in `main()` |
| 18 | Runtime < 20 min on 2vCPU/8GB | ✅ LIKELY OK | Few LLM calls, short prompts |

---

## 🚨 Critical Issues (Could Cause Disqualification)

### 1. `[STEP]` log format deviation — Phase 2 scorer will misread

The problem statement's **Sample Inference Script** emits:
```
[STEP]  step=<N>   action=<str>     reward=<float>  done=<bool>  error=<None|str>
```

Your current `log_step()` in `inference.py` (line 51):
```python
print(f"[STEP]  step={step} action={action!r:.80} "
      f"reward={reward:.4f} done={done} error={error}", flush=True)
```

**Problem:** The `!r:.80` format spec applies repr AND width simultaneously, which is non-standard Python and may either error or produce `action='...'` (with quotes) rather than a clean string. The sample script uses `action=<str>` without repr quotes.

**Fix:** Change to `action={action:.80}` (no `!r`), or `action={action!r}` but confirm Phase 2 parser expects repr-quoted strings.

---

### 2. `[START]` format — spacing inconsistency

Your `log_start()`:
```python
print(f"[START] task={task} env={env} model={model}", flush=True)
```

The spec shows:
```
[START] task=<id>  env=scholar-env  model=<MODEL_NAME>
```

The spec has **double spaces** between fields. While minor, automated parsers that rely on exact field separators may fail. **Align spacing to match spec exactly.**

---

### 3. `HF_SPACE_URL` env var not in the mandatory list

The problem statement mandates **only** these 3 variables:
```
API_BASE_URL   MODEL_NAME   HF_TOKEN
```

Your `inference.py` line 33–34 does:
```python
SPACE_URL = os.environ.get("HF_SPACE_URL", "https://your-username-scholar-env.hf.space")
```

This is **fine** (uses `.get()` with a default), but the default placeholder URL `your-username-scholar-env.hf.space` will cause the baseline to fail in evaluation if `HF_SPACE_URL` is not set. The evaluator will run your inference script — if the default URL doesn't resolve, all 3 tasks fail.

**Fix:** Either make `HF_SPACE_URL` required (`os.environ["HF_SPACE_URL"]`) and document it, OR point the default to your actual deployed Space URL after deployment.

---

### 4. `inference.py` calls `HF_TOKEN` but OpenAI client usage wraps raw HTTP calls

Your inference script **does NOT use `from_docker_image` or the OpenEnv Python SDK pattern** shown in the sample. Instead, it calls the env via raw `httpx` HTTP calls. This is **fine** — the spec says use the OpenAI client for LLM calls, not necessarily for env calls. However, confirm the evaluator expects the standard pattern.

---

## ⚠️ Warnings (Non-Critical but Should Fix)

### 5. `openenv.yaml` missing `baseline_script` field

Many OpenEnv YAML schemas include:
```yaml
baseline_script: inference.py
```
Add this for `openenv validate` compatibility (check the installed `openenv-core` schema).

### 6. Docker: corpus generated at build time

```dockerfile
RUN python scripts/generate_corpus.py
```

The corpus is baked into the image. This is fine for reproducibility, **but** if `data/papers/` already exists in the repo (it does — 3 JSON files are present), running `generate_corpus.py` might **overwrite** the hand-crafted ground truth. Verify that `generate_corpus.py` skips existing files.

### 7. `validate-submission.sh` is Linux-only

The script uses bash and Unix tools (`curl`, `openenv`). It won't run on Windows. This is fine for HF/CI evaluation, but note in README that this must be run in a Linux/Mac environment or WSL.

### 8. `requirements.txt` missing `openenv-core`

The server and graders don't import `openenv-core`, so it's not needed at runtime. But `openenv validate` (the validator CLI) needs it installed in the evaluation environment. This is likely handled by the evaluator's own env, but it's worth documenting.

### 9. Single-threaded server (`--workers 1`)

```dockerfile
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

The environment stores state in a global `_ENV` singleton. Using `--workers 1` is **correct** for this design — using multiple workers would break state. This is consistent but limits throughput (fine for hackathon).

---

## ✅ Things Done Well

| Category | Detail |
|---|---|
| **OpenEnv spec** | `reset()`, `step()`, `state()`, `health` all present and correct |
| **Typed models** | Full Pydantic v2 models with discriminated union (`AnyAction`) |
| **3 graders** | `FormattingGrader` (PRS), `ConsistencyGrader` (F-beta), `AuditGrader` (F-beta + coverage) |
| **Partial rewards** | PBRS shaping on navigation steps; PRS staged unlock on Task 1 |
| **Deterministic graders** | All regex/word-count based (Task 1); fuzzy match + F-beta (Tasks 2/3) |
| **Reward range** | Confirmed [0.0, 1.0] across all 3 graders |
| **`openenv.yaml`** | Name, version, description, tasks, action/observation spaces, reward, endpoints, docker, HF config |
| **Dockerfile** | Clean, slim image, health check, port 7860 |
| **Test suite** | 40+ assertions covering all components |
| **README** | Motivation, task table, reward design, API reference, file structure, references |
| **Corpus data** | 3 JSON papers pre-generated and committed |
| **Novelty** | Auditor-not-producer framing; claim-evidence auditing has no prior OpenEnv entry |

---

## 🔧 Recommended Fixes (Priority Order)

### P0 — Do before submission

1. **Fix `log_step` format** — remove `!r` from action repr, match exact spacing from spec sample
2. **Set real HF Space URL** — replace placeholder default in `SPACE_URL`
3. **Deploy to HF Space** — required for automated ping validation

### P1 — Should fix

4. **Verify `generate_corpus.py` doesn't clobber existing JSON** — add a `if not Path(output).exists(): skip` guard
5. **Add `baseline_script: inference.py`** to `openenv.yaml`
6. **Run `openenv validate` locally** — `pip install openenv-core && openenv validate`

### P2 — Nice to have

7. Align `[START]`/`[END]` spacing to match sample exactly
8. Add WSL/Linux note to README for `validate-submission.sh`
9. Add `HF_SPACE_URL` to mandatory env vars in README Quick Start

---

## Pre-Submission Checklist Status

| Check | Status |
|---|---|
| HF Space deploys + returns 200 on reset() | 🔲 Not yet deployed |
| `openenv validate` passes | 🔲 Not run locally |
| `docker build` succeeds | ✅ Likely (clean Dockerfile) |
| Baseline runs without error | ⚠️ Depends on HF_SPACE_URL being set |
| 3+ tasks with graders, scores in [0,1] | ✅ PASS |
