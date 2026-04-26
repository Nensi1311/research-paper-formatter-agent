"""
hf_space/app.py — ScholarEnv HuggingFace Space
Serves the ScholarEnv UI dashboard (Metronic HTML) + live demo API.

Routes:
  GET  /          → serve the dashboard UI (index.html)
  GET  /static/*  → static assets (CSS, JS, media)
  GET  /health    → {"status": "ok"}
  POST /audit     → run inference demo (claim-evidence audit)
  GET  /results   → return training results JSON
  GET  /tasks     → return task list JSON

Authors: Nensi Pansuriya, Krushna Parmar, Ishita Bhojani
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ── Config ────────────────────────────────────────────────────────────────────
HF_LORA_REPO = os.environ.get("HF_LORA_REPO", "")
BASE_DIR     = Path(__file__).parent

app = FastAPI(
    title="ScholarEnv",
    description="Research Paper Integrity Auditor — OpenEnv Hackathon 2026",
    version="6.7",
)

# Mount static assets from dist
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Training results (hardcoded from real Colab run) ─────────────────────────
TRAINING_RESULTS = {
    "smoke_run": {
        "description": "25 GRPO steps, T3 Claim Audit only, 416 completions",
        "start_reward": 0.0076,
        "peak_reward":  0.9051,
        "end_reward":   0.5114,
        "improvement_x": 67,
        "valid_json_pct": 95.4,
        "has_table_id_pct": 94.7,
        "duration_min": 25.7,
    },
    "multi_task": {
        "description": "200 GRPO steps, 5 tasks, 383 graded completions",
        "baseline_reward": 0.3436,
        "tasks": {
            "formatting_compliance":  {"baseline": 0.1709, "final": 0.2787, "n": 91,  "change": "+63%"},
            "internal_consistency":   {"baseline": 0.0187, "final": 0.0176, "n": 67,  "change": "-6%"},
            "claim_evidence_audit":   {"baseline": 0.8245, "final": 0.4932, "n": 91,  "change": "-40%"},
            "citation_verification":  {"baseline": 0.3604, "final": 0.4807, "n": 115, "change": "+33%"},
            "prompt_injection_audit": {"baseline": 0.1397, "final": 0.1771, "n": 19,  "change": "+27%"},
        },
        "notes": [
            "T3 regression caused by T2 system-prompt field-name bug (fixed in v8)",
            "T5 improved +27% with zero T5 training examples — zero-shot transfer confirmed",
            "T4 citation verification showed strongest clean improvement (+33%)",
        ],
    },
    "training_config": {
        "model":   "Qwen2.5-1.5B-Instruct (4-bit, Unsloth)",
        "lora_r":  16,
        "steps":   200,
        "loss":    "DAPO + scale_rewards=batch",
        "hardware": "Colab T4 (free tier, 14 GB VRAM)",
        "dataset": "200 rows (4 tasks × 50 papers, 5 domains)",
    },
}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard UI."""
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>ScholarEnv</h1><p>UI not found. "
                        "Check index.html is in the hf_space/ directory.</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "name": "scholar-env", "version": "6.7"}


@app.get("/results")
async def results():
    return JSONResponse(content=TRAINING_RESULTS)


@app.get("/tasks")
async def tasks():
    return JSONResponse(content={
        "tasks": [
            {"id": "formatting_compliance",  "max_steps": 3, "allows_navigation": False},
            {"id": "internal_consistency",   "max_steps": 4, "allows_navigation": True},
            {"id": "claim_evidence_audit",   "max_steps": 6, "allows_navigation": True},
            {"id": "citation_verification",  "max_steps": 8, "allows_navigation": True},
            {"id": "prompt_injection_audit", "max_steps": 5, "allows_navigation": True},
        ]
    })


@app.post("/audit")
async def audit(request: Request):
    """
    Live demo endpoint: run claim-evidence audit on user-supplied text.
    Falls back to rule-based scoring if model not loaded.
    """
    body = await request.json()
    abstract  = body.get("abstract", "").strip()
    table1    = body.get("table1", "").strip()

    if not abstract:
        return JSONResponse({"error": "abstract is required"}, status_code=400)

    # Extract numbers from abstract and table
    abs_nums = set(re.findall(r"\b(\d+(?:\.\d+)?)\b", abstract))
    tbl_nums = set(re.findall(r"\b(\d+(?:\.\d+)?)\b", table1))

    # Find numbers that appear in abstract but not in table (potential discrepancies)
    only_in_abstract = abs_nums - tbl_nums
    big_only = {n for n in only_in_abstract if float(n) > 50}  # focus on metric values

    findings = []
    for num in sorted(big_only, key=float, reverse=True)[:3]:
        # Find context around the number in abstract
        ctx_match = re.search(
            r'([^.!?]*\b' + re.escape(num) + r'\b[^.!?]*[.!?]?)',
            abstract
        )
        if ctx_match:
            findings.append({
                "type": "table_text_mismatch",
                "location": "abstract",
                "claim": ctx_match.group(1).strip()[:150],
                "table_id": "Table 1",
                "table_value": f"not found ({num} absent from table)",
                "confidence": 0.75,
            })

    return JSONResponse({
        "findings": findings,
        "n_findings": len(findings),
        "method": "rule-based (model not loaded)",
        "note": (
            "This is a rule-based demo. "
            "The full trained model (Qwen2.5-1.5B + GRPO LoRA) runs in the Colab notebook."
        ),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
