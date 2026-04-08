"""
server/app.py — FastAPI application for ScholarEnv.

Exposes the five endpoints required by OpenEnv / hackathon validation:
  POST /reset        — start a new episode
  POST /step         — submit an action
  GET  /state        — inspect current episode state
  GET  /health       — liveness probe (returns 200)
  GET  /action_space — action schema documentation
  GET  /tasks        — list all available tasks

All request/response bodies are JSON.
CORS is enabled for HuggingFace Spaces embedding.

Usage:
  uvicorn server.app:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

# Ensure root is on path when running from server/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.environment import ScholarEnvironment, TASK_CONFIG

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ScholarEnv",
    description=(
        "OpenEnv environment for scholarly integrity verification. "
        "Three tasks: formatting compliance, internal consistency, "
        "claim-evidence audit."
    ),
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance shared across requests
# (stateful — one active episode at a time, sufficient for hackathon eval)
_ENV: ScholarEnvironment | None = None


def get_env() -> ScholarEnvironment:
    global _ENV
    if _ENV is None:
        data_dir = os.environ.get("DATA_DIR", "data")
        _ENV = ScholarEnvironment(data_dir=data_dir)
    return _ENV


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Landing page — shows environment overview and API reference."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ScholarEnv — OpenEnv Research Integrity</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e0e0e0; padding: 40px 20px; }
  .container { max-width: 860px; margin: 0 auto; }
  h1 { font-size: 2.2rem; color: #fff; margin-bottom: 8px; }
  h1 span { color: #f97316; }
  .subtitle { color: #9ca3af; margin-bottom: 32px; font-size: 1.05rem; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
           font-size: 0.75rem; font-weight: 600; margin-right: 6px; }
  .badge-blue { background: #1d4ed8; color: #fff; }
  .badge-green { background: #166534; color: #86efac; }
  .badge-orange { background: #7c2d12; color: #fed7aa; }
  .badges { margin-bottom: 28px; }
  h2 { font-size: 1.2rem; color: #f97316; margin: 28px 0 12px; 
       border-bottom: 1px solid #1f2937; padding-bottom: 6px; }
  .task-card { background: #1f2937; border-radius: 8px; padding: 16px 20px;
               margin-bottom: 10px; border-left: 4px solid #f97316; }
  .task-card h3 { font-size: 1rem; color: #fff; margin-bottom: 4px; }
  .task-card p { color: #9ca3af; font-size: 0.875rem; }
  .task-meta { display: flex; gap: 16px; margin-top: 6px; font-size: 0.8rem; color: #6b7280; }
  code { background: #111827; padding: 2px 6px; border-radius: 4px;
         font-family: monospace; font-size: 0.875rem; color: #86efac; }
  .api-block { background: #111827; border-radius: 8px; padding: 16px 20px;
               margin-bottom: 10px; font-family: monospace; font-size: 0.82rem; color: #d1d5db; }
  .api-block .method { color: #60a5fa; font-weight: bold; margin-right: 8px; }
  .api-block .path { color: #f97316; }
  .links { display: flex; gap: 12px; margin-bottom: 28px; flex-wrap: wrap; }
  .link-btn { background: #1f2937; border: 1px solid #374151; color: #e0e0e0;
              padding: 8px 16px; border-radius: 6px; text-decoration: none;
              font-size: 0.875rem; transition: background 0.2s; }
  .link-btn:hover { background: #374151; }
  .authors { color: #9ca3af; font-size: 0.875rem; margin-top: 40px;
             border-top: 1px solid #1f2937; padding-top: 16px; }
</style>
</head>
<body>
<div class="container">
  <h1>🔬 <span>Scholar</span>Env</h1>
  <p class="subtitle">The first RL environment for AI-assisted peer review and scholarly integrity verification.</p>
  
  <div class="badges">
    <span class="badge badge-blue">OpenEnv v0.4.0</span>
    <span class="badge badge-green">4 Tasks</span>
    <span class="badge badge-green">Running</span>
    <span class="badge badge-orange">Meta × PyTorch Hackathon</span>
  </div>

  <div class="links">
    <a class="link-btn" href="/docs">📖 Interactive API Docs (Swagger)</a>
    <a class="link-btn" href="/health">❤️ Health Check</a>
    <a class="link-btn" href="/tasks">📋 List Tasks</a>
    <a class="link-btn" href="/state">📊 Current State</a>
  </div>

  <h2>Available Tasks</h2>

  <div class="task-card">
    <h3>formatting_compliance <span class="badge badge-green">EASY</span></h3>
    <p>Fix IEEE manuscript formatting violations — abstract length, section order, citation style, author block.</p>
    <div class="task-meta"><span>Max steps: 3</span><span>Frontier baseline: 0.80–0.95</span></div>
  </div>

  <div class="task-card">
    <h3>internal_consistency <span class="badge badge-blue">MEDIUM</span></h3>
    <p>Find internal contradictions — number mismatches, nonexistent references, inconsistent contribution counts.</p>
    <div class="task-meta"><span>Max steps: 4</span><span>Frontier baseline: 0.40–0.65</span></div>
  </div>

  <div class="task-card" style="border-left-color: #ef4444;">
    <h3>claim_evidence_audit <span class="badge badge-orange">HARD</span></h3>
    <p>Find where text claims don't match table values. RL training value: frontier LLMs score 0.20–0.45 with no training.</p>
    <div class="task-meta"><span>Max steps: 6</span><span>Frontier baseline: <strong style="color:#f97316">0.20–0.45</strong></span></div>
  </div>

  <div class="task-card">
    <h3>citation_verification <span class="badge badge-blue">MEDIUM</span></h3>
    <p>Identify ghost citations (fabricated) and misattributed references. SQLite cache stores verified citations across episodes.</p>
    <div class="task-meta"><span>Max steps: 8</span><span>Frontier baseline: 0.35–0.60</span></div>
  </div>

  <h2>API Usage</h2>

  <div class="api-block">
    <span class="method">POST</span><span class="path">/reset</span>&nbsp;&nbsp;
    {"task_id": "formatting_compliance"}
  </div>
  <div class="api-block">
    <span class="method">POST</span><span class="path">/step</span>&nbsp;&nbsp;&nbsp;&nbsp;
    {"task": "claim_evidence_audit", "action_type": "query_section", "section_name": "results"}
  </div>
  <div class="api-block">
    <span class="method">POST</span><span class="path">/step</span>&nbsp;&nbsp;&nbsp;&nbsp;
    {"task": "claim_evidence_audit", "action_type": "submit_findings", "findings": [...]}
  </div>
  <div class="api-block">
    <span class="method">GET</span>&nbsp;<span class="path">/state</span>&nbsp;&nbsp;&nbsp;
    Returns current episode state and curriculum summary
  </div>

  <p class="authors">
    <strong>Nensi Pansuriya · Krushna Parmar · Ishita Bhojani</strong><br>
    Meta × PyTorch OpenEnv Hackathon · Round 1 · April 2026
  </p>
</div>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/health")
async def health() -> dict:
    """Liveness probe — must return 200 for hackathon validation."""
    env = get_env()
    return {
        "status": "ok",
        "version": "0.4.0",
        "corpus_size": len(env.corpus),
        "tasks": list(TASK_CONFIG.keys()),
    }


# ── Reset ─────────────────────────────────────────────────────────────────────

@app.post("/reset")
async def reset(request: Request) -> JSONResponse:
    """
    Start a new episode.

    Body (JSON):
      { "task_id": "formatting_compliance" }   ← default if omitted

    Returns:
      { "observation": {...}, "info": {...} }
    """
    body    = await request.json() if request.headers.get("content-type") else {}
    task_id = body.get("task_id", "formatting_compliance")
    result  = get_env().reset(task_id=task_id)
    return JSONResponse(content=result, status_code=200)


# ── Step ──────────────────────────────────────────────────────────────────────

@app.post("/step")
async def step(request: Request) -> JSONResponse:
    """
    Submit one action.

    Body (JSON) — Task 1 example:
      {
        "task": "formatting_compliance",
        "formatted_text": "..."
      }

    Body (JSON) — Task 2/3 navigation example:
      {
        "task": "internal_consistency",
        "action_type": "query_section",
        "section_name": "results"
      }

    Body (JSON) — Task 2/3 submit example:
      {
        "task": "claim_evidence_audit",
        "action_type": "submit_findings",
        "findings": [
          {
            "type": "table_text_mismatch",
            "location": "results",
            "claim": "Table 2 shows 87% accuracy",
            "contradicts": "Table 2 value is 79%",
            "table_id": "Table 2",
            "table_value": "79%"
          }
        ]
      }

    Returns:
      { "observation": {...}, "reward": float, "done": bool, "info": {...} }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content={"error": "Request body must be valid JSON."},
            status_code=400,
        )
    result = get_env().step(body)
    # Always return 200 with a valid reward so evaluator never gets 400
    if "error" in result:
        result.setdefault("reward", 1e-4)
        result.setdefault("done", True)
        result.setdefault("info", {})
        result.setdefault("observation", {"task_id": "unknown", "task_description": "",
                                           "paper_id": "none", "step_count": 0, "max_steps": 1})
    return JSONResponse(content=result, status_code=200)


# ── State ─────────────────────────────────────────────────────────────────────

@app.get("/state")
async def state() -> dict:
    """Return current episode state (for debugging and logging)."""
    return get_env().state()


# ── Action space ──────────────────────────────────────────────────────────────

@app.get("/action_space")
async def action_space() -> dict:
    return {
        "type": "structured",
        "discriminator": "task",
        "variants": {
            "formatting_compliance": {
                "fields": {
                    "task": "Literal['formatting_compliance']",
                    "formatted_text": "str — complete reformatted manuscript",
                }
            },
            "internal_consistency": {
                "fields": {
                    "task":         "Literal['internal_consistency']",
                    "action_type":  "query_section | submit_findings",
                    "section_name": "str (for query_section)",
                    "findings":     "list[dict] (for submit_findings)",
                }
            },
            "claim_evidence_audit": {
                "fields": {
                    "task":         "Literal['claim_evidence_audit']",
                    "action_type":  "query_section | check_table | extract_claims | submit_findings",
                    "section_name": "str",
                    "table_id":     "str (e.g. 'Table 1')",
                    "findings":     "list[dict]",
                }
            },
        },
        "finding_schema": {
            "required": ["type", "location", "claim", "contradicts"],
            "optional_for_task3": ["table_id", "table_value"],
            "types": [
                "number_mismatch",
                "missing_reference",
                "contribution_count",
                "table_caption_mismatch",
                "table_text_mismatch",
            ],
        },
    }


# ── Tasks ─────────────────────────────────────────────────────────────────────

@app.get("/tasks")
async def tasks() -> dict:
    return {
        "tasks": [
            {
                "id":          tid,
                "description": cfg["description"][:120] + "...",
                "max_steps":   cfg["max_steps"],
                "navigable":   cfg["allows_navigation"],
            }
            for tid, cfg in TASK_CONFIG.items()
        ]
    }


# ── Entry point (required by openenv spec) ────────────────────────────────────

def main() -> None:
    """Server entry point — called by [project.scripts] and openenv runner."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        workers=1,
    )


if __name__ == "__main__":
    main()
