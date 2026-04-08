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
from fastapi.responses import JSONResponse

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
    status  = 400 if "error" in result else 200
    return JSONResponse(content=result, status_code=status)


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
    status = 400 if "error" in result else 200
    return JSONResponse(content=result, status_code=status)


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
