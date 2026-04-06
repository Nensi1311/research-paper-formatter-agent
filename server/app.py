"""
FastAPI server exposing the PaperFormatterEnv via HTTP.
Endpoints: POST /reset, POST /step, GET /state, GET /health, GET /tasks
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel

# Ensure root directory is in sys.path for relative imports when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import PaperAction, ActionType
from environment import PaperFormatterEnv
from tasks import list_tasks, get_task

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="Research Paper Formatter — OpenEnv",
    description="OpenEnv-compliant environment for formatting academic papers across conference styles.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance per server process
# (For multi-agent use, extend with session IDs)
_env: Optional[PaperFormatterEnv] = None


def get_env() -> PaperFormatterEnv:
    global _env
    if _env is None:
        _env = PaperFormatterEnv(task_id="task_easy")
    return _env


# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"


class StepRequest(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = {}


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with links to docs and environment info."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Research Paper Formatter — OpenEnv</title>
        <style>
            body { font-family: 'Georgia', serif; max-width: 860px; margin: 60px auto; 
                   background: #0f0e17; color: #fffffe; padding: 0 24px; }
            h1 { color: #ff8906; font-size: 2.2rem; margin-bottom: 0.3em; }
            h2 { color: #f25f4c; }
            a { color: #e53170; }
            code { background: #1a1a2e; padding: 2px 6px; border-radius: 4px; }
            .task { background: #1a1a2e; border-left: 4px solid #ff8906; 
                    padding: 12px 16px; margin: 12px 0; border-radius: 0 8px 8px 0; }
            .badge { display:inline-block; padding:2px 8px; border-radius:12px; 
                     font-size:0.8rem; font-weight:bold; }
            .easy { background:#2ecc71; color:#000; }
            .medium { background:#f39c12; color:#000; }
            .hard { background:#e74c3c; color:#fff; }
        </style>
    </head>
    <body>
        <h1>📄 Research Paper Formatter</h1>
        <p>An <strong>OpenEnv</strong> environment where AI agents learn to reformat academic papers 
        between conference styles (IEEE, ACM, NeurIPS, ICML, AAAI, arXiv).</p>

        <h2>Quick Links</h2>
        <ul>
            <li><a href="/docs">Interactive API Docs (Swagger UI)</a></li>
            <li><a href="/health">Health Check</a></li>
            <li><a href="/tasks">List Tasks</a></li>
        </ul>

        <h2>Available Tasks</h2>
        <div class="task">
            <strong>task_easy</strong> <span class="badge easy">EASY</span><br>
            NeurIPS → IEEE: Fix abstract length, column layout, citation style (4 issues)
        </div>
        <div class="task">
            <strong>task_medium</strong> <span class="badge medium">MEDIUM</span><br>
            ACM → NeurIPS: Fix author format, section names, references, layout (6 issues)
        </div>
        <div class="task">
            <strong>task_hard</strong> <span class="badge hard">HARD</span><br>
            IEEE → ICML: Full reformat — title case, sections, authors, citations (7+ issues)
        </div>

        <h2>API Usage</h2>
        <pre><code>POST /reset     {"task_id": "task_easy"}
POST /step      {"action_type": "set_column_layout", "parameters": {"columns": 2}}
GET  /state
GET  /health</code></pre>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "ok", "env": PaperFormatterEnv.ENV_ID, "version": PaperFormatterEnv.VERSION}


@app.get("/tasks")
async def list_all_tasks():
    """List available tasks with metadata."""
    tasks = []
    for tid in list_tasks():
        t = get_task(tid)
        tasks.append({
            "task_id": t.task_id,
            "name": t.name,
            "difficulty": t.difficulty,
            "description": t.description,
            "source_format": t.source_format.value,
            "target_format": t.target_format.value,
            "max_steps": t.max_steps,
            "issues_to_fix": t.issues_to_fix,
            "success_threshold": t.success_threshold,
        })
    return {"tasks": tasks}


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    """Reset environment and return initial observation."""
    global _env
    task_id = request.task_id or "task_easy"
    try:
        _env = PaperFormatterEnv(task_id=task_id)
        obs = _env.reset()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs.model_dump()


@app.post("/step")
async def step(request: StepRequest):
    """Apply an action and return observation, reward, done, info."""
    env = get_env()
    try:
        action_type = ActionType(request.action_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type '{request.action_type}'. "
                   f"Valid: {[a.value for a in ActionType]}"
        )

    action = PaperAction(action_type=action_type, parameters=request.parameters)
    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return result.model_dump()


@app.get("/state")
async def get_state():
    """Return full internal episode state."""
    env = get_env()
    try:
        s = env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return s.model_dump()


@app.get("/action_space")
async def action_space():
    """Describe all available actions and their parameters."""
    return {
        "actions": [
            {
                "action_type": "set_format",
                "description": "Declare the target conference format",
                "parameters": {"format": "string — one of IEEE|ACM|NeurIPS|ICML|AAAI|arXiv|Springer|Elsevier"},
            },
            {
                "action_type": "rename_section",
                "description": "Rename a section by its current name",
                "parameters": {"old_name": "string", "new_name": "string"},
            },
            {
                "action_type": "reorder_sections",
                "description": "Reorder sections by providing a new ordered list",
                "parameters": {"order": "list[string] — ordered section names"},
            },
            {
                "action_type": "format_references",
                "description": "Change reference list citation style",
                "parameters": {"style": "string — IEEE|ACM|APA|APA|AAAI"},
            },
            {
                "action_type": "set_title_case",
                "description": "Set title casing style",
                "parameters": {"style": "string — title_case|sentence_case|upper|lower"},
            },
            {
                "action_type": "set_abstract_word_limit",
                "description": "Trim abstract to specified word count",
                "parameters": {"limit": "int — maximum words"},
            },
            {
                "action_type": "remove_section",
                "description": "Remove a section by name",
                "parameters": {"name": "string"},
            },
            {
                "action_type": "add_section",
                "description": "Add a new section",
                "parameters": {"name": "string", "section_type": "string", "position": "int (optional)"},
            },
            {
                "action_type": "format_author_list",
                "description": "Change author name format",
                "parameters": {"style": "string — 'First Last' or 'F. Last'"},
            },
            {
                "action_type": "set_column_layout",
                "description": "Set 1 or 2 column layout",
                "parameters": {"columns": "int — 1 or 2"},
            },
            {
                "action_type": "format_citations",
                "description": "Switch in-text citation style",
                "parameters": {"style": "string — 'numeric' or 'author_year'"},
            },
            {
                "action_type": "submit",
                "description": "Submit the paper — ends the episode",
                "parameters": {},
            },
        ]
    }


def main():
    """Main entry point for the server script."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
