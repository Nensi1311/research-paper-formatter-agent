"""
server/app.py — ScholarEnv FastAPI server.

OpenEnv compliance (spec_version: 1):
  - Uses create_app() from openenv.core.env_server.http_server
  - ScholarEnvironment.SUPPORTS_CONCURRENT_SESSIONS = True
  - max_concurrent_envs=4 (configurable via MAX_ENVS env var)

Standard OpenEnv endpoints (from create_app):
  POST /reset  GET /state  GET /health  GET /schema  WS /ws

Custom endpoints added:
  GET  /             — demo HTML
  GET  /dashboard    — reward curve
  POST /reset_with_paper — training shortcut

Authors: Nensi Pansuriya · Krushna Parmar · Ishita Bhojani
"""
from __future__ import annotations
import json, os, sys, time
from collections import OrderedDict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from openenv.core.env_server.http_server import create_app
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False
    from fastapi import FastAPI
    def create_app(env_cls, action_cls, obs_cls, **kwargs):
        return FastAPI(title="ScholarEnv (standalone)", version="2.0.0")

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.environment import ScholarEnvironment, TASK_CONFIG
try:
    from models import ScholarAction, ScholarObservation
except ImportError:
    from ..models import ScholarAction, ScholarObservation

_MAX_ENVS = int(os.environ.get("MAX_ENVS", "4"))
_DATA_DIR  = os.environ.get("DATA_DIR", "data")

app = create_app(
    ScholarEnvironment,
    ScholarAction,
    ScholarObservation,
    env_name="scholar-env",
    max_concurrent_envs=_MAX_ENVS,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount the rich Metronic UI static assets (CSS / JS / media) from hf_space/static.
_HF_SPACE_DIR = _ROOT / "hf_space"
_HF_STATIC_DIR = _HF_SPACE_DIR / "static"
if _HF_STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_HF_STATIC_DIR)), name="static")

_REWARD_HISTORY: list[dict] = []
MAX_HISTORY = 500
_SESSIONS: OrderedDict[str, ScholarEnvironment] = OrderedDict()
MAX_SESSIONS = 128

def _get_or_create_session(session_id: str) -> ScholarEnvironment:
    if session_id not in _SESSIONS:
        if len(_SESSIONS) >= MAX_SESSIONS:
            _SESSIONS.popitem(last=False)
        _SESSIONS[session_id] = ScholarEnvironment(data_dir=_DATA_DIR)
    else:
        _SESSIONS.move_to_end(session_id)
    return _SESSIONS[session_id]

@app.get("/health")
async def health():
    return {"status":"ok","version":"2.0.0","active_sessions":len(_SESSIONS),
            "max_sessions":MAX_SESSIONS,"tasks":list(TASK_CONFIG.keys()),
            "openenv_core":_OPENENV_AVAILABLE}

@app.post("/reset_with_paper")
async def reset_with_paper(request: Request) -> JSONResponse:
    try: body = await request.json()
    except: body = {}
    task_id    = body.get("task_id", "claim_evidence_audit")
    session_id = body.get("session_id", f"s_{time.time_ns()}")
    paper_dict = body.get("paper", {})
    if task_id not in TASK_CONFIG:
        return JSONResponse({"error": f"Unknown task_id '{task_id}'"}, status_code=200)
    env = _get_or_create_session(session_id)
    if paper_dict:
        try:
            from corpus import Paper
            paper = Paper(
                id=paper_dict.get("id","injected"), title=paper_dict.get("title",""),
                source=paper_dict.get("source","training"), license=paper_dict.get("license","synthetic"),
                sections=paper_dict.get("sections",{}), tables=paper_dict.get("tables",{}),
                figures=paper_dict.get("figures",{}), ground_truth=paper_dict.get("ground_truth",{}),
                difficulty_score=paper_dict.get("difficulty_score",0.5),
                badly_formatted_text=paper_dict.get("badly_formatted_text"),
            )
            env.corpus.papers[paper.id] = paper
            env._use_procedural = False
            env._injected_paper_id = paper.id
        except Exception: pass
    result = env.reset(task_id=task_id, session_id=session_id)
    result["session_id"] = session_id
    return JSONResponse(content=result, status_code=200)

@app.post("/step")
async def step_action(request: Request) -> JSONResponse:
    try: body = await request.json()
    except: body = {}
    task_id    = body.get("task", body.get("task_id","claim_evidence_audit"))
    session_id = body.get("session_id","default")
    env = _get_or_create_session(session_id)
    result = env.step(body)
    reward = result.get("reward")
    if reward is not None:
        _REWARD_HISTORY.append({"t":time.time(),"task":task_id,"reward":float(reward),"session":session_id})
        if len(_REWARD_HISTORY) > MAX_HISTORY: _REWARD_HISTORY.pop(0)
    # G4: Save episode transcript on terminal step
    if result.get("done") and result.get("info", {}).get("action_log"):
        _save_transcript(session_id, task_id,
                         result["info"]["action_log"], float(reward or 0))
    return JSONResponse(content=result, status_code=200)

@app.post("/reset")
async def reset_env(request: Request) -> JSONResponse:
    try: body = await request.json()
    except: body = {}
    task_id    = body.get("task_id","claim_evidence_audit")
    session_id = body.get("session_id", f"s_{time.time_ns()}")
    if task_id not in TASK_CONFIG:
        return JSONResponse({"error":f"Unknown task_id '{task_id}'"}, status_code=200)
    env = _get_or_create_session(session_id)
    result = env.reset(task_id=task_id, session_id=session_id)
    result["session_id"] = session_id
    return JSONResponse(content=result, status_code=200)

@app.get("/state")
async def state(session_id: str = "default"):
    if session_id not in _SESSIONS: return {"status":"idle","session_id":session_id}
    env = _SESSIONS[session_id]
    s = env.state() if hasattr(env,"state") else {}
    s["n_sessions"] = len(_SESSIONS)
    return s

@app.get("/tasks")
async def tasks():
    return {tid:{"description":cfg["description"],"max_steps":cfg["max_steps"]}
            for tid,cfg in TASK_CONFIG.items()}

@app.get("/", response_class=HTMLResponse)
async def demo_ui():
    """Serve the full Metronic dashboard UI from hf_space/index.html."""
    index_html = _HF_SPACE_DIR / "index.html"
    if index_html.exists():
        return HTMLResponse(content=index_html.read_text(encoding="utf-8"))
    demo = Path(_ROOT)/"demo.html"
    if demo.exists(): return HTMLResponse(content=demo.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>ScholarEnv</h1><p><a href='/docs'>API Docs</a> | <a href='/dashboard'>Dashboard</a></p>")

_ASSET_DIR = Path(_ROOT) / "assets"


@app.get("/assets/{name:path}")
async def serve_asset(name: str):
    """Phase D6 (v6): static asset route for the Saccade-RL GIF + reward curve PNG."""
    from fastapi.responses import FileResponse, PlainTextResponse
    safe = (_ASSET_DIR / name).resolve()
    if not str(safe).startswith(str(_ASSET_DIR.resolve())) or not safe.exists():
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(str(safe))


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """v6 dashboard: live reward stream + (when present) Saccade-RL GIF + curve PNG.

    Phase D6 stub: the static assets/saccade_comparison.gif and
    assets/reward_curve.png are produced by the notebook (Cells 10/12) and
    rsync'd to the HF Space.  Full interactive dashboard is post-hackathon.
    """
    recent = _REWARD_HISTORY[-200:]
    avg = sum(r["reward"] for r in recent) / len(recent) if recent else 0
    gif_exists  = (_ASSET_DIR / "saccade_comparison.gif").exists()
    curve_exists = (_ASSET_DIR / "reward_curve.png").exists()
    tokens_exists = (_ASSET_DIR / "tokens_to_find.png").exists()
    static_html = ""
    if gif_exists:
        static_html += (
            '<div class="card"><div class="label">Saccade RL — baseline vs trained reading order</div>'
            '<img src="/assets/saccade_comparison.gif" style="width:100%;border-radius:6px;margin-top:8px"></div>'
        )
    if curve_exists:
        static_html += (
            '<div class="card"><div class="label">Reward curve (training)</div>'
            '<img src="/assets/reward_curve.png" style="width:100%;border-radius:6px;margin-top:8px"></div>'
        )
    if tokens_exists:
        static_html += (
            '<div class="card"><div class="label">Tokens-to-find-first-correct (per task)</div>'
            '<img src="/assets/tokens_to_find.png" style="width:100%;border-radius:6px;margin-top:8px"></div>'
        )
    if not static_html:
        static_html = (
            '<div class="card" style="border:1px dashed #374151;color:#9ca3af">'
            'Saccade-RL GIF, reward curve, and tokens-to-find chart will appear '
            'here once the training notebook has rsync\'d <code>assets/*.png|gif</code> '
            'to the Space. (Full interactive dashboard deferred — see README §Roadmap.)'
            '</div>'
        )
    return HTMLResponse(content=f"""<!DOCTYPE html><html><head><title>ScholarEnv Dashboard</title>
<style>body{{font-family:system-ui;background:#0f1117;color:#e5e7eb;margin:0;padding:20px;max-width:1100px;margin-left:auto;margin-right:auto}}
.card{{background:#1f2937;border-radius:8px;padding:16px;margin:12px 0}}
.big-num{{font-size:2.5rem;font-weight:700;color:#f97316}}
.label{{font-size:0.85rem;color:#9ca3af}}h1{{color:#f97316}}</style></head><body>
<h1>ScholarEnv Dashboard</h1>
<p style="color:#9ca3af;margin-top:-10px">
  Saccade RL: teaching a 1.5B model to read papers like a senior reviewer — out of order, evidence-first.
</p>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
<div class="card"><div class="label">Episodes</div><div class="big-num">{len(_REWARD_HISTORY)}</div></div>
<div class="card"><div class="label">Avg Reward</div><div class="big-num">{avg:.3f}</div></div>
<div class="card"><div class="label">Sessions</div><div class="big-num">{len(_SESSIONS)}</div></div>
</div>
{static_html}
<div class="card"><div class="label">Live reward stream (claim_evidence_audit)</div>
<canvas id="rc"></canvas></div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script>
fetch('/_reward_data').then(r=>r.json()).then(d=>{{
  new Chart(document.getElementById('rc').getContext('2d'),{{
    type:'line',data:{{labels:d.map((_,i)=>i+1),datasets:[{{label:'Reward',
    data:d.map(x=>x.reward),borderColor:'#f97316',backgroundColor:'rgba(249,115,22,0.1)',
    tension:0.3,fill:true,pointRadius:2}}]}},
    options:{{scales:{{y:{{min:0,max:1}}}},plugins:{{legend:{{display:true}}}}}}
  }});
}});
</script></body></html>""")

@app.get("/_reward_data")
async def reward_data():
    filtered = [r for r in _REWARD_HISTORY if r.get("task")=="claim_evidence_audit"]
    return JSONResponse(content=filtered[-200:])

def main(host="0.0.0.0", port=7860):
    import uvicorn
    uvicorn.run(app, host=host, port=int(os.environ.get("PORT", port)))

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(host=args.host, port=args.port)

# ── /transcripts — G4: action strategy logger ────────────────────────────────
import os as _os
_TRANSCRIPT_PATH = "/tmp/scholarenv_transcripts.jsonl"

@app.get("/transcripts")
async def transcripts(limit: int = 20):
    """
    G4: Episode transcripts showing agent action sequences.
    Enables behavioral comparison: untrained (random) vs trained (strategic).
    Each record: {action_sequence, sections_order, final_reward, n_steps}
    """
    if not _os.path.exists(_TRANSCRIPT_PATH):
        return JSONResponse(content={"transcripts": [], "total": 0})
    records = []
    try:
        with open(_TRANSCRIPT_PATH) as f:
            for line in f:
                try: records.append(json.loads(line.strip()))
                except Exception: pass
    except Exception: pass
    return JSONResponse(content={"transcripts": records[-limit:], "total": len(records)})

def _save_transcript(session_id: str, task_id: str,
                     action_log: list, final_reward: float) -> None:
    """Persist one episode transcript for behavioral dashboard."""
    try:
        import time as _t
        record = {
            "session_id":      session_id,
            "task_id":         task_id,
            "action_sequence": [a["action_type"] + ":" + a.get("target","") for a in action_log],
            "sections_order":  [a.get("target","") for a in action_log
                                 if a.get("action_type") == "query_section"],
            "final_reward":    round(final_reward, 4),
            "n_steps":         len(action_log),
            "ts":              _t.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(_TRANSCRIPT_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception: pass
