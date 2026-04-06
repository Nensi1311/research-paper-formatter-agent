#!/usr/bin/env python3
"""
inference.py — ScholarEnv baseline agent (OpenEnv hackathon submission).

MANDATORY REQUIREMENTS (problem statement):
  - Named inference.py, in root directory
  - Uses: API_BASE_URL, MODEL_NAME, HF_TOKEN environment variables
  - Emits structured stdout: [START] / [STEP] / [END]   ← auto-scored in Phase 2
  - Uses OpenAI client for all LLM calls
  - Completes in < 20 minutes on 2vCPU / 8GB

LOG FORMAT (exact — deviation = incorrect Phase 2 scoring):
  [START] task=<id>  env=scholar-env  model=<MODEL_NAME>
  [STEP]  step=<N>   action=<str>     reward=<float>  done=<bool>  error=<None|str>
  [END]   success=<bool>  steps=<N>   score=<float>   rewards=[...]
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.environ["MODEL_NAME"]
API_KEY      = os.environ["HF_TOKEN"]
SPACE_URL    = os.environ.get("HF_SPACE_URL",
               "https://your-username-scholar-env.hf.space").rstrip("/")

TEMPERATURE       = 0.1
MAX_TOKENS        = 4000
MAX_STEPS         = 6
SUCCESS_THRESHOLD = 0.60

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ── Structured logging (mandatory per problem statement) ─────────────────────

def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(*, step: int, action: str, reward: float,
             done: bool, error: Optional[str] = None) -> None:
    print(f"[STEP]  step={step} action={action!r:.80} "
          f"reward={reward:.4f} done={done} error={error}", flush=True)

def log_end(*, success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(f"[END]   success={success} steps={steps} "
          f"score={score:.4f} rewards={[round(r,4) for r in rewards]}", flush=True)


# ── LLM helper ────────────────────────────────────────────────────────────────

def llm(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system",
                     "content": "You are a scholarly research integrity expert."},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
                stream=False,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"[DEBUG] LLM attempt {attempt+1} failed: {exc}", flush=True)
            if attempt == 2:
                return ""
            time.sleep(2 ** attempt)
    return ""

def parse_json_safe(text: str) -> list | None:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n")
                         if not l.strip().startswith("```"))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'(\[.*?\])', text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return None


# ── Env client ────────────────────────────────────────────────────────────────

async def env_reset(task_id: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(f"{SPACE_URL}/reset", json={"task_id": task_id})
        r.raise_for_status()
        return r.json()

async def env_step(action: dict) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(f"{SPACE_URL}/step", json=action)
        r.raise_for_status()
        return r.json()


# ── Task runners ──────────────────────────────────────────────────────────────

async def run_task1() -> tuple[float, List[float], int]:
    """formatting_compliance: iterative refinement up to 3 steps."""
    result = await env_reset("formatting_compliance")
    obs = result["observation"]
    rewards: List[float] = []

    for s in range(1, 4):
        prompt = (
            f"You are an expert IEEE manuscript formatter.\n\n"
            f"TASK: {obs['task_description']}\n\n"
            f"RULES:\n{json.dumps(obs.get('style_guide', {}), indent=2)}\n\n"
            f"MANUSCRIPT:\n{obs.get('manuscript_text', '')}\n"
            + (f"\nFEEDBACK: {obs['feedback']}" if obs.get("feedback") else "")
            + (f"\nHINT: {obs['hint']}"         if obs.get("hint") else "")
            + "\n\nReturn ONLY the fully reformatted manuscript."
        )
        text = llm(prompt, max_tokens=4000)
        r = await env_step({"task": "formatting_compliance",
                             "formatted_text": text})
        rew  = float(r.get("reward", 0.0))
        done = bool(r.get("done", False))
        rewards.append(rew)
        log_step(step=s, action=f"[submit len={len(text)}]",
                 reward=rew, done=done)
        obs = r["observation"]
        if done:
            return rew, rewards, s

    return max(rewards), rewards, 3


async def run_task2() -> tuple[float, List[float], int]:
    """internal_consistency: navigate sections → precision-filtered submit."""
    result   = await env_reset("internal_consistency")
    obs      = result["observation"]
    findings: list = []
    rewards:  List[float] = []
    step_num  = 0

    for sec in obs.get("available_sections", [])[:3]:
        step_num += 1
        nav = await env_step({"task": "internal_consistency",
                               "action_type": "query_section",
                               "section_name": sec})
        content  = nav["observation"].get("current_section_content", "")
        nav_rew  = float(nav.get("reward", 0.0))
        nav_done = bool(nav.get("done", False))
        rewards.append(nav_rew)
        log_step(step=step_num, action=f"query_section:{sec}",
                 reward=nav_rew, done=nav_done)
        if nav_done:
            break

        if content:
            prompt = (
                f"Audit for INTERNAL contradictions only (no external knowledge).\n"
                f"SECTION: {sec}\nCONTENT:\n{content[:3000]}\n\n"
                f"ALREADY FOUND:\n{json.dumps(findings)}\n\n"
                "Types: number_mismatch, missing_reference, contribution_count, other\n"
                'JSON array only: [{"type":"..","location":"..","claim":"..","contradicts":".."}]'
                "\nIf none: []"
            )
            new = parse_json_safe(llm(prompt, max_tokens=600))
            if isinstance(new, list):
                seen = {(f.get("type"), f.get("location")) for f in findings}
                for item in new:
                    sig = (item.get("type"), item.get("location"))
                    if sig not in seen:
                        findings.append(item)
                        seen.add(sig)

    step_num += 1
    sub = await env_step({"task": "internal_consistency",
                           "action_type": "submit_findings",
                           "findings": findings})
    rew  = float(sub.get("reward", 0.0))
    done = bool(sub.get("done", True))
    rewards.append(rew)
    log_step(step=step_num, action=f"submit_findings:[{len(findings)}]",
             reward=rew, done=done)
    return rew, rewards, step_num


async def run_task3() -> tuple[float, List[float], int]:
    """claim_evidence_audit: strategic nav → extract_claims → tables → submit."""
    result   = await env_reset("claim_evidence_audit")
    obs      = result["observation"]
    findings: list = []
    rewards:  List[float] = []
    step_num  = 0

    avail_secs   = obs.get("available_sections", [])
    avail_tables = obs.get("available_tables",   [])

    # Priority: results > abstract > introduction > others
    priority = []
    for target in ("results", "abstract", "introduction", "methods"):
        for sec in avail_secs:
            if target in sec.lower() and sec not in priority:
                priority.append(sec)
                break
    for sec in avail_secs:
        if sec not in priority:
            priority.append(sec)

    section_contents: dict[str, str] = {}

    # A — query top 3 sections
    for sec in priority[:3]:
        step_num += 1
        nav = await env_step({"task": "claim_evidence_audit",
                               "action_type": "query_section",
                               "section_name": sec})
        content  = nav["observation"].get("current_section_content", "")
        nav_rew  = float(nav.get("reward", 0.0))
        nav_done = bool(nav.get("done", False))
        rewards.append(nav_rew)
        log_step(step=step_num, action=f"query_section:{sec}",
                 reward=nav_rew, done=nav_done)
        if content:
            section_contents[sec] = content
        if nav_done:
            break

    # B — extract_claims from results
    extracted: list = []
    res_sec = next((s for s in priority[:3] if "result" in s.lower()), None)
    if res_sec and step_num < MAX_STEPS - 2:
        step_num += 1
        ext = await env_step({"task": "claim_evidence_audit",
                               "action_type": "extract_claims",
                               "section_name": res_sec})
        extracted = ext["observation"].get("extracted_claims", []) or []
        ext_rew   = float(ext.get("reward", 0.0))
        ext_done  = bool(ext.get("done", False))
        rewards.append(ext_rew)
        log_step(step=step_num, action=f"extract_claims:{res_sec}",
                 reward=ext_rew, done=ext_done)

    # C — check tables referenced in claims (+ first 2 available)
    ref_tables: set[str] = set()
    for claim in extracted:
        for tref in claim.get("table_refs", []):
            ref_tables.add(tref)
    for t in avail_tables[:2]:
        ref_tables.add(t)

    table_data: dict[str, dict] = {}
    for tid in list(ref_tables)[:2]:
        if step_num >= MAX_STEPS - 1:
            break
        step_num += 1
        tc = await env_step({"task": "claim_evidence_audit",
                              "action_type": "check_table", "table_id": tid})
        tdata    = tc["observation"].get("current_table_content")
        tc_rew   = float(tc.get("reward", 0.0))
        tc_done  = bool(tc.get("done", False))
        rewards.append(tc_rew)
        log_step(step=step_num, action=f"check_table:{tid}",
                 reward=tc_rew, done=tc_done)
        if tdata and "error" not in tdata:
            table_data[tid] = tdata

    # D — LLM cross-reference → findings
    for sec, content in section_contents.items():
        if not content:
            continue
        prompt = (
            "Find CLAIM-EVIDENCE discrepancies: text claims ≠ table values.\n\n"
            f"SECTION: {sec}\nCONTENT:\n{content[:2500]}\n\n"
            f"TABLE DATA:\n{json.dumps(table_data, indent=2)[:2000]}\n\n"
            "Only report discrepancies you can CONFIRM from the table data above.\n"
            'JSON array: [{"type":"table_text_mismatch","location":"..","claim":"..","contradicts":"..","table_id":"..","table_value":".."}]\n'
            "If none confirmed: []"
        )
        new = parse_json_safe(llm(prompt, max_tokens=800))
        if isinstance(new, list):
            seen = {(f.get("table_id"), f.get("claim", "")[:40]) for f in findings}
            for item in new:
                sig = (item.get("table_id"), item.get("claim", "")[:40])
                if sig not in seen:
                    findings.append(item)
                    seen.add(sig)

    # E — submit
    step_num += 1
    sub = await env_step({"task": "claim_evidence_audit",
                           "action_type": "submit_findings",
                           "findings": findings})
    rew  = float(sub.get("reward", 0.0))
    done = bool(sub.get("done", True))
    rewards.append(rew)
    log_step(step=step_num, action=f"submit_findings:[{len(findings)}]",
             reward=rew, done=done)
    return rew, rewards, step_num


# ── Main ──────────────────────────────────────────────────────────────────────

RUNNERS = {
    "formatting_compliance": run_task1,
    "internal_consistency":  run_task2,
    "claim_evidence_audit":  run_task3,
}

async def main() -> None:
    overall: dict[str, float] = {}
    t0 = time.time()

    for task_id, runner in RUNNERS.items():
        log_start(task=task_id, env="scholar-env", model=MODEL_NAME)

        score = 0.0
        rewards: List[float] = []
        steps   = 0
        success = False

        try:
            score, rewards, steps = await runner()
            score   = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_THRESHOLD
        except Exception as exc:
            print(f"[DEBUG] {task_id} error: {exc}", flush=True)

        log_end(success=success, steps=steps, score=score, rewards=rewards)
        overall[task_id] = round(score, 4)
        print(f"[DEBUG] elapsed={time.time()-t0:.1f}s", flush=True)

    avg = sum(overall.values()) / len(overall)
    print(f"\n[SUMMARY] average={avg:.4f} scores={overall}", flush=True)

    with open("baseline_scores.json", "w") as f:
        json.dump({"scores": overall, "average": round(avg, 4),
                   "model": MODEL_NAME}, f, indent=2)
    print("[DEBUG] baseline_scores.json written", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
