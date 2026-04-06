#!/usr/bin/env python3
"""
Baseline inference script for the Research Paper Formatter OpenEnv environment.

Runs an LLM agent against all 3 tasks and emits structured logs per OpenEnv spec.

Required environment variables:
    API_BASE_URL   LLM API endpoint (default: HuggingFace router)
    MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key / HuggingFace token
    PAPER_FORMATTER_TASK   Override task (default: runs all 3)

Usage:
    python inference.py
    PAPER_FORMATTER_TASK=task_hard python inference.py
"""

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional
from openai import OpenAI
from environment import PaperFormatterEnv
from models import PaperAction, ActionType, PaperObservation
from tasks import list_tasks

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  



# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "paper-formatter-v1"
TEMPERATURE = 0.2
MAX_TOKENS = 512

TASK_OVERRIDE = os.getenv("PAPER_FORMATTER_TASK", "")

# ──────────────────────────────────────────────
# Logging helpers (OpenEnv stdout spec)
# ──────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert academic paper formatting agent. You receive observations about a research paper
that must be reformatted from one conference style to another.

Your job: issue ONE action per turn that makes progress toward full compliance.

Available actions (respond with ONLY a JSON object, no markdown, no explanation):
  {"action_type": "set_format", "parameters": {"format": "IEEE"}}
  {"action_type": "rename_section", "parameters": {"old_name": "I. Introduction", "new_name": "Introduction"}}
  {"action_type": "reorder_sections", "parameters": {"order": ["Abstract", "Introduction", ...]}}
  {"action_type": "format_references", "parameters": {"style": "IEEE"}}
  {"action_type": "set_title_case", "parameters": {"style": "title_case"}}
  {"action_type": "set_abstract_word_limit", "parameters": {"limit": 150}}
  {"action_type": "remove_section", "parameters": {"name": "SectionName"}}
  {"action_type": "add_section", "parameters": {"name": "Name", "section_type": "methodology"}}
  {"action_type": "format_author_list", "parameters": {"style": "First Last"}}
  {"action_type": "set_column_layout", "parameters": {"columns": 2}}
  {"action_type": "format_citations", "parameters": {"style": "numeric"}}
  {"action_type": "submit", "parameters": {}}

Citation styles: "numeric" (e.g. [1]) or "author_year" (e.g. (Liu et al., 2019))
Reference styles: "IEEE", "ACM", "APA", "AAAI"
Author formats: "First Last" (full names) or "F. Last" (abbreviated)
Title case styles: "title_case", "sentence_case", "upper", "lower"

Rules:
- Fix the MOST CRITICAL issue first (largest reward impact)
- When all issues are fixed, call submit
- Output ONLY the raw JSON. No backticks. No explanation.
""").strip()


def build_user_prompt(obs: PaperObservation, step: int, last_reward: float,
                      history: List[str]) -> str:
    history_block = "\n".join(history[-5:]) if history else "None"
    issues_block = "\n".join(f"  - {i}" for i in obs.issues) if obs.issues else "  (none — all clear!)"
    fixed_block = "\n".join(f"  ✓ {i}" for i in obs.fixed_issues) if obs.fixed_issues else "  (none yet)"

    return textwrap.dedent(f"""
    === STEP {step} ===
    Paper: "{obs.paper_title}"
    Target format: {obs.target_format.value}

    CURRENT STATE:
      Sections: {obs.section_order}
      Authors: {[a.name for a in obs.authors]}
      Abstract words: {obs.abstract_word_count} (limit: {obs.target_spec.max_abstract_words})
      Column layout: {obs.column_layout} (target: {obs.target_spec.columns})
      Citation style: {obs.citation_style} (target: {obs.target_spec.citation_style})
      Reference styles: {list(set(r.style for r in obs.references))} (target: {obs.target_spec.reference_style})
      Author format: (target: {obs.target_spec.author_format})
      Title case: {obs.title_case_style} (target: {obs.target_spec.title_case})
      Compliance score: {obs.compliance_score:.3f}

    OPEN ISSUES (fix these):
    {issues_block}

    ALREADY FIXED:
    {fixed_block}

    STEP HISTORY:
    {history_block}

    Last reward: {last_reward:.3f}
    Steps remaining: {obs.max_steps - obs.steps_taken}

    Issue ONE action as raw JSON:
    """).strip()


# ──────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────

def get_model_action(client: OpenAI, obs: PaperObservation, step: int,
                     last_reward: float, history: List[str]) -> Dict[str, Any]:
    """Call the LLM and parse its JSON action."""
    prompt = build_user_prompt(obs, step, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[DEBUG] Model call failed: {e}", file=sys.stderr, flush=True)
        # Fallback: no-op submit
        return {"action_type": "submit", "parameters": {}}


def run_task(client: OpenAI, task_id: str) -> tuple[float, bool, int, List[float]]:
    """Run one complete episode. Returns (final_score, success, steps, rewards)."""
    env = PaperFormatterEnv(task_id=task_id)
    task = env.task_spec

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    obs = env.reset()
    last_reward = 0.0
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    try:
        for step in range(1, task.max_steps + 1):
            if obs.done:
                break

            # Get action from model
            action_dict = get_model_action(client, obs, step, last_reward, history)
            action_str = json.dumps(action_dict)

            # Parse and apply action
            try:
                action = PaperAction(
                    action_type=ActionType(action_dict.get("action_type", "submit")),
                    parameters=action_dict.get("parameters", {}),
                )
                result = env.step(action)
                reward = result.reward
                done = result.done
                error = result.info.get("error")
                obs = result.observation

            except Exception as e:
                reward = 0.0
                done = False
                error = str(e)

            rewards.append(reward)
            last_reward = reward
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_dict.get('action_type')} "
                f"params={action_dict.get('parameters')} → reward={reward:.3f}"
            )

            if done:
                break

        # Final score is the last reward (represents current compliance)
        final_score = rewards[-1] if rewards else 0.0
        success = final_score >= task.success_threshold

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", file=sys.stderr, flush=True)
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return final_score, success, steps_taken, rewards


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    tasks_to_run = [TASK_OVERRIDE] if TASK_OVERRIDE else list_tasks()

    all_results = []
    for task_id in tasks_to_run:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)

        score, success, steps, rewards = run_task(client, task_id)
        all_results.append({
            "task_id": task_id,
            "score": score,
            "success": success,
            "steps": steps,
        })

    # Summary
    print("\n" + "="*60, flush=True)
    print("BASELINE RESULTS SUMMARY", flush=True)
    print("="*60, flush=True)
    for r in all_results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        print(f"  {status}  {r['task_id']:20s}  score={r['score']:.3f}  steps={r['steps']}", flush=True)

    avg = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0
    print(f"\n  Average score: {avg:.3f}", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()
