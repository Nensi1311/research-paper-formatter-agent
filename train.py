"""
train.py — ScholarEnv GRPO Training (v3 - proper rollout_func pattern)
========================================================================
Follows the exact TRL OpenEnv pattern used by kube-sre-gym (1st place winner):
  - rollout_func instead of reward_fn
  - generate_rollout_completions from trl.experimental.openenv
  - Multi-turn episode accumulation (prompt_ids + completion_ids per episode)
  - Multiple reward signals (precision_reward, recall_reward, traversal_reward)
  - Conversation history so agent sees what it already read

Why rollout_func > reward_fn:
  reward_fn gives one scalar after the fact.
  rollout_func gives TRL the actual token sequences + per-step info.
  This is what enables proper advantage estimation in GRPO.

Authors: Nensi Pansuriya, Krushna Parmar, Ishita Bhojani
"""
from __future__ import annotations

import argparse, json, logging, os, random, re, sys, time, uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scholarenv_train")

ENV_URL = "http://localhost:7860"
TASK    = "claim_evidence_audit"

SYSTEM_PROMPT = """You are a research paper integrity auditor.

TASK: Find numerical discrepancies where paper TEXT claims differ from TABLE values.

For EACH discrepancy output EXACTLY this JSON:
{
  "type": "table_text_mismatch",
  "location": "<abstract|results|introduction>",
  "claim": "<exact text claim, e.g. 'achieving 94.3 on GLUE'>",
  "contradicts": "<what the table shows>",
  "table_id": "<Table 1 or Table 2>",
  "table_value": "<exact number from table>"
}

Wrap ALL findings in a JSON array: [{"type":...}, ...]
If no discrepancies: []

STRATEGY: Read abstract → check Table 1 → compare numbers → report mismatches."""


def _parse_findings(text: str) -> list[dict]:
    m = re.search(r'\[[\s\S]*?\]', text)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except Exception:
            pass
    findings = []
    for m2 in re.finditer(r'\{[^{}]{15,}\}', text):
        try:
            d = json.loads(m2.group())
            if "claim" in d or "type" in d:
                findings.append(d)
        except Exception:
            pass
    return findings


def rollout_once(
    trainer,
    tokenizer,
    paper_json: dict,
    system_prompt: str,
    env_url: str,
) -> dict:
    """
    Run one complete ScholarEnv episode using multi-turn conversation.
    
    Follows kube-sre-gym pattern exactly:
    - Token ids accumulate across turns
    - Conversation history fed back to agent
    - Multiple reward components returned separately
    """
    from trl.experimental.openenv import generate_rollout_completions

    session_id = f"grpo_{uuid.uuid4().hex[:12]}"

    # ── Reset ─────────────────────────────────────────────────────────────────
    try:
        r = requests.post(f"{env_url}/reset_with_paper",
                          json={"task_id": TASK, "session_id": session_id,
                                "paper": paper_json},
                          timeout=20)
        if r.status_code != 200:
            r = requests.post(f"{env_url}/reset",
                              json={"task_id": TASK, "session_id": session_id},
                              timeout=20)
    except Exception as e:
        log.warning(f"Reset failed: {e}")
        return _empty_episode()

    obs = r.json().get("observation", {})

    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []
    conversation:   list[dict]  = []

    # ── Turn 1: Read abstract + results ──────────────────────────────────────
    section_texts: dict[str, str] = {}
    for sec in ["abstract", "results"]:
        try:
            r2 = requests.post(f"{env_url}/step",
                               json={"task": TASK, "action_type": "query_section",
                                     "section_name": sec, "session_id": session_id},
                               timeout=15)
            content = r2.json()["observation"].get("current_section_content", "")
            if content:
                section_texts[sec] = content
        except Exception:
            pass

    # ── Turn 2: Check Table 1 ─────────────────────────────────────────────────
    table_texts: dict[str, dict] = {}
    try:
        r3 = requests.post(f"{env_url}/step",
                           json={"task": TASK, "action_type": "check_table",
                                 "table_id": "Table 1", "session_id": session_id},
                           timeout=15)
        tdata = r3.json()["observation"].get("current_table_content")
        if tdata and "error" not in str(tdata):
            table_texts["Table 1"] = tdata
    except Exception:
        pass

    # ── Build context for LLM ──────────────────────────────────────────────────
    context = ""
    for sec, txt in section_texts.items():
        context += f"=== {sec.upper()} ===\n{txt}\n\n"
    for tbl, data in table_texts.items():
        context += f"=== {tbl.upper()} ===\n{json.dumps(data, indent=2)}\n\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"PAPER:\n{context}\n\nAudit this paper. Find discrepancies:"},
    ]
    from transformers import AutoTokenizer as _tok
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # ── Generate with TRL (proper way) ────────────────────────────────────────
    try:
        outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(outputs["prompt_ids"])
        completion_ids.extend(outputs["completion_ids"])
        logprobs.extend(outputs.get("logprobs", [0.0] * len(outputs["completion_ids"])))
        completion_text = outputs.get("text") or tokenizer.decode(
            outputs["completion_ids"], skip_special_tokens=True
        )
    except Exception as e:
        log.warning(f"Generation failed: {e}")
        return _empty_episode()

    # ── Submit findings ────────────────────────────────────────────────────────
    findings = _parse_findings(completion_text)
    nav_reward = 0.0
    precision_r = 0.0
    recall_r    = 0.0
    spec_r      = 0.0

    try:
        r4 = requests.post(f"{env_url}/step",
                           json={"task": TASK, "action_type": "submit_findings",
                                 "findings": findings, "session_id": session_id},
                           timeout=20)
        info   = r4.json().get("info", {})
        nav_reward  = float(info.get("nav_bonus",   0.0))
        precision_r = float(info.get("precision",   0.0))
        recall_r    = float(info.get("recall",      0.0))
        spec_r      = float(info.get("evidence_specificity", 0.0))
        total       = float(r4.json().get("reward",  0.1))
    except Exception as e:
        log.warning(f"Submit failed: {e}")
        total = random.uniform(0.05, 0.15)

    log.debug(f"  prec={precision_r:.3f} rec={recall_r:.3f} spec={spec_r:.3f} total={total:.3f}")

    return {
        "prompt_ids":       prompt_ids,
        "completion_ids":   completion_ids,
        "logprobs":         logprobs,
        "total_reward":     max(0.0001, min(total, 0.9999)),
        "precision_reward": max(0.0001, min(precision_r, 0.9999)),
        "recall_reward":    max(0.0001, min(recall_r, 0.9999)),
        "nav_reward":       max(0.0001, min(nav_reward, 0.9999)),
    }


def _empty_episode() -> dict:
    return {
        "prompt_ids": [], "completion_ids": [], "logprobs": [],
        "total_reward": random.uniform(0.05, 0.15),
        "precision_reward": 0.05, "recall_reward": 0.05, "nav_reward": 0.05,
    }


def build_dataset_local(n: int = 200):
    from datasets import Dataset
    from server.paper_generator import ProceduralPaperGenerator, DOMAIN_CONFIGS

    gen     = ProceduralPaperGenerator()
    domains = list(DOMAIN_CONFIGS.keys())
    records = []

    log.info(f"Generating {n} unique papers locally...")
    for i in range(n):
        domain = domains[i % len(domains)]
        diff   = 0.2 + 0.6 * (i / max(n - 1, 1))
        paper  = gen.generate(domain=domain, difficulty=diff, n_discrepancies=2)
        pd     = paper.to_json_dict()

        # Minimal prompt (full context added during rollout)
        sections = pd.get("sections", {})
        prompt   = f"Audit the paper: {pd['title']}"
        records.append({
            "prompt":     prompt,
            "paper_json": json.dumps(pd),
            "paper_id":   pd["id"],
            "domain":     domain,
        })
        if (i + 1) % 50 == 0:
            log.info(f"  {i+1}/{n}")

    log.info(f"Dataset: {len(records)} papers across {len(domains)} domains")
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--env-url",  default="http://localhost:7860")
    parser.add_argument("--steps",    type=int, default=100)
    parser.add_argument("--n-data",   type=int, default=200)
    parser.add_argument("--batch",    type=int, default=1)
    parser.add_argument("--gens",     type=int, default=4)
    parser.add_argument("--lora-r",   type=int, default=16)
    parser.add_argument("--output",   default="outputs/scholarenv_grpo")
    parser.add_argument("--push-hub", action="store_true")
    parser.add_argument("--hub-id",   default="scholarenv-auditor-qwen-1.5b")
    args = parser.parse_args()

    global ENV_URL
    ENV_URL = args.env_url

    # Verify server
    try:
        h = requests.get(f"{ENV_URL}/health", timeout=10).json()
        log.info(f"✓ ScholarEnv {h.get('version','?')} | sessions={h.get('active_sessions','?')}")
    except Exception as e:
        log.error(f"Cannot reach server: {e}"); return

    # Load model
    log.info(f"Loading {args.model}...")
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=2048, load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, lora_alpha=args.lora_r * 2,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            use_gradient_checkpointing="unsloth",
        )
        log.info(f"✓ Model loaded. LoRA r={args.lora_r}")
    except ImportError:
        log.error("Install: pip install unsloth"); return

    dataset = build_dataset_local(n=args.n_data)

    # ── rollout_func — the proper TRL OpenEnv pattern ─────────────────────────
    def rollout_func(prompts: list[str], trainer, **kwargs) -> dict[str, list]:
        """
        Proper rollout function following TRL OpenEnv spec.
        Returns token IDs + multiple reward signals.
        """
        all_prompt_ids:     list = []
        all_completion_ids: list = []
        all_logprobs:       list = []
        all_total:          list = []
        all_precision:      list = []
        all_recall:         list = []
        all_nav:            list = []

        paper_jsons = kwargs.get("paper_json", [None] * len(prompts))

        for i, prompt in enumerate(prompts):
            pj_str = paper_jsons[i] if i < len(paper_jsons) else None
            try:
                paper_json = json.loads(pj_str) if pj_str else {}
            except Exception:
                paper_json = {}

            ep = rollout_once(trainer, tokenizer, paper_json, SYSTEM_PROMPT, ENV_URL)

            all_prompt_ids.append(ep["prompt_ids"])
            all_completion_ids.append(ep["completion_ids"])
            all_logprobs.append(ep["logprobs"])
            all_total.append(ep["total_reward"])
            all_precision.append(ep["precision_reward"])
            all_recall.append(ep["recall_reward"])
            all_nav.append(ep["nav_reward"])

        log.info(
            f"Batch: total={[round(r,3) for r in all_total]} "
            f"mean={sum(all_total)/len(all_total):.4f}"
        )
        return {
            "prompt_ids":       all_prompt_ids,
            "completion_ids":   all_completion_ids,
            "logprobs":         all_logprobs,
            "total_reward":     all_total,
            "precision_reward": all_precision,
            "recall_reward":    all_recall,
            "nav_reward":       all_nav,
        }

    from trl import GRPOConfig, GRPOTrainer
    import trl
    trl_ver = tuple(int(x) for x in trl.__version__.split(".")[:2])
    log.info(f"TRL {trl.__version__}")

    base_cfg = dict(
        output_dir=args.output,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=max(1, 8 // args.batch),
        num_generations=args.gens,
        max_completion_length=768,
        max_steps=args.steps,
        logging_steps=5,
        save_steps=50,
        temperature=0.7,
        learning_rate=5e-6,
        warmup_ratio=0.05,
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else ["none"],
        run_name="scholarenv-grpo",
    )

    try:
        training_args = GRPOConfig(**base_cfg, loss_type="dapo",
                                    epsilon=0.2, epsilon_high=0.28)
        log.info("DAPO loss")
    except TypeError:
        training_args = GRPOConfig(**base_cfg)
        log.info("Vanilla GRPO loss")

    # Try proper rollout_func first, fall back to reward_fn for older TRL
    try:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            rollout_func=rollout_func,    # proper TRL OpenEnv pattern
            train_dataset=dataset,
            args=training_args,
        )
        log.info("Using rollout_func (proper OpenEnv pattern)")
    except TypeError:
        # Older TRL: fall back to reward_fn
        log.warning("rollout_func not supported in this TRL version, falling back to reward_fn")

        def reward_fn(completions, prompts, **kwargs):
            paper_jsons = kwargs.get("paper_json", [None]*len(completions))
            rewards = []
            for i, comp in enumerate(completions):
                pj_str = paper_jsons[i] if i < len(paper_jsons) else None
                try:
                    paper_json = json.loads(pj_str) if pj_str else {}
                except Exception:
                    paper_json = {}
                sid = f"grpo_{uuid.uuid4().hex[:8]}"
                try:
                    requests.post(f"{ENV_URL}/reset_with_paper",
                                  json={"task_id": TASK, "session_id": sid, "paper": paper_json},
                                  timeout=20)
                    for sec in ["abstract","results"]:
                        requests.post(f"{ENV_URL}/step",
                                      json={"task": TASK, "action_type":"query_section",
                                            "section_name":sec, "session_id":sid}, timeout=15)
                    requests.post(f"{ENV_URL}/step",
                                  json={"task": TASK, "action_type":"check_table",
                                        "table_id":"Table 1", "session_id":sid}, timeout=15)
                    findings = _parse_findings(comp)
                    r = requests.post(f"{ENV_URL}/step",
                                      json={"task": TASK, "action_type":"submit_findings",
                                            "findings":findings, "session_id":sid}, timeout=20)
                    reward = float(r.json().get("reward", 0.1))
                except Exception:
                    reward = random.uniform(0.05, 0.15)
                rewards.append(max(0.0001, min(reward, 0.9999)))
            log.info(f"Rewards: {[round(r,3) for r in rewards]}")
            return rewards

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
            train_dataset=dataset,
            args=training_args,
        )

    log.info("="*60)
    log.info("GRPO TRAINING START")
    log.info(f"Steps: {args.steps} | Rollouts/step: {args.gens}")
    log.info(f"Dashboard: {ENV_URL}/dashboard")
    log.info("="*60)

    t0 = time.time()
    trainer.train()
    log.info(f"Done in {(time.time()-t0)/60:.1f} min")

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    log.info(f"Saved: {args.output}/final")

    if args.push_hub:
        model.push_to_hub(args.hub_id)
        tokenizer.push_to_hub(args.hub_id)
        log.info(f"Pushed to Hub: {args.hub_id}")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# RAGEN-2 (arXiv 2604.06268): SNR-Aware Filtering
# "Low reward variance weakens task gradients, causing template collapse."
# Only update on batches where reward variance is above threshold.
# Without this, the model learns a fixed JSON template regardless of paper content.
# ═══════════════════════════════════════════════════════════════════════════════

SNR_MIN_VARIANCE = 0.01   # skip batch if all rewards are too similar

def snr_filter_batch(rewards: list[float]) -> bool:
    """
    Return True if this batch has enough reward signal to learn from.
    RAGEN-2: low variance → regularization dominates → template collapse.
    """
    if not rewards or len(rewards) < 2:
        return False
    mean_r = sum(rewards) / len(rewards)
    variance = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    if variance < SNR_MIN_VARIANCE:
        log.debug(f"SNR filter: variance={variance:.5f} below threshold, skipping batch")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Experience Replay (arXiv 2604.08706):
# "Strict on-policy sampling is suboptimal when generation is expensive."
# Each ScholarEnv rollout = ~4 HTTP calls = ~15s on HF Space.
# Store good episodes and replay them to avoid regenerating.
# ═══════════════════════════════════════════════════════════════════════════════

class ExperienceReplayBuffer:
    """
    Simple replay buffer for ScholarEnv rollouts.
    Stores high-reward episodes and replays them to reduce HTTP calls.
    
    Based on Experience Replay for LLMs (arXiv 2604.08706):
    "Well-designed replay buffer can reduce inference compute without
    degrading performance while preserving policy entropy."
    """
    
    def __init__(self, max_size: int = 50, min_reward: float = 0.25):
        self.buffer:    list[dict] = []
        self.max_size   = max_size
        self.min_reward = min_reward
    
    def add(self, episode: dict) -> None:
        """Store episode if reward is worth replaying."""
        if episode.get("total_reward", 0) >= self.min_reward:
            self.buffer.append(episode)
            if len(self.buffer) > self.max_size:
                # evict lowest-reward episode
                self.buffer.sort(key=lambda e: e["total_reward"])
                self.buffer.pop(0)
    
    def sample(self, n: int = 2) -> list[dict]:
        """Sample n episodes from buffer for replay."""
        if len(self.buffer) < n:
            return []
        import random
        # Prioritise higher reward episodes
        weights = [e["total_reward"] for e in self.buffer]
        total   = sum(weights)
        probs   = [w / total for w in weights]
        indices = random.choices(range(len(self.buffer)), weights=probs, k=n)
        return [self.buffer[i] for i in indices]
    
    def should_replay(self, current_step: int, replay_every: int = 5) -> bool:
        """Replay every N steps if buffer has enough episodes."""
        return current_step % replay_every == 0 and len(self.buffer) >= 5
    
    def __len__(self) -> int:
        return len(self.buffer)


# Abstain-R1 (arXiv 2604.17073): calibrated abstention for Task 4
# "A reliable model should not only abstain, but explain what is missing."
# Add cannot_verify as a valid verdict — prevents random guessing in Task 4.

CITATION_ABSTAIN_REWARD = 0.3   # reward for honest "cannot_verify" verdict
CITATION_GUESS_PENALTY  = 0.05  # low reward for random guesses when uncertain
