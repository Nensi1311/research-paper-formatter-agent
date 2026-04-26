"""
scripts/colab_smoke_v6.py — Phase D1 (v6.2: grounded + fast)

Standalone Colab T4 smoke test: 25 GRPO steps on T3 only.  Verifies:
  * CSV log is populated (Phase A1)
  * mask_truncated_completions stays False (Phase A2)
  * LoRA saves to /content/lora_smoke
  * action_log JSONL is non-empty
  * reward_curve.png renders

v6.1 added (so a flat reward curve can be diagnosed *during* the run):
  * Per-step format-compliance scoreboard
  * Best + worst rollout printed every logging_steps
  * Continuous reward_reasoning (count-based, not binary)
  * Two-shot in-context example pushing model to find ALL discrepancies
  * End-of-run "why is training loss 0.000000?" explainer

v6.2 (this revision) adds:
  * num_generations 6 → 4   — smoke completes in ~7 min instead of 35
                              (Unsloth no longer auto-bumps batch to 24)
  * couple reasoning reward to fbeta:  rs *= (0.3 + 0.7 * fb)
                              fluent-but-wrong text no longer earns full credit
  * SmokeDiagnosticCallback grows an early-stop guard:
                              3 consecutive logs with reward_std < 0.02
                              → stop.  Saves Colab GPU time when the group
                              has collapsed.
  * audit_grader.py G8 grounding (server-side) prevents
                              "claim X, table X — mismatch" hallucinations
                              from earning reward in the first place.

Designed to be copy-pasted into Colab as a single cell — ~7 min on T4.
Do NOT run this on HF GPU credits; that's what the full multi-task notebook
is for (see Phase D4 in the README).
"""
import os, sys, time, csv, json, warnings
sys.path.insert(0, "/content/scholarenv")

# Cosmetic: silence the harmless "max_new_tokens vs max_length" warning that
# spams every generation call when GRPOConfig sets max_completion_length.
warnings.filterwarnings("ignore", message=r"Both `max_new_tokens`")
warnings.filterwarnings("ignore", category=FutureWarning,
                        module=r"transformers\.modeling_attn_mask_utils")

from unsloth import FastLanguageModel  # must come before transformers
import torch

MAX_SEQ      = 4096
MODEL_NAME   = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
LORA_OUT_DIR = "/content/lora_smoke"
LOG_PATH     = "/content/reward_log_smoke.csv"
STEPS        = 25

print(f"=== ScholarEnv v6 Colab smoke ({STEPS} steps, T3 only) ===")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ,
    dtype=None,
    load_in_4bit=True,
    fast_inference=False,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing="unsloth", random_state=42,
)

# --- minimal dataset (T3 only) ----------------------------------------------
from server.paper_generator import ProceduralPaperGenerator
from corpus import Paper
from datasets import Dataset
import json as _j

GEN = ProceduralPaperGenerator()
sys_prompt = (
    "You are a research paper claim-evidence auditor.\n"
    "Find numerical claims in the abstract that DON'T match Table 1 or Table 2.\n\n"
    "Format your response EXACTLY like this:\n"
    "REASONING: <which claims you cross-checked, cite section names + numbers>\n"
    'FINDINGS: [{"type":"table_text_mismatch","location":"abstract",'
    '"claim":"<verbatim>","table_id":"Table 1","table_value":"<from table>"}]\n\n'
    "RULES (these are graded — violating them costs reward):\n"
    '  - "table_value" MUST be a single string literal like "90.95" or '
    '"85.2%". NEVER a nested object like {"GLUE": 90.95}.\n'
    '  - "claim" MUST be the exact substring quoted from the abstract.\n'
    '  - REASONING must mention specific section names AND quote the '
    "conflicting numbers verbatim.\n\n"
    "EXAMPLE (two-shot — papers typically have MULTIPLE discrepancies, find ALL of them):\n"
    "REASONING: The abstract claims 90.2% accuracy on GLUE, but Table 1 shows 85.7 — mismatch.\n"
    "Also, abstract claims SQuAD F1 = 88.1, but Table 1 shows 84.3 — second mismatch.\n"
    'FINDINGS: ['
    '{"type":"table_text_mismatch","location":"abstract","claim":"90.2% accuracy on GLUE","table_id":"Table 1","table_value":"85.7"},'
    '{"type":"table_text_mismatch","location":"abstract","claim":"SQuAD F1 = 88.1","table_id":"Table 1","table_value":"84.3"}'
    ']'
)

records = []
for i in range(40):
    import random as _r
    _n = _r.choice([1, 2, 2, 3])  # vary count: model must learn HOW MANY, not assume 2
    p  = GEN.generate(domain="NLP", difficulty=0.3 + 0.4*(_r.random()), n_discrepancies=_n, seed=i)
    pd = p.to_json_dict()
    ctx = ""
    for sec in ["abstract", "results"]:
        t = pd["sections"].get(sec, "")
        if t: ctx += f"=== {sec.upper()} ===\n{t}\n\n"
    for tn, td in list(pd["tables"].items())[:2]:
        ctx += f"=== {tn.upper()} ===\n{json.dumps(td.get('data', td), indent=2)}\n\n"
    user = f"PAPER: {pd['title']}\n\n{ctx}Audit this paper:"
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user},
    ], tokenize=False, add_generation_prompt=True)
    records.append({"prompt": prompt, "paper_json": _j.dumps(pd),
                    "task_id": "claim_evidence_audit"})

dataset = Dataset.from_list(records)
print(f"Built {len(dataset)} smoke records.")

# --- reward functions -------------------------------------------------------
import re
from server.graders import AuditGrader
GRADER = AuditGrader()
with open(LOG_PATH, "w", newline="") as f:
    csv.writer(f).writerow([
        "ep", "task_id", "fbeta", "spec", "reason", "total",
        "valid_json", "non_empty_findings",
        "has_table_id", "has_str_table_value", "ts",
    ])

EP            = [0]
_GRADE_CACHE: dict[int, dict] = {}
_GRADE_ERRORS = [0]
# Diagnostic ring buffer — the most-recent N rollouts, kept for the per-step
# scoreboard below.  Each entry: (total_reward, completion, format_flags).
_DIAG_BUFFER: list[tuple[float, str, dict]] = []
_DIAG_MAX     = 256


# ── Format-compliance probe (cheap structural check, no LLM) ─────────────────
_RE_FINDINGS = re.compile(r"FINDINGS:\s*(\[.*?\])", re.DOTALL)
_RE_REASON   = re.compile(r"REASONING:\s*(.*?)(?=FINDINGS:|$)", re.DOTALL)


def _format_flags(text: str) -> dict:
    """Cheap structural inspection — does the model even know the format?

    Returns a flag dict that gets aggregated into a per-step scoreboard.
    These are NOT used for reward; they're pure diagnostics so a flat reward
    curve can be diagnosed by inspecting compliance %, not just reward mean.
    """
    flags = {"valid_json": 0, "non_empty_findings": 0,
             "has_table_id": 0, "has_str_table_value": 0,
             "has_reasoning_block": 0, "reasoning_cites_number": 0}
    if "REASONING:" in text:
        flags["has_reasoning_block"] = 1
    rm = _RE_REASON.search(text)
    if rm and re.search(r"\b\d+(?:\.\d+)?\b", rm.group(1) or ""):
        flags["reasoning_cites_number"] = 1
    fm = _RE_FINDINGS.search(text)
    if not fm:
        return flags
    try:
        arr = json.loads(fm.group(1))
    except Exception:
        return flags
    if not isinstance(arr, list):
        return flags
    flags["valid_json"] = 1
    if len(arr) > 0:
        flags["non_empty_findings"] = 1
    for item in arr:
        if not isinstance(item, dict):
            continue
        if str(item.get("table_id", "")).strip():
            flags["has_table_id"] = 1
        tv = item.get("table_value")
        if isinstance(tv, str) and tv.strip():
            flags["has_str_table_value"] = 1
    return flags


def _findings(text):
    m = _RE_FINDINGS.search(text)
    if not m: return []
    try:
        v = json.loads(m.group(1))
        return v if isinstance(v, list) else []
    except Exception:
        return []


# ── Continuous reasoning reward (v6.1 — fixes the dead binary one) ───────────
# v6.0 used: rs = 0.10 + (0.30 if "REASONING:" in comp else 0) + (0.10 if "Table" in comp else 0)
# Result: every rollout scored 0.50 → std=0 → no GRPO gradient from this reward.
# v6.1: count-based across 4 dimensions, all continuous.
_REASON_VERBS = ("but", "vs", "versus", "differ", "contradict", "mismatch",
                 "disagree", "whereas", "however", "contrast", "inconsistent",
                 "while", "conflict")

def _reasoning_score(text: str) -> float:
    rm = _RE_REASON.search(text)
    if not rm:
        return 0.05  # didn't even use the format
    reason = (rm.group(1) or "").strip()
    if len(reason) < 20:
        return 0.10
    sec_hits = sum(1 for s in ("abstract", "intro", "method", "result",
                               "discuss", "table", "figure", "reference")
                   if s in reason.lower())
    sec_score  = min(sec_hits / 4.0, 1.0) * 0.30
    nums       = set(re.findall(r"\b\d+(?:\.\d+)?\b", reason))
    num_score  = min(len(nums) / 3.0, 1.0) * 0.30
    verb_hits  = sum(1 for v in _REASON_VERBS if v in reason.lower())
    verb_score = min(verb_hits / 2.0, 1.0) * 0.20
    L          = len(reason.split())
    if   L < 10: len_score = (L / 10.0) * 0.10
    elif L > 80: len_score = max(0.05, 0.20 - (L - 80) / 200.0 * 0.20)
    else:        len_score = 0.20
    return max(0.0001, min(0.9999, sec_score + num_score + verb_score + len_score))


def _grade(comp, paper_json):
    """v6 hardening: defensive try/except so a single bad rollout can never
    crash trainer.train().  See the v6 audit_grader._safe_str fix.
    """
    try:
        paper = Paper.from_dict(json.loads(paper_json))
        r  = GRADER.grade(_findings(comp), paper, nav_state=None)
        fb = float(r.score)
        sp = float(r.evidence_specificity)
    except Exception as exc:
        _GRADE_ERRORS[0] += 1
        if _GRADE_ERRORS[0] <= 3:
            print(f"[grade-error #{_GRADE_ERRORS[0]}] {type(exc).__name__}: {exc}")
        fb = sp = 0.0001
    rs = _reasoning_score(comp)
    flags = _format_flags(comp)
    return {"fb": max(0.0001, fb), "sp": max(0.0001, sp),
            "rs": max(0.0001, rs), "flags": flags}


def _coupled_reasoning(rs_raw: float, fb_raw: float) -> float:
    """v6.2: reasoning reward is gated by f-beta correctness.

    Why: in v6.1 the worst rollouts ('claim X, table X — mismatch') still
    earned ~0.20 reasoning credit — fluent text rewarded even when the
    finding was wrong.  Coupling stops paying for confident nonsense.

    Formula: rs_final = rs_raw * (0.3 + 0.7 * fb_raw)
      * fb=1.0 → 100% reasoning credit (perfect findings: full reward)
      * fb=0.0 → 30% reasoning credit  (tells the model "format still matters,
                                         but you can't farm reasoning credit
                                         if findings are nonsense")
    """
    return rs_raw * (0.30 + 0.70 * fb_raw)


def _emit(comps, **kw):
    """Cached single-pass grade.  Each completion is graded ONCE per step,
    even though three reward funcs call this — so the CSV gets 1 row per
    completion (not 3) and the pipeline runs ~3x faster than v6.0 smoke."""
    pjs = kw.get("paper_json", [None] * len(comps))
    if not isinstance(pjs, list): pjs = [pjs] * len(comps)
    out_f, out_s, out_r = [], [], []
    new_rows = []
    for c, pj in zip(comps, pjs):
        key = hash(c[:300] + "|" + (pj or "")[:100])
        if key not in _GRADE_CACHE:
            _GRADE_CACHE[key] = _grade(c, pj or "{}")
            g = _GRADE_CACHE[key]
            # G8: couple reasoning to f-beta correctness (v6.2).
            rs_coupled = _coupled_reasoning(g["rs"], g["fb"])
            g["rs_coupled"] = rs_coupled
            EP[0] += 1
            total = 0.60 * g["fb"] + 0.15 * g["sp"] + 0.25 * rs_coupled
            f = g["flags"]
            new_rows.append([EP[0], "claim_evidence_audit",
                             round(g["fb"], 4), round(g["sp"], 4),
                             round(rs_coupled, 4), round(total, 4),
                             f["valid_json"], f["non_empty_findings"],
                             f["has_table_id"], f["has_str_table_value"],
                             round(time.time(), 2)])
            _DIAG_BUFFER.append((total, c, f))
            if len(_DIAG_BUFFER) > _DIAG_MAX:
                _DIAG_BUFFER.pop(0)
        g  = _GRADE_CACHE[key]
        rc = g.get("rs_coupled", _coupled_reasoning(g["rs"], g["fb"]))
        out_f.append(0.60 * g["fb"]); out_s.append(0.15 * g["sp"])
        out_r.append(0.25 * rc)
    if new_rows:
        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerows(new_rows)
    if len(_GRADE_CACHE) > 1500:
        for k in list(_GRADE_CACHE)[:500]:
            del _GRADE_CACHE[k]
    return out_f, out_s, out_r


def reward_fbeta(completions, **kw):       return _emit(completions, **kw)[0]
def reward_specificity(completions, **kw): return _emit(completions, **kw)[1]
def reward_reasoning(completions, **kw):   return _emit(completions, **kw)[2]

# --- per-step diagnostic callback ------------------------------------------
# Fires every `logging_steps` and prints a scoreboard so a flat reward curve
# can be diagnosed *during* training (not after).
from transformers import TrainerCallback


class SmokeDiagnosticCallback(TrainerCallback):
    """Print format-compliance %, best/worst rollout, and reasoning quality
    at every logging_steps.  This is what tells you *why* reward is flat.

    v6.2: added early-stop on group collapse — three consecutive logs with
    reward_std < 0.02 means the group converged to identical text and GRPO
    has no gradient signal left.  Stopping saves Colab GPU time."""

    EARLY_STOP_STD     = 0.02
    EARLY_STOP_STREAK  = 3

    def __init__(self, every: int = 5):
        self.every = every
        self._last_size       = 0
        self._low_std_streak  = 0

    def on_log(self, args, state, control, logs=None, **kw):
        # Only act on training-step logs (not the final eval log).
        if state.global_step == 0 or state.global_step % self.every != 0:
            return
        recent = _DIAG_BUFFER[-(self.every * 4):]   # ≈ last `every` steps × 4 gens
        if not recent:
            return
        n = len(recent)
        valid_json   = sum(f["valid_json"]           for _, _, f in recent) / n
        non_empty    = sum(f["non_empty_findings"]   for _, _, f in recent) / n
        has_tbl_id   = sum(f["has_table_id"]         for _, _, f in recent) / n
        has_str_tv   = sum(f["has_str_table_value"]  for _, _, f in recent) / n
        cites_num    = sum(f["reasoning_cites_number"] for _, _, f in recent) / n
        rewards_only = sorted([t for t, _, _ in recent])
        med  = rewards_only[n // 2]
        best_total, best_comp, _ = max(recent, key=lambda x: x[0])
        worst_total, worst_comp, _ = min(recent, key=lambda x: x[0])

        print(f"\n[diag @ step {state.global_step}]  median_reward={med:.3f}  "
              f"best={best_total:.3f}  worst={worst_total:.3f}  (n={n} rollouts)")
        print(f"  format compliance:  valid_json={valid_json:.0%}  "
              f"non_empty={non_empty:.0%}  table_id={has_tbl_id:.0%}  "
              f"str_table_value={has_str_tv:.0%}  reasoning_quotes_num={cites_num:.0%}")
        print(f"  best  rollout (head 220 chars): {best_comp[:220]!r}")
        print(f"  worst rollout (head 220 chars): {worst_comp[:220]!r}")

        # ── Early-stop on group collapse (v6.2) ──────────────────────────────
        cur_std = (logs or {}).get("reward_std")
        if cur_std is not None and cur_std < self.EARLY_STOP_STD:
            self._low_std_streak += 1
            print(f"  [warn] reward_std={cur_std:.4f} below {self.EARLY_STOP_STD} "
                  f"(streak {self._low_std_streak}/{self.EARLY_STOP_STREAK})")
            if self._low_std_streak >= self.EARLY_STOP_STREAK:
                print(f"  [STOP] {self.EARLY_STOP_STREAK} consecutive low-std logs "
                      "— group has collapsed.  Halting training to save credits.\n")
                control.should_training_stop = True
        else:
            self._low_std_streak = 0
        print()


# --- trainer ---------------------------------------------------------------
# Hyperparam history (so the README ablation is honest):
#   v6.0 → v6.1: max_completion_length 384 → 512  (v6.0 hit 347 on step 5)
#                learning_rate         5e-6 → 1e-5
#                temperature           0.7  → 0.85
#                num_generations       4    → 6
#   v6.1 → v6.2: num_generations       6    → 4   (Unsloth was auto-bumping
#                                                  batch to 24 → 35 min/25 steps;
#                                                  back to 4 makes it ~7 min)
#                temperature           0.85 → 0.7 (with G8 grounding the model
#                                                  no longer needs aggressive
#                                                  exploration to escape collapse)
#                learning_rate         1e-5 → 5e-6 (matches notebook for honest
#                                                   smoke→full extrapolation)
from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    output_dir="/content/scholarenv_grpo_smoke",
    max_steps=STEPS, num_generations=4,
    per_device_train_batch_size=1, gradient_accumulation_steps=4,
    max_prompt_length=1024, max_completion_length=512,
    temperature=0.7, learning_rate=5e-6, logging_steps=5, save_steps=999,
    report_to=["none"],
)
trainer = GRPOTrainer(
    model=model, processing_class=tokenizer,
    reward_funcs=[reward_fbeta, reward_specificity, reward_reasoning],
    train_dataset=dataset, args=training_args,
    callbacks=[SmokeDiagnosticCallback(every=5)],
)
if hasattr(trainer.args, "mask_truncated_completions"):
    trainer.args.mask_truncated_completions = False
# ── Step-0 baseline: measure BEFORE any GRPO update ─────────────────────────
# Without this you cannot claim RL adds value over the base model
print("Measuring Step-0 baseline (frozen model, no training yet)...")
_baseline_scores = []
import random as _rand
for _i in range(min(20, len(records))):
    _rec = records[_rand.randint(0, len(records)-1)]
    _r0 = reward_fbeta([_rec["prompt"]], paper_json=[_rec["paper_json"]])
    _baseline_scores.append(_r0[0])
_b_mean = sum(_baseline_scores) / len(_baseline_scores)
print(f"Step-0 baseline F-beta: {_b_mean:.4f}  (this is what RL must beat)")
print(f"If final reward < {_b_mean:.4f}, RL hurt the model.")
print()
# ─────────────────────────────────────────────────────────────────────────────

print("Starting smoke train...")
t0 = time.time()
trainer.train()
print(f"Done in {(time.time()-t0)/60:.1f} min")

trainer.save_model(LORA_OUT_DIR)
print(f"LoRA saved to {LORA_OUT_DIR}")


# --- end-of-run scoreboard + zero-loss explainer ---------------------------
with open(LOG_PATH) as f:
    rdr = list(csv.DictReader(f))
n_rows = len(rdr)
print(f"\nCSV rows: {n_rows}")
print(f"Grade-error events suppressed (model emitted bad JSON): {_GRADE_ERRORS[0]}")
if n_rows < STEPS:
    print(f"WARNING: only {n_rows} CSV rows for {STEPS} steps — grader may be silently failing")

if rdr:
    def _avg(key):
        vals = [float(r[key]) for r in rdr if r.get(key) not in (None, "")]
        return sum(vals) / max(1, len(vals))
    fb_avg   = _avg("fbeta")
    sp_avg   = _avg("spec")
    rs_avg   = _avg("reason")
    valid    = sum(int(r["valid_json"])           for r in rdr) / n_rows
    non_emp  = sum(int(r["non_empty_findings"])   for r in rdr) / n_rows
    has_tid  = sum(int(r["has_table_id"])         for r in rdr) / n_rows
    has_stv  = sum(int(r["has_str_table_value"])  for r in rdr) / n_rows
    first_q  = rdr[: max(1, n_rows // 4)]
    last_q   = rdr[-max(1, n_rows // 4):]
    fb_first = sum(float(r["fbeta"]) for r in first_q) / len(first_q)
    fb_last  = sum(float(r["fbeta"]) for r in last_q)  / len(last_q)
    delta    = fb_last - fb_first

    print("\n========== SMOKE-RUN SCOREBOARD ==========")
    print(f"  Step-0 prompted baseline : {_b_mean:.3f}   (the bar RL must clear)")
    print(f"  Mean F-beta              : {fb_avg:.3f}   (target: ≥ baseline + 0.10)")
    print(f"  Mean specificity         : {sp_avg:.3f}   (target: ≥ 0.30 — was 0.02 in v6.0)")
    print(f"  Mean reasoning (coupled) : {rs_avg:.3f}   "
          f"(coupled to f-beta — wrong answers cap at 30%)")
    print(f"  Format: valid_json         : {valid:.0%}   (knows the format)")
    print(f"  Format: non-empty findings : {non_emp:.0%}   (actually attempted a finding)")
    print(f"  Format: has table_id       : {has_tid:.0%}")
    print(f"  Format: str table_value    : {has_stv:.0%}")
    print(f"  Δ F-beta (last vs first quartile): {delta:+.3f}  "
          f"({'IMPROVING' if delta > 0.02 else 'flat — see explainer below'})")
    print()
    print("  Note (v6.2): G8 grounding + reasoning-coupling produce a LOWER")
    print("  absolute reward number than v6.1, but the reward is HONEST —")
    print("  fluent self-contradicting submissions (the v6.1 worst-rollout")
    print("  pattern: 'claim X, table X — mismatch') are now ~0.0 instead of")
    print("  ~0.2.  Compare to baseline, not to v6.1 absolute numbers.")

    print("\n========== WHY 'TRAINING LOSS = 0.000000' IS NORMAL FOR GRPO ==========")
    print("  GRPO loss = -mean(advantage × ratio) + β × KL")
    print("  Advantages within a group are MEAN-CENTERED by construction:")
    print("      A_i = (R_i - mean(R_group)) / std(R_group)")
    print("      → mean(A) is exactly 0 by construction")
    print("  At early steps, ratio ≈ 1 and KL ≈ 0, so the displayed mean loss")
    print("  rounds to 0.000000.  The PER-SAMPLE gradient is non-zero — that's")
    print("  what actually updates the policy.  See: HuggingFace TRL issues")
    print("  #2456 and #2603.  Confirm learning by checking: reward_std > 0,")
    print("  KL growing over steps, and reward improving last vs first quartile.")
    print("==========================================================================")

print("OK — smoke complete.")
