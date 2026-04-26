# HF GPU credit plan ($30 budget) — Phase D4

This file tracks how the $30 HuggingFace GPU credits get spent on the v6
ScholarEnv runs.  Designed so the dev iteration happens on a cheap A10G
and only the **final** run touches A100 80GB.

## Current status

| Slot | GPU                | Estimated cost | Run                                         | Status   |
|:----:|:-------------------|:---------------:|:--------------------------------------------|:---------|
| 1    | Colab T4 (free)    | $0              | 25-step T3 smoke (Phase D1)                 | pending  |
| 2    | HF A10G            | ~$4             | 200-step 4-task GRPO (Round 1)              | pending  |
| 3    | HF A100 80GB       | ~$12            | 300-step 4-task + T5 final run (Round 2)    | pending  |
| —    | reserve            | ~$13            | re-runs / inference eval / dashboard host    | pending  |

Total committed = ~$16, reserve = ~$13–14.  Avoid temptation to do
everything on A100 — A10G is enough for dev iteration on a 1.5B base.

## Round 1 — A10G ($1.05/hr, ~4hr)

**Purpose:** confirm multi-task interleaving + cross-task transfer numbers.

```
GPU                : A10G
TRL                : pip install -U trl  (>=0.27 → enables GDPO)
N_PER_TASK         : 50
TRAIN_TASKS        : T1, T2, T3, T4
max_steps          : 200
num_generations    : 4
gradient_accum     : 8
expected_runtime   : ~3.5 hr
expected_cost      : ~$3.70
artifacts          : /content/reward_log.csv,
                     /content/reward_curve.png,
                     /content/lora_v6/  (uploaded by Cell 9)
                     /content/transfer_matrix.json (Cell 11)
                     /content/tokens_to_find.png  (Cell 11)
```

After Round 1 finishes:
1. Inspect transfer_matrix.json — does T5 (zero-shot) beat baseline?  If
   yes, we have the saccade story for the README.  If no, drop T5 from
   the headline claim and report it under §honest-red-team.
2. Update README per-task table with real numbers.
3. Push LoRA to `flyingmaverick/scholarenv-auditor-qwen-1.5b` (Cell 9 inline
   or `python scripts/push_lora_to_hf.py`).

## Round 2 — A100 80GB ($4.13/hr, ~3hr)

**Purpose:** final demo run with more diversity + T5 in the training mix.

```
GPU                : A100 80GB
N_PER_TASK         : 60
TRAIN_TASKS        : T1, T2, T3, T4, T5      ← T5 promoted from held-out
max_steps          : 300
num_generations    : 8                         ← double the diversity
gradient_accum     : 4
expected_runtime   : ~2.8 hr
expected_cost      : ~$11.50
artifacts          : final reward_curve.png,
                     final assets/saccade_comparison.gif,
                     final LoRA in /content/lora_v6_final/
```

Notes:
* If A100 also OOMs at the multi-task batch, drop `num_generations` to
  4 first — that preserves more LoRA rank than reducing per-device batch.
* Save the A10G LoRA artifact even after A100 finishes — useful for the
  cross-LoRA ablation in the writeup.

## Risk register

| Risk | Mitigation |
|---|---|
| TRL version on HF GPU image doesn't expose `reward_aggregation_method` | `Final_last_run.ipynb` Cell 8 falls through three configs (DAPO+GDPO → DAPO → vanilla) and prints `config_label`.  Drop GDPO claim from writeup if config_label != "DAPO + GDPO + scale_rewards=batch". |
| Unsloth flips `mask_truncated_completions` back to True | Re-asserted in Cell 8 (post-construct) AND Cell 9 (pre-train). |
| HF Space build cache is stale after `.dockerignore` fix | First push: `git push --force` once with `[hf_skip_build_cache]` in the commit message. |
| Round 1 transfer matrix is bad → narrative collapses | Have a "negative result" fallback paragraph in README §honest-red-team — Round 2 then becomes the bid for full transfer with T5 in train. |
