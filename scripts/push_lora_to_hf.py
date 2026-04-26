"""
scripts/push_lora_to_hf.py — Phase D3

Push a saved LoRA adapter folder to the HuggingFace model repo
`flyingmaverick/scholarenv-auditor-qwen-1.5b`.

This is safe to run from outside the training notebook (e.g. after the
notebook is shut down) — the notebook also includes inline upload logic in
Cell 9 that uses the same env vars.

Usage:
    export HF_TOKEN=hf_...
    export LORA_DIR=/content/lora_v6           # default if unset
    export HF_LORA_REPO=flyingmaverick/scholarenv-auditor-qwen-1.5b
    python scripts/push_lora_to_hf.py

Per the OpenEnv help guide §16, this uploads ADAPTERS only — never upcast
+ merge a 4-bit base.  The repo card lists the base model.

v6.2 hardening:
  * Verifies adapter_config.json is present (catches the
    'forgot trainer.save_model' bug before pushing junk to Hub)
  * Auto-generates a README.md model card if the LoRA folder lacks one
    (so the Hub page isn't blank)
  * Optionally bundles the reward CSV + transfer matrix as evidence
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"


def _ensure_model_card(lora_dir: Path, repo_id: str) -> None:
    """Write a minimal HF model card if the user didn't supply one."""
    card = lora_dir / "README.md"
    if card.exists():
        return
    txt = f"""---
base_model: {BASE_MODEL}
library_name: peft
tags:
  - lora
  - grpo
  - openenv
  - scholarenv
license: apache-2.0
---

# ScholarEnv Auditor (Qwen2.5-1.5B-Instruct LoRA)

LoRA adapters trained with GRPO on [ScholarEnv](https://huggingface.co/spaces/{repo_id.split('/')[0]}/scholar-env),
a 5-task procedural research-paper auditing environment.

## Tasks
1. `formatting_compliance` — IEEE rewrite
2. `internal_consistency` — abstract↔results contradictions
3. `claim_evidence_audit` — abstract claims vs Table 1/2 (G8 grounded)
4. `citation_verification` — ghost-author / impossible-claim refs
5. `prompt_injection_audit` — held-out, zero-shot

## Training
* Base: `{BASE_MODEL}`
* Method: GRPO + DAPO loss (when TRL ≥ 0.27)
* β-audit = 1.5 (precision-recall tilt to penalise under-reporting)
* G8 substring grounding active in `audit_grader.py`
* See `Final_last_run.ipynb` Cells 0-13 in the source repo for the full pipeline

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{BASE_MODEL}", device_map="auto")
tok  = AutoTokenizer.from_pretrained("{BASE_MODEL}")
model = PeftModel.from_pretrained(base, "{repo_id}")
```

Pair with the [ScholarEnv environment](https://huggingface.co/spaces/{repo_id.split('/')[0]}/scholar-env) for end-to-end evaluation.
"""
    card.write_text(txt, encoding="utf-8")
    print(f"[INFO] Generated model card: {card}")


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] Set HF_TOKEN before running.", file=sys.stderr)
        return 1

    lora_dir = Path(os.environ.get("LORA_DIR", "/content/lora_v6"))
    if not lora_dir.exists():
        print(f"[ERROR] LoRA dir not found: {lora_dir}", file=sys.stderr)
        return 1

    # Sanity: a real LoRA save has adapter_config.json + adapter weights.
    if not (lora_dir / "adapter_config.json").exists():
        print(f"[ERROR] {lora_dir}/adapter_config.json missing — "
              f"this isn't a saved LoRA.  Did trainer.save_model() run?",
              file=sys.stderr)
        return 1

    repo_id = os.environ.get("HF_LORA_REPO",
                              "flyingmaverick/scholarenv-auditor-qwen-1.5b")

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("[ERROR] pip install huggingface_hub first.", file=sys.stderr)
        return 1

    _ensure_model_card(lora_dir, repo_id)

    login(token=token, add_to_git_credential=False)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True,
                    private=False)
    api.upload_folder(
        folder_path=str(lora_dir),
        repo_id=repo_id,
        commit_message="v6.2 LoRA (G8 grounding + reasoning-couple)",
        ignore_patterns=["*.pt", "optimizer.pt"],   # save Hub bandwidth
    )
    print(f"[DONE] Pushed: https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
