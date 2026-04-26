"""
scripts/deploy_to_hf_space.py — Phase D2

Pushes the v6 ScholarEnv server to the HuggingFace Space
`flyingmaverick/scholar-env` (Docker SDK).  Honors `.dockerignore` so refs/,
notebooks, zips, etc. are never uploaded.

Usage:
    export HF_TOKEN=hf_...
    python scripts/deploy_to_hf_space.py

Optional env vars:
    HF_SPACE_REPO   default: flyingmaverick/scholar-env
    HF_TOKEN        required to push
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE_REPO = os.environ.get("HF_SPACE_REPO", "flyingmaverick/scholar-env")

# Files / directories that must NEVER be uploaded to the Space, regardless of
# what the user has lying around in the working tree.  Mirrors .dockerignore.
EXCLUDE_PATTERNS = {
    ".git", ".github", "__pycache__", ".pytest_cache",
    "refs", "_nb_src.txt", "friends_advice_from_cursor.txt",
    "SCHOLARENV_COMPLETE.md", "_apply_nb_v6.py",
    "_smoke_grading.py", "_smoke_nb.py", "_inspect_nb.py", "_smoke_test.py",
    "venv", ".venv", "scholarenv_grpo", "lora_v6",
}


def _filter(item: Path) -> bool:
    parts = set(item.parts)
    if parts & EXCLUDE_PATTERNS:
        return False
    if item.suffix in {".zip", ".tar", ".tar.gz"}:
        return False
    if item.suffix == ".ipynb" and item.name not in {"Final_last_run.ipynb"}:
        return False
    if item.name.startswith("_"):
        # local scratch files like _cell5.txt
        return False
    return True


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] Set HF_TOKEN before running this script.", file=sys.stderr)
        return 1

    # Pre-flight: a Docker Space must have a Dockerfile at the repo root.
    if not (ROOT / "Dockerfile").exists():
        print(f"[ERROR] {ROOT}/Dockerfile missing — Docker Space won't build.",
              file=sys.stderr)
        return 1
    if not (ROOT / "requirements.txt").exists():
        print(f"[WARN] requirements.txt missing — Docker build may fail")

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("[ERROR] pip install huggingface_hub first.", file=sys.stderr)
        return 1

    login(token=token, add_to_git_credential=False)
    api = HfApi()
    api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="docker",
                    exist_ok=True, private=False)
    print(f"[OK] Space exists: https://huggingface.co/spaces/{SPACE_REPO}")

    # Single source of truth for exclusions: glob patterns passed to upload_folder.
    # The Hub will respect these AND the .dockerignore at build time.
    ignore = [
        # Obvious junk
        ".git/**", ".github/**", "**/__pycache__/**", "**/.pytest_cache/**",
        "**/.ipynb_checkpoints/**",
        # Local research notes / refs / scratch
        "refs/**", "_*.py", "_*.txt", "_*.md", "_*.json",
        "friends_advice_from_cursor.txt", "SCHOLARENV_COMPLETE.md",
        # Trained artifacts (way too big for a Space; live on Model Hub instead)
        "scholarenv_grpo/**", "scholarenv_grpo_smoke/**",
        "lora_v6/**", "lora_smoke/**",
        # Notebooks (Final_last_run.ipynb included; checkpoints excluded)
        "**/.ipynb_checkpoints/**",
        # Archives + venvs
        "*.zip", "*.tar", "*.tar.gz", "venv/**", ".venv/**",
        # Skill / hook scaffolding (not part of the env)
        ".cursor/**",
    ]

    print(f"[INFO] Uploading repo from {ROOT} → {SPACE_REPO}...")
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=SPACE_REPO,
        repo_type="space",
        commit_message="ScholarEnv v6.2: G8 grounding + reasoning-couple + Saccade-RL",
        ignore_patterns=ignore,
    )
    space_host = SPACE_REPO.replace("/", "-")
    print("[DONE] Pushed.  Verify with:")
    print(f"  curl https://{space_host}.hf.space/health")
    print(f"  curl https://{space_host}.hf.space/v1/info")
    print(f"  open  https://huggingface.co/spaces/{SPACE_REPO}")
    return 0


# `_filter` is no longer used by main() but kept for backwards-compat with
# any external scripts that import it.  Will be removed in v6.3.
_LEGACY_FILTER_KEPT_FOR_BACKCOMPAT = _filter


if __name__ == "__main__":
    sys.exit(main())
