"""
scripts/make_saccade_gif.py — Phase B3 standalone

Renders the side-by-side Saccade-RL GIF (baseline reading order vs trained
reading order) on a fixed contradiction paper, without needing the full
training notebook to be open.  Reads two completion strings from disk
(or accepts them as CLI args).

The notebook (Cell 12) has the same logic inline; this script exists so the
GIF can be rebuilt at any time once you have one baseline + one trained
completion saved as JSON files.

Usage:
    python scripts/make_saccade_gif.py \
        --baseline /content/baseline_completion.txt \
        --trained  /content/trained_completion.txt \
        --paper    /content/baseline_paper.json \
        --out      assets/saccade_comparison.gif
"""
from __future__ import annotations
import argparse, json, re, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

SECTION_TARGETS = ["Abstract", "Introduction", "Methods", "Results",
                   "Discussion", "Table 1", "Table 2", "References"]


def parse_reading_order(text: str) -> list[str]:
    seen, order = set(), []
    pat = (r"\b(abstract|introduction|methods?|results?|discussion|"
           r"references?|table\s*1|table\s*2)\b")
    for m in re.finditer(pat, text.lower()):
        token = m.group(1)
        if token.startswith("table"):
            normalized = "Table 1" if "1" in token else "Table 2"
        else:
            normalized = token.capitalize().rstrip("s")
            if normalized == "Reference": normalized = "References"
        if normalized not in seen:
            seen.add(normalized); order.append(normalized)
    return order


def step_reward(order: list[str], frame_idx: int) -> float:
    seen = set(order[:frame_idx + 1])
    have_table = any(s.startswith("Table") for s in seen)
    have_evid  = "Results" in seen
    return 0.10 + 0.45 * have_table + 0.30 * have_evid + \
           0.15 * (len(seen) / len(SECTION_TARGETS))


def step_tokens(order: list[str], frame_idx: int, paper: dict) -> int:
    secs = paper.get("sections", {})
    total = 0
    for s in order[:frame_idx + 1]:
        total += 16 if s.startswith("Table") else len(secs.get(s.lower(), "").split())
    return total


def build_gif(base_text: str, trained_text: str, paper: dict, out: str) -> None:
    base_order  = parse_reading_order(base_text)    or SECTION_TARGETS[:1]
    train_order = parse_reading_order(trained_text) or SECTION_TARGETS[:1]
    n_frames    = max(len(base_order), len(train_order))

    fig, axes = plt.subplots(2, 2, figsize=(13, 7),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0f1117")
    ax_b, ax_t, ax_br, ax_tr = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    ax_b.set_title("BASELINE (untrained Qwen2.5-1.5B)", color="#fb7185")
    ax_t.set_title("TRAINED (v6 LoRA, evidence-first saccade)", color="#34d399")

    def draw_panel(ax, order, frame, color):
        ax.clear(); ax.set_facecolor("#1f2937"); ax.set_xticks([]); ax.set_yticks([])
        for j, sec in enumerate(SECTION_TARGETS):
            is_seen = sec in order[:frame + 1]
            idx     = order.index(sec) if sec in order[:frame + 1] else None
            rect_color = color if is_seen else "#374151"
            ax.add_patch(plt.Rectangle((0.1, 0.85 - j * 0.10), 0.8, 0.08,
                                        color=rect_color,
                                        alpha=0.9 if is_seen else 0.4))
            ax.text(0.5, 0.85 - j * 0.10 + 0.04, sec,
                    ha="center", va="center", color="white", fontsize=10,
                    fontweight="bold" if is_seen else "normal")
            if is_seen:
                ax.text(0.95, 0.85 - j * 0.10 + 0.04, f"#{idx + 1}",
                        ha="right", va="center", color="white", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def draw_bar(ax, val, label, color):
        ax.clear(); ax.set_facecolor("#1f2937")
        ax.barh([label], [val], color=color)
        ax.set_xlim(0, 1); ax.tick_params(colors="#9ca3af")

    def update(frame):
        draw_panel(ax_b, base_order,  frame, "#fb7185")
        draw_panel(ax_t, train_order, frame, "#34d399")
        rb = step_reward(base_order,  frame); tb = step_tokens(base_order,  frame, paper)
        rt = step_reward(train_order, frame); tt = step_tokens(train_order, frame, paper)
        draw_bar(ax_br, rb, f"reward {rb:.2f}", "#fb7185")
        draw_bar(ax_tr, rt, f"reward {rt:.2f}", "#34d399")
        ax_br.set_xlabel(f"baseline tokens: {tb}", color="#9ca3af")
        ax_tr.set_xlabel(f"trained  tokens: {tt}", color="#9ca3af")
        return [ax_b, ax_t, ax_br, ax_tr]

    out_path = pathlib.Path(out); out_path.parent.mkdir(parents=True, exist_ok=True)
    ani = FuncAnimation(fig, update, frames=n_frames, interval=900, blit=False,
                        repeat=True)
    ani.save(str(out_path), writer=PillowWriter(fps=1))
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def _load(path: str) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--trained",  required=True)
    ap.add_argument("--paper",    required=True)
    ap.add_argument("--out",      default="assets/saccade_comparison.gif")
    args = ap.parse_args()

    paper = json.loads(_load(args.paper))
    build_gif(_load(args.baseline), _load(args.trained), paper, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
