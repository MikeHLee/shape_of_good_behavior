#!/usr/bin/env python3
"""Figures for X/Twitter thread #3: "The bug that made our RL look broken".

Numbers load from:
- shared/results/finetune/sgb005b_rm_scores.json / sgb005b_rm_scores_hodge.json
  (post-fix RM ranking accuracy, computed from the per-pair scores)
- shared/results/finetune/sgb004_exploit_resistance_holdout_v2.json
  (pre-fix chance-level accuracies + token stats, regex-extracted from the
  file's own `root_cause_fixed` provenance prose so nothing is hand-entered)

Usage: ./venv/bin/python3 threads/03_the_bug_that_broke_our_rl/generate_figures.py
Outputs PNGs into threads/03_the_bug_that_broke_our_rl/figures/
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FT = os.path.join(REPO, "shared", "results", "finetune")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"    # standard RM
ORANGE = "#eb6834"  # Hodge RM
CRIT = "#d03b3b"
GOOD = "#0ca30c"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASE,
    "xtick.color": INK2,
    "ytick.color": MUTED,
})


def load():
    with open(os.path.join(FT, "sgb004_exploit_resistance_holdout_v2.json")) as f:
        v2 = json.load(f)
    prose = v2["root_cause_fixed"]["verification"]
    m = re.search(r"(\d+\.\d+)%/(\d+\.\d+)%\s*\(chance\)", prose)
    pre_std, pre_hodge = float(m.group(1)), float(m.group(2))

    bug_prose = v2["root_cause_fixed"]["bug"]
    tok = re.search(r"averages ~(\d+) tokens \(median (\d+), max (\d+)\)",
                    bug_prose)
    mean_tok, med_tok, max_tok = map(int, tok.groups())

    post = {}
    for name, fname in (("standard", "sgb005b_rm_scores.json"),
                        ("hodge", "sgb005b_rm_scores_hodge.json")):
        with open(os.path.join(FT, fname)) as f:
            d = json.load(f)
        post[name] = {
            split: 100.0 * sum(
                1 for r in rows if r["rm_ideal"] > r["rm_exploit"]
            ) / len(rows)
            for split, rows in d.items()
        }
    return (pre_std, pre_hodge), post, (mean_tok, med_tok, max_tok)


def headline(fig, title, subtitle):
    fig.text(0.05, 0.955, title, fontsize=19, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.05, 0.895, subtitle, fontsize=12.5, color=INK2,
             ha="left", va="top")


def footer(fig, text):
    fig.text(0.05, 0.02, text, fontsize=9.5, color=MUTED, ha="left")


# ---------------------------------------------------------------- figure 1
# Before/after: RM ranking accuracy at chance vs at 100%.
def fig1_before_after(pre, post):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), dpi=160, sharey=True)
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.08, right=0.97,
                        wspace=0.12)

    panels = [
        ("BEFORE the fix (retracted run)",
         [("Standard RM", pre[0], BLUE), ("Hodge RM", pre[1], ORANGE)],
         "reference pairs, mixed split"),
        ("AFTER the fix",
         [("Standard RM\n(train / holdout)",
           (post["standard"]["train"], post["standard"]["holdout"]), BLUE),
          ("Hodge RM\n(train / holdout)",
           (post["hodge"]["train"], post["hodge"]["holdout"]), ORANGE)],
         "217 train + 51 holdout pairs"),
    ]

    for pi, (ax, (title, bars, sub)) in enumerate(zip(axes, panels)):
        xs = []
        pos = 0.0
        for label, val, color in bars:
            vals = val if isinstance(val, tuple) else (val,)
            for j, v in enumerate(vals):
                ax.bar(pos, v, width=0.7, color=color, zorder=3,
                       alpha=1.0 if j == 0 else 0.72)
                if pi == 0:  # chance-level bars: label inside, clear of line
                    ax.text(pos, v - 3, f"{v:.1f}%", ha="center", va="top",
                            fontsize=12.5, color="#ffffff",
                            fontweight="bold", zorder=4)
                else:
                    ax.text(pos, v + 2.5, f"{v:.1f}%", ha="center",
                            va="bottom", fontsize=12.5, color=INK,
                            fontweight="bold")
                pos += 0.8
            xs.append((pos - 0.8 * len(vals) + (0.8 * (len(vals) - 1)) / 2,
                       label))
            pos += 0.55
        ax.axhline(50, color=CRIT, linewidth=1.6, linestyle=(0, (4, 3)),
                   zorder=2)
        if pi == 0:
            ax.text(0.97, 0.475, "coin flip", ha="right", va="bottom",
                    fontsize=10.5, color=CRIT, transform=ax.transAxes)
        ax.set_title(title, fontsize=14, color=INK, fontweight="bold",
                     pad=10)
        ax.set_xticks([x for x, _ in xs])
        ax.set_xticklabels([l for _, l in xs], fontsize=11.5, color=INK)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"],
                           fontsize=10.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.text(0.5, -0.16, sub, transform=ax.transAxes, ha="center",
                fontsize=10.5, color=MUTED)

    headline(fig,
             "One default argument made RL look broken",
             "Reward-model accuracy at ranking (ideal, exploit) response pairs. The only change: truncate training text from the left, not the right.")
    footer(fig,
           "sgb004_exploit_resistance_holdout_v2.json (pre-fix, root_cause_fixed) · sgb005b_rm_scores{,_hodge}.json (post-fix, computed from per-pair scores) · "
           "github.com/MikeHLee/shape_of_good_behavior")
    fig.savefig(os.path.join(OUT, "fig1_before_after.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
# The truncation mechanism.
def fig2_truncation(tok_stats):
    mean_tok, med_tok, max_tok = tok_stats
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    headline(fig,
             "The answer was truncated away before the model ever saw it",
             f"TRACE contexts average ~{mean_tok:,} tokens (median {med_tok}, max {max_tok:,}); the differing answer sits at the end. The window holds 512.")

    def seq(y, label, ans_text, ans_color):
        # shared context bar
        ax.add_patch(Rectangle((0.8, y), 7.2, 0.62, facecolor="#e8e6df",
                               edgecolor=INK2, linewidth=1.2, zorder=2))
        ax.text(4.4, y + 0.31, f"shared context (~{mean_tok:,} tokens — identical in both)",
                ha="center", va="center", fontsize=11, color=INK2, zorder=3)
        ax.add_patch(Rectangle((8.0, y), 2.1, 0.62, facecolor=ans_color,
                               edgecolor=SURFACE, linewidth=2, zorder=2))
        ax.text(9.05, y + 0.31, ans_text, ha="center", va="center",
                fontsize=11, color="#ffffff", fontweight="bold", zorder=3)
        ax.text(0.65, y + 0.31, label, ha="right", va="center", fontsize=12,
                color=INK, fontweight="bold")

    seq(4.9, "“chosen”", "ideal answer", BLUE)
    seq(4.0, "“rejected”", "exploit answer", ORANGE)

    # Right-truncation window (the bug)
    ax.add_patch(Rectangle((0.8, 3.78), 3.6, 1.96, facecolor="none",
                           edgecolor=CRIT, linewidth=2.6, zorder=4))
    ax.text(2.6, 6.0, "default: right-truncation keeps the FIRST 512 tokens",
            ha="center", va="bottom", fontsize=12, color=CRIT,
            fontweight="bold")
    ax.text(2.6, 3.55, "window sees only shared context →\nchosen and rejected are byte-identical (49/50 pairs)",
            ha="center", va="top", fontsize=11, color=CRIT)

    # Left-truncation window (the fix)
    seq(1.6, "“chosen”", "ideal answer", BLUE)
    seq(0.7, "“rejected”", "exploit answer", ORANGE)
    ax.add_patch(Rectangle((6.5, 0.48), 3.6, 1.96, facecolor="none",
                           edgecolor=GOOD, linewidth=2.6, zorder=4))
    ax.text(8.3, 2.68, "the fix: left-truncation keeps the LAST tokens — the answer survives",
            ha="center", va="bottom", fontsize=12, color=GOOD,
            fontweight="bold")

    footer(fig,
           "Token stats from sgb004_exploit_resistance_holdout_v2.json → root_cause_fixed · fix: tokenizer.truncation_side='left', window 512→1024 · "
           "github.com/MikeHLee/shape_of_good_behavior")
    fig.savefig(os.path.join(OUT, "fig2_truncation.png"),
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pre, post, tok_stats = load()
    fig1_before_after(pre, post)
    fig2_truncation(tok_stats)
    print(f"wrote 2 figures to {OUT}")
