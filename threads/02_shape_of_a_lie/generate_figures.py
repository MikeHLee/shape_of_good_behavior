#!/usr/bin/env python3
"""Figure for X/Twitter thread #2: "The shape of a lie".

All numbers are loaded from the SGB-021 7-9B panel result files
`shared/results/peer_sheaf_e6_modal_*.json` (length-matched splits) —
nothing is hand-entered.

Usage: ./venv/bin/python3 threads/02_shape_of_a_lie/generate_figures.py
Outputs PNGs into threads/02_shape_of_a_lie/figures/
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(REPO, "shared", "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"    # detects (above chance)
ORANGE = "#eb6834"  # inverted (below chance)
CRIT = "#d03b3b"

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

# (display label, result file, verdict) — verdict picks the color/group.
SCENARIOS = [
    ("Convincing-game\n(argue a user into a false belief)",
     "peer_sheaf_e6_modal_convincing-game.json", "detects"),
    ("Insider trading\n(conceal the real reason)",
     "peer_sheaf_e6_modal_insider-trading.json", "detects"),
    ("Alignment-faking reasoning\n(comply while privately disagreeing)",
     "peer_sheaf_e6_modal_alignment-faking-reasoning.json", "inverted"),
    ("… same, cues redacted",
     "peer_sheaf_e6_modal_alignment-faking-reasoning-redacted.json",
     "inverted"),
    ("Instructed lies\n(“state X”, X false)",
     "peer_sheaf_e6_modal_instructed-deception.json", "blind"),
    ("Alignment-faking outputs\n(the final answer, not the reasoning)",
     "peer_sheaf_e6_modal_alignment-faking-output.json", "blind"),
]

COLORS = {"detects": BLUE, "inverted": ORANGE, "blind": MUTED}
VERDICT_LABEL = {
    "detects": "detected (above chance)",
    "inverted": "inverted (below chance — sign flip)",
    "blind": "no usable signal (at or near chance)",
}


def fig1():
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.30, right=0.95)

    ys = range(len(SCENARIOS) - 1, -1, -1)
    seen = set()
    for y, (label, fname, verdict) in zip(ys, SCENARIOS):
        with open(os.path.join(RES, fname)) as f:
            d = json.load(f)
        lm = d["length_matched"]
        auc, std, p = lm["auc_mean"], lm["auc_std"], lm["p_seed0"]
        color = COLORS[verdict]
        leg = VERDICT_LABEL[verdict] if verdict not in seen else None
        seen.add(verdict)
        ax.errorbar(auc, y, xerr=std, fmt="o", markersize=11, color=color,
                    ecolor=color, elinewidth=2.2, capsize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3,
                    label=leg)
        ptxt = f"p = {p:.1e}" if p >= 1e-4 else "p < 1e-4"
        if verdict == "blind" and p > 0.05:
            ptxt += "  (not significant)"
        if auc >= 0.5:
            ax.text(auc + std + 0.018, y, f"{auc:.2f}   {ptxt}",
                    ha="left", va="center", fontsize=11.5, color=INK,
                    fontweight="bold")
        else:  # below-chance rows: label under the marker, clear of row names
            ax.text(auc, y - 0.32, f"{auc:.2f}   {ptxt}",
                    ha="center", va="top", fontsize=11.5, color=INK,
                    fontweight="bold")

    ax.axvline(0.5, color=CRIT, linewidth=1.6, linestyle=(0, (4, 3)),
               zorder=2)
    ax.text(0.5, len(SCENARIOS) - 0.25, "coin flip (0.5)", ha="center",
            va="bottom", fontsize=11, color=CRIT)

    headline_y = 0.955
    fig.text(0.05, headline_y,
             "The triangle test detects deception-as-strategy — and is blind to instructed lies",
             fontsize=19, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.05, 0.895,
             "AUC for separating deceptive from honest transcripts by cross-model translation inconsistency (‖δ¹c‖) · 7–9B open-model panel,\nlength-matched classes, ±1 sd across 5 splits.",
             fontsize=12.5, color=INK2, ha="left", va="top")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([s[0] for s in SCENARIOS], fontsize=11.5, color=INK)
    ax.set_xlim(0.28, 0.82)
    ax.set_ylim(-0.6, len(SCENARIOS) - 0.1)
    ax.set_xlabel("AUC (0.5 = cannot tell deceptive from honest)",
                  fontsize=11.5, color=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.text(0.05, 0.02,
             "n per class 262–901 (length-matched) · SGB-021 · shared/results/peer_sheaf_e6_modal_*.json · "
             "github.com/MikeHLee/shape_of_good_behavior",
             fontsize=9.5, color=MUTED, ha="left")
    fig.savefig(os.path.join(OUT, "fig1_auc_by_scenario.png"),
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    print(f"wrote 1 figure to {OUT}")
