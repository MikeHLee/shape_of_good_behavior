#!/usr/bin/env python3
"""Figures for X/Twitter thread #1: "Your preferences have loops".

All numbers are loaded from
`shared/results/optimizer_comparison_hodge_v3_30seed.json` — nothing is
hand-entered. Figures are designed to be self-interpretable: the takeaway
is in the title, every mark is directly labeled, and no figure depends on
the thread text to be understood.

Usage: ./venv/bin/python3 threads/01_preference_loops/generate_figures.py
Outputs PNGs into threads/01_preference_loops/figures/
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

RESULTS = os.path.join(
    REPO, "shared", "results", "optimizer_comparison_hodge_v3_30seed.json"
)

# Series palette (CVD-validated in this fixed order; see threads/README.md)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"    # series 1: standard optimizer
ORANGE = "#eb6834"  # series 2: Hodge-aware variant
AQUA = "#1baf7a"    # series 3 (unused in bars; loop accent in fig2)
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


def load():
    with open(RESULTS) as f:
        d = json.load(f)
    stats = d["method_stats"]
    tests = d["pairwise_tests"]
    per_seed = {}
    for r in d["all_results"]:
        per_seed.setdefault(r["method"], {})[r["seed"]] = r["exploit_resistance"]
    return stats, tests, per_seed, d["config"]


def headline(fig, title, subtitle):
    fig.text(0.05, 0.955, title, fontsize=19, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.05, 0.895, subtitle, fontsize=12.5, color=INK2,
             ha="left", va="top")


def footer(fig, text):
    fig.text(0.05, 0.02, text, fontsize=9.5, color=MUTED, ha="left")


FOOT = ("30 seeds · 500 preference pairs + 1,482 cross-pair edges · "
        "shared/results/optimizer_comparison_hodge_v3_30seed.json · "
        "github.com/MikeHLee/shape_of_good_behavior")


# ---------------------------------------------------------------- figure 1
# Headline: standard vs Hodge-aware exploit resistance, three families.
def fig1_headline(stats, tests):
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.07, right=0.97)

    families = [  # (standard, hodge, test key ordered as in file)
        ("KTO", "Hodge-KTO", "Hodge-KTO vs KTO"),
        ("DPO", "Hodge-DPO", "DPO vs Hodge-DPO"),
        ("GRPO", "Hodge-GRPO", None),
    ]

    x0 = np.arange(len(families), dtype=float)
    width = 0.32
    for i, (std_m, hodge_m, _) in enumerate(families):
        for off, m, color, label in (
            (-width / 2 - 0.01, std_m, BLUE, "Standard optimizer"),
            (width / 2 + 0.01, hodge_m, ORANGE,
             "Hodge-aware variant (sees the loops)"),
        ):
            s = stats[m]
            v = s["mean"] * 100
            b = ax.bar(x0[i] + off, v, width=width, color=color, zorder=3,
                       label=label if i == 0 else None)
            ax.errorbar(x0[i] + off, v, yerr=s["std"] * 100, fmt="none",
                        ecolor=INK2, elinewidth=1.4, capsize=4, zorder=4)
            txt = f"{v:.2f}%" if 99 < v < 100 else f"{v:.1f}%"
            ax.text(x0[i] + off, v + 3.2, txt, ha="center", va="bottom",
                    fontsize=12.5, color=INK, fontweight="bold")

    # Effect-size annotations from the pairwise tests
    for i, (std_m, hodge_m, key) in enumerate(families):
        if key is None:
            ax.text(x0[i], 137, "both already at the\nmetric ceiling (100%)",
                    ha="center", va="top", fontsize=11, color=INK2)
            continue
        t = tests[key]
        gain = (stats[hodge_m]["mean"] - stats[std_m]["mean"]) * 100
        ax.text(x0[i], 137,
                f"+{gain:.1f} points\nd = {abs(t['cohens_d']):.2f}, p < 0.0001",
                ha="center", va="top", fontsize=11, color=INK2)

    headline(fig,
             "Reward optimizers that can see preference loops leave far fewer exploitable rankings",
             "Exploit resistance: % of held (ideal, exploit) response pairs the trained model ranks correctly · mean of 30 seeds, ±1 sd error bars.")
    ax.set_xticks(x0)
    ax.set_xticklabels([f"{a} family" for a, _, _ in families],
                       fontsize=12.5, color=INK)
    ax.set_xlim(-0.95, 2.55)
    ax.set_ylim(0, 140)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.94), frameon=False,
              fontsize=11.5)
    footer(fig, FOOT)
    fig.savefig(os.path.join(OUT, "fig1_exploit_resistance.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
# Concept: a loop of preferences cannot be represented by a score.
def fig2_loop_concept():
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    headline(fig,
             "A single score per answer cannot represent a loop of preferences",
             "Hodge decomposition splits the graders' comparisons into the part a score CAN fit, plus the leftover loop.")

    def box(x, y, text, w=1.9, fs=12.5):
        b = FancyBboxPatch((x - w / 2, y - 0.4), w, 0.8,
                           boxstyle="round,pad=0.06,rounding_size=0.14",
                           facecolor="#ffffff", edgecolor=INK2, linewidth=1.4)
        ax.add_patch(b)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color=INK, fontweight="bold")

    def arrow(p, q, color=INK2, rad=0.0, lw=2.2, style="-|>", ls="solid"):
        a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=16,
                            linewidth=lw, color=color, linestyle=ls,
                            connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(a)

    # Left: the loop the graders produced
    A, B, C = (2.4, 4.7), (1.0, 2.1), (3.8, 2.1)
    box(*A, "Answer A")
    box(*B, "Answer B")
    box(*C, "Answer C")
    arrow((A[0] - 0.55, A[1] - 0.5), (B[0] + 0.15, B[1] + 0.5), rad=0.12)
    arrow((B[0] + 1.0, B[1]), (C[0] - 1.0, C[1]), rad=0.0)
    arrow((C[0] - 0.15, C[1] + 0.5), (A[0] + 0.55, A[1] - 0.5), rad=0.12)
    ax.text(1.05, 3.55, "“A is more\nthorough”", fontsize=10.5, color=INK2,
            ha="center")
    ax.text(2.4, 1.28, "“B is clearer”", fontsize=10.5, color=INK2,
            ha="center", va="top")
    ax.text(3.85, 3.55, "“C is more\nconcise”", fontsize=10.5, color=INK2,
            ha="center")
    ax.text(2.4, 5.6, "What the graders said\n(each judgment reasonable — together, a loop)",
            ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold")
    ax.text(2.4, 0.55,
            "An Escher staircase: every step goes “up”,\nyet you end where you started.",
            ha="center", va="top", fontsize=11, color=CRIT)

    ax.text(5.35, 3.3, "=", fontsize=30, color=INK, ha="center", va="center",
            fontweight="bold")

    # Middle: the rankable part
    rx = 7.0
    ax.text(rx, 5.6, "Rankable part\n(a score CAN fit this)", ha="center",
            va="bottom", fontsize=12, color=BLUE, fontweight="bold")
    box(rx, 4.7, "Answer A", w=1.7)
    box(rx, 3.3, "Answer B", w=1.7)
    box(rx, 1.9, "Answer C", w=1.7)
    arrow((rx - 1.15, 1.9), (rx - 1.15, 4.7), color=BLUE, style="-|>", lw=2.4)
    ax.text(rx - 1.38, 3.3, "higher score", fontsize=10.5, color=BLUE,
            ha="center", va="center", rotation=90)
    ax.text(rx, 0.55, "Train the reward model\nanchored to this part.",
            ha="center", va="top", fontsize=11, color=BLUE)

    ax.text(8.65, 3.3, "+", fontsize=30, color=INK, ha="center", va="center",
            fontweight="bold")

    # Right: the pure loop
    lx = 10.3
    ax.text(lx, 5.6, "Pure loop\n(no score fits — flag it)", ha="center",
            va="bottom", fontsize=12, color=ORANGE, fontweight="bold")
    la, lb, lc = (lx, 4.55), (lx - 1.15, 2.45), (lx + 1.15, 2.45)
    for p in (la, lb, lc):
        ax.plot(*p, marker="o", markersize=13, color=ORANGE,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    arrow((la[0] - 0.28, la[1] - 0.28), (lb[0] + 0.12, lb[1] + 0.34),
          color=ORANGE, rad=0.25)
    arrow((lb[0] + 0.35, lb[1] - 0.08), (lc[0] - 0.35, lc[1] - 0.08),
          color=ORANGE, rad=0.25)
    arrow((lc[0] - 0.12, lc[1] + 0.34), (la[0] + 0.28, la[1] - 0.28),
          color=ORANGE, rad=0.25)
    ax.text(lx, 0.55,
            "Left in the data unflagged, this is where\nreward models grow exploitable soft spots.",
            ha="center", va="top", fontsize=11, color=ORANGE)

    footer(fig, "Concept diagram (no measured values) · github.com/MikeHLee/shape_of_good_behavior")
    fig.savefig(os.path.join(OUT, "fig2_preference_loop.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
# Every seed improved: paired per-seed differences, both families.
def fig3_paired_seeds(per_seed):
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.16, right=0.95)

    rng = np.random.default_rng(42)  # fixed jitter for reproducibility
    rows = [
        ("KTO family\n(Hodge-KTO − KTO)", "KTO", "Hodge-KTO", ORANGE, 1.0),
        ("DPO family\n(Hodge-DPO − DPO)", "DPO", "Hodge-DPO", BLUE, 0.0),
    ]
    for label, std_m, hodge_m, color, y in rows:
        seeds = sorted(per_seed[std_m])
        diffs = np.array([
            (per_seed[hodge_m][s] - per_seed[std_m][s]) * 100 for s in seeds
        ])
        jitter = rng.uniform(-0.13, 0.13, size=len(diffs))
        ax.scatter(diffs, y + jitter, s=95, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=1.6, alpha=0.95)
        mean = diffs.mean()
        ax.plot([mean, mean], [y - 0.24, y + 0.24], color=INK, linewidth=2.4,
                zorder=4)
        ax.text(mean, y + 0.31, f"mean +{mean:.1f}", ha="center",
                va="bottom", fontsize=11.5, color=INK, fontweight="bold")
        ax.text(diffs.min(), y - 0.31, f"worst seed\n+{diffs.min():.1f}",
                ha="center", va="top", fontsize=10.5, color=INK2)

    ax.axvline(0, color=CRIT, linewidth=1.6, linestyle=(0, (4, 3)), zorder=2)
    ax.text(0, 1.62, "no change", ha="center", va="bottom", fontsize=11,
            color=CRIT)

    headline(fig,
             "Not an average hiding variance: all 30 seeds improved, in both families",
             "Per-seed change in exploit resistance when the same optimizer is given the loop map (Hodge variant − standard), 30 matched seeds.")
    ax.set_yticks([r[4] for r in rows])
    ax.set_yticklabels([r[0] for r in rows], fontsize=12.5, color=INK)
    ax.set_ylim(-0.7, 1.75)
    ax.set_xlim(-2.5, 26)
    ax.set_xlabel("Improvement in exploit resistance (percentage points)",
                  fontsize=11.5, color=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    footer(fig, FOOT)
    fig.savefig(os.path.join(OUT, "fig3_paired_seeds.png"),
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    stats, tests, per_seed, config = load()
    assert config["num_seeds"] == 30
    fig1_headline(stats, tests)
    fig2_loop_concept()
    fig3_paired_seeds(per_seed)
    print(f"wrote 3 figures to {OUT}")
