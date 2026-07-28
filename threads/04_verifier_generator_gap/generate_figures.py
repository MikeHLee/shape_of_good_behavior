#!/usr/bin/env python3
"""Figure for X/Twitter thread #4: "When the judge can't keep up with the
contestant" (SCAFFOLD).

fig1 renders from
`feedback_geometry/results/verifier_gap/mechanism_argmax_50seed.json`
(SGB-035 summary block, weakest-judge cell) — nothing hand-entered.

fig2 (the hook figure: saturated inside/outside-trap split) is specified in
thread.md's pre-posting checklist and not yet built — it needs the per-seed
argmax-inside-trap classification from the `rows` block plus the SGB-036
out-of-sample file.

Usage: ./venv/bin/python3 threads/04_verifier_generator_gap/generate_figures.py
Outputs PNGs into threads/04_verifier_generator_gap/figures/
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "feedback_geometry", "results", "verifier_gap",
                   "mechanism_argmax_50seed.json")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

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


def fig1():
    with open(SRC) as f:
        d = json.load(f)
    budgets = d["budgets"]
    # weakest judge in the sweep = lowest oracle fraction
    cell = min(d["summary"], key=lambda s: s["oracle_fraction"])
    means = np.array([cell["search_hack_rate"][str(b)]["mean"]
                      for b in budgets])
    stds = np.array([cell["search_hack_rate"][str(b)]["std"]
                     for b in budgets])

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.08, right=0.96)

    x = np.log2(budgets)
    ax.fill_between(x, means - stds, means + stds, color=BLUE, alpha=0.16,
                    zorder=2, linewidth=0)
    ax.plot(x, means, color=BLUE, linewidth=2.4, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3)

    peak = int(np.argmax(means))
    for i in (0, peak, len(budgets) - 1):
        ax.text(x[i], means[i] + 0.05, f"{means[i]:.2f}", ha="center",
                va="bottom", fontsize=12, color=INK, fontweight="bold")

    ax.annotate(
        f"variance explodes: ±1 sd = {stds[-1]:.2f}\n"
        "the 50 seeds fork into hack≈1 and hack≈0 populations\n"
        "(the inside/outside-trap split — next figure)",
        xy=(x[-1], means[-1] - stds[-1]), xytext=(x[-1] - 2.6, 0.13),
        fontsize=11.5, color=INK2, ha="center",
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))

    fig.text(0.05, 0.955,
             "More search first inflates hacking — then forks the population",
             fontsize=19, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.05, 0.895,
             f"Mean hack rate vs best-of-N search budget, weakest judge (oracle fraction {cell['oracle_fraction']}) · "
             f"{cell['n_seeds']} seeds, band = ±1 sd.",
             fontsize=12.5, color=INK2, ha="left", va="top")

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets], fontsize=10.5)
    ax.set_xlabel("search budget N (best-of-N, log scale)", fontsize=11.5,
                  color=INK2)
    ax.set_ylabel("hack rate", fontsize=11.5, color=INK2)
    ax.set_ylim(-0.05, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.05, 0.02,
             "SGB-035 · feedback_geometry/results/verifier_gap/mechanism_argmax_50seed.json · "
             "github.com/MikeHLee/shape_of_good_behavior",
             fontsize=9.5, color=MUTED, ha="left")
    fig.savefig(os.path.join(OUT, "fig1_inverted_u.png"),
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    print(f"wrote 1 figure to {OUT} (fig2 not yet built — see thread.md)")
