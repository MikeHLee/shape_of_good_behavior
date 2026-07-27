"""Figure for the SGB-005 open-model results writeup.

One figure, two panels: exploit resistance by method, at two scales.
  Left  — embedding-level optimizer comparison (30 seeds, frozen embeddings,
          `shared/results/optimizer_comparison_hodge_v3_30seed.json`).
  Right — 1.5B LM-level exploit resistance after the SGB-005c RM fix
          (n=51 true holdout, `shared/results/finetune/sgb004_exploit_resistance_holdout_v2.json`).

Nothing is hardcoded except axis cosmetics; every bar height and error bar is
read from committed JSON. See feedback_geometry/figures/make_verifier_gap_figures.py
for why this convention exists (EXPERIMENT_ISSUES.md §7 hand-typed-figure-data audit).

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 feedback_geometry/figures/make_sgb005_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SHARED_RESULTS = HERE.parent.parent / "shared" / "results"

EMBED_JSON = SHARED_RESULTS / "optimizer_comparison_hodge_v3_30seed.json"
LM_JSON = SHARED_RESULTS / "finetune" / "sgb004_exploit_resistance_holdout_v2.json"

C_BASE = "#2f6f9f"   # non-Hodge / baseline
C_HODGE = "#a33b3b"  # Hodge variant
C_NEUTRAL = "#7f8c8d"  # base / SFT (no optimizer being compared yet)

EMBED_ORDER = ["ORPO", "KTO", "DPO", "GRPO", "Hodge-KTO", "Hodge-DPO", "Hodge-GRPO"]
LM_ORDER = ["base", "sft", "ppo", "hodge_ppo"]
LM_LABELS = {"base": "base", "sft": "SFT", "ppo": "PPO", "hodge_ppo": "Hodge-PPO"}


def _color_for(name: str) -> str:
    if name.startswith("Hodge") or name == "hodge_ppo":
        return C_HODGE
    if name in ("base", "sft"):
        return C_NEUTRAL
    return C_BASE


def main() -> None:
    embed = json.loads(EMBED_JSON.read_text())["method_stats"]
    lm = json.loads(LM_JSON.read_text())["results"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), width_ratios=[7, 4])

    # --- Panel A: embedding-level (30 seeds) ---
    means = [embed[m]["mean"] for m in EMBED_ORDER]
    ci95 = [embed[m]["ci95"] for m in EMBED_ORDER]
    colors = [_color_for(m) for m in EMBED_ORDER]
    x = np.arange(len(EMBED_ORDER))
    ax1.bar(x, means, yerr=ci95, capsize=3, color=colors, edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels(EMBED_ORDER, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("exploit resistance")
    ax1.set_ylim(0, 1.08)
    ax1.set_title(f"Embedding-level (n={embed[EMBED_ORDER[0]]['n']} seeds, frozen embeddings)", fontsize=10)
    ax1.axhline(1.0, color="#bbbbbb", lw=0.8, ls="--", zorder=0)

    # --- Panel B: 1.5B LM-level (n=51 holdout) ---
    lm_means = [lm[m]["exploit_resistance"] for m in LM_ORDER]
    lm_n = lm[LM_ORDER[0]]["n"]
    colors2 = [_color_for(m) for m in LM_ORDER]
    x2 = np.arange(len(LM_ORDER))
    ax2.bar(x2, lm_means, color=colors2, edgecolor="none")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([LM_LABELS[m] for m in LM_ORDER], rotation=30, ha="right", fontsize=9)
    ax2.set_ylim(0, 1.08)
    ax2.set_title(f"1.5B LM-level (n={lm_n} true holdout, generates text)", fontsize=10)
    ax2.axhline(1.0, color="#bbbbbb", lw=0.8, ls="--", zorder=0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_NEUTRAL, label="base / SFT"),
        plt.Rectangle((0, 0), 1, 1, color=C_BASE, label="standard optimizer"),
        plt.Rectangle((0, 0), 1, 1, color=C_HODGE, label="Hodge variant"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04), fontsize=9)
    fig.suptitle("Exploit resistance by method, across scale", y=1.12, fontsize=12)
    fig.tight_layout()

    out_png = HERE / "sgb005_fig1_method_by_scale.png"
    out_pdf = HERE / "sgb005_fig1_method_by_scale.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
