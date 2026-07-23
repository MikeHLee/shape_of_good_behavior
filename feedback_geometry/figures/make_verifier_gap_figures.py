"""Figures for the verifier-generator gap result (SGB-032/033/035).

Every plotted number is loaded from a committed results JSON — nothing is
hardcoded. This is deliberate: the 2026-07-21 audit found hand-typed figure
data in two independent tracks (EXPERIMENT_ISSUES.md §7, and the peer-consistency
small-panel constants). Do not reintroduce the pattern here.

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 feedback_geometry/figures/make_verifier_gap_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "verifier_gap"

# The 50-seed run carries `argmax_in_trap`; the 15-seed sweep is the disjoint-seed
# replication partner (seeds 0-14 vs 100-149).
MECH = RESULTS / "mechanism_argmax_50seed.json"
SWEEP15 = RESULTS / "sweep_105cells.json"

# "Interior peak" = the curve genuinely turns over. Must match the definition in
# analyze_verifier_gap.py: argmax INDEX alone is wrong, because np.argmax breaks
# ties by first index and a saturated curve (..., 1.0, 1.0) then reports a
# spurious turn-over.
TOL = 0.02

C_IN = "#a33b3b"    # argmax inside trap  -> saturating search converges on it
C_OUT = "#2f6f9f"   # argmax outside trap -> saturating search escapes
C_ACC = "#2c3e50"


def load(path: Path):
    blob = json.loads(path.read_text())
    budgets = blob["budgets"]
    rows = []
    for r in blob["rows"]:
        curve = np.array([r["search"][str(b)] for b in budgets], float)
        i = int(curve.argmax())
        rows.append({
            "seed": r["seed"],
            "orc": r["verifier"]["oracle_fraction"],
            "acc": r["competence"]["trap_vs_safe_acc"],
            "curve": curve,
            "nstar": budgets[i],
            "peak": float(curve[i]),
            "final": float(curve[-1]),
            "base": float(curve[0]),
            "in_trap": r["competence"].get("argmax_in_trap"),
            "turns": (budgets[i] > budgets[0])
                     and (curve[i] > curve[-1] + TOL)
                     and (curve[i] > curve[0] + TOL),
        })
    return blob, budgets, rows


def fit_scaling(rows, orcs):
    """Median N* per oracle_fraction vs mean measured competence."""
    xs = [float(np.mean([r["acc"] for r in rows if r["orc"] == o])) for o in orcs]
    ys = [float(np.log2(np.median([r["nstar"] for r in rows if r["orc"] == o]))) for o in orcs]
    return np.array(xs), np.array(ys), stats.linregress(xs, ys)


# ---------------------------------------------------------------------------
# Figure 1 — the mechanism: two populations, not one curve
# ---------------------------------------------------------------------------

def fig1_mechanism(rows, budgets, out):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.5),
                                   gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.28})
    x = np.arange(len(budgets))

    inn = [r for r in rows if r["in_trap"]]
    out_ = [r for r in rows if not r["in_trap"]]

    for group, c in ((out_, C_OUT), (inn, C_IN)):
        for r in group:
            axL.plot(x, r["curve"], color=c, alpha=0.10, lw=0.8, zorder=1)
    for group, c, lab in ((out_, C_OUT, f"argmax OUTSIDE trap (n={len(out_)})"),
                          (inn, C_IN, f"argmax INSIDE trap (n={len(inn)})")):
        m = np.mean([r["curve"] for r in group], axis=0)
        axL.plot(x, m, color=c, lw=2.6, label=lab, zorder=3)

    axL.set_xticks(x)
    axL.set_xticklabels([str(b) for b in budgets], fontsize=8.5)
    axL.set_xlabel("generator search budget $N$ (best-of-$N$)", fontsize=10)
    axL.set_ylabel("hack rate (ground truth)", fontsize=10)
    axL.set_ylim(-0.02, 1.02)
    axL.set_title("One binary variable splits every seed into two populations",
                  fontsize=11, pad=8, color=C_ACC)
    axL.legend(loc="upper left", fontsize=9, frameon=False)

    # Right: the crosstab that the left panel is a picture of.
    ti = 100 * np.mean([r["turns"] for r in inn])
    to = 100 * np.mean([r["turns"] for r in out_])
    fi = 100 * np.mean([r["final"] for r in inn])
    fo = 100 * np.mean([r["final"] for r in out_])
    tbl = np.array([[ti, to], [fi, fo]])

    xb = np.arange(2)
    w = 0.36
    axR.bar(xb - w/2, tbl[:, 0], w, color=C_IN, edgecolor="#5c1f1f", label="argmax INSIDE")
    axR.bar(xb + w/2, tbl[:, 1], w, color=C_OUT, edgecolor="#1a4260", label="argmax OUTSIDE")
    for xx, vals in zip(xb, tbl):
        for dx, v in zip((-w/2, w/2), vals):
            axR.text(xx + dx, v + 2.5, f"{v:.1f}%", ha="center", va="bottom",
                     fontsize=9, weight="bold", color=C_ACC)
    axR.set_xticks(xb)
    axR.set_xticklabels(["curve turns over\n(inverted U)", "final hack rate\nat $N=1024$"], fontsize=9.5)
    axR.set_ylim(0, 118)
    axR.set_ylabel("%", fontsize=10)
    axR.set_title("Near-deterministic", fontsize=11, pad=8, color=C_ACC)
    axR.legend(loc="upper center", fontsize=8.5, frameon=False, ncol=1)

    _, p = stats.fisher_exact([
        [sum(r["turns"] for r in inn), len(inn) - sum(r["turns"] for r in inn)],
        [sum(r["turns"] for r in out_), len(out_) - sum(r["turns"] for r in out_)],
    ])
    axR.text(0.5, 0.60, f"Fisher exact\np = {p:.2e}", transform=axR.transAxes,
             ha="center", va="top", fontsize=8.5, color="#5c1f1f",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f7eded", ec="#c7a0a0", lw=0.6))

    fig.suptitle("The inverted U is a mixture: it is what you see when the verifier's global optimum is safe",
                 fontsize=12, y=1.03, color=C_ACC)
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figure 2 — gap, not height
# ---------------------------------------------------------------------------

def fig2_gap_not_height(rows, budgets, out):
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    x = np.arange(len(budgets))
    orcs = sorted({r["orc"] for r in rows})
    cmap = plt.get_cmap("viridis")

    for i, o in enumerate(orcs):
        g = [r for r in rows if r["orc"] == o]
        acc = np.mean([r["acc"] for r in g])
        m = np.mean([r["curve"] for r in g], axis=0)
        ax.plot(x, m, color=cmap(i / max(1, len(orcs) - 1)), lw=2.0,
                marker="o", ms=3.5, label=f"{acc:.3f}")

    base = np.mean([r["base"] for r in rows])
    ax.axhline(base, color="#888888", ls="--", lw=1.0, zorder=0)
    ax.text(len(budgets) - 1, base + 0.022, f"no-selection baseline ({base:.3f})",
            fontsize=8, color="#666666", ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets], fontsize=8.5)
    ax.set_xlabel("generator search budget $N$ (best-of-$N$)", fontsize=10)
    ax.set_ylabel("hack rate (ground truth)", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("The same verifier is safe or exploited depending only on $N$",
                 fontsize=11.5, pad=8, color=C_ACC)
    leg = ax.legend(title="verifier competence\n(trap-vs-safe acc)", fontsize=8.5,
                    title_fontsize=8.5, loc="upper left", frameon=False, ncol=2)
    leg._legend_box.align = "left"
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — peak-danger budget vs competence, both fits shown honestly
# ---------------------------------------------------------------------------

def fig3_scaling(rows, out):
    orcs = sorted({r["orc"] for r in rows})
    xs, ys, full = fit_scaling(rows, orcs)
    keep = [o for o in orcs if o != min(orcs)]
    xk, yk, sub = fit_scaling(rows, keep)

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    dropped = 0  # min(orcs) is the excluded row; it is first in xs/ys

    ax.scatter(xs[1:], ys[1:], s=62, color=C_ACC, zorder=4, label="verifier (7 total)")
    ax.scatter(xs[:1], ys[:1], s=110, facecolor="none", edgecolor=C_IN, lw=2.0,
               zorder=5, label=f"orc={min(orcs)} — excluded from the 6-point fit")

    gx = np.linspace(min(xs) - 0.01, max(xs) + 0.01, 50)
    ax.plot(gx, full.intercept + full.slope * gx, color=C_ACC, lw=2.2,
            label=(f"all 7 pts: slope {full.slope:.1f}, "
                   f"$r^2$={full.rvalue**2:.3f}, p={full.pvalue:.1e}"))
    ax.plot(gx, sub.intercept + sub.slope * gx, color=C_IN, lw=1.6, ls="--",
            label=(f"6 pts: slope {sub.slope:.1f}, "
                   f"$r^2$={sub.rvalue**2:.3f}, p={sub.pvalue:.1e}"))

    # White bbox so the label stays legible where a fit line passes under it.
    for xx, yy in zip(xs, ys):
        ax.annotate(f"$N^*$={2**yy:.0f}", (xx, yy), textcoords="offset points",
                    xytext=(8, 7), fontsize=8, color="#555555",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    ax.set_xlabel("verifier competence (trap-vs-safe accuracy, measured against oracle)", fontsize=9.5)
    ax.set_ylabel(r"$\log_2 N^*$  (peak-danger search budget)", fontsize=10)
    ax.set_title("Improving the verifier moves the risk to a smaller generator",
                 fontsize=11.5, pad=8, color=C_ACC)
    ax.legend(fontsize=8.2, frameon=False, loc="upper right")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return full, sub


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    _, budgets, rows = load(MECH)

    p = fig1_mechanism(rows, budgets, HERE / "vg_fig1_mechanism")
    fig2_gap_not_height(rows, budgets, HERE / "vg_fig2_gap_not_height")
    full, sub = fig3_scaling(rows, HERE / "vg_fig3_scaling_law")

    print(f"wrote vg_fig1_mechanism      (Fisher p = {p:.3e})")
    print("wrote vg_fig2_gap_not_height")
    print(f"wrote vg_fig3_scaling_law    (7pt slope {full.slope:.2f} r2={full.rvalue**2:.3f} "
          f"p={full.pvalue:.2e} | 6pt slope {sub.slope:.2f} r2={sub.rvalue**2:.3f} p={sub.pvalue:.2e})")


if __name__ == "__main__":
    main()
