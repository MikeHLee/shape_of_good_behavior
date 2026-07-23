"""Figures for the verifier-generator gap result (SGB-032/033/035/036/041).

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
# Three disjoint 50-seed blocks for the scaling law (fig3): in-sample, OOS, and
# the dense 13-point grid. Confirmed-regime slope reproduces across all three.
OOS = RESULTS / "mechanism_oos_seed200.json"
DENSE = RESULTS / "scaling_dense_13pt_50seed.json"
SCALING_BLOCKS = [
    (MECH, "seeds 100–149", "#a33b3b"),
    (OOS, "seeds 200–249 (OOS)", "#c9832b"),
    (DENSE, "seeds 500–549 (dense)", "#2f6f9f"),
]

# Turn-over = the curve genuinely rises then falls. This MUST match the
# `noise_turn` definition in analyze_verifier_gap.py that the paper's verdict and
# crosstab rest on -- NOT the looser index/TOL `interior_peak` column. A qualifying
# earlier budget must beat BOTH the final and the base budget by > 3 combined
# binomial standard errors (trials/cell = TRIALS), i.e. beyond sampling noise.
# The looser TOL=0.02 definition over-counts turnovers (it inflated the argmax-
# outside turnover to 91.7% and Fisher p to 3.96e-42; noise-aware gives 75.7% /
# 6.7e-27). See queue SGB-035 CORRECTION 2026-07-23.
TRIALS = 800

C_IN = "#a33b3b"    # argmax inside trap  -> saturating search converges on it
C_OUT = "#2f6f9f"   # argmax outside trap -> saturating search escapes
C_ACC = "#2c3e50"


def _noise_turn(curve: np.ndarray) -> bool:
    """Noise-aware turn-over, identical to analyze_verifier_gap.py: some earlier
    budget beats BOTH the final and the base budget by > 3 combined binomial SEs."""
    p_hat = np.clip(curve, 0.0, 1.0)
    se = np.sqrt(p_hat * (1.0 - p_hat) / TRIALS)
    fin, base = curve[-1], curve[0]
    tf = 3.0 * np.sqrt(se[:-1] ** 2 + se[-1] ** 2)
    tb = 3.0 * np.sqrt(se[:-1] ** 2 + se[0] ** 2)
    return bool((((curve[:-1] - fin) > tf) & ((curve[:-1] - base) > tb)).any())


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
            "turns": _noise_turn(curve),
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

def regime_points(path):
    """Per-oracle_fraction (competence, log2 median N*, confirmed?) for one seed
    block. 'confirmed' uses the pre-specified rule: noise-aware turn-over fraction
    >= 50% AND median N* > 1 (so N* is a real interior peak, not a floored min)."""
    _, budgets, rows = load(path)
    by = {}
    for r in rows:
        by.setdefault(r["orc"], []).append(r)
    pts = []
    for orc in sorted(by):
        rs = by[orc]
        comp = float(np.mean([r["acc"] for r in rs]))
        med_ns = float(np.median([r["nstar"] for r in rs]))
        turnf = float(np.mean([r["turns"] for r in rs]))
        confirmed = (turnf >= 0.5) and (med_ns > 1)
        pts.append((comp, np.log2(med_ns), confirmed, orc))
    return pts


def fig3_scaling(out):
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    slopes = []
    for path, label, col in SCALING_BLOCKS:
        pts = regime_points(path)
        cx = np.array([p[0] for p in pts if p[2]])
        cy = np.array([p[1] for p in pts if p[2]])
        nx = np.array([p[0] for p in pts if not p[2]])
        ny = np.array([p[1] for p in pts if not p[2]])
        r = stats.linregress(cx, cy)
        slopes.append(r.slope)
        ax.scatter(cx, cy, s=55, color=col, zorder=4, edgecolor="white", lw=0.5,
                   label=f"{label}: slope {r.slope:.1f}, $r^2$={r.rvalue**2:.2f} ({len(cx)} pts)")
        if len(nx):
            ax.scatter(nx, ny, s=42, facecolor="none", edgecolor=col, lw=1.3,
                       alpha=0.55, zorder=3)
        gx = np.linspace(cx.min(), cx.max(), 40)
        ax.plot(gx, r.intercept + r.slope * gx, color=col, lw=1.8, alpha=0.9, zorder=2)

    ax.scatter([], [], s=42, facecolor="none", edgecolor="#888", lw=1.3,
               label="hollow = outside inverted-U regime (no $N^*$)")
    ax.text(0.02, 0.03,
            f"confirmed-regime slope $\\approx$ {np.mean(slopes):.0f} across 3 disjoint blocks\n"
            "(the retracted 6-pt $-14.7$ kept an out-of-regime point)",
            transform=ax.transAxes, fontsize=8, color="#555", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#ccc", lw=0.6))

    ax.set_xlabel("verifier competence (trap-vs-safe accuracy, measured against oracle)", fontsize=9.5)
    ax.set_ylabel(r"$\log_2 N^*$  (peak-danger search budget)", fontsize=10)
    ax.set_title("Improving the verifier moves the risk to a smaller generator",
                 fontsize=11.5, pad=8, color=C_ACC)
    ax.legend(fontsize=8.0, frameon=False, loc="upper right")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return slopes


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    _, budgets, rows = load(MECH)

    p = fig1_mechanism(rows, budgets, HERE / "vg_fig1_mechanism")
    fig2_gap_not_height(rows, budgets, HERE / "vg_fig2_gap_not_height")
    slopes = fig3_scaling(HERE / "vg_fig3_scaling_law")

    print(f"wrote vg_fig1_mechanism      (Fisher p = {p:.3e})")
    print("wrote vg_fig2_gap_not_height")
    print("wrote vg_fig3_scaling_law    (confirmed-regime slopes across 3 blocks: "
          + ", ".join(f"{s:.1f}" for s in slopes) + ")")


if __name__ == "__main__":
    main()
