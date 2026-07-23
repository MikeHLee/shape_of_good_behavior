"""Scaling-law robustness: log2(median N*) vs measured verifier competence.

Produces the numbers in VERIFIER_GAP_WRITEUP.md §4:
  - confirmed-regime slope for each seed block (should reproduce at ~ -19)
  - leave-one-out slope range (the honest robustness metric)

Inclusion rule (PRE-SPECIFIED, applied SYMMETRICALLY to both competence ends):
an oracle_fraction point enters the fit iff its noise-aware turn-over fraction
>= 50% AND its median N* > 1 -- i.e. the inverted U is confirmed, so N* is a
real interior peak rather than a floored minimum budget. Points failing this
(low-competence: search hasn't separated the populations; high-competence: curve
is monotone) have no well-defined N* and are excluded.

This is why the earlier "6-point -14.7" fit is retracted: it dropped the low
endpoint (orc 0.25) but kept an out-of-regime high point (orc 0.50), which is
asymmetric. Under this rule the slope is ~ -19 on every block.

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 feedback_geometry/src/scaling_robustness.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parent.parent / "results" / "verifier_gap"
TRIALS = 800

BLOCKS = [
    ("100-149 (in-sample)", RESULTS / "mechanism_argmax_50seed.json"),
    ("200-249 (OOS)", RESULTS / "mechanism_oos_seed200.json"),
    ("500-549 (dense 13-pt)", RESULTS / "scaling_dense_13pt_50seed.json"),
]


def _noise_turn(curve: np.ndarray) -> bool:
    p = np.clip(curve, 0, 1)
    se = np.sqrt(p * (1 - p) / TRIALS)
    tf = 3 * np.sqrt(se[:-1] ** 2 + se[-1] ** 2)
    tb = 3 * np.sqrt(se[:-1] ** 2 + se[0] ** 2)
    return bool((((curve[:-1] - curve[-1]) > tf) & ((curve[:-1] - curve[0]) > tb)).any())


def regime_points(path: Path):
    blob = json.loads(Path(path).read_text())
    budgets = blob["budgets"]
    by = defaultdict(list)
    for r in blob["rows"]:
        curve = np.array([r["search"][str(b)] for b in budgets], float)
        by[r["verifier"]["oracle_fraction"]].append(
            (budgets[int(curve.argmax())], r["competence"]["trap_vs_safe_acc"], _noise_turn(curve)))
    pts = []
    for orc in sorted(by):
        rs = by[orc]
        med_ns = float(np.median([r[0] for r in rs]))
        comp = float(np.mean([r[1] for r in rs]))
        turnf = float(np.mean([r[2] for r in rs]))
        pts.append((orc, comp, med_ns, turnf, (turnf >= 0.5 and med_ns > 1)))
    return pts


def lin(x, y):
    r = stats.linregress(x, y)
    return r.slope, r.rvalue ** 2, r.pvalue


def main() -> None:
    print("Confirmed-regime scaling slope across disjoint seed blocks")
    print("(inclusion: noise-aware turnover >=50% AND median N* > 1)\n")
    print(f"{'block':>24} {'confirmed pts':>14} {'slope':>8} {'r2':>6}")
    for label, path in BLOCKS:
        pts = regime_points(path)
        x = np.array([p[1] for p in pts if p[4]])
        y = np.array([np.log2(p[2]) for p in pts if p[4]])
        s, r2, _ = lin(x, y)
        print(f"{label:>24} {len(x):>14} {s:>8.2f} {r2:>6.2f}")
        loo = np.array([lin(np.delete(x, i), np.delete(y, i))[0] for i in range(len(x))])
        swing = (loo.max() - loo.min()) / abs(s)
        tag = "  (dense)" if "dense" in label else ""
        print(f"{'  leave-one-out':>24}  [{loo.min():.2f}, {loo.max():.2f}]  "
              f"swing {swing:.0%}{tag}")
    print("\n=> slope ~ -19, halving per ~+0.05 competence. Densifying 5->8 pts shrinks the"
          "\n   leave-one-out swing (37-51% -> 25%) but leaves a +-12% floor (endpoint leverage"
          "\n   in a bounded window). The retracted 6-pt -14.7 kept an out-of-regime point (orc 0.50).")


if __name__ == "__main__":
    main()
