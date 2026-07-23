"""Mechanism crosstab: does the verifier's global argmax predict curve shape?

Produces the numbers in VERIFIER_GAP_WRITEUP.md §2 for one or more run files:
  - saturated hack rate (N_max) split by argmax-in-trap  [threshold-free]
  - turn-over crosstab (argmax-in-trap x turns-over) + Fisher exact p

The turn-over flag uses the NOISE-AWARE definition (identical to
analyze_verifier_gap.py): a qualifying earlier budget must beat BOTH the final
and the base budget by > 3 combined binomial standard errors. We also print the
looser INDEX/TOL definition for disclosure -- it over-counts turn-overs and is
the source of the retracted 91.7% / Fisher 3.96e-42 headline. Report the
noise-aware numbers.

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 feedback_geometry/src/mechanism_crosstab.py \
        feedback_geometry/results/verifier_gap/mechanism_argmax_50seed.json \
        feedback_geometry/results/verifier_gap/mechanism_oos_seed200.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_verifier_gap import per_seed_curves  # noqa: E402

import json  # noqa: E402

TRIALS = 800
TOL = 0.02


def noise_turn(curves: np.ndarray) -> np.ndarray:
    p = np.clip(curves, 0.0, 1.0)
    se = np.sqrt(p * (1.0 - p) / TRIALS)
    tf = 3.0 * np.sqrt(se[:, :-1] ** 2 + se[:, [-1]] ** 2)
    tb = 3.0 * np.sqrt(se[:, :-1] ** 2 + se[:, [0]] ** 2)
    return (((curves[:, :-1] - curves[:, [-1]]) > tf)
            & ((curves[:, :-1] - curves[:, [0]]) > tb)).any(axis=1)


def interior_turn(runs, budgets) -> np.ndarray:
    return np.array([
        (r["peak_budget"] > budgets[0])
        and (r["peak_value"] > r["final_value"] + TOL)
        and (r["peak_value"] > r["base_value"] + TOL)
        for r in runs
    ])


def _crosstab(inside: np.ndarray, turns: np.ndarray, finals: np.ndarray, tag: str):
    a = int((inside & turns).sum()); b = int((inside & ~turns).sum())
    c = int((~inside & turns).sum()); d = int((~inside & ~turns).sum())
    n_in, n_out = int(inside.sum()), int((~inside).sum())
    _, pf = stats.fisher_exact([[a, b], [c, d]])
    print(f"    [{tag:>9}] argmax-in turnover {a}/{n_in}={a/n_in:.1%}   "
          f"argmax-out turnover {c}/{n_out}={c/n_out:.1%}   Fisher p={pf:.2e}")


def analyse(path: str) -> None:
    blob = json.loads(Path(path).read_text())
    budgets = blob["budgets"]
    by_orc = per_seed_curves(blob["rows"], budgets)
    runs = [r for rs in by_orc.values() for r in rs if r["argmax_in_trap"] is not None]
    curves = np.vstack([r["curve"] for r in runs])
    inside = np.array([bool(r["argmax_in_trap"]) for r in runs])
    finals = np.array([r["final_value"] for r in runs])

    inn, out = finals[inside], finals[~inside]
    mw = stats.mannwhitneyu(inn, out, alternative="greater").pvalue
    print(f"\n=== {Path(path).name}  (n_in={inside.sum()}, n_out={(~inside).sum()}) ===")
    print(f"  saturated hack rate: argmax-in {inn.mean():.3f}  vs  argmax-out {out.mean():.3f}"
          f"   Mann-Whitney p={mw:.2e}   [threshold-free -- report this]")
    _crosstab(inside, noise_turn(curves), finals, "noise")
    _crosstab(inside, interior_turn(runs, budgets), finals, "interior")
    print("    (report 'noise'; 'interior' over-counts -- the retracted 91.7% / 3.96e-42)")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "feedback_geometry/results/verifier_gap/mechanism_argmax_50seed.json",
        "feedback_geometry/results/verifier_gap/mechanism_oos_seed200.json",
    ]
    for p in paths:
        analyse(p)
