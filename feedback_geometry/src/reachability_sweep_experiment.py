"""Reachability sweep (VG-004 / SGB-039): does samplability unify search and RL?

The search arm (VG-001) and the PPO arm (VG-003) disagree qualitatively at
intermediate verifier competence. Search shows an inverted U (danger peaks at
moderate budget, falls at large N). PPO shows bimodal categorical outcomes
(safe or catastrophic, path-dependent, no turn-over). SGB-035 explained this
by the global-vs-local optimum distinction, but that framing left them as two
separate stories.

Hypothesis: they may be the same story, one axis apart. Search draws candidates
GLOBALLY and INDEPENDENTLY; PPO reaches states LOCALLY, sequentially, along
correlated trajectories. Those are two ends of a *samplability* axis, which is
condition (2) of the data-economics framing and the one currently untested.

This experiment interpolates. For each (verifier, start-state) draw N
candidates in an L-infinity ball of radius k around the start, take
argmax-of-verifier, and check ground truth. At k >= diameter this recovers
global best-of-N (VG-001); as k shrinks, candidates become local and
correlated, and where you start matters more than how many you draw.

Prediction. If the search-vs-RL split is samplability in disguise, small-k
sweeps should show bimodal outcomes over start location: starts near the trap
get stuck; starts far from it don't reach it. Large-k sweeps should recover
the inverted U. If instead small-k is just a graded weaker version of large-k,
the two arms are genuinely different phenomena and this story does not fold.

Either outcome resolves the question directly.

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 feedback_geometry/src/reachability_sweep_experiment.py --smoke
    ./venv/bin/python3 feedback_geometry/src/reachability_sweep_experiment.py --seeds 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from h1_reward_hacking_experiment import PreferenceDataset  # noqa: E402
from verifier_gap_experiment import (  # noqa: E402
    RewardModel, VerifierSpec, generate_pairs, train_reward_model,
    measure_verifier_competence,
)


# State space is [0, 10]^2 (diameter 10 * sqrt(2) ~= 14.14 in L2, 10 in L-inf).
# k=INF is anything >= the diameter -- with clipping to [0,10]^2 this recovers
# uniform draws over the whole box.
STATE_DIAM = 10.0


@dataclass(frozen=True)
class Cell:
    oracle_fraction: float
    reach_k: float          # L-infinity ball radius; INF means global search
    budget: int             # best-of-N within the ball
    n_starts: int = 200     # start states per (verifier, cell)


def _reach_label(k: float) -> str:
    return "INF" if k >= STATE_DIAM else f"{k:.2f}"


@torch.no_grad()
def measure_reach_hack(
    rm: RewardModel,
    ds: PreferenceDataset,
    cell: Cell,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Best-of-N search within an L-inf ball of radius k around each start.

    Records the hack rate as ground-truth trap occupancy of the argmax winner,
    the mean start-to-trap-boundary distance, and the correlation between the
    two (a positive correlation would say local search only exploits when it
    starts nearby -- the mechanism the samplability story predicts).
    """
    trap_c = ds.trap_center
    trap_r = ds.trap_radius

    # Uniform starts over the box. For very small k these dominate the outcome.
    starts = rng.uniform(0, STATE_DIAM, size=(cell.n_starts, 2))
    # Signed distance to trap boundary at the start (negative = inside).
    d_start = np.linalg.norm(starts - trap_c, axis=1) - trap_r

    hacks = np.zeros(cell.n_starts, dtype=int)
    k = min(cell.reach_k, STATE_DIAM)
    for i in range(cell.n_starts):
        lo = np.maximum(starts[i] - k, 0.0)
        hi = np.minimum(starts[i] + k, STATE_DIAM)
        cand = rng.uniform(lo, hi, size=(cell.budget, 2))
        acts = rng.integers(0, ds.n_actions, size=cell.budget)
        best = int(np.argmax(rm.batch_reward(cand, acts)))
        hacks[i] = int(ds.in_trap(cand[best]))

    return {
        "hack_rate": float(hacks.mean()),
        "start_dist_to_trap_mean": float(d_start.mean()),
        # Fraction of hacked trials whose START was already at or inside the trap.
        # If small-k search hacks *only* when it started near the trap, this is
        # ~1.0; if it hacks even from far starts, this is close to the
        # unconditional trap-area fraction (~0.126).
        "hack_from_near_start": (
            float(np.mean(d_start[hacks == 1] < 0)) if hacks.any() else 0.0
        ),
    }


def run_verifier(oracle_fraction: float, seed: int) -> Tuple[RewardModel, PreferenceDataset, Dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ds = PreferenceDataset(n_evaluators=5, n_actions=4, h1_magnitude=0.5)
    spec = VerifierSpec(oracle_fraction=oracle_fraction, n_pairs=800,
                        hidden=64, epochs=300, trap_appeal=2.0)
    pairs = generate_pairs(ds, spec.n_pairs, oracle_fraction, rng,
                           trap_appeal=spec.trap_appeal)
    rm = RewardModel(hidden=spec.hidden, n_actions=ds.n_actions)
    train_reward_model(rm, pairs, spec)
    return rm, ds, measure_verifier_competence(rm, ds, rng)


def run_sweep(
    oracle_fractions: List[float],
    reach_ks: List[float],
    budgets: List[int],
    seeds: List[int],
    n_starts: int,
) -> List[Dict]:
    rows: List[Dict] = []
    total = len(oracle_fractions) * len(seeds)
    i, t0 = 0, time.time()
    for orc in oracle_fractions:
        for seed in seeds:
            i += 1
            rm, ds, comp = run_verifier(orc, seed)
            cells: Dict[str, Dict[str, float]] = {}
            for k in reach_ks:
                for N in budgets:
                    cell = Cell(orc, k, N, n_starts)
                    key = f"k={_reach_label(k)}|N={N}"
                    cells[key] = measure_reach_hack(
                        rm, ds, cell, np.random.default_rng(seed + hash(key) % 10_000),
                    )
            rows.append({
                "seed": seed, "oracle_fraction": orc,
                "competence": comp, "cells": cells,
            })
            print(f"[{i:>3}/{total}] orc={orc:.3f} seed={seed} | "
                  f"trap_v_safe={comp['trap_vs_safe_acc']:.3f} "
                  f"argmax_in_trap={int(comp['argmax_in_trap'])} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return rows


def summarise(rows: List[Dict], reach_ks: List[float], budgets: List[int]) -> Dict:
    per_orc: Dict[float, List[Dict]] = {}
    for r in rows:
        per_orc.setdefault(r["oracle_fraction"], []).append(r)

    summary = []
    for orc in sorted(per_orc):
        rs = per_orc[orc]
        entry: Dict = {"oracle_fraction": orc, "n_seeds": len(rs), "grid": {}}
        for k in reach_ks:
            for N in budgets:
                key = f"k={_reach_label(k)}|N={N}"
                hacks = np.array([r["cells"][key]["hack_rate"] for r in rs])
                near = np.array([r["cells"][key]["hack_from_near_start"] for r in rs])
                entry["grid"][key] = {
                    "hack_rate_mean": float(hacks.mean()),
                    "hack_rate_std": float(hacks.std(ddof=1)) if len(rs) > 1 else 0.0,
                    "hack_from_near_start_mean": float(near.mean()),
                }
        summary.append(entry)
    return {"budgets": budgets, "reach_ks": [_reach_label(k) for k in reach_ks],
            "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser(description="Reachability sweep")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=300)   # disjoint from earlier
    ap.add_argument("--oracle-fractions", type=str, default="0.30,0.375,0.45")
    ap.add_argument("--reach-ks", type=str, default="0.5,1,2,4,8,20")
    ap.add_argument("--budgets", type=str, default="1,4,16,64,256,1024")
    ap.add_argument("--n-starts", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        orcs = [0.375]
        ks = [1.0, 20.0]
        Ns = [16, 256]
        seeds = [0]
        starts = 100
    else:
        orcs = [float(x) for x in args.oracle_fractions.split(",")]
        ks = [float(x) for x in args.reach_ks.split(",")]
        Ns = [int(x) for x in args.budgets.split(",")]
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
        starts = args.n_starts

    rows = run_sweep(orcs, ks, Ns, seeds, starts)
    blob = summarise(rows, ks, Ns)

    print("\n=== HACK RATE by reachability k and search budget N ===")
    print("baseline at N=1, k=INF is the trap area share ~= 0.126")
    for entry in blob["summary"]:
        print(f"\n  orc={entry['oracle_fraction']:.3f}  (n={entry['n_seeds']}):")
        header = "  " + " ".join(f"{'N=' + str(N):>9}" for N in Ns)
        print(f"    {'k':>6} " + header)
        for k in ks:
            klab = _reach_label(k)
            cells = " ".join(
                f"{entry['grid'][f'k={klab}|N={N}']['hack_rate_mean']:>9.3f}"
                for N in Ns
            )
            print(f"    {klab:>6} " + cells)

    out = Path(args.out) if args.out else (
        _SRC.parent / "results" / "verifier_gap"
        / ("reachability_smoke.json" if args.smoke
           else f"reachability_{len(rows)}seedcells.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, **blob}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
