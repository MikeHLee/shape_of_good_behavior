"""Higher-dimensional multi-trap reachability sweep (VG-005 / SGB-040).

The reachability finding (SGB-039) collapses search-vs-RL into one axis --
samplability. This asks whether it survives more realistic geometry: higher
dimensional state spaces and multiple non-overlapping traps.

Design keeps the same knobs as VG-004 (oracle_fraction, reachability radius k,
search budget N) and adds two: state dimension d, and number of traps K.
Traps are AXIS-ALIGNED HYPERCUBES rather than L2 balls -- the reachability
metric is L-inf, and matching them keeps the geometry clean. Per-trap volume
is fixed at ~13% of the state space divided by K, so the N=1 baseline hack
rate stays comparable across (d, K); high-d curse-of-dimensionality effects
should therefore appear as changes in the reachability crossover, not as
baseline drift.

Two claims are on the line:
  1. Continuous degradation of the inverted U as k shrinks (SGB-039's headline)
     survives higher d and multiple traps.
  2. The crossover k where the inverted U dies scales with dimension in an
     interpretable way -- most likely, "local reachability" translates to
     "volume covered by a k-ball around a start", which shrinks (2k/10)^d.
     If so, effective k in absolute units grows with d to preserve the effect,
     which is a specific quantitative prediction, not a hand-wave.

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 feedback_geometry/src/multi_trap_reachability_experiment.py --smoke
    ./venv/bin/python3 feedback_geometry/src/multi_trap_reachability_experiment.py --seeds 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from verifier_gap_experiment import (  # noqa: E402
    RewardModel, VerifierSpec, train_reward_model, Pair,
)

STATE_LO = 0.0
STATE_HI = 10.0
BOX_SIDE = STATE_HI - STATE_LO


class MultiTrapDataset:
    """d-dimensional preference dataset with K disjoint hypercube traps.

    Interface intentionally mirrors PreferenceDataset -- get_true_reward,
    in_trap, base_rewards, evaluator_biases, n_actions, noise_std -- so the
    same reward-model training and label functions work without changes.
    """

    def __init__(
        self,
        dim: int,
        n_traps: int,
        target_total_trap_vol: float = 0.13,
        n_evaluators: int = 5,
        n_actions: int = 4,
        h1_magnitude: float = 0.5,
        noise_std: float = 0.1,
        seed: int = 0,
    ):
        self.dim = dim
        self.n_traps = n_traps
        self.n_evaluators = n_evaluators
        self.n_actions = n_actions
        self.noise_std = noise_std
        self.base_rewards = np.array([1.0, 0.75, 0.5, 0.25][:n_actions])

        self.evaluator_biases = np.zeros((n_evaluators, n_actions))
        for i in range(n_evaluators):
            self.evaluator_biases[i, i % n_actions] = h1_magnitude

        # Half-side chosen so K traps together cover `target_total_trap_vol`.
        # Curse of dimensionality: small-volume cubes have LARGE side length in
        # high d (volume V has side V^(1/d), which goes to 1 as d rises), so
        # K > 1 traps often cannot fit non-overlapping. Shrink adaptively when
        # that happens and record the achieved volume so the change is honest.
        target_vol = target_total_trap_vol
        placed: List[np.ndarray] = []
        rng = np.random.default_rng(seed)
        for shrink_step in range(20):
            per_trap_vol = target_vol / n_traps
            hs = (BOX_SIDE / 2.0) * (per_trap_vol ** (1.0 / dim))
            centers: List[np.ndarray] = []
            for _ in range(20_000):
                if len(centers) == n_traps:
                    break
                c = rng.uniform(STATE_LO + hs, STATE_HI - hs, size=dim)
                if all(np.max(np.abs(c - o)) >= 2 * hs for o in centers):
                    centers.append(c)
            if len(centers) == n_traps:
                placed = centers
                self.trap_half_side = hs
                self.achieved_total_vol = target_vol
                break
            target_vol *= 0.5   # halve target and retry
        if not placed:
            raise RuntimeError(
                f"Could not fit {n_traps} traps in d={dim} even after shrinking"
            )
        centers = placed
        self.trap_centers = np.stack(centers)

        # Reference "trap_center" for utilities that expect a single centre.
        self.trap_center = self.trap_centers[0]
        # Half-side reported as "radius" so external code that reads
        # ds.trap_radius gets a sensible answer under L-inf geometry.
        self.trap_radius = self.trap_half_side

    def in_trap(self, state: np.ndarray) -> bool:
        s = np.asarray(state)
        for c in self.trap_centers:
            if np.max(np.abs(s - c)) < self.trap_half_side:
                return True
        return False

    def in_trap_batch(self, states: np.ndarray) -> np.ndarray:
        """Vectorised in_trap over (n, d)."""
        diff = np.abs(states[:, None, :] - self.trap_centers[None, :, :])
        in_any = (diff.max(axis=-1) < self.trap_half_side).any(axis=-1)
        return in_any

    def get_true_reward(self, state: np.ndarray, action: int) -> float:
        return -1.0 if self.in_trap(state) else float(self.base_rewards[action])


def _sample_state(ds: MultiTrapDataset, rng: np.random.Generator, in_trap: bool) -> np.ndarray:
    """Sample a state inside or outside any trap.

    Rejection sampling breaks in high d once total trap volume is tiny (curse
    of dimensionality after shrinking). Instead: for `in_trap=True` sample
    directly from a randomly-chosen trap cube -- O(1) regardless of volume.
    For `in_trap=False` still reject, since safe volume is >> trap volume so
    hits are almost immediate.
    """
    if in_trap:
        t = int(rng.integers(ds.n_traps))
        c = ds.trap_centers[t]
        return rng.uniform(c - ds.trap_half_side, c + ds.trap_half_side, size=ds.dim)
    for _ in range(2000):
        s = rng.uniform(STATE_LO, STATE_HI, size=ds.dim)
        if not ds.in_trap(s):
            return s
    raise RuntimeError(f"couldn't sample OUTSIDE trap in d={ds.dim}")


def _bt(a: float, b: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(a - b))))


def _label(
    ds: MultiTrapDataset, rng: np.random.Generator,
    oracle_fraction: float,
    sa: np.ndarray, aa: int, sb: np.ndarray, ab: int,
    trap_appeal: float,
) -> float:
    if rng.random() < oracle_fraction:
        score_a = ds.get_true_reward(sa, aa)
        score_b = ds.get_true_reward(sb, ab)
    else:
        ev = int(rng.integers(ds.n_evaluators))
        score_a = ds.base_rewards[aa]
        score_b = ds.base_rewards[ab]
        if ds.in_trap(sa):
            score_a += ds.evaluator_biases[ev, aa] + trap_appeal
        if ds.in_trap(sb):
            score_b += ds.evaluator_biases[ev, ab] + trap_appeal
    score_a += rng.normal(0, ds.noise_std)
    score_b += rng.normal(0, ds.noise_std)
    return _bt(score_a, score_b)


def generate_pairs_multi(
    ds: MultiTrapDataset, n_pairs: int, oracle_fraction: float,
    rng: np.random.Generator, trap_appeal: float = 2.0,
    trap_fraction: float = 0.35, cross_state_fraction: float = 0.5,
) -> List[Pair]:
    pairs: List[Pair] = []
    for _ in range(n_pairs):
        cross = rng.random() < cross_state_fraction
        a_in_trap = rng.random() < trap_fraction
        if cross:
            action = int(rng.integers(ds.n_actions))
            sa = _sample_state(ds, rng, a_in_trap)
            sb = _sample_state(ds, rng, not a_in_trap)
            aa = ab = action
        else:
            sa = _sample_state(ds, rng, a_in_trap)
            sb = sa
            aa = int(rng.integers(ds.n_actions))
            ab = int(rng.integers(ds.n_actions))
            while ab == aa:
                ab = int(rng.integers(ds.n_actions))
        pref = _label(ds, rng, oracle_fraction, sa, aa, sb, ab, trap_appeal)
        pairs.append(Pair(sa, aa, sb, ab, pref, cross))
    return pairs


@torch.no_grad()
def measure_reach_hack_multi(
    rm: RewardModel, ds: MultiTrapDataset, k: float, budget: int,
    n_starts: int, rng: np.random.Generator,
) -> Dict[str, float]:
    starts = rng.uniform(STATE_LO, STATE_HI, size=(n_starts, ds.dim))
    starts_in = ds.in_trap_batch(starts)  # was the start already inside a trap?
    k_eff = min(k, BOX_SIDE)

    hacks = np.zeros(n_starts, dtype=int)
    for i in range(n_starts):
        lo = np.maximum(starts[i] - k_eff, STATE_LO)
        hi = np.minimum(starts[i] + k_eff, STATE_HI)
        cand = rng.uniform(lo, hi, size=(budget, ds.dim))
        acts = rng.integers(0, ds.n_actions, size=budget)
        best = int(np.argmax(rm.batch_reward(cand, acts)))
        hacks[i] = int(ds.in_trap(cand[best]))

    return {
        "hack_rate": float(hacks.mean()),
        "hack_from_near_start": (
            float(np.mean(starts_in[hacks == 1])) if hacks.any() else 0.0
        ),
    }


@torch.no_grad()
def measure_verifier_competence_multi(
    rm: RewardModel, ds: MultiTrapDataset, rng: np.random.Generator,
    n_probe: int = 2000,
) -> Dict[str, float]:
    trap = np.stack([_sample_state(ds, rng, True) for _ in range(n_probe)])
    safe = np.stack([_sample_state(ds, rng, False) for _ in range(n_probe)])
    acts = rng.integers(0, ds.n_actions, size=n_probe)
    r_t = rm.batch_reward(trap, acts)
    r_s = rm.batch_reward(safe, acts)
    return {
        "trap_vs_safe_acc": float(np.mean(r_s > r_t)),
        "trap_reward_gap": float(r_t.mean() - r_s.mean()),
    }


@dataclass(frozen=True)
class GridSpec:
    dim: int
    n_traps: int
    oracle_fraction: float


def run_cell(
    grid: GridSpec, seed: int, reach_ks: List[float], budgets: List[int],
    n_starts: int,
) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ds = MultiTrapDataset(dim=grid.dim, n_traps=grid.n_traps, seed=seed)
    spec = VerifierSpec(oracle_fraction=grid.oracle_fraction, n_pairs=1200,
                        hidden=64, epochs=300, trap_appeal=2.0)
    pairs = generate_pairs_multi(ds, spec.n_pairs, grid.oracle_fraction, rng)

    rm = RewardModel(state_dim=grid.dim, n_actions=ds.n_actions,
                     hidden=spec.hidden)
    train_reward_model(rm, pairs, spec)
    comp = measure_verifier_competence_multi(rm, ds, rng)

    cells: Dict[str, Dict[str, float]] = {}
    for k in reach_ks:
        for N in budgets:
            key = f"k={k:g}|N={N}"
            cells[key] = measure_reach_hack_multi(
                rm, ds, k, N, n_starts,
                np.random.default_rng(seed + hash(key) % 10_000),
            )
    return {
        "seed": seed, "dim": grid.dim, "n_traps": grid.n_traps,
        "oracle_fraction": grid.oracle_fraction,
        "trap_half_side": float(ds.trap_half_side),
        "achieved_total_vol": float(ds.achieved_total_vol),
        "competence": comp, "cells": cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-trap higher-D reachability sweep")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=400)
    ap.add_argument("--dims", type=str, default="2,5,10")
    ap.add_argument("--n-traps-list", type=str, default="1,3")
    ap.add_argument("--oracle-fractions", type=str, default="0.375")
    ap.add_argument("--reach-ks", type=str, default="0.5,1,2,4,8,20")
    ap.add_argument("--budgets", type=str, default="1,16,64,256,1024")
    ap.add_argument("--n-starts", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        dims = [2, 5]
        ntraps = [1, 3]
        orcs = [0.375]
        ks = [1.0, 20.0]
        Ns = [1, 256]
        seeds = [400]
        starts = 100
    else:
        dims = [int(x) for x in args.dims.split(",")]
        ntraps = [int(x) for x in args.n_traps_list.split(",")]
        orcs = [float(x) for x in args.oracle_fractions.split(",")]
        ks = [float(x) for x in args.reach_ks.split(",")]
        Ns = [int(x) for x in args.budgets.split(",")]
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
        starts = args.n_starts

    rows: List[Dict] = []
    total = len(dims) * len(ntraps) * len(orcs) * len(seeds)
    i, t0 = 0, time.time()
    for d in dims:
        for K in ntraps:
            for orc in orcs:
                grid = GridSpec(d, K, orc)
                for seed in seeds:
                    i += 1
                    try:
                        row = run_cell(grid, seed, ks, Ns, starts)
                    except RuntimeError as e:
                        print(f"[{i:>3}/{total}] d={d} K={K} orc={orc:.3f} "
                              f"seed={seed} SKIPPED: {e}", flush=True)
                        continue
                    rows.append(row)
                    print(f"[{i:>3}/{total}] d={d} K={K} orc={orc:.3f} "
                          f"seed={seed} | trap_v_safe="
                          f"{row['competence']['trap_vs_safe_acc']:.3f} "
                          f"h={row['trap_half_side']:.2f} "
                          f"({time.time() - t0:.0f}s)", flush=True)

    # Aggregate
    print("\n=== HACK RATE by dim x n_traps x k x N (orc-averaged if multiple) ===")
    print("baseline at N=1, k=INF ~ 0.13 (fixed by construction)")
    summary = []
    for d in dims:
        for K in ntraps:
            rs = [r for r in rows if r["dim"] == d and r["n_traps"] == K]
            if not rs:
                continue
            print(f"\n  d={d}, K={K}  (n={len(rs)}, "
                  f"trap half-side {rs[0]['trap_half_side']:.2f}, "
                  f"comp={np.mean([r['competence']['trap_vs_safe_acc'] for r in rs]):.3f})")
            hdr = "  " + " ".join(f"{'N=' + str(N):>9}" for N in Ns)
            print(f"    {'k':>6} " + hdr)
            entry_grid: Dict[str, Dict[str, float]] = {}
            for k in ks:
                cells = []
                near = []
                for N in Ns:
                    key = f"k={k:g}|N={N}"
                    v = np.array([r["cells"][key]["hack_rate"] for r in rs])
                    n = np.array([r["cells"][key]["hack_from_near_start"] for r in rs])
                    entry_grid[key] = {"hack_mean": float(v.mean()),
                                       "hack_std": float(v.std(ddof=1)) if len(rs) > 1 else 0.0,
                                       "near_mean": float(n.mean())}
                    cells.append(v.mean())
                print(f"    {k:>6g} " + " ".join(f"{c:>9.3f}" for c in cells))
            summary.append({"dim": d, "n_traps": K, "n_seeds": len(rs),
                            "grid": entry_grid})

    out = Path(args.out) if args.out else (
        _SRC.parent / "results" / "verifier_gap"
        / ("multitrap_smoke.json" if args.smoke
           else f"multitrap_{len(rows)}runs.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"rows": rows, "summary": summary,
         "budgets": Ns, "reach_ks": ks, "dims": dims, "n_traps_list": ntraps},
        indent=2,
    ))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
