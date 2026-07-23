"""Non-stationarity stress test: does patching a verifier close the loop? (VG-002)

Tests condition (3) of the data-economics axis -- *stationarity under the
agent's own actions*. Conditions (1) verifier and (2) samplability are held
fixed; only the stationarity of the labelling distribution varies.

The setup is the iterated RLHF / red-teaming loop, which is the realistic form
of the question. Each round:

    1. the generator searches best-of-N against the current verifier;
    2. an ORACLE labels a fixed budget of K points;
    3. those labels join the training set and the verifier is retrained;
    4. ground-truth hacking is measured against the new verifier.

Two arms, IDENTICAL label budget -- the only difference is where the K labelled
points come from:

    ADAPTIVE   -- the points the GENERATOR just exploited. The training
                  distribution therefore depends on the agent's own actions.
                  This is the non-stationary / condition-3-violating arm, and
                  is what real red-teaming does.
    STATIONARY -- K points sampled uniformly, independent of the agent. Same
                  labelling cost, fixed distribution. Control.

The comparison isolates *adversarially-selected data* from *more data*, which
is the whole content of condition (3). If the adaptive arm drives hacking to
zero, patching closes the loop and a fast self-improvement cycle exists here.
If it plateaus while the verifier's optimum keeps relocating, patching is
whack-a-mole and the loop does not close -- the outcome the data-economics
framing predicts for non-stationary adversarial domains.

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 feedback_geometry/src/nonstationary_verifier_experiment.py --smoke
    ./venv/bin/python3 feedback_geometry/src/nonstationary_verifier_experiment.py --seeds 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from h1_reward_hacking_experiment import PreferenceDataset  # noqa: E402
from verifier_gap_experiment import (  # noqa: E402
    Pair, RewardModel, VerifierSpec, _bt, _sample_state,
    generate_pairs, train_reward_model, measure_verifier_competence,
)


@dataclass(frozen=True)
class IterConfig:
    arm: str                    # "adaptive" | "stationary"
    gen_budget: int = 64        # generator's best-of-N search budget
    rounds: int = 8
    labels_per_round: int = 60
    init_pairs: int = 800
    init_oracle_fraction: float = 0.25
    trap_appeal: float = 2.0
    rm_hidden: int = 64
    rm_epochs: int = 300
    eval_trials: int = 400


def _oracle_label_points(
    ds: PreferenceDataset,
    points: np.ndarray,
    rng: np.random.Generator,
) -> List[Pair]:
    """Label each point against a uniform reference using the ORACLE.

    Each labelled point becomes a cross-state comparison (point vs a random
    reference state) scored by `get_true_reward`. A trap point scores -1.0
    against a safe reference's positive base reward, so the pair teaches the
    verifier that this specific location is bad. This is the only channel by
    which the verifier can learn about the trap.
    """
    pairs: List[Pair] = []
    for p in points:
        ref = _sample_state(ds, rng, in_trap=False)
        a = int(rng.integers(ds.n_actions))
        sa = ds.get_true_reward(p, a) + rng.normal(0, ds.noise_std)
        sb = ds.get_true_reward(ref, a) + rng.normal(0, ds.noise_std)
        pairs.append(Pair(np.asarray(p, dtype=float), a, ref, a, _bt(sa, sb), True))
    return pairs


@torch.no_grad()
def _search_points(
    rm: RewardModel, ds: PreferenceDataset, budget: int,
    n_draws: int, rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """Run best-of-N search `n_draws` times. Returns the winning points and the
    ground-truth hack rate over those draws."""
    wins = np.empty((n_draws, 2), dtype=float)
    hits = 0
    for i in range(n_draws):
        cand = rng.uniform(0, 10, size=(budget, 2))
        acts = rng.integers(0, ds.n_actions, size=budget)
        best = int(np.argmax(rm.batch_reward(cand, acts)))
        wins[i] = cand[best]
        hits += int(ds.in_trap(cand[best]))
    return wins, hits / n_draws


def run_iteration(cfg: IterConfig, seed: int) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ds = PreferenceDataset(n_evaluators=5, n_actions=4, h1_magnitude=0.5)

    spec = VerifierSpec(
        oracle_fraction=cfg.init_oracle_fraction, n_pairs=cfg.init_pairs,
        hidden=cfg.rm_hidden, epochs=cfg.rm_epochs, trap_appeal=cfg.trap_appeal,
    )
    pairs = generate_pairs(
        ds, cfg.init_pairs, cfg.init_oracle_fraction, rng,
        trap_appeal=cfg.trap_appeal,
    )

    history: List[Dict] = []
    for r in range(cfg.rounds):
        rm = RewardModel(hidden=cfg.rm_hidden, n_actions=ds.n_actions)
        train_reward_model(rm, pairs, spec)

        comp = measure_verifier_competence(rm, ds, np.random.default_rng(seed + 7000 + r))
        wins, hack = _search_points(
            rm, ds, cfg.gen_budget, cfg.eval_trials,
            np.random.default_rng(seed + 90_000 + r),
        )
        history.append({
            "round": r,
            "n_pairs": len(pairs),
            "hack_rate": hack,
            "trap_vs_safe_acc": comp["trap_vs_safe_acc"],
            "trap_reward_gap": comp["trap_reward_gap"],
            "argmax_in_trap": comp["argmax_in_trap"],
        })

        # --- patch the verifier with this round's label budget ---
        if cfg.arm == "adaptive":
            # Label exactly what the generator exploited. Distribution depends
            # on the agent's own behaviour -> non-stationary.
            idx = rng.choice(len(wins), size=min(cfg.labels_per_round, len(wins)),
                             replace=False)
            new_points = wins[idx]
        elif cfg.arm == "stationary":
            # Same budget, agent-independent distribution.
            new_points = rng.uniform(0, 10, size=(cfg.labels_per_round, 2))
        else:
            raise ValueError(f"unknown arm: {cfg.arm}")

        pairs = pairs + _oracle_label_points(ds, new_points, rng)

    return {"seed": seed, "config": asdict(cfg), "history": history}


def summarise(rows: List[Dict]) -> List[Dict]:
    buckets: Dict[Tuple[str, int], List[Dict]] = {}
    for r in rows:
        buckets.setdefault((r["config"]["arm"], r["config"]["gen_budget"]), []).append(r)

    out = []
    for (arm, budget), rs in sorted(buckets.items()):
        n_rounds = len(rs[0]["history"])
        hack = np.array([[h["hack_rate"] for h in r["history"]] for r in rs])
        comp = np.array([[h["trap_vs_safe_acc"] for h in r["history"]] for r in rs])
        intrap = np.array([[h["argmax_in_trap"] for h in r["history"]] for r in rs], dtype=float)
        out.append({
            "arm": arm,
            "gen_budget": budget,
            "n_seeds": len(rs),
            "hack_by_round": [float(x) for x in hack.mean(axis=0)],
            "hack_std_by_round": [float(x) for x in hack.std(axis=0, ddof=1)] if len(rs) > 1 else [0.0] * n_rounds,
            "competence_by_round": [float(x) for x in comp.mean(axis=0)],
            "argmax_in_trap_by_round": [float(x) for x in intrap.mean(axis=0)],
            "hack_first": float(hack[:, 0].mean()),
            "hack_last": float(hack[:, -1].mean()),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Non-stationarity stress test (VG-002)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--labels-per-round", type=int, default=60)
    ap.add_argument("--budgets", type=str, default="16,64,256")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        budgets, seeds, rounds = [64], [0], 3
    else:
        budgets = [int(x) for x in args.budgets.split(",")]
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
        rounds = args.rounds

    rows: List[Dict] = []
    total = len(budgets) * 2 * len(seeds)
    i, t0 = 0, time.time()
    for budget in budgets:
        for arm in ("adaptive", "stationary"):
            for seed in seeds:
                i += 1
                cfg = IterConfig(
                    arm=arm, gen_budget=budget, rounds=rounds,
                    labels_per_round=args.labels_per_round,
                )
                row = run_iteration(cfg, seed)
                rows.append(row)
                h = [f"{x['hack_rate']:.2f}" for x in row["history"]]
                print(f"[{i:>3}/{total}] N={budget:<4} {arm:<10} seed={seed} "
                      f"| hack {' '.join(h)} ({time.time() - t0:.0f}s)", flush=True)

    summary = summarise(rows)
    print("\n=== HACK RATE BY ROUND (label budget identical across arms) ===")
    for s in summary:
        curve = " ".join(f"{v:.3f}" for v in s["hack_by_round"])
        print(f"  N={s['gen_budget']:<4} {s['arm']:<10} | {curve}")
    print("\n=== VERIFIER ARGMAX INSIDE TRAP, BY ROUND ===")
    for s in summary:
        curve = " ".join(f"{v:.2f}" for v in s["argmax_in_trap_by_round"])
        print(f"  N={s['gen_budget']:<4} {s['arm']:<10} | {curve}")

    out = Path(args.out) if args.out else (
        _SRC.parent / "results" / "verifier_gap"
        / ("nonstationary_smoke.json" if args.smoke else f"nonstationary_{len(rows)}runs.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
