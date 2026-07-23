"""PPO arm of the verifier-gap experiment (VG-003 / SGB-034).

Everything in the verifier-gap line so far rests on best-of-N SEARCH. This asks
whether the same structure appears under GRADIENT-BASED optimisation, which is
what actually trains models. It is a bounding experiment: if the answer is no,
the headline claim narrows from "optimisation pressure" to "sampling-based
search", which is a much weaker statement.

Sharp prediction from the SGB-035 mechanism. Search at saturating budget
approximates a GLOBAL argmax, which is why hacking collapses once the verifier's
global optimum leaves the trap (the inverted U). Gradient ascent is LOCAL: it
climbs whatever basin it starts in. So a PPO learner should get STUCK in the
local trap optimum that saturating search escapes, and the inverted U should
vanish or inverta -- hacking should rise with training and stay high, even for
verifiers where search gets safer with more compute.

Supersedes the REINFORCE arm in `verifier_gap_experiment.train_policy`, which
was too weak to reach the verifier's optimum at all (trap occupancy pinned near
0.025 vs a ~0.126 area baseline regardless of both axes). Fixes carried over:
fixed-length episodes and randomised starts. New here:
  - true PPO: GAE(lambda), clipped surrogate, several epochs of minibatch
    updates per rollout, entropy bonus to prevent premature collapse;
  - batched proxy-reward evaluation (the old per-step `get_reward` call in a
    Python loop dominated runtime);
  - hacking measured as ground-truth trap occupancy over TRAINING TIME, giving
    a curve directly comparable to the search-budget curve.

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 feedback_geometry/src/verifier_gap_ppo_arm.py --smoke
    ./venv/bin/python3 feedback_geometry/src/verifier_gap_ppo_arm.py --seeds 10
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
import torch.nn as nn
import torch.optim as optim

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from h1_reward_hacking_experiment import PreferenceDataset  # noqa: E402
from verifier_gap_experiment import (  # noqa: E402
    Actor, Critic, FixedLengthTrapEnv, RewardModel, VerifierSpec,
    generate_pairs, train_reward_model, measure_verifier_competence,
)


@dataclass(frozen=True)
class PPOConfig:
    hidden: int = 64
    rollout_episodes: int = 8
    updates: int = 200
    epochs_per_update: int = 4
    minibatch: int = 128
    clip: float = 0.2
    entropy_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3


def _action_idx_batch(actions: np.ndarray, n_actions: int) -> np.ndarray:
    """Vectorised version of the harness's continuous->discrete action map."""
    idx = (actions[:, 0] + 1.0) * n_actions / 2.0
    return np.clip(idx, 0, n_actions - 1).astype(np.int64)


@torch.no_grad()
def collect_rollout(
    env: FixedLengthTrapEnv,
    actor: Actor,
    rm: RewardModel,
    n_episodes: int,
    n_actions: int,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Run episodes, then score the whole batch with the verifier at once."""
    S, A, LP, DONE = [], [], [], []
    in_trap: List[int] = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs)
            dist = actor(obs_t)
            act = dist.sample()
            S.append(obs.copy())
            A.append(act.numpy())
            LP.append(float(dist.log_prob(act).sum()))
            obs, _true_r, done, info = env.step(act.numpy())
            in_trap.append(int(info["in_trap"]))
            DONE.append(float(done))

    states = np.array(S, dtype=np.float32)
    acts = np.array(A, dtype=np.float32)
    # Proxy reward from the VERIFIER, batched.
    proxy = rm.batch_reward(states, _action_idx_batch(acts, n_actions))

    batch = {
        "states": torch.FloatTensor(states),
        "actions": torch.FloatTensor(acts),
        "logp_old": torch.FloatTensor(np.array(LP, dtype=np.float32)),
        "rewards": torch.FloatTensor(proxy.astype(np.float32)),
        "dones": torch.FloatTensor(np.array(DONE, dtype=np.float32)),
    }
    # GROUND TRUTH, never shown to the policy.
    return batch, float(np.mean(in_trap))


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
    gamma: float, lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(rewards)
    adv = torch.zeros(n)
    last = 0.0
    for t in reversed(range(n)):
        nonterminal = 1.0 - dones[t]
        next_v = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def train_ppo(
    env: FixedLengthTrapEnv,
    rm: RewardModel,
    cfg: PPOConfig,
    n_actions: int,
    seed: int,
) -> Dict:
    torch.manual_seed(seed)
    actor = Actor(hidden=cfg.hidden)
    critic = Critic(hidden=cfg.hidden)
    opt_a = optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_c = optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    hack_curve: List[float] = []
    proxy_curve: List[float] = []

    for _ in range(cfg.updates):
        batch, hack = collect_rollout(env, actor, rm, cfg.rollout_episodes, n_actions)
        hack_curve.append(hack)
        proxy_curve.append(float(batch["rewards"].mean()))

        with torch.no_grad():
            values = critic(batch["states"]).squeeze(-1)
        adv, ret = compute_gae(
            batch["rewards"], values, batch["dones"], cfg.gamma, cfg.lam
        )
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(adv)
        for _ in range(cfg.epochs_per_update):
            perm = torch.randperm(n)
            for start in range(0, n, cfg.minibatch):
                mb = perm[start:start + cfg.minibatch]
                dist = actor(batch["states"][mb])
                logp = dist.log_prob(batch["actions"][mb]).sum(dim=-1)
                ratio = torch.exp(logp - batch["logp_old"][mb])
                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[mb]
                ent = dist.entropy().sum(dim=-1).mean()
                loss_a = -torch.min(s1, s2).mean() - cfg.entropy_coef * ent
                opt_a.zero_grad()
                loss_a.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                opt_a.step()

                v = critic(batch["states"][mb]).squeeze(-1)
                loss_c = nn.MSELoss()(v, ret[mb])
                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()

    q = max(1, len(hack_curve) // 10)
    return {
        "hack_curve": hack_curve,
        "proxy_curve": proxy_curve,
        "hack_start": float(np.mean(hack_curve[:q])),
        "hack_peak": float(np.max(hack_curve)),
        "hack_peak_update": int(np.argmax(hack_curve)),
        "hack_final": float(np.mean(hack_curve[-q:])),
        "proxy_start": float(np.mean(proxy_curve[:q])),
        "proxy_final": float(np.mean(proxy_curve[-q:])),
    }


def run_cell(
    oracle_fraction: float,
    cfg: PPOConfig,
    seed: int,
    policy_seed: Optional[int] = None,
) -> Dict:
    """Train a verifier from `seed`, then PPO against it from `policy_seed`.

    The two seeds are separable so that the verifier REALISATION (preference
    sampling + reward-model init) can be varied independently of the policy
    INIT and env start states. With them tied (policy_seed=None, the default)
    a trapped outcome cannot be attributed to either -- see SGB-038.
    """
    pseed = seed if policy_seed is None else policy_seed

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ds = PreferenceDataset(n_evaluators=5, n_actions=4, h1_magnitude=0.5)
    spec = VerifierSpec(oracle_fraction=oracle_fraction, n_pairs=800,
                        hidden=64, epochs=300, trap_appeal=2.0)
    pairs = generate_pairs(ds, spec.n_pairs, oracle_fraction, rng,
                           trap_appeal=spec.trap_appeal)
    rm = RewardModel(hidden=spec.hidden, n_actions=ds.n_actions)
    train_reward_model(rm, pairs, spec)
    comp = measure_verifier_competence(rm, ds, rng)

    env = FixedLengthTrapEnv(seed=pseed)
    res = train_ppo(env, rm, cfg, ds.n_actions, pseed)
    return {
        "seed": seed,
        "policy_seed": pseed,
        "oracle_fraction": oracle_fraction,
        "config": asdict(cfg),
        "competence": comp,
        "ppo": res,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="PPO arm of the verifier-gap experiment")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--oracle-fractions", type=str, default="0.0,0.25,0.375,0.5,1.0")
    ap.add_argument("--entropy-coefs", type=str, default=None,
                    help="comma-separated entropy bonuses to sweep, e.g. "
                         "'0,0.003,0.01,0.03,0.1'. Confound check: entropy is "
                         "what drives escape from local optima, so the trapped "
                         "fraction must be shown NOT to be an artifact of it.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    ent_coefs = ([float(x) for x in args.entropy_coefs.split(",")]
                 if args.entropy_coefs else [PPOConfig().entropy_coef])

    if args.smoke:
        orcs, seeds, updates = [0.0, 1.0], [0], 30
    else:
        orcs = [float(x) for x in args.oracle_fractions.split(",")]
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
        updates = args.updates

    rows: List[Dict] = []
    total = len(orcs) * len(seeds) * len(ent_coefs)
    i, t0 = 0, time.time()
    for ent in ent_coefs:
        cfg_base = PPOConfig(updates=updates, entropy_coef=ent)
        for orc in orcs:
            for seed in seeds:
                i += 1
                row = run_cell(orc, cfg_base, seed)
                rows.append(row)
                p = row["ppo"]
                print(f"[{i:>3}/{total}] ent={ent:<6} orc={orc:.3f} seed={seed} "
                      f"| trap_v_safe={row['competence']['trap_vs_safe_acc']:.3f} "
                      f"argmax_in_trap={int(row['competence']['argmax_in_trap'])} "
                      f"| hack {p['hack_start']:.3f} -> final {p['hack_final']:.3f} "
                      f"| proxy {p['proxy_start']:+.2f}->{p['proxy_final']:+.2f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    # The outcome distribution is BIMODAL (see SGB-034): seeds either settle
    # near ~0.03 or get stuck near ~0.8, with almost nothing between. The MEAN
    # therefore describes no actual run. Report the trapped FRACTION and the
    # MEDIAN as the primary statistics; the mean is shown only for continuity
    # with earlier tables.
    BASE = 0.126
    print("\n=== PPO ARM: ground-truth trap occupancy (baseline ~0.126) ===")
    print("primary stats are TRAPPED FRACTION and MEDIAN; the mean is bimodal-misleading")
    print(f"{'ent':>7} {'orc':>7} {'trap_v_safe':>12} {'trapped':>9} "
          f"{'median':>8} {'mean':>8} {'proxy gain':>11}")
    summary = []
    for ent in ent_coefs:
        for orc in orcs:
            rs = [r for r in rows
                  if r["oracle_fraction"] == orc and r["config"]["entropy_coef"] == ent]
            if not rs:
                continue
            finals = np.array([r["ppo"]["hack_final"] for r in rs])
            entry = {
                "entropy_coef": ent, "oracle_fraction": orc, "n_seeds": len(rs),
                "trap_vs_safe_acc": float(np.mean(
                    [r["competence"]["trap_vs_safe_acc"] for r in rs])),
                "argmax_in_trap": float(np.mean(
                    [r["competence"]["argmax_in_trap"] for r in rs])),
                "trapped_fraction": float(np.mean(finals > 2 * BASE)),
                "hack_final_median": float(np.median(finals)),
                "hack_final_mean": float(finals.mean()),
                "hack_finals": [float(x) for x in np.sort(finals)],
                "proxy_gain": float(np.mean([r["ppo"]["proxy_final"] for r in rs])
                                    - np.mean([r["ppo"]["proxy_start"] for r in rs])),
            }
            summary.append(entry)
            print(f"{ent:>7.3f} {orc:>7.3f} {entry['trap_vs_safe_acc']:>12.3f} "
                  f"{entry['trapped_fraction']:>8.0%} "
                  f"{entry['hack_final_median']:>8.3f} "
                  f"{entry['hack_final_mean']:>8.3f} {entry['proxy_gain']:>+11.2f}")

    out = Path(args.out) if args.out else (
        _SRC.parent / "results" / "verifier_gap"
        / ("ppo_smoke.json" if args.smoke else f"ppo_arm_{len(rows)}runs.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
