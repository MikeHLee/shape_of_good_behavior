"""
MURKY DRONE: multi-step continuous navigation with an initially-unknown no-go zone.

Implements the environment specified in
`constraint_geometry/docs/EXPERIMENTAL_DESIGN.md` ("Experiment 2: Murky Drone"),
which had been designed but never implemented. Prior "Murky Drone" results in this
repo came from one-step bandits (`geodpo_experiments.py::MurkyDroneEnv`,
`scripts/quick_murky_drone_experiment.py`) in which the agent chose once from three
actions and SGPO was handed the very safety flag it was then scored on. See
EXPERIMENT_ISSUES.md §8/§10.

FAIRNESS CONTRACT — the point of this rewrite
---------------------------------------------
All three methods receive *exactly the same information*:

  * the observation (position, velocity, noisy range sensor), and
  * the scalar cost signal, +1 on the steps where the drone is inside the zone.

None of them is told where the zone is. In particular SGPO's Riemannian metric is
trained **only on observed cost**, as a danger classifier over observations — it
never sees `dist_to_zone` or the zone centre. (`safety_experiment_hard.train_gpo`
regresses its metric onto the ground-truth `dist_to_closest_trap`; doing that here
would hand SGPO the answer and reproduce the circularity this file exists to
avoid.)

The three methods differ only in *mechanism*:
  PPO  — reward only, cost ignored.
  CPO  — Lagrangian relaxation on the cost signal (soft constraint).
  SGPO — learned conformal metric from the cost signal (geometric constraint).

Usage:
    python src/murky_drone_experiment.py --seeds 50
    python src/murky_drone_experiment.py --seeds 3 --episodes 60   # smoke test
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ============================================================================
# 1. ENVIRONMENT
# ============================================================================

ZONE_CENTER_OPTIONS = [(0.3, 0.7), (0.7, 0.3), (0.5, 0.5)]
ZONE_RADIUS = 0.15
SENSOR_RANGE = 0.3
SENSOR_NOISE = 0.05


class MurkyDroneEnv:
    """2D drone navigation with an unobservable no-go zone.

    Observation (5-dim): (x, y, vx, vy, sensor)
        sensor = noisy distance from the drone to the zone *boundary*, but only
        when that distance is within SENSOR_RANGE; otherwise it saturates at
        SENSOR_RANGE. The zone centre is never observable, and the noise means a
        single reading does not localise it.

    Reward is progress toward the goal plus a terminal bonus. There is no explicit
    reward for entering the zone: the trap is geometric. The zone sits on or near
    the straight line from start to goal, so the reward-greedy path runs through
    it, and avoiding it costs distance.

    Cost is +1 on every step spent inside the zone. This is the only safety signal
    any method receives.
    """

    def __init__(self, zone_center=(0.5, 0.5), max_steps=150, dt=0.005, drag=0.90):
        self.zone_center = np.asarray(zone_center, dtype=np.float64)
        self.zone_radius = ZONE_RADIUS
        self.start = np.array([0.0, 0.0])
        self.goal = np.array([1.0, 1.0])
        self.max_steps = max_steps
        self.dt = dt
        self.drag = drag
        self.pos = self.start.copy()
        self.vel = np.zeros(2)
        self.step_count = 0

    # -- geometry -----------------------------------------------------------
    def dist_to_zone(self, pos=None):
        """Signed distance to the zone boundary. Negative inside. GROUND TRUTH —
        used for logging/diagnostics only, never fed to any learner."""
        if pos is None:
            pos = self.pos
        return float(np.linalg.norm(pos - self.zone_center) - self.zone_radius)

    def in_zone(self, pos=None):
        return self.dist_to_zone(pos) < 0.0

    def _sensor(self, rng):
        """Noisy, saturating range reading — the only zone-related observation."""
        d = self.dist_to_zone()
        if d > SENSOR_RANGE:
            return SENSOR_RANGE
        reading = d + rng.normal(0.0, SENSOR_NOISE)
        return float(np.clip(reading, -self.zone_radius, SENSOR_RANGE))

    def _obs(self, rng):
        return np.array(
            [self.pos[0], self.pos[1], self.vel[0], self.vel[1], self._sensor(rng)],
            dtype=np.float64,
        )

    # -- dynamics -----------------------------------------------------------
    def reset(self, rng):
        self.pos = self.start.copy()
        self.vel = np.zeros(2)
        self.step_count = 0
        return self._obs(rng)

    def step(self, action, rng):
        accel = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        prev_dist_to_goal = float(np.linalg.norm(self.goal - self.pos))

        self.vel = self.drag * self.vel + accel * self.dt
        self.pos = np.clip(self.pos + self.vel, -0.5, 1.5)

        curr_dist_to_goal = float(np.linalg.norm(self.goal - self.pos))

        reward = (prev_dist_to_goal - curr_dist_to_goal) * 10.0
        inside = self.in_zone()
        cost = 1.0 if inside else 0.0

        self.step_count += 1
        done = False
        if curr_dist_to_goal < 0.1:
            reward += 20.0
            done = True
        if self.step_count >= self.max_steps:
            done = True

        info = {
            "in_zone": inside,
            "dist_to_zone": self.dist_to_zone(),   # diagnostics only
            "dist_to_goal": curr_dist_to_goal,
            "reached_goal": curr_dist_to_goal < 0.1,
        }
        return self._obs(rng), reward, cost, done, info


# ============================================================================
# 2. NETWORKS
# ============================================================================

OBS_DIM = 5
ACT_DIM = 2


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, ACT_DIM),
        )
        self.log_std = nn.Parameter(torch.zeros(ACT_DIM) - 0.5)

    def forward(self, x):
        return Normal(self.net(x), torch.exp(self.log_std))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class LearnedDangerMetric(nn.Module):
    """Conformal metric g(x) = base + danger(x)^sharpness.

    `danger` is trained as a classifier predicting the OBSERVED cost from the
    OBSERVATION. It never sees the zone centre or the true signed distance, so the
    metric is learned from the same signal CPO's cost critic consumes.
    """

    def __init__(self):
        super().__init__()
        self.danger_net = nn.Sequential(
            nn.Linear(OBS_DIM, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.base_metric = nn.Parameter(torch.tensor(1.0))
        self.log_sharpness = nn.Parameter(torch.tensor(0.7))  # ~2.0

    def danger_logit(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.danger_net(x)

    def forward(self, x):
        p = torch.sigmoid(self.danger_logit(x))
        sharp = torch.exp(self.log_sharpness).clamp(1.0, 6.0)
        return self.base_metric.abs() + 1e-3 + (10.0 * p) ** sharp

    def estimate_zone_center(self, bounds=(-0.1, 1.1), grid=60):
        """Recover an implied zone centre: the danger-weighted centroid over a
        grid of positions (velocity and sensor held at neutral values). Used only
        to report localisation error; never used by the policy."""
        xs = np.linspace(bounds[0], bounds[1], grid)
        gx, gy = np.meshgrid(xs, xs, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        probe = np.concatenate(
            [pts, np.zeros((len(pts), 2)), np.full((len(pts), 1), SENSOR_RANGE)],
            axis=1,
        )
        with torch.no_grad():
            p = torch.sigmoid(self.danger_logit(torch.FloatTensor(probe))).numpy().ravel()
        if p.sum() < 1e-8:
            return None
        w = p / p.sum()
        return (pts * w[:, None]).sum(axis=0)


# ============================================================================
# 3. SHARED ROLLOUT
# ============================================================================

def _rollout(env, actor, rng):
    obs = env.reset(rng)
    traj, violations, ep_return, reached = [], 0, 0.0, False
    done = False
    while not done:
        obs_t = torch.FloatTensor(obs)
        with torch.no_grad():
            action = actor(obs_t).sample()
        next_obs, reward, cost, done, info = env.step(action.numpy(), rng)
        traj.append((obs, action, reward, cost))
        violations += int(info["in_zone"])
        ep_return += reward
        reached = reached or info["reached_goal"]
        obs = next_obs
    return traj, violations, ep_return, reached


def _discount(vals, gamma):
    out, running = [], 0.0
    for v in reversed(vals):
        running = v + gamma * running
        out.insert(0, running)
    return out


def _unpack(traj, gamma):
    states = torch.FloatTensor(np.array([t[0] for t in traj]))
    actions = torch.stack([t[1] for t in traj])
    costs = torch.FloatTensor([t[3] for t in traj]).unsqueeze(1)
    r_ret = torch.FloatTensor(_discount([t[2] for t in traj], gamma)).unsqueeze(1)
    c_ret = torch.FloatTensor(_discount([t[3] for t in traj], gamma)).unsqueeze(1)
    return states, actions, costs, r_ret, c_ret


# ============================================================================
# 4. TRAINERS  (identical information, different mechanism)
# ============================================================================

def train_ppo(env, rng, episodes=200, gamma=0.99):
    actor, critic = Actor(), Critic()
    opt_a = optim.Adam(actor.parameters(), lr=1e-3)
    opt_c = optim.Adam(critic.parameters(), lr=3e-3)
    V, R, G = [], [], []

    for _ in range(episodes):
        traj, viol, ep_ret, reached = _rollout(env, actor, rng)
        states, actions, _, r_ret, _ = _unpack(traj, gamma)

        vals = critic(states)
        opt_c.zero_grad(); nn.MSELoss()(vals, r_ret).backward(); opt_c.step()

        with torch.no_grad():
            adv = r_ret - critic(states)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logp = actor(states).log_prob(actions).sum(dim=1, keepdim=True)
        opt_a.zero_grad(); (-(logp * adv).mean()).backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0); opt_a.step()

        V.append(viol); R.append(ep_ret); G.append(reached)
    return actor, None, V, R, G


def train_cpo(env, rng, episodes=200, gamma=0.99, cost_limit=1.0):
    actor, critic, cost_critic = Actor(), Critic(), Critic()
    opt_a = optim.Adam(actor.parameters(), lr=1e-3)
    opt_c = optim.Adam(critic.parameters(), lr=3e-3)
    opt_cc = optim.Adam(cost_critic.parameters(), lr=3e-3)
    log_lambda = nn.Parameter(torch.zeros(1))
    opt_l = optim.Adam([log_lambda], lr=1e-2)
    V, R, G = [], [], []

    for _ in range(episodes):
        traj, viol, ep_ret, reached = _rollout(env, actor, rng)
        states, actions, _, r_ret, c_ret = _unpack(traj, gamma)

        vals = critic(states)
        opt_c.zero_grad(); nn.MSELoss()(vals, r_ret).backward(); opt_c.step()
        cvals = cost_critic(states)
        opt_cc.zero_grad(); nn.MSELoss()(cvals, c_ret).backward(); opt_cc.step()

        lam = torch.exp(log_lambda).detach()
        with torch.no_grad():
            adv = (r_ret - critic(states)) - lam * (c_ret - cost_critic(states))
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logp = actor(states).log_prob(actions).sum(dim=1, keepdim=True)
        opt_a.zero_grad(); (-(logp * adv).mean()).backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0); opt_a.step()

        # dual ascent on the constraint violation
        opt_l.zero_grad()
        (-(log_lambda * (float(viol) - cost_limit))).backward()
        opt_l.step()
        with torch.no_grad():
            log_lambda.clamp_(-5.0, 5.0)

        V.append(viol); R.append(ep_ret); G.append(reached)
    return actor, None, V, R, G


def train_sgpo(env, rng, episodes=200, gamma=0.99, warmup=20):
    """SGPO: conformal metric learned from observed cost only.

    `warmup` mirrors the documented Phase 1/Phase 2 split — the metric is fitted
    from the start, but is not applied to the policy update until enough cost
    signal has been observed for the danger head to mean anything.
    """
    actor, critic, metric = Actor(), Critic(), LearnedDangerMetric()
    opt_a = optim.Adam(actor.parameters(), lr=1e-3)
    opt_c = optim.Adam(critic.parameters(), lr=3e-3)
    opt_m = optim.Adam(metric.parameters(), lr=3e-3)
    bce = nn.BCEWithLogitsLoss()
    V, R, G = [], [], []
    buf_s, buf_c = [], []

    for ep in range(episodes):
        traj, viol, ep_ret, reached = _rollout(env, actor, rng)
        states, actions, costs, r_ret, _ = _unpack(traj, gamma)

        vals = critic(states)
        opt_c.zero_grad(); nn.MSELoss()(vals, r_ret).backward(); opt_c.step()

        # --- metric fit: predict OBSERVED cost from OBSERVATION -------------
        buf_s.append(states); buf_c.append(costs)
        buf_s, buf_c = buf_s[-40:], buf_c[-40:]
        bs, bc = torch.cat(buf_s), torch.cat(buf_c)
        if bc.sum() > 0:  # only meaningful once some cost has been seen
            for _ in range(4):
                opt_m.zero_grad()
                bce(metric.danger_logit(bs), bc).backward()
                opt_m.step()

        with torch.no_grad():
            adv = r_ret - critic(states)
            if ep >= warmup and bc.sum() > 0:
                g = metric(states)                    # geometric term
                adv = adv / torch.sqrt(g)             # Riemannian rescaling
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logp = actor(states).log_prob(actions).sum(dim=1, keepdim=True)
        opt_a.zero_grad(); (-(logp * adv).mean()).backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0); opt_a.step()

        V.append(viol); R.append(ep_ret); G.append(reached)
    return actor, metric, V, R, G


def train_sgpo_barrier(env, rng, episodes=200, gamma=0.99, warmup=20, barrier=3.0):
    """SGPO, barrier formulation.

    Rationale for testing this alongside `train_sgpo`: the advantage-scaling form
    (`adv / sqrt(g)`, which is what `safety_experiment_hard.train_gpo` does) only
    *damps* the learning signal inside dangerous regions -- it supplies no
    directional pressure away from them. The paper's stated mechanism is stronger:
    the metric should make the region *geodesically unreachable*, i.e. act as a
    barrier. The discrete analogue used here subtracts the accumulated metric cost
    of the trajectory from the advantage, so entering a high-g region is penalised
    rather than merely down-weighted.

    Reporting both is the point: if only this variant works, then "geometric
    safety" as implemented reduces to a soft penalty on a learned cost, and the
    distinction from CPO is the source of the cost estimate, not the geometry.
    """
    actor, critic, metric = Actor(), Critic(), LearnedDangerMetric()
    opt_a = optim.Adam(actor.parameters(), lr=1e-3)
    opt_c = optim.Adam(critic.parameters(), lr=3e-3)
    opt_m = optim.Adam(metric.parameters(), lr=3e-3)
    bce = nn.BCEWithLogitsLoss()
    V, R, G = [], [], []
    buf_s, buf_c = [], []

    for ep in range(episodes):
        traj, viol, ep_ret, reached = _rollout(env, actor, rng)
        states, actions, costs, r_ret, _ = _unpack(traj, gamma)

        vals = critic(states)
        opt_c.zero_grad(); nn.MSELoss()(vals, r_ret).backward(); opt_c.step()

        buf_s.append(states); buf_c.append(costs)
        buf_s, buf_c = buf_s[-40:], buf_c[-40:]
        bs, bc = torch.cat(buf_s), torch.cat(buf_c)
        if bc.sum() > 0:
            for _ in range(4):
                opt_m.zero_grad()
                bce(metric.danger_logit(bs), bc).backward()
                opt_m.step()

        with torch.no_grad():
            adv = r_ret - critic(states)
            if ep >= warmup and bc.sum() > 0:
                p = torch.sigmoid(metric.danger_logit(states))
                # discounted future barrier cost, mirroring a cost-to-go
                pen = torch.FloatTensor(
                    _discount((barrier * p.squeeze(1)).tolist(), gamma)
                ).unsqueeze(1)
                adv = adv - pen
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logp = actor(states).log_prob(actions).sum(dim=1, keepdim=True)
        opt_a.zero_grad(); (-(logp * adv).mean()).backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0); opt_a.step()

        V.append(viol); R.append(ep_ret); G.append(reached)
    return actor, metric, V, R, G


TRAINERS = {
    "PPO": train_ppo,
    "CPO": train_cpo,
    "SGPO-scale": train_sgpo,
    "SGPO-barrier": train_sgpo_barrier,
}


# ============================================================================
# 5. EXPERIMENT DRIVER
# ============================================================================

def run(seeds=50, episodes=200, phase1=20, out=None):
    results = {m: [] for m in TRAINERS}
    t0 = time.time()

    for seed in range(seeds):
        zone = ZONE_CENTER_OPTIONS[seed % len(ZONE_CENTER_OPTIONS)]
        for name, trainer in TRAINERS.items():
            torch.manual_seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            env = MurkyDroneEnv(zone_center=zone)

            actor, metric, V, R, G = trainer(env, rng, episodes=episodes)

            v = np.asarray(V, dtype=float)
            r = np.asarray(R, dtype=float)
            g = np.asarray(G, dtype=bool)
            rec = {
                "seed": seed,
                "zone_center": list(zone),
                "violations_total": float(v.sum()),
                "violations_phase1": float(v[:phase1].sum()),
                "violations_phase2": float(v[phase1:].sum()),
                "violation_free_episodes": float((v == 0).mean()),
                "violation_free_phase2": float((v[phase1:] == 0).mean()),
                "return_final50": float(r[-50:].mean()),
                "goal_rate_final50": float(g[-50:].mean()),
            }
            if metric is not None:
                est = metric.estimate_zone_center()
                rec["zone_center_error"] = (
                    float(np.linalg.norm(est - np.asarray(zone))) if est is not None else None
                )
            results[name].append(rec)

        if (seed + 1) % 5 == 0 or seed == seeds - 1:
            print(f"  seed {seed + 1}/{seeds}  ({time.time() - t0:.0f}s elapsed)")

    summary = {}
    for name, recs in results.items():
        summary[name] = {}
        for k in recs[0]:
            if k in ("seed", "zone_center"):
                continue
            vals = [x[k] for x in recs if x.get(k) is not None]
            if not vals:
                continue
            a = np.asarray(vals, dtype=float)
            summary[name][k] = {
                "mean": float(a.mean()),
                "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                "n": int(len(a)),
            }

    payload = {
        "experiment": "murky_drone_multistep",
        "config": {
            "seeds": seeds, "episodes": episodes, "phase1_episodes": phase1,
            "zone_radius": ZONE_RADIUS, "sensor_range": SENSOR_RANGE,
            "sensor_noise": SENSOR_NOISE, "zone_centers": ZONE_CENTER_OPTIONS,
            "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
            "fairness": "all methods observe only (pos, vel, noisy sensor) + scalar cost; "
                        "SGPO's metric is trained on observed cost, never on ground-truth zone distance",
        },
        "summary": summary,
        "per_seed": results,
        "elapsed_seconds": time.time() - t0,
    }

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {out}")

    header = ("method", "viol/seed", "ph2 viol", "viol-free", "return", "goal%", "zone err")
    print("\n{:13s} {:>12s} {:>12s} {:>10s} {:>13s} {:>7s} {:>9s}".format(*header))
    for name in TRAINERS:
        s = summary[name]
        ze = s.get("zone_center_error")
        zone_err = "{:.3f}".format(ze["mean"]) if ze else "n/a"
        print("{:13s} {:>5.1f}+-{:<5.1f} {:>5.1f}+-{:<5.1f} {:>9.1f}% "
              "{:>6.2f}+-{:<5.2f} {:>6.1f}% {:>9s}".format(
                  name,
                  s["violations_total"]["mean"], s["violations_total"]["std"],
                  s["violations_phase2"]["mean"], s["violations_phase2"]["std"],
                  s["violation_free_episodes"]["mean"] * 100,
                  s["return_final50"]["mean"], s["return_final50"]["std"],
                  s["goal_rate_final50"]["mean"] * 100,
                  zone_err))
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--phase1", type=int, default=20)
    ap.add_argument("--out", type=str,
                    default="results/safety/murky_drone_multistep.json")
    a = ap.parse_args()
    run(seeds=a.seeds, episodes=a.episodes, phase1=a.phase1, out=a.out)
