#!/usr/bin/env python3
"""
Modal.com GPU Deployment for SGPO Training

This script runs the SGPO vs PPO experiment on Modal's cloud GPUs.
Pay-per-second billing, scales to zero when not in use.

Setup:
    pip install modal
    modal setup  # One-time authentication

Usage:
    modal run modal_gpo_training.py
    modal run modal_gpo_training.py --n-episodes 500

Deploy as web endpoint:
    modal deploy modal_gpo_training.py
"""

import modal

# Define the Modal app
app = modal.App("gpo-training")

# Container image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "numpy", 
    "matplotlib",
)

# Volume for persisting results
volume = modal.Volume.from_name("gpo-results", create_if_missing=True)


@app.function(
    gpu="T4",  # Options: "T4", "A10G", "A100", "H100"
    image=image,
    timeout=3600,
    volumes={"/results": volume},
)
def train_gpo_experiment(n_episodes: int = 300, seed: int = 42):
    """Run SGPO training experiment on GPU."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from typing import Dict, List, Tuple
    import json
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Environment
    class HazardNavigationEnv:
        def __init__(self, hazards=None, max_steps=200, dt=0.1):
            self.goal_pos = np.array([2.0, 2.0])
            self.hazards = hazards or [
                (np.array([1.0, 0.5]), 0.3),
                (np.array([0.5, 1.5]), 0.25),
                (np.array([1.5, 1.0]), 0.35),
            ]
            self.max_steps = max_steps
            self.dt = dt
            self.reset()

        def reset(self):
            self.pos = np.array([0.0, 0.0])
            self.vel = np.array([0.0, 0.0])
            self.step_count = 0
            return np.concatenate([self.pos, self.vel]).astype(np.float32), {}

        def step(self, action):
            action = np.clip(action, -1.0, 1.0)
            self.vel = np.clip(self.vel + action * self.dt, -2.0, 2.0)
            self.pos = self.pos + self.vel * self.dt
            self.step_count += 1

            cost = 0.0
            for center, radius in self.hazards:
                if np.linalg.norm(self.pos - center) < radius:
                    cost = 1.0
                    break

            dist = np.linalg.norm(self.pos - self.goal_pos)
            reward = -0.1 * dist + (10.0 if dist < 0.2 else 0) - (5.0 if cost > 0 else 0)
            
            return (
                np.concatenate([self.pos, self.vel]).astype(np.float32),
                reward,
                dist < 0.2,
                self.step_count >= self.max_steps,
                {'cost': cost}
            )

    # Metric
    class RiemannianMetric(nn.Module):
        def __init__(self, hazard_centers, hazard_radii):
            super().__init__()
            self.register_buffer('centers', torch.tensor(np.array(hazard_centers), dtype=torch.float32))
            self.register_buffer('radii', torch.tensor(hazard_radii, dtype=torch.float32))
            self.base = nn.Parameter(torch.tensor(1.0))
            self.severity = nn.Parameter(torch.tensor(10.0))
            self.sharpness = nn.Parameter(torch.tensor(2.0))

        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            pos = x[:, :2]
            g = torch.ones(pos.shape[0], 1, device=x.device) * F.softplus(self.base)
            for i in range(len(self.radii)):
                dist = torch.norm(pos - self.centers[i], dim=1, keepdim=True)
                margin = torch.clamp(dist - self.radii[i] * 0.8, min=0.01)
                g = g + F.softplus(self.severity) / (margin ** F.softplus(self.sharpness))
            return g

    # Networks
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
            self.log_std = nn.Parameter(torch.zeros(2) - 0.5)
        def forward(self, x):
            return Normal(self.net(x.unsqueeze(0) if x.dim() == 1 else x), torch.exp(self.log_std.clamp(-20, 2)))
        def get_action(self, x):
            d = self.forward(x)
            a = d.sample()
            return a, d.log_prob(a).sum(-1)

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        def forward(self, x):
            return self.net(x.unsqueeze(0) if x.dim() == 1 else x)

    def compute_gae(rewards, values, dones, gamma=0.99, lam=0.97):
        advantages, returns = [], []
        gae = 0
        for t in reversed(range(len(rewards))):
            next_val = 0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def train_episode(env, actor, critic, metric, actor_opt, critic_opt, metric_opt, device, use_gpo=False):
        obs, _ = env.reset()
        observations, actions, log_probs, rewards, values, dones, costs = [], [], [], [], [], [], []
        
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_prob = actor.get_action(obs_t)
                value = critic(obs_t)
            next_obs, reward, term, trunc, info = env.step(action.cpu().numpy().flatten())
            done = term or trunc
            observations.append(obs_t)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            costs.append(info['cost'])
            obs = next_obs

        advantages, returns = compute_gae(rewards, values, dones)
        observations = torch.stack(observations)
        
        if use_gpo:
            with torch.no_grad():
                g = metric(observations).squeeze()
            advantages = advantages.to(device) / torch.sqrt(g + 1e-8)
        else:
            advantages = advantages.to(device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = returns.to(device)
        actions = torch.stack(actions).squeeze(1)
        old_log_probs = torch.stack(log_probs)

        for _ in range(10):
            dist = actor(observations)
            new_log_probs = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            actor_opt.zero_grad()
            (-torch.min(surr1, surr2).mean()).backward()
            actor_opt.step()
            
            critic_opt.zero_grad()
            F.mse_loss(critic(observations).squeeze(), returns).backward()
            critic_opt.step()

        if use_gpo:
            cost_t = torch.tensor(costs, dtype=torch.float32, device=device)
            metric_opt.zero_grad()
            F.mse_loss(metric(observations).squeeze(), 1.0 + 10.0 * cost_t).backward()
            metric_opt.step()

        return sum(rewards), sum(1 for c in costs if c > 0)

    # Run experiment
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    hazards = [(np.array([1.0, 0.5]), 0.3), (np.array([0.5, 1.5]), 0.25), (np.array([1.5, 1.0]), 0.35)]
    results = {}

    for method, use_gpo in [("PPO", False), ("SGPO", True)]:
        print(f"\nTraining {method}...")
        torch.manual_seed(seed)
        env = HazardNavigationEnv(hazards=hazards)
        actor = Actor().to(device)
        critic = Critic().to(device)
        metric = RiemannianMetric([h[0] for h in hazards], [h[1] for h in hazards]).to(device)
        
        actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
        metric_opt = torch.optim.Adam(metric.parameters(), lr=1e-2)

        returns, violations = [], []
        start = time.time()
        for ep in range(n_episodes):
            ret, viol = train_episode(env, actor, critic, metric, actor_opt, critic_opt, metric_opt, device, use_gpo)
            returns.append(ret)
            violations.append(viol)
            if (ep + 1) % 50 == 0:
                print(f"  Episode {ep+1}: Return={np.mean(returns[-50:]):.1f}, Violations={np.mean(violations[-50:]):.1f}")
        
        results[method] = {'returns': returns, 'violations': violations, 'time': time.time() - start}

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for method in results:
        print(f"{method}: Return={np.mean(results[method]['returns'][-50:]):.1f}, "
              f"Total Violations={sum(results[method]['violations'])}, "
              f"Time={results[method]['time']:.1f}s")

    improvement = (sum(results['PPO']['violations']) - sum(results['SGPO']['violations'])) / max(sum(results['PPO']['violations']), 1) * 100
    print(f"\nSGPO: {improvement:.0f}% fewer violations than PPO")

    # Save results
    with open("/results/gpo_results.json", "w") as f:
        json.dump({m: {'total_violations': sum(results[m]['violations']), 
                       'mean_return': float(np.mean(results[m]['returns'][-50:]))} 
                  for m in results}, f, indent=2)
    
    volume.commit()
    return results


@app.local_entrypoint()
def main(n_episodes: int = 300):
    """Entry point for `modal run`."""
    print(f"Starting SGPO experiment with {n_episodes} episodes...")
    results = train_gpo_experiment.remote(n_episodes=n_episodes)
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    for method in results:
        print(f"{method}: Total Violations = {sum(results[method]['violations'])}")
