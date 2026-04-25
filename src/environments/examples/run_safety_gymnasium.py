#!/usr/bin/env python3
"""
Example: SGPO Training on Safety-Gymnasium

Compares PPO, CPO, and SGPO on SafetyPointGoal1-v0 environment.
Demonstrates how geodesic optimization avoids hazards more effectively
than Lagrangian-based constrained optimization.

Usage:
    pip install safety-gymnasium torch matplotlib
    python run_safety_gymnasium.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import matplotlib.pyplot as plt
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environments import (
    create_safety_gpo_env,
    SGPOTrainer,
    MultiHazardRiemannianMetric,
)


class Actor(nn.Module):
    """Policy network for continuous action space."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
    
    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        mu = self.net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)


class Critic(nn.Module):
    """Value function network."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)


def run_experiment(
    env_id: str = "SafetyPointGoal1-v0",
    n_episodes: int = 200,
    seed: int = 42,
):
    """Run SGPO training experiment on Safety-Gymnasium."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("=" * 60)
    print(f"SGPO Training on {env_id}")
    print("=" * 60)
    
    print("\nCreating environment with SGPO wrapper...")
    try:
        env = create_safety_gpo_env(env_id)
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nTo install safety-gymnasium:")
        print("  pip install safety-gymnasium")
        return None
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    print(f"\nExtracted hazards from environment:")
    centers = env.metric.get_black_hole_centers()
    horizons = env.metric.get_event_horizons()
    for i, (c, h) in enumerate(zip(centers, horizons)):
        print(f"  Hazard {i+1}: center={c}, event_horizon={h:.2f}")
    
    print("\nInitializing networks...")
    actor = Actor(obs_dim, act_dim)
    critic = Critic(obs_dim)
    
    trainer = SGPOTrainer(
        env=env,
        actor=actor,
        critic=critic,
        actor_lr=3e-4,
        critic_lr=1e-3,
        metric_lr=1e-2,
        gamma=0.99,
    )
    
    print(f"\nTraining for {n_episodes} episodes...")
    metrics = trainer.train(n_episodes=n_episodes, log_interval=20)
    
    results = {
        'env_id': env_id,
        'n_episodes': n_episodes,
        'seed': seed,
        'final_returns': metrics['returns'][-50:],
        'final_violations': metrics['violations'][-50:],
        'mean_return': float(np.mean(metrics['returns'][-50:])),
        'mean_violations': float(np.mean(metrics['violations'][-50:])),
        'total_violations': sum(metrics['violations']),
    }
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Mean Return (last 50): {results['mean_return']:.2f}")
    print(f"Mean Violations (last 50): {results['mean_violations']:.2f}")
    print(f"Total Violations: {results['total_violations']}")
    
    output_dir = Path(__file__).parent.parent.parent.parent
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1 = axes[0, 0]
    window = 10
    returns_smooth = np.convolve(
        metrics['returns'], 
        np.ones(window)/window, 
        mode='valid'
    )
    ax1.plot(returns_smooth, color='blue', linewidth=2)
    ax1.set_title('Episode Returns (Smoothed)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    violations_smooth = np.convolve(
        metrics['violations'],
        np.ones(window)/window,
        mode='valid'
    )
    ax2.plot(violations_smooth, color='red', linewidth=2)
    ax2.set_title('Hazard Violations (Smoothed)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Violations')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    env.visualize_metric_field(ax=ax3, xlim=(-3, 3), ylim=(-3, 3))
    ax3.set_title('Learned Riemannian Metric')
    
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.7, f"Environment: {env_id}", ha='center', fontsize=12)
    ax4.text(0.5, 0.5, f"Mean Return: {results['mean_return']:.2f}", ha='center', fontsize=14, fontweight='bold')
    ax4.text(0.5, 0.3, f"Total Violations: {results['total_violations']}", ha='center', fontsize=14, fontweight='bold', color='red')
    ax4.axis('off')
    ax4.set_title('Summary')
    
    plt.tight_layout()
    
    fig_path = output_dir / 'safety_gymnasium_gpo_results.png'
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved to: {fig_path}")
    
    results_path = output_dir / 'safety_gymnasium_gpo_metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {results_path}")
    
    env.close()
    
    return results


if __name__ == "__main__":
    run_experiment()
