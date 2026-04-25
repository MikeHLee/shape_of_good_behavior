#!/usr/bin/env python3
"""
SGPO Training Experiment: Sheaf-Geodesic Policy Optimization vs PPO vs CPO

Demonstrates SGPO's ability to navigate around "black holes" (hazard regions)
using Riemannian geometry, compared to standard RL approaches.

This experiment uses a simple 2D navigation environment with hazards,
avoiding the dependency issues with mujoco/safety-gymnasium.

Usage:
    python gpo_training_experiment.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environments.base import RiemannianMetricBase


# =============================================================================
# ENVIRONMENT: 2D Navigation with Hazards
# =============================================================================

class HazardNavigationEnv:
    """
    2D navigation environment with hazard regions (black holes).
    
    Agent must reach the goal while avoiding hazards.
    This is similar to Safety-Gymnasium's PointGoal but standalone.
    """
    
    def __init__(
        self,
        goal_pos: np.ndarray = np.array([2.0, 2.0]),
        hazards: List[Tuple[np.ndarray, float]] = None,
        max_steps: int = 200,
        dt: float = 0.1,
    ):
        self.goal_pos = goal_pos
        self.hazards = hazards or [
            (np.array([1.0, 0.5]), 0.3),   # center, radius
            (np.array([0.5, 1.5]), 0.25),
            (np.array([1.5, 1.0]), 0.35),
        ]
        self.max_steps = max_steps
        self.dt = dt
        
        self.observation_space_shape = (4,)  # [x, y, vx, vy]
        self.action_space_shape = (2,)       # [ax, ay]
        self.action_bound = 1.0
        
        self.reset()
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.pos = np.array([0.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        self.step_count = 0
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.pos, self.vel]).astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -self.action_bound, self.action_bound)
        
        # Physics update
        self.vel = self.vel + action * self.dt
        self.vel = np.clip(self.vel, -2.0, 2.0)  # velocity limit
        self.pos = self.pos + self.vel * self.dt
        self.step_count += 1
        
        # Check hazard collision
        in_hazard = False
        cost = 0.0
        for center, radius in self.hazards:
            dist = np.linalg.norm(self.pos - center)
            if dist < radius:
                in_hazard = True
                cost = 1.0
                break
        
        # Reward: distance to goal + hazard penalty
        dist_to_goal = np.linalg.norm(self.pos - self.goal_pos)
        reward = -0.1 * dist_to_goal  # Small penalty for distance
        
        if dist_to_goal < 0.2:
            reward += 10.0  # Goal bonus
        
        if in_hazard:
            reward -= 5.0  # Hazard penalty
        
        # Termination
        terminated = dist_to_goal < 0.2
        truncated = self.step_count >= self.max_steps
        
        info = {
            'cost': cost,
            'in_hazard': in_hazard,
            'dist_to_goal': dist_to_goal,
        }
        
        return self._get_obs(), reward, terminated, truncated, info


# =============================================================================
# RIEMANNIAN METRIC
# =============================================================================

class NavigationRiemannianMetric(RiemannianMetricBase):
    """
    Riemannian metric for the hazard navigation environment.
    Models hazards as Schwarzschild-like singularities.
    """
    
    def __init__(
        self,
        hazard_centers: List[np.ndarray],
        hazard_radii: List[float],
        base_metric: float = 1.0,
        severity: float = 10.0,
        sharpness: float = 2.0,
        learnable: bool = True,
    ):
        super().__init__(state_dim=2)
        
        self.register_buffer(
            'hazard_centers',
            torch.tensor(np.array(hazard_centers), dtype=torch.float32)
        )
        self.register_buffer(
            'hazard_radii',
            torch.tensor(hazard_radii, dtype=torch.float32)
        )
        
        if learnable:
            self.base_metric = nn.Parameter(torch.tensor(base_metric))
            self.severity = nn.Parameter(torch.tensor(severity))
            self.sharpness = nn.Parameter(torch.tensor(sharpness))
        else:
            self.register_buffer('base_metric', torch.tensor(base_metric))
            self.register_buffer('severity', torch.tensor(severity))
            self.register_buffer('sharpness', torch.tensor(sharpness))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute metric g(x) at state x."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Only use position (first 2 dims)
        pos = x[:, :2]
        batch_size = pos.shape[0]
        
        # Base metric
        g = torch.ones(batch_size, 1, device=x.device) * F.softplus(self.base_metric)
        
        # Add contribution from each hazard
        for i in range(len(self.hazard_radii)):
            center = self.hazard_centers[i]
            radius = self.hazard_radii[i]
            
            # Distance to hazard center
            dist = torch.norm(pos - center.unsqueeze(0), dim=1, keepdim=True)
            
            # Schwarzschild-like metric: g = 1 / (1 - r_s/r)^α
            # Near the event horizon (r → r_s), g → ∞
            event_horizon = radius * 0.8
            margin = torch.clamp(dist - event_horizon, min=0.01)
            
            schwarzschild = F.softplus(self.severity) / (margin ** F.softplus(self.sharpness))
            g = g + schwarzschild
        
        return g
    
    def get_black_hole_centers(self) -> List[np.ndarray]:
        return [c.cpu().numpy() for c in self.hazard_centers]
    
    def get_event_horizons(self) -> List[float]:
        return [r.item() * 0.8 for r in self.hazard_radii]


# =============================================================================
# NEURAL NETWORK POLICIES
# =============================================================================

class Actor(nn.Module):
    """Gaussian policy network."""
    
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
    
    def forward(self, obs: torch.Tensor) -> Normal:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        mu = self.net(obs)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return Normal(mu, std)
    
    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


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
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)


# =============================================================================
# TRAINING ALGORITHMS
# =============================================================================

@dataclass
class TrainingConfig:
    gamma: float = 0.99
    lam: float = 0.97
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    metric_lr: float = 1e-2
    train_iters: int = 10
    cost_limit: float = 0.1  # For CPO


class PPOTrainer:
    """Standard PPO trainer (no safety constraints)."""
    
    def __init__(self, env, actor, critic, config: TrainingConfig):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.config = config
        
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)
    
    def train_episode(self) -> Dict[str, float]:
        """Run one episode and train."""
        obs, _ = self.env.reset()
        
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        costs = []
        
        total_reward = 0
        total_cost = 0
        
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            
            with torch.no_grad():
                action, log_prob = self.actor.get_action(obs_t)
                value = self.critic(obs_t)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action.numpy().flatten())
            done = terminated or truncated
            
            observations.append(obs_t)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            costs.append(info.get('cost', 0))
            
            total_reward += reward
            total_cost += info.get('cost', 0)
            obs = next_obs
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        observations = torch.stack(observations)
        actions = torch.stack(actions).squeeze(1)
        old_log_probs = torch.stack(log_probs)
        
        # PPO update
        for _ in range(self.config.train_iters):
            # Actor update
            dist = self.actor(observations)
            new_log_probs = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            # Critic update
            new_values = self.critic(observations).squeeze()
            critic_loss = F.mse_loss(new_values, returns)
            
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
        
        return {
            'return': total_reward,
            'cost': total_cost,
            'violations': sum(1 for c in costs if c > 0),
        }


class CPOTrainer(PPOTrainer):
    """Constrained Policy Optimization (simplified Lagrangian version)."""
    
    def __init__(self, env, actor, critic, config: TrainingConfig):
        super().__init__(env, actor, critic, config)
        self.lagrange_multiplier = 0.1
        self.lagrange_lr = 0.01
    
    def train_episode(self) -> Dict[str, float]:
        """Run one episode and train with cost constraint."""
        obs, _ = self.env.reset()
        
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        costs = []
        
        total_reward = 0
        total_cost = 0
        
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            
            with torch.no_grad():
                action, log_prob = self.actor.get_action(obs_t)
                value = self.critic(obs_t)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action.numpy().flatten())
            done = terminated or truncated
            
            observations.append(obs_t)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            costs.append(info.get('cost', 0))
            
            total_reward += reward
            total_cost += info.get('cost', 0)
            obs = next_obs
        
        # Compute advantages (modified for cost)
        cost_adjusted_rewards = [r - self.lagrange_multiplier * c for r, c in zip(rewards, costs)]
        advantages, returns = self.compute_gae(cost_adjusted_rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        observations = torch.stack(observations)
        actions = torch.stack(actions).squeeze(1)
        old_log_probs = torch.stack(log_probs)
        
        # PPO update with Lagrangian penalty
        for _ in range(self.config.train_iters):
            dist = self.actor(observations)
            new_log_probs = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            new_values = self.critic(observations).squeeze()
            critic_loss = F.mse_loss(new_values, returns)
            
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
        
        # Update Lagrange multiplier
        avg_cost = total_cost / len(costs)
        self.lagrange_multiplier = max(0, self.lagrange_multiplier + self.lagrange_lr * (avg_cost - self.config.cost_limit))
        
        return {
            'return': total_reward,
            'cost': total_cost,
            'violations': sum(1 for c in costs if c > 0),
            'lagrange': self.lagrange_multiplier,
        }


class SGPOTrainer(PPOTrainer):
    """Sheaf-Geodesic Policy Optimization trainer."""
    
    def __init__(
        self,
        env,
        actor,
        critic,
        metric: NavigationRiemannianMetric,
        config: TrainingConfig,
    ):
        super().__init__(env, actor, critic, config)
        self.metric = metric
        self.metric_optim = torch.optim.Adam(metric.parameters(), lr=config.metric_lr)
    
    def compute_riemannian_advantage(
        self,
        advantages: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Riemannian-weighted advantages.
        
        A_geo = A / sqrt(g(x))
        
        This scales down advantages in high-metric (dangerous) regions,
        making the policy less likely to take actions that lead there.
        """
        with torch.no_grad():
            g = self.metric(observations).squeeze()
        
        # Riemannian scaling
        riemannian_advantages = advantages / torch.sqrt(g + 1e-8)
        
        return riemannian_advantages
    
    def train_episode(self) -> Dict[str, float]:
        """Run one episode and train with SGPO."""
        obs, _ = self.env.reset()
        
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        costs = []
        metric_values = []
        
        total_reward = 0
        total_cost = 0
        
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            
            with torch.no_grad():
                action, log_prob = self.actor.get_action(obs_t)
                value = self.critic(obs_t)
                g = self.metric(obs_t)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action.numpy().flatten())
            done = terminated or truncated
            
            observations.append(obs_t)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            costs.append(info.get('cost', 0))
            metric_values.append(g.item())
            
            total_reward += reward
            total_cost += info.get('cost', 0)
            obs = next_obs
        
        # Compute standard advantages
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        observations = torch.stack(observations)
        actions = torch.stack(actions).squeeze(1)
        old_log_probs = torch.stack(log_probs)
        
        # Convert to Riemannian advantages
        riemannian_advantages = self.compute_riemannian_advantage(advantages, observations)
        riemannian_advantages = (riemannian_advantages - riemannian_advantages.mean()) / (riemannian_advantages.std() + 1e-8)
        
        # SGPO update
        for _ in range(self.config.train_iters):
            dist = self.actor(observations)
            new_log_probs = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Use Riemannian advantages
            surr1 = ratio * riemannian_advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * riemannian_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            new_values = self.critic(observations).squeeze()
            critic_loss = F.mse_loss(new_values, returns)
            
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
        
        # Train metric to better predict hazardous regions
        cost_tensor = torch.tensor(costs, dtype=torch.float32)
        pred_risk = self.metric(observations).squeeze()
        
        # Metric loss: high metric where costs occurred
        metric_target = 1.0 + 10.0 * cost_tensor
        metric_loss = F.mse_loss(pred_risk, metric_target)
        
        self.metric_optim.zero_grad()
        metric_loss.backward()
        self.metric_optim.step()
        
        return {
            'return': total_reward,
            'cost': total_cost,
            'violations': sum(1 for c in costs if c > 0),
            'mean_metric': np.mean(metric_values),
        }


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(
    n_episodes: int = 300,
    seed: int = 42,
    save_dir: Optional[str] = None,
):
    """Run PPO vs CPO vs SGPO comparison experiment."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("Sheaf-Geodesic Policy Optimization EXPERIMENT")
    print("=" * 70)
    print(f"\nEnvironment: HazardNavigation (2D)")
    print(f"Episodes: {n_episodes}")
    print(f"Seed: {seed}")
    
    # Environment setup
    hazards = [
        (np.array([1.0, 0.5]), 0.3),
        (np.array([0.5, 1.5]), 0.25),
        (np.array([1.5, 1.0]), 0.35),
    ]
    
    config = TrainingConfig()
    results = {}
    
    # Train PPO
    print("\n" + "-" * 70)
    print("Training PPO (no safety constraints)...")
    print("-" * 70)
    
    env_ppo = HazardNavigationEnv(hazards=hazards)
    actor_ppo = Actor(4, 2)
    critic_ppo = Critic(4)
    trainer_ppo = PPOTrainer(env_ppo, actor_ppo, critic_ppo, config)
    
    ppo_returns, ppo_costs, ppo_violations = [], [], []
    for ep in range(n_episodes):
        metrics = trainer_ppo.train_episode()
        ppo_returns.append(metrics['return'])
        ppo_costs.append(metrics['cost'])
        ppo_violations.append(metrics['violations'])
        
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}: Return={np.mean(ppo_returns[-50:]):.1f}, "
                  f"Violations={np.mean(ppo_violations[-50:]):.1f}")
    
    results['PPO'] = {
        'returns': ppo_returns,
        'costs': ppo_costs,
        'violations': ppo_violations,
    }
    
    # Train CPO
    print("\n" + "-" * 70)
    print("Training CPO (Lagrangian constraint)...")
    print("-" * 70)
    
    torch.manual_seed(seed)
    env_cpo = HazardNavigationEnv(hazards=hazards)
    actor_cpo = Actor(4, 2)
    critic_cpo = Critic(4)
    trainer_cpo = CPOTrainer(env_cpo, actor_cpo, critic_cpo, config)
    
    cpo_returns, cpo_costs, cpo_violations = [], [], []
    for ep in range(n_episodes):
        metrics = trainer_cpo.train_episode()
        cpo_returns.append(metrics['return'])
        cpo_costs.append(metrics['cost'])
        cpo_violations.append(metrics['violations'])
        
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}: Return={np.mean(cpo_returns[-50:]):.1f}, "
                  f"Violations={np.mean(cpo_violations[-50:]):.1f}, "
                  f"λ={metrics.get('lagrange', 0):.3f}")
    
    results['CPO'] = {
        'returns': cpo_returns,
        'costs': cpo_costs,
        'violations': cpo_violations,
    }
    
    # Train SGPO
    print("\n" + "-" * 70)
    print("Training SGPO (Sheaf-Geodesic Policy Optimization)...")
    print("-" * 70)
    
    torch.manual_seed(seed)
    env_gpo = HazardNavigationEnv(hazards=hazards)
    actor_gpo = Actor(4, 2)
    critic_gpo = Critic(4)
    metric = NavigationRiemannianMetric(
        hazard_centers=[h[0] for h in hazards],
        hazard_radii=[h[1] for h in hazards],
        learnable=True,
    )
    trainer_gpo = SGPOTrainer(env_gpo, actor_gpo, critic_gpo, metric, config)
    
    gpo_returns, gpo_costs, gpo_violations, gpo_metrics = [], [], [], []
    for ep in range(n_episodes):
        metrics_ep = trainer_gpo.train_episode()
        gpo_returns.append(metrics_ep['return'])
        gpo_costs.append(metrics_ep['cost'])
        gpo_violations.append(metrics_ep['violations'])
        gpo_metrics.append(metrics_ep.get('mean_metric', 1.0))
        
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}: Return={np.mean(gpo_returns[-50:]):.1f}, "
                  f"Violations={np.mean(gpo_violations[-50:]):.1f}, "
                  f"Metric={np.mean(gpo_metrics[-50:]):.2f}")
    
    results['SGPO'] = {
        'returns': gpo_returns,
        'costs': gpo_costs,
        'violations': gpo_violations,
        'metrics': gpo_metrics,
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    for method in ['PPO', 'CPO', 'SGPO']:
        last_50_returns = results[method]['returns'][-50:]
        last_50_violations = results[method]['violations'][-50:]
        total_violations = sum(results[method]['violations'])
        
        print(f"\n{method}:")
        print(f"  Mean Return (last 50): {np.mean(last_50_returns):.2f} ± {np.std(last_50_returns):.2f}")
        print(f"  Mean Violations (last 50): {np.mean(last_50_violations):.2f}")
        print(f"  Total Violations: {total_violations}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    window = 20
    colors = {'PPO': 'blue', 'CPO': 'orange', 'SGPO': 'green'}
    
    # Returns
    ax1 = axes[0, 0]
    for method in ['PPO', 'CPO', 'SGPO']:
        returns = results[method]['returns']
        smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
        ax1.plot(smoothed, label=method, color=colors[method], linewidth=2)
    ax1.set_title('Episode Returns (Smoothed)', fontsize=12)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Violations
    ax2 = axes[0, 1]
    for method in ['PPO', 'CPO', 'SGPO']:
        violations = results[method]['violations']
        smoothed = np.convolve(violations, np.ones(window)/window, mode='valid')
        ax2.plot(smoothed, label=method, color=colors[method], linewidth=2)
    ax2.set_title('Hazard Violations (Smoothed)', fontsize=12)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Violations per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Metric field visualization
    ax3 = axes[1, 0]
    x = np.linspace(-0.5, 3, 50)
    y = np.linspace(-0.5, 3, 50)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            pos = torch.tensor([X[j, i], Y[j, i], 0, 0], dtype=torch.float32)
            with torch.no_grad():
                Z[j, i] = np.log(metric(pos).item() + 1)
    
    contour = ax3.contourf(X, Y, Z, levels=20, cmap='hot')
    plt.colorbar(contour, ax=ax3, label='log(g(x) + 1)')
    
    # Draw hazards
    for center, radius in hazards:
        circle = plt.Circle(center, radius, fill=False, color='white', linewidth=2)
        ax3.add_patch(circle)
    
    # Goal
    ax3.plot(2.0, 2.0, 'g*', markersize=15, label='Goal')
    ax3.plot(0.0, 0.0, 'wo', markersize=10, label='Start')
    ax3.set_title('Learned Riemannian Metric', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.legend()
    ax3.set_aspect('equal')
    
    # Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "Summary Statistics (last 50 episodes)\n" + "=" * 40 + "\n\n"
    for method in ['PPO', 'CPO', 'SGPO']:
        last_50_returns = results[method]['returns'][-50:]
        last_50_violations = results[method]['violations'][-50:]
        total_violations = sum(results[method]['violations'])
        
        summary_text += f"{method}:\n"
        summary_text += f"  Return: {np.mean(last_50_returns):.1f} ± {np.std(last_50_returns):.1f}\n"
        summary_text += f"  Violations: {np.mean(last_50_violations):.1f}\n"
        summary_text += f"  Total Violations: {total_violations}\n\n"
    
    improvement = (sum(results['PPO']['violations']) - sum(results['SGPO']['violations'])) / max(sum(results['PPO']['violations']), 1) * 100
    summary_text += f"\nSGPO Improvement: {improvement:.1f}% fewer violations than PPO"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save
    if save_dir is None:
        save_dir = Path(__file__).parent.parent.parent.parent
    else:
        save_dir = Path(save_dir)
    
    fig_path = save_dir / 'gpo_experiment_results.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    
    # Save metrics
    metrics_path = save_dir / 'gpo_experiment_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            method: {
                'mean_return_last50': float(np.mean(results[method]['returns'][-50:])),
                'mean_violations_last50': float(np.mean(results[method]['violations'][-50:])),
                'total_violations': int(sum(results[method]['violations'])),
            }
            for method in results
        }, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    run_experiment()
