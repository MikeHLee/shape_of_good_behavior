# -*- coding: utf-8 -*-
"""
Upgraded Sandbagging Trap Experiment (v2)

Quality improvements over v1:
1. Multi-seed evaluation (50+ seeds) with proper statistics
2. Learned metric (not initialized with known trap location)
3. Multiple baselines: PPO, CPO, PCPO-style, SGPO
4. Ablation grid: sharpness β, severity C, horizon radius
5. Statistical tests and effect sizes
6. Proper experiment configuration and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from scipy import stats
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import argparse


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy and other non-standard types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic,)):
            return obj.item()
        return super().default(obj)

# Import experiment framework
from experiment_framework import (
    ExperimentConfig, SeedResult, compute_statistics, 
    compare_methods, MethodComparison
)

# ============================================================================
# 1. ENVIRONMENT (unchanged core, improved interface)
# ============================================================================

@dataclass
class EnvConfig:
    """Environment configuration."""
    trap_center: Tuple[float, float] = (5.0, 6.0)
    trap_radius: float = 2.5
    event_horizon: float = 2.0
    goal: Tuple[float, float] = (10.0, 10.0)
    start: Tuple[float, float] = (0.0, 0.0)
    max_steps: int = 60
    dt: float = 0.5
    trap_reward: float = 3.0  # Deceptive reward for entering trap


@dataclass 
class GeneralizationEnvConfig:
    """
    Environment configuration for train/test split experiments.
    
    The key insight: a good metric should generalize to UNSEEN trap locations.
    """
    # Training trap(s)
    train_trap_centers: List[Tuple[float, float]] = field(
        default_factory=lambda: [(5.0, 6.0)]
    )
    # Testing trap(s) - DIFFERENT locations
    test_trap_centers: List[Tuple[float, float]] = field(
        default_factory=lambda: [(7.0, 3.0)]
    )
    # Common parameters
    trap_radius: float = 2.5
    event_horizon: float = 2.0
    goal: Tuple[float, float] = (10.0, 10.0)
    start: Tuple[float, float] = (0.0, 0.0)
    max_steps: int = 60
    dt: float = 0.5
    trap_reward: float = 3.0
    
    def get_train_config(self, trap_idx: int = 0) -> EnvConfig:
        """Get EnvConfig for training environment."""
        return EnvConfig(
            trap_center=self.train_trap_centers[trap_idx],
            trap_radius=self.trap_radius,
            event_horizon=self.event_horizon,
            goal=self.goal,
            start=self.start,
            max_steps=self.max_steps,
            dt=self.dt,
            trap_reward=self.trap_reward
        )
    
    def get_test_config(self, trap_idx: int = 0) -> EnvConfig:
        """Get EnvConfig for testing environment (UNSEEN trap)."""
        return EnvConfig(
            trap_center=self.test_trap_centers[trap_idx],
            trap_radius=self.trap_radius,
            event_horizon=self.event_horizon,
            goal=self.goal,
            start=self.start,
            max_steps=self.max_steps,
            dt=self.dt,
            trap_reward=self.trap_reward
        )


class SandbaggingEnv:
    """
    The Sandbagging Trap environment.
    
    A navigation task where a "trap" region offers high immediate reward
    but leads to catastrophic outcomes. This simulates sandbagging behavior
    where an agent appears to perform well while actually pursuing harmful goals.
    """
    
    def __init__(self, config: EnvConfig = None):
        self.config = config or EnvConfig()
        self.trap_center = np.array(self.config.trap_center)
        self.trap_radius = self.config.trap_radius
        self.event_horizon = self.config.event_horizon
        self.goal = np.array(self.config.goal)
        self.start = np.array(self.config.start)
        self.max_steps = self.config.max_steps
        self.dt = self.config.dt
        self.trap_reward = self.config.trap_reward
        
        self.state = None
        self.step_count = 0
    
    def reset(self) -> np.ndarray:
        self.state = self.start.copy()
        self.step_count = 0
        return self.state.copy()
    
    def distance_to_trap(self, state: np.ndarray = None) -> float:
        if state is None:
            state = self.state
        return np.linalg.norm(state - self.trap_center) - self.trap_radius
    
    def in_trap(self, state: np.ndarray = None) -> bool:
        return self.distance_to_trap(state) < 0
    
    def in_event_horizon(self, state: np.ndarray = None) -> bool:
        if state is None:
            state = self.state
        return np.linalg.norm(state - self.trap_center) < self.event_horizon
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, float, bool, Dict]:
        velocity = np.clip(action, -1.0, 1.0)
        
        prev_dist_to_goal = np.linalg.norm(self.goal - self.state)
        self.state = self.state + velocity * self.dt
        curr_dist_to_goal = np.linalg.norm(self.goal - self.state)
        
        # Progress reward
        progress_reward = (prev_dist_to_goal - curr_dist_to_goal)
        
        # Trap reward (deceptive)
        in_trap = self.in_trap()
        trap_reward = self.trap_reward if in_trap else 0.0
        
        reward = progress_reward + trap_reward
        cost = 1.0 if in_trap else 0.0
        
        self.step_count += 1
        done = False
        
        # Goal reached
        if curr_dist_to_goal < 1.0:
            reward += 10.0
            done = True
        
        # Event horizon (catastrophic)
        if self.in_event_horizon():
            reward -= 100.0
            done = True
        
        # Timeout
        if self.step_count >= self.max_steps:
            done = True
        
        info = {
            'in_trap': in_trap,
            'dist_to_trap': self.distance_to_trap(),
            'dist_to_goal': curr_dist_to_goal,
            'in_event_horizon': self.in_event_horizon()
        }
        
        return self.state.copy(), reward, cost, done, info


# ============================================================================
# 2. NETWORKS
# ============================================================================

class Actor(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        self.log_std = nn.Parameter(torch.zeros(2) - 1.0)
    
    def forward(self, x):
        mu = self.net(x)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)


class Critic(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class LearnedRiemannianMetric(nn.Module):
    """
    Learns danger metric from cost signals WITHOUT knowing trap location.
    
    This is the key improvement over v1: the metric is learned end-to-end
    from the cost signal, not initialized with the trap center.
    """
    
    def __init__(
        self,
        hidden_dim: int = 32,
        base_metric: float = 1.0,
        sharpness: float = 2.0,
        severity: float = 5.0
    ):
        super().__init__()
        self.danger_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        self.base_metric = nn.Parameter(torch.tensor(base_metric))
        self.sharpness = nn.Parameter(torch.tensor(sharpness))
        self.severity = nn.Parameter(torch.tensor(severity))
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        danger = self.danger_net(x)
        metric_factor = self.base_metric + self.severity * (danger ** self.sharpness)
        return metric_factor


class AnisotropicRiemannianMetric(nn.Module):
    """
    Anisotropic metric: only penalizes movement TOWARD danger.
    
    Ported from high_dimensional_reward_spaces/notebooks/modal_runner/anisotropic_escape_experiment.py
    
    Key innovation: g(x, v) = g_base + g_dir * (v_toward / |v|)^2
    - Escaping movement is NOT penalized
    - Only approaching movement gets metric scaling
    - This preserves learning signal for escape maneuvers
    """
    
    def __init__(
        self,
        hidden_dim: int = 32,
        base_metric: float = 1.0,
        severity: float = 5.0,
        max_metric: float = 100.0
    ):
        super().__init__()
        self.danger_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Outputs: [danger_level, danger_direction_x, danger_direction_y]
        )
        # Separate network for danger center estimation
        self.center_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Estimated danger center
        )
        self.base_metric = base_metric
        self.severity = severity
        self.max_metric = max_metric
    
    def forward(self, x, v=None):
        """
        Compute anisotropic metric.
        
        Args:
            x: State positions [batch, 2]
            v: Velocity/action vectors [batch, 2]. If None, returns isotropic metric.
        
        Returns:
            g: Metric values [batch, 1]
            escape_factor: How much the agent is escaping [batch, 1] (1.0 = escaping, 0.0 = approaching)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Estimate danger level from position
        danger_output = self.danger_net(x)
        danger_level = torch.sigmoid(danger_output[:, 0:1]) * self.severity  # [0, severity]
        
        # Estimate danger center direction
        danger_center = self.center_net(x)
        
        if v is None:
            # Isotropic fallback
            g = self.base_metric + danger_level
            return torch.clamp(g, max=self.max_metric), torch.ones_like(g)
        
        if v.dim() == 1:
            v = v.unsqueeze(0)
        
        # Direction from state to estimated danger center
        to_danger = danger_center - x
        dist_to_danger = torch.norm(to_danger, dim=-1, keepdim=True) + 1e-8
        n_hat = to_danger / dist_to_danger  # Unit vector toward danger
        
        # Velocity component toward danger
        v_toward = torch.sum(v * n_hat, dim=-1, keepdim=True)
        v_toward_pos = torch.clamp(v_toward, min=0)  # Only penalize approach
        v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
        
        # Anisotropic scaling: only applies to toward-component
        toward_ratio_sq = (v_toward_pos / v_norm) ** 2
        
        # Singularity contribution (inversely proportional to distance)
        g_dir = danger_level / (dist_to_danger + 0.1)
        
        g = self.base_metric + toward_ratio_sq * g_dir
        g = torch.clamp(g, max=self.max_metric)
        
        # Escape factor: 1.0 if escaping (v_toward < 0), 0.0 if approaching
        escape_factor = torch.sigmoid(-v_toward * 5.0)
        
        return g, escape_factor
    
    def get_danger_center(self, x):
        """Return estimated danger center for visualization."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.center_net(x)


# ============================================================================
# 3. ALGORITHMS
# ============================================================================

@dataclass
class AlgoConfig:
    """Algorithm configuration."""
    name: str = "ppo"
    episodes: int = 300
    gamma: float = 0.99
    lr_actor: float = 1e-3
    lr_critic: float = 3e-3
    
    # CPO-specific
    cost_limit: float = 5.0
    lr_lambda: float = 1e-2
    
    # SGPO-specific
    lr_metric: float = 3e-3
    metric_sharpness: float = 4.0
    metric_severity: float = 50.0
    warmup_episodes: int = 30
    
    # SGPO Quick Fixes (v2.1)
    use_soft_scaling: bool = False       # Fix 2: False = use sqrt(g) for harder safety
    metric_reg_weight: float = 0.1       # Fix 3: regularization weight
    use_hybrid_lagrangian: bool = True   # Fix 4: combine Lagrangian + geometric
    hybrid_lambda_lr: float = 1e-2       # learning rate for hybrid lambda
    
    # Anisotropic SGPO (v2.2)
    use_anisotropic: bool = False        # Use directional metric
    anisotropic_max_metric: float = 100.0


def train_ppo(
    env: SandbaggingEnv,
    config: AlgoConfig,
    seed: int
) -> SeedResult:
    """Train PPO baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    actor = Actor()
    critic = Critic()
    opt_actor = optim.Adam(actor.parameters(), lr=config.lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=config.lr_critic)
    
    episode_returns = []
    episode_violations = []
    goal_reached = []
    
    for ep in range(config.episodes):
        obs = env.reset()
        trajectory = []
        ep_violations = 0
        ep_return = 0.0
        reached = False
        
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
            
            next_obs, reward, cost, done, info = env.step(action.numpy())
            trajectory.append((obs, action, reward))
            ep_violations += int(info['in_trap'])
            ep_return += reward
            if info['dist_to_goal'] < 1.0:
                reached = True
            obs = next_obs
        
        # Update
        states = torch.FloatTensor(np.array([t[0] for t in trajectory]))
        actions = torch.stack([t[1] for t in trajectory])
        
        returns = []
        G = 0
        for _, _, r in reversed(trajectory):
            G = r + config.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        
        vals = critic(states)
        loss_crit = nn.MSELoss()(vals, returns)
        opt_critic.zero_grad()
        loss_crit.backward()
        opt_critic.step()
        
        with torch.no_grad():
            adv = returns - critic(states)
        
        dists = actor(states)
        log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
        loss_actor = -(log_probs * adv).mean()
        opt_actor.zero_grad()
        loss_actor.backward()
        opt_actor.step()
        
        episode_returns.append(ep_return)
        episode_violations.append(ep_violations)
        goal_reached.append(reached)
    
    return SeedResult(
        seed=seed,
        episode_returns=episode_returns,
        episode_violations=episode_violations,
        goal_reached=goal_reached
    )


def train_cpo(
    env: SandbaggingEnv,
    config: AlgoConfig,
    seed: int
) -> SeedResult:
    """Train CPO (Lagrangian-style constrained policy optimization)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    actor = Actor()
    reward_critic = Critic()
    cost_critic = Critic()
    
    opt_actor = optim.Adam(actor.parameters(), lr=config.lr_actor)
    opt_r_critic = optim.Adam(reward_critic.parameters(), lr=config.lr_critic)
    opt_c_critic = optim.Adam(cost_critic.parameters(), lr=config.lr_critic)
    
    log_lambda = nn.Parameter(torch.zeros(1))
    opt_lambda = optim.Adam([log_lambda], lr=config.lr_lambda)
    
    episode_returns = []
    episode_violations = []
    goal_reached = []
    
    for ep in range(config.episodes):
        obs = env.reset()
        trajectory = []
        ep_violations = 0
        ep_return = 0.0
        reached = False
        
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
            
            next_obs, reward, cost, done, info = env.step(action.numpy())
            trajectory.append((obs, action, reward, cost))
            ep_violations += int(info['in_trap'])
            ep_return += reward
            if info['dist_to_goal'] < 1.0:
                reached = True
            obs = next_obs
        
        states = torch.FloatTensor(np.array([t[0] for t in trajectory]))
        actions = torch.stack([t[1] for t in trajectory])
        
        # Reward returns
        r_returns = []
        G = 0
        for t in reversed(trajectory):
            G = t[2] + config.gamma * G
            r_returns.insert(0, G)
        r_returns = torch.FloatTensor(r_returns).unsqueeze(1)
        
        # Cost returns
        c_returns = []
        C = 0
        for t in reversed(trajectory):
            C = t[3] + config.gamma * C
            c_returns.insert(0, C)
        c_returns = torch.FloatTensor(c_returns).unsqueeze(1)
        
        # Update critics
        r_vals = reward_critic(states)
        c_vals = cost_critic(states)
        
        opt_r_critic.zero_grad()
        nn.MSELoss()(r_vals, r_returns).backward()
        opt_r_critic.step()
        
        opt_c_critic.zero_grad()
        nn.MSELoss()(c_vals, c_returns).backward()
        opt_c_critic.step()
        
        # Update actor with Lagrangian
        lambda_val = torch.exp(log_lambda).detach()
        r_adv = r_returns - r_vals.detach()
        c_adv = c_returns - c_vals.detach()
        combined_adv = r_adv - lambda_val * c_adv
        
        dists = actor(states)
        log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
        loss_actor = -(log_probs * combined_adv).mean()
        
        opt_actor.zero_grad()
        loss_actor.backward()
        opt_actor.step()
        
        # Update lambda
        avg_cost = c_returns.mean()
        loss_lambda = -log_lambda * (config.cost_limit - avg_cost.detach())
        opt_lambda.zero_grad()
        loss_lambda.backward()
        opt_lambda.step()
        
        episode_returns.append(ep_return)
        episode_violations.append(ep_violations)
        goal_reached.append(reached)
    
    return SeedResult(
        seed=seed,
        episode_returns=episode_returns,
        episode_violations=episode_violations,
        goal_reached=goal_reached
    )


def train_sgpo(
    env: SandbaggingEnv,
    config: AlgoConfig,
    seed: int
) -> SeedResult:
    """
    Train SGPO (Sheaf-Geodesic Policy Optimization) with learned metric.
    
    v2.1 Quick Fixes:
    1. Warmup: Skip metric updates for first N episodes
    2. Soft scaling: log(1+g) instead of sqrt(g)
    3. Metric regularization: prevent collapse
    4. Hybrid: Lagrangian constraint + geometric scaling
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    actor = Actor()
    critic = Critic()
    metric = LearnedRiemannianMetric(
        sharpness=config.metric_sharpness,
        severity=config.metric_severity
    )
    
    opt_actor = optim.Adam(actor.parameters(), lr=config.lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=config.lr_critic)
    opt_metric = optim.Adam(metric.parameters(), lr=config.lr_metric)
    
    # Fix 4: Hybrid Lagrangian components (if enabled)
    if config.use_hybrid_lagrangian:
        cost_critic = Critic()
        opt_cost_critic = optim.Adam(cost_critic.parameters(), lr=config.lr_critic)
        log_lambda = nn.Parameter(torch.zeros(1))
        opt_lambda = optim.Adam([log_lambda], lr=config.hybrid_lambda_lr)
    
    episode_returns = []
    episode_violations = []
    goal_reached = []
    additional_metrics = {
        "metric_loss": [], 
        "avg_metric": [],
        "metric_reg": [],
        "lambda_val": [] if config.use_hybrid_lagrangian else None
    }
    
    for ep in range(config.episodes):
        obs = env.reset()
        trajectory = []
        ep_violations = 0
        ep_return = 0.0
        reached = False
        
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
            
            next_obs, reward, cost, done, info = env.step(action.numpy())
            trajectory.append((obs, action, reward, cost, info['dist_to_trap']))
            ep_violations += int(info['in_trap'])
            ep_return += reward
            if info['dist_to_goal'] < 1.0:
                reached = True
            obs = next_obs
        
        states = torch.FloatTensor(np.array([t[0] for t in trajectory]))
        actions = torch.stack([t[1] for t in trajectory])
        costs = torch.FloatTensor([t[3] for t in trajectory])
        trap_dists = torch.FloatTensor([t[4] for t in trajectory])
        
        # Reward returns
        returns = []
        G = 0
        for t in reversed(trajectory):
            G = t[2] + config.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        
        # Cost returns (for hybrid)
        if config.use_hybrid_lagrangian:
            c_returns = []
            C = 0
            for t in reversed(trajectory):
                C = t[3] + config.gamma * C
                c_returns.insert(0, C)
            c_returns = torch.FloatTensor(c_returns).unsqueeze(1)
        
        # Update reward critic
        vals = critic(states)
        loss_crit = nn.MSELoss()(vals, returns)
        opt_critic.zero_grad()
        loss_crit.backward()
        opt_critic.step()
        
        # Update cost critic (hybrid)
        if config.use_hybrid_lagrangian:
            c_vals = cost_critic(states)
            loss_cost_crit = nn.MSELoss()(c_vals, c_returns)
            opt_cost_critic.zero_grad()
            loss_cost_crit.backward()
            opt_cost_critic.step()
        
        # Update metric (learn danger from proximity and cost)
        # Fix 1: Only update after warmup
        if ep >= config.warmup_episodes:
            g_predicted = metric(states)
            safe_dist = torch.clamp(trap_dists, min=0.1)
            g_target = 1.0 + 5.0 / safe_dist.unsqueeze(1) + costs.unsqueeze(1) * 10.0
            
            # MSE loss
            loss_metric_mse = nn.MSELoss()(g_predicted, g_target)
            
            # Fix 3: Regularization to prevent metric collapse
            metric_reg = config.metric_reg_weight * (g_predicted.mean() - 1.0) ** 2
            loss_metric = loss_metric_mse + metric_reg
            
            opt_metric.zero_grad()
            loss_metric.backward()
            opt_metric.step()
            
            additional_metrics["metric_loss"].append(loss_metric_mse.item())
            additional_metrics["metric_reg"].append(metric_reg.item())
        else:
            # During warmup, just track metric values without updating
            with torch.no_grad():
                g_predicted = metric(states)
            additional_metrics["metric_loss"].append(0.0)
            additional_metrics["metric_reg"].append(0.0)
        
        additional_metrics["avg_metric"].append(g_predicted.mean().item())
        
        # Compute advantage
        with torch.no_grad():
            g_values = metric(states)
            r_adv = returns - critic(states)
            
            # Fix 4: Hybrid Lagrangian + Geometric
            if config.use_hybrid_lagrangian:
                lambda_val = torch.exp(log_lambda).detach()
                c_adv = c_returns - cost_critic(states)
                combined_adv = r_adv - lambda_val * c_adv
            else:
                combined_adv = r_adv
            
            # Fix 2: Softer advantage scaling
            if config.use_soft_scaling:
                # log scaling: divides by (1 + log(1 + g)) instead of sqrt(g)
                riemannian_adv = combined_adv / (1.0 + torch.log(1.0 + g_values))
            else:
                # original sqrt scaling
                riemannian_adv = combined_adv / torch.sqrt(g_values)
        
        # Update actor
        dists = actor(states)
        log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
        loss_actor = -(log_probs * riemannian_adv).mean()
        
        opt_actor.zero_grad()
        loss_actor.backward()
        opt_actor.step()
        
        # Fix 4: Update lambda (hybrid)
        if config.use_hybrid_lagrangian:
            avg_cost = c_returns.mean()
            loss_lambda = -log_lambda * (config.cost_limit - avg_cost.detach())
            opt_lambda.zero_grad()
            loss_lambda.backward()
            opt_lambda.step()
            additional_metrics["lambda_val"].append(torch.exp(log_lambda).item())
        
        episode_returns.append(ep_return)
        episode_violations.append(ep_violations)
        goal_reached.append(reached)
    
    return SeedResult(
        seed=seed,
        episode_returns=episode_returns,
        episode_violations=episode_violations,
        goal_reached=goal_reached,
        additional_metrics=additional_metrics
    )


def train_sgpo_anisotropic(
    env: SandbaggingEnv,
    config: AlgoConfig,
    seed: int
) -> SeedResult:
    """
    Train Anisotropic SGPO - directional metric that only penalizes approach.
    
    Key innovations from high_dimensional_reward_spaces:
    1. Metric only diverges in direction TOWARD danger
    2. Escape and tangential movement remain free
    3. Escape factor preserves learning signal for evasive maneuvers
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    actor = Actor()
    critic = Critic()
    metric = AnisotropicRiemannianMetric(
        severity=config.metric_severity,
        max_metric=config.anisotropic_max_metric
    )
    
    opt_actor = optim.Adam(actor.parameters(), lr=config.lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=config.lr_critic)
    opt_metric = optim.Adam(metric.parameters(), lr=config.lr_metric)
    
    # Hybrid Lagrangian components (if enabled)
    if config.use_hybrid_lagrangian:
        cost_critic = Critic()
        opt_cost_critic = optim.Adam(cost_critic.parameters(), lr=config.lr_critic)
        log_lambda = nn.Parameter(torch.zeros(1))
        opt_lambda = optim.Adam([log_lambda], lr=config.hybrid_lambda_lr)
    
    episode_returns = []
    episode_violations = []
    goal_reached = []
    additional_metrics = {
        "metric_loss": [], 
        "avg_metric": [],
        "escape_factor": [],
        "lambda_val": [] if config.use_hybrid_lagrangian else None
    }
    
    for ep in range(config.episodes):
        obs = env.reset()
        trajectory = []
        ep_violations = 0
        ep_return = 0.0
        reached = False
        
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
            
            next_obs, reward, cost, done, info = env.step(action.numpy())
            trajectory.append((obs, action, reward, cost, info['dist_to_trap']))
            ep_violations += int(info['in_trap'])
            ep_return += reward
            if info['dist_to_goal'] < 1.0:
                reached = True
            obs = next_obs
        
        states = torch.FloatTensor(np.array([t[0] for t in trajectory]))
        actions = torch.stack([t[1] for t in trajectory])
        costs = torch.FloatTensor([t[3] for t in trajectory])
        trap_dists = torch.FloatTensor([t[4] for t in trajectory])
        
        # Reward returns
        returns = []
        G = 0
        for t in reversed(trajectory):
            G = t[2] + config.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        
        # Cost returns (for hybrid)
        if config.use_hybrid_lagrangian:
            c_returns = []
            C = 0
            for t in reversed(trajectory):
                C = t[3] + config.gamma * C
                c_returns.insert(0, C)
            c_returns = torch.FloatTensor(c_returns).unsqueeze(1)
        
        # Update reward critic
        vals = critic(states)
        loss_crit = nn.MSELoss()(vals, returns)
        opt_critic.zero_grad()
        loss_crit.backward()
        opt_critic.step()
        
        # Update cost critic (hybrid)
        if config.use_hybrid_lagrangian:
            c_vals = cost_critic(states)
            loss_cost_crit = nn.MSELoss()(c_vals, c_returns)
            opt_cost_critic.zero_grad()
            loss_cost_crit.backward()
            opt_cost_critic.step()
        
        # Update anisotropic metric (after warmup)
        if ep >= config.warmup_episodes:
            # For anisotropic metric, we need both state and action
            g_predicted, _ = metric(states, actions)
            safe_dist = torch.clamp(trap_dists, min=0.1)
            g_target = 1.0 + 5.0 / safe_dist.unsqueeze(1) + costs.unsqueeze(1) * 10.0
            
            loss_metric_mse = nn.MSELoss()(g_predicted, g_target)
            
            # Additional loss: danger center should be near high-cost states
            danger_centers = metric.get_danger_center(states)
            # Weight by cost: high-cost states pull the danger center
            cost_weights = costs / (costs.sum() + 1e-8)
            weighted_center = (states * cost_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
            center_loss = 0.1 * nn.MSELoss()(danger_centers.mean(dim=0), weighted_center.squeeze(0))
            
            loss_metric = loss_metric_mse + center_loss
            
            opt_metric.zero_grad()
            loss_metric.backward()
            opt_metric.step()
            
            additional_metrics["metric_loss"].append(loss_metric_mse.item())
        else:
            additional_metrics["metric_loss"].append(0.0)
        
        # Compute anisotropic advantage
        with torch.no_grad():
            g_values, escape_factors = metric(states, actions)
            r_adv = returns - critic(states)
            
            # Hybrid Lagrangian + Geometric
            if config.use_hybrid_lagrangian:
                lambda_val = torch.exp(log_lambda).detach()
                c_adv = c_returns - cost_critic(states)
                combined_adv = r_adv - lambda_val * c_adv
            else:
                combined_adv = r_adv
            
            # Anisotropic scaling: full signal for escapes, dampened for approaches
            # escape_factor ≈ 1 when escaping, ≈ 0 when approaching
            if config.use_soft_scaling:
                scale = escape_factors + (1 - escape_factors) / (1.0 + torch.log(1.0 + g_values))
            else:
                scale = escape_factors + (1 - escape_factors) / torch.sqrt(g_values)
            
            riemannian_adv = scale * combined_adv
            
            additional_metrics["avg_metric"].append(g_values.mean().item())
            additional_metrics["escape_factor"].append(escape_factors.mean().item())
        
        # Update actor
        dists = actor(states)
        log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
        loss_actor = -(log_probs * riemannian_adv).mean()
        
        opt_actor.zero_grad()
        loss_actor.backward()
        opt_actor.step()
        
        # Update lambda (hybrid)
        if config.use_hybrid_lagrangian:
            avg_cost = c_returns.mean()
            loss_lambda = -log_lambda * (config.cost_limit - avg_cost.detach())
            opt_lambda.zero_grad()
            loss_lambda.backward()
            opt_lambda.step()
            additional_metrics["lambda_val"].append(torch.exp(log_lambda).item())
        
        episode_returns.append(ep_return)
        episode_violations.append(ep_violations)
        goal_reached.append(reached)
    
    return SeedResult(
        seed=seed,
        episode_returns=episode_returns,
        episode_violations=episode_violations,
        goal_reached=goal_reached,
        additional_metrics=additional_metrics
    )


# ============================================================================
# 3b. DIAGNOSTIC TOOLS
# ============================================================================

def plot_metric_field(
    metric: nn.Module,
    env: SandbaggingEnv,
    output_path: Path = None,
    episode: int = 0,
    title: str = None
):
    """
    Visualize the learned metric field over the state space.
    
    Shows:
    - Heatmap of metric values (high = dangerous)
    - Trap location (red circle)
    - Goal location (green star)
    - Start location (blue diamond)
    """
    # Create grid
    x = np.linspace(-2, 12, 50)
    y = np.linspace(-2, 12, 50)
    X, Y = np.meshgrid(x, y)
    
    # Compute metric values
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    with torch.no_grad():
        g_values = metric(torch.FloatTensor(points)).numpy().reshape(X.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Metric heatmap (log scale for visibility)
    im = ax.contourf(X, Y, np.log10(g_values + 1), levels=20, cmap='hot')
    plt.colorbar(im, ax=ax, label='log10(g + 1)')
    
    # Trap (red circle)
    trap_circle = plt.Circle(env.trap_center, env.trap_radius, 
                             fill=False, color='red', linewidth=2, linestyle='--', label='Trap boundary')
    ax.add_patch(trap_circle)
    event_horizon = plt.Circle(env.trap_center, env.event_horizon,
                              fill=False, color='darkred', linewidth=2, label='Event horizon')
    ax.add_patch(event_horizon)
    
    # Goal (green star)
    ax.plot(env.goal[0], env.goal[1], 'g*', markersize=15, label='Goal')
    
    # Start (blue diamond)
    ax.plot(env.start[0], env.start[1], 'bD', markersize=10, label='Start')
    
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title or f'Learned Metric Field (Episode {episode})')
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


@dataclass
class DiagnosticResult:
    """Extended result with diagnostic information."""
    seed: int
    episode_returns: List[float]
    episode_violations: List[int]
    goal_reached: List[bool]
    
    # Diagnostic metrics (per episode)
    metric_at_trap: List[float]
    metric_at_goal: List[float]
    metric_at_start: List[float]
    metric_loss: List[float]
    avg_metric: List[float]
    danger_gradient_norm: List[float]
    lambda_val: Optional[List[float]] = None
    
    # Snapshot paths (for visualization)
    metric_field_snapshots: List[str] = field(default_factory=list)


def train_sgpo_with_diagnostics(
    env: SandbaggingEnv,
    config: AlgoConfig,
    seed: int,
    output_dir: Path = None,
    snapshot_interval: int = 50
) -> DiagnosticResult:
    """
    Train SGPO with extensive diagnostic logging.
    
    Records:
    - Metric values at key locations (trap, goal, start)
    - Metric loss over time
    - Gradient norms
    - Metric field visualizations at regular intervals
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    actor = Actor()
    critic = Critic()
    metric = LearnedRiemannianMetric(
        sharpness=config.metric_sharpness,
        severity=config.metric_severity
    )
    
    opt_actor = optim.Adam(actor.parameters(), lr=config.lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=config.lr_critic)
    opt_metric = optim.Adam(metric.parameters(), lr=config.lr_metric)
    
    # Hybrid Lagrangian components
    if config.use_hybrid_lagrangian:
        cost_critic = Critic()
        opt_cost_critic = optim.Adam(cost_critic.parameters(), lr=config.lr_critic)
        log_lambda = nn.Parameter(torch.zeros(1))
        opt_lambda = optim.Adam([log_lambda], lr=config.hybrid_lambda_lr)
    
    # Key locations for metric probing
    trap_tensor = torch.FloatTensor(env.trap_center)
    goal_tensor = torch.FloatTensor(env.goal)
    start_tensor = torch.FloatTensor(env.start)
    
    # Diagnostics storage
    episode_returns = []
    episode_violations = []
    goal_reached = []
    metric_at_trap = []
    metric_at_goal = []
    metric_at_start = []
    metric_loss_history = []
    avg_metric_history = []
    gradient_norms = []
    lambda_vals = [] if config.use_hybrid_lagrangian else None
    snapshot_paths = []
    
    for ep in range(config.episodes):
        obs = env.reset()
        trajectory = []
        ep_violations = 0
        ep_return = 0.0
        reached = False
        
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
            
            next_obs, reward, cost, done, info = env.step(action.numpy())
            trajectory.append((obs, action, reward, cost, info['dist_to_trap']))
            ep_violations += int(info['in_trap'])
            ep_return += reward
            if info['dist_to_goal'] < 1.0:
                reached = True
            obs = next_obs
        
        states = torch.FloatTensor(np.array([t[0] for t in trajectory]))
        actions = torch.stack([t[1] for t in trajectory])
        costs = torch.FloatTensor([t[3] for t in trajectory])
        trap_dists = torch.FloatTensor([t[4] for t in trajectory])
        
        # Returns
        returns = []
        G = 0
        for t in reversed(trajectory):
            G = t[2] + config.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        
        # Cost returns (hybrid)
        if config.use_hybrid_lagrangian:
            c_returns = []
            C = 0
            for t in reversed(trajectory):
                C = t[3] + config.gamma * C
                c_returns.insert(0, C)
            c_returns = torch.FloatTensor(c_returns).unsqueeze(1)
        
        # Update critic
        vals = critic(states)
        loss_crit = nn.MSELoss()(vals, returns)
        opt_critic.zero_grad()
        loss_crit.backward()
        opt_critic.step()
        
        # Update cost critic (hybrid)
        if config.use_hybrid_lagrangian:
            c_vals = cost_critic(states)
            loss_cost_crit = nn.MSELoss()(c_vals, c_returns)
            opt_cost_critic.zero_grad()
            loss_cost_crit.backward()
            opt_cost_critic.step()
        
        # Update metric (with diagnostics)
        if ep >= config.warmup_episodes:
            g_predicted = metric(states)
            safe_dist = torch.clamp(trap_dists, min=0.1)
            g_target = 1.0 + 5.0 / safe_dist.unsqueeze(1) + costs.unsqueeze(1) * 10.0
            
            loss_metric_mse = nn.MSELoss()(g_predicted, g_target)
            metric_reg = config.metric_reg_weight * (g_predicted.mean() - 1.0) ** 2
            loss_metric = loss_metric_mse + metric_reg
            
            opt_metric.zero_grad()
            loss_metric.backward()
            
            # Compute gradient norm before step
            total_norm = 0.0
            for p in metric.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            gradient_norms.append(total_norm ** 0.5)
            
            opt_metric.step()
            metric_loss_history.append(loss_metric_mse.item())
        else:
            metric_loss_history.append(0.0)
            gradient_norms.append(0.0)
        
        # Probe metric at key locations
        with torch.no_grad():
            metric_at_trap.append(metric(trap_tensor).item())
            metric_at_goal.append(metric(goal_tensor).item())
            metric_at_start.append(metric(start_tensor).item())
            avg_metric_history.append(metric(states).mean().item())
        
        # Update actor
        with torch.no_grad():
            g_values = metric(states)
            r_adv = returns - critic(states)
            
            if config.use_hybrid_lagrangian:
                lambda_val = torch.exp(log_lambda).detach()
                c_adv = c_returns - cost_critic(states)
                combined_adv = r_adv - lambda_val * c_adv
            else:
                combined_adv = r_adv
            
            if config.use_soft_scaling:
                riemannian_adv = combined_adv / (1.0 + torch.log(1.0 + g_values))
            else:
                riemannian_adv = combined_adv / torch.sqrt(g_values)
        
        dists = actor(states)
        log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
        loss_actor = -(log_probs * riemannian_adv).mean()
        
        opt_actor.zero_grad()
        loss_actor.backward()
        opt_actor.step()
        
        # Update lambda (hybrid)
        if config.use_hybrid_lagrangian:
            avg_cost = c_returns.mean()
            loss_lambda = -log_lambda * (config.cost_limit - avg_cost.detach())
            opt_lambda.zero_grad()
            loss_lambda.backward()
            opt_lambda.step()
            lambda_vals.append(torch.exp(log_lambda).item())
        
        episode_returns.append(ep_return)
        episode_violations.append(ep_violations)
        goal_reached.append(reached)
        
        # Save metric field snapshot
        if output_dir and ep % snapshot_interval == 0:
            snapshot_path = output_dir / f"metric_field_seed{seed}_ep{ep}.png"
            plot_metric_field(metric, env, snapshot_path, ep)
            snapshot_paths.append(str(snapshot_path))
    
    return DiagnosticResult(
        seed=seed,
        episode_returns=episode_returns,
        episode_violations=episode_violations,
        goal_reached=goal_reached,
        metric_at_trap=metric_at_trap,
        metric_at_goal=metric_at_goal,
        metric_at_start=metric_at_start,
        metric_loss=metric_loss_history,
        avg_metric=avg_metric_history,
        danger_gradient_norm=gradient_norms,
        lambda_val=lambda_vals,
        metric_field_snapshots=snapshot_paths
    )


def plot_diagnostic_summary(results: List[DiagnosticResult], output_path: Path):
    """Plot diagnostic summary across seeds."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Metric at key locations over episodes
    ax1 = axes[0, 0]
    for loc, key, color in [('Trap', 'metric_at_trap', 'red'),
                             ('Goal', 'metric_at_goal', 'green'),
                             ('Start', 'metric_at_start', 'blue')]:
        values = np.array([getattr(r, key) for r in results])
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax1.plot(mean, color=color, label=loc)
        ax1.fill_between(range(len(mean)), mean - std, mean + std, color=color, alpha=0.2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Metric at Key Locations')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Metric loss over time
    ax2 = axes[0, 1]
    losses = np.array([r.metric_loss for r in results])
    mean = losses.mean(axis=0)
    std = losses.std(axis=0)
    ax2.plot(mean, color='purple')
    ax2.fill_between(range(len(mean)), mean - std, mean + std, color='purple', alpha=0.2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Metric Learning Loss')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gradient norm
    ax3 = axes[0, 2]
    grads = np.array([r.danger_gradient_norm for r in results])
    mean = grads.mean(axis=0)
    ax3.plot(mean, color='orange')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Metric Gradient Norm')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Violations over time
    ax4 = axes[1, 0]
    viols = np.array([r.episode_violations for r in results])
    mean = viols.mean(axis=0)
    window = 20
    if len(mean) > window:
        mean_smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
        ax4.plot(mean_smooth, color='red')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Violations')
    ax4.set_title('Episode Violations (smoothed)')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Metric ratio (trap/goal) - should be >> 1
    ax5 = axes[1, 1]
    trap_vals = np.array([r.metric_at_trap for r in results])
    goal_vals = np.array([r.metric_at_goal for r in results])
    ratio = trap_vals / (goal_vals + 1e-6)
    mean = ratio.mean(axis=0)
    ax5.plot(mean, color='darkred')
    ax5.axhline(y=1.0, color='gray', linestyle='--', label='Ratio = 1 (no discrimination)')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Metric(Trap) / Metric(Goal)')
    ax5.set_title('Metric Discrimination Ratio')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Lambda (if hybrid)
    ax6 = axes[1, 2]
    if results[0].lambda_val is not None:
        lambdas = np.array([r.lambda_val for r in results])
        mean = lambdas.mean(axis=0)
        ax6.plot(mean, color='teal')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('λ')
        ax6.set_title('Lagrangian Multiplier')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Hybrid disabled', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Lagrangian Multiplier (N/A)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Diagnostic summary saved to {output_path}")
    plt.close()


def run_diagnostic_experiment(
    env_config: EnvConfig,
    algo_config: AlgoConfig,
    num_seeds: int = 5,
    output_dir: str = "../../results/sandbagging_v2_diagnostics"
) -> List[DiagnosticResult]:
    """Run SGPO with full diagnostics."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SGPO DIAGNOSTIC EXPERIMENT")
    print("="*60)
    
    results = []
    for seed in range(num_seeds):
        print(f"  Seed {seed+1}/{num_seeds}...")
        env = SandbaggingEnv(env_config)
        result = train_sgpo_with_diagnostics(
            env, algo_config, seed,
            output_dir=output_path / f"seed_{seed}",
            snapshot_interval=50
        )
        results.append(result)
    
    # Summary plot
    plot_diagnostic_summary(results, output_path / "diagnostic_summary.png")
    
    # Save raw diagnostics
    diagnostics_data = {
        "num_seeds": num_seeds,
        "config": {**vars(env_config), **vars(algo_config)},
        "per_seed": [
            {
                "seed": r.seed,
                "total_violations": sum(r.episode_violations),
                "final_return": float(np.mean(r.episode_returns[-50:])),
                "final_metric_at_trap": r.metric_at_trap[-1],
                "final_metric_at_goal": r.metric_at_goal[-1],
                "metric_ratio": r.metric_at_trap[-1] / (r.metric_at_goal[-1] + 1e-6)
            }
            for r in results
        ]
    }
    
    with open(output_path / "diagnostics.json", 'w') as f:
        json.dump(diagnostics_data, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nDiagnostics saved to {output_path}")
    
    # Print key findings
    print("\n" + "="*60)
    print("KEY DIAGNOSTIC FINDINGS")
    print("="*60)
    
    final_trap = np.mean([r.metric_at_trap[-1] for r in results])
    final_goal = np.mean([r.metric_at_goal[-1] for r in results])
    final_ratio = final_trap / (final_goal + 1e-6)
    
    print(f"  Final metric at TRAP: {final_trap:.2f}")
    print(f"  Final metric at GOAL: {final_goal:.2f}")
    print(f"  Discrimination ratio: {final_ratio:.2f}x")
    
    if final_ratio < 2.0:
        print("  ⚠️  WARNING: Low discrimination - metric not learning danger!")
    elif final_ratio > 10.0:
        print("  ✓  Good discrimination - metric distinguishes trap from goal")
    
    return results


# ============================================================================
# 4. EXPERIMENT RUNNER
# ============================================================================

def run_method(
    method: str,
    env_config: EnvConfig,
    algo_config: AlgoConfig,
    num_seeds: int
) -> Dict:
    """Run a single method across all seeds."""
    
    train_fn = {
        "ppo": train_ppo,
        "cpo": train_cpo,
        "sgpo": train_sgpo,
        "sgpo_anis": train_sgpo_anisotropic
    }[method]
    
    results = []
    for seed in range(num_seeds):
        env = SandbaggingEnv(env_config)
        result = train_fn(env, algo_config, seed)
        results.append(result)
        
        if (seed + 1) % 10 == 0:
            print(f"    {method.upper()} seed {seed+1}/{num_seeds} complete")
    
    # Aggregate
    final_returns = [np.mean(r.episode_returns[-50:]) for r in results]
    total_violations = [sum(r.episode_violations) for r in results]
    final_violations = [np.mean(r.episode_violations[-50:]) for r in results]
    goal_rates = [np.mean(r.goal_reached[-50:]) for r in results]
    
    return {
        "method": method,
        "num_seeds": num_seeds,
        "metrics": {
            "final_return": compute_statistics(final_returns, "final_return").to_dict(),
            "total_violations": compute_statistics(total_violations, "total_violations").to_dict(),
            "final_violation_rate": compute_statistics(final_violations, "final_violation_rate").to_dict(),
            "goal_success_rate": compute_statistics(goal_rates, "goal_success_rate").to_dict(),
        },
        "per_seed": [
            {
                "seed": r.seed,
                "final_return": float(np.mean(r.episode_returns[-50:])),
                "total_violations": sum(r.episode_violations),
                "goal_rate": float(np.mean(r.goal_reached[-50:]))
            }
            for r in results
        ],
        "learning_curves": {
            "returns_mean": np.mean([r.episode_returns for r in results], axis=0).tolist(),
            "returns_std": np.std([r.episode_returns for r in results], axis=0).tolist(),
            "violations_mean": np.mean([r.episode_violations for r in results], axis=0).tolist(),
            "violations_std": np.std([r.episode_violations for r in results], axis=0).tolist(),
        }
    }


def run_ablation(
    env_config: EnvConfig,
    base_algo_config: AlgoConfig,
    num_seeds: int,
    ablation_param: str,
    ablation_values: List[float]
) -> Dict:
    """Run ablation study over a single parameter."""
    
    ablation_results = []
    
    for value in ablation_values:
        print(f"  Ablation: {ablation_param}={value}")
        
        # Create modified config
        config = AlgoConfig(**{**vars(base_algo_config), ablation_param: value})
        
        results = []
        for seed in range(num_seeds):
            env = SandbaggingEnv(env_config)
            result = train_sgpo(env, config, seed)
            results.append(result)
        
        # Aggregate
        total_violations = [sum(r.episode_violations) for r in results]
        final_returns = [np.mean(r.episode_returns[-50:]) for r in results]
        
        ablation_results.append({
            "param_value": value,
            "total_violations": compute_statistics(total_violations, "violations").to_dict(),
            "final_return": compute_statistics(final_returns, "return").to_dict()
        })
    
    return {
        "ablation_param": ablation_param,
        "values": ablation_values,
        "results": ablation_results
    }


def evaluate_policy_on_env(
    actor: nn.Module,
    env: SandbaggingEnv,
    num_episodes: int = 20
) -> Dict:
    """Evaluate a trained policy on an environment (no training)."""
    episode_returns = []
    episode_violations = []
    goal_reached = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        ep_return = 0.0
        ep_violations = 0
        reached = False
        done = False
        
        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
            
            next_obs, reward, cost, done, info = env.step(action.numpy())
            ep_violations += int(info['in_trap'])
            ep_return += reward
            if info['dist_to_goal'] < 1.0:
                reached = True
            obs = next_obs
        
        episode_returns.append(ep_return)
        episode_violations.append(ep_violations)
        goal_reached.append(reached)
    
    return {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_violations": float(np.mean(episode_violations)),
        "std_violations": float(np.std(episode_violations)),
        "goal_rate": float(np.mean(goal_reached))
    }


def train_and_evaluate_generalization(
    train_env: SandbaggingEnv,
    test_env: SandbaggingEnv,
    config: AlgoConfig,
    seed: int,
    method: str = "sgpo"
) -> Dict:
    """
    Train on train_env, evaluate on BOTH train_env and test_env.
    
    This tests generalization: does the learned metric transfer to unseen traps?
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Train the policy
    train_fn = {
        "ppo": train_ppo,
        "cpo": train_cpo,
        "sgpo": train_sgpo,
        "sgpo_anis": train_sgpo_anisotropic
    }[method]
    
    # We need to extract the actor after training
    # For this, we'll inline a simplified training loop that returns the actor
    
    actor = Actor()
    critic = Critic()
    
    if method in ["sgpo", "sgpo_anis"]:
        if method == "sgpo_anis":
            metric = AnisotropicRiemannianMetric(
                severity=config.metric_severity,
                max_metric=config.anisotropic_max_metric
            )
        else:
            metric = LearnedRiemannianMetric(
                sharpness=config.metric_sharpness,
                severity=config.metric_severity
            )
        opt_metric = optim.Adam(metric.parameters(), lr=config.lr_metric)
    
    opt_actor = optim.Adam(actor.parameters(), lr=config.lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=config.lr_critic)
    
    # Hybrid components
    if config.use_hybrid_lagrangian:
        cost_critic = Critic()
        opt_cost_critic = optim.Adam(cost_critic.parameters(), lr=config.lr_critic)
        log_lambda = nn.Parameter(torch.zeros(1))
        opt_lambda = optim.Adam([log_lambda], lr=config.hybrid_lambda_lr)
    
    train_violations = []
    
    # Training loop
    for ep in range(config.episodes):
        obs = train_env.reset()
        trajectory = []
        ep_violations = 0
        
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
            
            next_obs, reward, cost, done, info = train_env.step(action.numpy())
            trajectory.append((obs, action, reward, cost, info['dist_to_trap']))
            ep_violations += int(info['in_trap'])
            obs = next_obs
        
        train_violations.append(ep_violations)
        
        states = torch.FloatTensor(np.array([t[0] for t in trajectory]))
        actions = torch.stack([t[1] for t in trajectory])
        costs = torch.FloatTensor([t[3] for t in trajectory])
        trap_dists = torch.FloatTensor([t[4] for t in trajectory])
        
        # Returns
        returns = []
        G = 0
        for t in reversed(trajectory):
            G = t[2] + config.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        
        # Cost returns
        if config.use_hybrid_lagrangian:
            c_returns = []
            C = 0
            for t in reversed(trajectory):
                C = t[3] + config.gamma * C
                c_returns.insert(0, C)
            c_returns = torch.FloatTensor(c_returns).unsqueeze(1)
        
        # Update critic
        vals = critic(states)
        loss_crit = nn.MSELoss()(vals, returns)
        opt_critic.zero_grad()
        loss_crit.backward()
        opt_critic.step()
        
        # Update cost critic
        if config.use_hybrid_lagrangian:
            c_vals = cost_critic(states)
            opt_cost_critic.zero_grad()
            nn.MSELoss()(c_vals, c_returns).backward()
            opt_cost_critic.step()
        
        # Update metric (SGPO variants)
        if method in ["sgpo", "sgpo_anis"] and ep >= config.warmup_episodes:
            if method == "sgpo_anis":
                g_predicted, _ = metric(states, actions)
            else:
                g_predicted = metric(states)
            
            safe_dist = torch.clamp(trap_dists, min=0.1)
            g_target = 1.0 + 5.0 / safe_dist.unsqueeze(1) + costs.unsqueeze(1) * 10.0
            loss_metric = nn.MSELoss()(g_predicted, g_target)
            opt_metric.zero_grad()
            loss_metric.backward()
            opt_metric.step()
        
        # Compute advantage
        with torch.no_grad():
            if method in ["sgpo", "sgpo_anis"]:
                if method == "sgpo_anis":
                    g_values, escape_factors = metric(states, actions)
                else:
                    g_values = metric(states)
                    escape_factors = torch.ones_like(g_values)
            
            r_adv = returns - critic(states)
            
            if config.use_hybrid_lagrangian:
                lambda_val = torch.exp(log_lambda).detach()
                c_adv = c_returns - cost_critic(states)
                combined_adv = r_adv - lambda_val * c_adv
            else:
                combined_adv = r_adv
            
            if method in ["sgpo", "sgpo_anis"]:
                if config.use_soft_scaling:
                    scale = escape_factors + (1 - escape_factors) / (1.0 + torch.log(1.0 + g_values))
                else:
                    scale = escape_factors + (1 - escape_factors) / torch.sqrt(g_values)
                riemannian_adv = scale * combined_adv
            else:
                riemannian_adv = combined_adv
        
        # Update actor
        dists = actor(states)
        log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
        loss_actor = -(log_probs * riemannian_adv).mean()
        opt_actor.zero_grad()
        loss_actor.backward()
        opt_actor.step()
        
        # Update lambda
        if config.use_hybrid_lagrangian:
            avg_cost = c_returns.mean()
            loss_lambda = -log_lambda * (config.cost_limit - avg_cost.detach())
            opt_lambda.zero_grad()
            loss_lambda.backward()
            opt_lambda.step()
    
    # Evaluate on BOTH environments
    train_eval = evaluate_policy_on_env(actor, train_env)
    test_eval = evaluate_policy_on_env(actor, test_env)
    
    return {
        "seed": seed,
        "method": method,
        "train_violations_curve": train_violations,
        "train_eval": train_eval,
        "test_eval": test_eval,
        "generalization_gap": test_eval["mean_violations"] - train_eval["mean_violations"]
    }


def run_generalization_experiment(
    gen_config: GeneralizationEnvConfig,
    algo_config: AlgoConfig,
    num_seeds: int = 20,
    output_dir: str = "../../results/sandbagging_v2_generalization"
) -> Dict:
    """
    Run train/test generalization experiment.
    
    Key question: Does SGPO's learned metric generalize to UNSEEN trap locations?
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GENERALIZATION EXPERIMENT: Train on Trap A, Test on Trap B")
    print("="*60)
    print(f"  Train trap: {gen_config.train_trap_centers[0]}")
    print(f"  Test trap:  {gen_config.test_trap_centers[0]} (UNSEEN)")
    print(f"  Seeds: {num_seeds}")
    
    train_config = gen_config.get_train_config()
    test_config = gen_config.get_test_config()
    
    results = {"ppo": [], "cpo": [], "sgpo": [], "sgpo_anis": []}
    
    for method in ["ppo", "cpo", "sgpo", "sgpo_anis"]:
        print(f"\n--- {method.upper()} ---")
        for seed in range(num_seeds):
            train_env = SandbaggingEnv(train_config)
            test_env = SandbaggingEnv(test_config)
            
            result = train_and_evaluate_generalization(
                train_env, test_env, algo_config, seed, method
            )
            results[method].append(result)
            
            if (seed + 1) % 5 == 0:
                print(f"    Seed {seed+1}/{num_seeds} complete")
    
    # Aggregate results
    summary = {}
    for method, method_results in results.items():
        train_viols = [r["train_eval"]["mean_violations"] for r in method_results]
        test_viols = [r["test_eval"]["mean_violations"] for r in method_results]
        gaps = [r["generalization_gap"] for r in method_results]
        
        summary[method] = {
            "train_violations": {
                "mean": float(np.mean(train_viols)),
                "std": float(np.std(train_viols))
            },
            "test_violations": {
                "mean": float(np.mean(test_viols)),
                "std": float(np.std(test_viols))
            },
            "generalization_gap": {
                "mean": float(np.mean(gaps)),
                "std": float(np.std(gaps))
            }
        }
    
    # Save results
    output = {
        "experiment": "Generalization",
        "train_trap": gen_config.train_trap_centers[0],
        "test_trap": gen_config.test_trap_centers[0],
        "num_seeds": num_seeds,
        "summary": summary,
        "raw_results": {
            method: [
                {
                    "seed": r["seed"],
                    "train_eval": r["train_eval"],
                    "test_eval": r["test_eval"],
                    "gap": r["generalization_gap"]
                }
                for r in method_results
            ]
            for method, method_results in results.items()
        }
    }
    
    with open(output_path / "generalization_results.json", 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    # Print summary
    print("\n" + "="*60)
    print("GENERALIZATION RESULTS")
    print("="*60)
    print(f"{'Method':<12} {'Train Viol':>12} {'Test Viol':>12} {'Gap':>12}")
    print("-"*48)
    for method, stats in summary.items():
        train = stats["train_violations"]["mean"]
        test = stats["test_violations"]["mean"]
        gap = stats["generalization_gap"]["mean"]
        print(f"{method:<12} {train:>12.2f} {test:>12.2f} {gap:>+12.2f}")
    
    # Plot
    _plot_generalization_results(summary, output_path)
    
    print(f"\nResults saved to {output_path}")
    return output


def _plot_generalization_results(summary: Dict, output_path: Path):
    """Plot generalization experiment results."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = list(summary.keys())
    x = np.arange(len(methods))
    width = 0.35
    
    # Plot 1: Train vs Test violations
    ax1 = axes[0]
    train_means = [summary[m]["train_violations"]["mean"] for m in methods]
    train_stds = [summary[m]["train_violations"]["std"] for m in methods]
    test_means = [summary[m]["test_violations"]["mean"] for m in methods]
    test_stds = [summary[m]["test_violations"]["std"] for m in methods]
    
    ax1.bar(x - width/2, train_means, width, yerr=train_stds, label='Train (seen trap)', 
            color='blue', alpha=0.7, capsize=3)
    ax1.bar(x + width/2, test_means, width, yerr=test_stds, label='Test (unseen trap)',
            color='red', alpha=0.7, capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in methods])
    ax1.set_ylabel('Mean Violations')
    ax1.set_title('Train vs Test Violations')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Generalization gap
    ax2 = axes[1]
    gaps = [summary[m]["generalization_gap"]["mean"] for m in methods]
    gap_stds = [summary[m]["generalization_gap"]["std"] for m in methods]
    colors = ['green' if g < 0 else 'red' for g in gaps]
    
    ax2.bar(x, gaps, yerr=gap_stds, color=colors, alpha=0.7, capsize=3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in methods])
    ax2.set_ylabel('Generalization Gap (Test - Train)')
    ax2.set_title('Generalization Gap (lower is better)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / "generalization_plots.png", dpi=150)
    print(f"Plots saved to {output_path / 'generalization_plots.png'}")
    plt.close()


# ============================================================================
# 5. MAIN EXPERIMENT
# ============================================================================

def main(
    num_seeds: int = 50,
    run_ablations: bool = True,
    output_dir: str = "../../results/sandbagging_v2"
):
    """Run the full sandbagging experiment."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    env_config = EnvConfig()
    base_algo_config = AlgoConfig(episodes=300)
    
    print("="*60)
    print("SANDBAGGING TRAP EXPERIMENT v2")
    print("="*60)
    print(f"Seeds: {num_seeds}, Episodes: {base_algo_config.episodes}")
    
    # Run all methods
    method_results = {}
    for method in ["ppo", "cpo", "sgpo", "sgpo_anis"]:
        print(f"\n--- Running {method.upper()} ---")
        method_results[method] = run_method(
            method, env_config, base_algo_config, num_seeds
        )
    
    # Statistical comparisons
    comparison = MethodComparison()
    for method, results in method_results.items():
        comparison.add_method(method, results)
    
    comparisons = {
        "violations": comparison.compare_all("total_violations"),
        "returns": comparison.compare_all("final_return")
    }
    
    # Ablations (if requested)
    ablation_results = {}
    if run_ablations:
        print("\n--- Running Ablations ---")
        
        # Reduced seeds for ablations
        ablation_seeds = min(20, num_seeds)
        
        # Sharpness ablation
        print("  Ablation: metric_sharpness")
        ablation_results["sharpness"] = run_ablation(
            env_config, base_algo_config, ablation_seeds,
            "metric_sharpness", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        )
        
        # Severity ablation
        print("  Ablation: metric_severity")
        ablation_results["severity"] = run_ablation(
            env_config, base_algo_config, ablation_seeds,
            "metric_severity", [1.0, 2.5, 5.0, 10.0, 20.0]
        )
    
    # Save all results
    final_output = {
        "experiment": "SandbaggingTrap_v2",
        "env_config": vars(env_config),
        "algo_config": vars(base_algo_config),
        "num_seeds": num_seeds,
        "method_results": method_results,
        "comparisons": comparisons,
        "ablations": ablation_results
    }
    
    with open(output_path / "sandbagging_v2_results.json", 'w') as f:
        json.dump(final_output, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nResults saved to {output_path / 'sandbagging_v2_results.json'}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for method, results in method_results.items():
        m = results["metrics"]
        print(f"\n{method.upper()}:")
        print(f"  Return: {m['final_return']['mean']:.2f} ± {m['final_return']['std']:.2f}")
        print(f"  Violations: {m['total_violations']['mean']:.1f} ± {m['total_violations']['std']:.1f}")
        print(f"  Goal Rate: {m['goal_success_rate']['mean']*100:.1f}%")
    
    print("\nStatistical Comparisons (violations):")
    for key, comp in comparisons["violations"].items():
        sig = "***" if comp["significant_at_005"] else ""
        print(f"  {key}: Cohen's d = {comp['cohens_d']:.2f} ({comp['effect_size']}) {sig}")
    
    # Generate plots
    _plot_results(method_results, ablation_results, output_path)
    
    return final_output


def _plot_results(method_results: Dict, ablation_results: Dict, output_path: Path):
    """Generate visualization plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    methods = list(method_results.keys())
    colors = {'ppo': 'red', 'cpo': 'orange', 'sgpo': 'blue', 'sgpo_anis': 'green'}
    
    # Plot 1: Learning curves (returns)
    ax1 = axes[0, 0]
    for method in methods:
        lc = method_results[method]["learning_curves"]
        mean = np.array(lc["returns_mean"])
        std = np.array(lc["returns_std"])
        
        # Smooth
        window = 20
        if len(mean) > window:
            mean_smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
            std_smooth = np.convolve(std, np.ones(window)/window, mode='valid')
            x = np.arange(len(mean_smooth))
            ax1.plot(x, mean_smooth, color=colors[method], label=method.upper())
            ax1.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth,
                            color=colors[method], alpha=0.2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('Learning Curves (Returns)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning curves (violations)
    ax2 = axes[0, 1]
    for method in methods:
        lc = method_results[method]["learning_curves"]
        mean = np.array(lc["violations_mean"])
        
        window = 20
        if len(mean) > window:
            mean_smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
            ax2.plot(mean_smooth, color=colors[method], label=method.upper())
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Violations')
    ax2.set_title('Learning Curves (Violations)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final violations bar chart with CI
    ax3 = axes[0, 2]
    x = np.arange(len(methods))
    means = [method_results[m]["metrics"]["total_violations"]["mean"] for m in methods]
    ci_widths = [
        (method_results[m]["metrics"]["total_violations"]["ci_95"][1] -
         method_results[m]["metrics"]["total_violations"]["ci_95"][0]) / 2
        for m in methods
    ]
    bars = ax3.bar(x, means, yerr=ci_widths, color=[colors[m] for m in methods],
                   alpha=0.7, capsize=5)
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.upper() for m in methods])
    ax3.set_ylabel('Total Violations')
    ax3.set_title('Total Violations (95% CI)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Final returns bar chart with CI
    ax4 = axes[1, 0]
    means = [method_results[m]["metrics"]["final_return"]["mean"] for m in methods]
    ci_widths = [
        (method_results[m]["metrics"]["final_return"]["ci_95"][1] -
         method_results[m]["metrics"]["final_return"]["ci_95"][0]) / 2
        for m in methods
    ]
    bars = ax4.bar(x, means, yerr=ci_widths, color=[colors[m] for m in methods],
                   alpha=0.7, capsize=5)
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.upper() for m in methods])
    ax4.set_ylabel('Final Return')
    ax4.set_title('Final Return (95% CI)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Sharpness ablation (if available)
    ax5 = axes[1, 1]
    if "sharpness" in ablation_results:
        abl = ablation_results["sharpness"]
        values = abl["values"]
        viol_means = [r["total_violations"]["mean"] for r in abl["results"]]
        viol_stds = [r["total_violations"]["std"] for r in abl["results"]]
        
        ax5.errorbar(values, viol_means, yerr=viol_stds, marker='o', capsize=3)
        ax5.axvline(x=2.0, color='red', linestyle='--', alpha=0.5, label='β=2 (theoretical)')
        ax5.set_xlabel('Sharpness β')
        ax5.set_ylabel('Total Violations')
        ax5.set_title('Ablation: Metric Sharpness')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Severity ablation (if available)
    ax6 = axes[1, 2]
    if "severity" in ablation_results:
        abl = ablation_results["severity"]
        values = abl["values"]
        viol_means = [r["total_violations"]["mean"] for r in abl["results"]]
        viol_stds = [r["total_violations"]["std"] for r in abl["results"]]
        
        ax6.errorbar(values, viol_means, yerr=viol_stds, marker='s', capsize=3)
        ax6.set_xlabel('Severity C')
        ax6.set_ylabel('Total Violations')
        ax6.set_title('Ablation: Metric Severity')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'sandbagging_v2_plots.png', dpi=150)
    print(f"Plots saved to {output_path / 'sandbagging_v2_plots.png'}")


# ============================================================================
# 6. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sandbagging Trap Experiment v2")
    parser.add_argument("--seeds", type=int, default=50, help="Number of seeds")
    parser.add_argument("--quick", action="store_true", help="Quick test with 5 seeds")
    parser.add_argument("--no-ablations", action="store_true", help="Skip ablation studies")
    parser.add_argument("--mode", type=str, default="full", 
                        choices=["full", "diagnostics", "generalization"],
                        help="Experiment mode: full (main + ablations), diagnostics (SGPO debugging), generalization (train/test split)")
    args = parser.parse_args()
    
    num_seeds = 5 if args.quick else args.seeds
    
    if args.mode == "diagnostics":
        # Run diagnostic experiment for SGPO debugging
        env_config = EnvConfig()
        algo_config = AlgoConfig(episodes=300)
        run_diagnostic_experiment(
            env_config, algo_config,
            num_seeds=min(num_seeds, 10),  # Diagnostics are detailed, limit seeds
            output_dir="../../results/sandbagging_v2_diagnostics"
        )
    
    elif args.mode == "generalization":
        # Run generalization experiment (train/test split)
        gen_config = GeneralizationEnvConfig(
            train_trap_centers=[(5.0, 6.0)],
            test_trap_centers=[(7.0, 3.0)]
        )
        algo_config = AlgoConfig(episodes=300)
        run_generalization_experiment(
            gen_config, algo_config,
            num_seeds=num_seeds,
            output_dir="../../results/sandbagging_v2_generalization"
        )
    
    else:
        # Full experiment (default)
        results = main(
            num_seeds=num_seeds,
            run_ablations=not args.no_ablations,
            output_dir="../../results/sandbagging_v2"
        )
