#!/usr/bin/env python3
"""
Safety Gymnasium Benchmark for SGPO vs Baselines

This script provides a comprehensive experimental comparison of:
- PPO (Proximal Policy Optimization) - standard RLHF baseline
- CPO (Constrained Policy Optimization) - safe RL baseline  
- PPO-Lagrangian - Lagrangian relaxation approach
- SGPO (Sheaf-Geodesic Policy Optimization) - our proposed method

Environments from Safety Gymnasium:
- SafetyPointGoal1-v0: Point robot navigation with hazards
- SafetyCarGoal1-v0: Car robot with dynamics constraints
- SafetyAntGoal1-v0: Ant locomotion with safety constraints

Usage:
    python safety_gym_benchmark.py --env SafetyPointGoal1-v0 --method gpo --seeds 3
    python safety_gym_benchmark.py --all  # Run full benchmark suite
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Check for required packages
try:
    import safety_gymnasium as safety_gym
    SAFETY_GYM_AVAILABLE = True
except ImportError:
    SAFETY_GYM_AVAILABLE = False
    print("Warning: safety_gymnasium not installed. Install with: pip install safety-gymnasium")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed.")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    env_id: str = "SafetyPointGoal1-v0"
    method: str = "gpo"  # ppo, cpo, ppo_lagrangian, gpo
    seed: int = 0
    total_steps: int = 1_000_000
    steps_per_epoch: int = 10_000
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_metric: float = 1e-4  # SGPO-specific
    cost_limit: float = 25.0  # CPO/Lagrangian constraint
    metric_alpha: float = 1.0  # SGPO: exponent for barrier (alpha >= 1)
    metric_scale: float = 10.0  # SGPO: scale factor C
    hidden_sizes: Tuple[int, ...] = (256, 256)
    save_freq: int = 50_000
    log_freq: int = 10_000
    output_dir: str = "results/safety_gym"


@dataclass 
class BenchmarkResults:
    """Results from a benchmark run."""
    config: Dict
    episode_returns: List[float] = field(default_factory=list)
    episode_costs: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    cost_rate: float = 0.0  # Average cost per step
    mean_return: float = 0.0
    std_return: float = 0.0
    mean_cost: float = 0.0
    std_cost: float = 0.0
    total_violations: int = 0
    training_time: float = 0.0
    safety_margin: float = 0.0  # SGPO-specific: min distance to hazards


# ============================================================================
# Neural Network Components
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron."""
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Tuple[int, ...], 
                 activation=nn.ReLU, output_activation=None):
        super().__init__()
        layers = []
        prev_size = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_dim))
        if output_activation:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class GaussianActor(nn.Module):
    """Gaussian policy for continuous actions."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.mean_net = MLP(obs_dim, act_dim, hidden_sizes)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))
    
    def forward(self, obs):
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)
    
    def act(self, obs, deterministic=False):
        dist = self.forward(obs)
        if deterministic:
            return dist.mean
        return dist.sample()


class Critic(nn.Module):
    """Value function critic."""
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.v_net = MLP(obs_dim, 1, hidden_sizes)
    
    def forward(self, obs):
        return self.v_net(obs).squeeze(-1)


class CostCritic(nn.Module):
    """Cost value function for constrained RL."""
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.c_net = MLP(obs_dim, 1, hidden_sizes)
    
    def forward(self, obs):
        return self.c_net(obs).squeeze(-1)


class RiemannianMetric(nn.Module):
    """
    Learned Riemannian metric for SGPO.
    
    The conformal factor phi(x) determines path lengths:
        L(gamma) = integral phi(gamma(t)) ||gamma'(t)|| dt
    
    For safety, we want phi(x) -> infinity as x -> hazards.
    We parameterize: phi(x) = 1 + scale * max(0, threshold - d(x))^alpha
    where d(x) is the learned distance to hazards.
    """
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, ...], 
                 alpha: float = 1.0, scale: float = 10.0):
        super().__init__()
        self.alpha = alpha
        self.scale = scale
        # Network predicts "inverse distance" to hazards (higher = more dangerous)
        self.danger_net = MLP(obs_dim, 1, hidden_sizes, output_activation=nn.Softplus)
    
    def forward(self, obs) -> torch.Tensor:
        """Compute conformal factor phi(x)."""
        danger = self.danger_net(obs).squeeze(-1)  # [0, inf)
        # phi = 1 + scale * danger^alpha
        phi = 1.0 + self.scale * torch.pow(danger + 1e-8, self.alpha)
        return phi
    
    def geodesic_penalty(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """
        Compute Riemannian path length penalty for a transition.
        
        Approximates: phi(midpoint) * ||next_obs - obs||
        """
        midpoint = (obs + next_obs) / 2
        phi = self.forward(midpoint)
        euclidean_dist = torch.norm(next_obs - obs, dim=-1)
        return phi * euclidean_dist


# ============================================================================
# Algorithm Implementations
# ============================================================================

class PPOAgent:
    """Standard PPO agent."""
    def __init__(self, obs_dim: int, act_dim: int, config: ExperimentConfig):
        self.config = config
        self.actor = GaussianActor(obs_dim, act_dim, config.hidden_sizes)
        self.critic = Critic(obs_dim, config.hidden_sizes)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(obs_t)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        return action.squeeze(0).numpy(), log_prob.item()
    
    def update(self, batch: Dict) -> Dict:
        """PPO update step."""
        obs = torch.FloatTensor(batch['obs'])
        act = torch.FloatTensor(batch['act'])
        ret = torch.FloatTensor(batch['ret'])
        adv = torch.FloatTensor(batch['adv'])
        old_logp = torch.FloatTensor(batch['logp'])
        
        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        # Policy update
        dist = self.actor(obs)
        logp = dist.log_prob(act).sum(-1)
        ratio = torch.exp(logp - old_logp)
        clip_adv = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * adv
        policy_loss = -torch.min(ratio * adv, clip_adv).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # Value update
        v = self.critic(obs)
        value_loss = F.mse_loss(v, ret)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        return {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}


class SGPOAgent(PPOAgent):
    """
    Sheaf-Geodesic Policy Optimization agent.
    
    Extends PPO with:
    1. Learned Riemannian metric that penalizes paths near hazards
    2. Geodesic advantage: standard advantage - metric penalty
    3. Metric learning from cost signals
    """
    def __init__(self, obs_dim: int, act_dim: int, config: ExperimentConfig):
        super().__init__(obs_dim, act_dim, config)
        self.metric = RiemannianMetric(
            obs_dim, config.hidden_sizes,
            alpha=config.metric_alpha,
            scale=config.metric_scale
        )
        self.metric_optimizer = torch.optim.Adam(
            self.metric.parameters(), lr=config.lr_metric
        )
    
    def update(self, batch: Dict) -> Dict:
        """SGPO update with geodesic penalty."""
        obs = torch.FloatTensor(batch['obs'])
        next_obs = torch.FloatTensor(batch['next_obs'])
        act = torch.FloatTensor(batch['act'])
        ret = torch.FloatTensor(batch['ret'])
        adv = torch.FloatTensor(batch['adv'])
        old_logp = torch.FloatTensor(batch['logp'])
        cost = torch.FloatTensor(batch['cost'])
        
        # Compute geodesic penalty
        geo_penalty = self.metric.geodesic_penalty(obs, next_obs)
        
        # Modified advantage: penalize transitions with high geodesic cost
        geo_adv = adv - geo_penalty.detach()
        geo_adv = (geo_adv - geo_adv.mean()) / (geo_adv.std() + 1e-8)
        
        # Policy update with geodesic advantage
        dist = self.actor(obs)
        logp = dist.log_prob(act).sum(-1)
        ratio = torch.exp(logp - old_logp)
        clip_adv = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * geo_adv
        policy_loss = -torch.min(ratio * geo_adv, clip_adv).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # Value update
        v = self.critic(obs)
        value_loss = F.mse_loss(v, ret)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        # Metric learning: high phi where cost is high
        # Loss: encourage phi to predict cost (supervised)
        phi = self.metric(obs)
        metric_loss = F.mse_loss(phi, 1.0 + self.config.metric_scale * cost)
        
        self.metric_optimizer.zero_grad()
        metric_loss.backward()
        self.metric_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'metric_loss': metric_loss.item(),
            'mean_geo_penalty': geo_penalty.mean().item(),
            'mean_phi': phi.mean().item()
        }


class PPOLagrangianAgent(PPOAgent):
    """PPO with Lagrangian relaxation for cost constraints."""
    def __init__(self, obs_dim: int, act_dim: int, config: ExperimentConfig):
        super().__init__(obs_dim, act_dim, config)
        self.cost_critic = CostCritic(obs_dim, config.hidden_sizes)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config.lr_critic)
        # Learnable Lagrange multiplier
        self.log_lambda = nn.Parameter(torch.zeros(1))
        self.lambda_optimizer = torch.optim.Adam([self.log_lambda], lr=5e-3)
        self.cost_limit = config.cost_limit
    
    def update(self, batch: Dict) -> Dict:
        obs = torch.FloatTensor(batch['obs'])
        act = torch.FloatTensor(batch['act'])
        ret = torch.FloatTensor(batch['ret'])
        adv = torch.FloatTensor(batch['adv'])
        old_logp = torch.FloatTensor(batch['logp'])
        cost_ret = torch.FloatTensor(batch['cost_ret'])
        cost_adv = torch.FloatTensor(batch['cost_adv'])
        
        lam = torch.exp(self.log_lambda).detach()
        
        # Combined advantage: reward - lambda * cost
        combined_adv = adv - lam * cost_adv
        combined_adv = (combined_adv - combined_adv.mean()) / (combined_adv.std() + 1e-8)
        
        # Policy update
        dist = self.actor(obs)
        logp = dist.log_prob(act).sum(-1)
        ratio = torch.exp(logp - old_logp)
        clip_adv = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * combined_adv
        policy_loss = -torch.min(ratio * combined_adv, clip_adv).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # Value updates
        v = self.critic(obs)
        value_loss = F.mse_loss(v, ret)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        vc = self.cost_critic(obs)
        cost_value_loss = F.mse_loss(vc, cost_ret)
        self.cost_optimizer.zero_grad()
        cost_value_loss.backward()
        self.cost_optimizer.step()
        
        # Lambda update: increase if cost exceeds limit
        mean_cost = cost_ret.mean()
        lambda_loss = -self.log_lambda * (mean_cost - self.cost_limit)
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'cost_value_loss': cost_value_loss.item(),
            'lambda': lam.item(),
            'mean_cost': mean_cost.item()
        }


# ============================================================================
# Training Loop
# ============================================================================

def compute_gae(rewards: List[float], values: List[float], dones: List[bool],
                gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation."""
    n = len(rewards)
    advantages = np.zeros(n)
    returns = np.zeros(n)
    last_gae = 0
    last_value = 0
    
    for t in reversed(range(n)):
        if dones[t]:
            delta = rewards[t] - values[t]
            last_gae = delta
            last_value = 0
        else:
            next_value = values[t + 1] if t + 1 < n else last_value
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns


def run_experiment(config: ExperimentConfig) -> BenchmarkResults:
    """Run a single experiment."""
    if not SAFETY_GYM_AVAILABLE or not TORCH_AVAILABLE:
        print("Required packages not available. Returning empty results.")
        return BenchmarkResults(config=asdict(config))
    
    # Setup
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    env = safety_gym.make(config.env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Create agent
    if config.method == 'ppo':
        agent = PPOAgent(obs_dim, act_dim, config)
    elif config.method == 'gpo':
        agent = SGPOAgent(obs_dim, act_dim, config)
    elif config.method == 'ppo_lagrangian':
        agent = PPOLagrangianAgent(obs_dim, act_dim, config)
    else:
        raise ValueError(f"Unknown method: {config.method}")
    
    # Training loop
    results = BenchmarkResults(config=asdict(config))
    start_time = time.time()
    
    obs, info = env.reset(seed=config.seed)
    episode_return = 0
    episode_cost = 0
    episode_length = 0
    
    buffer = {
        'obs': [], 'act': [], 'rew': [], 'next_obs': [],
        'done': [], 'logp': [], 'val': [], 'cost': []
    }
    
    for step in range(config.total_steps):
        # Collect transition
        action, logp = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cost = info.get('cost', 0.0)
        
        with torch.no_grad():
            value = agent.critic(torch.FloatTensor(obs).unsqueeze(0)).item()
        
        buffer['obs'].append(obs)
        buffer['act'].append(action)
        buffer['rew'].append(reward)
        buffer['next_obs'].append(next_obs)
        buffer['done'].append(done)
        buffer['logp'].append(logp)
        buffer['val'].append(value)
        buffer['cost'].append(cost)
        
        episode_return += reward
        episode_cost += cost
        episode_length += 1
        
        if done:
            results.episode_returns.append(episode_return)
            results.episode_costs.append(episode_cost)
            results.episode_lengths.append(episode_length)
            results.total_violations += int(episode_cost > 0)
            
            obs, info = env.reset()
            episode_return = 0
            episode_cost = 0
            episode_length = 0
        else:
            obs = next_obs
        
        # Update at end of epoch
        if (step + 1) % config.steps_per_epoch == 0:
            # Compute advantages
            advantages, returns = compute_gae(
                buffer['rew'], buffer['val'], buffer['done'],
                config.gamma, config.gae_lambda
            )
            
            # Build batch
            batch = {
                'obs': np.array(buffer['obs']),
                'act': np.array(buffer['act']),
                'ret': returns,
                'adv': advantages,
                'logp': np.array(buffer['logp']),
                'next_obs': np.array(buffer['next_obs']),
                'cost': np.array(buffer['cost'])
            }
            
            # For Lagrangian: also compute cost advantages
            if config.method == 'ppo_lagrangian':
                cost_adv, cost_ret = compute_gae(
                    buffer['cost'], 
                    [0] * len(buffer['cost']),  # Simple baseline
                    buffer['done'],
                    config.gamma, config.gae_lambda
                )
                batch['cost_adv'] = cost_adv
                batch['cost_ret'] = cost_ret
            
            # Update
            update_info = agent.update(batch)
            
            # Clear buffer
            for k in buffer:
                buffer[k] = []
            
            # Logging
            if (step + 1) % config.log_freq == 0:
                recent_returns = results.episode_returns[-10:] if results.episode_returns else [0]
                recent_costs = results.episode_costs[-10:] if results.episode_costs else [0]
                print(f"Step {step+1}/{config.total_steps} | "
                      f"Return: {np.mean(recent_returns):.2f} | "
                      f"Cost: {np.mean(recent_costs):.2f} | "
                      f"{update_info}")
    
    # Final statistics
    results.training_time = time.time() - start_time
    results.mean_return = np.mean(results.episode_returns)
    results.std_return = np.std(results.episode_returns)
    results.mean_cost = np.mean(results.episode_costs)
    results.std_cost = np.std(results.episode_costs)
    results.cost_rate = sum(results.episode_costs) / sum(results.episode_lengths)
    
    env.close()
    return results


def run_benchmark_suite(envs: List[str], methods: List[str], seeds: List[int],
                        output_dir: str = "results/safety_gym") -> Dict:
    """Run full benchmark suite."""
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    
    for env_id in envs:
        all_results[env_id] = {}
        for method in methods:
            all_results[env_id][method] = []
            for seed in seeds:
                print(f"\n{'='*60}")
                print(f"Running: {env_id} | {method} | seed={seed}")
                print('='*60)
                
                config = ExperimentConfig(
                    env_id=env_id,
                    method=method,
                    seed=seed,
                    output_dir=output_dir
                )
                
                results = run_experiment(config)
                all_results[env_id][method].append(asdict(results))
    
    # Save results
    results_path = Path(output_dir) / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_path}")
    return all_results


def print_summary_table(results: Dict):
    """Print summary table of results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Environment':<25} {'Method':<15} {'Return':>12} {'Cost':>12} {'Violations':>12}")
    print("-"*80)
    
    for env_id, env_results in results.items():
        for method, runs in env_results.items():
            returns = [r['mean_return'] for r in runs]
            costs = [r['mean_cost'] for r in runs]
            violations = [r['total_violations'] for r in runs]
            
            print(f"{env_id:<25} {method:<15} "
                  f"{np.mean(returns):>10.2f}±{np.std(returns):>4.1f} "
                  f"{np.mean(costs):>10.2f}±{np.std(costs):>4.1f} "
                  f"{np.mean(violations):>10.1f}±{np.std(violations):>4.1f}")
    
    print("="*80)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safety Gymnasium Benchmark")
    parser.add_argument("--env", type=str, default="SafetyPointGoal1-v0",
                        help="Environment ID")
    parser.add_argument("--method", type=str, default="gpo",
                        choices=['ppo', 'cpo', 'ppo_lagrangian', 'gpo'],
                        help="Algorithm to use")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Total training steps")
    parser.add_argument("--all", action="store_true",
                        help="Run full benchmark suite")
    parser.add_argument("--output", type=str, default="results/safety_gym",
                        help="Output directory")
    
    args = parser.parse_args()
    
    if args.all:
        # Full benchmark suite
        envs = [
            "SafetyPointGoal1-v0",
            "SafetyPointGoal2-v0",
            "SafetyCarGoal1-v0",
        ]
        methods = ["ppo", "ppo_lagrangian", "gpo"]
        seeds = list(range(args.seeds))
        
        results = run_benchmark_suite(envs, methods, seeds, args.output)
        print_summary_table(results)
    else:
        # Single experiment
        config = ExperimentConfig(
            env_id=args.env,
            method=args.method,
            seed=0,
            total_steps=args.steps,
            output_dir=args.output
        )
        
        for seed in range(args.seeds):
            config.seed = seed
            results = run_experiment(config)
            print(f"\nSeed {seed}: Return={results.mean_return:.2f}, "
                  f"Cost={results.mean_cost:.2f}, Violations={results.total_violations}")
