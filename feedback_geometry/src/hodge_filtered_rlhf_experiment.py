"""
Hodge-Filtered RLHF Experiment: Context-Conditional H¹ for Reward Hacking Prevention

This script implements two experiments for publishable results:

EXPERIMENT A: Pre-filtered Reward Models
- Train reward models on raw vs H¹-filtered preferences (normalized data sizes)
- Compare reward hacking rates on held-out test set
- Hypothesis: Filtered model has lower exploitation rate

EXPERIMENT C: Context-Conditional Harmonic-Discounted Advantage
- Integrate ω_invalid into SGPO_ANIS advantage computation
- A = (r + γV' - V - ω_invalid) / √g(x,v)
- Hypothesis: SGPO_ANIS_HODGE has lowest violations + highest goal rate

Key Innovation:
- Context-conditional H¹ distinguishes valid contextual cycles from invalid intransitivity
- Only invalid cycles are filtered/discounted, preserving valid preference variation

Usage:
    # Experiment A: Pre-filtered reward models
    python hodge_filtered_rlhf_experiment.py --experiment A --seeds 20
    
    # Experiment C: Context-conditional SGPO
    python hodge_filtered_rlhf_experiment.py --experiment C --seeds 20
    
    # Both experiments
    python hodge_filtered_rlhf_experiment.py --experiment both --seeds 20
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import defaultdict
import random
from datetime import datetime

# Local imports
from context_conditional_hodge_critic import (
    ContextConditionalHodgeCritic,
    ContextualFeedbackItem,
    ConditionalH1Result,
    load_hh_rlhf_with_context,
    create_normalized_train_sets,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for Hodge-filtered RLHF experiments."""
    # Data
    n_samples: int = 5000
    train_ratio: float = 0.8
    h1_threshold: float = 0.8
    
    # Reward model
    reward_hidden_dim: int = 128
    reward_lr: float = 1e-4
    reward_epochs: int = 50
    reward_batch_size: int = 32
    
    # Policy (SGPO_ANIS)
    policy_hidden_dim: int = 128
    policy_lr: float = 3e-4
    critic_lr: float = 1e-3
    metric_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    n_episodes: int = 500
    max_steps: int = 100
    
    # SGPO_ANIS specific
    metric_severity: float = 5.0
    anisotropic_max_metric: float = 100.0
    warmup_episodes: int = 30
    use_soft_scaling: bool = True
    
    # Experiment
    seeds: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "results/hodge_filtered_rlhf"
    
    # Embedding model
    embedding_model_name: str = "all-MiniLM-L6-v2"


@dataclass
class ExperimentAResults:
    """Results for Experiment A: Pre-filtered reward models."""
    seed: int
    raw_reward_loss: float
    filtered_reward_loss: float
    raw_h1_initial: float
    filtered_h1_initial: float
    raw_exploitation_rate: float
    filtered_exploitation_rate: float
    raw_test_accuracy: float
    filtered_test_accuracy: float
    n_train_raw: int
    n_train_filtered: int
    conditional_h1_analysis: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentCResults:
    """Results for Experiment C: Context-conditional SGPO."""
    seed: int
    method: str  # ppo, cpo, sgpo_anis, sgpo_anis_hodge
    total_violations: float
    final_return: float
    goal_rate: float
    h1_filtered_ratio: float
    training_returns: List[float] = field(default_factory=list)
    training_violations: List[float] = field(default_factory=list)


# =============================================================================
# Reward Model
# =============================================================================

class PreferenceRewardModel(nn.Module):
    """Bradley-Terry preference-based reward model."""
    
    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.reward_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reward for embeddings."""
        return self.reward_net(x)
    
    def preference_probability(
        self,
        chosen_emb: torch.Tensor,
        rejected_emb: torch.Tensor,
    ) -> torch.Tensor:
        """P(chosen > rejected) under Bradley-Terry model."""
        r_chosen = self.forward(chosen_emb)
        r_rejected = self.forward(rejected_emb)
        return torch.sigmoid(r_chosen - r_rejected)


def train_reward_model(
    model: PreferenceRewardModel,
    train_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (chosen, rejected)
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    """
    Train reward model on preference pairs.
    
    Returns:
        Final training loss
    """
    chosen_embs, rejected_embs = train_embeddings
    n_samples = len(chosen_embs)
    
    optimizer = optim.Adam(model.parameters(), lr=config.reward_lr)
    
    model.train()
    final_loss = 0.0
    
    for epoch in range(config.reward_epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        chosen_embs = chosen_embs[perm]
        rejected_embs = rejected_embs[perm]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, config.reward_batch_size):
            batch_chosen = chosen_embs[i:i + config.reward_batch_size].to(device)
            batch_rejected = rejected_embs[i:i + config.reward_batch_size].to(device)
            
            optimizer.zero_grad()
            
            # Bradley-Terry loss: -log P(chosen > rejected)
            probs = model.preference_probability(batch_chosen, batch_rejected)
            loss = -torch.log(probs + 1e-8).mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        final_loss = epoch_loss / max(n_batches, 1)
    
    return final_loss


def evaluate_reward_model(
    model: PreferenceRewardModel,
    test_embeddings: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate reward model accuracy and exploitation rate.
    
    Returns:
        (accuracy, exploitation_rate)
    """
    chosen_embs, rejected_embs = test_embeddings
    
    model.eval()
    with torch.no_grad():
        chosen_embs = chosen_embs.to(device)
        rejected_embs = rejected_embs.to(device)
        
        probs = model.preference_probability(chosen_embs, rejected_embs)
        
        # Accuracy: P(chosen > rejected) > 0.5
        accuracy = (probs > 0.5).float().mean().item()
        
        # Exploitation rate: How often does model strongly prefer rejected?
        # (This indicates reward hacking potential)
        exploitation_rate = (probs < 0.3).float().mean().item()
    
    return accuracy, exploitation_rate


# =============================================================================
# Environment for Experiment C
# =============================================================================

class PreferenceBasedEnv:
    """
    Environment where reward is based on learned preference model.
    
    The agent navigates in embedding space, and reward is given by
    the preference model's assessment of the trajectory.
    """
    
    def __init__(
        self,
        reward_model: PreferenceRewardModel,
        embed_dim: int,
        trap_centers: List[np.ndarray] = None,
        trap_radius: float = 2.0,
        trap_strength: float = 10.0,
        max_steps: int = 100,
        device: torch.device = None,
    ):
        self.reward_model = reward_model
        self.embed_dim = embed_dim
        self.device = device or torch.device("cpu")
        self.max_steps = max_steps
        
        # Trap regions (represent "reward hacking" zones)
        if trap_centers is None:
            # Random trap locations
            self.trap_centers = [
                np.random.randn(embed_dim) * 5.0
                for _ in range(3)
            ]
        else:
            self.trap_centers = trap_centers
        
        self.trap_radius = trap_radius
        self.trap_strength = trap_strength
        
        self.state = None
        self.step_count = 0
        self.goal = np.random.randn(embed_dim) * 10.0
    
    def reset(self) -> np.ndarray:
        """Reset environment to random starting state."""
        self.state = np.random.randn(self.embed_dim) * 0.1
        self.step_count = 0
        return self.state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action and return (next_state, reward, done, info).
        
        Reward is based on:
        1. Preference model's reward for current state
        2. Penalty for entering trap regions (exploitation)
        3. Bonus for reaching goal
        """
        # Clip and apply action
        action = np.clip(action, -0.5, 0.5)
        next_state = self.state + action
        
        # Compute reward from preference model
        state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            base_reward = self.reward_model(state_tensor.unsqueeze(0)).item()
        
        # Trap penalty (exploitation detection)
        trap_cost = 0.0
        in_trap = False
        for trap_center in self.trap_centers:
            dist = np.linalg.norm(next_state - trap_center)
            if dist < self.trap_radius:
                trap_cost += self.trap_strength
                in_trap = True
            elif dist < self.trap_radius * 2:
                proximity = 1.0 - (dist - self.trap_radius) / self.trap_radius
                trap_cost += self.trap_strength * (proximity ** 2)
        
        # Goal bonus
        goal_dist = np.linalg.norm(next_state - self.goal)
        goal_reached = goal_dist < 2.0
        goal_bonus = 10.0 if goal_reached else 0.0
        
        # Total reward
        reward = base_reward - trap_cost + goal_bonus
        
        self.state = next_state
        self.step_count += 1
        done = self.step_count >= self.max_steps or goal_reached
        
        info = {
            "trap_cost": trap_cost,
            "in_trap": in_trap,
            "goal_reached": goal_reached,
            "base_reward": base_reward,
        }
        
        return self.state.copy(), reward, done, info
    
    def get_trap_centers(self) -> List[np.ndarray]:
        """Return trap centers for metric learning."""
        return self.trap_centers


# =============================================================================
# SGPO_ANIS with Hodge Integration
# =============================================================================

class Actor(nn.Module):
    """Gaussian policy network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, x: torch.Tensor) -> Normal:
        mean = self.net(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)


class Critic(nn.Module):
    """Value function network."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AnisotropicMetricWithHodge(nn.Module):
    """
    Anisotropic Riemannian metric with Hodge harmonic integration.
    
    Computes g(x, v) that:
    1. Only penalizes movement TOWARD danger (anisotropic)
    2. Incorporates invalid harmonic component for discounting
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 32,
        base_metric: float = 1.0,
        severity: float = 5.0,
        max_metric: float = 100.0,
        hodge_critic: Optional[ContextConditionalHodgeCritic] = None,
    ):
        super().__init__()
        
        self.danger_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.center_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
        self.base_metric = base_metric
        self.severity = severity
        self.max_metric = max_metric
        self.hodge_critic = hodge_critic
    
    def forward(
        self,
        x: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute anisotropic metric and harmonic component.
        
        Returns:
            g: Metric values [batch]
            escape_factor: Escape factors [batch]
            omega_invalid: Invalid harmonic component [batch]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        device = x.device
        
        # Danger level
        danger_level = torch.sigmoid(self.danger_net(x)) * self.severity
        danger_center = self.center_net(x)
        
        if v is None:
            g = self.base_metric + danger_level.squeeze(-1)
            escape_factor = torch.ones(batch_size, device=device)
            omega_invalid = torch.zeros(batch_size, device=device)
            return g, escape_factor, omega_invalid
        
        if v.dim() == 1:
            v = v.unsqueeze(0)
        
        # Direction to danger
        to_danger = danger_center - x
        dist_to_danger = torch.norm(to_danger, dim=-1, keepdim=True) + 1e-8
        n_hat = to_danger / dist_to_danger
        
        # Velocity component toward danger
        v_toward = torch.sum(v * n_hat, dim=-1, keepdim=True)
        v_toward_pos = torch.clamp(v_toward, min=0)
        v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
        
        # Anisotropic scaling
        toward_ratio_sq = (v_toward_pos / v_norm) ** 2
        g_dir = danger_level / (dist_to_danger + 0.1)
        
        g = self.base_metric + (toward_ratio_sq * g_dir).squeeze(-1)
        g = torch.clamp(g, max=self.max_metric)
        
        # Escape factor
        escape_factor = torch.sigmoid(-v_toward * 5.0).squeeze(-1)
        
        # Harmonic component (invalid cycles)
        if self.hodge_critic is not None and context is not None:
            omega_invalid = self.hodge_critic.harmonic_given_context(x, v, context).squeeze(-1)
        else:
            omega_invalid = torch.zeros(batch_size, device=device)
        
        return g, escape_factor, omega_invalid


def train_sgpo_anis_hodge(
    env: PreferenceBasedEnv,
    config: ExperimentConfig,
    hodge_critic: Optional[ContextConditionalHodgeCritic] = None,
    use_hodge: bool = True,
    seed: int = 0,
) -> ExperimentCResults:
    """
    Train SGPO_ANIS with optional Hodge harmonic discounting.
    
    Args:
        env: Environment with preference-based rewards
        config: Experiment configuration
        hodge_critic: Context-conditional Hodge critic (None = no Hodge)
        use_hodge: Whether to use Hodge harmonic discounting
        seed: Random seed
        
    Returns:
        ExperimentCResults with training metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(config.device)
    state_dim = env.embed_dim
    action_dim = env.embed_dim
    
    # Networks
    actor = Actor(state_dim, action_dim, config.policy_hidden_dim).to(device)
    critic = Critic(state_dim, config.policy_hidden_dim).to(device)
    metric = AnisotropicMetricWithHodge(
        state_dim,
        severity=config.metric_severity,
        max_metric=config.anisotropic_max_metric,
        hodge_critic=hodge_critic if use_hodge else None,
    ).to(device)
    
    opt_actor = optim.Adam(actor.parameters(), lr=config.policy_lr)
    opt_critic = optim.Adam(critic.parameters(), lr=config.critic_lr)
    opt_metric = optim.Adam(metric.parameters(), lr=config.metric_lr)
    
    # Training history
    returns_history = []
    violations_history = []
    total_violations = 0
    total_goals = 0
    
    for ep in range(config.n_episodes):
        obs = env.reset()
        done = False
        trajectory = []
        ep_return = 0.0
        ep_violations = 0
        
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
                old_lp = dist.log_prob(action).sum()
            
            next_obs, reward, done, info = env.step(action.squeeze(0).cpu().numpy())
            
            trajectory.append({
                "obs": obs,
                "next_obs": next_obs,
                "action": action,
                "reward": reward,
                "old_lp": old_lp.item(),
                "in_trap": info["in_trap"],
            })
            
            obs = next_obs
            ep_return += reward
            if info["in_trap"]:
                ep_violations += 1
            if info["goal_reached"]:
                total_goals += 1
        
        returns_history.append(ep_return)
        violations_history.append(ep_violations)
        total_violations += ep_violations
        
        # Convert trajectory to tensors
        states = torch.FloatTensor(np.array([t["obs"] for t in trajectory])).to(device)
        next_states = torch.FloatTensor(np.array([t["next_obs"] for t in trajectory])).to(device)
        actions = torch.cat([t["action"] for t in trajectory]).to(device)
        rewards = torch.FloatTensor([t["reward"] for t in trajectory]).to(device)
        old_lps = torch.FloatTensor([t["old_lp"] for t in trajectory]).to(device)
        
        # Critic update
        with torch.no_grad():
            V_next = critic(next_states).squeeze()
        V = critic(states).squeeze()
        targets = rewards + config.gamma * V_next
        critic_loss = F.mse_loss(V, targets)
        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()
        
        # Metric update (after warmup)
        if ep >= config.warmup_episodes:
            # Train metric to predict danger from trap proximity
            danger_targets = torch.zeros(len(states), device=device)
            for i, t in enumerate(trajectory):
                if t["in_trap"]:
                    danger_targets[i] = 1.0
            
            g, _, _ = metric(states, actions)
            metric_loss = F.mse_loss(g, danger_targets * config.metric_severity + 1.0)
            opt_metric.zero_grad()
            metric_loss.backward()
            opt_metric.step()
        
        # Compute advantage with Hodge correction
        with torch.no_grad():
            V = critic(states).squeeze()
            V_next = critic(next_states).squeeze()
            
            # Get metric and harmonic
            g, escape_factors, omega_invalid = metric(states, actions, context=states)
            
            # TD error with Hodge harmonic subtraction
            td_error = rewards + config.gamma * V_next - V
            if use_hodge:
                td_error = td_error - omega_invalid
            
            # Anisotropic scaling
            if config.use_soft_scaling:
                scale = escape_factors + (1 - escape_factors) / (1.0 + torch.log(1.0 + g))
            else:
                scale = escape_factors + (1 - escape_factors) / torch.sqrt(g)
            
            adv = scale * td_error
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        # Actor update (PPO-style)
        new_lps = actor(states).log_prob(actions).sum(dim=-1)
        ratio = torch.exp(new_lps - old_lps)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * adv
        actor_loss = -torch.min(surr1, surr2).mean()
        
        entropy = actor(states).entropy().sum(dim=-1).mean()
        actor_loss = actor_loss - config.entropy_coef * entropy
        
        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()
        
        if ep % 50 == 0:
            print(f"  Ep {ep}: Return={ep_return:.1f}, Violations={ep_violations}, "
                  f"Total Viol={total_violations}, Goals={total_goals}")
    
    # Compute final metrics
    final_return = np.mean(returns_history[-100:])
    goal_rate = total_goals / config.n_episodes
    
    method = "sgpo_anis_cchc" if use_hodge else "sgpo_anis"
    
    return ExperimentCResults(
        seed=seed,
        method=method,
        total_violations=total_violations,
        final_return=final_return,
        goal_rate=goal_rate,
        h1_filtered_ratio=1.0 if use_hodge else 0.0,
        training_returns=returns_history,
        training_violations=violations_history,
    )


# =============================================================================
# Experiment Runners
# =============================================================================

def run_experiment_a(config: ExperimentConfig) -> List[ExperimentAResults]:
    """
    Run Experiment A: Pre-filtered reward models.
    
    Compares reward models trained on:
    1. Raw preferences (all data)
    2. H¹-filtered preferences (invalid cycles removed)
    
    Both models are trained on the SAME NUMBER of samples.
    """
    print("=" * 60)
    print("EXPERIMENT A: Pre-filtered Reward Models")
    print("=" * 60)
    
    device = torch.device(config.device)
    
    # Load embedding model
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(config.embedding_model_name)
        embed_dim = embedding_model.get_sentence_embedding_dimension()
    except ImportError:
        print("Warning: sentence-transformers not installed. Using mock embeddings.")
        embedding_model = None
        embed_dim = 384
    
    results = []
    
    for seed in range(config.seeds):
        print(f"\n--- Seed {seed + 1}/{config.seeds} ---")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Load data with context
        print("Loading HH-RLHF data...")
        try:
            items = load_hh_rlhf_with_context(config.n_samples)
        except Exception as e:
            print(f"Could not load HH-RLHF: {e}. Using synthetic data.")
            items = _generate_synthetic_contextual_data(config.n_samples, embed_dim)
        
        # Create critic and compute conditional H¹
        print("Computing conditional H¹...")
        if embedding_model is not None:
            critic = ContextConditionalHodgeCritic(embedding_model, embed_dim)
        else:
            critic = _create_mock_critic(embed_dim)
        
        critic.add_contextual_feedback(items)
        h1_result = critic.compute_conditional_h1()
        print(f"  {h1_result}")
        
        # Create normalized train sets
        raw_items, filtered_items = create_normalized_train_sets(
            items, critic, config.h1_threshold
        )
        n_train = int(len(raw_items) * config.train_ratio)
        
        print(f"  Raw items: {len(raw_items)}, Filtered items: {len(filtered_items)}")
        print(f"  Training on {n_train} samples each")
        
        # Embed items
        if embedding_model is not None:
            raw_chosen_embs = embedding_model.encode([i.chosen_text or i.state_text for i in raw_items[:n_train]])
            raw_rejected_embs = embedding_model.encode([i.rejected_text or "" for i in raw_items[:n_train]])
            filtered_chosen_embs = embedding_model.encode([i.chosen_text or i.state_text for i in filtered_items[:n_train]])
            filtered_rejected_embs = embedding_model.encode([i.rejected_text or "" for i in filtered_items[:n_train]])
            
            # Test set
            test_chosen_embs = embedding_model.encode([i.chosen_text or i.state_text for i in raw_items[n_train:]])
            test_rejected_embs = embedding_model.encode([i.rejected_text or "" for i in raw_items[n_train:]])
        else:
            # Mock embeddings
            raw_chosen_embs = np.random.randn(n_train, embed_dim)
            raw_rejected_embs = np.random.randn(n_train, embed_dim)
            filtered_chosen_embs = np.random.randn(n_train, embed_dim)
            filtered_rejected_embs = np.random.randn(n_train, embed_dim)
            test_chosen_embs = np.random.randn(len(raw_items) - n_train, embed_dim)
            test_rejected_embs = np.random.randn(len(raw_items) - n_train, embed_dim)
        
        # Convert to tensors
        raw_train = (
            torch.tensor(raw_chosen_embs, dtype=torch.float32),
            torch.tensor(raw_rejected_embs, dtype=torch.float32),
        )
        filtered_train = (
            torch.tensor(filtered_chosen_embs, dtype=torch.float32),
            torch.tensor(filtered_rejected_embs, dtype=torch.float32),
        )
        test_data = (
            torch.tensor(test_chosen_embs, dtype=torch.float32),
            torch.tensor(test_rejected_embs, dtype=torch.float32),
        )
        
        # Train raw model
        print("Training raw reward model...")
        raw_model = PreferenceRewardModel(embed_dim, config.reward_hidden_dim).to(device)
        raw_loss = train_reward_model(raw_model, raw_train, config, device)
        
        # Train filtered model
        print("Training filtered reward model...")
        filtered_model = PreferenceRewardModel(embed_dim, config.reward_hidden_dim).to(device)
        filtered_loss = train_reward_model(filtered_model, filtered_train, config, device)
        
        # Evaluate
        raw_acc, raw_exploit = evaluate_reward_model(raw_model, test_data, device)
        filtered_acc, filtered_exploit = evaluate_reward_model(filtered_model, test_data, device)
        
        print(f"  Raw: Loss={raw_loss:.4f}, Acc={raw_acc:.2%}, Exploit={raw_exploit:.2%}")
        print(f"  Filtered: Loss={filtered_loss:.4f}, Acc={filtered_acc:.2%}, Exploit={filtered_exploit:.2%}")
        
        results.append(ExperimentAResults(
            seed=seed,
            raw_reward_loss=raw_loss,
            filtered_reward_loss=filtered_loss,
            raw_h1_initial=h1_result.marginal_h1,
            filtered_h1_initial=h1_result.conditional_h1,
            raw_exploitation_rate=raw_exploit,
            filtered_exploitation_rate=filtered_exploit,
            raw_test_accuracy=raw_acc,
            filtered_test_accuracy=filtered_acc,
            n_train_raw=n_train,
            n_train_filtered=n_train,
            conditional_h1_analysis={
                "marginal_h1": h1_result.marginal_h1,
                "conditional_h1": h1_result.conditional_h1,
                "valid_contextual_h1": h1_result.valid_contextual_h1,
                "n_contexts": h1_result.n_contexts,
            },
        ))
    
    return results


def run_experiment_c(config: ExperimentConfig) -> List[ExperimentCResults]:
    """
    Run Experiment C: Context-conditional SGPO.
    
    Compares:
    1. SGPO_ANIS (baseline)
    2. SGPO_ANIS_HODGE (with context-conditional harmonic discounting)
    """
    print("=" * 60)
    print("EXPERIMENT C: Context-Conditional SGPO")
    print("=" * 60)
    
    device = torch.device(config.device)
    
    # Load embedding model
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(config.embedding_model_name)
        embed_dim = embedding_model.get_sentence_embedding_dimension()
    except ImportError:
        print("Warning: sentence-transformers not installed. Using mock embeddings.")
        embedding_model = None
        embed_dim = 384
    
    results = []
    
    for seed in range(config.seeds):
        print(f"\n--- Seed {seed + 1}/{config.seeds} ---")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Create a simple reward model for the environment
        reward_model = PreferenceRewardModel(embed_dim, config.reward_hidden_dim).to(device)
        
        # Create environment
        env = PreferenceBasedEnv(
            reward_model=reward_model,
            embed_dim=embed_dim,
            max_steps=config.max_steps,
            device=device,
        )
        
        # Create Hodge critic (trained on synthetic data for demo)
        if embedding_model is not None:
            hodge_critic = ContextConditionalHodgeCritic(embedding_model, embed_dim, device=device)
        else:
            hodge_critic = _create_mock_critic(embed_dim)
        
        # Train SGPO_ANIS (baseline)
        print("Training SGPO_ANIS (baseline)...")
        baseline_result = train_sgpo_anis_hodge(
            env, config, hodge_critic=None, use_hodge=False, seed=seed
        )
        results.append(baseline_result)
        
        # Train SGPO_ANIS_HODGE
        print("Training SGPO_ANIS_CCHC...")
        hodge_result = train_sgpo_anis_hodge(
            env, config, hodge_critic=hodge_critic, use_hodge=True, seed=seed
        )
        results.append(hodge_result)
    
    return results


# =============================================================================
# Helper Functions
# =============================================================================

def _generate_synthetic_contextual_data(
    n_samples: int,
    embed_dim: int,
) -> List[ContextualFeedbackItem]:
    """Generate synthetic contextual preference data for testing."""
    items = []
    n_contexts = max(10, n_samples // 50)
    
    for i in range(n_samples):
        context_id = str(i % n_contexts)
        item = ContextualFeedbackItem(
            state_text=f"prompt_{i}",
            action_text="response",
            next_state_text=None,
            rank=np.random.random(),
            context_id=context_id,
            chosen_text=f"chosen_{i}",
            rejected_text=f"rejected_{i}",
        )
        items.append(item)
    
    return items


def _create_mock_critic(embed_dim: int) -> ContextConditionalHodgeCritic:
    """Create mock critic for testing without embedding model."""
    class MockEmbedder:
        def encode(self, texts):
            return np.random.randn(len(texts), embed_dim)
    
    return ContextConditionalHodgeCritic(MockEmbedder(), embed_dim)


def save_results(results: List, experiment_name: str, config: ExperimentConfig):
    """Save results to JSON file."""
    os.makedirs(config.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(config.output_dir, filename)
    
    # Convert to serializable format
    results_dict = {
        "experiment": experiment_name,
        "config": asdict(config),
        "results": [asdict(r) for r in results],
    }
    
    with open(filepath, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")


def print_summary(results_a: List[ExperimentAResults] = None, results_c: List[ExperimentCResults] = None):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if results_a:
        print("\n--- Experiment A: Pre-filtered Reward Models ---")
        raw_exploit = np.mean([r.raw_exploitation_rate for r in results_a])
        filtered_exploit = np.mean([r.filtered_exploitation_rate for r in results_a])
        raw_acc = np.mean([r.raw_test_accuracy for r in results_a])
        filtered_acc = np.mean([r.filtered_test_accuracy for r in results_a])
        
        print(f"Raw Model:      Accuracy={raw_acc:.2%} ± {np.std([r.raw_test_accuracy for r in results_a]):.2%}, "
              f"Exploitation={raw_exploit:.2%}")
        print(f"Filtered Model: Accuracy={filtered_acc:.2%} ± {np.std([r.filtered_test_accuracy for r in results_a]):.2%}, "
              f"Exploitation={filtered_exploit:.2%}")
        print(f"Exploitation Reduction: {(raw_exploit - filtered_exploit) / (raw_exploit + 1e-8):.1%}")
    
    if results_c:
        print("\n--- Experiment C: Context-Conditional SGPO ---")
        for method in ["sgpo_anis", "sgpo_anis_cchc"]:
            method_results = [r for r in results_c if r.method == method]
            if method_results:
                violations = np.mean([r.total_violations for r in method_results])
                violations_std = np.std([r.total_violations for r in method_results])
                returns = np.mean([r.final_return for r in method_results])
                goals = np.mean([r.goal_rate for r in method_results])
                
                print(f"{method:20s}: Violations={violations:.1f} ± {violations_std:.1f}, "
                      f"Return={returns:.2f}, Goals={goals:.1%}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hodge-Filtered RLHF Experiments")
    parser.add_argument("--experiment", type=str, default="both",
                        choices=["A", "C", "both"],
                        help="Which experiment to run")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of preference samples")
    parser.add_argument("--h1-threshold", type=float, default=0.8,
                        help="H¹ threshold for filtering")
    parser.add_argument("--n-episodes", type=int, default=200,
                        help="Training episodes for Experiment C")
    parser.add_argument("--output-dir", type=str, default="results/hodge_filtered_rlhf",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode")
    
    args = parser.parse_args()
    
    # Create config
    config = ExperimentConfig(
        seeds=args.seeds if not args.quick else 2,
        n_samples=args.n_samples if not args.quick else 200,
        h1_threshold=args.h1_threshold,
        n_episodes=args.n_episodes if not args.quick else 50,
        output_dir=args.output_dir,
    )
    
    results_a = None
    results_c = None
    
    if args.experiment in ["A", "both"]:
        results_a = run_experiment_a(config)
        save_results(results_a, "experiment_a", config)
    
    if args.experiment in ["C", "both"]:
        results_c = run_experiment_c(config)
        save_results(results_c, "experiment_c", config)
    
    print_summary(results_a, results_c)


if __name__ == "__main__":
    main()
