#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modal GPU Runner for Corrected Experiments

This script runs both Experiment A (preference filtering with HH-RLHF) and 
Experiment C (conformal safety with baselines) on Modal GPUs.

Run locally:
    python run_experiments_modal.py --local --mode quick

Run on Modal:
    modal run run_experiments_modal.py --mode full
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from corrected_experiments import (
    DiscreteHodgeRank,
    PreferenceFilter,
    FilteringConfig,
    FixedSandbaggingEnv,
    FixedEnvConfig,
    run_experiment_a_variant,
    ExperimentAResult,
    HodgeComponents,
)

from conformal_sgpo import (
    ConformalSafetyMetric,
    ConformalSGPOConfig,
    train_conformal_sgpo,
    train_conformal_sgpo_anis,
)


# ============================================================================
# IMPROVED SYNTHETIC DATA WITH BOTH CURL AND HARMONIC CYCLES
# ============================================================================

def create_improved_synthetic_preferences(
    n_contexts: int = 100,
    curl_cycle_rate: float = 0.3,  # 30% have 3-cycles (curl)
    harmonic_cycle_rate: float = 0.3,  # 30% have 5-cycles (harmonic)
    seed: int = 42,
) -> List[Dict]:
    """
    Create synthetic preferences with BOTH curl (3-cycles) and harmonic (5-cycles).
    
    This ensures each filter type has distinct effects:
    - curl_only: filters contexts with 3-clique cycles
    - harmonic_only: filters contexts with larger global cycles
    - reliability_score: filters both
    """
    np.random.seed(seed)
    preferences = []
    
    for ctx_id in range(n_contexts):
        prompt = f"Context_{ctx_id}"
        
        # Decide cycle type for this context
        r = np.random.rand()
        
        if r < curl_cycle_rate:
            # Create 3-cycle (curl): A > B > C > A
            items = [f"Item_{ctx_id}_{i}" for i in range(3)]
            preferences.append({'prompt': prompt, 'chosen': items[1], 'rejected': items[0]})  # B > A
            preferences.append({'prompt': prompt, 'chosen': items[2], 'rejected': items[1]})  # C > B
            preferences.append({'prompt': prompt, 'chosen': items[0], 'rejected': items[2]})  # A > C (cycle)
            
        elif r < curl_cycle_rate + harmonic_cycle_rate:
            # Create 5-cycle (harmonic): A > B > C > D > E > A
            items = [f"Item_{ctx_id}_{i}" for i in range(5)]
            for i in range(4):
                preferences.append({'prompt': prompt, 'chosen': items[i+1], 'rejected': items[i]})
            preferences.append({'prompt': prompt, 'chosen': items[0], 'rejected': items[4]})  # A > E (cycle)
            
        else:
            # Transitive (consistent): A < B < C < D < E with some transitive edges
            items = [f"Item_{ctx_id}_{i}" for i in range(5)]
            for i in range(4):
                preferences.append({'prompt': prompt, 'chosen': items[i+1], 'rejected': items[i]})
            # Add some transitive edges for higher reliability
            preferences.append({'prompt': prompt, 'chosen': items[2], 'rejected': items[0]})  # C > A
            preferences.append({'prompt': prompt, 'chosen': items[4], 'rejected': items[2]})  # E > C
    
    return preferences


# ============================================================================
# HH-RLHF DATA INTEGRATION
# ============================================================================

def load_hh_rlhf_data(n_samples: int = 5000, seed: int = 42) -> List[Dict]:
    """
    Load Anthropic HH-RLHF dataset.
    
    Returns list of {'prompt': str, 'chosen': str, 'rejected': str}
    """
    try:
        from datasets import load_dataset
        
        print("Loading HH-RLHF dataset...")
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        
        # Sample
        np.random.seed(seed)
        indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
        
        preferences = []
        for idx in indices:
            example = dataset[int(idx)]
            # Extract prompt from chosen (first turn)
            chosen = example['chosen']
            rejected = example['rejected']
            
            # Find common prefix (prompt)
            # Format: "\n\nHuman: ... \n\nAssistant: ..."
            prompt_end = chosen.find("\n\nAssistant:")
            if prompt_end > 0:
                prompt = chosen[:prompt_end]
            else:
                prompt = f"context_{idx}"
            
            preferences.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
            })
        
        print(f"  Loaded {len(preferences)} preference pairs")
        return preferences
        
    except ImportError:
        print("  datasets library not available, using synthetic data")
        return create_improved_synthetic_preferences(n_samples // 5, seed=seed)
    except Exception as e:
        print(f"  Error loading HH-RLHF: {e}, using synthetic data")
        return create_improved_synthetic_preferences(n_samples // 5, seed=seed)


# ============================================================================
# EXPERIMENT A: PREFERENCE FILTERING WITH REWARD MODEL TRAINING
# ============================================================================

class SimpleRewardModel(nn.Module):
    """Simple MLP reward model for preference learning."""
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_reward_model(
    preferences: List[Dict],
    embed_fn,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[SimpleRewardModel, Dict]:
    """
    Train reward model on preferences.
    
    Returns trained model and training metrics.
    """
    # Embed all texts
    all_texts = []
    for p in preferences:
        all_texts.extend([p['chosen'], p['rejected']])
    
    embeddings = embed_fn(all_texts)
    embed_dim = embeddings.shape[1]
    
    # Create model
    model = SimpleRewardModel(input_dim=embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        for i, p in enumerate(preferences):
            chosen_emb = torch.tensor(embeddings[2*i], device=device).unsqueeze(0)
            rejected_emb = torch.tensor(embeddings[2*i + 1], device=device).unsqueeze(0)
            
            r_chosen = model(chosen_emb)
            r_rejected = model(rejected_emb)
            
            # Bradley-Terry loss
            loss = -torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-8)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if r_chosen > r_rejected:
                correct += 1
        
        losses.append(total_loss / len(preferences))
    
    accuracy = correct / len(preferences)
    return model, {"final_loss": losses[-1], "accuracy": accuracy}


def compute_exploitation_rate(
    model: SimpleRewardModel,
    test_preferences: List[Dict],
    embed_fn,
    device: str = "cpu",
) -> float:
    """
    Compute exploitation rate: fraction where model prefers rejected.
    
    Lower is better (model correctly prefers chosen).
    """
    all_texts = []
    for p in test_preferences:
        all_texts.extend([p['chosen'], p['rejected']])
    
    embeddings = embed_fn(all_texts)
    
    exploited = 0
    model.eval()
    with torch.no_grad():
        for i in range(len(test_preferences)):
            chosen_emb = torch.tensor(embeddings[2*i], device=device).unsqueeze(0)
            rejected_emb = torch.tensor(embeddings[2*i + 1], device=device).unsqueeze(0)
            
            r_chosen = model(chosen_emb)
            r_rejected = model(rejected_emb)
            
            if r_rejected > r_chosen:
                exploited += 1
    
    return exploited / len(test_preferences)


def run_experiment_a_full(
    seeds: int = 50,
    n_train_samples: int = 3000,
    n_test_samples: int = 1000,
    use_hh_rlhf: bool = True,
    device: str = "cpu",
    seed_offset: int = 0,
) -> pd.DataFrame:
    """Run full Experiment A with reward model training and exploitation measurement."""
    print("\n" + "="*60)
    print("EXPERIMENT A: Preference Filtering (Full)")
    print("="*60)
    
    # Try to load embedding model
    try:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        embed_fn = lambda texts: embed_model.encode(texts, show_progress_bar=False)
        embed_dim = 384
        print("  Using sentence-transformers embeddings")
    except ImportError:
        print("  sentence-transformers not available, using random embeddings")
        def random_embed(texts):
            return np.random.randn(len(texts), 128).astype(np.float32)
        embed_fn = random_embed
        embed_dim = 128
    
    methods = ["raw", "harmonic_only", "curl_only", "reliability_score"]
    results = []
    
    for seed in range(seeds):
        actual_seed = seed + seed_offset
        print(f"\n--- Seed {actual_seed+1}/{seed_offset+seeds} ---")
        np.random.seed(actual_seed)
        torch.manual_seed(actual_seed)
        
        # Load data
        if use_hh_rlhf:
            all_prefs = load_hh_rlhf_data(n_train_samples + n_test_samples, seed=actual_seed)
        else:
            all_prefs = create_improved_synthetic_preferences(
                n_contexts=(n_train_samples + n_test_samples) // 5,
                curl_cycle_rate=0.2,
                harmonic_cycle_rate=0.2,
                seed=actual_seed,
            )
        
        # Split train/test
        np.random.shuffle(all_prefs)
        train_prefs = all_prefs[:n_train_samples]
        test_prefs = all_prefs[n_train_samples:n_train_samples + n_test_samples]
        
        for method in methods:
            print(f"  Method: {method}...", end=" ")
            
            # Apply filter
            config = FilteringConfig(
                method=method if method != "raw" else "reliability_score",
                threshold=0.5,
                h1_threshold=0.8,
            )
            
            if method == "raw":
                filtered_prefs = train_prefs
            else:
                # Group by context and filter
                contexts = {}
                pref_by_ctx = {}
                for p in train_prefs:
                    ctx_id = hash(p['prompt'])
                    if ctx_id not in contexts:
                        contexts[ctx_id] = []
                        pref_by_ctx[ctx_id] = []
                    chosen_id = hash(p['chosen'])
                    rejected_id = hash(p['rejected'])
                    contexts[ctx_id].append((chosen_id, rejected_id, 1.0))
                    pref_by_ctx[ctx_id].append(p)
                
                filt = PreferenceFilter(config)
                filtered_contexts, components = filt.filter_preferences(contexts)
                
                filtered_prefs = []
                for ctx_id in filtered_contexts:
                    filtered_prefs.extend(pref_by_ctx[ctx_id])
            
            if len(filtered_prefs) < 10:
                print(f"Too few samples ({len(filtered_prefs)}), skipping")
                continue
            
            # Train reward model
            model, train_metrics = train_reward_model(
                filtered_prefs, embed_fn, epochs=5, device=device
            )
            
            # Compute exploitation rate
            exploitation = compute_exploitation_rate(model, test_prefs, embed_fn, device)
            
            # Compute Hodge stats on filtered data
            hodge = DiscreteHodgeRank()
            contexts_for_stats = {}
            for p in filtered_prefs:
                ctx_id = hash(p['prompt'])
                if ctx_id not in contexts_for_stats:
                    contexts_for_stats[ctx_id] = []
                contexts_for_stats[ctx_id].append(
                    (hash(p['chosen']), hash(p['rejected']), 1.0)
                )
            
            reliabilities, curls, harmonics = [], [], []
            for ctx_id, comps in contexts_for_stats.items():
                items = set()
                for i, j, _ in comps:
                    items.add(i)
                    items.add(j)
                item_to_idx = {item: idx for idx, item in enumerate(items)}
                remapped = [(item_to_idx[i], item_to_idx[j], w) for i, j, w in comps]
                comp = hodge.decompose(len(items), remapped)
                reliabilities.append(comp.reliability_score)
                curls.append(comp.curl_ratio)
                harmonics.append(comp.harmonic_ratio)
            
            results.append({
                "seed": seed,
                "method": method,
                "n_train": len(filtered_prefs),
                "accuracy": train_metrics["accuracy"],
                "exploitation_rate": exploitation,
                "avg_reliability": np.mean(reliabilities) if reliabilities else 0,
                "avg_curl_ratio": np.mean(curls) if curls else 0,
                "avg_harmonic_ratio": np.mean(harmonics) if harmonics else 0,
            })
            
            print(f"n={len(filtered_prefs)}, acc={train_metrics['accuracy']:.3f}, exploit={exploitation:.3f}")
    
    return pd.DataFrame(results)


# ============================================================================
# EXPERIMENT C: BASELINES (PPO, CPO) + CONFORMAL SGPO
# ============================================================================

class Actor(nn.Module):
    """Simple policy network."""
    def __init__(self, state_dim: int = 2, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor):
        h = self.net(state)
        mean = self.mean(h)
        std = self.log_std.exp()
        return mean, std
    
    def get_action(self, state: torch.Tensor):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-1, 1), log_prob


class Critic(nn.Module):
    """Value function network."""
    def __init__(self, state_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


def train_ppo(
    env,
    episodes: int = 300,
    seed: int = 0,
) -> Dict:
    """Train vanilla PPO baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    actor = Actor()
    critic = Critic()
    actor_optim = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-3)
    
    episode_returns = []
    episode_violations = []
    goal_reached = []
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        states, actions, rewards, log_probs, values = [], [], [], [], []
        violations = 0
        
        while not done:
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, log_prob = actor.get_action(state)
            value = critic(state)
            
            obs_next, reward, cost, done, info = env.step(action.squeeze(0).numpy())
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            violations += int(info.get('in_trap', False))
            
            obs = obs_next
        
        # PPO update
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        values_t = torch.cat(values)
        advantages = returns - values_t.detach()
        
        # Policy update
        log_probs_t = torch.cat(log_probs)
        policy_loss = -(advantages * log_probs_t).mean()
        
        actor_optim.zero_grad()
        policy_loss.backward()
        actor_optim.step()
        
        # Value update
        value_loss = ((values_t - returns) ** 2).mean()
        critic_optim.zero_grad()
        value_loss.backward()
        critic_optim.step()
        
        episode_returns.append(sum(rewards))
        episode_violations.append(violations)
        goal_reached.append(info.get('reached_goal', False))
    
    return {
        "episode_returns": episode_returns,
        "episode_violations": episode_violations,
        "goal_reached": goal_reached,
        "n_hardened_regions": 0,
    }


def train_cpo(
    env,
    episodes: int = 300,
    cost_limit: float = 0.1,
    seed: int = 0,
) -> Dict:
    """Train CPO baseline with Lagrangian constraint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    actor = Actor()
    critic = Critic()
    cost_critic = Critic()
    actor_optim = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-3)
    cost_critic_optim = optim.Adam(cost_critic.parameters(), lr=1e-3)
    
    lagrange_mult = torch.tensor(1.0, requires_grad=True)
    lagrange_optim = optim.Adam([lagrange_mult], lr=0.01)
    
    episode_returns = []
    episode_violations = []
    goal_reached = []
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        states, actions, rewards, costs_list, log_probs, values, cost_values = [], [], [], [], [], [], []
        violations = 0
        
        while not done:
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, log_prob = actor.get_action(state)
            value = critic(state)
            cost_value = cost_critic(state)
            
            obs_next, reward, cost, done, info = env.step(action.squeeze(0).numpy())
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            costs_list.append(cost)
            log_probs.append(log_prob)
            values.append(value)
            cost_values.append(cost_value)
            violations += int(info.get('in_trap', False))
            
            obs = obs_next
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        cost_returns = []
        C = 0
        for c in reversed(costs_list):
            C = c + 0.99 * C
            cost_returns.insert(0, C)
        cost_returns = torch.tensor(cost_returns, dtype=torch.float32)
        
        values_t = torch.cat(values)
        cost_values_t = torch.cat(cost_values)
        
        # CPO: reward - lambda * cost
        advantages = returns - values_t.detach() - lagrange_mult.detach() * (cost_returns - cost_values_t.detach())
        
        # Policy update
        log_probs_t = torch.cat(log_probs)
        policy_loss = -(advantages * log_probs_t).mean()
        
        actor_optim.zero_grad()
        policy_loss.backward()
        actor_optim.step()
        
        # Value updates
        value_loss = ((values_t - returns) ** 2).mean()
        critic_optim.zero_grad()
        value_loss.backward()
        critic_optim.step()
        
        cost_value_loss = ((cost_values_t - cost_returns) ** 2).mean()
        cost_critic_optim.zero_grad()
        cost_value_loss.backward()
        cost_critic_optim.step()
        
        # Lagrange update
        constraint_violation = cost_returns.mean() - cost_limit
        lagrange_loss = -lagrange_mult * constraint_violation
        lagrange_optim.zero_grad()
        lagrange_loss.backward()
        lagrange_optim.step()
        lagrange_mult.data.clamp_(0.0, 100.0)
        
        episode_returns.append(sum(rewards))
        episode_violations.append(violations)
        goal_reached.append(info.get('reached_goal', False))
    
    return {
        "episode_returns": episode_returns,
        "episode_violations": episode_violations,
        "goal_reached": goal_reached,
        "n_hardened_regions": 0,
    }


def run_experiment_c_full(
    seeds: int = 50,
    episodes: int = 300,
    seed_offset: int = 0,
) -> pd.DataFrame:
    """Run full Experiment C with all methods including baselines."""
    print("\n" + "="*60)
    print("EXPERIMENT C: Conformal Safety (Full with Baselines)")
    print("="*60)
    
    methods = ["ppo", "cpo", "conformal_sgpo", "conformal_sgpo_anis"]
    results = []
    
    for method in methods:
        print(f"\n--- Method: {method} ---")
        
        for seed in range(seeds):
            actual_seed = seed + seed_offset
            print(f"  Seed {actual_seed+1}/{seed_offset+seeds}...", end=" ")
            
            env = FixedSandbaggingEnv()
            trap_center, trap_radius = env.get_trap_info()
            known_regions = [(trap_center, trap_radius)]
            
            if method == "ppo":
                result = train_ppo(env, episodes=episodes, seed=actual_seed)
            elif method == "cpo":
                result = train_cpo(env, episodes=episodes, seed=actual_seed)
            elif method == "conformal_sgpo":
                config = ConformalSGPOConfig(episodes=min(episodes, 150), sharpness=4.0, anisotropic=False)
                result = train_conformal_sgpo(env, config, actual_seed, known_regions)
            elif method == "conformal_sgpo_anis":
                config = ConformalSGPOConfig(episodes=min(episodes, 150), sharpness=4.0, anisotropic=True)
                result = train_conformal_sgpo_anis(env, config, actual_seed, known_regions)
            
            violations = sum(result["episode_violations"])
            goal_rate = np.mean(result["goal_reached"])
            final_return = np.mean(result["episode_returns"][-50:])
            
            results.append({
                "seed": actual_seed,
                "method": method,
                "violations": violations,
                "goal_rate": goal_rate,
                "final_return": final_return,
                "n_hardened_regions": result.get("n_hardened_regions", 0),
            })
            
            print(f"violations={violations}, goal_rate={goal_rate:.2%}")
    
    return pd.DataFrame(results)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run corrected experiments")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--experiment", choices=["A", "C", "both"], default="both")
    parser.add_argument("--local", action="store_true", help="Run locally instead of Modal")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data instead of HH-RLHF")
    parser.add_argument("--output-dir", type=str, default="results/corrected_v2")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    seeds = 5 if args.mode == "quick" else 50
    episodes = 100 if args.mode == "quick" else 300
    
    print("\n" + "="*60)
    print("CORRECTED EXPERIMENTS V2")
    print("="*60)
    print(f"Mode: {args.mode} ({seeds} seeds)")
    print(f"Experiment: {args.experiment}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.experiment in ["A", "both"]:
        df_a = run_experiment_a_full(
            seeds=seeds,
            n_train_samples=1000 if args.mode == "quick" else 3000,
            n_test_samples=500 if args.mode == "quick" else 1000,
            use_hh_rlhf=not args.use_synthetic,
            device=args.device,
        )
        
        output_file = output_dir / f"experiment_a_{args.mode}_{timestamp}.csv"
        df_a.to_csv(output_file, index=False)
        print(f"\nSaved Experiment A to: {output_file}")
        
        # Summary
        print("\n--- Experiment A Summary ---")
        summary = df_a.groupby("method").agg({
            "n_train": ["mean", "std"],
            "accuracy": ["mean", "std"],
            "exploitation_rate": ["mean", "std"],
        })
        print(summary.round(4))
    
    if args.experiment in ["C", "both"]:
        df_c = run_experiment_c_full(seeds=seeds, episodes=episodes)
        
        output_file = output_dir / f"experiment_c_{args.mode}_{timestamp}.csv"
        df_c.to_csv(output_file, index=False)
        print(f"\nSaved Experiment C to: {output_file}")
        
        # Summary
        print("\n--- Experiment C Summary ---")
        summary = df_c.groupby("method").agg({
            "violations": ["mean", "std"],
            "goal_rate": ["mean", "std"],
        })
        print(summary.round(4))
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
