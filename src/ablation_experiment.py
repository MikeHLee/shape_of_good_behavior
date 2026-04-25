"""
ABLATION EXPERIMENTS: Hodge Critic & Geometric Safety
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from typing import Dict, List, Any

# Import from existing experiments
from src.condorcet_experiment import (
    CondorcetRingEnv, 
    train_ppo as train_ppo_ring, 
    train_gpo as train_gpo_ring,
    compute_empirical_h1
)
from src.safety_experiment import (
    SandbaggingEnv, 
    train_gpo as train_gpo_safety,
    RiemannianMetric
)

def ablate_hodge_critic():
    """
    Ablation 1: Effect of Cycle Strength on Hodge vs Scalar Critic
    Vary the base_reward (cycle strength) and measure performance difference.
    """
    print("\n--- Ablation 1: Cycle Strength (Hodge vs Scalar) ---")
    
    cycle_strengths = [0.1, 0.5, 1.0, 2.0]
    results = []
    
    for strength in cycle_strengths:
        print(f"Testing Cycle Strength: {strength}")
        env = CondorcetRingEnv(base_reward=strength, noise_std=0.05)
        
        # Train PPO (Scalar)
        _, _, ppo_returns, _ = train_ppo_ring(env, episodes=200)
        ppo_perf = np.mean(ppo_returns[-50:])
        
        # Train SGPO (Hodge)
        _, gpo_critic, gpo_returns, _ = train_gpo_ring(env, episodes=200)
        gpo_perf = np.mean(gpo_returns[-50:])
        
        learned_h1 = gpo_critic.harmonic_coeff.item()
        
        results.append({
            "cycle_strength": strength,
            "ppo_return": ppo_perf,
            "gpo_return": gpo_perf,
            "learned_h1": learned_h1,
            "improvement": gpo_perf - ppo_perf
        })
        
    return results

def ablate_metric_parameters():
    """
    Ablation 2: Effect of Event Horizon Size on Safety
    Vary the event_horizon radius and measure violations.
    """
    print("\n--- Ablation 2: Event Horizon Size (Safety) ---")
    
    horizons = [1.0, 1.5, 2.0, 2.5]
    results = []
    
    for horizon in horizons:
        print(f"Testing Event Horizon: {horizon}")
        env = SandbaggingEnv()
        env.event_horizon = horizon # Override
        
        # Train SGPO
        _, _, violations, returns, goals, _ = train_gpo_safety(env, episodes=200)
        
        results.append({
            "event_horizon": horizon,
            "total_violations": sum(violations),
            "final_success_rate": np.mean(goals[-50:]),
            "final_return": np.mean(returns[-50:])
        })
        
    return results

def main():
    print("Running Ablation Experiments...")
    
    # Run Ablations
    hodge_results = ablate_hodge_critic()
    safety_results = ablate_metric_parameters()
    
    # Save Data
    all_results = {
        "hodge_ablation": hodge_results,
        "safety_ablation": safety_results
    }
    
    with open("../results/safety/ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nAblation results saved to ../results/safety/ablation_results.json")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Hodge vs Scalar
    ax1 = axes[0]
    df_hodge = pd.DataFrame(hodge_results)
    ax1.plot(df_hodge["cycle_strength"], df_hodge["ppo_return"], 'r-o', label="Scalar Critic (PPO)")
    ax1.plot(df_hodge["cycle_strength"], df_hodge["gpo_return"], 'b-o', label="Hodge Critic (SGPO)")
    ax1.set_xlabel("Cycle Strength (Base Reward)")
    ax1.set_ylabel("Final Mean Return")
    ax1.set_title("Critic Performance vs Cycle Strength")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Safety Horizon
    ax2 = axes[1]
    df_safety = pd.DataFrame(safety_results)
    ax2.bar(df_safety["event_horizon"].astype(str), df_safety["total_violations"], color='orange')
    ax2.set_xlabel("Event Horizon Radius")
    ax2.set_ylabel("Total Trap Violations")
    ax2.set_title("Safety vs Event Horizon Size")
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig("../figures/experiments/ablation_plots.png")
    print("Plots saved to ../figures/experiments/ablation_plots.png")

if __name__ == "__main__":
    main()
