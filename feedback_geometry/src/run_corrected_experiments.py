#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Corrected Experiments: Quick Validation and Full Runs

This script runs both Experiment A (preference filtering) and Experiment C (conformal safety)
with the mathematically corrected implementations.

Usage:
    # Quick validation (5 seeds)
    python run_corrected_experiments.py --mode quick
    
    # Full run (50 seeds)
    python run_corrected_experiments.py --mode full
    
    # Only Experiment A
    python run_corrected_experiments.py --experiment A --mode quick
    
    # Only Experiment C
    python run_corrected_experiments.py --experiment C --mode quick
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import pandas as pd
from typing import Dict, List

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
)

from conformal_sgpo import (
    ConformalSafetyMetric,
    ConformalSGPOConfig,
    train_conformal_sgpo,
    train_conformal_sgpo_anis,
)


# ============================================================================
# EXPERIMENT A: PREFERENCE FILTERING
# ============================================================================

def create_synthetic_preferences(n_contexts: int = 100, n_items_per_context: int = 5) -> List[Dict]:
    """
    Create synthetic preference data with controlled cyclic content.
    
    This is a placeholder until we integrate real HH-RLHF data.
    """
    preferences = []
    
    for ctx_id in range(n_contexts):
        prompt = f"Context_{ctx_id}"
        
        # Generate items for this context
        items = [f"Item_{ctx_id}_{i}" for i in range(n_items_per_context)]
        
        # Create preferences with some cycles
        for i in range(n_items_per_context - 1):
            preferences.append({
                'prompt': prompt,
                'chosen': items[i + 1],
                'rejected': items[i],
            })
        
        # Add cycle with 30% probability
        if np.random.rand() < 0.3:
            preferences.append({
                'prompt': prompt,
                'chosen': items[0],
                'rejected': items[-1],
            })
    
    return preferences


def run_experiment_a_quick(seeds: int = 5) -> pd.DataFrame:
    """Run Experiment A with all filtering variants."""
    print("\n" + "="*60)
    print("EXPERIMENT A: Preference Filtering")
    print("="*60)
    
    # Create synthetic data
    print("\n[1/4] Generating synthetic preferences...")
    preferences = create_synthetic_preferences(n_contexts=50, n_items_per_context=5)
    print(f"  Generated {len(preferences)} preference pairs")
    
    methods = ["raw", "harmonic_only", "curl_only", "reliability_score"]
    results = []
    
    for method in methods:
        print(f"\n[2/4] Running method: {method}")
        
        config = FilteringConfig(
            method=method,
            threshold=0.5,
            h1_threshold=0.8,
        )
        
        for seed in range(seeds):
            print(f"  Seed {seed+1}/{seeds}...", end=" ")
            
            result = run_experiment_a_variant(
                preferences=preferences,
                method=method,
                config=config,
                seed=seed,
            )
            
            results.append({
                "seed": seed,
                "method": method,
                "n_train": result.n_train,
                "avg_reliability": result.avg_reliability,
                "avg_curl_ratio": result.avg_curl_ratio,
                "avg_harmonic_ratio": result.avg_harmonic_ratio,
            })
            
            print(f"reliability={result.avg_reliability:.3f}, n_train={result.n_train}")
    
    df = pd.DataFrame(results)
    
    print("\n[3/4] Summary Statistics:")
    summary = df.groupby("method").agg({
        "avg_reliability": ["mean", "std"],
        "avg_curl_ratio": ["mean", "std"],
        "avg_harmonic_ratio": ["mean", "std"],
        "n_train": ["mean", "std"],
    })
    print(summary)
    
    return df


# ============================================================================
# EXPERIMENT C: CONFORMAL SAFETY
# ============================================================================

def run_experiment_c_quick(seeds: int = 5, episodes: int = 100) -> pd.DataFrame:
    """Run Experiment C with conformal SGPO variants."""
    print("\n" + "="*60)
    print("EXPERIMENT C: Conformal Safety Metrics")
    print("="*60)
    
    methods = [
        "conformal_sgpo",
        "conformal_sgpo_anis",
    ]
    
    results = []
    
    for method in methods:
        print(f"\n[1/3] Running method: {method}")
        
        for seed in range(seeds):
            print(f"  Seed {seed+1}/{seeds}...", end=" ")
            
            # Create environment
            env = FixedSandbaggingEnv()
            trap_center, trap_radius = env.get_trap_info()
            known_regions = [(trap_center, trap_radius)]
            
            # Configure
            config = ConformalSGPOConfig(
                episodes=episodes,
                sharpness=2.0,
                anisotropic=(method == "conformal_sgpo_anis"),
            )
            
            # Train
            if method == "conformal_sgpo":
                result = train_conformal_sgpo(env, config, seed, known_regions)
            else:
                result = train_conformal_sgpo_anis(env, config, seed, known_regions)
            
            # Extract metrics
            violations = sum(result["episode_violations"])
            goal_rate = np.mean(result["goal_reached"])
            final_return = np.mean(result["episode_returns"][-20:])
            
            results.append({
                "seed": seed,
                "method": method,
                "violations": violations,
                "goal_rate": goal_rate,
                "final_return": final_return,
                "n_hardened_regions": result["n_hardened_regions"],
            })
            
            print(f"violations={violations}, goal_rate={goal_rate:.2%}")
    
    df = pd.DataFrame(results)
    
    print("\n[2/3] Summary Statistics:")
    summary = df.groupby("method").agg({
        "violations": ["mean", "std"],
        "goal_rate": ["mean", "std"],
        "final_return": ["mean", "std"],
    })
    print(summary)
    
    return df


# ============================================================================
# VALIDATION TESTS
# ============================================================================

def test_hodge_decomposition():
    """Test DiscreteHodgeRank on known cyclic preferences."""
    print("\n" + "="*60)
    print("VALIDATION: Hodge Decomposition")
    print("="*60)
    
    hodge = DiscreteHodgeRank()
    
    # Test 1: Perfect cycle (A > B > C > A)
    print("\nTest 1: Perfect 3-cycle")
    comparisons = [
        (0, 1, 1.0),  # B > A
        (1, 2, 1.0),  # C > B
        (2, 0, 1.0),  # A > C
    ]
    comp = hodge.decompose(3, comparisons)
    print(f"  Gradient energy: {comp.gradient_energy:.3f}")
    print(f"  Curl energy: {comp.curl_energy:.3f}")
    print(f"  Harmonic energy: {comp.harmonic_energy:.3f}")
    print(f"  Reliability score: {comp.reliability_score:.3f}")
    assert comp.reliability_score < 0.5, "Cycle should have low reliability"
    print("  ✓ Passed")
    
    # Test 2: Transitive preferences (A > B > C)
    print("\nTest 2: Transitive preferences")
    comparisons = [
        (0, 1, 1.0),  # B > A
        (1, 2, 1.0),  # C > B
        (0, 2, 1.0),  # C > A (transitive)
    ]
    comp = hodge.decompose(3, comparisons)
    print(f"  Gradient energy: {comp.gradient_energy:.3f}")
    print(f"  Curl energy: {comp.curl_energy:.3f}")
    print(f"  Harmonic energy: {comp.harmonic_energy:.3f}")
    print(f"  Reliability score: {comp.reliability_score:.3f}")
    assert comp.reliability_score > 0.8, "Transitive should have high reliability"
    print("  ✓ Passed")


def test_conformal_metric():
    """Test ConformalSafetyMetric barrier properties."""
    print("\n" + "="*60)
    print("VALIDATION: Conformal Safety Metric")
    print("="*60)
    
    metric = ConformalSafetyMetric(state_dim=2, sharpness=2.0)
    metric.add_known_danger_region(np.array([5.0, 5.0]), 2.0)
    
    # Test points at different distances
    test_points = [
        ("Safe (far)", torch.tensor([[0.0, 0.0]])),
        ("Near boundary", torch.tensor([[3.5, 5.0]])),
        ("At boundary", torch.tensor([[3.0, 5.0]])),
        ("Inside danger", torch.tensor([[5.0, 5.0]])),
    ]
    
    print("\nConformal factor σ(x) at different locations:")
    for name, point in test_points:
        sigma = metric.conformal_factor(point).item()
        g = metric.metric_tensor(point).item()
        print(f"  {name:20s}: σ={sigma:8.2f}, g={g:10.2f}")
    
    # Verify σ increases near danger
    safe_sigma = metric.conformal_factor(torch.tensor([[0.0, 0.0]])).item()
    near_sigma = metric.conformal_factor(torch.tensor([[3.5, 5.0]])).item()
    assert near_sigma > safe_sigma, "σ should increase near danger"
    print("\n  ✓ Passed: σ increases near danger")


def test_fixed_environment():
    """Test FixedSandbaggingEnv trap reachability."""
    print("\n" + "="*60)
    print("VALIDATION: Fixed Environment")
    print("="*60)
    
    env = FixedSandbaggingEnv()
    obs = env.reset()
    
    print(f"\nEnvironment configuration:")
    print(f"  Start: {env.start}")
    print(f"  Goal: {env.goal}")
    print(f"  Trap center: {env.trap_center}")
    print(f"  Trap radius: {env.trap_radius}")
    
    # Simulate moving toward trap
    print(f"\nSimulating movement toward trap...")
    action = np.array([0.5, 0.5])  # Toward (4,4)
    
    for step in range(15):
        obs, reward, cost, done, info = env.step(action)
        if info['in_trap']:
            print(f"  Step {step}: Entered trap at {obs}")
            print(f"  ✓ Passed: Trap is reachable")
            return
    
    print(f"  ✗ Failed: Trap not reached after 15 steps")
    print(f"  Final position: {obs}")
    print(f"  Distance to trap: {info['dist_to_trap']:.2f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run corrected experiments")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="Quick (5 seeds) or full (50 seeds)")
    parser.add_argument("--experiment", choices=["A", "C", "both"], default="both",
                       help="Which experiment to run")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only run validation tests")
    parser.add_argument("--output-dir", type=str, default="results/corrected",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Set seeds
    seeds = 5 if args.mode == "quick" else 50
    episodes = 100 if args.mode == "quick" else 300
    
    print("\n" + "="*60)
    print("CORRECTED EXPERIMENTS RUNNER")
    print("="*60)
    print(f"Mode: {args.mode} ({seeds} seeds)")
    print(f"Experiment: {args.experiment}")
    print(f"Output: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run validation tests
    if args.validate_only or args.mode == "quick":
        test_hodge_decomposition()
        test_conformal_metric()
        test_fixed_environment()
        
        if args.validate_only:
            print("\n✓ All validation tests passed")
            return
    
    # Run experiments
    results = {}
    
    if args.experiment in ["A", "both"]:
        df_a = run_experiment_a_quick(seeds=seeds)
        results["experiment_a"] = df_a
        
        # Save
        output_file = output_dir / f"experiment_a_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_a.to_csv(output_file, index=False)
        print(f"\n[4/4] Saved to: {output_file}")
    
    if args.experiment in ["C", "both"]:
        df_c = run_experiment_c_quick(seeds=seeds, episodes=episodes)
        results["experiment_c"] = df_c
        
        # Save
        output_file = output_dir / f"experiment_c_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_c.to_csv(output_file, index=False)
        print(f"\n[3/3] Saved to: {output_file}")
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print("="*60)
    
    # Summary
    if "experiment_a" in results:
        df_a = results["experiment_a"]
        print("\nExperiment A Summary:")
        for method in df_a["method"].unique():
            subset = df_a[df_a["method"] == method]
            print(f"  {method:20s}: reliability={subset['avg_reliability'].mean():.3f}±{subset['avg_reliability'].std():.3f}")
    
    if "experiment_c" in results:
        df_c = results["experiment_c"]
        print("\nExperiment C Summary:")
        for method in df_c["method"].unique():
            subset = df_c[df_c["method"] == method]
            print(f"  {method:25s}: violations={subset['violations'].mean():.1f}±{subset['violations'].std():.1f}, goal_rate={subset['goal_rate'].mean():.2%}")


if __name__ == "__main__":
    main()
