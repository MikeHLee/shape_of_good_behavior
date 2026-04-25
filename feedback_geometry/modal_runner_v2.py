# -*- coding: utf-8 -*-
"""
Modal GPU Runner for Safe RLHF Experiments (v2 - Modal 1.0+ API)

Deploy and run experiments on Modal GPUs:
    modal run modal_runner_v2.py --experiment both --mode full

Uses Image.add_local_dir (not deprecated Mount).
"""

import modal

# Define Modal app
app = modal.App("safe-rlhf-experiments")

# Create image with dependencies AND local source code embedded
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "sentence-transformers>=2.2.0",
        "datasets>=2.0.0",
        "matplotlib>=3.7.0",
    )
    .add_local_dir("src", remote_path="/app/src")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=28800,  # 8 hours
    memory=16384,
)
def run_experiment_a(seeds: int = 50, use_synthetic: bool = False, seed_offset: int = 0):
    """Run Experiment A on Modal GPU."""
    import sys
    sys.path.insert(0, "/app/src")
    
    import numpy as np
    import pandas as pd
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Experiment A on device: {device}")
    print(f"Seeds: {seeds}, Use synthetic: {use_synthetic}")
    
    from run_experiments_modal import run_experiment_a_full
    
    df = run_experiment_a_full(
        seeds=seeds,
        n_train_samples=3000,
        n_test_samples=1000,
        use_hh_rlhf=not use_synthetic,
        device=device,
        seed_offset=seed_offset,
    )
    
    # Summary statistics
    summary = df.groupby("method").agg({
        "n_train": ["mean", "std"],
        "accuracy": ["mean", "std"],
        "exploitation_rate": ["mean", "std"],
    }).round(4)
    
    print("\n" + "="*60)
    print("EXPERIMENT A RESULTS")
    print("="*60)
    print(summary)
    
    # Statistical tests
    from scipy import stats
    
    raw = df[df.method == "raw"]["exploitation_rate"]
    reliability = df[df.method == "reliability_score"]["exploitation_rate"]
    
    if len(raw) > 1 and len(reliability) > 1:
        t_stat, p_value = stats.ttest_ind(raw, reliability, equal_var=False)
        cohens_d = (raw.mean() - reliability.mean()) / np.sqrt((raw.std()**2 + reliability.std()**2) / 2)
        
        print(f"\nStatistical Comparison (raw vs reliability_score):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Cohen's d: {cohens_d:.3f}")
        print(f"  Exploitation reduction: {(raw.mean() - reliability.mean()) / max(raw.mean(), 0.001) * 100:.1f}%")
    
    return df.to_dict()


@app.function(
    image=image,
    gpu="T4",
    timeout=28800,  # 8 hours
    memory=16384,
)
def run_experiment_c(seeds: int = 50, episodes: int = 300, seed_offset: int = 0):
    """Run Experiment C on Modal GPU."""
    import sys
    sys.path.insert(0, "/app/src")
    
    import numpy as np
    import pandas as pd
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Experiment C on device: {device}")
    print(f"Seeds: {seeds}, Episodes: {episodes}")
    
    from run_experiments_modal import run_experiment_c_full
    
    df = run_experiment_c_full(seeds=seeds, episodes=episodes, seed_offset=seed_offset)
    
    # Summary statistics
    summary = df.groupby("method").agg({
        "violations": ["mean", "std"],
        "goal_rate": ["mean", "std"],
        "final_return": ["mean", "std"],
    }).round(4)
    
    print("\n" + "="*60)
    print("EXPERIMENT C RESULTS")
    print("="*60)
    print(summary)
    
    # Statistical tests
    from scipy import stats
    
    for baseline in ["ppo", "cpo"]:
        base = df[df.method == baseline]["violations"]
        conf = df[df.method == "conformal_sgpo_anis"]["violations"]
        
        if len(base) > 1 and len(conf) > 1:
            t_stat, p_value = stats.ttest_ind(base, conf, equal_var=False)
            cohens_d = (base.mean() - conf.mean()) / np.sqrt((base.std()**2 + conf.std()**2) / 2)
            
            print(f"\n{baseline.upper()} vs conformal_sgpo_anis:")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Cohen's d: {cohens_d:.3f}")
            print(f"  Violation reduction: {(base.mean() - conf.mean()) / max(base.mean(), 0.001) * 100:.1f}%")
    
    return df.to_dict()


@app.local_entrypoint()
def main(
    experiment: str = "both",
    mode: str = "full",
    use_synthetic: bool = False,
    batch_size: int = 10,
    batch_num: int = 0,
):
    """
    Run experiments on Modal GPU cloud.
    
    Args:
        experiment: "A", "C", or "both"
        mode: "quick" (5 seeds) or "full" (50 seeds)
        use_synthetic: Use synthetic data instead of HH-RLHF
    """
    import pandas as pd
    from pathlib import Path
    from datetime import datetime
    
    seeds = 5 if mode == "quick" else batch_size
    episodes = 100 if mode == "quick" else 300
    seed_offset = batch_num * batch_size
    
    print(f"\n{'='*60}")
    print(f"Modal GPU Experiment Runner")
    print(f"{'='*60}")
    print(f"Experiment: {experiment}")
    print(f"Mode: {mode}")
    print(f"Batch: {batch_num + 1} (seeds {seed_offset + 1}-{seed_offset + seeds})")
    print(f"Use synthetic: {use_synthetic}")
    print(f"Running on: Modal T4 GPU")
    
    output_dir = Path("results/modal")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if experiment.upper() == "A":
        print("\nLaunching Experiment A on Modal...")
        results = run_experiment_a.remote(seeds=seeds, use_synthetic=use_synthetic, seed_offset=seed_offset)
        df = pd.DataFrame(results)
        output_file = output_dir / f"experiment_a_batch{batch_num}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")
        
    elif experiment.upper() == "C":
        print("\nLaunching Experiment C on Modal...")
        results = run_experiment_c.remote(seeds=seeds, episodes=episodes, seed_offset=seed_offset)
        df = pd.DataFrame(results)
        output_file = output_dir / f"experiment_c_batch{batch_num}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")
        
    else:  # both
        print("\nLaunching BOTH experiments on Modal (parallel)...")
        
        # Launch both in parallel using spawn
        future_a = run_experiment_a.spawn(seeds=seeds, use_synthetic=use_synthetic, seed_offset=seed_offset)
        future_c = run_experiment_c.spawn(seeds=seeds, episodes=episodes, seed_offset=seed_offset)
        
        # Wait for results
        print("Waiting for Experiment A...")
        results_a = future_a.get()
        df_a = pd.DataFrame(results_a)
        df_a.to_csv(output_dir / f"experiment_a_batch{batch_num}_{timestamp}.csv", index=False)
        print("Experiment A complete!")
        
        print("Waiting for Experiment C...")
        results_c = future_c.get()
        df_c = pd.DataFrame(results_c)
        df_c.to_csv(output_dir / f"experiment_c_batch{batch_num}_{timestamp}.csv", index=False)
        print("Experiment C complete!")
        
        print(f"\nResults saved to {output_dir}/")
    
    print("\n" + "="*60)
    print("MODAL EXPERIMENTS COMPLETE")
    print("="*60)
