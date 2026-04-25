"""
Modal GPU Runner for Safe RLHF Experiments

Deploy and run experiments on Modal GPUs:
    modal run modal_runner.py

For local testing:
    python modal_runner.py --local
"""

import modal
import os

# Define Modal app
app = modal.App("safe-rlhf-experiments")

# Create image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
    "sentence-transformers>=2.2.0",
    "datasets>=2.0.0",
    "matplotlib>=3.7.0",
)

@app.function(
    image=image,
    mounts=[modal.Mount.from_local_dir("src", remote_path="/app/src")],
    gpu="T4",  # Use T4 for cost efficiency
    timeout=7200,  # 2 hours
    memory=16384,  # 16GB RAM
)
def run_experiment_a(seeds: int = 50, use_synthetic: bool = False):
    """Run Experiment A on Modal GPU."""
    import sys
    sys.path.insert(0, "/app/src")
    
    import numpy as np
    import pandas as pd
    import torch
    from datetime import datetime
    
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
    mounts=[modal.Mount.from_local_dir("src", remote_path="/app/src")],
    gpu="T4",
    timeout=7200,
    memory=16384,
)
def run_experiment_c(seeds: int = 50, episodes: int = 300):
    """Run Experiment C on Modal GPU."""
    import sys
    sys.path.insert(0, "/app/src")
    
    import numpy as np
    import pandas as pd
    import torch
    from datetime import datetime
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Experiment C on device: {device}")
    print(f"Seeds: {seeds}, Episodes: {episodes}")
    
    from run_experiments_modal import run_experiment_c_full
    
    df = run_experiment_c_full(seeds=seeds, episodes=episodes)
    
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
    
    # Compare conformal_anis vs baselines
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


@app.function(
    image=image,
    mounts=[modal.Mount.from_local_dir("src", remote_path="/app/src")],
    gpu="T4",
    timeout=14400,  # 4 hours for full run
    memory=32768,
)
def run_both_experiments(seeds: int = 50, use_synthetic: bool = False):
    """Run both experiments and save results."""
    import sys
    sys.path.insert(0, "/app/src")
    
    import pandas as pd
    from datetime import datetime
    
    print("="*60)
    print("RUNNING BOTH EXPERIMENTS")
    print("="*60)
    
    # Run Experiment A
    results_a = run_experiment_a.local(seeds=seeds, use_synthetic=use_synthetic)
    df_a = pd.DataFrame(results_a)
    
    # Run Experiment C
    results_c = run_experiment_c.local(seeds=seeds, episodes=300)
    df_c = pd.DataFrame(results_c)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    return {
        "experiment_a": df_a.to_dict(),
        "experiment_c": df_c.to_dict(),
        "timestamp": timestamp,
    }


@app.local_entrypoint()
def main(
    experiment: str = "both",
    mode: str = "full",
    use_synthetic: bool = False,
):
    """
    Run experiments on Modal.
    
    Args:
        experiment: "A", "C", or "both"
        mode: "quick" (5 seeds) or "full" (50 seeds)
        use_synthetic: Use synthetic data instead of HH-RLHF
    """
    import pandas as pd
    from pathlib import Path
    from datetime import datetime
    
    seeds = 5 if mode == "quick" else 50
    episodes = 100 if mode == "quick" else 300
    
    print(f"\n{'='*60}")
    print(f"Modal Experiment Runner")
    print(f"{'='*60}")
    print(f"Experiment: {experiment}")
    print(f"Mode: {mode} ({seeds} seeds)")
    print(f"Use synthetic: {use_synthetic}")
    
    output_dir = Path("results/modal")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if experiment == "A":
        results = run_experiment_a.remote(seeds=seeds, use_synthetic=use_synthetic)
        df = pd.DataFrame(results)
        output_file = output_dir / f"experiment_a_{mode}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")
        
    elif experiment == "C":
        results = run_experiment_c.remote(seeds=seeds, episodes=episodes)
        df = pd.DataFrame(results)
        output_file = output_dir / f"experiment_c_{mode}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")
        
    else:  # both
        results = run_both_experiments.remote(seeds=seeds, use_synthetic=use_synthetic)
        
        df_a = pd.DataFrame(results["experiment_a"])
        df_c = pd.DataFrame(results["experiment_c"])
        
        df_a.to_csv(output_dir / f"experiment_a_{mode}_{timestamp}.csv", index=False)
        df_c.to_csv(output_dir / f"experiment_c_{mode}_{timestamp}.csv", index=False)
        
        print(f"\nSaved results to {output_dir}/")
    
    print("\nDone!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run locally instead of Modal")
    parser.add_argument("--experiment", choices=["A", "C", "both"], default="both")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--use-synthetic", action="store_true")
    
    args = parser.parse_args()
    
    if args.local:
        # Run locally using the direct script
        import subprocess
        cmd = [
            "python", "src/run_experiments_modal.py",
            "--mode", args.mode,
            "--experiment", args.experiment,
            "--local",
        ]
        if args.use_synthetic:
            cmd.append("--use-synthetic")
        subprocess.run(cmd)
    else:
        print("Run with: modal run modal_runner.py --experiment both --mode full")
