# -*- coding: utf-8 -*-
"""
Modal GPU Runner for V2 Experiments (March 2026)

Runs:
1. sandbagging_experiment_v2.py - SGPO vs CPO vs PPO comparison
2. h1_reward_hacking_experiment_v2.py - H1 correlation validation

Usage:
    modal run modal_runner_v2_experiments.py --experiment both --seeds 50
"""

import modal

app = modal.App("feedback-geometry-v2-experiments")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
    )
    .add_local_dir("src", remote_path="/app/src")
)

results_volume = modal.Volume.from_name("feedback-geometry-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=14400,
    memory=16384,
    volumes={"/results": results_volume},
)
def run_sandbagging_v2(seeds: int = 50, run_ablations: bool = True):
    """Run Sandbagging V2 on Modal GPU."""
    import sys
    import json
    from pathlib import Path
    from datetime import datetime
    
    sys.path.insert(0, "/app/src")
    
    import torch
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}, Seeds: {seeds}")
    
    from sandbagging_experiment_v2 import main as sandbagging_main
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"/results/sandbagging_v2_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = sandbagging_main(num_seeds=seeds, run_ablations=run_ablations, output_dir=str(output_dir))
    results_volume.commit()
    
    return {"experiment": "sandbagging_v2", "seeds": seeds, "output_dir": str(output_dir)}


@app.function(
    image=image,
    gpu="T4",
    timeout=14400,
    memory=16384,
    volumes={"/results": results_volume},
)
def run_h1_exploitation_v2(seeds: int = 50, h1_threshold: float = 0.0):
    """Run H1 Exploitation V2 on Modal GPU."""
    import sys
    from pathlib import Path
    from datetime import datetime
    
    sys.path.insert(0, "/app/src")
    
    import torch
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}, Seeds: {seeds}")
    
    from h1_reward_hacking_experiment_v2 import run_full_experiment
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"/results/h1_exploitation_v2_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_full_experiment(
        num_seeds=seeds,
        h1_magnitudes=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        h1_threshold=h1_threshold,
        output_dir=str(output_dir)
    )
    results_volume.commit()
    
    return {"experiment": "h1_exploitation_v2", "seeds": seeds, "correlations": results.get("correlations", {})}


@app.local_entrypoint()
def main(experiment: str = "both", seeds: int = 50, quick: bool = False):
    """Run V2 experiments on Modal."""
    actual_seeds = 5 if quick else seeds
    
    print(f"Experiment: {experiment}, Seeds: {actual_seeds}, GPU: Modal T4")
    
    if experiment.lower() in ["sandbagging", "both"]:
        print("Launching Sandbagging V2...")
        result = run_sandbagging_v2.remote(seeds=actual_seeds, run_ablations=True)
        print(f"Sandbagging complete: {result}")
    
    if experiment.lower() in ["h1", "both"]:
        print("Launching H1 Exploitation V2...")
        result = run_h1_exploitation_v2.remote(seeds=actual_seeds, h1_threshold=0.0)
        print(f"H1 complete: {result}")
    
    print("ALL EXPERIMENTS COMPLETE")
