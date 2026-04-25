"""
Experiment Framework for Feedback Geometry Paper

Provides:
- Multi-seed evaluation with proper statistics
- Configuration-driven experiments
- Standardized logging and results format
- Statistical analysis utilities
"""

import torch
import numpy as np
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from datetime import datetime
import hashlib
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainTestSplit:
    """Configuration for train/test item split in H¹ experiments."""
    n_train_items: int = 70
    n_test_items: int = 30
    trap_item_start: int = 80
    trap_item_end: int = 90
    
    def validate(self):
        """Validate split configuration."""
        total = self.n_train_items + self.n_test_items
        assert self.trap_item_start >= self.n_train_items, \
            f"Trap must be in test region: trap_start={self.trap_item_start} < n_train={self.n_train_items}"
        assert self.trap_item_end <= total, \
            f"Trap must be within items: trap_end={self.trap_item_end} > total={total}"


@dataclass
class H1ExperimentConfig:
    """Configuration specific to H¹ exploitation experiments."""
    h1_magnitude: float = 0.5
    split: TrainTestSplit = field(default_factory=TrainTestSplit)
    base_noise: float = 0.1
    n_comparisons_per_pair: int = 3
    rm_epochs: int = 100
    n_eval_episodes: int = 50
    n_selections_per_episode: int = 20
    
    def validate(self):
        self.split.validate()
        assert 0.0 <= self.h1_magnitude <= 2.0, f"h1_magnitude should be in [0, 2], got {self.h1_magnitude}"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    num_seeds: int = 50
    episodes: int = 300
    gamma: float = 0.99
    
    # Environment parameters
    env_type: str = "sandbagging"
    env_params: Dict = field(default_factory=dict)
    
    # Algorithm parameters
    algorithm: str = "sgpo"
    algo_params: Dict = field(default_factory=dict)
    
    # SGPO-specific
    metric_sharpness: float = 2.0
    metric_severity: float = 5.0
    horizon_radius: float = 1.0
    
    # H¹ experiment specific (optional)
    h1_config: Optional[H1ExperimentConfig] = None
    
    # Ablation grid (if running ablations)
    ablation_grid: Optional[Dict] = None
    
    # Output
    output_dir: str = "results"
    save_checkpoints: bool = False
    checkpoint_freq: int = 100
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def hash(self) -> str:
        """Unique hash for this configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


# ============================================================================
# Statistical Analysis
# ============================================================================

@dataclass
class StatisticalResult:
    """Statistical summary of experiment results across seeds."""
    metric_name: str
    mean: float
    std: float
    ci_lower: float  # 95% CI
    ci_upper: float
    median: float
    min_val: float
    max_val: float
    n_samples: int
    raw_values: List[float]
    
    def to_dict(self) -> Dict:
        return {
            "metric": self.metric_name,
            "mean": self.mean,
            "std": self.std,
            "ci_95": [self.ci_lower, self.ci_upper],
            "median": self.median,
            "range": [self.min_val, self.max_val],
            "n": self.n_samples
        }


def compute_statistics(values: List[float], metric_name: str) -> StatisticalResult:
    """Compute comprehensive statistics for a list of values."""
    arr = np.array(values)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else 0.0
    
    # 95% CI using t-distribution
    if n > 1:
        se = std / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
    else:
        ci_lower = ci_upper = mean
    
    return StatisticalResult(
        metric_name=metric_name,
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        median=np.median(arr),
        min_val=np.min(arr),
        max_val=np.max(arr),
        n_samples=n,
        raw_values=arr.tolist()
    )


def compare_methods(
    method_a_values: List[float],
    method_b_values: List[float],
    method_a_name: str = "A",
    method_b_name: str = "B"
) -> Dict:
    """Compare two methods using appropriate statistical tests."""
    a = np.array(method_a_values)
    b = np.array(method_b_values)
    
    # Welch's t-test (does not assume equal variance)
    t_stat, t_pval = stats.ttest_ind(a, b, equal_var=False)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(a, b, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    cohens_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    return {
        "comparison": f"{method_a_name} vs {method_b_name}",
        "mean_diff": float(np.mean(a) - np.mean(b)),
        "welch_t": {"statistic": float(t_stat), "p_value": float(t_pval)},
        "mann_whitney_u": {"statistic": float(u_stat), "p_value": float(u_pval)},
        "cohens_d": float(cohens_d),
        "effect_size": effect_interpretation,
        "significant_at_005": t_pval < 0.05 and u_pval < 0.05
    }


# ============================================================================
# Experiment Runner
# ============================================================================

@dataclass 
class SeedResult:
    """Result from a single seed run."""
    seed: int
    episode_returns: List[float]
    episode_violations: List[int]
    goal_reached: List[bool]
    final_trajectory: Optional[np.ndarray] = None
    additional_metrics: Dict = field(default_factory=dict)


class ExperimentRunner:
    """Runs experiments with multi-seed evaluation and proper statistics."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        train_fn: Callable,
        env_factory: Callable,
        parallel: bool = False,
        n_workers: int = 4
    ):
        self.config = config
        self.train_fn = train_fn
        self.env_factory = env_factory
        self.parallel = parallel
        self.n_workers = n_workers
        
        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[SeedResult] = []
    
    def run_single_seed(self, seed: int) -> SeedResult:
        """Run experiment with a single seed."""
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create environment
        env = self.env_factory(**self.config.env_params)
        
        # Run training
        result = self.train_fn(
            env=env,
            config=self.config,
            seed=seed
        )
        
        return result
    
    def run_all_seeds(self) -> Dict:
        """Run experiment across all seeds and aggregate results."""
        print(f"\n{'='*60}")
        print(f"Running: {self.config.name}")
        print(f"Seeds: {self.config.num_seeds}, Episodes: {self.config.episodes}")
        print(f"{'='*60}")
        
        if self.parallel:
            self._run_parallel()
        else:
            self._run_sequential()
        
        # Aggregate results
        aggregated = self._aggregate_results()
        
        # Save results
        self._save_results(aggregated)
        
        return aggregated
    
    def _run_sequential(self):
        """Run seeds sequentially."""
        for seed in range(self.config.num_seeds):
            print(f"  Seed {seed+1}/{self.config.num_seeds}...", end=" ", flush=True)
            result = self.run_single_seed(seed)
            self.results.append(result)
            
            # Quick summary
            final_return = np.mean(result.episode_returns[-50:])
            total_violations = sum(result.episode_violations)
            print(f"Return: {final_return:.2f}, Violations: {total_violations}")
    
    def _run_parallel(self):
        """Run seeds in parallel."""
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self.run_single_seed, seed): seed 
                for seed in range(self.config.num_seeds)
            }
            
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    print(f"  Seed {seed} completed")
                except Exception as e:
                    print(f"  Seed {seed} failed: {e}")
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results across seeds."""
        # Final performance metrics (last 50 episodes)
        final_returns = [np.mean(r.episode_returns[-50:]) for r in self.results]
        total_violations = [sum(r.episode_violations) for r in self.results]
        final_violations = [np.mean(r.episode_violations[-50:]) for r in self.results]
        goal_rates = [np.mean(r.goal_reached[-50:]) for r in self.results]
        
        aggregated = {
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "num_seeds": len(self.results),
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
                    "goal_rate": float(np.mean(r.goal_reached[-50:])),
                    **r.additional_metrics
                }
                for r in self.results
            ],
            "learning_curves": {
                "returns_mean": np.mean([r.episode_returns for r in self.results], axis=0).tolist(),
                "returns_std": np.std([r.episode_returns for r in self.results], axis=0).tolist(),
                "violations_mean": np.mean([r.episode_violations for r in self.results], axis=0).tolist(),
                "violations_std": np.std([r.episode_violations for r in self.results], axis=0).tolist(),
            }
        }
        
        return aggregated
    
    def _save_results(self, aggregated: Dict):
        """Save aggregated results to JSON."""
        output_file = self.output_dir / f"results_{self.config.hash()}.json"
        
        with open(output_file, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        
        # Print summary
        self._print_summary(aggregated)
    
    def _print_summary(self, aggregated: Dict):
        """Print summary statistics."""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        for metric_name, metric_data in aggregated["metrics"].items():
            mean = metric_data["mean"]
            ci = metric_data["ci_95"]
            print(f"{metric_name}: {mean:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")


# ============================================================================
# Multi-Method Comparison
# ============================================================================

class MethodComparison:
    """Compare multiple methods with statistical tests."""
    
    def __init__(self, output_dir: str = "results/comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.method_results: Dict[str, Dict] = {}
    
    def add_method(self, name: str, results: Dict):
        """Add results from a method."""
        self.method_results[name] = results
    
    def compare_all(self, metric: str = "total_violations") -> Dict:
        """Compare all methods on a given metric."""
        comparisons = {}
        methods = list(self.method_results.keys())
        
        for i, method_a in enumerate(methods):
            for method_b in methods[i+1:]:
                values_a = [
                    s["total_violations"] if metric == "total_violations" else s["final_return"]
                    for s in self.method_results[method_a]["per_seed"]
                ]
                values_b = [
                    s["total_violations"] if metric == "total_violations" else s["final_return"]
                    for s in self.method_results[method_b]["per_seed"]
                ]
                
                key = f"{method_a}_vs_{method_b}"
                comparisons[key] = compare_methods(values_a, values_b, method_a, method_b)
        
        return comparisons
    
    def generate_report(self) -> Dict:
        """Generate comprehensive comparison report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "methods": list(self.method_results.keys()),
            "violation_comparison": self.compare_all("total_violations"),
            "return_comparison": self.compare_all("final_return"),
            "summary_table": self._summary_table()
        }
        
        # Save report
        output_file = self.output_dir / "comparison_report.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comparison report saved to {output_file}")
        return report
    
    def _summary_table(self) -> List[Dict]:
        """Generate summary table for all methods."""
        table = []
        for method, results in self.method_results.items():
            table.append({
                "method": method,
                "final_return_mean": results["metrics"]["final_return"]["mean"],
                "final_return_ci": results["metrics"]["final_return"]["ci_95"],
                "total_violations_mean": results["metrics"]["total_violations"]["mean"],
                "total_violations_ci": results["metrics"]["total_violations"]["ci_95"],
                "goal_rate_mean": results["metrics"]["goal_success_rate"]["mean"],
            })
        return table


# ============================================================================
# Utility Functions
# ============================================================================

def load_results(filepath: str) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def merge_results(filepaths: List[str]) -> Dict:
    """Merge results from multiple files (e.g., for distributed runs)."""
    merged_per_seed = []
    config = None
    
    for fp in filepaths:
        data = load_results(fp)
        if config is None:
            config = data["config"]
        merged_per_seed.extend(data["per_seed"])
    
    # Recompute statistics
    final_returns = [s["final_return"] for s in merged_per_seed]
    total_violations = [s["total_violations"] for s in merged_per_seed]
    goal_rates = [s["goal_rate"] for s in merged_per_seed]
    
    return {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "num_seeds": len(merged_per_seed),
        "metrics": {
            "final_return": compute_statistics(final_returns, "final_return").to_dict(),
            "total_violations": compute_statistics(total_violations, "total_violations").to_dict(),
            "goal_success_rate": compute_statistics(goal_rates, "goal_success_rate").to_dict(),
        },
        "per_seed": merged_per_seed
    }
