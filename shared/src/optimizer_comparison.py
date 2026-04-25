"""Multi-method benchmark runner: all preference optimizers × N seeds.

Runs DPO, GRPO, ORPO, KTO and their Hodge-aware variants on the same
preference data across multiple seeds, producing a comparison table with
statistical tests (Welch t, Cohen's d).
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import PipelineConfig
from .preference_mapper import MappingResult

logger = logging.getLogger(__name__)


@dataclass
class MethodResult:
    """Per-seed result for one method."""
    method: str
    seed: int
    exploit_resistance: float
    final_loss: float
    extra_metrics: Dict = field(default_factory=dict)


@dataclass
class ComparisonTable:
    """Full comparison across all methods and seeds."""
    method_stats: Dict[str, Dict]  # method → {mean, std, ci95, values}
    pairwise_tests: Dict[str, Dict]  # "A vs B" → {welch_t, cohens_d, ...}
    all_results: List[MethodResult]
    config: Dict = field(default_factory=dict)


class OptimizerBenchmark:
    """Run all preference optimization methods on the same data."""

    def __init__(self, config: PipelineConfig, mapping: MappingResult):
        self.config = config
        self.mapping = mapping

    def run(
        self,
        num_seeds: int = 5,
        methods: Optional[List[str]] = None,
    ) -> ComparisonTable:
        """Run full benchmark.

        Args:
            num_seeds: Number of random seeds per method
            methods: List of method names to run (None = all)

        Returns:
            ComparisonTable with statistical comparison
        """
        import torch
        from .preference_optimizers import (
            DPOTrainer,
            GRPOTrainer,
            ORPOTrainer,
            KTOTrainer,
            PreferenceSample,
            mapping_to_preference_samples,
        )
        from .hodge_preference_optimizers import (
            HodgeDPOTrainer,
            HodgeGRPOTrainer,
            HodgeKTOTrainer,
        )
        from .hodge_diagnostic import HodgeDiagnosticCritic

        embed_dim = self.config.embed_dim
        hidden_dim = self.config.rm_hidden_dim
        lr = self.config.rm_lr
        epochs = self.config.rm_epochs

        # Prepare preference samples from mapping
        samples = mapping_to_preference_samples(self.mapping)
        if not samples:
            logger.error("No preference samples available")
            return ComparisonTable(
                method_stats={}, pairwise_tests={}, all_results=[]
            )

        logger.info(f"Benchmark: {len(samples)} samples, {num_seeds} seeds")

        # Run Hodge diagnostic once (shared across seeds)
        edges = self.mapping.preference_edges
        n_items = self.mapping.n_items

        diag_critic = HodgeDiagnosticCritic(self.config)
        diagnosis = diag_critic.diagnose_for_samples(
            edges, n_items,
            embedding_pairs=self.mapping.embedding_pairs,
        )

        # Define all methods
        all_methods = {
            "DPO": lambda seed: DPOTrainer(
                beta=self.config.dpo_beta,
                embed_dim=embed_dim, hidden_dim=hidden_dim,
                lr=lr, epochs=epochs,
            ),
            "GRPO": lambda seed: GRPOTrainer(
                beta=self.config.grpo_beta,
                group_size=self.config.grpo_group_size,
                embed_dim=embed_dim, hidden_dim=hidden_dim,
                lr=lr, epochs=epochs,
            ),
            "ORPO": lambda seed: ORPOTrainer(
                lambda_align=self.config.orpo_lambda,
                embed_dim=embed_dim, hidden_dim=hidden_dim,
                lr=lr, epochs=epochs,
            ),
            "KTO": lambda seed: KTOTrainer(
                beta=self.config.kto_beta,
                lambda_good=self.config.kto_lambda_good,
                lambda_bad=self.config.kto_lambda_bad,
                embed_dim=embed_dim, hidden_dim=hidden_dim,
                lr=lr, epochs=epochs,
            ),
            "Hodge-DPO": lambda seed: HodgeDPOTrainer(
                diagnosis=diagnosis,
                hodge_lambda=self.config.hodge_lambda,
                beta=self.config.dpo_beta,
                embed_dim=embed_dim, hidden_dim=hidden_dim,
                lr=lr, epochs=epochs,
            ),
            "Hodge-GRPO": lambda seed: HodgeGRPOTrainer(
                diagnosis=diagnosis,
                hodge_lambda=self.config.hodge_lambda,
                beta=self.config.grpo_beta,
                group_size=self.config.grpo_group_size,
                embed_dim=embed_dim, hidden_dim=hidden_dim,
                lr=lr, epochs=epochs,
            ),
            "Hodge-KTO": lambda seed: HodgeKTOTrainer(
                diagnosis=diagnosis,
                hodge_lambda=self.config.hodge_lambda,
                beta=self.config.kto_beta,
                embed_dim=embed_dim, hidden_dim=hidden_dim,
                lr=lr, epochs=epochs,
            ),
        }

        if methods is not None:
            all_methods = {k: v for k, v in all_methods.items() if k in methods}

        # Run all methods × seeds
        all_results: List[MethodResult] = []

        for method_name, make_trainer in all_methods.items():
            logger.info(f"Running {method_name}...")
            for seed in range(num_seeds):
                np.random.seed(seed)
                torch.manual_seed(seed)

                t0 = time.time()
                trainer = make_trainer(seed)
                result = trainer.train(samples)
                elapsed = time.time() - t0

                mr = MethodResult(
                    method=method_name,
                    seed=seed,
                    exploit_resistance=result.exploit_resistance,
                    final_loss=result.losses[-1] if result.losses else float("nan"),
                    extra_metrics={
                        **result.extra_metrics,
                        "elapsed_s": elapsed,
                    },
                )
                all_results.append(mr)
                logger.info(
                    f"  {method_name} seed={seed}: "
                    f"resistance={result.exploit_resistance:.3f} "
                    f"({elapsed:.1f}s)"
                )

        # Compute statistics
        method_stats = self._compute_stats(all_results)
        pairwise_tests = self._pairwise_comparisons(all_results, method_stats)

        return ComparisonTable(
            method_stats=method_stats,
            pairwise_tests=pairwise_tests,
            all_results=all_results,
            config={
                "num_seeds": num_seeds,
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "lr": lr,
                "epochs": epochs,
                "n_samples": len(samples),
                "diagnosis_exploit_fraction": diagnosis.exploit_fraction,
            },
        )

    def _compute_stats(
        self, results: List[MethodResult]
    ) -> Dict[str, Dict]:
        """Compute per-method statistics."""
        from collections import defaultdict
        from scipy import stats as sp_stats

        by_method = defaultdict(list)
        for r in results:
            by_method[r.method].append(r.exploit_resistance)

        method_stats = {}
        for method, values in by_method.items():
            arr = np.array(values)
            n = len(arr)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if n > 1 else 0.0
            se = std / np.sqrt(n) if n > 1 else 0.0
            ci95 = float(sp_stats.t.ppf(0.975, df=max(n - 1, 1)) * se)

            method_stats[method] = {
                "mean": mean,
                "std": std,
                "se": se,
                "ci95": ci95,
                "n": n,
                "values": values,
            }

        return method_stats

    def _pairwise_comparisons(
        self, results: List[MethodResult], method_stats: Dict
    ) -> Dict[str, Dict]:
        """Pairwise Welch t-tests and Cohen's d."""
        from scipy import stats as sp_stats

        methods = sorted(method_stats.keys())
        tests = {}

        for i, a in enumerate(methods):
            for b in methods[i + 1 :]:
                va = np.array(method_stats[a]["values"])
                vb = np.array(method_stats[b]["values"])

                if len(va) < 2 or len(vb) < 2:
                    continue

                t_stat, p_value = sp_stats.ttest_ind(va, vb, equal_var=False)

                # Cohen's d
                pooled_std = np.sqrt(
                    ((len(va) - 1) * va.std(ddof=1) ** 2
                     + (len(vb) - 1) * vb.std(ddof=1) ** 2)
                    / (len(va) + len(vb) - 2)
                )
                cohens_d = (va.mean() - vb.mean()) / max(pooled_std, 1e-10)

                effect = (
                    "negligible" if abs(cohens_d) < 0.2
                    else "small" if abs(cohens_d) < 0.5
                    else "medium" if abs(cohens_d) < 0.8
                    else "large"
                )

                tests[f"{a} vs {b}"] = {
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "cohens_d": float(cohens_d),
                    "effect_size": effect,
                    "significant_005": bool(p_value < 0.05),
                }

        return tests

    @staticmethod
    def print_table(table: ComparisonTable) -> str:
        """Format comparison table as markdown."""
        lines = [
            "| Method | Exploit Resistance | Std | 95% CI | N |",
            "|--------|-------------------|-----|--------|---|",
        ]
        for method in sorted(table.method_stats.keys()):
            s = table.method_stats[method]
            lines.append(
                f"| {method} | {s['mean']:.3f} | {s['std']:.3f} | "
                f"±{s['ci95']:.3f} | {s['n']} |"
            )

        lines.append("")
        lines.append("**Pairwise comparisons:**")
        for pair, test in sorted(table.pairwise_tests.items()):
            sig = "*" if test["significant_005"] else ""
            lines.append(
                f"- {pair}: d={test['cohens_d']:.2f} ({test['effect_size']}), "
                f"p={test['p_value']:.4f}{sig}"
            )

        result = "\n".join(lines)
        logger.info(f"\n{result}")
        return result

    def save_results(
        self, table: ComparisonTable, output_dir: Optional[str] = None
    ) -> Path:
        """Save results to JSON."""
        out_dir = Path(output_dir or self.config.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "optimizer_comparison.json"

        data = {
            "method_stats": {
                k: {kk: vv for kk, vv in v.items() if kk != "values"}
                for k, v in table.method_stats.items()
            },
            "pairwise_tests": table.pairwise_tests,
            "all_results": [
                {
                    "method": r.method,
                    "seed": r.seed,
                    "exploit_resistance": r.exploit_resistance,
                    "final_loss": r.final_loss,
                }
                for r in table.all_results
            ],
            "config": table.config,
        }

        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Results saved to {path}")
        return path
