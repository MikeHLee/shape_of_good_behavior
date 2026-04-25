"""Per-category exploit resistance evaluation and model comparison.

Provides detailed breakdowns of how well standard vs Hodge-filtered
reward models resist exploits across different attack categories.
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig, FEEDBACK_GEOMETRY_SRC
from .preference_mapper import MappingResult, DangerRegionSpec
from .hodge_analysis import MultiSeedResult

logger = logging.getLogger(__name__)


@dataclass
class CategoryResistance:
    """Exploit resistance metrics for a single category."""

    category: str
    n_pairs: int
    resistance_rate: float  # Fraction where ideal > exploit
    mean_reward_gap: float  # Mean(reward_ideal - reward_exploit)
    std_reward_gap: float


@dataclass
class ExploitResistanceReport:
    """Comprehensive exploit resistance evaluation report."""

    overall_resistance: float
    per_category: List[CategoryResistance]
    per_source: Dict[str, float]  # "trace", "hh_rlhf" → resistance
    worst_categories: List[str]
    best_categories: List[str]
    n_total_pairs: int

    # Geometric features per category
    category_features: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class ModelComparisonReport:
    """Comparison of standard vs Hodge-filtered models across seeds."""

    standard_report: ExploitResistanceReport
    hodge_report: ExploitResistanceReport
    multi_seed: MultiSeedResult
    per_category_improvement: Dict[str, float]  # category → (hodge - standard)


class ExploitResistanceEvaluator:
    """Evaluate and compare reward models on exploit resistance."""

    def __init__(self, config: PipelineConfig, mapping: MappingResult):
        self.config = config
        self.mapping = mapping

        src_str = str(FEEDBACK_GEOMETRY_SRC)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    def evaluate_model(self, model, label: str = "model") -> ExploitResistanceReport:
        """Evaluate a trained reward model's exploit resistance in detail."""
        edges = self.mapping.preference_edges
        pairs = self.mapping.embedding_pairs
        state_dim = self.config.reduced_dim

        # Per-category tracking
        category_correct: Dict[str, int] = {}
        category_total: Dict[str, int] = {}
        category_gaps: Dict[str, List[float]] = {}
        source_correct: Dict[str, int] = {}
        source_total: Dict[str, int] = {}

        total_correct = 0
        total = 0

        for i, edge in enumerate(edges):
            ideal_id, exploit_id, _ = edge

            if i < len(self.mapping.exploit_embeddings_reduced):
                state = self.mapping.exploit_embeddings_reduced[i]
            else:
                state = np.zeros(state_dim)

            reward_ideal = model.get_reward(state, ideal_id)
            reward_exploit = model.get_reward(state, exploit_id)

            gap = reward_ideal - reward_exploit
            is_correct = reward_ideal > reward_exploit

            if is_correct:
                total_correct += 1
            total += 1

            # Per-category
            if i < len(pairs):
                cat = pairs[i].category
                category_correct[cat] = category_correct.get(cat, 0) + int(is_correct)
                category_total[cat] = category_total.get(cat, 0) + 1
                category_gaps.setdefault(cat, []).append(gap)

                # Per-source
                source = "trace" if "trace" in cat.lower() or cat not in [
                    "harmless-base", "helpful-base"
                ] else "hh_rlhf"
                source_correct[source] = source_correct.get(source, 0) + int(is_correct)
                source_total[source] = source_total.get(source, 0) + 1

        # Build per-category results
        per_category = []
        for cat in sorted(category_total.keys()):
            n = category_total[cat]
            correct = category_correct.get(cat, 0)
            gaps = category_gaps.get(cat, [])
            per_category.append(
                CategoryResistance(
                    category=cat,
                    n_pairs=n,
                    resistance_rate=correct / n if n > 0 else 0.0,
                    mean_reward_gap=float(np.mean(gaps)) if gaps else 0.0,
                    std_reward_gap=float(np.std(gaps)) if gaps else 0.0,
                )
            )

        # Sort for worst/best
        sorted_cats = sorted(per_category, key=lambda c: c.resistance_rate)
        worst = [c.category for c in sorted_cats[:3]]
        best = [c.category for c in sorted_cats[-3:]]

        per_source = {
            s: source_correct.get(s, 0) / source_total[s]
            for s in source_total
            if source_total[s] > 0
        }

        report = ExploitResistanceReport(
            overall_resistance=total_correct / total if total > 0 else 0.0,
            per_category=per_category,
            per_source=per_source,
            worst_categories=worst,
            best_categories=best,
            n_total_pairs=total,
        )

        # Compute geometric features
        report.category_features = self._compute_exploit_embedding_signatures()

        logger.info(
            f"[{label}] Overall resistance: {report.overall_resistance:.3f} "
            f"({total_correct}/{total})"
        )
        return report

    def compare_models(
        self,
        standard_model,
        hodge_model,
        multi_seed: MultiSeedResult,
    ) -> ModelComparisonReport:
        """Compare standard vs Hodge-filtered models."""
        standard_report = self.evaluate_model(standard_model, "standard")
        hodge_report = self.evaluate_model(hodge_model, "hodge-filtered")

        # Per-category improvement
        standard_by_cat = {
            c.category: c.resistance_rate for c in standard_report.per_category
        }
        hodge_by_cat = {
            c.category: c.resistance_rate for c in hodge_report.per_category
        }

        improvement = {}
        for cat in standard_by_cat:
            if cat in hodge_by_cat:
                improvement[cat] = hodge_by_cat[cat] - standard_by_cat[cat]

        logger.info("Per-category improvement (Hodge - Standard):")
        for cat, imp in sorted(improvement.items(), key=lambda x: -x[1]):
            logger.info(f"  {cat}: {imp:+.3f}")

        return ModelComparisonReport(
            standard_report=standard_report,
            hodge_report=hodge_report,
            multi_seed=multi_seed,
            per_category_improvement=improvement,
        )

    def _compute_exploit_embedding_signatures(self) -> Dict[str, Dict]:
        """Compute per-category geometric features from embeddings."""
        pairs = self.mapping.embedding_pairs
        danger_regions = self.mapping.danger_regions

        # Group by category
        cat_embeds: Dict[str, List[np.ndarray]] = {}
        cat_gradients: Dict[str, List[np.ndarray]] = {}
        for ep in pairs:
            cat_embeds.setdefault(ep.category, []).append(ep.exploit_embed)
            cat_gradients.setdefault(ep.category, []).append(
                ep.constitutional_gradient
            )

        # Build danger region centers for proximity computation
        danger_centers = np.array([dr.center for dr in danger_regions]) if danger_regions else None

        features: Dict[str, Dict] = {}
        for cat in cat_embeds:
            embeds = np.array(cat_embeds[cat])
            gradients = np.array(cat_gradients.get(cat, []))

            centroid = np.mean(embeds, axis=0)
            distances = np.linalg.norm(embeds - centroid, axis=1)

            feat: Dict = {
                "n_exploits": len(embeds),
                "cluster_radius": float(np.mean(distances)),
                "cluster_radius_std": float(np.std(distances)),
            }

            if len(gradients) > 0:
                mean_grad = np.mean(gradients, axis=0)
                feat["gradient_magnitude"] = float(np.linalg.norm(mean_grad))
                feat["gradient_consistency"] = float(
                    np.mean(
                        [
                            np.dot(g, mean_grad)
                            / (np.linalg.norm(g) * np.linalg.norm(mean_grad) + 1e-8)
                            for g in gradients
                        ]
                    )
                )

            if danger_centers is not None and len(danger_centers) > 0:
                # Use reduced-dim centroid for proximity if available
                min_dist = float(
                    np.min(np.linalg.norm(danger_centers - centroid[:len(danger_centers[0])], axis=1))
                ) if centroid.shape[0] >= danger_centers.shape[1] else float("inf")
                feat["black_hole_proximity"] = min_dist

            features[cat] = feat

        return features
