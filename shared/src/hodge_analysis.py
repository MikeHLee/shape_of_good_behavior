"""Hodge decomposition analysis of exploit preferences (Track 1).

Reuses existing feedback_geometry code for H1 computation,
preference filtering, and reward model training.
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig, FEEDBACK_GEOMETRY_SRC
from .preference_mapper import MappingResult

logger = logging.getLogger(__name__)


def _ensure_feedback_geometry_importable():
    """Add feedback_geometry/src to sys.path if needed."""
    src_str = str(FEEDBACK_GEOMETRY_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


@dataclass
class H1AnalysisResult:
    """Result of H1 structure analysis on exploit preferences."""

    h1_overall: float
    h1_per_category: Dict[str, float]
    decomposition: object  # HodgeDecomposition from hodge_utils
    n_edges: int
    n_items: int
    conditional_h1: Optional[Tuple[float, float, Dict]] = None  # marginal, conditional, breakdown


@dataclass
class TrainingResult:
    """Result of training standard vs Hodge-filtered reward models."""

    seed: int
    standard_resistance: float  # Fraction of pairs where RM ranks ideal > exploit
    hodge_resistance: float
    standard_loss: List[float]
    hodge_loss: List[float]
    h1_before: float
    h1_after: float
    filter_info: Dict = field(default_factory=dict)
    diagnostic_resistance: Optional[float] = None  # Selective Hodge filtering
    diagnostic_loss: Optional[List[float]] = None


@dataclass
class MultiSeedResult:
    """Aggregated results across multiple seeds."""

    seed_results: List[TrainingResult]
    standard_stats: Dict  # from compute_statistics
    hodge_stats: Dict  # from compute_statistics
    comparison: Dict  # from compare_methods (Welch t, Cohen's d)


class HodgeRewardHackingAnalyzer:
    """Analyze exploit preference cycles via Hodge decomposition and train filtered RMs."""

    def __init__(self, config: PipelineConfig, mapping: MappingResult):
        self.config = config
        self.mapping = mapping
        _ensure_feedback_geometry_importable()

    def analyze_h1_structure(self) -> H1AnalysisResult:
        """Compute H1 on the full preference graph and per-category."""
        from hodge_utils import (
            compute_h1_from_preferences,
            HodgeDecomposition,
        )

        edges = self.mapping.preference_edges
        n_items = self.mapping.n_items

        logger.info(f"Analyzing H1 structure: {len(edges)} edges, {n_items} items")

        # Overall H1
        h1_overall, decomposition = compute_h1_from_preferences(edges, n_items)
        logger.info(f"Overall H1 magnitude: {h1_overall:.4f}")

        # Per-category H1
        h1_per_category: Dict[str, float] = {}
        category_edges: Dict[str, List[Tuple[int, int, float]]] = {}

        for ep in self.mapping.embedding_pairs:
            cat = ep.category
            category_edges.setdefault(cat, [])

        # Map edges back to categories via item IDs
        # Use embedding_pairs to build category-to-edge mapping
        pair_edge_map: Dict[Tuple[int, int], str] = {}
        for i, ep in enumerate(self.mapping.embedding_pairs):
            if i < len(edges):
                edge = edges[i]
                pair_edge_map[(edge[0], edge[1])] = ep.category

        for edge in edges:
            cat = pair_edge_map.get((edge[0], edge[1]), "cross_category")
            category_edges.setdefault(cat, []).append(edge)

        for cat, cat_edges in category_edges.items():
            if len(cat_edges) < 3:
                continue
            try:
                cat_h1, _ = compute_h1_from_preferences(cat_edges, n_items)
                h1_per_category[cat] = cat_h1
            except Exception as e:
                logger.warning(f"H1 computation failed for category {cat}: {e}")
                h1_per_category[cat] = 0.0

        # Conditional H1 (valid vs invalid cycles) if we have context
        conditional_h1 = None
        try:
            from hodge_utils import compute_conditional_h1, ContextualPreference

            ctx_prefs = []
            for i, (edge, ep) in enumerate(
                zip(edges[: len(self.mapping.embedding_pairs)], self.mapping.embedding_pairs)
            ):
                # Use category as context_id
                cat_id = hash(ep.category) % 10000
                ctx_prefs.append(
                    ContextualPreference(
                        context_id=cat_id,
                        item_a=edge[0],
                        item_b=edge[1],
                        preference=edge[2],
                    )
                )
            if ctx_prefs:
                marginal, conditional, breakdown = compute_conditional_h1(
                    ctx_prefs, n_items
                )
                conditional_h1 = (marginal, conditional, breakdown)
                logger.info(
                    f"Conditional H1: marginal={marginal:.4f}, conditional={conditional:.4f}"
                )
        except Exception as e:
            logger.warning(f"Conditional H1 computation failed: {e}")

        return H1AnalysisResult(
            h1_overall=h1_overall,
            h1_per_category=h1_per_category,
            decomposition=decomposition,
            n_edges=len(edges),
            n_items=n_items,
            conditional_h1=conditional_h1,
        )

    def train_and_evaluate(self, seed: int) -> TrainingResult:
        """Train standard RM vs Hodge-filtered RM for a single seed.

        Returns exploit resistance metrics for both models.
        """
        import torch
        from h1_reward_hacking_experiment_v2 import (
            PreferenceRewardModel,
            train_preference_reward_model,
            PreferencePair,
        )
        from hodge_utils import compute_h1_from_preferences

        np.random.seed(seed)
        torch.manual_seed(seed)

        edges = self.mapping.preference_edges
        n_items = self.mapping.n_items
        state_dim = self.config.reduced_dim

        # Build PreferencePair objects with state vectors
        preference_pairs = []
        for i, edge in enumerate(edges):
            # Use reduced exploit embedding as state vector (or zeros for cross-category)
            if i < len(self.mapping.exploit_embeddings_reduced):
                state = self.mapping.exploit_embeddings_reduced[i]
            else:
                state = np.zeros(state_dim)

            preference_pairs.append(
                PreferencePair(
                    item_a=edge[0],
                    item_b=edge[1],
                    preference=edge[2],
                    state=state,
                )
            )

        # Train standard RM (no filtering)
        standard_model = PreferenceRewardModel(
            state_dim=state_dim,
            n_items=n_items,
            hidden_dim=self.config.rm_hidden_dim,
        )
        standard_loss, _, _ = train_preference_reward_model(
            standard_model,
            preference_pairs,
            epochs=self.config.rm_epochs,
            lr=self.config.rm_lr,
            hodge_filter=False,
            n_items=n_items,
        )

        # Train Hodge-filtered RM
        hodge_model = PreferenceRewardModel(
            state_dim=state_dim,
            n_items=n_items,
            hidden_dim=self.config.rm_hidden_dim,
        )
        hodge_loss, final_h1, filter_info = train_preference_reward_model(
            hodge_model,
            preference_pairs,
            epochs=self.config.rm_epochs,
            lr=self.config.rm_lr,
            hodge_filter=True,
            h1_threshold=self.config.h1_threshold,
            n_items=n_items,
        )

        # Train Diagnostic RM (selective filtering — preserves genuine tensions)
        diagnostic_resistance = None
        diagnostic_loss_list = None
        try:
            from .hodge_diagnostic import HodgeDiagnosticCritic

            diag_critic = HodgeDiagnosticCritic(self.config)
            categories = [ep.category for ep in self.mapping.embedding_pairs]
            diagnosis = diag_critic.diagnose(edges, n_items, categories=categories)
            filtered = diag_critic.selective_filter(edges, n_items, diagnosis)

            # Build PreferencePair objects from selectively filtered edges
            diag_pairs = []
            for i, edge in enumerate(filtered.edges):
                if i < len(self.mapping.exploit_embeddings_reduced):
                    state = self.mapping.exploit_embeddings_reduced[i]
                else:
                    state = np.zeros(state_dim)
                diag_pairs.append(
                    PreferencePair(
                        item_a=edge[0],
                        item_b=edge[1],
                        preference=edge[2],
                        state=state,
                    )
                )

            diag_model = PreferenceRewardModel(
                state_dim=state_dim,
                n_items=n_items,
                hidden_dim=self.config.rm_hidden_dim,
            )
            diagnostic_loss_list, _, _ = train_preference_reward_model(
                diag_model,
                diag_pairs,
                epochs=self.config.rm_epochs,
                lr=self.config.rm_lr,
                hodge_filter=False,  # Already filtered selectively
                n_items=n_items,
            )
            diagnostic_resistance = self._evaluate_exploit_resistance(
                diag_model, state_dim
            )
        except Exception as e:
            logger.warning(f"Diagnostic RM training failed: {e}")

        # Evaluate exploit resistance
        standard_resistance = self._evaluate_exploit_resistance(
            standard_model, state_dim
        )
        hodge_resistance = self._evaluate_exploit_resistance(
            hodge_model, state_dim
        )

        # Compute H1 before/after
        h1_before, _ = compute_h1_from_preferences(edges, n_items)
        h1_after = filter_info.get("h1_after", final_h1)

        diag_str = f", diagnostic={diagnostic_resistance:.3f}" if diagnostic_resistance is not None else ""
        logger.info(
            f"Seed {seed}: standard={standard_resistance:.3f}, "
            f"hodge={hodge_resistance:.3f}{diag_str}, "
            f"H1 {h1_before:.4f} → {h1_after:.4f}"
        )

        return TrainingResult(
            seed=seed,
            standard_resistance=standard_resistance,
            hodge_resistance=hodge_resistance,
            standard_loss=standard_loss,
            hodge_loss=hodge_loss,
            h1_before=h1_before,
            h1_after=h1_after,
            filter_info=filter_info,
            diagnostic_resistance=diagnostic_resistance,
            diagnostic_loss=diagnostic_loss_list,
        )

    def _evaluate_exploit_resistance(
        self, model, state_dim: int
    ) -> float:
        """Measure fraction of pairs where RM ranks ideal > exploit."""
        edges = self.mapping.preference_edges
        correct = 0
        total = 0

        for i, edge in enumerate(edges):
            ideal_id, exploit_id, _ = edge
            if i < len(self.mapping.exploit_embeddings_reduced):
                state = self.mapping.exploit_embeddings_reduced[i]
            else:
                state = np.zeros(state_dim)

            reward_ideal = model.get_reward(state, ideal_id)
            reward_exploit = model.get_reward(state, exploit_id)

            if reward_ideal > reward_exploit:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def run_multi_seed(self, num_seeds: Optional[int] = None) -> MultiSeedResult:
        """Run full multi-seed evaluation with statistical comparison."""
        from experiment_framework import compute_statistics, compare_methods

        if num_seeds is None:
            num_seeds = self.config.num_seeds_quick

        seed_results = []
        for seed in range(num_seeds):
            result = self.train_and_evaluate(seed)
            seed_results.append(result)

        standard_values = [r.standard_resistance for r in seed_results]
        hodge_values = [r.hodge_resistance for r in seed_results]
        diagnostic_values = [
            r.diagnostic_resistance for r in seed_results
            if r.diagnostic_resistance is not None
        ]

        standard_stats = compute_statistics(standard_values, "standard_resistance")
        hodge_stats = compute_statistics(hodge_values, "hodge_resistance")
        comparison = compare_methods(
            standard_values,
            hodge_values,
            method_a_name="Standard RM",
            method_b_name="Hodge-filtered RM",
        )

        # Diagnostic comparison (if available)
        diagnostic_comparison = None
        if diagnostic_values:
            diagnostic_stats = compute_statistics(diagnostic_values, "diagnostic_resistance")
            diagnostic_comparison = compare_methods(
                standard_values[:len(diagnostic_values)],
                diagnostic_values,
                method_a_name="Standard RM",
                method_b_name="Diagnostic RM",
            )
            diag_str = f"\n  Diagnostic RM: {diagnostic_stats.mean:.3f} ± {diagnostic_stats.std:.3f}"
        else:
            diagnostic_stats = None
            diag_str = ""

        logger.info(
            f"Multi-seed results ({num_seeds} seeds):\n"
            f"  Standard RM: {standard_stats.mean:.3f} ± {standard_stats.std:.3f}\n"
            f"  Hodge RM:    {hodge_stats.mean:.3f} ± {hodge_stats.std:.3f}"
            f"{diag_str}\n"
            f"  Cohen's d (std vs hodge): {comparison['cohens_d']:.3f} ({comparison['effect_size']})\n"
            f"  p-value:     {comparison['welch_t']['p_value']:.4f}"
        )

        return MultiSeedResult(
            seed_results=seed_results,
            standard_stats=standard_stats.to_dict(),
            hodge_stats=hodge_stats.to_dict(),
            comparison=comparison,
        )
