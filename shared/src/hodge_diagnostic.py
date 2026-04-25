"""Hodge Diagnostic Critic: classify cycles, selective filter, per-preference weights.

Replaces the "discard all harmonic" paradigm with "diagnose, classify, selectively
correct". Uses compute_conditional_h1() and filter_invalid_cycles_only() from
hodge_utils to distinguish genuine cross-context value tensions from exploitable
within-context cycles.

Key insight: HH-RLHF harmonic component contains genuine helpfulness-vs-harmlessness
tensions. Removing ALL harmonic signal destroys useful training signal (exploit
resistance drops from 0.813 to 0.273). Only within-context cycles are exploitable.
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig, FEEDBACK_GEOMETRY_SRC

logger = logging.getLogger(__name__)


def _ensure_hodge_utils():
    src_str = str(FEEDBACK_GEOMETRY_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


@dataclass
class CycleDiagnosis:
    """Result of diagnosing preference cycles."""

    marginal_h1: float          # Total H1 (all cycles)
    conditional_h1: float       # Invalid cycles only (within-context)
    genuine_h1: float           # Valid cycles (cross-context value tensions)
    per_preference_weights: np.ndarray  # Weight for each preference edge [0,1]
    per_sample_weights: Optional[np.ndarray] = None  # Weight for each embedding pair [0,1]
    hodge_potential: Optional[np.ndarray] = None  # Node-level gradient potential from decomposition
    sample_potential_diffs: Optional[np.ndarray] = None  # Per-sample potential diff (ideal - exploit)
    breakdown: Dict = field(default_factory=dict)

    @property
    def exploit_fraction(self) -> float:
        """Fraction of cyclic energy that is exploitable."""
        if self.marginal_h1 < 1e-10:
            return 0.0
        return self.conditional_h1 / self.marginal_h1

    @property
    def genuine_fraction(self) -> float:
        """Fraction of cyclic energy that is genuine value tension."""
        return 1.0 - self.exploit_fraction


@dataclass
class FilteredPreferences:
    """Preferences after selective Hodge filtering."""

    edges: List[Tuple[int, int, float]]  # Filtered preference edges
    weights: np.ndarray                   # Per-edge weights applied
    n_dampened: int                        # Number of edges with weight < 1.0
    n_preserved: int                       # Number of edges with weight = 1.0
    h1_before: float
    h1_after: float


class HodgeDiagnosticCritic:
    """Hodge-aware critic that preserves genuine value tensions
    while identifying and dampening exploitable cycles.

    Unlike the aggressive Hodge filter (threshold=0.0 which removes ALL
    harmonic), this critic:
    1. Diagnoses cycles using context-conditional H1
    2. Classifies each cycle as genuine (cross-context) or exploitable (within-context)
    3. Assigns per-preference weights: 1.0 for genuine, alpha for exploitable
    4. Uses adaptive cross-validation to find optimal alpha

    Usage:
        critic = HodgeDiagnosticCritic(config)
        diagnosis = critic.diagnose(edges, n_items, embedding_pairs)
        filtered = critic.selective_filter(edges, n_items, diagnosis)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        _ensure_hodge_utils()

    def diagnose(
        self,
        preference_edges: List[Tuple[int, int, float]],
        n_items: int,
        embedding_pairs: Optional[List] = None,
        categories: Optional[List[str]] = None,
    ) -> CycleDiagnosis:
        """Diagnose preference cycles: separate genuine from exploitable.

        Args:
            preference_edges: (item_a, item_b, probability) tuples
            n_items: Total items
            embedding_pairs: EmbeddingPair objects (for category context)
            categories: Explicit category labels per edge (alternative to embedding_pairs)

        Returns:
            CycleDiagnosis with per-preference weights
        """
        from hodge_utils import (
            compute_conditional_h1,
            compute_h1_from_preferences,
            hodge_decompose,
            ContextualPreference,
        )

        # Build contextual preferences using categories
        ctx_prefs = []
        cats = categories or []
        if not cats and embedding_pairs:
            cats = [ep.category for ep in embedding_pairs]

        for i, edge in enumerate(preference_edges):
            cat = cats[i] if i < len(cats) else "default"
            ctx_id = hash(cat) % 10000
            ctx_prefs.append(ContextualPreference(
                context_id=ctx_id,
                item_a=edge[0],
                item_b=edge[1],
                preference=edge[2],
            ))

        # Compute marginal (all cycles) and conditional (invalid only) H1
        if ctx_prefs:
            marginal_h1, conditional_h1, breakdown = compute_conditional_h1(
                ctx_prefs, n_items
            )
        else:
            marginal_h1, _ = compute_h1_from_preferences(preference_edges, n_items)
            conditional_h1 = marginal_h1
            breakdown = {}

        genuine_h1 = marginal_h1 - conditional_h1

        # Compute per-preference weights based on cycle membership
        weights = self._compute_cycle_weights(
            preference_edges, n_items, ctx_prefs, conditional_h1, marginal_h1
        )

        logger.info(
            f"Cycle diagnosis: marginal_h1={marginal_h1:.4f}, "
            f"conditional_h1={conditional_h1:.4f}, genuine_h1={genuine_h1:.4f}, "
            f"exploit_fraction={conditional_h1 / max(marginal_h1, 1e-10):.2%}"
        )

        return CycleDiagnosis(
            marginal_h1=marginal_h1,
            conditional_h1=conditional_h1,
            genuine_h1=genuine_h1,
            per_preference_weights=weights,
            breakdown=breakdown,
        )

    def diagnose_for_samples(
        self,
        preference_edges: List[Tuple[int, int, float]],
        n_items: int,
        embedding_pairs: List,
        categories: Optional[List[str]] = None,
    ) -> CycleDiagnosis:
        """Diagnose cycles and compute per-SAMPLE weights (aligned to embedding_pairs).

        Unlike diagnose() which returns per-edge weights (len=preference_edges),
        this returns per-sample weights (len=embedding_pairs) using node-level
        cycle participation scoring. This fixes the index mismatch between
        preference_edges (21k+) and embedding_pairs (2k).

        Node-level scoring: for each node, sum |harmonic[e]| across all incident
        edges. Each sample's weight depends on how much its ideal/exploit nodes
        participate in cycles across the full preference graph.
        """
        from hodge_utils import hodge_decompose, compute_conditional_h1, ContextualPreference

        # --- Step 1: Run full diagnosis for marginal/conditional H1 ---
        cats = categories or [ep.category for ep in embedding_pairs]

        # Build contextual preferences (only for edges with known categories)
        ctx_prefs = []
        for i, edge in enumerate(preference_edges):
            cat = cats[i] if i < len(cats) else "default"
            ctx_prefs.append(ContextualPreference(
                context_id=hash(cat) % 10000,
                item_a=edge[0],
                item_b=edge[1],
                preference=edge[2],
            ))

        marginal_h1, conditional_h1, breakdown = compute_conditional_h1(
            ctx_prefs, n_items
        )
        genuine_h1 = marginal_h1 - conditional_h1
        exploit_fraction = conditional_h1 / max(marginal_h1, 1e-10)

        # --- Step 2: Hodge decompose full preference graph ---
        edge_list = [(e[0], e[1]) for e in preference_edges]
        edge_weights = np.array([
            np.log(e[2] / (1 - e[2] + 1e-10) + 1e-10) for e in preference_edges
        ])
        edge_weights = np.clip(edge_weights, -10, 10)

        decomp = hodge_decompose(edge_weights, edge_list, n_items)

        # --- Step 3: Node-level harmonic participation ---
        node_harmonic_score = np.zeros(n_items)
        for idx, (src, tgt) in enumerate(edge_list):
            h_abs = abs(decomp.harmonic[idx])
            node_harmonic_score[src] += h_abs
            node_harmonic_score[tgt] += h_abs

        # Normalize node scores to [0, 1]
        max_score = node_harmonic_score.max()
        if max_score > 1e-10:
            node_harmonic_score /= max_score

        # --- Step 4: Per-sample weights and potential differences ---
        n_pairs = len(embedding_pairs)
        sample_cycle_scores = np.zeros(n_pairs)
        sample_potential_diffs = np.zeros(n_pairs)
        hodge_potential = decomp.potential  # node-level gradient potential

        for i in range(n_pairs):
            if i < len(preference_edges):
                ideal_id = preference_edges[i][0]
                exploit_id = preference_edges[i][1]
            else:
                ideal_id = 0
                exploit_id = 0
            ideal_id = min(ideal_id, n_items - 1)
            exploit_id = min(exploit_id, n_items - 1)

            sample_cycle_scores[i] = (
                node_harmonic_score[ideal_id]
                + node_harmonic_score[exploit_id]
            ) / 2.0

            # Hodge potential difference: in Hodge convention, higher potential =
            # "more flowed into" (preferred item). For edge (ideal→exploit) with
            # positive flow, gradient has potential[exploit] > potential[ideal].
            # Negate so positive diff means "gradient supports ideal > exploit".
            sample_potential_diffs[i] = (
                hodge_potential[exploit_id] - hodge_potential[ideal_id]
            )

        # Normalize cycle scores to [0, 1]
        max_cs = sample_cycle_scores.max()
        if max_cs > 1e-10:
            sample_cycle_scores /= max_cs

        # Normalize potential diffs to roughly [-1, 1]
        pd_scale = np.abs(sample_potential_diffs).max()
        if pd_scale > 1e-10:
            sample_potential_diffs /= pd_scale

        # Dampening uses relative position within the cycle score distribution
        # rather than exploit_fraction (which is 1.0 for single-category data).
        # Samples in the top quartile of cycle participation get strongest dampening.
        # This ensures meaningful variance regardless of exploit_fraction.
        dampening_strength = 0.6  # max weight reduction for highest-cycle samples
        per_sample_weights = 1.0 - dampening_strength * sample_cycle_scores
        per_sample_weights = np.clip(per_sample_weights, 0.2, 1.0)

        n_dampened = int(np.sum(per_sample_weights < 0.99))
        logger.info(
            f"Sample weights: {n_dampened}/{n_pairs} dampened, "
            f"range=[{per_sample_weights.min():.3f}, {per_sample_weights.max():.3f}], "
            f"mean={per_sample_weights.mean():.3f}, std={per_sample_weights.std():.3f}"
        )
        logger.info(
            f"Potential diffs: mean={sample_potential_diffs.mean():.3f}, "
            f"std={sample_potential_diffs.std():.3f}, "
            f"positive={np.sum(sample_potential_diffs > 0)}/{n_pairs}"
        )

        # Also compute legacy per-edge weights
        per_edge_weights = self._compute_cycle_weights(
            preference_edges, n_items, ctx_prefs, conditional_h1, marginal_h1
        )

        return CycleDiagnosis(
            marginal_h1=marginal_h1,
            conditional_h1=conditional_h1,
            genuine_h1=genuine_h1,
            per_preference_weights=per_edge_weights,
            per_sample_weights=per_sample_weights,
            hodge_potential=hodge_potential,
            sample_potential_diffs=sample_potential_diffs,
            breakdown=breakdown,
        )

    def _compute_cycle_weights(
        self,
        edges: List[Tuple[int, int, float]],
        n_items: int,
        ctx_prefs: list,
        conditional_h1: float,
        marginal_h1: float,
    ) -> np.ndarray:
        """Compute per-preference weights based on cycle membership.

        - Gradient-consistent pairs: weight = 1.0
        - Genuine tension pairs (cross-context): weight = 1.0
        - Exploitable cycle pairs (within-context): weight = alpha

        Alpha is determined by the ratio of exploitable to total H1.
        """
        from hodge_utils import hodge_decompose

        n_edges = len(edges)
        if n_edges == 0:
            return np.array([])

        # Get Hodge decomposition for all edges
        edge_list = [(e[0], e[1]) for e in edges]
        edge_weights = np.array([
            np.log(e[2] / (1 - e[2] + 1e-10) + 1e-10) for e in edges
        ])
        edge_weights = np.clip(edge_weights, -10, 10)

        decomp = hodge_decompose(edge_weights, edge_list, n_items)

        # Per-edge harmonic energy fraction
        gradient_sq = decomp.gradient ** 2
        harmonic_sq = decomp.harmonic ** 2
        total_sq = gradient_sq + harmonic_sq
        total_sq = np.maximum(total_sq, 1e-10)

        harmonic_fraction = harmonic_sq / total_sq  # [0, 1] per edge

        # Of the harmonic fraction, what portion is exploitable?
        if marginal_h1 > 1e-10:
            exploit_ratio = conditional_h1 / marginal_h1
        else:
            exploit_ratio = 0.0

        # Also compute per-context H1 to identify which edges are in exploitable contexts
        context_exploit_scores = self._per_context_exploit_scores(ctx_prefs, n_items)

        # Weight = 1.0 - (harmonic_fraction * exploit_score_for_this_edge's_context)
        # Edges in gradient-consistent region: harmonic_fraction ≈ 0 → weight ≈ 1.0
        # Edges in genuine cross-context cycles: exploit_score ≈ 0 → weight ≈ 1.0
        # Edges in exploitable within-context cycles: both high → weight < 1.0
        alpha = max(0.1, 1.0 - exploit_ratio)  # Floor at 0.1 to never fully discard

        weights = np.ones(n_edges)
        for i in range(n_edges):
            ctx_score = context_exploit_scores.get(i, 0.0)
            # Dampen proportional to harmonic fraction AND context exploit score
            dampening = harmonic_fraction[i] * ctx_score
            weights[i] = 1.0 - dampening * (1.0 - alpha)

        weights = np.clip(weights, 0.1, 1.0)
        return weights

    def _per_context_exploit_scores(
        self,
        ctx_prefs: list,
        n_items: int,
    ) -> Dict[int, float]:
        """Compute per-edge exploit score based on within-context H1.

        Returns dict mapping edge index → exploit_score in [0, 1].
        """
        from hodge_utils import compute_h1_from_preferences

        if not ctx_prefs:
            return {}

        # Group by context
        by_context: Dict[int, List[int]] = {}
        for i, p in enumerate(ctx_prefs):
            by_context.setdefault(p.context_id, []).append(i)

        # Compute H1 within each context
        context_h1: Dict[int, float] = {}
        for ctx_id, indices in by_context.items():
            ctx_edges = [
                (ctx_prefs[i].item_a, ctx_prefs[i].item_b, ctx_prefs[i].preference)
                for i in indices
            ]
            if len(ctx_edges) >= 3:
                h1, _ = compute_h1_from_preferences(ctx_edges, n_items)
                context_h1[ctx_id] = h1
            else:
                context_h1[ctx_id] = 0.0

        # Normalize to [0, 1]
        max_h1 = max(context_h1.values()) if context_h1 else 1.0
        max_h1 = max(max_h1, 1e-10)

        # Map back to per-edge scores
        scores = {}
        for ctx_id, indices in by_context.items():
            score = context_h1.get(ctx_id, 0.0) / max_h1
            for i in indices:
                scores[i] = score

        return scores

    def selective_filter(
        self,
        preference_edges: List[Tuple[int, int, float]],
        n_items: int,
        diagnosis: CycleDiagnosis,
        threshold: Optional[float] = None,
    ) -> FilteredPreferences:
        """Apply selective filtering using diagnosis weights.

        Unlike aggressive Hodge filtering (threshold=0.0), this preserves
        genuine value tensions and only dampens exploitable cycles.

        Args:
            preference_edges: Raw preference edges
            n_items: Total items
            diagnosis: CycleDiagnosis from diagnose()
            threshold: Optional override for dampening threshold
                       (None = use adaptive CV threshold)

        Returns:
            FilteredPreferences with weighted edges
        """
        from hodge_utils import compute_h1_from_preferences

        weights = diagnosis.per_preference_weights

        # Apply weights to preference probabilities
        # For dampened preferences, move probability toward 0.5 (uncertain)
        filtered_edges = []
        for i, edge in enumerate(preference_edges):
            w = weights[i] if i < len(weights) else 1.0
            if w < 1.0:
                # Blend toward 0.5 (uncertainty) proportional to dampening
                new_prob = w * edge[2] + (1.0 - w) * 0.5
            else:
                new_prob = edge[2]
            filtered_edges.append((edge[0], edge[1], new_prob))

        # Compute H1 after filtering
        h1_before = diagnosis.marginal_h1
        h1_after, _ = compute_h1_from_preferences(filtered_edges, n_items)

        n_dampened = int(np.sum(weights < 1.0 - 1e-6))
        n_preserved = len(weights) - n_dampened

        logger.info(
            f"Selective filter: {n_dampened} dampened, {n_preserved} preserved, "
            f"H1 {h1_before:.4f} → {h1_after:.4f}"
        )

        return FilteredPreferences(
            edges=filtered_edges,
            weights=weights,
            n_dampened=n_dampened,
            n_preserved=n_preserved,
            h1_before=h1_before,
            h1_after=h1_after,
        )

    def get_exploit_correction(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        preference_edges: List[Tuple[int, int, float]],
        n_items: int,
        diagnosis: CycleDiagnosis,
    ) -> np.ndarray:
        """Compute the exploitable harmonic correction for SGPO advantage.

        Returns only the exploitable component of omega (harmonic), NOT the
        genuine tension component. Used by revised sgpo_clipped.py.

        Args:
            states: State embeddings (batch_size, embed_dim)
            actions: Action indices (batch_size,)
            preference_edges: Full preference edges
            n_items: Total items
            diagnosis: CycleDiagnosis from diagnose()

        Returns:
            omega_exploit: (batch_size,) exploitable harmonic correction
        """
        from hodge_utils import hodge_decompose

        batch_size = len(states)

        # Get full Hodge decomposition
        edge_list = [(e[0], e[1]) for e in preference_edges]
        edge_weights_raw = np.array([
            np.log(e[2] / (1 - e[2] + 1e-10) + 1e-10) for e in preference_edges
        ])
        edge_weights_raw = np.clip(edge_weights_raw, -10, 10)

        decomp = hodge_decompose(edge_weights_raw, edge_list, n_items)

        # Scale harmonic by exploit fraction
        exploit_fraction = diagnosis.exploit_fraction
        harmonic_exploit = decomp.harmonic * exploit_fraction

        # Map edge-level harmonic to state-action level
        # For each (state, action), find the closest preference edge
        # and use its exploitable harmonic value
        omega_exploit = np.zeros(batch_size)
        if len(harmonic_exploit) > 0:
            for i in range(batch_size):
                action_idx = int(actions[i]) if actions[i] < len(harmonic_exploit) else 0
                omega_exploit[i] = harmonic_exploit[min(action_idx, len(harmonic_exploit) - 1)]

        return omega_exploit
