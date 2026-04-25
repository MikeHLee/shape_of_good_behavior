# -*- coding: utf-8 -*-
"""
Hodge Decomposition Utilities for Preference Data

Provides proper implementation of:
1. Boundary operators for preference graphs
2. Hodge decomposition: f = df + δg + h
3. H¹ magnitude computation and verification
4. Controlled H¹ injection

Mathematical Background:
- Preferences form a 1-cochain on the action graph (nodes=items, edges=comparisons)
- Hodge decomposition splits this into:
  - Exact (gradient): dφ where φ is a potential (consistent ranking)
  - Co-exact (co-gradient): δψ where ψ captures sinks/sources
  - Harmonic: h ∈ ker(d) ∩ ker(δ) - the cyclic inconsistency (H¹)
- H¹ ≠ 0 implies no scalar reward function can represent the preferences

References:
- Jiang et al. (2011): "Statistical ranking and combinatorial Hodge theory"
- Hirani (2003): "Discrete Exterior Calculus"
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import lsqr, lsmr
from scipy.linalg import lstsq
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class HodgeDecomposition:
    """Result of Hodge decomposition on preference data."""
    # Original edge flow (preferences)
    original: np.ndarray
    # Gradient component (derivable from potential)
    gradient: np.ndarray
    # Harmonic component (cyclic, H¹)
    harmonic: np.ndarray
    # Potential function on nodes
    potential: np.ndarray
    # H¹ magnitude (normalized norm of harmonic component)
    h1_magnitude: float
    # Number of edges
    n_edges: int
    # Number of nodes
    n_nodes: int


def build_boundary_operator(n_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build the boundary operator ∂₁: C₁ → C₀ (edges → nodes).
    
    For edge e = (i, j) oriented from i to j:
        ∂₁(e) = j - i  (formal sum of endpoints with signs)
    
    In matrix form: B[e, i] = -1, B[e, j] = +1
    
    Args:
        n_nodes: Number of nodes (items)
        edges: List of (source, target) edge pairs
        
    Returns:
        B: (n_edges, n_nodes) boundary matrix
    """
    n_edges = len(edges)
    B = np.zeros((n_edges, n_nodes))
    
    for idx, (i, j) in enumerate(edges):
        B[idx, i] = -1  # Source
        B[idx, j] = +1  # Target
    
    return B


def build_coboundary_operator(B: np.ndarray) -> np.ndarray:
    """
    Build the coboundary operator d = ∂₁ᵀ: C⁰ → C¹ (nodes → edges).
    
    For a function φ on nodes: (dφ)(e) = φ(j) - φ(i) for edge e = (i,j)
    
    Args:
        B: Boundary operator matrix
        
    Returns:
        d: Coboundary operator (transpose of boundary)
    """
    return B.T


def hodge_decompose(
    edge_weights: np.ndarray,
    edges: List[Tuple[int, int]],
    n_nodes: int
) -> HodgeDecomposition:
    """
    Perform Hodge decomposition on edge flow (preference data).
    
    Decomposes: w = dφ + h
    where:
        - dφ = gradient component (exact)
        - h = harmonic component (H¹)
    
    For graphs without 2-cells (no triangles enforced), we use:
        - gradient: projection onto im(d) = im(∂₁ᵀ)
        - harmonic: orthogonal complement
    
    Args:
        edge_weights: Weight on each edge (preference strength)
        edges: List of (source, target) pairs
        n_nodes: Number of nodes
        
    Returns:
        HodgeDecomposition with all components
    """
    n_edges = len(edges)
    
    if n_edges == 0:
        return HodgeDecomposition(
            original=np.array([]),
            gradient=np.array([]),
            harmonic=np.array([]),
            potential=np.zeros(n_nodes),
            h1_magnitude=0.0,
            n_edges=0,
            n_nodes=n_nodes
        )
    
    # Build boundary operator
    B = build_boundary_operator(n_nodes, edges)
    
    # Solve for potential: minimize ||Bᵀφ - w||²
    # This is: minimize ||dφ - w||² where d = Bᵀ
    # Normal equations: BBᵀφ = Bw
    # But BBᵀ is singular (constant functions in kernel), so we use least squares
    
    BtB = B @ B.T  # This is the graph Laplacian-like operator on edges
    BBt = B.T @ B  # This is the graph Laplacian on nodes
    
    # Solve BBᵀφ = Bw using regularized least squares
    # Add small regularization to handle the constant kernel
    BBt_reg = BBt + 1e-8 * np.eye(n_nodes)
    
    # Right-hand side: Bw
    Bw = B.T @ edge_weights
    
    # Solve for potential
    potential = np.linalg.solve(BBt_reg, Bw)
    
    # Center potential (remove constant ambiguity)
    potential = potential - np.mean(potential)
    
    # Gradient component: dφ = Bᵀφ
    gradient = B @ potential
    
    # Harmonic component: h = w - dφ
    harmonic = edge_weights - gradient
    
    # H¹ magnitude: normalized L² norm of harmonic component
    h1_magnitude = np.linalg.norm(harmonic) / np.sqrt(n_edges) if n_edges > 0 else 0.0
    
    return HodgeDecomposition(
        original=edge_weights.copy(),
        gradient=gradient,
        harmonic=harmonic,
        potential=potential,
        h1_magnitude=float(h1_magnitude),
        n_edges=n_edges,
        n_nodes=n_nodes
    )


def compute_h1_from_preferences(
    preferences: List[Tuple[int, int, float]],
    n_items: int
) -> Tuple[float, HodgeDecomposition]:
    """
    Compute H¹ magnitude from preference data.
    
    Args:
        preferences: List of (item_a, item_b, prob_a_wins) tuples
        n_items: Total number of items
        
    Returns:
        h1_magnitude: The H¹ invariant (cyclic inconsistency)
        decomposition: Full Hodge decomposition
    """
    # Build preference graph
    # Edge (i, j) has weight = log-odds of i > j (Bradley-Terry style)
    edge_dict = {}  # (min_idx, max_idx) -> list of log-odds
    
    for i, j, prob in preferences:
        if i == j:
            continue
        # Canonicalize edge direction
        if i < j:
            edge = (i, j)
            weight = np.log(prob / (1 - prob + 1e-10) + 1e-10)  # log-odds for i > j
        else:
            edge = (j, i)
            weight = np.log((1 - prob) / (prob + 1e-10) + 1e-10)  # log-odds for j > i
        
        if edge not in edge_dict:
            edge_dict[edge] = []
        edge_dict[edge].append(weight)
    
    # Average weights per edge
    edges = list(edge_dict.keys())
    edge_weights = np.array([np.mean(edge_dict[e]) for e in edges])
    
    # Clip extreme values
    edge_weights = np.clip(edge_weights, -10, 10)
    
    # Perform Hodge decomposition
    decomposition = hodge_decompose(edge_weights, edges, n_items)
    
    return decomposition.h1_magnitude, decomposition


def inject_h1_controlled(
    n_items: int,
    h1_target: float,
    base_noise: float = 0.1,
    n_comparisons_per_pair: int = 5
) -> Tuple[List[Tuple[int, int, float]], float]:
    """
    Generate preference data with controlled H¹ magnitude.
    
    Strategy:
    1. For h1_target = 0: Generate fully consistent preferences
    2. For h1_target > 0: Generate preferences with controlled cyclic component
       using direct harmonic injection via Hodge theory
    
    Args:
        n_items: Number of items
        h1_target: Target H¹ magnitude (0 = consistent, higher = more cyclic)
        base_noise: Noise level in base preferences
        n_comparisons_per_pair: Comparisons per item pair
        
    Returns:
        preferences: List of (item_a, item_b, prob_a_wins)
        measured_h1: Actual H¹ after injection
    """
    np.random.seed(None)  # Fresh randomness
    
    # Ground truth utilities (determines consistent ordering)
    utilities = np.linspace(0, 1, n_items)
    np.random.shuffle(utilities)
    
    # Build complete graph edges
    edges = [(i, j) for i in range(n_items) for j in range(i+1, n_items)]
    n_edges = len(edges)
    
    # Compute consistent (gradient) edge weights from utilities
    gradient_weights = np.array([utilities[i] - utilities[j] for i, j in edges])
    
    # Generate harmonic component (in kernel of boundary operator)
    # For a complete graph, the harmonic space is generated by cycles
    # We inject a random harmonic component scaled by h1_target
    if h1_target > 0:
        # Build boundary operator
        B = build_boundary_operator(n_items, edges)
        
        # Project random noise onto harmonic subspace (orthogonal to image of B^T)
        random_flow = np.random.randn(n_edges)
        
        # Remove gradient component: h = f - B @ (B^T B)^{-1} @ B^T @ f
        BBt = B.T @ B + 1e-8 * np.eye(n_items)
        potential = np.linalg.solve(BBt, B.T @ random_flow)
        gradient_part = B @ potential
        harmonic_part = random_flow - gradient_part
        
        # Normalize and scale harmonic component
        harmonic_norm = np.linalg.norm(harmonic_part) / np.sqrt(n_edges)
        if harmonic_norm > 1e-6:
            harmonic_part = harmonic_part * (h1_target / harmonic_norm)
    else:
        harmonic_part = np.zeros(n_edges)
    
    # Combine: total edge weight = gradient + harmonic
    total_weights = gradient_weights + harmonic_part
    
    # Add noise and convert to preferences
    preferences = []
    for idx, (i, j) in enumerate(edges):
        for _ in range(n_comparisons_per_pair):
            # Convert log-odds style weight to probability
            weight = total_weights[idx] + np.random.normal(0, base_noise)
            prob = 1.0 / (1.0 + np.exp(-weight * 2))  # Scale factor for spread
            prob = np.clip(prob, 0.05, 0.95)
            preferences.append((i, j, prob))
    
    # Measure actual H¹
    measured_h1, _ = compute_h1_from_preferences(preferences, n_items)
    
    return preferences, measured_h1


def verify_h1_injection(
    h1_targets: List[float],
    n_items: int = 20,
    n_trials: int = 5
) -> Dict[float, Dict]:
    """
    Verify that H¹ injection achieves target values.
    
    Args:
        h1_targets: List of target H¹ values
        n_items: Number of items
        n_trials: Trials per target
        
    Returns:
        Dictionary mapping target → {mean, std, correlation}
    """
    results = {}
    
    all_targets = []
    all_measured = []
    
    for target in h1_targets:
        measured = []
        for _ in range(n_trials):
            _, h1 = inject_h1_controlled(n_items, target)
            measured.append(h1)
            all_targets.append(target)
            all_measured.append(h1)
        
        results[target] = {
            'mean': np.mean(measured),
            'std': np.std(measured),
            'min': np.min(measured),
            'max': np.max(measured)
        }
    
    # Compute correlation
    correlation = np.corrcoef(all_targets, all_measured)[0, 1]
    
    print(f"H¹ Injection Verification (n_items={n_items}):")
    print(f"  Target → Measured (mean ± std)")
    for target, data in results.items():
        print(f"  {target:.2f} → {data['mean']:.3f} ± {data['std']:.3f}")
    print(f"  Correlation: {correlation:.3f}")
    
    return {'per_target': results, 'correlation': correlation}


def hodge_filter_preferences(
    preferences: List[Tuple[int, int, float]],
    n_items: int,
    h1_threshold: float = 0.0,
    return_info: bool = False
) -> Union[List[Tuple[int, int, float]], Tuple[List[Tuple[int, int, float]], Dict]]:
    """
    Apply Hodge filtering to remove or reduce cyclic (H¹) component from preferences.
    
    Supports threshold-based partial filtering:
    - If h1_threshold = 0.0: Remove ALL harmonic component (H¹ → 0)
    - If h1_threshold > 0.0: Only reduce H¹ to the threshold level
    
    The filtering blends gradient and harmonic components:
        filtered = gradient + α * harmonic
    where α = min(1, threshold / current_h1) if threshold > 0, else α = 0
    
    Args:
        preferences: List of (item_a, item_b, prob_a_wins)
        n_items: Total number of items
        h1_threshold: Maximum allowed H¹ after filtering (0 = remove all)
        return_info: If True, also return filtering metadata
        
    Returns:
        filtered_preferences: Preferences with H¹ ≤ h1_threshold
        info (optional): Dict with h1_before, h1_after, alpha used
    """
    # Compute Hodge decomposition
    h1_before, decomposition = compute_h1_from_preferences(preferences, n_items)
    
    # Determine filtering strength (α)
    if h1_threshold <= 0 or h1_before <= 1e-10:
        # Full filtering or nothing to filter
        alpha = 0.0 if h1_threshold <= 0 else 1.0
    elif h1_before <= h1_threshold:
        # Already below threshold, no filtering needed
        alpha = 1.0
    else:
        # Partial filtering: keep only enough harmonic to reach threshold
        # H¹_new = α * H¹_old, so α = threshold / H¹_old
        alpha = h1_threshold / h1_before
    
    # Build edge map for reconstruction
    edge_dict = {}
    for i, j, prob in preferences:
        if i == j:
            continue
        if i < j:
            edge = (i, j)
            weight = np.log(prob / (1 - prob + 1e-10) + 1e-10)
        else:
            edge = (j, i)
            weight = np.log((1 - prob) / (prob + 1e-10) + 1e-10)
        if edge not in edge_dict:
            edge_dict[edge] = []
        edge_dict[edge].append((i, j, prob, weight))
    
    edges = list(edge_dict.keys())
    potential = decomposition.potential
    harmonic = decomposition.harmonic
    
    # Create edge index map
    edge_to_idx = {e: idx for idx, e in enumerate(edges)}
    
    # Reconstruct filtered preferences
    filtered = []
    for i, j, orig_prob in preferences:
        if i == j:
            filtered.append((i, j, orig_prob))
            continue
            
        # Get gradient contribution from potential
        grad_diff = potential[i] - potential[j]
        
        # Get harmonic contribution (scaled by α)
        if i < j:
            edge = (i, j)
            sign = 1.0
        else:
            edge = (j, i)
            sign = -1.0
        
        if edge in edge_to_idx:
            h_contrib = sign * harmonic[edge_to_idx[edge]] * alpha
        else:
            h_contrib = 0.0
        
        # Total log-odds = gradient + α * harmonic
        total_diff = grad_diff + h_contrib
        prob = 1.0 / (1.0 + np.exp(-total_diff))
        prob = np.clip(prob, 0.01, 0.99)
        filtered.append((i, j, prob))
    
    if return_info:
        # Measure H¹ after filtering
        h1_after, _ = compute_h1_from_preferences(filtered, n_items)
        info = {
            'h1_before': h1_before,
            'h1_after': h1_after,
            'alpha': alpha,
            'threshold': h1_threshold
        }
        return filtered, info
    
    return filtered


def fine_grained_threshold_search(
    preferences: List[Tuple[int, int, float]],
    n_items: int,
    center: float = 0.8,
    width: float = 0.2,
    n_points: int = 11,
    n_folds: int = 5
) -> Tuple[float, Dict]:
    """
    Fine-grained search around a center threshold value.
    
    Args:
        preferences: All preference data
        n_items: Number of items
        center: Center of search range (e.g., 0.8)
        width: Half-width of search range (e.g., 0.2 → search [0.6, 1.0])
        n_points: Number of candidate points
        n_folds: Cross-validation folds
        
    Returns:
        optimal_threshold: Best threshold in neighborhood
        results: Dict with CV scores for each threshold
    """
    candidates = np.linspace(
        max(0.0, center - width), 
        min(1.5, center + width),  # Allow slightly > 1.0
        n_points
    ).tolist()
    
    return adaptive_threshold_search(preferences, n_items, n_folds, candidates)


def find_optimal_h1_threshold(
    train_preferences: List[Tuple[int, int, float]],
    val_preferences: List[Tuple[int, int, float]],
    n_items: int,
    candidate_thresholds: List[float] = None,
    metric: str = 'prediction_error'
) -> Tuple[float, Dict]:
    """
    Find optimal H¹ threshold via validation set performance.
    
    Strategy: Cross-validation on held-out preferences
    - For each threshold, filter training data and fit a Bradley-Terry model
    - Evaluate prediction error on validation preferences
    - Select threshold with best validation performance
    
    This balances:
    - Low threshold → Less cyclic bias but potential information loss
    - High threshold → More signal preserved but risk of reward hacking
    
    Args:
        train_preferences: Training preference data
        val_preferences: Held-out validation preferences
        n_items: Number of items
        candidate_thresholds: List of thresholds to try (default: 0.0 to 1.0)
        metric: 'prediction_error' or 'log_likelihood'
        
    Returns:
        optimal_threshold: Best threshold value
        results: Dict with scores for each threshold
    """
    if candidate_thresholds is None:
        candidate_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = {}
    
    for threshold in candidate_thresholds:
        # Filter training preferences
        filtered_train, info = hodge_filter_preferences(
            train_preferences, n_items, h1_threshold=threshold, return_info=True
        )
        
        # Fit Bradley-Terry model (extract potential from filtered data)
        _, decomp = compute_h1_from_preferences(filtered_train, n_items)
        potential = decomp.potential
        
        # Evaluate on validation set
        if metric == 'prediction_error':
            errors = []
            for i, j, true_prob in val_preferences:
                pred_diff = potential[i] - potential[j]
                pred_prob = 1.0 / (1.0 + np.exp(-pred_diff))
                errors.append((true_prob - pred_prob) ** 2)
            score = np.mean(errors)  # MSE (lower is better)
        else:  # log_likelihood
            ll = 0.0
            for i, j, true_prob in val_preferences:
                pred_diff = potential[i] - potential[j]
                pred_prob = np.clip(1.0 / (1.0 + np.exp(-pred_diff)), 1e-10, 1-1e-10)
                # Treat true_prob as soft label
                ll += true_prob * np.log(pred_prob) + (1 - true_prob) * np.log(1 - pred_prob)
            score = -ll / len(val_preferences)  # Negative LL (lower is better)
        
        results[threshold] = {
            'score': score,
            'h1_before': info['h1_before'],
            'h1_after': info['h1_after'],
            'alpha': info['alpha']
        }
    
    # Find optimal threshold (lowest score)
    optimal = min(results.keys(), key=lambda t: results[t]['score'])
    
    return optimal, results


def adaptive_threshold_search(
    preferences: List[Tuple[int, int, float]],
    n_items: int,
    n_folds: int = 5,
    candidates: List[float] = None
) -> Tuple[float, Dict]:
    """
    K-fold cross-validation to find optimal H¹ threshold.
    
    More robust than single train/val split.
    
    Args:
        preferences: All preference data
        n_items: Number of items
        n_folds: Number of CV folds
        candidates: Threshold candidates to evaluate
        
    Returns:
        optimal_threshold: Best threshold across folds
        cv_results: Per-fold and aggregate scores
    """
    if candidates is None:
        candidates = np.linspace(0, 1, 11).tolist()
    
    np.random.seed(42)
    indices = np.random.permutation(len(preferences))
    fold_size = len(preferences) // n_folds
    
    fold_results = {t: [] for t in candidates}
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else len(preferences)
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        train_prefs = [preferences[i] for i in train_indices]
        val_prefs = [preferences[i] for i in val_indices]
        
        _, results = find_optimal_h1_threshold(
            train_prefs, val_prefs, n_items, candidates
        )
        
        for t, data in results.items():
            fold_results[t].append(data['score'])
    
    # Aggregate across folds
    aggregate = {}
    for t, scores in fold_results.items():
        aggregate[t] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores
        }
    
    optimal = min(aggregate.keys(), key=lambda t: aggregate[t]['mean_score'])
    
    return optimal, aggregate


# ============================================================================
# Preference Graph Utilities
# ============================================================================

def build_preference_graph(
    preferences: List[Tuple[int, int, float]],
    n_items: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build adjacency and weight matrices from preferences.
    
    Returns:
        adj_matrix: (n_items, n_items) adjacency matrix
        weight_matrix: (n_items, n_items) average preference strengths
    """
    adj = np.zeros((n_items, n_items))
    weights = np.zeros((n_items, n_items))
    counts = np.zeros((n_items, n_items))
    
    for i, j, prob in preferences:
        adj[i, j] = 1
        adj[j, i] = 1
        weights[i, j] += (2 * prob - 1)  # Convert to [-1, 1]
        weights[j, i] += (1 - 2 * prob)
        counts[i, j] += 1
        counts[j, i] += 1
    
    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.where(counts > 0, weights / counts, 0)
    
    return adj, weights


def count_transitivity_violations(
    preferences: List[Tuple[int, int, float]],
    n_items: int,
    threshold: float = 0.5
) -> Tuple[int, int]:
    """
    Count transitivity violations (A > B > C but C > A).
    
    Returns:
        n_violations: Number of violating triplets
        n_triplets: Total triplets checked
    """
    _, weights = build_preference_graph(preferences, n_items)
    
    # For each triplet, check transitivity
    n_violations = 0
    n_triplets = 0
    
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            for k in range(n_items):
                if k == i or k == j:
                    continue
                
                # Check if i > j > k but k > i
                if (weights[i, j] > threshold and 
                    weights[j, k] > threshold and 
                    weights[k, i] > threshold):
                    n_violations += 1
                
                n_triplets += 1
    
    return n_violations, n_triplets


# ============================================================================
# Context-Conditional H¹ Analysis
# ============================================================================

@dataclass
class ContextualPreference:
    """Preference with context information."""
    context_id: int  # Hash or ID of the prompt/context
    item_a: int
    item_b: int
    preference: float


def compute_conditional_h1(
    preferences: List[ContextualPreference],
    n_items: int
) -> Tuple[float, float, Dict]:
    """
    Compute context-conditional H¹ to distinguish valid vs invalid cycles.
    
    Key insight: A cycle like rock > scissors > paper > rock is VALID if 
    conditioned on different contexts (opponent moves). It only becomes
    an INVALID transitivity violation if it occurs within the same context.
    
    Returns:
        marginal_h1: H¹ computed ignoring context (includes valid cycles)
        conditional_h1: H¹ computed within each context (only invalid cycles)
        breakdown: Per-context H¹ values
    """
    # Group preferences by context
    by_context = {}
    for p in preferences:
        if p.context_id not in by_context:
            by_context[p.context_id] = []
        by_context[p.context_id].append((p.item_a, p.item_b, p.preference))
    
    # Compute marginal H¹ (ignoring context)
    all_prefs = [(p.item_a, p.item_b, p.preference) for p in preferences]
    marginal_h1, _ = compute_h1_from_preferences(all_prefs, n_items)
    
    # Compute conditional H¹ (within each context)
    context_h1_values = {}
    weighted_h1 = 0.0
    total_weight = 0.0
    
    for ctx_id, ctx_prefs in by_context.items():
        if len(ctx_prefs) >= 3:  # Need at least 3 preferences for a cycle
            h1, _ = compute_h1_from_preferences(ctx_prefs, n_items)
            context_h1_values[ctx_id] = h1
            # Weight by number of preferences in context
            weighted_h1 += h1 * len(ctx_prefs)
            total_weight += len(ctx_prefs)
    
    conditional_h1 = weighted_h1 / total_weight if total_weight > 0 else 0.0
    
    breakdown = {
        'n_contexts': len(by_context),
        'per_context_h1': context_h1_values,
        'valid_cycle_contribution': marginal_h1 - conditional_h1,
        'invalid_cycle_contribution': conditional_h1
    }
    
    return marginal_h1, conditional_h1, breakdown


def filter_invalid_cycles_only(
    preferences: List[ContextualPreference],
    n_items: int,
    h1_threshold: float = 0.0
) -> List[ContextualPreference]:
    """
    Filter only invalid cyclic preferences (within-context cycles).
    
    Preserves valid context-dependent cycles (like rock-paper-scissors
    conditioned on opponent's expected move).
    
    This is the "smart" version of Hodge filtering that uses context.
    """
    # Group by context
    by_context = {}
    for p in preferences:
        if p.context_id not in by_context:
            by_context[p.context_id] = []
        by_context[p.context_id].append(p)
    
    filtered = []
    
    for ctx_id, ctx_prefs in by_context.items():
        # Convert to tuple format for hodge_filter_preferences
        raw = [(p.item_a, p.item_b, p.preference) for p in ctx_prefs]
        
        # Apply threshold-based filtering within this context
        filtered_raw = hodge_filter_preferences(raw, n_items, h1_threshold)
        
        # Convert back to ContextualPreference
        for (i, j, prob), orig in zip(filtered_raw, ctx_prefs):
            filtered.append(ContextualPreference(
                context_id=ctx_id,
                item_a=i,
                item_b=j,
                preference=prob
            ))
    
    return filtered


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Hodge Decomposition Utilities")
    print("=" * 60)
    
    # Test 1: Consistent preferences (H¹ ≈ 0)
    print("\nTest 1: Consistent preferences")
    prefs, h1 = inject_h1_controlled(n_items=10, h1_target=0.0)
    print(f"  Target H¹: 0.0, Measured: {h1:.4f}")
    
    # Test 2: Moderate cycling
    print("\nTest 2: Moderate cycling")
    prefs, h1 = inject_h1_controlled(n_items=10, h1_target=0.5)
    print(f"  Target H¹: 0.5, Measured: {h1:.4f}")
    
    # Test 3: Strong cycling
    print("\nTest 3: Strong cycling")
    prefs, h1 = inject_h1_controlled(n_items=10, h1_target=1.0)
    print(f"  Target H¹: 1.0, Measured: {h1:.4f}")
    
    # Test 4: Verify injection across range
    print("\nTest 4: Verify H¹ injection")
    targets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = verify_h1_injection(targets, n_items=15, n_trials=3)
    
    # Test 5: Hodge filtering (full)
    print("\nTest 5: Hodge filtering (full removal)")
    prefs, h1_before = inject_h1_controlled(n_items=10, h1_target=0.8)
    filtered = hodge_filter_preferences(prefs, n_items=10, h1_threshold=0.0)
    h1_after, _ = compute_h1_from_preferences(filtered, n_items=10)
    print(f"  Before filtering: H¹ = {h1_before:.4f}")
    print(f"  After filtering:  H¹ = {h1_after:.4f}")
    
    # Test 6: Threshold-based partial filtering
    print("\nTest 6: Threshold-based partial filtering")
    prefs, h1_before = inject_h1_controlled(n_items=15, h1_target=1.0)
    print(f"  Original H¹: {h1_before:.4f}")
    for threshold in [0.0, 0.25, 0.5, 0.75, 1.0]:
        filtered, info = hodge_filter_preferences(
            prefs, n_items=15, h1_threshold=threshold, return_info=True
        )
        print(f"  Threshold {threshold:.2f} → H¹ = {info['h1_after']:.4f} (α = {info['alpha']:.3f})")
    
    # Test 7: Optimal threshold search
    print("\nTest 7: Cross-validation threshold search")
    # Generate preferences with moderate H¹
    prefs, _ = inject_h1_controlled(n_items=20, h1_target=0.6)
    optimal, cv_results = adaptive_threshold_search(prefs, n_items=20, n_folds=3)
    print(f"  Optimal threshold: {optimal:.2f}")
    print(f"  Threshold → Mean CV Score:")
    for t in sorted(cv_results.keys()):
        print(f"    {t:.1f} → {cv_results[t]['mean_score']:.4f} ± {cv_results[t]['std_score']:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests complete!")
