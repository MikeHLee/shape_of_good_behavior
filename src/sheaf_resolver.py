"""
Sheaf Resolver: Multi-Perspective Consensus via Cohomology

This module implements Proposal (D): learning to assign consistent, perspective-invariant
rewards when feedback agrees, and appropriately variant rewards when perspectives differ.

Key Concepts:
- Perspectives: Different evaluators/personas/objectives (nodes in sheaf base space)
- Restriction Maps: How preferences translate between perspectives (learned or fixed)
- H^0 (Global Sections): The consensus policy that all perspectives can agree on
- H^1 (Cohomology): The "conflict energy" measuring disagreement
- Condorcet Cycles: Circular preferences (A > B > C > A) that prevent global ordering

The sheaf formalism provides:
1. A principled way to aggregate multiple objectives/evaluators
2. Detection of irreconcilable conflicts (non-zero H^1)
3. Identification of which perspectives are the outliers
4. Suggestions for resolving conflicts
"""

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
import warnings


@dataclass
class CondorcetCycleInfo:
    """Information about a detected Condorcet cycle."""
    perspectives: List[str]    # Perspectives involved in the cycle
    actions: List[int]         # Actions forming the cycle
    circulation: float         # Net preference flow around cycle
    
    def __str__(self) -> str:
        return f"Condorcet({' > '.join(self.perspectives)}, circ={self.circulation:.3f})"


@dataclass
class Perspective:
    """A node in the base space (e.g., a specific constraint or persona)."""
    name: str
    weight: float = 1.0
    preference_distribution: Optional[np.ndarray] = None  # Probability dist over actions
    
    # For learned restriction maps
    embedding: Optional[np.ndarray] = None  # Perspective embedding (for learning)
    trust_score: float = 1.0                # Dynamic trust based on consistency history

class RestrictionMap:
    """
    Learnable restriction map between perspectives.
    
    In sheaf theory, a restriction map ρ: F(U) → F(V) for V ⊂ U
    describes how data on a larger set restricts to a smaller set.
    
    For preference learning:
    - ρ maps one perspective's preferences to another's
    - Identity map = perspectives should agree exactly
    - Learned map = perspectives have systematic differences
    """
    
    def __init__(
        self,
        source_perspective: str,
        target_perspective: str,
        n_actions: int,
        learnable: bool = False,
    ):
        self.source = source_perspective
        self.target = target_perspective
        self.n_actions = n_actions
        self.learnable = learnable
        
        if learnable:
            # Learnable linear transformation
            self.matrix = np.eye(n_actions) + 0.01 * np.random.randn(n_actions, n_actions)
        else:
            # Identity restriction (perspectives should agree)
            self.matrix = np.eye(n_actions)
    
    def apply(self, preference: np.ndarray) -> np.ndarray:
        """Apply restriction map to a preference distribution."""
        restricted = self.matrix @ preference
        # Re-normalize to probability distribution
        restricted = np.clip(restricted, 0, None)
        if restricted.sum() > 0:
            restricted = restricted / restricted.sum()
        else:
            restricted = np.ones(self.n_actions) / self.n_actions
        return restricted
    
    def update(self, gradient: np.ndarray, learning_rate: float = 0.01):
        """Update restriction map parameters."""
        if self.learnable:
            self.matrix -= learning_rate * gradient


class SheafResolver:
    """
    Implements a Sheaf-Theoretic conflict resolution mechanism.
    
    Theory:
    - Base Space (X): A graph where nodes are 'perspectives' and edges represent 
      the requirement for consistency between them.
    - Sheaf (F): Associates a vector space (preferences/logits) to each node/edge.
    - Restriction Maps: Define how preferences translates between perspectives. 
      (Can be identity maps or learned transformations).
    - Cohomology (H^1): Measures global inconsistency (obstruction to consensus).
    
    Goal:
    - Compute H^0 (Global Sections): The consensus policy.
    - Compute H^1 (Cohomology): The "Conflict Energy".
    - Harmonic Decomposition: Resolve conflict by projecting out the curl component.
    - Detect Condorcet Cycles: Identify circular preferences that prevent ordering.
    
    Tensor RL Connection:
    - Multiple perspectives = multiple reward functions (multi-objective RL)
    - Consensus = Pareto-optimal compromise
    - H^1 = irreducible conflict that requires explicit value judgments
    """
    
    def __init__(
        self, 
        perspectives: List[Perspective], 
        action_space_size: int,
        use_learned_restrictions: bool = False,
    ):
        self.perspectives = perspectives
        self.n_actions = action_space_size
        self.n_perspectives = len(perspectives)
        self.use_learned_restrictions = use_learned_restrictions
        
        # Build a complete graph between perspectives for all-to-all consistency check
        self.adjacency = np.ones((self.n_perspectives, self.n_perspectives)) - np.eye(self.n_perspectives)
        
        # Initialize restriction maps (one per edge)
        self.restriction_maps: Dict[Tuple[str, str], RestrictionMap] = {}
        for i, p1 in enumerate(perspectives):
            for j, p2 in enumerate(perspectives):
                if i != j:
                    self.restriction_maps[(p1.name, p2.name)] = RestrictionMap(
                        p1.name, p2.name, action_space_size, learnable=use_learned_restrictions
                    )
        
        # History for learning restriction maps
        self.conflict_history: List[Dict] = []
        
        # Detected Condorcet cycles
        self.condorcet_cycles: List[CondorcetCycleInfo] = []
        
    def set_preferences(self, perspective_name: str, distribution: np.ndarray):
        """Update the preference distribution for a specific perspective."""
        for p in self.perspectives:
            if p.name == perspective_name:
                p.preference_distribution = distribution
                return
        raise ValueError(f"Perspective {perspective_name} not found")

    def compute_cohomology(self) -> Dict[str, Any]:
        """
        Compute the 0-th and 1-st cohomology to analyze conflict.
        
        We treat the preference distributions as a 0-cochain (data on nodes).
        We compute the coboundary (delta) to get a 1-cochain (data on edges).
        
        d0: C^0 -> C^1
        (df)(u, v) = ρ_{u→v}(f(u)) - f(v)  [with restriction maps]
        
        Returns:
            - consensus: The harmonic representative (best compromise)
            - obstruction: Magnitude of H^1 (conflict)
            - pairwise_conflicts: Specific edges with high disagreement
            - condorcet_cycles: Detected preference cycles
        """
        # 1. Construct 0-cochain (features on nodes)
        # Shape: (n_perspectives, n_actions)
        c0 = np.array([p.preference_distribution for p in self.perspectives])
        if np.any(c0 == None):
            raise ValueError("All perspectives must have preferences set")
            
        # 2. Compute 1-cochain (differences on edges) via Coboundary map d0
        # With restriction maps: d0(f)(u,v) = ρ_{u→v}(f(u)) - f(v)
        
        total_energy = 0.0
        pairwise_conflicts = []
        edge_residuals = {}  # For Hodge decomposition
        
        for i in range(self.n_perspectives):
            for j in range(self.n_perspectives):
                if i != j:
                    p1, p2 = self.perspectives[i], self.perspectives[j]
                    
                    # Apply restriction map
                    rmap = self.restriction_maps.get((p1.name, p2.name))
                    if rmap:
                        restricted_p1 = rmap.apply(c0[i])
                    else:
                        restricted_p1 = c0[i]
                    
                    # Coboundary: difference between restricted and target
                    residual = restricted_p1 - c0[j]
                    edge_residuals[(i, j)] = residual
        
        # Consensus computation (H^0 projection)
        # Use trust-weighted average
        weights = np.array([p.weight * p.trust_score for p in self.perspectives])
        norm_weights = weights / (weights.sum() + 1e-8)
        consensus = np.average(c0, axis=0, weights=norm_weights)
        
        # Calculate Obstruction (H^1 norm)
        residuals = c0 - consensus
        obstruction_energy = np.sum(weights[:, None] * (residuals ** 2))
        
        # Identify specific conflicts (edges with high difference)
        for i in range(self.n_perspectives):
            for j in range(i + 1, self.n_perspectives):
                # Use the edge residual if available
                if (i, j) in edge_residuals:
                    diff = np.linalg.norm(edge_residuals[(i, j)])
                else:
                    diff = np.linalg.norm(c0[i] - c0[j])
                    
                pairwise_conflicts.append({
                    "p1": self.perspectives[i].name,
                    "p2": self.perspectives[j].name,
                    "disagreement": float(diff)
                })
        
        # Sort conflicts
        pairwise_conflicts.sort(key=lambda x: x["disagreement"], reverse=True)
        
        # Detect Condorcet cycles in preferences
        self.condorcet_cycles = self._detect_condorcet_cycles(c0)
        
        return {
            "consensus_distribution": consensus,
            "obstruction_energy": float(obstruction_energy),
            "is_consistent": float(obstruction_energy) < 1e-3,
            "pairwise_conflicts": pairwise_conflicts,
            "per_perspective_divergence": {
                p.name: float(np.linalg.norm(r)) 
                for p, r in zip(self.perspectives, residuals)
            },
            "condorcet_cycles": [str(c) for c in self.condorcet_cycles],
            "has_condorcet_cycles": len(self.condorcet_cycles) > 0,
        }
    
    def _detect_condorcet_cycles(self, preferences: np.ndarray) -> List[CondorcetCycleInfo]:
        """
        Detect Condorcet cycles across perspectives.
        
        A Condorcet cycle occurs when:
        - Perspective A prefers action 1 > 2
        - Perspective B prefers action 2 > 3
        - Perspective C prefers action 3 > 1
        
        This makes it impossible to find a global ordering that satisfies everyone.
        """
        cycles = []
        
        # Build pairwise preference matrix for each action pair
        # Entry [a1, a2] = number of perspectives that prefer a1 over a2
        pref_matrix = np.zeros((self.n_actions, self.n_actions))
        
        for i, p in enumerate(self.perspectives):
            prefs = preferences[i]
            for a1 in range(self.n_actions):
                for a2 in range(self.n_actions):
                    if prefs[a1] > prefs[a2]:
                        pref_matrix[a1, a2] += p.weight
        
        # Look for cycles in the majority preference graph
        # A cycle exists if we can find a1 > a2 > a3 > a1 (by majority vote)
        n = self.n_actions
        total_weight = sum(p.weight for p in self.perspectives)
        
        for a1 in range(n):
            for a2 in range(n):
                if a1 == a2:
                    continue
                for a3 in range(n):
                    if a3 == a1 or a3 == a2:
                        continue
                    
                    # Check if a1 > a2 > a3 > a1 by majority
                    m12 = pref_matrix[a1, a2] > total_weight / 2
                    m23 = pref_matrix[a2, a3] > total_weight / 2
                    m31 = pref_matrix[a3, a1] > total_weight / 2
                    
                    if m12 and m23 and m31:
                        # Found a Condorcet cycle!
                        circulation = (
                            pref_matrix[a1, a2] + 
                            pref_matrix[a2, a3] + 
                            pref_matrix[a3, a1] -
                            pref_matrix[a2, a1] -
                            pref_matrix[a3, a2] -
                            pref_matrix[a1, a3]
                        )
                        
                        # Only record if not already found (avoid duplicates)
                        cycle_key = tuple(sorted([a1, a2, a3]))
                        existing = [c for c in cycles if tuple(sorted(c.actions)) == cycle_key]
                        
                        if not existing:
                            cycles.append(CondorcetCycleInfo(
                                perspectives=[f"majority"],
                                actions=[a1, a2, a3],
                                circulation=float(circulation),
                            ))
        
        return cycles

    def propose_resolution_path(self, obstruction_info: Dict) -> List[str]:
        """
        Suggest a path to resolve conflicts based on cohomology.
        
        Resolution strategies:
        1. If consistent: proceed with consensus
        2. If single outlier: down-weight or re-query that perspective
        3. If Condorcet cycle: need explicit value judgment (cannot resolve automatically)
        4. If learned restriction maps: suggest map updates
        """
        suggestions = []
        
        if obstruction_info["is_consistent"]:
            suggestions.append("✓ Perspectives are consistent. Proceed with consensus action.")
            return suggestions
        
        # Check for Condorcet cycles (fundamental irreconcilability)
        if obstruction_info.get("has_condorcet_cycles", False):
            suggestions.append(
                "⚠ CONDORCET CYCLES DETECTED: Preferences are fundamentally cyclic. "
                "No global ordering exists that satisfies all perspectives."
            )
            for cycle_str in obstruction_info.get("condorcet_cycles", []):
                suggestions.append(f"  Cycle: {cycle_str}")
            suggestions.append(
                "Resolution requires explicit value judgment: which perspective should dominate?"
            )
            return suggestions
            
        # Analyze the conflicts
        conflicts = obstruction_info["pairwise_conflicts"]
        top_conflict = conflicts[0]
        
        suggestions.append(
            f"Major conflict detected between '{top_conflict['p1']}' and '{top_conflict['p2']}' "
            f"(magnitude: {top_conflict['disagreement']:.4f})."
        )
        
        # Find the outlier
        divergences = obstruction_info["per_perspective_divergence"]
        sorted_divs = sorted(divergences.items(), key=lambda x: x[1], reverse=True)
        outlier_name = sorted_divs[0][0]
        outlier_div = sorted_divs[0][1]
        
        # Check if it's a single outlier or widespread disagreement
        if len(sorted_divs) > 1 and sorted_divs[0][1] > 2 * sorted_divs[1][1]:
            # Single dominant outlier
            suggestions.append(
                f"→ Perspective '{outlier_name}' is the primary outlier (div={outlier_div:.3f}). "
                f"Consider: (1) re-querying this perspective, (2) lowering its weight, "
                f"(3) learning a restriction map to align it."
            )
            
            # Update trust score
            for p in self.perspectives:
                if p.name == outlier_name:
                    p.trust_score *= 0.9  # Decrease trust
                    suggestions.append(f"  (Trust score for '{outlier_name}' reduced to {p.trust_score:.2f})")
        else:
            # Widespread disagreement
            suggestions.append(
                f"→ Multiple perspectives disagree significantly. "
                f"This may indicate genuinely different values, not noise."
            )
            suggestions.append(
                f"→ Consider: (1) conditioning policy on perspective, "
                f"(2) Pareto-optimal compromise, (3) human arbitration."
            )
        
        return suggestions
    
    def learn_restriction_maps(
        self,
        observed_transitions: List[Dict],
        learning_rate: float = 0.01,
    ) -> Dict[str, float]:
        """
        Learn restriction maps from observed transitions.
        
        When perspective A's preferences predict perspective B's behavior,
        we can learn the transformation between them.
        
        Args:
            observed_transitions: List of {"perspective": str, "action": int, "outcome": float}
            learning_rate: Learning rate for restriction map updates
            
        Returns:
            Dict of learning metrics
        """
        if not self.use_learned_restrictions:
            return {"skipped": True, "reason": "Learned restrictions disabled"}
        
        total_loss = 0.0
        n_updates = 0
        
        # Group transitions by perspective
        by_perspective = {}
        for t in observed_transitions:
            p_name = t.get("perspective")
            if p_name not in by_perspective:
                by_perspective[p_name] = []
            by_perspective[p_name].append(t)
        
        # For each pair of perspectives, compute restriction map gradient
        for p1_name, p1_trans in by_perspective.items():
            for p2_name, p2_trans in by_perspective.items():
                if p1_name == p2_name:
                    continue
                
                rmap = self.restriction_maps.get((p1_name, p2_name))
                if rmap is None:
                    continue
                
                # Compute gradient: how well does ρ(p1) predict p2?
                # This is simplified - full implementation would use proper gradients
                p1 = next((p for p in self.perspectives if p.name == p1_name), None)
                p2 = next((p for p in self.perspectives if p.name == p2_name), None)
                
                if p1 and p2 and p1.preference_distribution is not None and p2.preference_distribution is not None:
                    predicted = rmap.apply(p1.preference_distribution)
                    error = predicted - p2.preference_distribution
                    loss = np.sum(error ** 2)
                    
                    # Simple gradient descent
                    gradient = np.outer(error, p1.preference_distribution)
                    rmap.update(gradient, learning_rate)
                    
                    total_loss += loss
                    n_updates += 1
        
        return {
            "total_loss": float(total_loss),
            "n_updates": n_updates,
            "avg_loss": float(total_loss / max(n_updates, 1)),
        }
