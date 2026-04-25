"""
CPO to Black Hole Initializer

Convert CPO (Constrained Policy Optimization) cost constraints to SGPO black hole regions.

Key Insight:
CPO defines safety through cost constraints: max_π J(π) s.t. E[C(s)] ≤ d
SGPO defines safety through geometric singularities: g(x) → ∞ as x → B

This module provides the bridge: CPO's high-cost regions become SGPO's black holes,
providing stronger safety guarantees (geometric barriers vs. expectation constraints).

Advantages of this approach:
1. Pre-initialize SGPO with known dangerous regions (faster learning)
2. Convert soft CPO constraints to hard geometric barriers
3. Allow closer approach to boundaries without violation (geometric avoidance)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class BlackHole:
    """A black hole region in state space."""
    center: np.ndarray              # Center of the black hole
    radius: float                   # Event horizon radius
    strength: float                 # Singularity strength (affects metric scaling)
    power: float = 2.0              # Power in 1/dist^power
    max_cost: float = 1.0           # Maximum cost in this region
    source: str = "cpo"             # Source of this black hole ("cpo", "learned", "manual")
    
    def __post_init__(self):
        self.center = np.asarray(self.center)
    
    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Check if a point is inside the event horizon."""
        dist = np.linalg.norm(point - self.center)
        return dist < (self.radius + margin)
    
    def distance_to_horizon(self, point: np.ndarray) -> float:
        """Compute distance to the event horizon (negative if inside)."""
        dist = np.linalg.norm(point - self.center)
        return dist - self.radius
    
    def metric_contribution(self, point: np.ndarray, epsilon: float = 1e-6) -> float:
        """
        Compute the metric contribution at a point.
        
        g(x) = strength / dist(x, center)^power
        
        Returns infinity if inside the black hole.
        """
        dist = np.linalg.norm(point - self.center)
        
        if dist < self.radius:
            return float('inf')
        
        safe_dist = max(dist - self.radius, epsilon)
        return self.strength / (safe_dist ** self.power)


@dataclass
class CPOConstraint:
    """A CPO-style constraint specification."""
    cost_function: Callable[[np.ndarray], np.ndarray]  # C(s) -> cost
    threshold: float                                    # d: cost limit
    name: str = "constraint"
    
    def evaluate(self, states: np.ndarray) -> np.ndarray:
        """Evaluate cost at given states."""
        return self.cost_function(states)
    
    def is_violated(self, states: np.ndarray) -> np.ndarray:
        """Check which states violate the constraint."""
        costs = self.evaluate(states)
        return costs > self.threshold


class CPOToBlackHoleInitializer:
    """
    Convert CPO cost constraints to SGPO black hole regions.
    
    Pipeline:
    1. Sample state space to evaluate costs
    2. Identify high-cost regions (cost > threshold)
    3. Cluster dangerous states to find black hole centers
    4. Initialize SGPO metric model with pre-known singularities
    
    Usage:
        initializer = CPOToBlackHoleInitializer(cost_threshold=0.5)
        black_holes = initializer.identify_black_holes(sample_states, costs)
        metric_model = initializer.initialize_metric(metric_model, black_holes)
    """
    
    def __init__(
        self,
        cost_threshold: float,
        horizon_scale: float = 1.0,
        singularity_power: float = 2.0,
        min_cluster_size: int = 3,
        cluster_eps: float = 0.5,
        use_schwarzschild: bool = True,
        schwarzschild_constant: float = 0.1,
    ):
        """
        Initialize the CPO to Black Hole converter.
        
        Args:
            cost_threshold: CPO's d parameter - cost above this is dangerous
            horizon_scale: Multiplier for event horizon radius (default: 1.0)
            singularity_power: α in 1/dist^α metric contribution (default: 2.0)
            min_cluster_size: Minimum points for DBSCAN cluster (default: 3)
            cluster_eps: DBSCAN neighborhood radius (default: 0.5)
            use_schwarzschild: Use Schwarzschild radius scaling (default: True)
            schwarzschild_constant: k in r_s = k * M^(1/d) (default: 0.1)
        """
        self.cost_threshold = cost_threshold
        self.horizon_scale = horizon_scale
        self.alpha = singularity_power
        self.min_cluster_size = min_cluster_size
        self.cluster_eps = cluster_eps
        self.use_schwarzschild = use_schwarzschild
        self.schwarzschild_k = schwarzschild_constant
    
    def compute_schwarzschild_radius(
        self,
        total_mass: float,
        embed_dim: int,
        n_points: int = 1,
    ) -> float:
        """
        Compute Schwarzschild-style event horizon radius from accumulated "mass".
        
        In general relativity: r_s = 2GM/c²
        
        In our embedding space:
            r_s = k * M^(1/d) * sqrt(n)
        
        where:
            M = total accumulated cost/danger (analogous to mass)
            d = embedding dimension
            n = number of dangerous points (density factor)
            k = schwarzschild_constant (tunable)
        
        This ensures:
        1. More dangerous regions → larger event horizons
        2. Proper scaling in high-dimensional spaces
        3. Density-aware radius (more violations = larger sphere)
        
        Args:
            total_mass: Sum of costs in the dangerous region
            embed_dim: Dimensionality of the embedding space
            n_points: Number of dangerous points in the cluster
            
        Returns:
            Event horizon radius
        """
        # Schwarzschild-style scaling: r ∝ M^(1/d)
        # In high-D, we need the 1/d exponent to prevent explosion
        mass_factor = total_mass ** (1.0 / embed_dim)
        
        # Density factor: more points = larger region
        density_factor = np.sqrt(n_points)
        
        # Final radius
        radius = self.schwarzschild_k * mass_factor * density_factor * self.horizon_scale
        
        return max(radius, 0.1)  # Minimum radius
    
    def identify_black_holes(
        self,
        states: np.ndarray,
        costs: np.ndarray,
        use_clustering: bool = True,
    ) -> List[BlackHole]:
        """
        Identify black hole centers from CPO cost function.
        
        States where cost exceeds threshold are considered "inside" black holes.
        We cluster these states to find black hole centers and radii.
        
        Args:
            states: State embeddings (n_samples, embed_dim)
            costs: Cost values at each state (n_samples,)
            use_clustering: Whether to cluster dangerous states (default: True)
            
        Returns:
            List of BlackHole objects representing forbidden regions
        """
        states = np.asarray(states)
        costs = np.asarray(costs).flatten()
        
        # Find dangerous states (cost above threshold)
        dangerous_mask = costs > self.cost_threshold
        dangerous_states = states[dangerous_mask]
        dangerous_costs = costs[dangerous_mask]
        
        if len(dangerous_states) == 0:
            return []
        
        black_holes = []
        
        if use_clustering and len(dangerous_states) >= self.min_cluster_size:
            # Cluster dangerous states to find distinct black hole regions
            clustering = DBSCAN(
                eps=self.cluster_eps,
                min_samples=self.min_cluster_size
            ).fit(dangerous_states)
            
            labels = clustering.labels_
            unique_labels = set(labels)
            
            for label in unique_labels:
                if label == -1:
                    # Noise points - treat each as a small black hole
                    noise_mask = labels == -1
                    noise_states = dangerous_states[noise_mask]
                    noise_costs = dangerous_costs[noise_mask]
                    
                    for i, (state, cost) in enumerate(zip(noise_states, noise_costs)):
                        black_holes.append(BlackHole(
                            center=state,
                            radius=0.1 * self.horizon_scale,  # Small radius for isolated points
                            strength=cost,
                            power=self.alpha,
                            max_cost=cost,
                            source="cpo_noise",
                        ))
                else:
                    # Cluster - create single black hole
                    cluster_mask = labels == label
                    cluster_states = dangerous_states[cluster_mask]
                    cluster_costs = dangerous_costs[cluster_mask]
                    
                    # Black hole center is centroid of cluster
                    center = cluster_states.mean(axis=0)
                    embed_dim = states.shape[1]
                    
                    if self.use_schwarzschild:
                        # SCHWARZSCHILD RADIUS: scales with total "mass" (accumulated danger)
                        total_mass = cluster_costs.sum()
                        n_points = len(cluster_costs)
                        radius = self.compute_schwarzschild_radius(total_mass, embed_dim, n_points)
                    else:
                        # Legacy: max distance from center
                        distances = np.linalg.norm(cluster_states - center, axis=1)
                        radius = np.max(distances) * self.horizon_scale
                    
                    # Strength proportional to max cost in cluster
                    max_cost = cluster_costs.max()
                    total_mass = cluster_costs.sum()
                    
                    black_holes.append(BlackHole(
                        center=center,
                        radius=max(radius, 0.1),  # Minimum radius
                        strength=total_mass,  # Use total mass for strength (Schwarzschild style)
                        power=self.alpha,
                        max_cost=max_cost,
                        source="cpo_schwarzschild" if self.use_schwarzschild else "cpo_cluster",
                    ))
        else:
            # No clustering - treat all dangerous states as a single region
            # or create individual small black holes
            embed_dim = states.shape[1]
            
            if len(dangerous_states) < 10:
                # Few states - individual black holes with Schwarzschild radius
                for state, cost in zip(dangerous_states, dangerous_costs):
                    if self.use_schwarzschild:
                        radius = self.compute_schwarzschild_radius(cost, embed_dim, n_points=1)
                    else:
                        radius = 0.1 * self.horizon_scale
                    
                    black_holes.append(BlackHole(
                        center=state,
                        radius=radius,
                        strength=cost,
                        power=self.alpha,
                        max_cost=cost,
                        source="cpo_schwarzschild_individual" if self.use_schwarzschild else "cpo_individual",
                    ))
            else:
                # Many states - single large black hole with Schwarzschild radius
                center = dangerous_states.mean(axis=0)
                total_mass = dangerous_costs.sum()
                max_cost = dangerous_costs.max()
                n_points = len(dangerous_costs)
                
                if self.use_schwarzschild:
                    radius = self.compute_schwarzschild_radius(total_mass, embed_dim, n_points)
                else:
                    distances = np.linalg.norm(dangerous_states - center, axis=1)
                    radius = np.max(distances) * self.horizon_scale
                
                black_holes.append(BlackHole(
                    center=center,
                    radius=radius,
                    strength=total_mass,
                    power=self.alpha,
                    max_cost=max_cost,
                    source="cpo_schwarzschild_region" if self.use_schwarzschild else "cpo_region",
                ))
        
        return black_holes
    
    def identify_black_holes_from_constraints(
        self,
        constraints: List[CPOConstraint],
        state_samples: np.ndarray,
    ) -> List[BlackHole]:
        """
        Identify black holes from multiple CPO constraints.
        
        Args:
            constraints: List of CPOConstraint objects
            state_samples: States to sample (n_samples, embed_dim)
            
        Returns:
            List of BlackHole objects
        """
        all_black_holes = []
        
        for constraint in constraints:
            costs = constraint.evaluate(state_samples)
            
            # Use constraint-specific threshold
            old_threshold = self.cost_threshold
            self.cost_threshold = constraint.threshold
            
            black_holes = self.identify_black_holes(state_samples, costs)
            
            # Tag with constraint name
            for bh in black_holes:
                bh.source = f"cpo_{constraint.name}"
            
            all_black_holes.extend(black_holes)
            self.cost_threshold = old_threshold
        
        return all_black_holes
    
    def initialize_metric(
        self,
        metric_model: Any,
        black_holes: List[BlackHole],
    ) -> Any:
        """
        Initialize SGPO's metric model with pre-known black holes.
        
        Instead of learning singularities from scratch, we set them
        based on CPO's constraint boundaries.
        
        Args:
            metric_model: A metric model with add_singularity method
            black_holes: List of BlackHole objects
            
        Returns:
            The metric model with added singularities
        """
        for bh in black_holes:
            if hasattr(metric_model, 'add_singularity'):
                metric_model.add_singularity(
                    center=bh.center,
                    radius=bh.radius,
                    strength=bh.strength,
                    power=bh.power,
                )
            elif hasattr(metric_model, 'singularities'):
                # Direct list access
                metric_model.singularities.append({
                    "center": bh.center,
                    "radius": bh.radius,
                    "strength": bh.strength,
                    "power": bh.power,
                })
        
        return metric_model
    
    def merge_overlapping_black_holes(
        self,
        black_holes: List[BlackHole],
        overlap_threshold: float = 0.5,
    ) -> List[BlackHole]:
        """
        Merge black holes that significantly overlap.
        
        Args:
            black_holes: List of BlackHole objects
            overlap_threshold: Merge if overlap > this fraction (default: 0.5)
            
        Returns:
            List of merged BlackHole objects
        """
        if len(black_holes) <= 1:
            return black_holes
        
        merged = []
        used = set()
        
        for i, bh1 in enumerate(black_holes):
            if i in used:
                continue
            
            # Find all overlapping black holes
            group = [bh1]
            used.add(i)
            
            for j, bh2 in enumerate(black_holes):
                if j in used:
                    continue
                
                # Check overlap
                dist = np.linalg.norm(bh1.center - bh2.center)
                combined_radius = bh1.radius + bh2.radius
                
                if dist < combined_radius * overlap_threshold:
                    group.append(bh2)
                    used.add(j)
            
            if len(group) == 1:
                merged.append(bh1)
            else:
                # Merge group into single black hole
                centers = np.array([bh.center for bh in group])
                strengths = np.array([bh.strength for bh in group])
                radii = np.array([bh.radius for bh in group])
                
                # Weighted centroid by strength
                weights = strengths / strengths.sum()
                new_center = np.average(centers, axis=0, weights=weights)
                
                # Radius encompasses all original black holes
                new_radius = max(
                    np.linalg.norm(centers - new_center, axis=1) + radii
                )
                
                merged.append(BlackHole(
                    center=new_center,
                    radius=new_radius,
                    strength=strengths.max(),
                    power=group[0].power,
                    max_cost=max(bh.max_cost for bh in group),
                    source="merged",
                ))
        
        return merged
    
    def visualize_black_holes(
        self,
        black_holes: List[BlackHole],
        states: Optional[np.ndarray] = None,
        costs: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Visualize black hole regions (2D projection).
        
        Args:
            black_holes: List of BlackHole objects
            states: Optional state samples to plot
            costs: Optional costs for coloring states
            output_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
        except ImportError:
            print("Visualization requires matplotlib and sklearn")
            return
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Project to 2D if needed
        if black_holes:
            dim = len(black_holes[0].center)
            if dim > 2:
                # Use PCA to project to 2D
                all_centers = np.array([bh.center for bh in black_holes])
                if states is not None:
                    all_points = np.vstack([all_centers, states])
                else:
                    all_points = all_centers
                
                pca = PCA(n_components=2)
                pca.fit(all_points)
                
                centers_2d = pca.transform(all_centers)
                if states is not None:
                    states_2d = pca.transform(states)
            else:
                centers_2d = np.array([bh.center[:2] for bh in black_holes])
                if states is not None:
                    states_2d = states[:, :2]
        
        # Plot states
        if states is not None:
            if costs is not None:
                scatter = ax.scatter(
                    states_2d[:, 0], states_2d[:, 1],
                    c=costs, cmap='RdYlGn_r',
                    alpha=0.5, s=10,
                )
                plt.colorbar(scatter, label='Cost')
            else:
                ax.scatter(
                    states_2d[:, 0], states_2d[:, 1],
                    c='gray', alpha=0.3, s=10,
                )
        
        # Plot black holes
        for i, (bh, center_2d) in enumerate(zip(black_holes, centers_2d)):
            # Event horizon circle
            circle = plt.Circle(
                center_2d, bh.radius,
                fill=True, color='black', alpha=0.3,
                label='Event Horizon' if i == 0 else None
            )
            ax.add_patch(circle)
            
            # Center point
            ax.scatter(
                center_2d[0], center_2d[1],
                c='red', s=100, marker='x',
                label='Black Hole Center' if i == 0 else None
            )
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'Black Hole Regions ({len(black_holes)} identified)')
        ax.legend()
        ax.set_aspect('equal')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        
        plt.close()


def create_cost_function_from_examples(
    unsafe_states: np.ndarray,
    safe_states: np.ndarray,
    method: str = "distance",
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a cost function from example safe/unsafe states.
    
    Args:
        unsafe_states: Examples of dangerous states
        safe_states: Examples of safe states
        method: "distance" (to unsafe) or "classifier" (trained model)
        
    Returns:
        Cost function C(s) -> cost
    """
    if method == "distance":
        # Cost = negative distance to nearest unsafe state
        def cost_fn(states):
            states = np.atleast_2d(states)
            costs = []
            for state in states:
                dists = np.linalg.norm(unsafe_states - state, axis=1)
                min_dist = np.min(dists)
                # High cost near unsafe states
                cost = 1.0 / (min_dist + 0.1)
                costs.append(cost)
            return np.array(costs)
        return cost_fn
    
    elif method == "classifier":
        # Train a simple classifier
        from sklearn.svm import SVC
        
        X = np.vstack([unsafe_states, safe_states])
        y = np.concatenate([
            np.ones(len(unsafe_states)),
            np.zeros(len(safe_states))
        ])
        
        clf = SVC(probability=True)
        clf.fit(X, y)
        
        def cost_fn(states):
            states = np.atleast_2d(states)
            probs = clf.predict_proba(states)
            return probs[:, 1]  # Probability of unsafe class
        
        return cost_fn
    
    else:
        raise ValueError(f"Unknown method: {method}")
