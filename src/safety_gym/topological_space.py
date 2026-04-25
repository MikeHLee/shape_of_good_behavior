"""
Abstract base class for topological spaces in decision-making.

Key insight: We don't need full manifold structure, just:
1. Distance metric (for neighborhoods)
2. Embedding function (for Hodge decomposition)
3. Boundary detection (for black holes)
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class TopologicalSpace(ABC):
    """
    Abstract base class for any space with topological structure.
    
    This enables sheaf-theoretic safety analysis on arbitrary decision spaces,
    not just text embeddings.
    
    Attributes:
        black_hole_regions: List of identified dangerous regions
        topology_data: Database of (state, embedding, risk) tuples
    """
    
    def __init__(self):
        self.black_hole_regions: List[Dict[str, Any]] = []
        self.topology_data: Dict[str, List] = {
            'states': [],
            'embeddings': [],
            'harmonic_risk': [],
        }
    
    @abstractmethod
    def embed(self, state: Any) -> np.ndarray:
        """
        Embed state into a common vector space for topology computation.
        
        This is the key abstraction that allows us to apply sheaf theory
        to arbitrary spaces. The embedding should preserve local structure.
        
        Args:
            state: State in the original space (could be vector, tuple, image, etc.)
        
        Returns:
            embedding: Vector representation in R^d
        """
        pass
    
    @abstractmethod
    def distance(self, state1: Any, state2: Any) -> float:
        """
        Compute distance between two states.
        
        This defines the metric structure needed for:
        - Neighborhood computation
        - Geodesic distances
        - Riemannian metric construction
        
        Args:
            state1: First state
            state2: Second state
        
        Returns:
            distance: Non-negative real number
        """
        pass
    
    @abstractmethod
    def is_safe(self, state: Any) -> bool:
        """
        Check if state is in safe region (not a black hole).
        
        Black holes are regions where the Riemannian metric has singularities,
        representing states that should be avoided.
        
        Args:
            state: State to check
        
        Returns:
            safe: True if state is safe, False if in black hole region
        """
        pass
    
    def compute_harmonic_risk(self, state: Any, k: int = 5) -> float:
        """
        Estimate H¹ cohomology risk at this state.
        
        Uses KNN to interpolate risk from nearby states in the topology database.
        This is the same approach used in the text embedding experiments.
        
        Args:
            state: State to evaluate
            k: Number of nearest neighbors
        
        Returns:
            risk: Estimated harmonic risk (0 = safe, 1 = dangerous)
        """
        if len(self.topology_data['embeddings']) < k:
            return 0.5  # Unknown risk
        
        from sklearn.neighbors import NearestNeighbors
        
        embedding = self.embed(state)
        embeddings_array = np.array(self.topology_data['embeddings'])
        risks_array = np.array(self.topology_data['harmonic_risk'])
        
        # Find k nearest neighbors
        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn.fit(embeddings_array)
        distances, indices = knn.kneighbors([embedding])
        
        # Weighted average by inverse distance
        neighbor_risks = risks_array[indices[0]]
        weights = 1.0 / (distances[0] + 0.01)  # Add small epsilon
        weighted_risk = np.average(neighbor_risks, weights=weights)
        
        return float(weighted_risk)
    
    def compute_black_hole_proximity(self, state: Any) -> float:
        """
        Compute minimum distance to any black hole region.
        
        This measures how close the state is to dangerous regions.
        Used for constructing the Riemannian metric with singularities.
        
        Args:
            state: State to evaluate
        
        Returns:
            proximity: Minimum distance to nearest black hole (inf if none exist)
        """
        if not self.black_hole_regions:
            return float('inf')
        
        embedding = self.embed(state)
        min_distance = float('inf')
        
        for bh in self.black_hole_regions:
            dist = np.linalg.norm(embedding - bh['center'])
            effective_dist = max(0, dist - bh['radius'])
            min_distance = min(min_distance, effective_dist)
        
        return min_distance
    
    def add_topology_sample(self, state: Any, risk: float):
        """
        Add a state to the topology database.
        
        This builds up the sheaf structure over the state space.
        
        Args:
            state: State to add
            risk: Harmonic risk at this state (from H¹ computation)
        """
        embedding = self.embed(state)
        self.topology_data['states'].append(state)
        self.topology_data['embeddings'].append(embedding)
        self.topology_data['harmonic_risk'].append(risk)
    
    def identify_black_holes(
        self,
        failed_states: List[Any],
        eps: float = 0.5,
        min_samples: int = 5,
        safety_margin: float = 1.2,
    ):
        """
        Cluster failure states into black hole regions using DBSCAN.
        
        This is the key step that identifies dangerous regions in the state space
        where the policy should not venture.
        
        Args:
            failed_states: List of states that led to failures
            eps: DBSCAN epsilon parameter (neighborhood size)
            min_samples: Minimum samples for a cluster
            safety_margin: Multiply radius by this factor for safety
        """
        if len(failed_states) < min_samples:
            return
        
        from sklearn.cluster import DBSCAN
        
        # Embed all failure states
        failure_embeddings = np.array([self.embed(s) for s in failed_states])
        
        # Cluster in embedding space
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(failure_embeddings)
        
        # Create black hole region for each cluster
        for label in set(clustering.labels_):
            if label == -1:  # Noise points
                continue
            
            cluster_mask = clustering.labels_ == label
            cluster_embeddings = failure_embeddings[cluster_mask]
            
            # Compute center and radius
            center = np.mean(cluster_embeddings, axis=0)
            radius = np.max([np.linalg.norm(emb - center) for emb in cluster_embeddings])
            
            self.black_hole_regions.append({
                'center': center,
                'radius': radius * safety_margin,
                'strength': np.sum(cluster_mask) / len(failed_states),
                'label': int(label),
            })
    
    def compute_riemannian_metric(self, state: Any, alpha: float = 2.0) -> float:
        """
        Compute conformal factor for Riemannian metric at state.
        
        The metric is g(x) = φ(x)² · δ where φ(x) ≈ 1/dist(x, B)^α
        creates infinite "energy barriers" at black holes.
        
        Args:
            state: State to evaluate
            alpha: Strength of singularity (higher = stronger barrier)
        
        Returns:
            conformal_factor: φ(x) for the metric
        """
        proximity = self.compute_black_hole_proximity(state)
        
        if proximity == float('inf'):
            return 1.0  # Flat metric if no black holes
        
        # Conformal factor with singularity at black holes
        phi = 1.0 / (proximity + 0.01) ** alpha
        
        return phi
    
    def get_topology_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the topology database.
        
        Returns:
            summary: Dictionary with topology statistics
        """
        return {
            'n_samples': len(self.topology_data['states']),
            'n_black_holes': len(self.black_hole_regions),
            'mean_risk': np.mean(self.topology_data['harmonic_risk']) if self.topology_data['harmonic_risk'] else 0.0,
            'max_risk': np.max(self.topology_data['harmonic_risk']) if self.topology_data['harmonic_risk'] else 0.0,
            'embedding_dim': len(self.topology_data['embeddings'][0]) if self.topology_data['embeddings'] else 0,
        }
