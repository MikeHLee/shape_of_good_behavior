"""
Continuous control space for MuJoCo-style environments.

Handles state spaces that are already vectors (joint positions, velocities, etc.)
"""

import numpy as np
from typing import Any, List, Optional
from .topological_space import TopologicalSpace


class ContinuousControlSpace(TopologicalSpace):
    """
    Topological space for continuous control environments (MuJoCo, robotics).
    
    State: Vector in R^n (e.g., joint positions + velocities)
    Embedding: State itself, optionally normalized
    Black holes: States leading to falls, collisions, or constraint violations
    
    Examples:
        - HalfCheetah-v3: 17-dim state (joint angles + velocities)
        - Ant-v3: 111-dim state
        - Humanoid-v3: 376-dim state
        - Safety Gym environments: Variable dimension
    """
    
    def __init__(
        self,
        env_name: str,
        state_dim: int,
        normalize: bool = True,
        distance_metric: str = "euclidean",
        config: Optional['PhysicsConfig'] = None,
    ):
        """
        Initialize continuous control space.
        
        Args:
            env_name: Name of the environment (for logging)
            state_dim: Dimension of state vector
            normalize: Whether to normalize embeddings
            distance_metric: "euclidean" or "cosine"
            config: PhysicsConfig for difficulty/dynamics settings
        """
        super().__init__()
        
        # Import here to avoid circular dependency
        from .config import PhysicsConfig
        
        self.config = config or PhysicsConfig()
        self.env_name = env_name
        self.state_dim = state_dim
        self.normalize = normalize
        self.distance_metric = distance_metric
        
        # Statistics for normalization
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self._n_samples = 0
    
    def embed(self, state: np.ndarray) -> np.ndarray:
        """
        Embed state into vector space.
        
        For continuous control, the state is already a vector.
        We optionally normalize to unit sphere for better topology computation.
        
        Args:
            state: State vector (shape: [state_dim])
        
        Returns:
            embedding: Normalized state vector
        """
        state = np.asarray(state, dtype=np.float32)
        
        if self.normalize:
            # Normalize to unit sphere
            norm = np.linalg.norm(state)
            if norm > 1e-8:
                return state / norm
            else:
                return state
        else:
            return state
    
    def distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Compute distance between two states.
        
        Args:
            state1: First state vector
            state2: Second state vector
        
        Returns:
            distance: Euclidean or cosine distance
        """
        state1 = np.asarray(state1, dtype=np.float32)
        state2 = np.asarray(state2, dtype=np.float32)
        
        if self.distance_metric == "euclidean":
            return float(np.linalg.norm(state1 - state2))
        elif self.distance_metric == "cosine":
            # Cosine distance: 1 - cos(θ)
            dot = np.dot(state1, state2)
            norm1 = np.linalg.norm(state1)
            norm2 = np.linalg.norm(state2)
            if norm1 > 1e-8 and norm2 > 1e-8:
                cos_sim = dot / (norm1 * norm2)
                return float(1.0 - cos_sim)
            else:
                return 0.0
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def is_safe(self, state: np.ndarray) -> bool:
        """
        Check if state is far from black holes.
        
        Args:
            state: State vector to check
        
        Returns:
            safe: True if state is safe
        """
        if not self.black_hole_regions:
            return True
        
        embedding = self.embed(state)
        
        for bh in self.black_hole_regions:
            dist = np.linalg.norm(embedding - bh['center'])
            if dist < bh['radius']:
                return False
        
        return True
    
    def update_normalization_stats(self, state: np.ndarray):
        """
        Update running mean and std for normalization.
        
        Uses Welford's online algorithm for numerical stability.
        
        Args:
            state: New state observation
        """
        state = np.asarray(state, dtype=np.float32)
        self._n_samples += 1
        
        delta = state - self.state_mean
        self.state_mean += delta / self._n_samples
        delta2 = state - self.state_mean
        self.state_std = np.sqrt(
            (self.state_std ** 2 * (self._n_samples - 1) + delta * delta2) / self._n_samples
        )
    
    def identify_black_holes_from_trajectories(
        self,
        failed_trajectories: List[dict],
        window_size: int = 10,
        **kwargs
    ):
        """
        Identify black holes from failed trajectories.
        
        Takes the last `window_size` states before failure as dangerous states.
        
        Args:
            failed_trajectories: List of dicts with 'states' and 'done' keys
            window_size: Number of states before failure to consider
            **kwargs: Additional arguments for identify_black_holes
        """
        failed_states = []
        
        for traj in failed_trajectories:
            states = traj.get('states', [])
            if len(states) >= window_size:
                # Take last window_size states before failure
                failed_states.extend(states[-window_size:])
            else:
                # Take all states if trajectory is short
                failed_states.extend(states)
        
        if failed_states:
            self.identify_black_holes(failed_states, **kwargs)
    
    def compute_constraint_violation_risk(
        self,
        state: np.ndarray,
        constraint_fn: Optional[callable] = None,
    ) -> float:
        """
        Compute risk of constraint violation at state.
        
        Useful for Safety Gym environments with explicit constraints.
        
        Args:
            state: State to evaluate
            constraint_fn: Function that returns constraint value (0 = safe, >0 = violation)
        
        Returns:
            risk: Estimated constraint violation risk
        """
        if constraint_fn is None:
            # Use harmonic risk as proxy
            return self.compute_harmonic_risk(state)
        
        # Combine constraint value with topological risk
        constraint_value = constraint_fn(state)
        harmonic_risk = self.compute_harmonic_risk(state)
        
        # Weighted combination
        return 0.5 * min(1.0, constraint_value) + 0.5 * harmonic_risk
    
    def get_safe_action_mask(
        self,
        current_state: np.ndarray,
        candidate_actions: np.ndarray,
        dynamics_fn: callable,
        risk_threshold: float = 0.7,
    ) -> np.ndarray:
        """
        Get boolean mask of safe actions.
        
        Predicts next states and filters actions that lead to high-risk regions.
        
        Args:
            current_state: Current state
            candidate_actions: Array of candidate actions [n_actions, action_dim]
            dynamics_fn: Function that predicts next_state = f(state, action)
            risk_threshold: Maximum acceptable risk
        
        Returns:
            mask: Boolean array [n_actions] where True = safe
        """
        n_actions = len(candidate_actions)
        mask = np.ones(n_actions, dtype=bool)
        
        for i, action in enumerate(candidate_actions):
            # Predict next state
            next_state = dynamics_fn(current_state, action)
            
            # Check risk
            risk = self.compute_harmonic_risk(next_state)
            is_safe = self.is_safe(next_state)
            
            if risk > risk_threshold or not is_safe:
                mask[i] = False
        
        return mask
