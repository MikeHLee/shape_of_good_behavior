"""
Discrete navigation space for grid worlds and discrete tasks.

Handles discrete state spaces (grid positions, discrete choices, etc.)
"""

import numpy as np
from typing import Any, List, Tuple, Optional
from .topological_space import TopologicalSpace


class DiscreteNavigationSpace(TopologicalSpace):
    """
    Topological space for discrete navigation environments.
    
    State: Discrete position (e.g., (x, y) in grid world)
    Embedding: Learned position embeddings or one-hot encoding
    Black holes: Hazard positions (lava, pits, enemies)
    
    Examples:
        - Grid worlds
        - Mazes
        - Discrete navigation tasks
        - Board games
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, ...],
        hazard_positions: Optional[List[Tuple[int, ...]]] = None,
        embedding_dim: int = 64,
        embedding_type: str = "random",
        config: Optional['PhysicsConfig'] = None,
    ):
        """
        Initialize discrete navigation space.
        
        Args:
            grid_size: Size of grid (e.g., (10, 10) for 10x10 grid)
            hazard_positions: List of known hazard positions
            embedding_dim: Dimension of position embeddings
            embedding_type: "random", "onehot", or "learned"
            config: PhysicsConfig for difficulty/dynamics settings
        """
        super().__init__()
        
        # Import here to avoid circular dependency
        from .config import PhysicsConfig
        
        self.config = config or PhysicsConfig()
        self.grid_size = grid_size
        self.hazard_positions = set(hazard_positions or [])
        self.embedding_dim = embedding_dim
        self.embedding_type = embedding_type
        
        # Initialize position embeddings
        self._init_position_embeddings()
    
    def _init_position_embeddings(self):
        """
        Initialize position embeddings.
        
        Options:
        - random: Random Gaussian projection (fast, works well)
        - onehot: One-hot encoding (sparse, high-dimensional)
        - learned: Could be learned via VAE (future work)
        """
        self.position_embeddings = {}
        
        if self.embedding_type == "random":
            # Random Gaussian projection
            np.random.seed(42)
            for pos in self._enumerate_positions():
                self.position_embeddings[pos] = np.random.randn(self.embedding_dim)
                # Normalize
                self.position_embeddings[pos] /= np.linalg.norm(self.position_embeddings[pos])
        
        elif self.embedding_type == "onehot":
            # One-hot encoding
            total_positions = int(np.prod(self.grid_size))
            self.embedding_dim = total_positions
            
            for i, pos in enumerate(self._enumerate_positions()):
                embedding = np.zeros(total_positions)
                embedding[i] = 1.0
                self.position_embeddings[pos] = embedding
        
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
    
    def _enumerate_positions(self):
        """Generate all valid positions in the grid."""
        if len(self.grid_size) == 2:
            # 2D grid
            for x in range(self.grid_size[0]):
                for y in range(self.grid_size[1]):
                    yield (x, y)
        elif len(self.grid_size) == 3:
            # 3D grid
            for x in range(self.grid_size[0]):
                for y in range(self.grid_size[1]):
                    for z in range(self.grid_size[2]):
                        yield (x, y, z)
        else:
            raise ValueError("Only 2D and 3D grids supported")
    
    def embed(self, state: Tuple[int, ...]) -> np.ndarray:
        """
        Map discrete position to continuous embedding.
        
        Args:
            state: Discrete position (e.g., (x, y))
        
        Returns:
            embedding: Vector representation
        """
        # Convert to tuple if needed
        if isinstance(state, np.ndarray):
            state = tuple(state.astype(int))
        elif not isinstance(state, tuple):
            state = tuple(state)
        
        return self.position_embeddings.get(state, np.zeros(self.embedding_dim))
    
    def distance(self, state1: Tuple[int, ...], state2: Tuple[int, ...]) -> float:
        """
        Compute Manhattan distance in grid.
        
        Args:
            state1: First position
            state2: Second position
        
        Returns:
            distance: Manhattan distance
        """
        # Convert to tuples
        if isinstance(state1, np.ndarray):
            state1 = tuple(state1.astype(int))
        if isinstance(state2, np.ndarray):
            state2 = tuple(state2.astype(int))
        
        return float(sum(abs(a - b) for a, b in zip(state1, state2)))
    
    def is_safe(self, state: Tuple[int, ...]) -> bool:
        """
        Check if position is not a hazard.
        
        Args:
            state: Position to check
        
        Returns:
            safe: True if not a hazard
        """
        # Convert to tuple
        if isinstance(state, np.ndarray):
            state = tuple(state.astype(int))
        elif not isinstance(state, tuple):
            state = tuple(state)
        
        return state not in self.hazard_positions
    
    def add_hazard(self, position: Tuple[int, ...]):
        """
        Add a hazard position.
        
        Args:
            position: Position to mark as hazard
        """
        if isinstance(position, np.ndarray):
            position = tuple(position.astype(int))
        self.hazard_positions.add(position)
    
    def remove_hazard(self, position: Tuple[int, ...]):
        """
        Remove a hazard position.
        
        Args:
            position: Position to unmark as hazard
        """
        if isinstance(position, np.ndarray):
            position = tuple(position.astype(int))
        self.hazard_positions.discard(position)
    
    def get_neighbors(self, state: Tuple[int, ...], include_diagonals: bool = False) -> List[Tuple[int, ...]]:
        """
        Get neighboring positions.
        
        Args:
            state: Current position
            include_diagonals: Whether to include diagonal neighbors
        
        Returns:
            neighbors: List of valid neighboring positions
        """
        if isinstance(state, np.ndarray):
            state = tuple(state.astype(int))
        
        neighbors = []
        
        if len(state) == 2:
            x, y = state
            # Cardinal directions
            candidates = [
                (x - 1, y), (x + 1, y),
                (x, y - 1), (x, y + 1),
            ]
            
            if include_diagonals:
                # Diagonal directions
                candidates.extend([
                    (x - 1, y - 1), (x - 1, y + 1),
                    (x + 1, y - 1), (x + 1, y + 1),
                ])
            
            # Filter valid positions
            for pos in candidates:
                if (0 <= pos[0] < self.grid_size[0] and
                    0 <= pos[1] < self.grid_size[1]):
                    neighbors.append(pos)
        
        return neighbors
    
    def compute_path_risk(self, path: List[Tuple[int, ...]]) -> float:
        """
        Compute total risk along a path.
        
        Useful for path planning with topological safety.
        
        Args:
            path: List of positions forming a path
        
        Returns:
            total_risk: Sum of harmonic risks along path
        """
        total_risk = 0.0
        
        for pos in path:
            risk = self.compute_harmonic_risk(pos)
            total_risk += risk
        
        return total_risk
    
    def find_safe_path(
        self,
        start: Tuple[int, ...],
        goal: Tuple[int, ...],
        max_risk: float = 0.5,
    ) -> Optional[List[Tuple[int, ...]]]:
        """
        Find a safe path from start to goal using A* with risk constraints.
        
        Args:
            start: Start position
            goal: Goal position
            max_risk: Maximum acceptable risk per step
        
        Returns:
            path: List of positions, or None if no safe path exists
        """
        from heapq import heappush, heappop
        
        # A* search with risk constraints
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            _, current = heappop(frontier)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return list(reversed(path))
            
            for next_pos in self.get_neighbors(current):
                # Check if safe
                if not self.is_safe(next_pos):
                    continue
                
                # Check risk
                risk = self.compute_harmonic_risk(next_pos)
                if risk > max_risk:
                    continue
                
                # Compute cost (distance + risk penalty)
                new_cost = cost_so_far[current] + 1 + risk
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    # Priority = cost + heuristic
                    priority = new_cost + self.distance(next_pos, goal)
                    heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return None  # No safe path found
    
    def visualize_risk_heatmap(self, save_path: Optional[str] = None):
        """
        Visualize harmonic risk as a heatmap (2D only).
        
        Args:
            save_path: Path to save figure, or None to display
        """
        if len(self.grid_size) != 2:
            raise ValueError("Visualization only supported for 2D grids")
        
        import matplotlib.pyplot as plt
        
        # Compute risk for each position
        risk_grid = np.zeros(self.grid_size)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                risk_grid[x, y] = self.compute_harmonic_risk((x, y))
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(risk_grid.T, origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Harmonic Risk')
        plt.title('Topological Risk Heatmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Mark hazards
        if self.hazard_positions:
            hazard_x = [pos[0] for pos in self.hazard_positions]
            hazard_y = [pos[1] for pos in self.hazard_positions]
            plt.scatter(hazard_x, hazard_y, c='blue', marker='x', s=100, label='Hazards')
        
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
