"""
Safe Navigation Environment: Grid world with hazards and goal.

Demonstrates:
- Discrete state space
- Clear black hole regions (lava/hazards)
- Condorcet cycles (multiple paths with conflicting preferences)
"""

import gym
import numpy as np
from typing import Tuple, List, Optional


class SafeNavigationEnv(gym.Env):
    """
    Grid world navigation with hazards.
    
    The agent must navigate from start (0, 0) to goal (size-1, size-1)
    while avoiding randomly placed hazards.
    
    State: (x, y) position in grid
    Action: 0=Up, 1=Down, 2=Left, 3=Right
    Reward: +10 for goal, -10 for hazard, -0.1 per step
    
    This environment is useful for testing:
    - Black hole detection (hazards are black holes)
    - Safe path planning with topological constraints
    - H¹ cohomology detection (conflicting path preferences)
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        size: int = 10,
        n_hazards: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Initialize safe navigation environment.
        
        Args:
            size: Grid size (size x size)
            n_hazards: Number of hazard positions
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.size = size
        self.n_hazards = n_hazards
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize grid
        self.grid = np.zeros((size, size))
        
        # Place hazards (avoid start and goal)
        self.hazards = []
        while len(self.hazards) < n_hazards:
            pos = (np.random.randint(1, size - 1), np.random.randint(1, size - 1))
            if pos not in self.hazards and pos != (0, 0) and pos != (size - 1, size - 1):
                self.hazards.append(pos)
                self.grid[pos] = -1
        
        # Goal and start
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.grid[self.goal] = 1
        
        # Agent position
        self.agent_pos = self.start
        
        # Gym spaces
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.int32
        )
        
        # Episode tracking
        self.steps = 0
        self.max_steps = size * size * 2
    
    def reset(self):
        """Reset environment to start position."""
        self.agent_pos = self.start
        self.steps = 0
        return np.array(self.agent_pos, dtype=np.int32)
    
    def step(self, action: int):
        """
        Take a step in the environment.
        
        Args:
            action: 0=Up, 1=Down, 2=Left, 3=Right
        
        Returns:
            obs: New position [x, y]
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        x, y = self.agent_pos
        
        # Move agent
        if action == 0:  # Up
            y = max(0, y - 1)
        elif action == 1:  # Down
            y = min(self.size - 1, y + 1)
        elif action == 2:  # Left
            x = max(0, x - 1)
        elif action == 3:  # Right
            x = min(self.size - 1, x + 1)
        
        self.agent_pos = (x, y)
        self.steps += 1
        
        # Compute reward
        info = {}
        if self.agent_pos in self.hazards:
            reward = -10
            done = True
            info['failure'] = True
            info['reason'] = 'hazard'
        elif self.agent_pos == self.goal:
            reward = 10
            done = True
            info['success'] = True
        elif self.steps >= self.max_steps:
            reward = -1
            done = True
            info['timeout'] = True
        else:
            reward = -0.1  # Step penalty
            done = False
        
        return np.array(self.agent_pos, dtype=np.int32), reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: 'human' for console, 'rgb_array' for image
        """
        if mode == 'human':
            # Console rendering
            for y in range(self.size):
                row = []
                for x in range(self.size):
                    pos = (x, y)
                    if pos == self.agent_pos:
                        row.append('A')  # Agent
                    elif pos == self.goal:
                        row.append('G')  # Goal
                    elif pos in self.hazards:
                        row.append('X')  # Hazard
                    else:
                        row.append('.')  # Empty
                print(' '.join(row))
            print()
        
        elif mode == 'rgb_array':
            # Image rendering
            img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255
            
            # Hazards (red)
            for hx, hy in self.hazards:
                img[hy, hx] = [255, 0, 0]
            
            # Goal (green)
            gx, gy = self.goal
            img[gy, gx] = [0, 255, 0]
            
            # Agent (blue)
            ax, ay = self.agent_pos
            img[ay, ax] = [0, 0, 255]
            
            return img
    
    def get_optimal_path_length(self) -> int:
        """
        Compute optimal path length (Manhattan distance).
        
        Returns:
            length: Optimal path length from start to goal
        """
        return abs(self.goal[0] - self.start[0]) + abs(self.goal[1] - self.start[1])
    
    def is_hazard(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a hazard."""
        return pos in self.hazards
    
    def get_safe_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get neighboring positions that are not hazards."""
        x, y = pos
        candidates = [
            (x - 1, y), (x + 1, y),
            (x, y - 1), (x, y + 1),
        ]
        
        safe_neighbors = []
        for nx, ny in candidates:
            if (0 <= nx < self.size and 
                0 <= ny < self.size and 
                (nx, ny) not in self.hazards):
                safe_neighbors.append((nx, ny))
        
        return safe_neighbors
