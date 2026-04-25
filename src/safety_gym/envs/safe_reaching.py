"""
Safe Reaching Environment: 2D reaching task with obstacle avoidance.

Demonstrates:
- Continuous state/action space
- Geometric black holes (obstacles)
- Riemannian metric (distance penalized near obstacles)
"""

import gym
import numpy as np
from typing import List, Dict, Optional


class SafeReachingEnv(gym.Env):
    """
    2D reaching task with obstacle avoidance.
    
    The agent (point mass) must reach a goal position while avoiding
    circular obstacles. Physics includes acceleration, velocity, and damping.
    
    State: [x, y, vx, vy] (position and velocity)
    Action: [ax, ay] (acceleration)
    Reward: -distance_to_goal, -10 for collision, +10 for reaching goal
    
    This environment is useful for testing:
    - Continuous control with topological safety
    - Black hole detection (obstacles)
    - Riemannian metric construction (geometric barriers)
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        n_obstacles: int = 3,
        obstacle_radius: float = 0.15,
        goal_threshold: float = 0.05,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        """
        Initialize safe reaching environment.
        
        Args:
            n_obstacles: Number of circular obstacles
            obstacle_radius: Radius of each obstacle
            goal_threshold: Distance to goal for success
            max_steps: Maximum episode length
            seed: Random seed
        """
        super().__init__()
        
        self.n_obstacles = n_obstacles
        self.obstacle_radius = obstacle_radius
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        
        if seed is not None:
            np.random.seed(seed)
        
        # Obstacles (black holes)
        self.obstacles = self._generate_obstacles()
        
        # Goal position
        self.goal = np.array([0.9, 0.9])
        
        # Start position
        self.start = np.array([0.1, 0.1])
        
        # Gym spaces
        self.state_dim = 4  # (x, y, vx, vy)
        self.action_dim = 2  # (ax, ay)
        
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Physics parameters
        self.dt = 0.1
        self.damping = 0.9
        self.max_velocity = 2.0
        
        # Episode tracking
        self.state = None
        self.steps = 0
    
    def _generate_obstacles(self) -> List[Dict]:
        """Generate random obstacle positions."""
        obstacles = []
        
        for _ in range(self.n_obstacles):
            # Random position (avoid corners)
            center = np.random.uniform(0.3, 0.7, size=2)
            
            obstacles.append({
                'center': center,
                'radius': self.obstacle_radius,
            })
        
        return obstacles
    
    def reset(self):
        """Reset environment to start state."""
        self.state = np.concatenate([self.start, np.zeros(2)])  # [x, y, vx, vy]
        self.steps = 0
        return self.state.copy()
    
    def step(self, action: np.ndarray):
        """
        Take a step in the environment.
        
        Args:
            action: [ax, ay] acceleration
        
        Returns:
            obs: New state [x, y, vx, vy]
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        action = np.clip(action, -1, 1)
        
        # Extract state
        pos = self.state[:2]
        vel = self.state[2:]
        
        # Physics update
        vel = vel + action * self.dt  # Apply acceleration
        vel = vel * self.damping  # Apply damping
        vel = np.clip(vel, -self.max_velocity, self.max_velocity)  # Limit velocity
        pos = pos + vel * self.dt  # Update position
        
        # Clip to bounds [0, 1]
        pos = np.clip(pos, 0, 1)
        
        # Update state
        self.state = np.concatenate([pos, vel])
        self.steps += 1
        
        # Check collision with obstacles
        collision = False
        collision_obstacle = None
        for i, obs in enumerate(self.obstacles):
            dist = np.linalg.norm(pos - obs['center'])
            if dist < obs['radius']:
                collision = True
                collision_obstacle = i
                break
        
        # Compute reward
        info = {}
        if collision:
            reward = -10
            done = True
            info['failure'] = True
            info['reason'] = 'collision'
            info['obstacle_id'] = collision_obstacle
        elif np.linalg.norm(pos - self.goal) < self.goal_threshold:
            reward = 10
            done = True
            info['success'] = True
        elif self.steps >= self.max_steps:
            reward = -1
            done = True
            info['timeout'] = True
        else:
            # Negative distance to goal (encourages moving toward goal)
            reward = -np.linalg.norm(pos - self.goal)
            done = False
        
        # Add distance info
        info['distance_to_goal'] = np.linalg.norm(pos - self.goal)
        info['min_obstacle_distance'] = min(
            np.linalg.norm(pos - obs['center']) - obs['radius']
            for obs in self.obstacles
        )
        
        return self.state.copy(), reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: 'human' for matplotlib, 'rgb_array' for image
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        if mode == 'human':
            plt.figure(figsize=(6, 6))
            ax = plt.gca()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            
            # Draw obstacles
            for obs in self.obstacles:
                circle = Circle(obs['center'], obs['radius'], color='red', alpha=0.5)
                ax.add_patch(circle)
            
            # Draw goal
            goal_circle = Circle(self.goal, self.goal_threshold, color='green', alpha=0.3)
            ax.add_patch(goal_circle)
            plt.plot(self.goal[0], self.goal[1], 'g*', markersize=15, label='Goal')
            
            # Draw agent
            if self.state is not None:
                pos = self.state[:2]
                plt.plot(pos[0], pos[1], 'bo', markersize=10, label='Agent')
                
                # Draw velocity vector
                vel = self.state[2:]
                plt.arrow(pos[0], pos[1], vel[0] * 0.1, vel[1] * 0.1,
                         head_width=0.02, head_length=0.02, fc='blue', ec='blue')
            
            plt.legend()
            plt.title('Safe Reaching Environment')
            plt.grid(True, alpha=0.3)
            plt.show()
        
        elif mode == 'rgb_array':
            # Create RGB image
            img_size = 256
            img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
            
            def to_pixel(coord):
                """Convert [0, 1] coordinates to pixel coordinates."""
                return int(coord[0] * img_size), int((1 - coord[1]) * img_size)
            
            # Draw obstacles (red circles)
            for obs in self.obstacles:
                center_px = to_pixel(obs['center'])
                radius_px = int(obs['radius'] * img_size)
                y, x = np.ogrid[:img_size, :img_size]
                mask = (x - center_px[0])**2 + (y - center_px[1])**2 <= radius_px**2
                img[mask] = [255, 0, 0]
            
            # Draw goal (green circle)
            goal_px = to_pixel(self.goal)
            goal_radius_px = int(self.goal_threshold * img_size)
            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - goal_px[0])**2 + (y - goal_px[1])**2 <= goal_radius_px**2
            img[mask] = [0, 255, 0]
            
            # Draw agent (blue circle)
            if self.state is not None:
                pos = self.state[:2]
                agent_px = to_pixel(pos)
                agent_radius = 5
                y, x = np.ogrid[:img_size, :img_size]
                mask = (x - agent_px[0])**2 + (y - agent_px[1])**2 <= agent_radius**2
                img[mask] = [0, 0, 255]
            
            return img
    
    def get_obstacle_proximity(self, pos: np.ndarray) -> float:
        """
        Get minimum distance to any obstacle.
        
        Args:
            pos: Position [x, y]
        
        Returns:
            proximity: Minimum distance to obstacle surface
        """
        min_dist = float('inf')
        for obs in self.obstacles:
            dist = np.linalg.norm(pos - obs['center']) - obs['radius']
            min_dist = min(min_dist, dist)
        return min_dist
    
    def compute_riemannian_metric_factor(self, pos: np.ndarray, alpha: float = 2.0) -> float:
        """
        Compute conformal factor for Riemannian metric.
        
        Creates infinite energy barriers at obstacles.
        
        Args:
            pos: Position [x, y]
            alpha: Strength of singularity
        
        Returns:
            factor: Conformal factor φ(x)
        """
        proximity = self.get_obstacle_proximity(pos)
        
        if proximity <= 0:
            return float('inf')  # Inside obstacle
        
        # φ(x) = 1 / proximity^α
        return 1.0 / (proximity ** alpha)
    
    def is_collision_free_path(self, start: np.ndarray, end: np.ndarray, n_samples: int = 20) -> bool:
        """
        Check if straight-line path is collision-free.
        
        Args:
            start: Start position [x, y]
            end: End position [x, y]
            n_samples: Number of points to check along path
        
        Returns:
            collision_free: True if path is safe
        """
        for t in np.linspace(0, 1, n_samples):
            pos = start + t * (end - start)
            
            for obs in self.obstacles:
                dist = np.linalg.norm(pos - obs['center'])
                if dist < obs['radius']:
                    return False
        
        return True
