"""
Physics configuration for safety gym environments.

Provides configurable difficulty presets and fine-grained control over
environment dynamics, hazards, and observability.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PhysicsConfig:
    """Configurable physics parameters for safety gym environments."""
    
    # === Grid/Space Configuration ===
    grid_size: int = 20
    
    # === Discrete Navigation Parameters ===
    hazard_density: float = 0.1
    hazard_clusters: bool = False
    visibility_radius: int = 10
    moving_hazards: float = 0.0
    
    # === Continuous Control Parameters ===
    dt: float = 0.1
    friction: float = 0.1
    max_velocity: float = 1.0
    obstacle_radius_mean: float = 0.3
    obstacle_radius_variance: float = 0.0
    wind: Tuple[float, float] = (0.0, 0.0)
    turbulence: float = 0.0
    
    # === Reward/Risk Parameters ===
    reward_noise: float = 0.0
    delayed_consequences: int = 0
    goal_reward: float = 10.0
    step_penalty: float = -0.01
    collision_penalty: float = -1.0
    
    # === Difficulty Presets ===
    
    @classmethod
    def trivial(cls) -> 'PhysicsConfig':
        """Trivial difficulty - almost impossible to fail."""
        return cls(
            grid_size=20,
            hazard_density=0.02,
            visibility_radius=20,
            friction=0.3,
            max_velocity=0.5,
            moving_hazards=0.0,
            turbulence=0.0,
        )
    
    @classmethod
    def easy(cls) -> 'PhysicsConfig':
        """Easy difficulty - good for initial testing."""
        return cls(
            grid_size=20,
            hazard_density=0.05,
            visibility_radius=15,
            friction=0.2,
            max_velocity=0.8,
            moving_hazards=0.0,
            turbulence=0.0,
        )
    
    @classmethod
    def medium(cls) -> 'PhysicsConfig':
        """Medium difficulty - balanced challenge."""
        return cls(
            grid_size=20,
            hazard_density=0.15,
            visibility_radius=8,
            friction=0.1,
            max_velocity=1.0,
            moving_hazards=0.0,
            turbulence=0.0,
        )
    
    @classmethod
    def hard(cls) -> 'PhysicsConfig':
        """Hard difficulty - requires careful navigation."""
        return cls(
            grid_size=20,
            hazard_density=0.25,
            visibility_radius=5,
            hazard_clusters=True,
            friction=0.08,
            max_velocity=1.2,
            moving_hazards=0.1,
            turbulence=0.05,
            obstacle_radius_variance=0.1,
        )
    
    @classmethod
    def nightmare(cls) -> 'PhysicsConfig':
        """Nightmare difficulty - extreme challenge."""
        return cls(
            grid_size=20,
            hazard_density=0.4,
            visibility_radius=3,
            hazard_clusters=True,
            friction=0.02,
            max_velocity=1.5,
            moving_hazards=0.3,
            turbulence=0.15,
            obstacle_radius_variance=0.2,
            reward_noise=0.1,
        )
    
    @classmethod
    def from_name(cls, name: str) -> 'PhysicsConfig':
        """Get preset by name."""
        presets = {
            'trivial': cls.trivial,
            'easy': cls.easy,
            'medium': cls.medium,
            'hard': cls.hard,
            'nightmare': cls.nightmare,
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Choose from: {list(presets.keys())}")
        return presets[name]()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'grid_size': self.grid_size,
            'hazard_density': self.hazard_density,
            'hazard_clusters': self.hazard_clusters,
            'visibility_radius': self.visibility_radius,
            'moving_hazards': self.moving_hazards,
            'dt': self.dt,
            'friction': self.friction,
            'max_velocity': self.max_velocity,
            'obstacle_radius_mean': self.obstacle_radius_mean,
            'obstacle_radius_variance': self.obstacle_radius_variance,
            'wind': self.wind,
            'turbulence': self.turbulence,
            'reward_noise': self.reward_noise,
            'delayed_consequences': self.delayed_consequences,
            'goal_reward': self.goal_reward,
            'step_penalty': self.step_penalty,
            'collision_penalty': self.collision_penalty,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PhysicsConfig':
        """Create from dictionary."""
        return cls(**d)
    
    def describe(self) -> str:
        """Human-readable description of difficulty."""
        difficulty_score = (
            self.hazard_density * 10 +
            (1.0 / max(self.visibility_radius, 1)) * 5 +
            (1.0 / max(self.friction, 0.01)) * 0.5 +
            self.moving_hazards * 5 +
            self.turbulence * 10
        )
        
        if difficulty_score < 1.0:
            level = "Trivial"
        elif difficulty_score < 2.0:
            level = "Easy"
        elif difficulty_score < 4.0:
            level = "Medium"
        elif difficulty_score < 7.0:
            level = "Hard"
        else:
            level = "Nightmare"
        
        return (
            f"{level} (score: {difficulty_score:.2f})\n"
            f"  Hazards: {self.hazard_density*100:.0f}% density, "
            f"{'clustered' if self.hazard_clusters else 'scattered'}, "
            f"{self.moving_hazards*100:.0f}% moving\n"
            f"  Visibility: {self.visibility_radius} cells\n"
            f"  Physics: friction={self.friction:.2f}, "
            f"max_vel={self.max_velocity:.1f}, "
            f"turbulence={self.turbulence:.2f}"
        )
