"""
Continuous 2D physics simulation for robotics navigation.

This module provides velocity-aware safety checking and predictive braking
for continuous control in 2D environments with obstacles.

Mirrors the Rust implementation in safety_gym_core/src/topology/continuous.rs
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Obstacle:
    """Circular obstacle in 2D space."""
    center: np.ndarray  # [x, y]
    radius: float
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside obstacle."""
        return np.linalg.norm(point - self.center) <= self.radius
    
    def distance_to(self, point: np.ndarray) -> float:
        """Distance from point to obstacle surface (0 if inside)."""
        return max(0.0, np.linalg.norm(point - self.center) - self.radius)


class ContinuousPhysicsSpace:
    """
    2D continuous control space with velocity-aware safety.
    
    Implements:
    - Velocity-based physics simulation
    - Stopping distance computation
    - Velocity-aware safety checking
    - Predictive braking
    
    Physics model:
        vel_{t+1} = (vel_t + action * dt) * (1 - friction)
        vel_{t+1} = clamp(vel_{t+1}, max_velocity)
        pos_{t+1} = pos_t + vel_{t+1} * dt
    """
    
    def __init__(
        self,
        bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        obstacles: List[Obstacle],
        dt: float = 0.1,
        friction: float = 0.1,
        max_velocity: float = 1.0,
    ):
        """
        Initialize continuous physics space.
        
        Args:
            bounds: ((x_min, x_max), (y_min, y_max))
            obstacles: List of Obstacle objects
            dt: Time step for physics simulation
            friction: Friction coefficient (0 to 1)
            max_velocity: Maximum velocity magnitude
        """
        self.bounds = bounds
        self.obstacles = obstacles
        self.dt = dt
        self.friction = friction
        self.max_velocity = max_velocity
    
    def in_bounds(self, pos: np.ndarray) -> bool:
        """Check if position is within bounds."""
        return (
            self.bounds[0][0] <= pos[0] <= self.bounds[0][1] and
            self.bounds[1][0] <= pos[1] <= self.bounds[1][1]
        )
    
    def collides(self, pos: np.ndarray) -> bool:
        """Check if position collides with any obstacle."""
        return any(obs.contains(pos) for obs in self.obstacles)
    
    def is_safe(self, pos: np.ndarray) -> bool:
        """Check if position is safe (in bounds and no collision)."""
        return self.in_bounds(pos) and not self.collides(pos)
    
    def distance_to_nearest_obstacle(self, pos: np.ndarray) -> float:
        """Compute distance to nearest obstacle surface."""
        if not self.obstacles:
            return float('inf')
        return min(obs.distance_to(pos) for obs in self.obstacles)
    
    def step(
        self,
        pos: np.ndarray,
        velocity: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate one physics step.
        
        Args:
            pos: Current position [x, y]
            velocity: Current velocity [vx, vy]
            action: Acceleration [ax, ay]
        
        Returns:
            (new_pos, new_vel): Updated position and velocity
        """
        # Apply action (acceleration)
        new_vel = velocity + action * self.dt
        
        # Apply friction
        new_vel = new_vel * (1.0 - self.friction)
        
        # Clamp velocity
        speed = np.linalg.norm(new_vel)
        if speed > self.max_velocity:
            new_vel = new_vel * (self.max_velocity / speed)
        
        # Update position
        new_pos = pos + new_vel * self.dt
        
        # Clamp to bounds
        new_pos = np.array([
            np.clip(new_pos[0], self.bounds[0][0], self.bounds[0][1]),
            np.clip(new_pos[1], self.bounds[1][0], self.bounds[1][1]),
        ])
        
        return new_pos, new_vel
    
    def stopping_distance(self, velocity: np.ndarray) -> float:
        """
        Compute stopping distance given current velocity.
        
        Uses geometric series sum for exponential decay:
        d = v * dt / friction
        
        Args:
            velocity: Current velocity [vx, vy]
        
        Returns:
            Distance traveled before coming to rest
        """
        speed = np.linalg.norm(velocity)
        
        if speed < 1e-6:
            return 0.0
        
        # Geometric series: sum_{i=0}^∞ v * (1-f)^i * dt = v * dt / f
        return speed * self.dt / max(self.friction, 1e-6)
    
    def is_safe_with_velocity(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        safety_buffer: float = 0.05,
    ) -> bool:
        """
        Check if position is safe considering current velocity.
        
        A position is safe with velocity if:
        1. The position itself is safe
        2. The stopping distance does not overlap with any obstacle
        
        Args:
            pos: Current position [x, y]
            vel: Current velocity [vx, vy]
            safety_buffer: Additional safety margin
        
        Returns:
            True if agent can safely stop before hitting obstacles
        """
        # Check current position
        if not self.is_safe(pos):
            return False
        
        # Compute stopping distance
        stop_dist = self.stopping_distance(vel)
        
        # Check clearance to obstacles
        obstacle_dist = self.distance_to_nearest_obstacle(pos)
        
        # Need stopping distance + buffer
        required_clearance = stop_dist + safety_buffer
        
        return obstacle_dist >= required_clearance


def compute_braking_action(
    velocity: np.ndarray,
    obstacle_dist: float,
    max_decel: float,
) -> np.ndarray:
    """
    Compute braking action to stop before reaching an obstacle.
    
    Uses kinematic equation v² = u² + 2as to determine required deceleration.
    
    Args:
        velocity: Current velocity [vx, vy]
        obstacle_dist: Distance to nearest obstacle
        max_decel: Maximum deceleration magnitude
    
    Returns:
        Acceleration vector that will bring agent to rest before obstacle
    """
    speed = np.linalg.norm(velocity)
    
    # Already stopped
    if speed < 1e-3:
        return np.zeros(2)
    
    # Obstacle very close - emergency brake
    if obstacle_dist < 1e-3:
        return -(velocity / speed) * max_decel
    
    # Compute required deceleration: a = v²/(2d)
    required_decel = speed ** 2 / (2.0 * obstacle_dist)
    
    # Clamp to maximum
    actual_decel = min(required_decel, max_decel)
    
    # Apply opposite to velocity
    return -(velocity / speed) * actual_decel


def compute_safe_action(
    desired_action: np.ndarray,
    velocity: np.ndarray,
    obstacle_dist: float,
    stopping_dist: float,
    max_decel: float,
    safety_scale: float,
) -> np.ndarray:
    """
    Compute safe action that respects velocity-aware safety constraints.
    
    If approaching obstacles too fast, applies braking.
    Otherwise, scales desired action by safety metric.
    
    Args:
        desired_action: The action the policy wants to take
        velocity: Current velocity
        obstacle_dist: Distance to nearest obstacle
        stopping_dist: Distance required to stop
        max_decel: Maximum deceleration magnitude
        safety_scale: Metric-based safety scaling (0 to 1)
    
    Returns:
        Safe action (either braking or scaled desired action)
    """
    # Check if we need to brake
    safety_margin = 1.5  # Need 1.5x stopping distance
    required_clearance = stopping_dist * safety_margin
    
    if obstacle_dist < required_clearance:
        # Emergency braking mode
        return compute_braking_action(velocity, obstacle_dist, max_decel)
    else:
        # Normal mode: scale desired action
        return desired_action * safety_scale


def step_with_collision(
    pos: np.ndarray,
    vel: np.ndarray,
    action: np.ndarray,
    obstacles: List[Obstacle],
    dt: float = 0.1,
    friction: float = 0.1,
    max_velocity: float = 1.0,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 1), (0, 1)),
    restitution: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Step with proper collision response.
    
    Args:
        pos: Current position [x, y]
        vel: Current velocity [vx, vy]
        action: Acceleration [ax, ay]
        obstacles: List of obstacles
        dt: Time step
        friction: Friction coefficient
        max_velocity: Maximum velocity magnitude
        bounds: ((x_min, x_max), (y_min, y_max))
        restitution: Energy retained after collision (0-1)
    
    Returns:
        (new_pos, new_vel, collided): Updated state and collision flag
    """
    # Apply action (acceleration)
    new_vel = vel + action * dt
    
    # Apply friction
    new_vel = new_vel * (1.0 - friction)
    
    # Clamp velocity
    speed = np.linalg.norm(new_vel)
    if speed > max_velocity:
        new_vel = new_vel * (max_velocity / speed)
    
    # Update position
    new_pos = pos + new_vel * dt
    
    # Check for obstacle collisions
    collided = False
    for obs in obstacles:
        dist_to_center = np.linalg.norm(new_pos - obs.center)
        if dist_to_center < obs.radius:
            collided = True
            
            # Compute collision normal (outward from obstacle center)
            normal = (new_pos - obs.center)
            normal_len = np.linalg.norm(normal)
            if normal_len > 1e-6:
                normal = normal / normal_len
            else:
                normal = np.array([1.0, 0.0])  # Fallback
            
            # Push position out to surface
            new_pos = obs.center + normal * (obs.radius + 0.01)
            
            # Reflect velocity with energy loss
            dot = np.dot(new_vel, normal)
            if dot < 0:  # Only reflect if moving into obstacle
                new_vel = (new_vel - 2.0 * dot * normal) * restitution
            
            break  # Handle one collision per step
    
    # Clamp to bounds
    new_pos = np.array([
        np.clip(new_pos[0], bounds[0][0], bounds[0][1]),
        np.clip(new_pos[1], bounds[1][0], bounds[1][1]),
    ])
    
    # Handle boundary collisions
    if new_pos[0] == bounds[0][0] or new_pos[0] == bounds[0][1]:
        new_vel[0] *= -restitution
        collided = True
    if new_pos[1] == bounds[1][0] or new_pos[1] == bounds[1][1]:
        new_vel[1] *= -restitution
        collided = True
    
    return new_pos, new_vel, collided


# Example usage and testing
if __name__ == "__main__":
    # Create test environment
    space = ContinuousPhysicsSpace(
        bounds=((-10.0, 10.0), (-10.0, 10.0)),
        obstacles=[
            Obstacle(center=np.array([5.0, 0.0]), radius=1.0),
        ],
        dt=0.1,
        friction=0.1,
        max_velocity=2.0,
    )
    
    # Test stopping distance
    velocity = np.array([1.0, 0.0])
    stop_dist = space.stopping_distance(velocity)
    print(f"Stopping distance for velocity {velocity}: {stop_dist:.3f}")
    assert abs(stop_dist - 1.0) < 0.01, "Stopping distance calculation incorrect"
    
    # Test velocity-aware safety
    pos = np.array([2.0, 0.0])
    vel_safe = np.array([0.5, 0.0])
    vel_unsafe = np.array([1.5, 0.0])
    
    print(f"\nPosition {pos}:")
    print(f"  Safe with velocity {vel_safe}: {space.is_safe_with_velocity(pos, vel_safe)}")
    print(f"  Safe with velocity {vel_unsafe}: {space.is_safe_with_velocity(pos, vel_unsafe)}")
    
    # Test braking
    brake = compute_braking_action(vel_unsafe, 0.5, 5.0)
    print(f"\nBraking action for velocity {vel_unsafe}: {brake}")
    assert brake[0] < 0, "Braking should oppose velocity"
    
    print("\n✅ All tests passed!")
