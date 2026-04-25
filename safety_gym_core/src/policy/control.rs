//! Control-theoretic safety functions for continuous control.
//!
//! This module provides low-level control primitives for safe navigation,
//! including predictive braking and collision avoidance.

/// Compute braking action to stop before reaching an obstacle.
///
/// Uses kinematic equation v² = u² + 2as to determine required deceleration.
/// The braking action opposes the current velocity direction.
///
/// # Arguments
/// * `velocity` - Current velocity vector [vx, vy]
/// * `obstacle_dist` - Distance to nearest obstacle (must be positive)
/// * `max_decel` - Maximum deceleration magnitude (positive value)
///
/// # Returns
/// Acceleration vector that will bring the agent to rest before the obstacle.
/// Returns zero if already stopped or obstacle is very close.
///
/// # Physics
/// From v² = u² + 2as with v=0 (final velocity):
/// - Required deceleration: a = -u²/(2s)
/// - Direction: opposite to velocity
///
/// # Example
/// ```
/// # use safety_gym_core::policy::control::compute_braking_action;
/// // Agent moving at 2.0 m/s toward obstacle 1.0 m away
/// let velocity = [2.0, 0.0];
/// let obstacle_dist = 1.0;
/// let max_decel = 5.0;
///
/// let brake = compute_braking_action(&velocity, obstacle_dist, max_decel);
///
/// // Should produce negative acceleration (opposite to velocity)
/// assert!(brake[0] < 0.0);
/// ```
pub fn compute_braking_action(
    velocity: &[f32; 2],
    obstacle_dist: f32,
    max_decel: f32,
) -> [f32; 2] {
    let speed = (velocity[0].powi(2) + velocity[1].powi(2)).sqrt();
    
    // Already stopped or very slow
    if speed < 1e-3 {
        return [0.0, 0.0];
    }
    
    // Obstacle too close - emergency brake at max deceleration
    if obstacle_dist < 1e-3 {
        let scale = -max_decel / speed;
        return [velocity[0] * scale, velocity[1] * scale];
    }
    
    // Compute required deceleration: a = v²/(2d)
    let required_decel = speed.powi(2) / (2.0 * obstacle_dist);
    
    // Clamp to maximum deceleration
    let actual_decel = required_decel.min(max_decel);
    
    // Apply in opposite direction of velocity
    let scale = -actual_decel / speed;
    [velocity[0] * scale, velocity[1] * scale]
}

/// Compute safe action that respects velocity-aware safety constraints.
///
/// If the current state is unsafe considering velocity, this function
/// returns a braking action. Otherwise, it returns the desired action
/// scaled by safety metrics.
///
/// # Arguments
/// * `desired_action` - The action the policy wants to take
/// * `velocity` - Current velocity
/// * `obstacle_dist` - Distance to nearest obstacle
/// * `stopping_dist` - Distance required to stop given current velocity
/// * `max_decel` - Maximum deceleration magnitude
/// * `safety_scale` - Metric-based safety scaling factor (0.0 to 1.0)
///
/// # Returns
/// Safe action that either brakes or scales the desired action
///
/// # Example
/// ```
/// # use safety_gym_core::policy::control::compute_safe_action;
/// let desired_action = [1.0, 0.0];  // Want to accelerate forward
/// let velocity = [1.5, 0.0];        // Already moving fast
/// let obstacle_dist = 0.8;          // Obstacle close
/// let stopping_dist = 1.5;          // Need 1.5 units to stop
/// let max_decel = 5.0;
/// let safety_scale = 0.5;           // Moderate danger
///
/// let safe_action = compute_safe_action(
///     &desired_action,
///     &velocity,
///     obstacle_dist,
///     stopping_dist,
///     max_decel,
///     safety_scale,
/// );
///
/// // Should brake instead of accelerating
/// assert!(safe_action[0] < 0.0);
/// ```
pub fn compute_safe_action(
    desired_action: &[f32; 2],
    velocity: &[f32; 2],
    obstacle_dist: f32,
    stopping_dist: f32,
    max_decel: f32,
    safety_scale: f32,
) -> [f32; 2] {
    // Check if we need to brake (stopping distance exceeds clearance with margin)
    let safety_margin = 1.5; // Need 1.5x stopping distance for safety
    let required_clearance = stopping_dist * safety_margin;
    
    if obstacle_dist < required_clearance {
        // Emergency braking mode
        compute_braking_action(velocity, obstacle_dist, max_decel)
    } else {
        // Normal mode: scale desired action by safety metric
        [
            desired_action[0] * safety_scale,
            desired_action[1] * safety_scale,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_braking_action_zero_velocity() {
        let velocity = [0.0, 0.0];
        let brake = compute_braking_action(&velocity, 1.0, 5.0);
        
        assert_eq!(brake[0], 0.0);
        assert_eq!(brake[1], 0.0);
    }
    
    #[test]
    fn test_braking_action_direction() {
        let velocity = [2.0, 0.0];
        let brake = compute_braking_action(&velocity, 1.0, 5.0);
        
        // Braking should oppose velocity
        assert!(brake[0] < 0.0);
        assert_eq!(brake[1], 0.0);
    }
    
    #[test]
    fn test_braking_action_magnitude() {
        let velocity = [2.0, 0.0];
        let obstacle_dist = 1.0;
        let max_decel = 5.0;
        
        let brake = compute_braking_action(&velocity, obstacle_dist, max_decel);
        
        // Required deceleration: v²/(2d) = 4/(2*1) = 2.0
        // Should be clamped to max_decel = 5.0, so actual = 2.0
        let brake_mag = (brake[0].powi(2) + brake[1].powi(2)).sqrt();
        assert!((brake_mag - 2.0).abs() < 0.01);
    }
    
    #[test]
    fn test_braking_action_max_decel_limit() {
        let velocity = [5.0, 0.0];
        let obstacle_dist = 1.0;
        let max_decel = 2.0;
        
        let brake = compute_braking_action(&velocity, obstacle_dist, max_decel);
        
        // Required: 25/(2*1) = 12.5, but clamped to max_decel = 2.0
        let brake_mag = (brake[0].powi(2) + brake[1].powi(2)).sqrt();
        assert!((brake_mag - 2.0).abs() < 0.01);
    }
    
    #[test]
    fn test_braking_action_diagonal_velocity() {
        let velocity = [3.0, 4.0]; // Speed = 5.0
        let obstacle_dist = 2.0;
        let max_decel = 10.0;
        
        let brake = compute_braking_action(&velocity, obstacle_dist, max_decel);
        
        // Required: 25/(2*2) = 6.25
        let brake_mag = (brake[0].powi(2) + brake[1].powi(2)).sqrt();
        assert!((brake_mag - 6.25).abs() < 0.01);
        
        // Direction should oppose velocity
        let dot = brake[0] * velocity[0] + brake[1] * velocity[1];
        assert!(dot < 0.0); // Opposite directions
    }
    
    #[test]
    fn test_braking_action_very_close_obstacle() {
        let velocity = [1.0, 0.0];
        let obstacle_dist = 0.0001; // Very close
        let max_decel = 5.0;
        
        let brake = compute_braking_action(&velocity, obstacle_dist, max_decel);
        
        // Should apply maximum deceleration
        let brake_mag = (brake[0].powi(2) + brake[1].powi(2)).sqrt();
        assert!((brake_mag - 5.0).abs() < 0.01);
    }
    
    #[test]
    fn test_safe_action_needs_braking() {
        let desired_action = [1.0, 0.0]; // Wants to accelerate
        let velocity = [1.5, 0.0];
        let obstacle_dist = 0.8;
        let stopping_dist = 1.5;
        let max_decel = 5.0;
        let safety_scale = 0.5;
        
        let safe_action = compute_safe_action(
            &desired_action,
            &velocity,
            obstacle_dist,
            stopping_dist,
            max_decel,
            safety_scale,
        );
        
        // Should brake (negative) instead of accelerate (positive)
        assert!(safe_action[0] < 0.0);
    }
    
    #[test]
    fn test_safe_action_normal_mode() {
        let desired_action = [1.0, 0.5];
        let velocity = [0.5, 0.0];
        let obstacle_dist = 5.0;  // Far away
        let stopping_dist = 0.5;  // Small stopping distance
        let max_decel = 5.0;
        let safety_scale = 0.7;
        
        let safe_action = compute_safe_action(
            &desired_action,
            &velocity,
            obstacle_dist,
            stopping_dist,
            max_decel,
            safety_scale,
        );
        
        // Should scale desired action
        assert!((safe_action[0] - 0.7).abs() < 0.01);
        assert!((safe_action[1] - 0.35).abs() < 0.01);
    }
    
    #[test]
    fn test_safe_action_boundary_case() {
        let desired_action = [1.0, 0.0];
        let velocity = [1.0, 0.0];
        let stopping_dist = 1.0;
        let obstacle_dist = 1.5; // Exactly at safety margin (1.5x stopping)
        let max_decel = 5.0;
        let safety_scale = 0.8;
        
        let safe_action = compute_safe_action(
            &desired_action,
            &velocity,
            obstacle_dist,
            stopping_dist,
            max_decel,
            safety_scale,
        );
        
        // At boundary, should still allow scaled action
        assert!(safe_action[0] > 0.0);
    }
}
