//! Godot GDExtension bindings using godot-rust.
//!
//! This module provides Godot classes for using the safety gym
//! in Godot 4 game projects.

use godot::prelude::*;
use godot::classes::{Node3D, INode3D};

use crate::topology::discrete::DiscreteNavigationSpace;
use crate::topology::continuous::ContinuousControlSpace;
use crate::topology::TopologicalSpace;
use crate::policy::gpo::{SGPOConfig, SGPOPolicy};
use crate::policy::control::{compute_braking_action, compute_safe_action};

/// Entry point for the GDExtension.
struct SafetyGymExtension;

#[gdextension]
unsafe impl ExtensionLibrary for SafetyGymExtension {}

// ============================================================================
// SafetyAgent3D - A 3D agent with SGPO-based safety constraints
// ============================================================================

/// A 3D agent that uses SGPO for safe navigation.
///
/// This node can be attached to any Node3D and will apply
/// safety constraints to its movement based on the topological
/// structure of the environment.
#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct SafetyAgent3D {
    /// The continuous control space for the environment
    #[var]
    bounds_min: Vector3,
    #[var]
    bounds_max: Vector3,
    
    /// Current velocity
    velocity: Vector3,
    
    /// SGPO configuration
    #[var]
    alpha: f32,
    #[var]
    risk_threshold: f32,
    
    /// Internal state
    space: Option<ContinuousControlSpace>,
    
    #[base]
    base: Base<Node3D>,
}

#[godot_api]
impl INode3D for SafetyAgent3D {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            bounds_min: Vector3::new(-10.0, 0.0, -10.0),
            bounds_max: Vector3::new(10.0, 10.0, 10.0),
            velocity: Vector3::ZERO,
            alpha: 2.0,
            risk_threshold: 0.5,
            space: None,
            base,
        }
    }
    
    fn ready(&mut self) {
        // Initialize the continuous control space
        self.space = Some(ContinuousControlSpace::new(
            [
                [self.bounds_min.x, self.bounds_max.x],
                [self.bounds_min.z, self.bounds_max.z],
            ],
            vec![],
            None,
        ));
        
        godot_print!("SafetyAgent3D initialized with bounds: {:?} to {:?}",
            self.bounds_min, self.bounds_max);
    }
    
    fn physics_process(&mut self, delta: f64) {
        // Get current position
        let pos = self.base().get_position();
        
        // Apply any pending actions (would come from AI policy)
        // For now, just demonstrate the safety check
        
        if let Some(ref space) = self.space {
            let state = [pos.x, pos.z];
            
            // Check safety
            let is_safe = space.is_safe(&state);
            let risk = space.compute_harmonic_risk(&state, 5);
            
            if !is_safe || risk > self.risk_threshold {
                // Would apply safety constraint here
                godot_print!("Warning: Agent at unsafe position! Risk: {}", risk);
            }
        }
    }
}

#[godot_api]
impl SafetyAgent3D {
    /// Add an obstacle to the environment.
    #[func]
    fn add_obstacle(&mut self, position: Vector2, radius: f32) {
        if let Some(ref mut space) = self.space {
            space.add_obstacle([position.x, position.y], radius);
        }
    }
    
    /// Check if a position is safe.
    #[func]
    fn is_position_safe(&self, position: Vector2) -> bool {
        self.space.as_ref()
            .map(|s| s.is_safe(&[position.x, position.y]))
            .unwrap_or(true)
    }
    
    /// Get the harmonic risk at a position.
    #[func]
    fn get_risk_at(&self, position: Vector2) -> f32 {
        self.space.as_ref()
            .map(|s| s.compute_harmonic_risk(&[position.x, position.y], 5))
            .unwrap_or(0.5)
    }
    
    /// Get the Riemannian metric at a position.
    #[func]
    fn get_metric_at(&self, position: Vector2) -> f32 {
        self.space.as_ref()
            .map(|s| s.compute_riemannian_metric(&[position.x, position.y], self.alpha))
            .unwrap_or(1.0)
    }
    
    /// Get stopping distance for current velocity.
    #[func]
    fn get_stopping_distance(&self) -> f32 {
        self.space.as_ref()
            .map(|s| s.stopping_distance(&[self.velocity.x, self.velocity.z]))
            .unwrap_or(0.0)
    }
    
    /// Check if current position is safe considering velocity.
    #[func]
    fn is_safe_with_current_velocity(&self) -> bool {
        let pos = self.base().get_position();
        self.space.as_ref()
            .map(|s| s.is_safe_with_velocity(
                &[pos.x, pos.z],
                &[self.velocity.x, self.velocity.z]
            ))
            .unwrap_or(true)
    }
    
    /// Check if a position would be safe with given velocity.
    #[func]
    fn is_position_safe_with_velocity(&self, position: Vector2, velocity: Vector2) -> bool {
        self.space.as_ref()
            .map(|s| s.is_safe_with_velocity(
                &[position.x, position.y],
                &[velocity.x, velocity.y]
            ))
            .unwrap_or(true)
    }
    
    /// Apply a movement action with safety constraints.
    ///
    /// This method integrates velocity-aware safety checking and predictive braking:
    /// 1. Checks if current velocity allows safe stopping
    /// 2. Applies braking if approaching obstacles too fast
    /// 3. Otherwise scales action by Riemannian metric
    #[func]
    fn move_safely(&mut self, direction: Vector2, speed: f32) {
        let Some(ref space) = self.space else { return };
        
        let pos = self.base().get_position();
        let state = [pos.x, pos.z];
        let velocity = [self.velocity.x, self.velocity.z];
        
        // Compute desired action
        let desired_action = [direction.x * speed, direction.y * speed];
        
        // Get safety metrics
        let obstacle_dist = space.distance_to_nearest_obstacle(&state);
        let stopping_dist = space.stopping_distance(&velocity);
        let metric = space.compute_riemannian_metric(&state, self.alpha);
        let safety_scale = (1.0 / metric).clamp(0.1, 1.0);
        
        // Compute safe action (brakes if needed, otherwise scales)
        let max_decel = 5.0; // Maximum deceleration
        let safe_action = compute_safe_action(
            &desired_action,
            &velocity,
            obstacle_dist,
            stopping_dist,
            max_decel,
            safety_scale,
        );
        
        // Simulate physics step
        let (new_pos, new_vel) = space.step(&state, &velocity, &safe_action);
        
        // Update position and velocity
        self.base_mut().set_position(Vector3::new(new_pos[0], pos.y, new_pos[1]));
        self.velocity = Vector3::new(new_vel[0], 0.0, new_vel[1]);
    }
    
    /// Convert obstacles to black hole regions.
    #[func]
    fn finalize_obstacles(&mut self, safety_margin: f32) {
        if let Some(ref mut space) = self.space {
            space.obstacles_to_black_holes(safety_margin);
        }
    }
}

// ============================================================================
// GridAgent - A 2D grid-based agent for discrete navigation
// ============================================================================

/// A 2D grid agent for discrete navigation tasks.
#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct GridAgent {
    /// Grid dimensions
    #[var]
    grid_width: i32,
    #[var]
    grid_height: i32,
    
    /// Current grid position
    #[var]
    grid_x: i32,
    #[var]
    grid_y: i32,
    
    /// Cell size for world positioning
    #[var]
    cell_size: f32,
    
    /// Internal state
    space: Option<DiscreteNavigationSpace>,
    
    #[base]
    base: Base<Node3D>,
}

#[godot_api]
impl INode3D for GridAgent {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            grid_width: 20,
            grid_height: 20,
            grid_x: 0,
            grid_y: 0,
            cell_size: 1.0,
            space: None,
            base,
        }
    }
    
    fn ready(&mut self) {
        self.space = Some(DiscreteNavigationSpace::new(
            (self.grid_width as usize, self.grid_height as usize),
            vec![],
            64,
            42,
        ));
        
        self.update_world_position();
        
        godot_print!("GridAgent initialized: {}x{} grid", self.grid_width, self.grid_height);
    }
}

#[godot_api]
impl GridAgent {
    fn update_world_position(&mut self) {
        let world_x = self.grid_x as f32 * self.cell_size;
        let world_z = self.grid_y as f32 * self.cell_size;
        self.base_mut().set_position(Vector3::new(world_x, 0.0, world_z));
    }
    
    /// Add a hazard at a grid position.
    #[func]
    fn add_hazard(&mut self, x: i32, y: i32) {
        if let Some(ref mut space) = self.space {
            space.add_hazard((x, y));
        }
    }
    
    /// Remove a hazard from a grid position.
    #[func]
    fn remove_hazard(&mut self, x: i32, y: i32) {
        if let Some(ref mut space) = self.space {
            space.remove_hazard((x, y));
        }
    }
    
    /// Check if a grid position is safe.
    #[func]
    fn is_safe(&self, x: i32, y: i32) -> bool {
        self.space.as_ref()
            .map(|s| s.is_safe(&(x, y)))
            .unwrap_or(true)
    }
    
    /// Get harmonic risk at a grid position.
    #[func]
    fn get_risk(&self, x: i32, y: i32) -> f32 {
        self.space.as_ref()
            .map(|s| s.compute_harmonic_risk(&(x, y), 5))
            .unwrap_or(0.5)
    }
    
    /// Move in a cardinal direction (0=up, 1=right, 2=down, 3=left).
    #[func]
    fn move_direction(&mut self, direction: i32) -> bool {
        let (dx, dy) = match direction {
            0 => (0, 1),   // Up
            1 => (1, 0),   // Right
            2 => (0, -1),  // Down
            3 => (-1, 0),  // Left
            _ => return false,
        };
        
        let new_x = self.grid_x + dx;
        let new_y = self.grid_y + dy;
        
        // Check bounds
        if new_x < 0 || new_x >= self.grid_width || new_y < 0 || new_y >= self.grid_height {
            return false;
        }
        
        // Check safety
        if !self.is_safe(new_x, new_y) {
            return false;
        }
        
        // Move
        self.grid_x = new_x;
        self.grid_y = new_y;
        self.update_world_position();
        
        true
    }
    
    /// Find a safe path to a goal position.
    #[func]
    fn find_path_to(&self, goal_x: i32, goal_y: i32, max_risk: f32) -> PackedVector2Array {
        let Some(ref space) = self.space else {
            return PackedVector2Array::new();
        };
        
        let start = (self.grid_x, self.grid_y);
        let goal = (goal_x, goal_y);
        
        match space.find_safe_path(start, goal, max_risk) {
            Some(path) => {
                let mut result = PackedVector2Array::new();
                for (x, y) in path {
                    result.push(Vector2::new(x as f32, y as f32));
                }
                result
            }
            None => PackedVector2Array::new(),
        }
    }
    
    /// Get neighboring positions.
    #[func]
    fn get_neighbors(&self) -> PackedVector2Array {
        let Some(ref space) = self.space else {
            return PackedVector2Array::new();
        };
        
        let mut result = PackedVector2Array::new();
        for (x, y) in space.get_neighbors((self.grid_x, self.grid_y)) {
            result.push(Vector2::new(x as f32, y as f32));
        }
        result
    }
}
