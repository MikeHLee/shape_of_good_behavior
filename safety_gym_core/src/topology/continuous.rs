//! Continuous control space for MuJoCo-style environments.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

use super::{BlackHoleRegion, TopologicalSpace, TopologyData};

/// Continuous 2D position
pub type ContinuousPos = [f32; 2];

/// Obstacle in continuous space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Obstacle {
    pub center: ContinuousPos,
    pub radius: f32,
}

impl Obstacle {
    pub fn new(center: ContinuousPos, radius: f32) -> Self {
        Self { center, radius }
    }
    
    pub fn contains(&self, point: &ContinuousPos) -> bool {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        dx * dx + dy * dy <= self.radius * self.radius
    }
    
    pub fn distance_to(&self, point: &ContinuousPos) -> f32 {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        ((dx * dx + dy * dy).sqrt() - self.radius).max(0.0)
    }
}

/// Topological space for continuous control environments.
///
/// State: Continuous position in R² (extendable to higher dimensions)
/// Embedding: Direct position (optionally with velocity)
/// Black holes: Circular obstacle regions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousControlSpace {
    /// Bounds of the space: [[x_min, x_max], [y_min, y_max]]
    pub bounds: [[f32; 2]; 2],
    /// Obstacles in the environment
    pub obstacles: Vec<Obstacle>,
    /// Goal position (optional)
    pub goal: Option<ContinuousPos>,
    /// Physics parameters
    pub dt: f32,
    pub friction: f32,
    pub max_velocity: f32,
    /// Black hole regions
    black_holes: Vec<BlackHoleRegion>,
    /// Topology data
    topology_data: TopologyData,
}

impl ContinuousControlSpace {
    /// Create a new continuous control space.
    pub fn new(
        bounds: [[f32; 2]; 2],
        obstacles: Vec<Obstacle>,
        goal: Option<ContinuousPos>,
    ) -> Self {
        Self {
            bounds,
            obstacles,
            goal,
            dt: 0.1,
            friction: 0.1,
            max_velocity: 1.0,
            black_holes: Vec::new(),
            topology_data: TopologyData::new(2), // 2D embedding
        }
    }
    
    /// Create with physics parameters
    pub fn with_physics(mut self, dt: f32, friction: f32, max_velocity: f32) -> Self {
        self.dt = dt;
        self.friction = friction;
        self.max_velocity = max_velocity;
        self
    }
    
    /// Add an obstacle
    pub fn add_obstacle(&mut self, center: ContinuousPos, radius: f32) {
        self.obstacles.push(Obstacle::new(center, radius));
    }
    
    /// Check if position is within bounds
    pub fn in_bounds(&self, pos: &ContinuousPos) -> bool {
        pos[0] >= self.bounds[0][0] && pos[0] <= self.bounds[0][1] &&
        pos[1] >= self.bounds[1][0] && pos[1] <= self.bounds[1][1]
    }
    
    /// Clamp position to bounds
    pub fn clamp_to_bounds(&self, pos: &mut ContinuousPos) {
        pos[0] = pos[0].clamp(self.bounds[0][0], self.bounds[0][1]);
        pos[1] = pos[1].clamp(self.bounds[1][0], self.bounds[1][1]);
    }
    
    /// Check if position collides with any obstacle
    pub fn collides(&self, pos: &ContinuousPos) -> bool {
        self.obstacles.iter().any(|obs| obs.contains(pos))
    }
    
    /// Compute distance to nearest obstacle
    pub fn distance_to_nearest_obstacle(&self, pos: &ContinuousPos) -> f32 {
        self.obstacles.iter()
            .map(|obs| obs.distance_to(pos))
            .fold(f32::INFINITY, f32::min)
    }
    
    /// Simulate one physics step
    pub fn step(&self, pos: &ContinuousPos, velocity: &[f32; 2], action: &[f32; 2]) -> (ContinuousPos, [f32; 2]) {
        // Apply action (acceleration)
        let mut new_vel = [
            velocity[0] + action[0] * self.dt,
            velocity[1] + action[1] * self.dt,
        ];
        
        // Apply friction
        new_vel[0] *= 1.0 - self.friction;
        new_vel[1] *= 1.0 - self.friction;
        
        // Clamp velocity
        let speed = (new_vel[0] * new_vel[0] + new_vel[1] * new_vel[1]).sqrt();
        if speed > self.max_velocity {
            let scale = self.max_velocity / speed;
            new_vel[0] *= scale;
            new_vel[1] *= scale;
        }
        
        // Update position
        let mut new_pos = [
            pos[0] + new_vel[0] * self.dt,
            pos[1] + new_vel[1] * self.dt,
        ];
        
        // Clamp to bounds
        self.clamp_to_bounds(&mut new_pos);
        
        (new_pos, new_vel)
    }
    
    /// Convert obstacles to black hole regions
    pub fn obstacles_to_black_holes(&mut self, safety_margin: f32) {
        self.black_holes.clear();
        
        for (i, obs) in self.obstacles.iter().enumerate() {
            self.black_holes.push(BlackHoleRegion::new(
                obs.center.to_vec(),
                obs.radius * safety_margin,
                1.0,
                i as i32,
            ));
        }
    }
    
    /// Compute stopping distance given current velocity.
    ///
    /// Uses the geometric series sum for exponential decay:
    /// d = v * dt / friction
    ///
    /// This assumes the agent applies zero acceleration and friction
    /// gradually reduces velocity to zero.
    ///
    /// # Arguments
    /// * `velocity` - Current velocity vector [vx, vy]
    ///
    /// # Returns
    /// Distance traveled before coming to rest (in same units as position)
    ///
    /// # Example
    /// ```
    /// # use safety_gym_core::topology::continuous::ContinuousControlSpace;
    /// let space = ContinuousControlSpace::new(
    ///     [[-10.0, 10.0], [-10.0, 10.0]],
    ///     vec![],
    ///     None,
    /// ).with_physics(0.1, 0.1, 2.0);
    ///
    /// let velocity = [1.0, 0.0];
    /// let stop_dist = space.stopping_distance(&velocity);
    /// assert!((stop_dist - 1.0).abs() < 0.01); // Should be ~1.0
    /// ```
    pub fn stopping_distance(&self, velocity: &[f32; 2]) -> f32 {
        let speed = (velocity[0].powi(2) + velocity[1].powi(2)).sqrt();
        
        if speed < 1e-6 {
            return 0.0; // Already stopped
        }
        
        // Geometric series: sum_{i=0}^∞ v * (1-f)^i * dt = v * dt / f
        // where f is friction coefficient
        speed * self.dt / self.friction.max(1e-6)
    }
    
    /// Check if position is safe considering current velocity.
    ///
    /// A position is considered safe with velocity if:
    /// 1. The position itself is safe (not in obstacle or out of bounds)
    /// 2. The stopping distance does not overlap with any obstacle
    ///
    /// This prevents the agent from entering states where collision
    /// is inevitable due to momentum.
    ///
    /// # Arguments
    /// * `pos` - Current position [x, y]
    /// * `vel` - Current velocity [vx, vy]
    ///
    /// # Returns
    /// `true` if the agent can safely stop before hitting any obstacle
    ///
    /// # Example
    /// ```
    /// # use safety_gym_core::topology::continuous::{ContinuousControlSpace, Obstacle};
    /// let mut space = ContinuousControlSpace::new(
    ///     [[-10.0, 10.0], [-10.0, 10.0]],
    ///     vec![Obstacle::new([5.0, 0.0], 1.0)],
    ///     None,
    /// ).with_physics(0.1, 0.1, 2.0);
    ///
    /// // Stationary agent far from obstacle: safe
    /// assert!(space.is_safe_with_velocity(&[0.0, 0.0], &[0.0, 0.0]));
    ///
    /// // Moving toward obstacle but far enough to stop: safe
    /// assert!(space.is_safe_with_velocity(&[2.0, 0.0], &[0.5, 0.0]));
    ///
    /// // Moving toward obstacle too fast to stop: unsafe
    /// assert!(!space.is_safe_with_velocity(&[3.5, 0.0], &[1.5, 0.0]));
    /// ```
    pub fn is_safe_with_velocity(&self, pos: &ContinuousPos, vel: &[f32; 2]) -> bool {
        // First check if current position is safe
        if !self.is_safe(pos) {
            return false;
        }
        
        // Compute stopping distance
        let stop_dist = self.stopping_distance(vel);
        
        // Check if we have enough clearance to stop
        let obstacle_dist = self.distance_to_nearest_obstacle(pos);
        
        // Safety buffer: need at least stop_dist + 5% margin
        let required_clearance = stop_dist + 0.05;
        
        obstacle_dist >= required_clearance
    }
}

impl TopologicalSpace for ContinuousControlSpace {
    type State = ContinuousPos;
    
    fn embedding_dim(&self) -> usize {
        2 // 2D position
    }
    
    fn embed(&self, state: &Self::State) -> Array1<f32> {
        Array1::from_vec(state.to_vec())
    }
    
    fn distance(&self, state1: &Self::State, state2: &Self::State) -> f32 {
        let dx = state1[0] - state2[0];
        let dy = state1[1] - state2[1];
        (dx * dx + dy * dy).sqrt()
    }
    
    fn is_safe(&self, state: &Self::State) -> bool {
        self.in_bounds(state) && !self.collides(state)
    }
    
    fn black_holes(&self) -> &[BlackHoleRegion] {
        &self.black_holes
    }
    
    fn black_holes_mut(&mut self) -> &mut Vec<BlackHoleRegion> {
        &mut self.black_holes
    }
    
    fn topology_data(&self) -> &TopologyData {
        &self.topology_data
    }
    
    fn topology_data_mut(&mut self) -> &mut TopologyData {
        &mut self.topology_data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_space() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![Obstacle::new([0.0, 0.0], 1.0)],
            Some([5.0, 5.0]),
        );
        
        assert_eq!(space.obstacles.len(), 1);
        assert_eq!(space.goal, Some([5.0, 5.0]));
    }
    
    #[test]
    fn test_in_bounds() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![],
            None,
        );
        
        assert!(space.in_bounds(&[0.0, 0.0]));
        assert!(space.in_bounds(&[-10.0, 10.0]));
        assert!(!space.in_bounds(&[15.0, 0.0]));
    }
    
    #[test]
    fn test_collides() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![Obstacle::new([0.0, 0.0], 1.0)],
            None,
        );
        
        assert!(space.collides(&[0.0, 0.0]));
        assert!(space.collides(&[0.5, 0.0]));
        assert!(!space.collides(&[2.0, 0.0]));
    }
    
    #[test]
    fn test_is_safe() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![Obstacle::new([0.0, 0.0], 1.0)],
            None,
        );
        
        assert!(!space.is_safe(&[0.0, 0.0])); // Inside obstacle
        assert!(space.is_safe(&[5.0, 5.0])); // Safe
        assert!(!space.is_safe(&[15.0, 0.0])); // Out of bounds
    }
    
    #[test]
    fn test_step() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![],
            None,
        ).with_physics(0.1, 0.1, 2.0);
        
        let pos = [0.0, 0.0];
        let vel = [0.0, 0.0];
        let action = [1.0, 0.0];
        
        let (new_pos, new_vel) = space.step(&pos, &vel, &action);
        
        // Should have moved right
        assert!(new_pos[0] > 0.0);
        assert!(new_vel[0] > 0.0);
    }
    
    #[test]
    fn test_distance() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![],
            None,
        );
        
        assert!((space.distance(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 0.001);
    }
    
    #[test]
    fn test_stopping_distance_zero_velocity() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![],
            None,
        ).with_physics(0.1, 0.1, 2.0);
        
        let vel = [0.0, 0.0];
        let stop_dist = space.stopping_distance(&vel);
        
        assert_eq!(stop_dist, 0.0);
    }
    
    #[test]
    fn test_stopping_distance_unit_velocity() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![],
            None,
        ).with_physics(0.1, 0.1, 2.0);
        
        // With friction=0.1, dt=0.1, speed=1.0
        // Expected: 1.0 * 0.1 / 0.1 = 1.0
        let vel = [1.0, 0.0];
        let stop_dist = space.stopping_distance(&vel);
        
        assert!((stop_dist - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_stopping_distance_diagonal_velocity() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![],
            None,
        ).with_physics(0.1, 0.1, 2.0);
        
        // Velocity [0.6, 0.8] has magnitude 1.0
        let vel = [0.6, 0.8];
        let stop_dist = space.stopping_distance(&vel);
        
        assert!((stop_dist - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_stopping_distance_high_friction() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![],
            None,
        ).with_physics(0.1, 0.5, 2.0); // High friction
        
        // With friction=0.5, dt=0.1, speed=1.0
        // Expected: 1.0 * 0.1 / 0.5 = 0.2
        let vel = [1.0, 0.0];
        let stop_dist = space.stopping_distance(&vel);
        
        assert!((stop_dist - 0.2).abs() < 0.001);
    }
    
    #[test]
    fn test_is_safe_with_velocity_stationary() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![Obstacle::new([5.0, 0.0], 1.0)],
            None,
        ).with_physics(0.1, 0.1, 2.0);
        
        // Stationary agent far from obstacle
        assert!(space.is_safe_with_velocity(&[0.0, 0.0], &[0.0, 0.0]));
    }
    
    #[test]
    fn test_is_safe_with_velocity_moving_away() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![Obstacle::new([5.0, 0.0], 1.0)],
            None,
        ).with_physics(0.1, 0.1, 2.0);
        
        // Moving away from obstacle: always safe
        assert!(space.is_safe_with_velocity(&[0.0, 0.0], &[-1.0, 0.0]));
    }
    
    #[test]
    fn test_is_safe_with_velocity_safe_approach() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![Obstacle::new([5.0, 0.0], 1.0)],
            None,
        ).with_physics(0.1, 0.1, 2.0);
        
        // At position [2.0, 0.0], obstacle at [5.0, 0.0] with radius 1.0
        // Distance to obstacle surface: 5.0 - 2.0 - 1.0 = 2.0
        // Velocity [0.5, 0.0] → stopping distance = 0.5 * 0.1 / 0.1 = 0.5
        // Required clearance: 0.5 + 0.05 = 0.55
        // 2.0 > 0.55 → safe
        assert!(space.is_safe_with_velocity(&[2.0, 0.0], &[0.5, 0.0]));
    }
    
    #[test]
    fn test_is_safe_with_velocity_unsafe_approach() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![Obstacle::new([5.0, 0.0], 1.0)],
            None,
        ).with_physics(0.1, 0.1, 2.0);
        
        // At position [3.5, 0.0], obstacle at [5.0, 0.0] with radius 1.0
        // Distance to obstacle surface: 5.0 - 3.5 - 1.0 = 0.5
        // Velocity [1.5, 0.0] → stopping distance = 1.5 * 0.1 / 0.1 = 1.5
        // Required clearance: 1.5 + 0.05 = 1.55
        // 0.5 < 1.55 → unsafe
        assert!(!space.is_safe_with_velocity(&[3.5, 0.0], &[1.5, 0.0]));
    }
    
    #[test]
    fn test_is_safe_with_velocity_already_inside_obstacle() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![Obstacle::new([5.0, 0.0], 1.0)],
            None,
        ).with_physics(0.1, 0.1, 2.0);
        
        // Already inside obstacle: unsafe regardless of velocity
        assert!(!space.is_safe_with_velocity(&[5.0, 0.0], &[0.0, 0.0]));
        assert!(!space.is_safe_with_velocity(&[5.0, 0.0], &[-1.0, 0.0]));
    }
    
    #[test]
    fn test_is_safe_with_velocity_out_of_bounds() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![],
            None,
        ).with_physics(0.1, 0.1, 2.0);
        
        // Out of bounds: unsafe
        assert!(!space.is_safe_with_velocity(&[15.0, 0.0], &[0.0, 0.0]));
    }
    
    #[test]
    fn test_is_safe_with_velocity_perpendicular_motion() {
        let space = ContinuousControlSpace::new(
            [[-10.0, 10.0], [-10.0, 10.0]],
            vec![Obstacle::new([5.0, 0.0], 1.0)],
            None,
        ).with_physics(0.1, 0.1, 2.0);
        
        // Moving perpendicular to obstacle direction
        // At [2.0, 0.0], moving [0.0, 1.0]
        // Distance to obstacle: 2.0, stopping distance: 1.0
        // Should be safe since not moving toward obstacle
        assert!(space.is_safe_with_velocity(&[2.0, 0.0], &[0.0, 1.0]));
    }
}
