//! Sheaf-Geodesic Policy Optimization (SGPO) policy implementation.
//!
//! SGPO uses the Riemannian metric derived from sheaf cohomology to
//! constrain policy updates away from dangerous regions.

use crate::topology::TopologicalSpace;
use serde::{Deserialize, Serialize};

/// Configuration for SGPO policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGPOConfig {
    /// Black hole strength (α in the conformal factor)
    pub alpha: f32,
    /// Risk threshold for safety constraint
    pub risk_threshold: f32,
    /// Clipping epsilon for Clipped-SGPO
    pub clip_epsilon: f32,
    /// Temperature for action scaling
    pub temperature: f32,
}

impl Default for SGPOConfig {
    fn default() -> Self {
        Self {
            alpha: 2.0,
            risk_threshold: 0.5,
            clip_epsilon: 0.2,
            temperature: 1.0,
        }
    }
}

/// SGPO Policy wrapper that applies safety constraints.
///
/// This wraps any base policy and applies the Riemannian metric
/// constraint to prevent entering black hole regions.
#[derive(Debug, Clone)]
pub struct SGPOPolicy<S: TopologicalSpace> {
    /// Topology space for computing safety metrics
    pub space: S,
    /// SGPO configuration
    pub config: SGPOConfig,
}

impl<S: TopologicalSpace> SGPOPolicy<S> {
    /// Create a new SGPO policy
    pub fn new(space: S, config: SGPOConfig) -> Self {
        Self { space, config }
    }
    
    /// Compute safety-adjusted action.
    ///
    /// Given a raw action from the base policy, apply safety constraints
    /// based on the current state's proximity to black holes.
    pub fn apply_safety_constraint(
        &self,
        state: &S::State,
        raw_action: &[f32],
    ) -> Vec<f32> {
        // Compute Riemannian metric (conformal factor)
        let phi = self.space.compute_riemannian_metric(state, self.config.alpha);
        
        // Compute harmonic risk
        let risk = self.space.compute_harmonic_risk(state, 5);
        
        // Scale action inversely with metric (slow down near black holes)
        let scale = if phi > 1.0 {
            (1.0 / phi).clamp(0.1, 1.0)
        } else {
            1.0
        };
        
        // Apply risk-based scaling
        let risk_scale = if risk > self.config.risk_threshold {
            1.0 - (risk - self.config.risk_threshold) / (1.0 - self.config.risk_threshold)
        } else {
            1.0
        };
        
        // Combine scales
        let total_scale = scale * risk_scale * self.config.temperature;
        
        raw_action.iter()
            .map(|&a| a * total_scale)
            .collect()
    }
    
    /// Compute the SGPO advantage for policy gradient.
    ///
    /// advantage = (V - ω) / sqrt(g)
    /// where V is value, ω is harmonic component, g is metric
    pub fn compute_advantage(
        &self,
        state: &S::State,
        value: f32,
        harmonic_component: f32,
    ) -> f32 {
        let metric = self.space.compute_riemannian_metric(state, self.config.alpha);
        (value - harmonic_component) / metric.sqrt().max(0.01)
    }
    
    /// Check if a state is safe to enter
    pub fn is_state_safe(&self, state: &S::State) -> bool {
        // Check explicit hazards
        if !self.space.is_safe(state) {
            return false;
        }
        
        // Check risk threshold
        let risk = self.space.compute_harmonic_risk(state, 5);
        risk <= self.config.risk_threshold
    }
    
    /// Compute safety score (higher = safer)
    pub fn safety_score(&self, state: &S::State) -> f32 {
        let risk = self.space.compute_harmonic_risk(state, 5);
        let proximity = self.space.compute_black_hole_proximity(state);
        
        // Combine risk and proximity
        let risk_score = 1.0 - risk;
        let proximity_score = (proximity / (proximity + 1.0)).min(1.0);
        
        (risk_score + proximity_score) / 2.0
    }
}

/// Clipped-SGPO extension for more stable training.
///
/// Combines the clipping mechanism of PPO with SGPO's geodesic constraints.
#[derive(Debug, Clone)]
pub struct ClippedSGPOPolicy<S: TopologicalSpace> {
    inner: SGPOPolicy<S>,
}

impl<S: TopologicalSpace> ClippedSGPOPolicy<S> {
    pub fn new(space: S, config: SGPOConfig) -> Self {
        Self {
            inner: SGPOPolicy::new(space, config),
        }
    }
    
    /// Compute clipped advantage ratio.
    ///
    /// Uses both PPO-style probability ratio clipping and
    /// SGPO's metric-based advantage scaling.
    pub fn compute_clipped_objective(
        &self,
        state: &S::State,
        value: f32,
        harmonic_component: f32,
        prob_ratio: f32,
    ) -> f32 {
        let advantage = self.inner.compute_advantage(state, value, harmonic_component);
        let epsilon = self.inner.config.clip_epsilon;
        
        // PPO-style clipping
        let unclipped = prob_ratio * advantage;
        let clipped = prob_ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantage;
        
        // Take minimum for pessimistic bound
        unclipped.min(clipped)
    }
    
    pub fn space(&self) -> &S {
        &self.inner.space
    }
    
    pub fn config(&self) -> &SGPOConfig {
        &self.inner.config
    }
    
    pub fn apply_safety_constraint(&self, state: &S::State, raw_action: &[f32]) -> Vec<f32> {
        self.inner.apply_safety_constraint(state, raw_action)
    }
    
    pub fn is_state_safe(&self, state: &S::State) -> bool {
        self.inner.is_state_safe(state)
    }
    
    pub fn safety_score(&self, state: &S::State) -> f32 {
        self.inner.safety_score(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::discrete::DiscreteNavigationSpace;
    
    #[test]
    fn test_gpo_policy() {
        let space = DiscreteNavigationSpace::new((10, 10), vec![(5, 5)], 64, 42);
        let policy = SGPOPolicy::new(space, SGPOConfig::default());
        
        // Safe state
        assert!(policy.is_state_safe(&(0, 0)));
        
        // Hazard state
        assert!(!policy.is_state_safe(&(5, 5)));
    }
    
    #[test]
    fn test_safety_constraint() {
        let space = DiscreteNavigationSpace::new((10, 10), vec![], 64, 42);
        let policy = SGPOPolicy::new(space, SGPOConfig::default());
        
        let raw_action = vec![1.0, 0.5];
        let safe_action = policy.apply_safety_constraint(&(0, 0), &raw_action);
        
        // Should be scaled by temperature
        assert_eq!(safe_action.len(), 2);
    }
    
    #[test]
    fn test_clipped_gpo() {
        let space = DiscreteNavigationSpace::new((10, 10), vec![], 64, 42);
        let config = SGPOConfig {
            clip_epsilon: 0.2,
            ..Default::default()
        };
        let policy = ClippedSGPOPolicy::new(space, config);
        
        let objective = policy.compute_clipped_objective(
            &(0, 0),
            1.0,  // value
            0.0,  // harmonic
            1.1,  // prob_ratio
        );
        
        assert!(objective.is_finite());
    }
}
