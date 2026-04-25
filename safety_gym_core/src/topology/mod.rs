//! Topological space abstractions for decision-making environments.
//!
//! Key insight: We don't need full manifold structure, just:
//! 1. Distance metric (for neighborhoods)
//! 2. Embedding function (for Hodge decomposition)
//! 3. Boundary detection (for black holes)

pub mod discrete;
pub mod continuous;

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// A region in the state space identified as dangerous ("black hole").
///
/// Black holes are regions where the Riemannian metric has singularities,
/// representing states that should be avoided at all costs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackHoleRegion {
    /// Center of the black hole in embedding space
    pub center: Vec<f32>,
    /// Radius of the dangerous region
    pub radius: f32,
    /// Strength of the singularity (higher = more dangerous)
    pub strength: f32,
    /// Label for identification
    pub label: i32,
}

impl BlackHoleRegion {
    pub fn new(center: Vec<f32>, radius: f32, strength: f32, label: i32) -> Self {
        Self { center, radius, strength, label }
    }
    
    /// Check if a point is inside this black hole region
    pub fn contains(&self, point: &[f32]) -> bool {
        if point.len() != self.center.len() {
            return false;
        }
        let dist_sq: f32 = point.iter()
            .zip(self.center.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        dist_sq <= self.radius.powi(2)
    }
    
    /// Compute distance from point to the boundary of this black hole
    pub fn distance_to_boundary(&self, point: &[f32]) -> f32 {
        let dist: f32 = point.iter()
            .zip(self.center.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        (dist - self.radius).max(0.0)
    }
}

/// Database of topology samples collected during exploration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopologyData {
    /// Collected state embeddings
    pub embeddings: Vec<Vec<f32>>,
    /// Harmonic risk at each state (from H¹ computation)
    pub harmonic_risk: Vec<f32>,
    /// Embedding dimension
    pub embedding_dim: usize,
}

impl TopologyData {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embeddings: Vec::new(),
            harmonic_risk: Vec::new(),
            embedding_dim,
        }
    }
    
    pub fn add_sample(&mut self, embedding: Vec<f32>, risk: f32) {
        debug_assert_eq!(embedding.len(), self.embedding_dim);
        self.embeddings.push(embedding);
        self.harmonic_risk.push(risk);
    }
    
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
    
    /// Find k nearest neighbors to a query point (brute force)
    pub fn find_k_nearest(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut distances: Vec<(usize, f32)> = self.embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                let dist = emb.iter()
                    .zip(query.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                (i, dist)
            })
            .collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }
}

/// Abstract trait for any space with topological structure.
///
/// This enables sheaf-theoretic safety analysis on arbitrary decision spaces,
/// not just text embeddings.
pub trait TopologicalSpace: Send + Sync {
    /// The state type for this space
    type State: Clone + Send + Sync;
    
    /// Get the embedding dimension
    fn embedding_dim(&self) -> usize;
    
    /// Embed state into a common vector space for topology computation.
    ///
    /// This is the key abstraction that allows us to apply sheaf theory
    /// to arbitrary spaces. The embedding should preserve local structure.
    fn embed(&self, state: &Self::State) -> Array1<f32>;
    
    /// Compute distance between two states.
    ///
    /// This defines the metric structure needed for:
    /// - Neighborhood computation
    /// - Geodesic distances
    /// - Riemannian metric construction
    fn distance(&self, state1: &Self::State, state2: &Self::State) -> f32;
    
    /// Check if state is in safe region (not a black hole).
    fn is_safe(&self, state: &Self::State) -> bool;
    
    /// Get black hole regions
    fn black_holes(&self) -> &[BlackHoleRegion];
    
    /// Get mutable access to black hole regions
    fn black_holes_mut(&mut self) -> &mut Vec<BlackHoleRegion>;
    
    /// Get topology data
    fn topology_data(&self) -> &TopologyData;
    
    /// Get mutable access to topology data
    fn topology_data_mut(&mut self) -> &mut TopologyData;
    
    /// Estimate H¹ cohomology risk at this state using KNN.
    fn compute_harmonic_risk(&self, state: &Self::State, k: usize) -> f32 {
        let topology = self.topology_data();
        if topology.len() < k {
            return 0.5; // Unknown risk
        }
        
        let embedding = self.embed(state);
        let neighbors = topology.find_k_nearest(embedding.as_slice().unwrap(), k);
        
        if neighbors.is_empty() {
            return 0.5;
        }
        
        // Weighted average by inverse distance
        let mut weighted_sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        
        for (idx, dist) in neighbors {
            let weight = 1.0 / (dist + 0.01);
            weighted_sum += topology.harmonic_risk[idx] * weight;
            weight_sum += weight;
        }
        
        weighted_sum / weight_sum
    }
    
    /// Compute minimum distance to any black hole region.
    fn compute_black_hole_proximity(&self, state: &Self::State) -> f32 {
        let black_holes = self.black_holes();
        if black_holes.is_empty() {
            return f32::INFINITY;
        }
        
        let embedding = self.embed(state);
        let emb_slice = embedding.as_slice().unwrap();
        
        black_holes.iter()
            .map(|bh| bh.distance_to_boundary(emb_slice))
            .fold(f32::INFINITY, f32::min)
    }
    
    /// Compute conformal factor for Riemannian metric at state.
    ///
    /// The metric is g(x) = φ(x)² · δ where φ(x) ≈ 1/dist(x, B)^α
    /// creates infinite "energy barriers" at black holes.
    fn compute_riemannian_metric(&self, state: &Self::State, alpha: f32) -> f32 {
        let proximity = self.compute_black_hole_proximity(state);
        
        if proximity.is_infinite() {
            return 1.0; // Flat metric if no black holes
        }
        
        // Conformal factor with singularity at black holes
        1.0 / (proximity + 0.01).powf(alpha)
    }
    
    /// Add a topology sample
    fn add_topology_sample(&mut self, state: &Self::State, risk: f32) {
        let embedding = self.embed(state);
        self.topology_data_mut().add_sample(embedding.to_vec(), risk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_black_hole_contains() {
        let bh = BlackHoleRegion::new(vec![0.0, 0.0], 1.0, 1.0, 0);
        assert!(bh.contains(&[0.0, 0.0]));
        assert!(bh.contains(&[0.5, 0.5]));
        assert!(!bh.contains(&[1.5, 0.0]));
    }
    
    #[test]
    fn test_black_hole_distance() {
        let bh = BlackHoleRegion::new(vec![0.0, 0.0], 1.0, 1.0, 0);
        assert!((bh.distance_to_boundary(&[2.0, 0.0]) - 1.0).abs() < 0.001);
        assert_eq!(bh.distance_to_boundary(&[0.5, 0.0]), 0.0);
    }
    
    #[test]
    fn test_topology_data() {
        let mut data = TopologyData::new(2);
        data.add_sample(vec![0.0, 0.0], 0.1);
        data.add_sample(vec![1.0, 0.0], 0.5);
        data.add_sample(vec![0.0, 1.0], 0.9);
        
        assert_eq!(data.len(), 3);
        
        let neighbors = data.find_k_nearest(&[0.1, 0.1], 2);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].0, 0); // Closest is origin
    }
}
