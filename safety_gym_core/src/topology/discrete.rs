//! Discrete navigation space for grid worlds and discrete tasks.

use ndarray::Array1;
use rand::prelude::*;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::{BlackHoleRegion, TopologicalSpace, TopologyData};

/// 2D grid position
pub type GridPos = (i32, i32);

/// Topological space for discrete navigation environments.
///
/// State: Discrete position (e.g., (x, y) in grid world)
/// Embedding: Random Gaussian projection
/// Black holes: Hazard positions (lava, pits, enemies)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteNavigationSpace {
    /// Grid dimensions (width, height)
    pub grid_size: (usize, usize),
    /// Known hazard positions
    pub hazard_positions: HashSet<GridPos>,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Position embeddings (precomputed)
    #[serde(skip)]
    position_embeddings: HashMap<GridPos, Vec<f32>>,
    /// Black hole regions
    black_holes: Vec<BlackHoleRegion>,
    /// Topology data
    topology_data: TopologyData,
    /// Random seed for reproducibility
    seed: u64,
}

impl DiscreteNavigationSpace {
    /// Create a new discrete navigation space.
    ///
    /// # Arguments
    /// * `grid_size` - Size of grid (width, height)
    /// * `hazard_positions` - List of known hazard positions
    /// * `embedding_dim` - Dimension of position embeddings
    /// * `seed` - Random seed for reproducibility
    pub fn new(
        grid_size: (usize, usize),
        hazard_positions: Vec<GridPos>,
        embedding_dim: usize,
        seed: u64,
    ) -> Self {
        let mut space = Self {
            grid_size,
            hazard_positions: hazard_positions.into_iter().collect(),
            embedding_dim,
            position_embeddings: HashMap::new(),
            black_holes: Vec::new(),
            topology_data: TopologyData::new(embedding_dim),
            seed,
        };
        space.init_position_embeddings();
        space
    }
    
    /// Initialize random Gaussian position embeddings
    fn init_position_embeddings(&mut self) {
        let mut rng = StdRng::seed_from_u64(self.seed);
        
        for x in 0..self.grid_size.0 as i32 {
            for y in 0..self.grid_size.1 as i32 {
                let pos = (x, y);
                let mut embedding: Vec<f32> = (0..self.embedding_dim)
                    .map(|_| rng.sample::<f32, _>(StandardNormal))
                    .collect();
                
                // Normalize
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                for v in &mut embedding {
                    *v /= norm;
                }
                
                self.position_embeddings.insert(pos, embedding);
            }
        }
    }
    
    /// Add a hazard position
    pub fn add_hazard(&mut self, pos: GridPos) {
        self.hazard_positions.insert(pos);
    }
    
    /// Remove a hazard position
    pub fn remove_hazard(&mut self, pos: GridPos) {
        self.hazard_positions.remove(&pos);
    }
    
    /// Check if position is within grid bounds
    pub fn in_bounds(&self, pos: GridPos) -> bool {
        pos.0 >= 0 && pos.0 < self.grid_size.0 as i32 &&
        pos.1 >= 0 && pos.1 < self.grid_size.1 as i32
    }
    
    /// Get neighboring positions (4-connected)
    pub fn get_neighbors(&self, pos: GridPos) -> Vec<GridPos> {
        let candidates = [
            (pos.0 - 1, pos.1),
            (pos.0 + 1, pos.1),
            (pos.0, pos.1 - 1),
            (pos.0, pos.1 + 1),
        ];
        
        candidates.into_iter()
            .filter(|&p| self.in_bounds(p))
            .collect()
    }
    
    /// Get neighboring positions (8-connected, includes diagonals)
    pub fn get_neighbors_8(&self, pos: GridPos) -> Vec<GridPos> {
        let candidates = [
            (pos.0 - 1, pos.1), (pos.0 + 1, pos.1),
            (pos.0, pos.1 - 1), (pos.0, pos.1 + 1),
            (pos.0 - 1, pos.1 - 1), (pos.0 - 1, pos.1 + 1),
            (pos.0 + 1, pos.1 - 1), (pos.0 + 1, pos.1 + 1),
        ];
        
        candidates.into_iter()
            .filter(|&p| self.in_bounds(p))
            .collect()
    }
    
    /// Find a safe path using A* with risk constraints
    pub fn find_safe_path(
        &self,
        start: GridPos,
        goal: GridPos,
        max_risk: f32,
    ) -> Option<Vec<GridPos>> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;
        
        #[derive(Clone, PartialEq)]
        struct Node {
            pos: GridPos,
            cost: f32,
            priority: f32,
        }
        
        impl Eq for Node {}
        
        impl Ord for Node {
            fn cmp(&self, other: &Self) -> Ordering {
                other.priority.partial_cmp(&self.priority)
                    .unwrap_or(Ordering::Equal)
            }
        }
        
        impl PartialOrd for Node {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        
        let mut frontier = BinaryHeap::new();
        let mut came_from: HashMap<GridPos, Option<GridPos>> = HashMap::new();
        let mut cost_so_far: HashMap<GridPos, f32> = HashMap::new();
        
        frontier.push(Node { pos: start, cost: 0.0, priority: 0.0 });
        came_from.insert(start, None);
        cost_so_far.insert(start, 0.0);
        
        while let Some(current) = frontier.pop() {
            if current.pos == goal {
                // Reconstruct path
                let mut path = Vec::new();
                let mut pos = Some(goal);
                while let Some(p) = pos {
                    path.push(p);
                    pos = came_from.get(&p).copied().flatten();
                }
                path.reverse();
                return Some(path);
            }
            
            for next in self.get_neighbors(current.pos) {
                // Skip hazards
                if !self.is_safe(&next) {
                    continue;
                }
                
                // Check risk
                let risk = self.compute_harmonic_risk(&next, 5);
                if risk > max_risk {
                    continue;
                }
                
                let new_cost = cost_so_far[&current.pos] + 1.0 + risk;
                
                if !cost_so_far.contains_key(&next) || new_cost < cost_so_far[&next] {
                    cost_so_far.insert(next, new_cost);
                    let heuristic = self.distance(&next, &goal);
                    frontier.push(Node {
                        pos: next,
                        cost: new_cost,
                        priority: new_cost + heuristic,
                    });
                    came_from.insert(next, Some(current.pos));
                }
            }
        }
        
        None // No safe path found
    }
}

impl TopologicalSpace for DiscreteNavigationSpace {
    type State = GridPos;
    
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    
    fn embed(&self, state: &Self::State) -> Array1<f32> {
        self.position_embeddings
            .get(state)
            .map(|v| Array1::from_vec(v.clone()))
            .unwrap_or_else(|| Array1::zeros(self.embedding_dim))
    }
    
    fn distance(&self, state1: &Self::State, state2: &Self::State) -> f32 {
        // Manhattan distance
        ((state1.0 - state2.0).abs() + (state1.1 - state2.1).abs()) as f32
    }
    
    fn is_safe(&self, state: &Self::State) -> bool {
        !self.hazard_positions.contains(state)
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
        let space = DiscreteNavigationSpace::new(
            (10, 10),
            vec![(5, 5), (3, 3)],
            64,
            42,
        );
        
        assert_eq!(space.grid_size, (10, 10));
        assert!(space.hazard_positions.contains(&(5, 5)));
        assert!(space.hazard_positions.contains(&(3, 3)));
        assert_eq!(space.embedding_dim, 64);
    }
    
    #[test]
    fn test_embed() {
        let space = DiscreteNavigationSpace::new((10, 10), vec![], 64, 42);
        
        let emb1 = space.embed(&(0, 0));
        let emb2 = space.embed(&(5, 5));
        
        assert_eq!(emb1.len(), 64);
        assert_eq!(emb2.len(), 64);
        
        // Embeddings should be normalized
        let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm1 - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_distance() {
        let space = DiscreteNavigationSpace::new((10, 10), vec![], 64, 42);
        
        assert_eq!(space.distance(&(0, 0), &(3, 4)), 7.0);
        assert_eq!(space.distance(&(5, 5), &(5, 5)), 0.0);
    }
    
    #[test]
    fn test_is_safe() {
        let space = DiscreteNavigationSpace::new((10, 10), vec![(5, 5)], 64, 42);
        
        assert!(space.is_safe(&(0, 0)));
        assert!(!space.is_safe(&(5, 5)));
    }
    
    #[test]
    fn test_neighbors() {
        let space = DiscreteNavigationSpace::new((10, 10), vec![], 64, 42);
        
        let neighbors = space.get_neighbors((5, 5));
        assert_eq!(neighbors.len(), 4);
        
        // Corner has only 2 neighbors
        let corner_neighbors = space.get_neighbors((0, 0));
        assert_eq!(corner_neighbors.len(), 2);
    }
    
    #[test]
    fn test_path_finding() {
        let space = DiscreteNavigationSpace::new((10, 10), vec![(5, 5)], 64, 42);
        
        let path = space.find_safe_path((0, 0), (9, 9), 1.0);
        assert!(path.is_some());
        
        let path = path.unwrap();
        assert_eq!(path[0], (0, 0));
        assert_eq!(*path.last().unwrap(), (9, 9));
        
        // Path should not go through hazard
        assert!(!path.contains(&(5, 5)));
    }
}
