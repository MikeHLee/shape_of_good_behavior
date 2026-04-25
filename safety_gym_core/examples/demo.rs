//! Demo example showing how to use the safety gym core library.
//!
//! Run with: cargo run --example demo

use safety_gym_core::{
    topology::{
        discrete::DiscreteNavigationSpace,
        continuous::ContinuousControlSpace,
        TopologicalSpace,
    },
    policy::gpo::{SGPOConfig, SGPOPolicy, ClippedSGPOPolicy},
};

fn demo_discrete_navigation() {
    println!("\n=== Discrete Navigation Demo ===\n");
    
    // Create a 20x20 grid with some hazards
    let hazards = vec![(5, 5), (5, 6), (5, 7), (10, 10), (15, 15)];
    let space = DiscreteNavigationSpace::new((20, 20), hazards.clone(), 64, 42);
    
    println!("Grid size: 20x20");
    println!("Hazards: {:?}", hazards);
    println!("Embedding dimension: {}", space.embedding_dim());
    
    // Check safety at various positions
    let test_positions = [(0, 0), (5, 5), (10, 5), (19, 19)];
    println!("\nSafety checks:");
    for pos in test_positions {
        let is_safe = space.is_safe(&pos);
        let risk = space.compute_harmonic_risk(&pos, 5);
        println!("  {:?}: safe={}, risk={:.3}", pos, is_safe, risk);
    }
    
    // Get embedding for a position
    let embedding = space.embed(&(0, 0));
    println!("\nEmbedding for (0,0): dim={}, norm={:.3}",
        embedding.len(),
        embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    
    // Find a safe path
    println!("\nPath finding from (0,0) to (19,19):");
    match space.find_safe_path((0, 0), (19, 19), 0.8) {
        Some(path) => {
            println!("  Found path with {} steps", path.len());
            println!("  First 5 steps: {:?}", &path[..5.min(path.len())]);
        }
        None => println!("  No safe path found!"),
    }
}

fn demo_continuous_control() {
    println!("\n=== Continuous Control Demo ===\n");
    
    // Create a 20x20 continuous space with obstacles
    let mut space = ContinuousControlSpace::new(
        [[-10.0, 10.0], [-10.0, 10.0]],
        vec![],
        Some([8.0, 8.0]),  // Goal position
    ).with_physics(0.1, 0.1, 2.0);
    
    // Add some circular obstacles
    space.add_obstacle([0.0, 0.0], 1.5);
    space.add_obstacle([3.0, 3.0], 1.0);
    space.add_obstacle([-3.0, 5.0], 0.8);
    
    println!("Bounds: [-10, 10] x [-10, 10]");
    println!("Goal: (8, 8)");
    println!("Obstacles: 3 circular regions");
    
    // Check safety at various positions
    let test_positions = [[0.0, 0.0], [5.0, 5.0], [0.5, 0.0], [-8.0, -8.0]];
    println!("\nSafety checks:");
    for pos in test_positions {
        let is_safe = space.is_safe(&pos);
        let dist = space.distance_to_nearest_obstacle(&pos);
        println!("  {:?}: safe={}, dist_to_obstacle={:.3}", pos, is_safe, dist);
    }
    
    // Simulate a trajectory
    println!("\nSimulating trajectory from (-8, -8):");
    let mut pos = [-8.0, -8.0];
    let mut vel = [0.0, 0.0];
    let action = [0.5, 0.5];  // Move toward goal
    
    for step in 0..10 {
        let (new_pos, new_vel) = space.step(&pos, &vel, &action);
        let dist_to_goal = space.distance(&new_pos, &[8.0, 8.0]);
        println!("  Step {}: pos=({:.2}, {:.2}), vel=({:.2}, {:.2}), dist_to_goal={:.2}",
            step, new_pos[0], new_pos[1], new_vel[0], new_vel[1], dist_to_goal);
        pos = new_pos;
        vel = new_vel;
    }
}

fn demo_gpo_policy() {
    println!("\n=== SGPO Policy Demo ===\n");
    
    // Create a discrete space with hazards
    let hazards = vec![(5, 5), (5, 6), (6, 5), (6, 6)];  // 2x2 danger zone
    let space = DiscreteNavigationSpace::new((10, 10), hazards, 64, 42);
    
    // Create SGPO policy
    let config = SGPOConfig {
        alpha: 2.0,
        risk_threshold: 0.5,
        clip_epsilon: 0.2,
        temperature: 1.0,
    };
    let policy = SGPOPolicy::new(space, config);
    
    println!("SGPO Policy Configuration:");
    println!("  alpha (black hole strength): {}", policy.config.alpha);
    println!("  risk_threshold: {}", policy.config.risk_threshold);
    
    // Test safety at various positions
    let test_positions = [(0, 0), (4, 4), (5, 5), (9, 9)];
    println!("\nState safety analysis:");
    for pos in test_positions {
        let is_safe = policy.is_state_safe(&pos);
        let safety_score = policy.safety_score(&pos);
        println!("  {:?}: safe={}, safety_score={:.3}", pos, is_safe, safety_score);
    }
    
    // Test action safety constraint
    let raw_action = vec![1.0, 0.5];
    println!("\nAction safety constraints (raw_action = {:?}):", raw_action);
    for pos in [(0, 0), (4, 4)] {
        let safe_action = policy.apply_safety_constraint(&pos, &raw_action);
        println!("  At {:?}: safe_action = {:?}", pos, safe_action);
    }
    
    // Clipped SGPO
    println!("\n--- Clipped-SGPO Extension ---");
    let space2 = DiscreteNavigationSpace::new((10, 10), vec![], 64, 42);
    let clipped_policy = ClippedSGPOPolicy::new(space2, SGPOConfig::default());
    
    let objective = clipped_policy.compute_clipped_objective(
        &(5, 5),
        1.0,   // value
        0.0,   // harmonic component
        1.15,  // probability ratio
    );
    println!("Clipped objective at (5,5): {:.4}", objective);
}

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║       Safety Gym Core - Rust Library Demo            ║");
    println!("║  Sheaf-Theoretic Safety for Reinforcement Learning   ║");
    println!("╚══════════════════════════════════════════════════════╝");
    
    demo_discrete_navigation();
    demo_continuous_control();
    demo_gpo_policy();
    
    println!("\n✅ Demo complete!");
}
