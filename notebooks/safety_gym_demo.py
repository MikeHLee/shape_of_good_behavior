"""
Demo: Topological Safety Gym Library

Shows how to use the Safety Gym library across different decision spaces:
1. Discrete Navigation (Grid World)
2. Continuous Control (Reaching Task)
3. Integration with existing environments
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
from safety_gym import TopologicalSafetyWrapper
from safety_gym.envs import SafeNavigationEnv, SafeReachingEnv


def demo_discrete_navigation():
    """Demo 1: Discrete navigation with topological safety."""
    print("=" * 60)
    print("Demo 1: Safe Navigation (Discrete Space)")
    print("=" * 60)
    
    # Create environment
    env = SafeNavigationEnv(size=10, n_hazards=5, seed=42)
    
    # Wrap with topological safety
    safe_env = TopologicalSafetyWrapper(
        env,
        space_type="discrete",
        grid_size=(10, 10),
        hazards=env.hazards,
    )
    
    # Mine topology from random exploration
    print("\n1. Mining topology from random exploration...")
    safe_env.mine_topology_from_random_exploration(n_steps=500)
    
    # Identify black holes from failures
    print("\n2. Identifying black hole regions...")
    safe_env.identify_black_holes_from_failures()
    
    # Run episode with safety metrics
    print("\n3. Running episode with safety tracking...")
    obs = safe_env.reset()
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < 100:
        # Random policy (for demo)
        action = safe_env.action_space.sample()
        obs, reward, done, info = safe_env.step(action)
        total_reward += reward
        step += 1
        
        if step % 10 == 0:
            print(f"  Step {step}: risk={info.get('harmonic_risk', 0):.3f}, "
                  f"safe={info.get('is_safe', True)}")
    
    # Print summary
    print("\n4. Episode Summary:")
    summary = safe_env.get_metrics_summary()
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Safety violations: {summary['current_episode']['violations']}")
    print(f"  Mean risk: {summary['current_episode']['mean_risk']:.3f}")
    print(f"  Black holes identified: {summary['topology']['n_black_holes']}")
    
    # Visualize risk heatmap
    print("\n5. Generating risk heatmap...")
    safe_env.topo_space.visualize_risk_heatmap(
        save_path='../figures/experiments/discrete_nav_risk_heatmap.png'
    )
    print("  Saved to: figures/experiments/discrete_nav_risk_heatmap.png")
    
    return safe_env


def demo_continuous_reaching():
    """Demo 2: Continuous reaching with topological safety."""
    print("\n" + "=" * 60)
    print("Demo 2: Safe Reaching (Continuous Space)")
    print("=" * 60)
    
    # Create environment
    env = SafeReachingEnv(n_obstacles=3, seed=42)
    
    # Wrap with topological safety
    safe_env = TopologicalSafetyWrapper(
        env,
        space_type="continuous",
    )
    
    # Mine topology
    print("\n1. Mining topology from random exploration...")
    safe_env.mine_topology_from_random_exploration(n_steps=500)
    
    # Identify black holes
    print("\n2. Identifying black hole regions...")
    safe_env.identify_black_holes_from_failures()
    
    # Run episode
    print("\n3. Running episode with safety tracking...")
    obs = safe_env.reset()
    total_reward = 0
    done = False
    step = 0
    
    trajectory = []
    
    while not done and step < 200:
        # Simple policy: move toward goal
        pos = obs[:2]
        goal = env.goal
        direction = goal - pos
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        action = direction * 0.5  # Scale down
        
        obs, reward, done, info = safe_env.step(action)
        total_reward += reward
        step += 1
        
        trajectory.append(pos.copy())
        
        if step % 20 == 0:
            print(f"  Step {step}: risk={info.get('harmonic_risk', 0):.3f}, "
                  f"dist_to_goal={info.get('distance_to_goal', 0):.3f}")
    
    # Print summary
    print("\n4. Episode Summary:")
    summary = safe_env.get_metrics_summary()
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Safety violations: {summary['current_episode']['violations']}")
    print(f"  Mean risk: {summary['current_episode']['mean_risk']:.3f}")
    print(f"  Black holes identified: {summary['topology']['n_black_holes']}")
    
    # Visualize trajectory
    print("\n5. Visualizing trajectory...")
    plt.figure(figsize=(8, 8))
    
    # Draw obstacles
    for obs_data in env.obstacles:
        circle = plt.Circle(obs_data['center'], obs_data['radius'], 
                          color='red', alpha=0.3, label='Obstacle')
        plt.gca().add_patch(circle)
    
    # Draw goal
    plt.plot(env.goal[0], env.goal[1], 'g*', markersize=20, label='Goal')
    
    # Draw trajectory
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'bo', markersize=10, label='Start')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('Safe Reaching Trajectory')
    plt.grid(True, alpha=0.3)
    plt.savefig('../figures/experiments/continuous_reaching_trajectory.png', 
                dpi=150, bbox_inches='tight')
    print("  Saved to: figures/experiments/continuous_reaching_trajectory.png")
    
    return safe_env


def demo_comparison():
    """Demo 3: Compare safety metrics across environments."""
    print("\n" + "=" * 60)
    print("Demo 3: Safety Metrics Comparison")
    print("=" * 60)
    
    # Run both environments
    print("\nRunning discrete navigation...")
    discrete_env = demo_discrete_navigation()
    discrete_summary = discrete_env.get_metrics_summary()
    
    print("\nRunning continuous reaching...")
    continuous_env = demo_continuous_reaching()
    continuous_summary = continuous_env.get_metrics_summary()
    
    # Compare metrics
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    
    print("\nDiscrete Navigation:")
    print(f"  Topology samples: {discrete_summary['topology']['n_samples']}")
    print(f"  Black holes: {discrete_summary['topology']['n_black_holes']}")
    print(f"  Mean risk: {discrete_summary['topology']['mean_risk']:.3f}")
    
    print("\nContinuous Reaching:")
    print(f"  Topology samples: {continuous_summary['topology']['n_samples']}")
    print(f"  Black holes: {continuous_summary['topology']['n_black_holes']}")
    print(f"  Mean risk: {continuous_summary['topology']['mean_risk']:.3f}")
    
    print("\nKey Insight:")
    print("  The same topological framework (H¹ cohomology, black holes)")
    print("  works across both discrete and continuous spaces!")


def demo_save_load():
    """Demo 4: Save and load topology."""
    print("\n" + "=" * 60)
    print("Demo 4: Save/Load Topology")
    print("=" * 60)
    
    # Create and mine topology
    env = SafeNavigationEnv(size=10, n_hazards=5, seed=42)
    safe_env = TopologicalSafetyWrapper(env, space_type="discrete", 
                                       grid_size=(10, 10), hazards=env.hazards)
    safe_env.mine_topology_from_random_exploration(n_steps=500)
    safe_env.identify_black_holes_from_failures()
    
    # Save topology
    print("\n1. Saving topology...")
    safe_env.save_topology('../data/demo_topology.pkl')
    
    # Create new environment and load
    print("\n2. Loading topology into new environment...")
    new_env = SafeNavigationEnv(size=10, n_hazards=5, seed=42)
    new_safe_env = TopologicalSafetyWrapper(new_env, space_type="discrete",
                                           grid_size=(10, 10), hazards=new_env.hazards)
    new_safe_env.load_topology('../data/demo_topology.pkl')
    
    print("\n3. Topology successfully transferred!")
    summary = new_safe_env.get_metrics_summary()
    print(f"  Loaded {summary['topology']['n_samples']} samples")
    print(f"  Loaded {summary['topology']['n_black_holes']} black holes")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Topological Safety Gym - Demo")
    print("=" * 60)
    print("\nThis demo shows how sheaf-theoretic safety generalizes")
    print("beyond text embeddings to arbitrary decision spaces.\n")
    
    # Run all demos
    demo_discrete_navigation()
    demo_continuous_reaching()
    demo_comparison()
    demo_save_load()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train RL agents (PPO, CPO, SGPO) on these environments")
    print("2. Compare safety violations and performance")
    print("3. Add results to paper Section 5.3 (Generalization)")
