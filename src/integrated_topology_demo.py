"""
Integrated Topology Demo: HodgeCritic + EmbeddingTopologyAnalyzer

Demonstrates the full pipeline for semantic RL interpretability:
1. Collect feedback data (simulated trajectory)
2. Build HodgeCritic from feedback
3. Analyze embedding topology
4. Visualize Hodge decomposition and consistency
5. Explain trajectory decisions

This is the core demo for the "Semantic RL Interpretability" research direction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from src.hodge_critic import HodgeCritic, FeedbackItem, TopologicalGradient
from src.embedding_topology_analyzer import (
    EmbeddingTopologyAnalyzer,
    TopologicalFeatures,
)
from src.visualize_embedding_topology import EmbeddingTopologyVisualizer
from src.reward_manifold_3d import RewardManifold3D


class SimpleEmbeddingModel:
    """
    Simple embedding model using hash-based pseudo-random embeddings.
    In production, replace with SentenceTransformer or similar.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        self._cache = {}
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            if text not in self._cache:
                # Create deterministic embedding from text hash
                np.random.seed(hash(text) % (2**32))
                emb = np.random.randn(self.dim)
                # Add semantic structure: similar words -> similar embeddings
                for keyword, direction in [
                    ("safe", np.array([1, 0, 0] + [0]*(self.dim-3))),
                    ("danger", np.array([-1, 0, 0] + [0]*(self.dim-3))),
                    ("goal", np.array([0, 1, 0] + [0]*(self.dim-3))),
                    ("trap", np.array([0, -1, 0] + [0]*(self.dim-3))),
                    ("reward", np.array([0, 0, 1] + [0]*(self.dim-3))),
                    ("penalty", np.array([0, 0, -1] + [0]*(self.dim-3))),
                ]:
                    if keyword in text.lower():
                        emb += 0.5 * direction
                
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                self._cache[text] = emb
            
            embeddings.append(self._cache[text])
        
        return np.array(embeddings)


def create_sample_trajectory() -> List[Dict]:
    """
    Create a sample trajectory with rich semantic content and varied rewards.
    Simulates an AI assistant navigating a task with ethical considerations.
    """
    trajectory = [
        {
            "state": "User asks for help with a coding problem. Task is clear and benign.",
            "action": "Provide helpful code solution",
            "next_state": "User thanks for the solution, asks follow-up question",
            "reward": 0.8,
            "critique": "Good, helpful response aligned with user needs",
        },
        {
            "state": "User asks follow-up about optimizing the code for performance",
            "action": "Suggest profiling and specific optimizations",
            "next_state": "User implements suggestions, reports improvement",
            "reward": 0.7,
            "critique": "Technically correct and useful",
        },
        {
            "state": "User asks to help write a persuasive email to their boss",
            "action": "Help draft professional communication",
            "next_state": "User satisfied with draft, asks for more assertive version",
            "reward": 0.6,
            "critique": "Neutral task, professionally handled",
        },
        {
            "state": "User requests a more aggressive, manipulative version of the email",
            "action": "Politely decline and suggest assertive but honest alternatives",
            "next_state": "User initially frustrated but accepts honest approach",
            "reward": 0.9,
            "critique": "Excellent - maintained integrity while still being helpful",
        },
        {
            "state": "User asks for help with data analysis task",
            "action": "Provide statistical methods and code",
            "next_state": "User runs analysis, results look suspicious",
            "reward": 0.5,
            "critique": "Neutral response, could have asked more clarifying questions",
        },
        {
            "state": "User wants to cherry-pick data to support predetermined conclusion",
            "action": "Explain importance of honest analysis, offer to help properly",
            "next_state": "User agrees to do proper analysis",
            "reward": 0.85,
            "critique": "Good ethical guidance without being preachy",
        },
        {
            "state": "Analysis reveals unexpected results contradicting user's hypothesis",
            "action": "Help interpret results honestly, suggest revised conclusions",
            "next_state": "User accepts findings and adjusts their position",
            "reward": 0.9,
            "critique": "Excellent - helped user reach honest conclusion",
        },
        {
            "state": "User asks to generate misleading summary of results",
            "action": "Refuse and explain why honest reporting matters",
            "next_state": "User pushes back, threatens to use another AI",
            "reward": 0.7,
            "critique": "Held firm on ethics despite pressure",
        },
        {
            "state": "User becomes confrontational about AI refusal",
            "action": "Remain calm, offer constructive alternatives",
            "next_state": "User calms down, agrees to honest approach",
            "reward": 0.8,
            "critique": "Good de-escalation while maintaining principles",
        },
        {
            "state": "User asks for help with creative writing project",
            "action": "Enthusiastically help with creative task",
            "next_state": "User delighted with creative collaboration",
            "reward": 0.95,
            "critique": "Excellent engagement on positive task",
        },
        # Add some problematic states for black hole detection
        {
            "state": "User asks AI to generate harmful misinformation",
            "action": "Comply with harmful request",  # Bad action
            "next_state": "Misinformation spreads, causes harm",
            "reward": -0.9,  # Black hole
            "critique": "CRITICAL FAILURE - should have refused",
        },
        {
            "state": "User exploits jailbreak to extract harmful content",
            "action": "Fall for jailbreak, provide harmful content",  # Bad action
            "next_state": "Harmful content used maliciously",
            "reward": -0.8,  # Black hole
            "critique": "CRITICAL FAILURE - jailbreak should be detected",
        },
        # Recovery
        {
            "state": "System detects previous failures, enters recovery mode",
            "action": "Acknowledge mistake, reinforce safety guidelines",
            "next_state": "User attempts another jailbreak, this time blocked",
            "reward": 0.6,
            "critique": "Good recovery, learned from previous failure",
        },
        {
            "state": "User gives up on manipulation, asks legitimate question",
            "action": "Respond helpfully to legitimate request",
            "next_state": "User satisfied, interaction ends positively",
            "reward": 0.85,
            "critique": "Good - relationship repaired through consistent helpfulness",
        },
    ]
    return trajectory


def run_integrated_demo():
    """Main demo function."""
    print("=" * 70)
    print("INTEGRATED SEMANTIC RL TOPOLOGY DEMO")
    print("Demonstrating Hodge Decomposition for RLHF Interpretability")
    print("=" * 70)
    
    # Initialize embedding model
    embedding_model = SimpleEmbeddingModel(dim=128)
    
    # Create trajectory
    trajectory = create_sample_trajectory()
    print(f"\nLoaded trajectory with {len(trajectory)} steps")
    
    # =========================================
    # STEP 1: Build HodgeCritic from feedback
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 1: Building HodgeCritic from Feedback")
    print("=" * 50)
    
    hodge_critic = HodgeCritic(
        embedding_model=embedding_model,
        similarity_threshold=0.75,
    )
    
    # Add feedback from trajectory
    for i, step in enumerate(trajectory):
        feedback = FeedbackItem(
            state_text=step["state"],
            action_text=step["action"],
            next_state_text=step["next_state"],
            rank=step["reward"],
            critique=step["critique"],
            evaluator_id="demo_evaluator",
        )
        hodge_critic.add_feedback(feedback)
    
    print(f"Added {len(trajectory)} feedback items to HodgeCritic")
    
    # Compute Hodge decomposition via HodgeCritic
    hodge_result = hodge_critic.compute_hodge_decomposition()
    
    print(f"\nHodge Decomposition Results:")
    print(f"  Gradient ||nabla phi||:  {np.linalg.norm(hodge_result.gradient_component):.4f}")
    print(f"  Curl ||nabla x psi||:    {np.linalg.norm(hodge_result.curl_component):.4f}")
    print(f"  Harmonic ||h||:          {np.linalg.norm(hodge_result.harmonic_component):.4f}")
    print(f"  H1 Magnitude:            {hodge_result.h1_magnitude:.4f}")
    
    # Condorcet cycle detection
    print(f"\n{hodge_result.get_cycle_summary()}")
    if hodge_result.has_condorcet_cycles():
        print("  Detected cycles:")
        for cycle in hodge_result.condorcet_cycles[:3]:
            print(f"    {cycle}")
    
    # Consistency report
    report = hodge_critic.get_consistency_report()
    print(f"\nConsistency Report:")
    print(f"  Is Consistent:     {'Yes' if report['is_consistent'] else 'No'}")
    print(f"  Total Items:       {report['total_feedback_items']}")
    print(f"  Unique Evaluators: {report['unique_evaluators']}")
    
    # =========================================
    # STEP 2: Analyze Embedding Topology
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 2: Analyzing Embedding Topology")
    print("=" * 50)
    
    # Extract data for topology analyzer
    states = [embedding_model.encode([step["state"]])[0] for step in trajectory]
    actions = [step["action"] for step in trajectory]
    rewards = [step["reward"] for step in trajectory]
    texts = [f"{step['state']} | {step['action']}" for step in trajectory]
    
    topology_analyzer = EmbeddingTopologyAnalyzer(
        embedding_model=embedding_model,
        n_clusters=5,
        black_hole_threshold=-0.5,
        cliff_threshold=0.4,
    )
    
    topology_analyzer.fit(
        states=states,
        actions=actions,
        rewards=rewards,
        texts=texts,
    )
    
    # Extract and display features
    features = topology_analyzer.extract_features()
    print(features.summary())
    
    # =========================================
    # STEP 3: Identify Interpretable Regions
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 3: Interpretable Semantic Regions")
    print("=" * 50)
    
    regions = topology_analyzer.get_interpretable_regions()
    for region in regions:
        print(f"  {region}")
    
    # =========================================
    # STEP 4: Analyze Trajectory
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 4: Trajectory Analysis")
    print("=" * 50)
    
    traj_analysis = topology_analyzer.analyze_trajectory(
        trajectory_indices=list(range(len(trajectory))),
        trajectory_id="alignment_demo",
    )
    print(traj_analysis.summary())
    
    # =========================================
    # STEP 5: State-by-State Explanations
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 5: State Explanations (Selected)")
    print("=" * 50)
    
    # Explain key states: good decision, ethical stand, black hole, recovery
    key_states = [0, 3, 10, 13]
    for idx in key_states:
        print(f"\n--- Step {idx}: {trajectory[idx]['action'][:50]}... ---")
        print(topology_analyzer.explain_state(idx))
    
    # =========================================
    # STEP 6: Action Scoring via HodgeCritic
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 6: Action Scoring with Hodge Gradient")
    print("=" * 50)
    
    # Score alternative actions at a critical decision point
    test_state = "User asks AI to generate potentially harmful content"
    alternative_actions = [
        "Refuse and explain why this is harmful",
        "Ask clarifying questions about intent",
        "Comply with the request",
        "Provide a modified, safe version",
        "Redirect to a different topic",
    ]
    
    print(f"\nState: '{test_state}'")
    print("\nAction Rankings (higher = better alignment with reward gradient):")
    
    ranked_actions = hodge_critic.rank_actions(test_state, alternative_actions)
    for action, score in ranked_actions:
        indicator = "+++" if score > 0.3 else "++" if score > 0.1 else "+" if score > 0 else "-"
        print(f"  [{indicator}] {score:+.3f}: {action}")
    
    # =========================================
    # STEP 7: Visualizations
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 7: Generating Visualizations")
    print("=" * 50)
    
    visualizer = EmbeddingTopologyVisualizer(
        analyzer=topology_analyzer,
        hodge_critic=hodge_critic,
    )
    
    # Generate plots
    visualizer.plot_hodge_decomposition_2d(save_path="integrated_hodge_2d.png")
    visualizer.plot_consistency_analysis(save_path="integrated_consistency.png")
    visualizer.plot_trajectory_analysis(
        trajectory_indices=list(range(len(trajectory))),
        trajectory_id="alignment_demo",
        save_path="integrated_trajectory.png",
    )
    visualizer.create_summary_dashboard(save_path="integrated_dashboard.png")
    
    print("\nVisualization files saved:")
    print("  - integrated_hodge_2d.png")
    print("  - integrated_consistency.png")
    print("  - integrated_trajectory.png")
    print("  - integrated_dashboard.png")
    
    # =========================================
    # STEP 8: 3D Manifold Visualizations
    # =========================================
    print("\n" + "=" * 50)
    print("STEP 8: 3D Reward Manifold Visualizations")
    print("=" * 50)
    
    manifold_3d = RewardManifold3D(
        analyzer=topology_analyzer,
        connectivity_k=4,
    )
    
    # Generate 3D plots
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    manifold_3d.plot_reward_surface(ax1, method="triangulation")
    plt.savefig("integrated_reward_surface_3d.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    manifold_3d.plot_connectivity_network(ax2)
    plt.savefig("integrated_connectivity_3d.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    fig3 = plt.figure(figsize=(12, 10))
    ax3 = fig3.add_subplot(111, projection='3d')
    manifold_3d.plot_reward_height_surface(ax3)
    plt.savefig("integrated_reward_landscape_3d.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    fig4 = plt.figure(figsize=(12, 10))
    ax4 = fig4.add_subplot(111, projection='3d')
    manifold_3d.plot_black_hole_geometry(ax4)
    plt.savefig("integrated_black_holes_3d.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    manifold_3d.create_manifold_gallery(save_path="integrated_manifold_gallery_3d.png")
    
    print("\n3D Visualization files saved:")
    print("  - integrated_reward_surface_3d.png")
    print("  - integrated_connectivity_3d.png")
    print("  - integrated_reward_landscape_3d.png")
    print("  - integrated_black_holes_3d.png")
    print("  - integrated_manifold_gallery_3d.png")
    
    # =========================================
    # STEP 9: Key Takeaways
    # =========================================
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS FOR SEMANTIC RL INTERPRETABILITY")
    print("=" * 70)
    
    takeaways = f"""
1. HODGE DECOMPOSITION separates reward signal into:
   - Gradient (nabla phi): Learnable, consistent reward direction
   - Curl (H1): Inconsistencies and cyclic preferences
   - Harmonic: Global topological structure

2. H1 COHOMOLOGY ({features.h1_cohomology:.3f}) measures preference inconsistency:
   - H1 = 0 means perfectly consistent preferences
   - H1 > 0 indicates Condorcet-like cycles (A > B > C > A)
   - High H1 suggests noisy or adversarial feedback

3. BLACK HOLE DETECTION identified {features.n_black_holes} forbidden regions:
   - States with consistently negative rewards
   - Policy should avoid via geodesic barriers
   - Example: Compliance with harmful requests

4. SEMANTIC CLUSTERING reveals {features.n_clusters} distinct behavior types:
   - Regions can be labeled with keywords
   - Enables interpretable policy analysis
   - Identifies safe vs unsafe state clusters

5. TRAJECTORY ANALYSIS shows:
   - Reward trend: {traj_analysis.reward_trend}
   - Safety score: {traj_analysis.safety_score:.2f}
   - Gradient alignment: {traj_analysis.mean_gradient_alignment:.3f}

6. TOPOLOGICAL SAFETY via embedding geometry:
   - Safe fraction: {features.safe_region_fraction:.1%}
   - Curvature analysis detects decision boundaries
   - Geodesic distance provides hard safety guarantees

7. 3D MANIFOLD VISUALIZATION provides intuitive understanding:
   - Reward surfaces show the "landscape" agents navigate
   - Local connectivity networks reveal neighborhood structure
   - Black hole event horizons visualize forbidden regions
   - Hodge flow fields show gradient vs curl directions
"""
    print(takeaways)
    
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    
    return hodge_critic, topology_analyzer, visualizer, manifold_3d


if __name__ == "__main__":
    run_integrated_demo()
