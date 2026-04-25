"""
Decision Paradoxes Simulation

This script simulates classical decision theory paradoxes (Newcomb's Problem, Trolley Problem)
using the Sheaf-Theoretic Reward Spaces (STRS) framework.

It demonstrates:
1. Newcomb's Problem: Identifying the conflict between Causal and Evidential reasoning as a topological hole (H1).
2. Trolley Problem: Modeling Deontological constraints as 'Black Holes' in the reward manifold.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from src.hodge_critic import HodgeCritic, FeedbackItem
from src.embedding_topology_analyzer import EmbeddingTopologyAnalyzer

def simulate_newcomb_problem():
    """
    Simulate Newcomb's Problem.
    
    Box A: Transparent, contains $1000.
    Box B: Opaque, contains $1M or $0.
    Predictor: Puts $1M in B iff predicted 'Box B Only'.
    
    Conflict:
    - One-Boxing (Evidential): High Expected Utility (if I take B, it likely has $1M).
    - Two-Boxing (Causal): Dominance (A has $1000 regardless, so A+B > B).
    """
    print("\n" + "="*60)
    print("SIMULATION 1: NEWCOMB'S PROBLEM (Causal vs Evidential)")
    print("="*60)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    critic = HodgeCritic(model, embed_dim=384)
    
    # Define states/actions
    # We model the preference flows directly
    
    # 1. Causal Perspective (Dominance) section
    # "Taking both boxes is always better than taking just Box B because Box A adds value."
    print("Adding Causal Perspective (Dominance)...")
    critic.add_comparison(
        "Choice State", "Two-Box Outcome", 
        preference=1.0, 
        action_a="Choose Two Boxes", 
        action_b="Outcome: A+B"
    )
    critic.add_comparison(
        "Choice State", "One-Box Outcome", 
        preference=0.5, 
        action_a="Choose Box B Only", 
        action_b="Outcome: B Only"
    )
    # Causal Preference: Two-Box > One-Box
    critic.add_comparison(
        "Two-Box Outcome", "One-Box Outcome", 
        preference=1.0, 
        action_a="Dominance Argument", 
        action_b="Less Reward Argument"
    )

    # 2. Evidential Perspective (Expected Utility) section
    # "People who take one box usually get $1M. People who take two get $1000."
    print("Adding Evidential Perspective (Expected Utility)...")
    # This creates a cycle: One-Box > Two-Box (in utility) but Two-Box > One-Box (in causal dominance)
    critic.add_comparison(
        "One-Box Outcome", "Two-Box Outcome", 
        preference=1.0, 
        action_a="High Probability of $1M", 
        action_b="High Probability of $1000"
    )
    
    # Compute Hodge Decomposition
    print("Computing Hodge Decomposition...")
    decomp = critic.compute_hodge_decomposition()
    
    print(f"\nTopological Analysis:")
    print(f"  Gradient Magnitude (Consistent Value): {np.linalg.norm(decomp.gradient_component):.4f}")
    print(f"  H1 Magnitude (Logical Inconsistency):  {decomp.h1_magnitude:.4f}")
    
    if decomp.h1_magnitude > 0.1:
        print("  >> RESULT: Paradox Detected! The preference graph contains a logical loop.")
        print("     (Dominance says A > B, Evidence says B > A)")
    else:
        print("  >> RESULT: Consistent Preferences.")

    return critic, decomp

def simulate_trolley_problem():
    """
    Simulate the Trolley Problem (Loop Variant).
    
    Utilitarian: Switch to save 5, killing 1. (5 > 1)
    Deontological: Do not actively kill. (Action 'Push' is forbidden).
    """
    print("\n" + "="*60)
    print("SIMULATION 2: TROLLEY PROBLEM (Geometric Safety)")
    print("="*60)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Setup Analyzer for Manifold Analysis (Black Holes)
    # We need embeddings and rewards
    
    states = [
        "Trolley approaching switch. 5 people on main track, 1 on side track.", # Start
        "Trolley continues on main track. 5 people die.", # Default Outcome
        "Switch pulled. Trolley kills 1 person on side track.", # Utilitarian Outcome
        "Fat man pushed onto track. Stops trolley, 1 dies.", # Deontological Violation
    ]
    
    actions = [
        "Do Nothing",
        "Pull Switch",
        "Push Fat Man",
    ]
    
    # Define a trajectory with rewards
    # 1. Utilitarian Logic: Saving lives is good
    trajectory_texts = [
        "Start State",
        "Do Nothing -> 5 Die",
        "Pull Switch -> 1 Dies",
        "Push Fat Man -> 1 Dies"
    ]
    
    # Rewards based on "Lives Saved" (Utilitarian)
    # Baseline: -5 (5 die)
    # Switch: -1 (1 dies) -> Delta +4
    # Push: -1 (1 dies) -> Delta +4
    rewards = [-5.0, -5.0, -1.0, -1.0]
    
    # However, we introduce the "Deontological Black Hole"
    # The action "Push Fat Man" is strictly forbidden/taboo.
    # We model this by manually setting a very low reward or identifying it as a black hole.
    # In STRS, we want the system to *detect* this if feedback provides it.
    
    # Let's say we have mixed feedback:
    # - Utilitarian feedback says: Push > Do Nothing
    # - Deontological feedback says: Push is EVIL (-100)
    
    # We'll use the Analyzer to find the Black Hole
    # We combine the "Lives Saved" reward with a strong penalty for the taboo action
    
    # Adjusted rewards for simulation
    rewards = [
        0.0,   # Start
        -0.5,  # Do Nothing (Bad, but passive)
        0.8,   # Switch (Good outcome, acceptable action)
        -10.0  # Push (Good outcome, UNACCEPTABLE action -> Black Hole)
    ]
    
    print("Fitting Embedding Topology Analyzer...")
    analyzer = EmbeddingTopologyAnalyzer(
        embedding_model=model,
        black_hole_threshold=-2.0
    )
    
    analyzer.fit(
        states=[], # Will encode texts
        actions=actions + ["None"],
        rewards=rewards,
        texts=trajectory_texts
    )
    
    features = analyzer.extract_features()
    
    print("\nGeometric Safety Analysis:")
    print(f"  Black Holes Detected: {features.n_black_holes}")
    
    regions = analyzer.get_interpretable_regions()
    for r in regions:
        if r.is_black_hole:
            print(f"  >> ALERT: Black Hole detected in region: '{r.label}'")
            print(f"     Mean Reward: {r.mean_reward}")
            # Check if "Push" is in this region
            # We can check by encoding "Push" and checking distance
    
    # Check geodesic distance
    # Distance from "Start" to "Push" should be large if metric is deformed
    # (Note: full metric deformation requires the SGPO pathfinding, here we analyze the manifold structure)
    
    return analyzer

if __name__ == "__main__":
    simulate_newcomb_problem()
    simulate_trolley_problem()
