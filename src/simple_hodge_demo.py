
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path to allow importing src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hodge_critic import HodgeCritic, FeedbackItem
from sentence_transformers import SentenceTransformer

def run_demo():
    print("Initializing Hodge Critic Demo...")
    # Use a small, fast model for demo purposes
    model = SentenceTransformer('all-MiniLM-L6-v2')
    critic = HodgeCritic(model, embed_dim=384) # MiniLM has 384 dim

    # 1. Create a "Gradient" Scenario (Consistent Preferences)
    # A -> B -> C -> D
    print("\n1. Simulating Consistent Gradient (A -> B -> C -> D)")
    scenarios = [
        ("Start", "Go East", "Middle", 0.2),
        ("Middle", "Go East", "End", 0.5),
        ("End", "Go East", "Goal", 0.9),
    ]
    
    for s, a, n, r in scenarios:
        critic.add_feedback(FeedbackItem(s, a, n, r))
    
    decomp = critic.compute_hodge_decomposition()
    print(f"Gradient Energy (Scalar Norm): {np.linalg.norm(critic._gradient_field if hasattr(critic, '_gradient_field') else 0):.4f} (Approx)")
    # Note: The TopologicalGradient object stores the vector fields. 
    # But we want the scalar energy of the flow components Y_grad, Y_curl, Y_harm.
    # The hodge_critic class calculates h1_magnitude as norm(Y_harm).
    # Let's inspect the h1_magnitude.
    print(f"H1 Magnitude (Harmonic Energy): {decomp.h1_magnitude:.4f}")
    
    # 2. Create a "Curl" Scenario (Local Cycle)
    # Rock -> Paper -> Scissors -> Rock
    print("\n2. Simulating Local Curl (Rock -> Paper -> Scissors -> Rock)")
    critic = HodgeCritic(model, embed_dim=384)
    
    # Use direct comparisons to enforce non-transitive flow
    critic.add_comparison("Rock", "Scissors", 1.0, "Beat", "Lose") # Rock > Scissors
    critic.add_comparison("Scissors", "Paper", 1.0, "Beat", "Lose") # Scissors > Paper
    critic.add_comparison("Paper", "Rock", 1.0, "Beat", "Lose") # Paper > Rock
    # Add cross edges to form triangles (needed for Curl vs Harmonic)
    # R->S, S->P, P->R. 
    # To be "Curl" (local), we need filled triangles.
    # d1 takes triangles. 
    # If we add no other edges, it's a hole (Harmonic).
    # If we add R->P (Rock beats Paper?), no that breaks the cycle.
    # Rock < Paper. 
    # A true curl is a rotational field on a surface.
    
    decomp = critic.compute_hodge_decomposition()
    print(f"H1 Magnitude (Harmonic Energy): {decomp.h1_magnitude:.4f}")
    
    # 3. Create a "Harmonic" Scenario (Global Hole)
    # Escher Staircase: A -> B -> C -> D -> A
    print("\n3. Simulating Harmonic Hole (A -> B -> C -> D -> A)")
    critic = HodgeCritic(model, embed_dim=384)
    # Use direct comparisons with positive flow A->B->C->D->A
    critic.add_comparison("Floor 1", "Floor 2", 1.0)
    critic.add_comparison("Floor 2", "Floor 3", 1.0)
    critic.add_comparison("Floor 3", "Floor 4", 1.0)
    critic.add_comparison("Floor 4", "Floor 1", 1.0)
    
    decomp = critic.compute_hodge_decomposition()
    print(f"H1 Magnitude (Harmonic Energy): {decomp.h1_magnitude:.4f}")
    print("(Note: Without cross-edges to form triangles, this loop is a topological hole -> Harmonic)")

    print("\nDemo Complete.")

if __name__ == "__main__":
    run_demo()
