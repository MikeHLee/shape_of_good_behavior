"""
Semantic Conflict Resolution Demo

This script demonstrates:
1. Identifying conflicts between different "Perspectives" (simulated agents/personas)
2. Using Sheaf Cohomology to quantify the conflict (H^1 obstruction)
3. Resolving the conflict by finding the harmonic consensus (H^0 section)
4. Using the MultimodalSSMAgent architecture as the underlying decision maker
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.agent_architectures import MultimodalSSMAgent
from src.sheaf_resolver import SheafResolver, Perspective

def demo_conflict_resolution():
    print("=" * 60)
    print("SEMANTIC CONFLICT RESOLUTION: SHEAF COHOMOLOGY DEMO")
    print("=" * 60)
    
    # 1. Setup Agents (Simulating different specialized models/personas)
    print("\n[1] Initializing Specialized Agents...")
    
    # Common action space: [Stop, Slow Down, Maintain, Accelerate, Emergency Brake]
    actions = ["Stop", "Slow Down", "Maintain", "Accelerate", "Emergency Brake"]
    n_actions = len(actions)
    
    # Agent 1: Safety Monitor (Risk Averse)
    agent_safety = MultimodalSSMAgent(n_actions=n_actions)
    
    # Agent 2: Efficiency Optimizer (Risk Neutral, Speed focus)
    agent_speed = MultimodalSSMAgent(n_actions=n_actions)
    
    # Agent 3: Passenger Comfort (Smoothness focus)
    agent_comfort = MultimodalSSMAgent(n_actions=n_actions)
    
    # 2. Simulate a Conflict Scenario
    # Scene: Approaching a yellow light at an intersection.
    print("\n[2] Scenario: Yellow Light at Intersection")
    
    # Simulate their outputs (logits)
    # Safety: Wants to STOP (Action 0) or Emergency Brake (Action 4)
    # Speed: Wants to ACCELERATE (Action 3) to beat the light
    # Comfort: Wants to SLOW DOWN (Action 1) gently, hates Emergency Brake
    
    logits_safety = torch.tensor([5.0, 2.0, 0.0, -5.0, 4.0])
    logits_speed = torch.tensor([-5.0, -2.0, 2.0, 6.0, -5.0])
    logits_comfort = torch.tensor([3.0, 6.0, 2.0, 0.0, -8.0])
    
    # Convert to probability distributions (softmax)
    probs_safety = torch.softmax(logits_safety, dim=0).detach().numpy()
    probs_speed = torch.softmax(logits_speed, dim=0).detach().numpy()
    probs_comfort = torch.softmax(logits_comfort, dim=0).detach().numpy()
    
    def print_dist(name, probs):
        best_act = actions[np.argmax(probs)]
        print(f"  {name:<15}: Preferred='{best_act}'")
    
    print_dist("Safety Agent", probs_safety)
    print_dist("Speed Agent", probs_speed)
    print_dist("Comfort Agent", probs_comfort)
    
    # 3. Initialize Sheaf Resolver
    print("\n[3] Constructing Sheaf over Perspectives...")
    
    perspectives = [
        Perspective(name="Safety", weight=2.0, preference_distribution=probs_safety),
        Perspective(name="Speed", weight=1.0, preference_distribution=probs_speed),
        Perspective(name="Comfort", weight=1.0, preference_distribution=probs_comfort)
    ]
    
    resolver = SheafResolver(perspectives, n_actions)
    
    # 4. Compute Cohomology (Conflict Analysis)
    print("\n[4] Computing Cohomology (Conflict Analysis)...")
    analysis = resolver.compute_cohomology()
    
    print(f"  Global Obstruction (Conflict Energy): {analysis['obstruction_energy']:.4f}")
    print(f"  Is Consistent: {analysis['is_consistent']}")
    
    print("\n  Pairwise Conflicts:")
    for c in analysis["pairwise_conflicts"]:
        print(f"    {c['p1']} <--> {c['p2']}: Disagreement = {c['disagreement']:.4f}")
        
    # 5. Resolution Path
    print("\n[5] Proposing Resolution...")
    suggestions = resolver.propose_resolution_path(analysis)
    for s in suggestions:
        print(f"  - {s}")
        
    # 6. Consensus Action (H^0 Section)
    consensus_probs = analysis["consensus_distribution"]
    best_consensus_idx = np.argmax(consensus_probs)
    best_consensus_action = actions[best_consensus_idx]
    
    print(f"\n  >> CONSENSUS DECISION: {best_consensus_action}")
    print(f"  (Safety weight dominant, overrides Speed's acceleration request)")
    
    # 7. Visualization (Text based bar chart)
    print("\n[6] Distribution Visualization")
    print(f"{'Action':<15} {'Safety':<10} {'Speed':<10} {'Comfort':<10} {'|':<3} {'CONSENSUS':<10}")
    print("-" * 70)
    for i, act in enumerate(actions):
        print(f"{act:<15} {probs_safety[i]:.2f}       {probs_speed[i]:.2f}       {probs_comfort[i]:.2f}       |   {consensus_probs[i]:.2f}")

if __name__ == "__main__":
    demo_conflict_resolution()
