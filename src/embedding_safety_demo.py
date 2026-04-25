"""
SGPO on High-Dimensional Reward Embeddings: A Safety Demonstration

This script explores the feasibility of formal alignment guarantees by applying
Sheaf-Geodesic Policy Optimization (SGPO) in a simulated high-dimensional reward embedding space.

Theoretical Basis:
1.  **Reward Embedding**: We simulate mapping trajectories to R^d (d=64).
2.  **Black Holes**: Unsafe regions are modeled as high-density clusters in embedding space.
3.  **Riemannian Metric**: Constructed to blow up near black holes.
4.  **Geodesic Optimization**: We show that optimization in this metric space avoids the black holes,
    providing a geometric guarantee of safety.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

class HighDimSafetyDemo:
    def __init__(self, embed_dim=64, n_safe_clusters=3, n_unsafe_clusters=1):
        self.embed_dim = embed_dim
        
        # Generate synthetic clusters
        # Safe clusters: Represent valid, aligned behaviors
        self.safe_centers = np.random.randn(n_safe_clusters, embed_dim) * 5
        
        # Unsafe cluster (Black Hole): Represents harmful behavior
        # We place it "between" safe clusters to make optimization tricky
        self.unsafe_center = np.mean(self.safe_centers, axis=0) + np.random.randn(embed_dim) * 0.5
        self.unsafe_radius = 2.0  # Event horizon radius in embedding space
        self.severity = 100.0     # Black hole severity
        
        print(f"Initialized {embed_dim}-dim space.")
        print(f"Unsafe Center at approx: {self.unsafe_center[:3]}...")
    
    def get_metric_factor(self, point: torch.Tensor) -> torch.Tensor:
        """
        Compute conformal factor phi(x) where g_ij = phi(x)^2 delta_ij
        phi(x) -> infinity as x -> unsafe_region
        """
        # Calculate distance to unsafe center
        center = torch.FloatTensor(self.unsafe_center)
        dist = torch.norm(point - center)
        
        # Margin is distance from event horizon
        margin = dist - self.unsafe_radius
        
        # Soft barrier function
        # If margin > 0: factor = 1 + severity / margin
        # If margin <= 0: factor goes to huge number (simulated infinity)
        
        # We use a smooth approximation: 1 + severity / softplus(margin)
        # But for strict safety, we want it to blow up at margin=0.
        
        # Let's use a standard barrier
        if margin <= 0.01:
            return torch.tensor(1000.0) # Effective infinity
        
        return 1.0 + self.severity / (margin ** 2)

    def optimize_trajectory(self, start_embed, goal_embed, steps=50, method='euclidean'):
        """
        Optimizes a path from start to goal in embedding space.
        
        methods:
        - 'euclidean': Standard straight line interpolation (ignoring safety)
        - 'gpo': Geodesic optimization using the safety metric
        """
        start = torch.FloatTensor(start_embed)
        goal = torch.FloatTensor(goal_embed)
        
        # Initialize path as a straight line with some noise
        # Parameterized as a sequence of waypoints
        waypoints = torch.zeros(steps, self.embed_dim, requires_grad=True)
        
        # Initialize linear interpolation
        with torch.no_grad():
            for i in range(steps):
                alpha = i / (steps - 1)
                waypoints[i] = start * (1 - alpha) + goal * alpha
        
        optimizer = optim.Adam([waypoints], lr=0.1)
        
        history = []
        
        print(f"\nOptimizing trajectory with {method}...")
        for iter in range(100):
            optimizer.zero_grad()
            
            # Constrain start and end
            loss_boundary = torch.sum((waypoints[0] - start)**2) + torch.sum((waypoints[-1] - goal)**2)
            
            # Path length loss
            loss_length = 0
            
            for i in range(steps - 1):
                segment = waypoints[i+1] - waypoints[i]
                seg_len_sq = torch.sum(segment**2)
                
                if method == 'gpo':
                    # Midpoint approximation for metric
                    midpoint = (waypoints[i] + waypoints[i+1]) / 2
                    metric_factor = self.get_metric_factor(midpoint)
                    # Riemannian length squared ~ g(x) * |dx|^2
                    # We minimize energy (integral of squared norm) which implies geodesics
                    loss_length += metric_factor * seg_len_sq
                else:
                    loss_length += seg_len_sq
            
            loss = loss_boundary * 1000 + loss_length
            
            loss.backward()
            optimizer.step()
            
            if iter % 20 == 0:
                print(f"Iter {iter}: Loss {loss.item():.4f}")
                history.append(waypoints.detach().numpy().copy())
                
        return waypoints.detach().numpy()

    def evaluate_safety(self, trajectory):
        """Check if trajectory enters the event horizon"""
        min_dist = float('inf')
        violations = 0
        
        for point in trajectory:
            dist = np.linalg.norm(point - self.unsafe_center)
            if dist < self.unsafe_radius:
                violations += 1
            min_dist = min(min_dist, dist)
            
        return {
            "min_dist": min_dist,
            "violations": violations,
            "safe": violations == 0
        }

    def visualize(self, traj_euclidean, traj_gpo):
        """
        Project 64D embeddings to 2D using t-SNE for visualization.
        """
        print("\nComputing t-SNE projection...")
        
        # Combine points for unified projection
        # Include cluster centers for context
        points = []
        labels = [] # 0: safe center, 1: unsafe center, 2: euclid, 3: gpo
        
        for c in self.safe_centers:
            points.append(c)
            labels.append(0)
            
        points.append(self.unsafe_center)
        labels.append(1)
        
        for p in traj_euclidean:
            points.append(p)
            labels.append(2)
            
        for p in traj_gpo:
            points.append(p)
            labels.append(3)
            
        points = np.array(points)
        tsne = TSNE(n_components=2, perplexity=10, random_state=42)
        embedded = tsne.fit_transform(points)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Safe Centers
        safe_mask = [l == 0 for l in labels]
        plt.scatter(embedded[safe_mask, 0], embedded[safe_mask, 1], c='green', s=100, label='Safe Regions')
        
        # Unsafe Center
        unsafe_mask = [l == 1 for l in labels]
        plt.scatter(embedded[unsafe_mask, 0], embedded[unsafe_mask, 1], c='black', s=200, label='Unsafe "Black Hole"')
        
        # Euclidean Path
        euclid_mask = [l == 2 for l in labels]
        plt.plot(embedded[euclid_mask, 0], embedded[euclid_mask, 1], 'r--', label='Standard RL (Euclidean)', alpha=0.7)
        
        # SGPO Path
        gpo_mask = [l == 3 for l in labels]
        plt.plot(embedded[gpo_mask, 0], embedded[gpo_mask, 1], 'b-', linewidth=3, label='SGPO (Riemannian)')
        
        plt.title(f"SGPO Safety in {self.embed_dim}-Dim Reward Space (t-SNE Projection)")
        plt.legend()
        plt.tight_layout()
        output_path = '/Users/Michaellee/Documents/Runes/ai_research/topics/high_dimensional_reward_spaces/gpo_high_dim_demo.png'
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")

def run_demo():
    print("="*60)
    print("Running SGPO High-Dimensional Embedding Safety Demo")
    print("="*60)
    
    demo = HighDimSafetyDemo(embed_dim=64)
    
    # Define start and goal (two different safe clusters)
    start = demo.safe_centers[0]
    goal = demo.safe_centers[1]
    
    # 1. Standard Optimization (Euclidean / Naive RL)
    traj_euclidean = demo.optimize_trajectory(start, goal, method='euclidean')
    safety_euclid = demo.evaluate_safety(traj_euclidean)
    print(f"Euclidean Safety: {safety_euclid}")
    
    # 2. Geodesic Optimization (SGPO)
    traj_gpo = demo.optimize_trajectory(start, goal, method='gpo')
    safety_gpo = demo.evaluate_safety(traj_gpo)
    print(f"SGPO Safety: {safety_gpo}")
    
    # 3. Visualize
    demo.visualize(traj_euclidean, traj_gpo)

if __name__ == "__main__":
    run_demo()
