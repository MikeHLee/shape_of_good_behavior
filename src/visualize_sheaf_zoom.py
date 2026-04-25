"""
Sheaf Zoom Visualization

This script illustrates the Sheaf-Theoretic concept of "Zooming In" to resolve paradoxes.

It visualizes a Condorcet Cycle (A > B > C > A) in two ways:
1. Global Projection (The Flat Loop): Inconsistent, H1 != 0.
2. The Sheaf Space (The Helicoid): "Zooming in" reveals a local gradient. 
   The paradox is resolved by lifting the cycle to a covering space (adding Context/Dimension).

This demonstrates how local consistency (restriction maps) can coexist with global inconsistency.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_sheaf_zoom():
    print("Generating Sheaf Zoom Visualization...")
    
    fig = plt.figure(figsize=(16, 8))
    
    # 1. The Global Inconsistency (Flat Loop)
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Define a circle of preferences
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    z_flat = np.zeros_like(theta)
    
    ax1.plot(x, y, z_flat, 'k--', alpha=0.5)
    
    # Points A, B, C
    labels = ['A', 'B', 'C']
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    points_x = np.cos(angles)
    points_y = np.sin(angles)
    points_z = [0, 0, 0]
    
    ax1.scatter(points_x, points_y, points_z, c=['r', 'g', 'b'], s=200)
    
    for i, txt in enumerate(labels):
        ax1.text(points_x[i]*1.2, points_y[i]*1.2, 0, txt, fontsize=15, fontweight='bold')
        
    # Draw cyclic arrows
    for i in range(3):
        p1 = i
        p2 = (i + 1) % 3
        
        # Draw curved arrow between points
        # Start and end
        start = angles[p1]
        end = angles[p2] if angles[p2] > start else angles[p2] + 2*np.pi
        
        mid_angle = (start + end) / 2
        ax1.quiver(
            np.cos(mid_angle), np.sin(mid_angle), 0,
            -np.sin(mid_angle), np.cos(mid_angle), 0, # Tangent vector
            length=0.5, normalize=True, color='purple', linewidth=2
        )
        
    ax1.set_title("1. Global View (The Paradox)\nA > B > C > A\nNo consistent value function V(x)", fontsize=14)
    ax1.set_zlim(-1, 4)
    ax1.axis('off')
    
    # 2. The Sheaf Space (The Lift / "Zooming In")
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Draw the Helicoid (The Riemann Surface of the Logarithm / Value)
    z_lift = theta / (2*np.pi) * 3 # Height increases with angle
    
    # Plot the surface track
    ax2.plot(x, y, z_lift, 'b-', linewidth=2, label='Consistent Section (Local)')
    
    # Plot the ghost of the next cycle
    ax2.plot(x, y, z_lift + 3, 'b--', alpha=0.3, label='Next Sheet')
    
    # Points lifted
    lifted_z = [0, 1, 2] # Values: V(A)=0, V(B)=1, V(C)=2
    ax2.scatter(points_x, points_y, lifted_z, c=['r', 'g', 'b'], s=200)
    
    # The "Ghost" A' (A in next context)
    ax2.scatter(points_x[0], points_y[0], 3, c='r', s=200, alpha=0.5, edgecolor='black')
    ax2.text(points_x[0]*1.2, points_y[0]*1.2, 3, "A' (Context 2)", fontsize=12)
    
    for i, txt in enumerate(labels):
        ax2.text(points_x[i]*1.2, points_y[i]*1.2, lifted_z[i], f"{txt}\nV={lifted_z[i]}", fontsize=12)
        
    # Draw consistent gradient arrows (upward)
    for i in range(3):
        idx = int(angles[i] / (2*np.pi) * 100)
        ax2.quiver(
            points_x[i], points_y[i], lifted_z[i],
            0, 0, 1, # Upward flow
            length=0.5, color='green', linewidth=3
        )

    # Restriction Map Annotation
    ax2.text(0, 0, 4, "Sheaf Section s ∈ F(U)", ha='center', fontsize=12, color='blue')
    
    ax2.set_title("2. Sheaf View (Zooming In)\nResolving paradox by adding Context/Value\nLocally Consistent Gradient", fontsize=14)
    ax2.set_zlim(-1, 4)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("sheaf_zoom_viz.png")
    print("Visualization saved to 'sheaf_zoom_viz.png'")

if __name__ == "__main__":
    visualize_sheaf_zoom()
