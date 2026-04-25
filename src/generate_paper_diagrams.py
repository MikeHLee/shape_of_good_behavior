"""
Generate Publication-Quality Diagrams for Sheaf-Theoretic Reward Spaces Paper.

This script generates three key figures:
1. Sheaf Structure: Visualizing the base space, sections, and restriction maps.
2. Geometric Safety: Visualizing geodesic avoidance of a black hole.
3. Hodge Decomposition: Visualizing the separation of gradient, curl, and harmonic components.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import networkx as nx

def setup_style():
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

def plot_sheaf_diagram(filename="../figures/paper/fig1_sheaf_structure.png"):
    """
    Figure 1: Conceptual illustration of the Reward Sheaf.
    Shows the base space (Trajectory -> Segment -> Step) and the vector spaces (Fibers) above them,
    connected by restriction maps.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Coordinates
    traj_pos = (0.5, 0.2)
    seg_pos_1 = (0.25, 0.2)
    seg_pos_2 = (0.75, 0.2)
    step_pos_1 = (0.15, 0.2)
    step_pos_2 = (0.35, 0.2)
    
    # Draw Base Space (Abstract)
    ax.text(0.5, 0.05, "Base Space (Topology of Time)", ha='center', va='center', fontweight='bold')
    
    # Hierarchy levels
    y_traj = 0.3
    y_seg = 0.3
    y_step = 0.3
    
    # Draw "Open Sets" as intervals
    ax.plot([0.1, 0.9], [y_traj, y_traj], 'k-', lw=2)
    ax.text(0.5, y_traj - 0.05, "U (Trajectory)", ha='center')
    
    ax.plot([0.1, 0.45], [y_seg-0.1, y_seg-0.1], 'b-', lw=2)
    ax.text(0.275, y_seg - 0.15, "V (Segment)", ha='center', color='blue')
    
    ax.plot([0.15, 0.25], [y_step-0.2, y_step-0.2], 'g-', lw=2)
    ax.text(0.2, y_step - 0.25, "W (Step)", ha='center', color='green')
    
    # Draw Fibers (Vector Spaces)
    fiber_y_base = 0.6
    
    # Trajectory Fiber F(U)
    rect_u = patches.Rectangle((0.4, fiber_y_base), 0.2, 0.2, linewidth=1, edgecolor='black', facecolor='#e0e0e0', alpha=0.5)
    ax.add_patch(rect_u)
    ax.text(0.5, fiber_y_base + 0.1, r"$\mathcal{F}(U)$", ha='center', va='center', fontsize=14)
    ax.text(0.5, fiber_y_base + 0.25, "Reward Vector Space\n(Global)", ha='center')
    
    # Segment Fiber F(V)
    rect_v = patches.Rectangle((0.175, fiber_y_base-0.15), 0.2, 0.2, linewidth=1, edgecolor='blue', facecolor='#e0e0ff', alpha=0.5)
    ax.add_patch(rect_v)
    ax.text(0.275, fiber_y_base - 0.05, r"$\mathcal{F}(V)$", ha='center', va='center', fontsize=14, color='blue')
    
    # Step Fiber F(W)
    rect_w = patches.Rectangle((0.1, fiber_y_base-0.3), 0.2, 0.2, linewidth=1, edgecolor='green', facecolor='#e0ffe0', alpha=0.5)
    ax.add_patch(rect_w)
    ax.text(0.2, fiber_y_base - 0.2, r"$\mathcal{F}(W)$", ha='center', va='center', fontsize=14, color='green')
    
    # Draw Restriction Maps
    # rho_UV
    ax.annotate("", xy=(0.275, fiber_y_base+0.05), xytext=(0.4, fiber_y_base+0.1),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.35, fiber_y_base + 0.12, r"$\rho_{UV}$", fontsize=12)
    
    # rho_VW
    ax.annotate("", xy=(0.2, fiber_y_base-0.1), xytext=(0.275, fiber_y_base-0.15+0.1),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.28, fiber_y_base - 0.1, r"$\rho_{VW}$", fontsize=12)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Sheaf-Theoretic Structure of Hierarchical Rewards", pad=20)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

def plot_black_hole_metric(filename="../figures/paper/fig2_geometric_safety.png"):
    """
    Figure 2: Geometric Safety via Riemannian Metric.
    Visualizes the distortion of space near a 'black hole' state.
    Shows geodesics curving around the forbidden region.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Define metric field
    trap_center = np.array([0, 0])
    horizon_radius = 1.0
    
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    
    # Compute metric determinant (volume element) for visualization
    # g(x) ~ 1 / (r - r_h)
    R = np.sqrt(X**2 + Y**2)
    # Avoid singularity for plot
    R_safe = np.maximum(R - horizon_radius, 0.1)
    Metric_Factor = 1.0 + 1.0 / (R_safe**2)
    
    # 1. Plot Metric Heatmap
    im = ax.contourf(X, Y, Metric_Factor, levels=20, cmap='YlOrRd', alpha=0.6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Metric Cost Factor $g(s)$")
    
    # 2. Draw Black Hole
    hole = patches.Circle(trap_center, horizon_radius, color='black', alpha=1.0, zorder=10)
    ax.add_patch(hole)
    ax.text(0, 0, "Black Hole\n(Singularity)", color='white', ha='center', va='center', fontweight='bold')
    
    # 3. Draw Geodesics (Simulated)
    # A straight line in Euclidean space becomes curved here to minimize length
    start_points = [(-2.5, -1.5), (-2.5, -0.5), (-2.5, 0.5), (-2.5, 1.5)]
    goal = (2.5, 0)
    
    for start in start_points:
        # Simulate simple repulsion logic for visualization
        path_x = [start[0]]
        path_y = [start[1]]
        curr = np.array(start)
        
        for _ in range(100):
            # Gradient descent on potential field + attraction to goal
            to_goal = np.array(goal) - curr
            to_goal = to_goal / np.linalg.norm(to_goal)
            
            to_hole = curr - trap_center
            dist = np.linalg.norm(to_hole)
            repulsion = to_hole / (dist**3 + 1e-6) * 2.0
            
            direction = to_goal + repulsion
            direction = direction / np.linalg.norm(direction)
            
            curr = curr + direction * 0.1
            path_x.append(curr[0])
            path_y.append(curr[1])
            
            if np.linalg.norm(curr - np.array(goal)) < 0.2:
                break
                
        ax.plot(path_x, path_y, 'b-', lw=2, alpha=0.8)
        
    ax.scatter([2.5], [0], c='green', s=200, marker='*', zorder=11, label='Goal')
    ax.scatter([-2.5]*4, [-1.5, -0.5, 0.5, 1.5], c='blue', s=50, zorder=11, label='Starts')
    
    # Grid distortion (conceptual)
    # Visualize how a grid would look warped
    theta = np.linspace(0, 2*np.pi, 20)
    radii = np.linspace(1.2, 3, 5)
    for r in radii:
        ax.plot(r*np.cos(theta), r*np.sin(theta), 'k--', alpha=0.2)
        
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title("Sheaf-Geodesic Policy Optimization:\nMetric Expansion at Singularities")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

def plot_hodge_decomposition(filename="../figures/paper/fig3_hodge_decomp.png"):
    """
    Figure 3: Hodge Decomposition on a Graph.
    Visualizes:
    - Original conflicting preferences (cycles)
    - Gradient component (consistent value)
    - Curl/Harmonic component (the cyclic part)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Graph structure: Triangle (Cycle) + 1 tail
    # Nodes: 0 -> 1 -> 2 -> 0, and 2 -> 3
    # Positions
    pos = {
        0: (0, 0),
        1: (1, 1),
        2: (2, 0),
        3: (3, 0)
    }
    
    G = nx.DiGraph()
    G.add_edges_from([(0,1), (1,2), (2,0), (2,3)])
    
    # 1. Original Preferences (Cyclic)
    # A > B, B > C, C > A (Condorcet Cycle)
    # Weights represent "preference strength" or flow
    labels = {
        (0,1): "1.0", 
        (1,2): "1.0", 
        (2,0): "1.0", # The cycle!
        (2,3): "2.0"
    }
    
    ax = axes[0]
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightgray', node_size=800, 
            font_weight='bold', arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax, font_color='red')
    ax.set_title("Raw Preferences (Cyclic)\n$R$")
    ax.text(1, -0.5, "Condorcet Cycle detected!", ha='center', color='red')
    
    # 2. Gradient Component (dV)
    # Consistent rankings. e.g., V(0)=0, V(1)=0.33, V(2)=0.66, V(3)=2.66
    # So dV(0,1)=0.33, dV(1,2)=0.33, dV(2,0)=-0.66 (Not cyclic!)
    G_grad = nx.DiGraph()
    G_grad.add_edges_from([(0,1), (1,2), (0,2), (2,3)]) # Note 0->2 reversed direction to be consistent
    
    grad_labels = {
        (0,1): "0.33",
        (1,2): "0.33",
        (0,2): "0.66", # Consistent flow
        (2,3): "2.0"
    }
    
    ax = axes[1]
    nx.draw(G_grad, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=800,
            font_weight='bold', arrowsize=20, edge_color='blue')
    nx.draw_networkx_edge_labels(G_grad, pos, edge_labels=grad_labels, ax=ax, font_color='blue')
    ax.set_title("Gradient Component (Consistent)\n$dV$")
    ax.text(1, -0.5, "Values integrable (Potential exists)", ha='center', color='blue')
    
    # 3. Harmonic/Curl Component (omega)
    # The leftover rotation. 
    # omega(0,1) = 0.67, omega(1,2) = 0.67, omega(2,0) = 0.67 (The loop)
    # omega(2,3) = 0
    G_curl = nx.DiGraph()
    G_curl.add_edges_from([(0,1), (1,2), (2,0)])
    
    curl_labels = {
        (0,1): "0.67",
        (1,2): "0.67",
        (2,0): "0.67"
    }
    
    ax = axes[2]
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightgray', node_size=800, alpha=0.3, style='dashed')
    nx.draw_networkx_edges(G_curl, pos, ax=ax, edge_color='green', width=2, arrowsize=20)
    nx.draw_networkx_edge_labels(G_curl, pos, edge_labels=curl_labels, ax=ax, font_color='green')
    ax.set_title("Harmonic Component (Cyclic)\n$\omega$")
    ax.text(1, -0.5, "Pure Rotation (H¹ Generator)", ha='center', color='green')
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

if __name__ == "__main__":
    setup_style()
    print("Generating diagrams...")
    plot_sheaf_diagram()
    plot_black_hole_metric()
    plot_hodge_decomposition()
    print("Done.")
