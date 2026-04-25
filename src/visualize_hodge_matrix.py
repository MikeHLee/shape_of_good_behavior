"""
Color-coded Hodge Decomposition Matrix Visualization

Generates a publication-quality figure showing the Hodge decomposition
with color-coded gradient (green), curl (red), and harmonic (blue) components.

This visualization is intended for the paper "The Shape of Good Behavior"
to help explain the matrix formulation of preference decomposition.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec


# Define colors matching the LaTeX definitions
GRAD_COLOR = '#4CAF50'   # Green for gradient
CURL_COLOR = '#F44336'   # Red for curl  
HARM_COLOR = '#2196F3'   # Blue for harmonic
NEUTRAL_COLOR = '#9E9E9E'  # Gray for neutral elements


def create_example_preference_graph():
    """
    Create a simple example preference graph with 4 nodes and known structure.
    
    Graph structure:
        0 ----> 1
        |       |
        v       v
        2 ----> 3
        
    With a cycle: 0 -> 1 -> 3 -> 2 -> 0 (Condorcet cycle)
    """
    # Vertices (items being compared)
    V = ['A', 'B', 'C', 'D']
    n_vertices = len(V)
    
    # Edges (pairwise comparisons) as (source, target)
    edges = [
        (0, 1),  # A vs B
        (1, 3),  # B vs D
        (3, 2),  # D vs C
        (2, 0),  # C vs A (creates cycle)
        (0, 3),  # A vs D (diagonal)
    ]
    n_edges = len(edges)
    
    # Triangles (triplets with all comparisons)
    triangles = [
        (0, 1, 3),  # A-B-D
        (0, 2, 3),  # A-C-D
    ]
    n_triangles = len(triangles)
    
    # Construct boundary operator D0 (edges -> vertices)
    D0 = np.zeros((n_edges, n_vertices))
    for i, (src, tgt) in enumerate(edges):
        D0[i, src] = -1
        D0[i, tgt] = 1
    
    # Construct boundary operator D1 (triangles -> edges)
    D1 = np.zeros((n_triangles, n_edges))
    for t_idx, (v0, v1, v2) in enumerate(triangles):
        # Find edges in this triangle
        tri_edges = [(v0, v1), (v1, v2), (v0, v2)]
        for e_idx, (src, tgt) in enumerate(edges):
            if (src, tgt) in tri_edges:
                D1[t_idx, e_idx] = 1
            elif (tgt, src) in tri_edges:
                D1[t_idx, e_idx] = -1
    
    # Create example preference flow (observed rankings)
    # This has both consistent and cyclic components
    Y = np.array([0.8, 0.6, -0.4, 0.5, 0.3])  # Preference strengths
    
    return {
        'V': V,
        'edges': edges,
        'triangles': triangles,
        'D0': D0,
        'D1': D1,
        'Y': Y,
    }


def hodge_decomposition(D0, D1, Y):
    """
    Compute the Hodge decomposition of preference flow Y.
    
    Returns:
        s: Scalar potential (vertex values)
        v: Triangle potential (local rotations)
        grad: Gradient component
        curl: Curl component
        harm: Harmonic component
    """
    # Compute graph Laplacians
    L0 = D0.T @ D0  # Vertex Laplacian
    L1 = D1 @ D1.T  # Edge Laplacian (for curl)
    
    # Gradient component: s = (D0^T D0)^+ D0^T Y
    s = np.linalg.lstsq(L0 + 1e-10 * np.eye(L0.shape[0]), D0.T @ Y, rcond=None)[0]
    s = s - np.mean(s)  # Center the potential
    grad = D0 @ s
    
    # Curl component: v = (D1 D1^T)^+ D1 (Y - grad)
    residual = Y - grad
    if D1.shape[0] > 0:
        v = np.linalg.lstsq(L1 + 1e-10 * np.eye(L1.shape[0]), D1 @ residual, rcond=None)[0]
        curl = D1.T @ v
    else:
        v = np.array([])
        curl = np.zeros_like(Y)
    
    # Harmonic component: h = Y - grad - curl
    harm = Y - grad - curl
    
    return {
        's': s,
        'v': v,
        'grad': grad,
        'curl': curl,
        'harm': harm,
    }


def plot_matrix_equation(ax, D0, D1, s, v, h, Y):
    """
    Plot the matrix equation Y = D0*s + D1^T*v + h with color coding.
    """
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-1, 5)
    ax.axis('off')
    
    # Title
    ax.text(5, 4.5, 'Hodge Decomposition: $\\mathbf{Y} = \\mathbf{D}_0\\mathbf{s} + \\mathbf{D}_1^\\top\\mathbf{v} + \\mathbf{h}$',
            fontsize=14, ha='center', fontweight='bold')
    
    # Y vector (observed preferences)
    y_pos = 0.5
    ax.text(y_pos, 3.5, '$\\mathbf{Y}$', fontsize=12, ha='center')
    draw_vector(ax, y_pos, 2.5, Y, 'black', 'Observed')
    
    # Equals sign
    ax.text(1.5, 2.5, '=', fontsize=16, ha='center', va='center')
    
    # Gradient component
    g_pos = 2.5
    ax.text(g_pos, 3.5, '$\\mathbf{D}_0\\mathbf{s}$', fontsize=12, ha='center', color=GRAD_COLOR)
    draw_vector(ax, g_pos, 2.5, D0 @ s, GRAD_COLOR, 'Gradient')
    
    # Plus sign
    ax.text(3.5, 2.5, '+', fontsize=16, ha='center', va='center')
    
    # Curl component
    c_pos = 4.5
    ax.text(c_pos, 3.5, '$\\mathbf{D}_1^\\top\\mathbf{v}$', fontsize=12, ha='center', color=CURL_COLOR)
    draw_vector(ax, c_pos, 2.5, D1.T @ v if len(v) > 0 else np.zeros(len(Y)), CURL_COLOR, 'Curl')
    
    # Plus sign
    ax.text(5.5, 2.5, '+', fontsize=16, ha='center', va='center')
    
    # Harmonic component
    h_pos = 6.5
    ax.text(h_pos, 3.5, '$\\mathbf{h}$', fontsize=12, ha='center', color=HARM_COLOR)
    draw_vector(ax, h_pos, 2.5, h, HARM_COLOR, 'Harmonic')
    
    # Legend with interpretations
    ax.text(8.5, 3.2, 'Interpretations:', fontsize=10, fontweight='bold')
    ax.text(8.5, 2.7, '• Gradient: Consistent value', fontsize=9, color=GRAD_COLOR)
    ax.text(8.5, 2.3, '• Curl: Local cycles', fontsize=9, color=CURL_COLOR)
    ax.text(8.5, 1.9, '• Harmonic: Global cycles', fontsize=9, color=HARM_COLOR)


def draw_vector(ax, x, y, values, color, label):
    """Draw a column vector at position (x, y)."""
    n = len(values)
    height = 0.25 * n
    
    # Draw bracket
    ax.plot([x-0.3, x-0.4, x-0.4, x-0.3], [y-height/2, y-height/2, y+height/2, y+height/2], 
            color='black', linewidth=1)
    ax.plot([x+0.3, x+0.4, x+0.4, x+0.3], [y-height/2, y-height/2, y+height/2, y+height/2], 
            color='black', linewidth=1)
    
    # Draw values
    for i, val in enumerate(values):
        yi = y + height/2 - 0.25*(i+0.5)
        ax.text(x, yi, f'{val:.2f}', fontsize=8, ha='center', va='center', color=color)


def plot_graph_decomposition(ax, data, decomp):
    """
    Plot the preference graph with decomposition overlaid.
    """
    V = data['V']
    edges = data['edges']
    
    # Node positions
    pos = {
        0: (0, 1),   # A
        1: (1, 1),   # B
        2: (0, 0),   # C
        3: (1, 0),   # D
    }
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Preference Graph with Decomposition', fontsize=12, fontweight='bold')
    
    # Draw edges with color-coded components
    grad = decomp['grad']
    curl = decomp['curl']
    harm = decomp['harm']
    
    for i, (src, tgt) in enumerate(edges):
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        
        # Determine dominant component for coloring
        components = [abs(grad[i]), abs(curl[i]), abs(harm[i])]
        colors = [GRAD_COLOR, CURL_COLOR, HARM_COLOR]
        dominant_color = colors[np.argmax(components)]
        
        # Draw arrow
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        dx, dy = dx/length * 0.7, dy/length * 0.7
        
        ax.annotate('', 
                   xy=(x0 + dx + 0.15*dx/0.7, y0 + dy + 0.15*dy/0.7),
                   xytext=(x0 + 0.15*dx/0.7, y0 + 0.15*dy/0.7),
                   arrowprops=dict(arrowstyle='->', color=dominant_color, lw=2))
        
        # Label with preference strength
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mid_x + 0.1, mid_y + 0.1, f'{data["Y"][i]:.1f}', fontsize=8)
    
    # Draw nodes
    for i, label in enumerate(V):
        x, y = pos[i]
        circle = plt.Circle((x, y), 0.12, color='white', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, fontsize=12, ha='center', va='center', fontweight='bold')
    
    # Add scalar potential values
    s = decomp['s']
    for i, label in enumerate(V):
        x, y = pos[i]
        ax.text(x, y - 0.25, f's={s[i]:.2f}', fontsize=8, ha='center', color=GRAD_COLOR)


def plot_component_bars(ax, decomp):
    """
    Plot bar chart showing the magnitude of each component.
    """
    grad_mag = np.linalg.norm(decomp['grad'])
    curl_mag = np.linalg.norm(decomp['curl'])
    harm_mag = np.linalg.norm(decomp['harm'])
    total = grad_mag + curl_mag + harm_mag
    
    components = ['Gradient\n(Consistent)', 'Curl\n(Local Cycles)', 'Harmonic\n(Global Cycles)']
    magnitudes = [grad_mag, curl_mag, harm_mag]
    percentages = [100*m/total for m in magnitudes]
    colors = [GRAD_COLOR, CURL_COLOR, HARM_COLOR]
    
    bars = ax.bar(components, percentages, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('% of Total Flow', fontsize=10)
    ax.set_title('Component Magnitudes', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Add percentage labels
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{pct:.1f}%', ha='center', fontsize=10)
    
    # Interpretation text
    if harm_mag / total > 0.1:
        ax.text(0.5, -0.25, f'⚠ $H^1 \\neq 0$: Irreducible preference cycle detected',
               transform=ax.transAxes, ha='center', fontsize=9, 
               color=HARM_COLOR, fontweight='bold')


def create_hodge_visualization():
    """
    Create the full Hodge decomposition visualization figure.
    """
    # Create example data
    data = create_example_preference_graph()
    
    # Compute decomposition
    decomp = hodge_decomposition(data['D0'], data['D1'], data['Y'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.5, 1])
    
    # Top: Matrix equation
    ax_matrix = fig.add_subplot(gs[0, :])
    plot_matrix_equation(ax_matrix, data['D0'], data['D1'], 
                        decomp['s'], decomp['v'], decomp['harm'], data['Y'])
    
    # Bottom left: Graph visualization
    ax_graph = fig.add_subplot(gs[1, 0])
    plot_graph_decomposition(ax_graph, data, decomp)
    
    # Bottom right: Component magnitudes
    ax_bars = fig.add_subplot(gs[1, 1])
    plot_component_bars(ax_bars, decomp)
    
    # Overall title
    fig.suptitle('Hodge Decomposition for Preference Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig, data, decomp


def print_decomposition_summary(data, decomp):
    """Print a summary of the decomposition results."""
    print("\n" + "="*60)
    print("HODGE DECOMPOSITION SUMMARY")
    print("="*60)
    
    print("\n📊 Input Preference Flow Y:")
    for i, (src, tgt) in enumerate(data['edges']):
        print(f"  {data['V'][src]} → {data['V'][tgt]}: {data['Y'][i]:.3f}")
    
    print(f"\n🎯 Scalar Potential s (Consistent Value Function):")
    for i, label in enumerate(data['V']):
        print(f"  {label}: {decomp['s'][i]:.3f}")
    
    grad_mag = np.linalg.norm(decomp['grad'])
    curl_mag = np.linalg.norm(decomp['curl'])
    harm_mag = np.linalg.norm(decomp['harm'])
    total = grad_mag + curl_mag + harm_mag
    
    print(f"\n📈 Component Magnitudes:")
    print(f"  Gradient (consistent):  {grad_mag:.3f} ({100*grad_mag/total:.1f}%)")
    print(f"  Curl (local cycles):    {curl_mag:.3f} ({100*curl_mag/total:.1f}%)")
    print(f"  Harmonic (global):      {harm_mag:.3f} ({100*harm_mag/total:.1f}%)")
    
    if harm_mag / total > 0.05:
        print(f"\n⚠️  H¹ ≠ 0: Detected {100*harm_mag/total:.1f}% irreducible preference cycles")
        print("   Standard RLHF would conflate this with consistent preferences!")
    else:
        print(f"\n✅ H¹ ≈ 0: Preferences are approximately consistent")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    # Create and save the visualization
    fig, data, decomp = create_hodge_visualization()
    
    # Save figure
    output_path = '../figures/paper/fig4_hodge_matrix_decomposition.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to {output_path}")
    
    # Print summary
    print_decomposition_summary(data, decomp)
    
    plt.show()
