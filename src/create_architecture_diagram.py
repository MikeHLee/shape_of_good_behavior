"""
Generate System Architecture Diagram (Figure 1) for the paper.
Shows the STRS framework: Reward Sheaf → Hodge Decomposition → SGPO
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors
c_input = '#E8F4FD'      # Light blue
c_sheaf = '#FFF3CD'      # Light yellow
c_hodge = '#D4EDDA'      # Light green
c_safety = '#F8D7DA'     # Light red
c_output = '#E2D5F1'     # Light purple
c_border = '#333333'

def add_box(ax, x, y, w, h, text, color, fontsize=10, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor=c_border, linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, wrap=True)

def add_arrow(ax, start, end, color='#555555', style='->'):
    arrow = FancyArrowPatch(start, end, arrowstyle=style, color=color,
                            mutation_scale=15, linewidth=2)
    ax.add_patch(arrow)

# Title
ax.text(7, 7.5, 'Sheaf-Theoretic Reward Spaces: System Architecture', 
        ha='center', va='center', fontsize=14, fontweight='bold')

# ============================================================================
# Layer 1: Inputs (left side)
# ============================================================================
ax.text(1.5, 6.5, 'Inputs', ha='center', fontsize=11, fontweight='bold', color='#0066CC')

add_box(ax, 0.3, 5.2, 2.4, 0.8, 'Human\nPreferences', c_input, fontsize=9)
add_box(ax, 0.3, 4.2, 2.4, 0.8, 'Trajectory\nData τ', c_input, fontsize=9)
add_box(ax, 0.3, 3.2, 2.4, 0.8, 'Safety\nSignals', c_input, fontsize=9)

# ============================================================================
# Layer 2: Reward Sheaf Construction
# ============================================================================
ax.text(5, 6.5, 'Reward Sheaf', ha='center', fontsize=11, fontweight='bold', color='#CC6600')

add_box(ax, 3.5, 4.8, 3, 1.2, 'Local Sections\nF(U) = {r: U → ℝᵈ}', c_sheaf, fontsize=9)
add_box(ax, 3.5, 3.3, 3, 1.2, 'Restriction Maps\nρ: F(U) → F(V)', c_sheaf, fontsize=9)

# Arrows from inputs to sheaf
add_arrow(ax, (2.7, 5.6), (3.5, 5.4))
add_arrow(ax, (2.7, 4.6), (3.5, 4.0))
add_arrow(ax, (2.7, 3.6), (3.5, 3.9))

# ============================================================================
# Layer 3: Topological Analysis (Hodge + Cohomology)
# ============================================================================
ax.text(9, 6.5, 'Topological Analysis', ha='center', fontsize=11, fontweight='bold', color='#006633')

# Hodge Decomposition box
add_box(ax, 7.2, 4.8, 3.6, 1.2, 'Hodge Decomposition\nr = dV + δα + ω', c_hodge, fontsize=9)

# Sub-components
add_box(ax, 7.2, 3.8, 1.1, 0.8, 'Exact\ndV', c_hodge, fontsize=8)
add_box(ax, 8.45, 3.8, 1.1, 0.8, 'Coexact\nδα', c_hodge, fontsize=8)
add_box(ax, 9.7, 3.8, 1.1, 0.8, 'Harmonic\nω', c_hodge, fontsize=8)

# H¹ Cohomology
add_box(ax, 7.2, 2.7, 3.6, 0.9, 'H¹ Cohomology\n(Cycle Detection)', c_hodge, fontsize=9)

# Arrows within topological analysis
add_arrow(ax, (6.5, 5.4), (7.2, 5.4))
add_arrow(ax, (6.5, 3.9), (7.2, 3.9))
add_arrow(ax, (9.0, 4.8), (9.0, 4.6))
add_arrow(ax, (9.0, 3.8), (9.0, 3.6))

# ============================================================================
# Layer 4: Safety Geometry
# ============================================================================
ax.text(5, 2.0, 'Safety Geometry', ha='center', fontsize=11, fontweight='bold', color='#990000')

add_box(ax, 3.5, 1.0, 3, 1.2, 'Riemannian Metric\ng(x) → ∞ at black holes', c_safety, fontsize=9)

# Arrow from restriction maps to safety
add_arrow(ax, (5.0, 3.3), (5.0, 2.2))

# ============================================================================
# Layer 5: SGPO Algorithm (output)
# ============================================================================
ax.text(12, 6.5, 'Output', ha='center', fontsize=11, fontweight='bold', color='#660099')

add_box(ax, 11.2, 4.0, 2.4, 2.0, 'SGPO\nPolicy\nπ(a|s)', c_output, fontsize=10, bold=True)

# Arrows to SGPO
add_arrow(ax, (10.8, 5.4), (11.2, 5.0))  # From Hodge
add_arrow(ax, (10.8, 3.2), (11.2, 4.5))  # From H¹
add_arrow(ax, (6.5, 1.6), (11.2, 4.2))   # From Safety (curved would be better but simplified)

# ============================================================================
# Key equations in boxes
# ============================================================================
# Value function box
ax.text(12.4, 2.5, 'Riemannian\nPolicy Gradient:', ha='center', fontsize=8, fontweight='bold')
ax.text(12.4, 1.8, r'$\nabla_\theta J = \mathbb{E}[\nabla \log\pi \cdot \frac{A}{g(s)}]$', 
        ha='center', fontsize=9, family='serif')

# Legend
legend_y = 0.5
ax.add_patch(FancyBboxPatch((0.3, legend_y), 0.4, 0.3, facecolor=c_input, edgecolor=c_border, linewidth=1))
ax.text(0.9, legend_y + 0.15, 'Input Data', fontsize=8, va='center')

ax.add_patch(FancyBboxPatch((2.5, legend_y), 0.4, 0.3, facecolor=c_sheaf, edgecolor=c_border, linewidth=1))
ax.text(3.1, legend_y + 0.15, 'Sheaf Structure', fontsize=8, va='center')

ax.add_patch(FancyBboxPatch((5.0, legend_y), 0.4, 0.3, facecolor=c_hodge, edgecolor=c_border, linewidth=1))
ax.text(5.6, legend_y + 0.15, 'Topology', fontsize=8, va='center')

ax.add_patch(FancyBboxPatch((7.0, legend_y), 0.4, 0.3, facecolor=c_safety, edgecolor=c_border, linewidth=1))
ax.text(7.6, legend_y + 0.15, 'Safety', fontsize=8, va='center')

ax.add_patch(FancyBboxPatch((8.8, legend_y), 0.4, 0.3, facecolor=c_output, edgecolor=c_border, linewidth=1))
ax.text(9.4, legend_y + 0.15, 'Policy', fontsize=8, va='center')

plt.tight_layout()
plt.savefig('figures/architecture_diagram.png', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('figures/architecture_diagram.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Architecture diagram saved to figures/architecture_diagram.png and .pdf")
