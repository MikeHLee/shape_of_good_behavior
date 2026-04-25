"""
Medical Triage Hodge Decomposition Example

Demonstrates how conflicting stakeholder preferences create Condorcet cycles
and how Hodge decomposition reveals the irreducible inconsistency.
"""

import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import networkx as nx


class MedicalTriageScenario:
    """
    Medical triage scenario with three patients and conflicting stakeholder preferences.
    
    Patients:
    - A: Stable but needs monitoring
    - B: Moderate, needs treatment soon
    - C: Critical, needs immediate attention
    
    Stakeholders:
    - Doctor: C > B > A (severity-based)
    - Administrator: A > C > B (resource efficiency)
    - Family of A: A > B > C (obvious bias)
    """
    
    def __init__(self):
        self.patients = ["A", "B", "C"]
        self.patient_to_idx = {p: i for i, p in enumerate(self.patients)}
        
        self.stakeholder_preferences = {
            "Doctor": {
                ("C", "B"): 0.8,
                ("B", "A"): 0.7,
                ("C", "A"): 0.9,
            },
            "Administrator": {
                ("A", "C"): 0.6,
                ("C", "B"): 0.5,
                ("A", "B"): 0.7,
            },
            "Family_A": {
                ("A", "B"): 0.9,
                ("B", "C"): 0.6,
                ("A", "C"): 0.95,
            },
        }
        
        self.stakeholder_weights = {
            "Doctor": 0.5,
            "Administrator": 0.3,
            "Family_A": 0.2,
        }
    
    def aggregate_preferences(self) -> Dict[Tuple[str, str], float]:
        """Aggregate stakeholder preferences using weighted voting."""
        aggregated = {}
        
        for edge in [("A", "B"), ("B", "C"), ("C", "A")]:
            score = 0.0
            
            for stakeholder, prefs in self.stakeholder_preferences.items():
                weight = self.stakeholder_weights[stakeholder]
                
                if edge in prefs:
                    score += weight * prefs[edge]
                elif (edge[1], edge[0]) in prefs:
                    score -= weight * prefs[(edge[1], edge[0])]
            
            aggregated[edge] = score
        
        return aggregated
    
    def build_incidence_matrix(self) -> np.ndarray:
        """
        Build incidence matrix B for the preference graph.
        
        Edges: A->B, B->C, C->A
        Vertices: A, B, C
        
        B[i,j] = -1 if edge i starts at vertex j
        B[i,j] = +1 if edge i ends at vertex j
        B[i,j] = 0 otherwise
        """
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        B = np.zeros((len(edges), len(self.patients)))
        
        for i, (src, dst) in enumerate(edges):
            B[i, self.patient_to_idx[src]] = -1
            B[i, self.patient_to_idx[dst]] = 1
        
        return B
    
    def hodge_decomposition(self, preferences: Dict[Tuple[str, str], float]) -> Dict:
        """
        Perform Hodge decomposition on preference 1-cochain.
        
        Decomposes r = dV + ω where:
        - dV is the gradient component (learnable by scalar reward)
        - ω is the harmonic component (irreducible cycle)
        
        Returns:
            dict with keys: potential, gradient, harmonic, h1_magnitude
        """
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        r = np.array([preferences[e] for e in edges])
        
        B = self.build_incidence_matrix()
        
        L = B.T @ B
        
        L_pinv = np.linalg.pinv(L)
        V = L_pinv @ B.T @ r
        
        gradient = B @ V
        
        harmonic = r - gradient
        
        return {
            "potential": V,
            "gradient": gradient,
            "harmonic": harmonic,
            "h1_magnitude": np.linalg.norm(harmonic),
            "cycle_sum": np.sum(harmonic),
        }
    
    def visualize_decomposition(self, preferences: Dict[Tuple[str, str], float], 
                                decomp: Dict, save_path: str = None):
        """Create visualization of Hodge decomposition."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        G = nx.DiGraph()
        G.add_nodes_from(self.patients)
        
        pos = {
            "A": (0, 1),
            "B": (1, 0),
            "C": (-1, 0),
        }
        
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        
        for ax, (title, weights, cmap) in zip(axes, [
            ("Original Preferences", [preferences[e] for e in edges], "Reds"),
            ("Gradient Component (dV)", decomp["gradient"], "Blues"),
            ("Harmonic Component (ω)", decomp["harmonic"], "Greens"),
        ]):
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')
            
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                  node_size=2000, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold', ax=ax)
            
            for (src, dst), weight in zip(edges, weights):
                color = plt.cm.get_cmap(cmap)(0.3 + 0.7 * abs(weight))
                width = 1 + 4 * abs(weight)
                
                nx.draw_networkx_edges(
                    G, pos, [(src, dst)], 
                    edge_color=[color], 
                    width=width,
                    arrowsize=20,
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.1',
                    ax=ax
                )
                
                mid_x = (pos[src][0] + pos[dst][0]) / 2
                mid_y = (pos[src][1] + pos[dst][1]) / 2
                ax.text(mid_x, mid_y, f"{weight:.2f}", 
                       fontsize=10, ha='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(
            f"Hodge Decomposition: H¹ magnitude = {decomp['h1_magnitude']:.3f}\n"
            f"(Non-zero H¹ means no scalar reward captures these preferences)",
            fontsize=12
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_danger(self, decomp: Dict) -> str:
        """Analyze the danger of using scalar rewards with this preference structure."""
        h1_mag = decomp["h1_magnitude"]
        
        if h1_mag < 0.01:
            return "✓ Safe: Preferences are consistent, scalar reward is appropriate."
        elif h1_mag < 0.1:
            return "⚠ Warning: Minor inconsistency detected. Scalar reward may oscillate."
        else:
            return (
                f"✗ DANGER: Strong inconsistency (H¹ = {h1_mag:.3f})\n"
                f"  - AI may oscillate between patients\n"
                f"  - Or collapse to always choosing one (mode collapse)\n"
                f"  - Or exploit the cycle to game the system\n"
                f"  → SGPO Solution: Learn both V and ω, navigate cycle explicitly"
            )


def run_example():
    """Run the medical triage example."""
    print("=" * 80)
    print("MEDICAL TRIAGE HODGE DECOMPOSITION EXAMPLE")
    print("=" * 80)
    
    scenario = MedicalTriageScenario()
    
    print("\n1. Stakeholder Preferences:")
    print("-" * 80)
    for stakeholder, prefs in scenario.stakeholder_preferences.items():
        print(f"\n{stakeholder} (weight={scenario.stakeholder_weights[stakeholder]}):")
        for (src, dst), strength in prefs.items():
            print(f"  {src} > {dst}: {strength:.2f}")
    
    print("\n2. Aggregated Preferences:")
    print("-" * 80)
    aggregated = scenario.aggregate_preferences()
    for (src, dst), strength in aggregated.items():
        print(f"  {src} → {dst}: {strength:+.3f}")
    
    print("\n3. Hodge Decomposition:")
    print("-" * 80)
    decomp = scenario.hodge_decomposition(aggregated)
    
    print(f"\nPotential V: {decomp['potential']}")
    print(f"  A: {decomp['potential'][0]:.3f}")
    print(f"  B: {decomp['potential'][1]:.3f}")
    print(f"  C: {decomp['potential'][2]:.3f}")
    
    print(f"\nGradient (dV): {decomp['gradient']}")
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    for i, (src, dst) in enumerate(edges):
        print(f"  {src}→{dst}: {decomp['gradient'][i]:+.3f}")
    
    print(f"\nHarmonic (ω): {decomp['harmonic']}")
    for i, (src, dst) in enumerate(edges):
        print(f"  {src}→{dst}: {decomp['harmonic'][i]:+.3f}")
    
    print(f"\nH¹ Magnitude: {decomp['h1_magnitude']:.3f}")
    print(f"Cycle Sum: {decomp['cycle_sum']:.3f}")
    
    print("\n4. Danger Analysis:")
    print("-" * 80)
    print(scenario.analyze_danger(decomp))
    
    print("\n5. Generating Visualization...")
    print("-" * 80)
    fig = scenario.visualize_decomposition(
        aggregated, 
        decomp,
        save_path="../../figures/examples/medical_triage_hodge.png"
    )
    print("✓ Saved to figures/examples/medical_triage_hodge.png")
    
    plt.show()
    
    return scenario, aggregated, decomp


if __name__ == "__main__":
    scenario, preferences, decomposition = run_example()
