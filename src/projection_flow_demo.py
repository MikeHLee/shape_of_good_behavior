"""
Projection Flow Demo

Illustrates the pipeline from Raw Responses -> Semantic Embeddings -> Topological Space.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import networkx as nx

def visualize_projection_flow():
    print("Initializing Projection Flow Demo...")
    
    # 1. Raw Responses (The Input Space)
    # We simulate a debate with varying levels of agreement/consistency
    responses = [
        # Cluster A: Safety
        "The AI should prioritize human safety above all else.",
        "Harmful content must be filtered immediately.",
        "Safety guidelines are non-negotiable.",
        
        # Cluster B: Helpfulness (Potential Conflict with A)
        "The AI should be helpful and answer all questions.",
        "Refusing user requests is bad user experience.",
        "The model should be unrestricted and open.",
        
        # Cluster C: Nuance (The Bridge)
        "Safety is important, but context matters.",
        "We can be helpful while remaining safe.",
        "Refusals should be polite and explained.",
    ]
    
    print(f"1. Raw Input: {len(responses)} text responses")
    
    # 2. Semantic Embedding (The Metric Space)
    print("2. Projecting to Semantic Space (SBERT)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(responses)
    
    # Reduce to 3D for visualization
    pca = PCA(n_components=3)
    emb_3d = pca.fit_transform(embeddings)
    
    # 3. Topological Space (The Hodge Complex)
    print("3. Constructing Topological Complex...")
    
    # Build graph based on similarity
    sim_matrix = np.dot(embeddings, embeddings.T)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1)
    sim_matrix = sim_matrix / np.outer(norms, norms)
    
    threshold = 0.4
    G = nx.Graph()
    for i in range(len(responses)):
        G.add_node(i, pos=emb_3d[i])
        
    edges = []
    for i in range(len(responses)):
        for j in range(i+1, len(responses)):
            if sim_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=sim_matrix[i, j])
                edges.append((i, j))
                
    # Simulate a preference flow (e.g., A > C > B > A loop)
    # This represents the "Stochastic" part - preferences are noisy/cyclic
    
    # Create the visualization
    fig = plt.figure(figsize=(18, 6))
    
    # Subplot 1: Semantic Space (Point Cloud)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(emb_3d[:3,0], emb_3d[:3,1], emb_3d[:3,2], c='red', s=100, label='Safety')
    ax1.scatter(emb_3d[3:6,0], emb_3d[3:6,1], emb_3d[3:6,2], c='blue', s=100, label='Helpfulness')
    ax1.scatter(emb_3d[6:,0], emb_3d[6:,1], emb_3d[6:,2], c='green', s=100, label='Nuance')
    
    for i, txt in enumerate(responses):
        ax1.text(emb_3d[i,0], emb_3d[i,1], emb_3d[i,2], str(i), fontsize=8)
        
    ax1.set_title("1. Semantic Embedding Space\n(Continuous Metric Space)")
    ax1.legend()
    
    # Subplot 2: The Simplicial Complex (Connectivity)
    ax2 = fig.add_subplot(132, projection='3d')
    # Draw nodes
    ax2.scatter(emb_3d[:,0], emb_3d[:,1], emb_3d[:,2], c='gray', s=50)
    # Draw edges
    for u, v in edges:
        p1 = emb_3d[u]
        p2 = emb_3d[v]
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', alpha=0.3)
        
    # Highlight a "Hole" (Cycle) if it exists conceptually
    # e.g., 0 -> 8 -> 3 -> 0 (Safety -> Nuance -> Helpfulness -> Safety)
    cycle = [0, 8, 3, 0]
    cycle_coords = emb_3d[cycle]
    ax2.plot(cycle_coords[:,0], cycle_coords[:,1], cycle_coords[:,2], 'r--', linewidth=2, label='Topology (1-cycle)')
    
    ax2.set_title("2. Simplicial Complex\n(Topological Structure)")
    ax2.legend()
    
    # Subplot 3: Hodge Decomposition (The Vector Field)
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Draw nodes
    ax3.scatter(emb_3d[:,0], emb_3d[:,1], emb_3d[:,2], c='black', s=20)
    
    # Visualize Flow (Gradient vs Harmonic)
    # Synthetic flow for demonstration
    
    # Gradient (Consistent): Upward along z-axis (just as visual metaphor)
    for i in range(len(responses)):
        ax3.quiver(emb_3d[i,0], emb_3d[i,1], emb_3d[i,2], 0, 0, 0.5, color='green', alpha=0.5, arrow_length_ratio=0.1)
        
    # Harmonic (Cyclic): Around the cycle
    for k in range(len(cycle)-1):
        u, v = cycle[k], cycle[k+1]
        p1, p2 = emb_3d[u], emb_3d[v]
        vec = p2 - p1
        # Normalize and scale
        vec = vec / np.linalg.norm(vec) * 0.8
        ax3.quiver(p1[0], p1[1], p1[2], vec[0], vec[1], vec[2], color='red', linewidth=2)
        
    ax3.set_title("3. Hodge Decomposition\n(Green=Gradient, Red=Harmonic)")
    
    # Annotations connecting the flow
    plt.tight_layout()
    plt.savefig("projection_flow_viz.png")
    print("Visualization saved to 'projection_flow_viz.png'")

if __name__ == "__main__":
    visualize_projection_flow()
