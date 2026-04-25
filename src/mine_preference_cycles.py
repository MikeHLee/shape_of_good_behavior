"""
Mine Preference Cycles from Anthropic HH-RLHF Dataset

This script:
1. Loads the Anthropic HH-RLHF dataset
2. Embeds prompts and responses to identify semantic neighborhoods
3. Detects preference cycles (H¹ ≠ 0) via inconsistent preference directions
4. Extracts specific cycle examples for paper illustrations
5. Generates graph + surface visualizations

Mathematical Framework:
- Each prompt p defines a node in the preference graph
- Each (chosen, rejected) pair defines a preference vector: v_p = embed(chosen) - embed(rejected)
- For k-NN neighbors, we check if preference vectors are consistent
- Cycles occur when local preferences contradict: p₁ → p₂ but neighbor(p₂) → p₁

Usage:
    python mine_preference_cycles.py --samples 10000 --output cycles/
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import faiss
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PreferenceCycle:
    """A detected preference cycle with associated data."""
    cycle_id: int
    nodes: List[int]  # Indices into the dataset
    prompts: List[str]
    chosen_responses: List[str]
    rejected_responses: List[str]
    preference_vectors: np.ndarray  # Shape: (cycle_len, embed_dim)
    embeddings: np.ndarray  # Shape: (cycle_len, embed_dim)
    h1_score: float  # Harmonic residual (higher = more cyclic)
    cycle_type: str  # 'condorcet', 'style', 'safety', 'mixed'
    
    def to_dict(self) -> dict:
        return {
            'cycle_id': self.cycle_id,
            'nodes': self.nodes,
            'prompts': self.prompts,
            'chosen_responses': self.chosen_responses,
            'rejected_responses': self.rejected_responses,
            'h1_score': float(self.h1_score),
            'cycle_type': self.cycle_type,
        }


class PreferenceCycleMiner:
    """
    Mine preference cycles from RLHF datasets.
    
    The key insight: In a consistent preference ordering, nearby prompts
    should have similar preference directions. When they don't, we have
    a "local cycle" that contributes to H¹ cohomology.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        k_neighbors: int = 10,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.encoder = SentenceTransformer(embedding_model, device=self.device)
        self.k_neighbors = k_neighbors
        
        # Data storage
        self.prompts: List[str] = []
        self.chosen_responses: List[str] = []
        self.rejected_responses: List[str] = []
        self.prompt_embeddings: np.ndarray = None
        self.preference_vectors: np.ndarray = None
        self.neighbor_indices: np.ndarray = None
        self.harmonic_risk_scores: np.ndarray = None
        
    def load_anthropic_hh(self, num_samples: int = 10000) -> int:
        """Load and preprocess Anthropic HH-RLHF dataset."""
        print("Loading Anthropic HH-RLHF dataset...")
        dataset = load_dataset("anthropic/hh-rlhf", split="train")
        
        if num_samples and num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))
        
        print(f"Processing {len(dataset)} examples...")
        for example in tqdm(dataset, desc="Extracting pairs"):
            try:
                # Extract prompt and responses from conversation format
                chosen = example["chosen"]
                rejected = example["rejected"]
                
                # Split on last Assistant response
                prompt = chosen.rpartition("\n\nAssistant:")[0]
                chosen_resp = chosen.rpartition("\n\nAssistant:")[2].strip()
                rejected_resp = rejected.rpartition("\n\nAssistant:")[2].strip()
                
                if prompt and chosen_resp and rejected_resp:
                    self.prompts.append(prompt)
                    self.chosen_responses.append(chosen_resp)
                    self.rejected_responses.append(rejected_resp)
            except Exception as e:
                continue
        
        print(f"Successfully extracted {len(self.prompts)} preference pairs")
        return len(self.prompts)
    
    def compute_embeddings(self, batch_size: int = 64):
        """Embed prompts and compute preference vectors."""
        print("\nComputing embeddings...")
        
        # Embed prompts
        print("  Embedding prompts...")
        self.prompt_embeddings = self.encoder.encode(
            self.prompts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        faiss.normalize_L2(self.prompt_embeddings)
        
        # Embed responses
        print("  Embedding chosen responses...")
        chosen_embs = self.encoder.encode(
            self.chosen_responses,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print("  Embedding rejected responses...")
        rejected_embs = self.encoder.encode(
            self.rejected_responses,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Preference vectors: direction from rejected to chosen
        self.preference_vectors = chosen_embs - rejected_embs
        
        # Normalize preference directions
        norms = np.linalg.norm(self.preference_vectors, axis=1, keepdims=True)
        self.preference_directions = self.preference_vectors / (norms + 1e-8)
        
        print(f"  Embedding dimension: {self.prompt_embeddings.shape[1]}")
        
    def build_preference_graph(self):
        """Build k-NN graph and compute harmonic risk scores."""
        print("\nBuilding preference graph...")
        
        d = self.prompt_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # Inner product for normalized vectors = cosine
        index.add(self.prompt_embeddings)
        
        # Find k nearest neighbors
        D, I = index.search(self.prompt_embeddings, self.k_neighbors + 1)
        self.neighbor_indices = I[:, 1:]  # Exclude self
        self.neighbor_distances = D[:, 1:]
        
        print(f"  Graph built with {len(self.prompts)} nodes, k={self.k_neighbors}")
        
        # Compute harmonic risk (local inconsistency)
        self._compute_harmonic_risk()
        
    def _compute_harmonic_risk(self):
        """
        Compute harmonic risk score for each node.
        
        Harmonic risk = 1 - avg(cos_sim(preference_i, mean_neighbor_preference))
        
        High risk indicates the preference direction disagrees with neighbors,
        suggesting a local cycle or inconsistency.
        """
        print("  Computing harmonic risk scores...")
        
        neighbor_prefs = self.preference_directions[self.neighbor_indices]  # (N, k, d)
        
        # Local mean preference direction
        local_mean = np.mean(neighbor_prefs, axis=1)  # (N, d)
        local_mean_norm = np.linalg.norm(local_mean, axis=1, keepdims=True)
        local_mean_dir = local_mean / (local_mean_norm + 1e-8)
        
        # Consistency: how aligned is each node's preference with local mean?
        self_alignment = np.sum(self.preference_directions * local_mean_dir, axis=1)
        
        # Also check neighbor-neighbor consistency
        neighbor_consistencies = np.sum(
            neighbor_prefs * local_mean_dir[:, np.newaxis, :], 
            axis=2
        )  # (N, k)
        avg_neighbor_consistency = np.mean(neighbor_consistencies, axis=1)
        
        # Combined harmonic risk
        self.harmonic_risk_scores = 1.0 - (0.5 * self_alignment + 0.5 * avg_neighbor_consistency)
        
        # Normalize to [0, 1]
        min_risk = np.min(self.harmonic_risk_scores)
        max_risk = np.max(self.harmonic_risk_scores)
        self.harmonic_risk_scores = (self.harmonic_risk_scores - min_risk) / (max_risk - min_risk + 1e-8)
        
        print(f"  Mean harmonic risk: {np.mean(self.harmonic_risk_scores):.4f}")
        print(f"  Max harmonic risk: {np.max(self.harmonic_risk_scores):.4f}")
        
    def find_cycles(
        self, 
        min_h1_score: float = 0.6,
        max_cycles: int = 20,
        min_cycle_length: int = 3,
        max_cycle_length: int = 5
    ) -> List[PreferenceCycle]:
        """
        Find explicit preference cycles in the graph.
        
        Strategy: Start from high harmonic-risk nodes and trace through
        neighbors with contradicting preferences.
        """
        print(f"\nSearching for preference cycles (min H¹ score: {min_h1_score})...")
        
        # Sort by harmonic risk (highest first)
        risk_order = np.argsort(self.harmonic_risk_scores)[::-1]
        
        cycles = []
        visited_sets = set()
        
        for start_idx in tqdm(risk_order, desc="Cycle search"):
            if len(cycles) >= max_cycles:
                break
                
            if self.harmonic_risk_scores[start_idx] < min_h1_score:
                break
            
            # Try to find a cycle starting from this node
            cycle = self._trace_cycle(
                start_idx, 
                min_length=min_cycle_length,
                max_length=max_cycle_length
            )
            
            if cycle is not None:
                # Check if we've seen this cycle (up to rotation)
                cycle_set = frozenset(cycle)
                if cycle_set not in visited_sets:
                    visited_sets.add(cycle_set)
                    
                    # Create PreferenceCycle object
                    pc = self._create_cycle_object(cycle, len(cycles))
                    if pc.h1_score >= min_h1_score:
                        cycles.append(pc)
                        print(f"  Found cycle {len(cycles)}: nodes {cycle}, H¹={pc.h1_score:.3f}, type={pc.cycle_type}")
        
        print(f"\nFound {len(cycles)} preference cycles")
        return cycles
    
    def _trace_cycle(
        self, 
        start: int, 
        min_length: int = 3, 
        max_length: int = 5
    ) -> Optional[List[int]]:
        """
        Trace a cycle from start node by following preference contradictions.
        
        We look for paths where moving to a neighbor and comparing preferences
        creates a contradiction (the neighbor prefers something the current node rejected).
        """
        path = [start]
        current = start
        
        for _ in range(max_length):
            neighbors = self.neighbor_indices[current]
            
            # Find neighbor with most contradicting preference
            current_pref = self.preference_directions[current]
            neighbor_prefs = self.preference_directions[neighbors]
            
            # Contradiction score: negative dot product means opposite preference
            contradictions = -np.dot(neighbor_prefs, current_pref)
            
            # Exclude already visited
            for i, n in enumerate(neighbors):
                if n in path[:-1]:  # Allow returning to start
                    contradictions[i] = -np.inf
            
            best_neighbor_idx = np.argmax(contradictions)
            best_neighbor = neighbors[best_neighbor_idx]
            
            # Check if we can close the cycle
            if best_neighbor == start and len(path) >= min_length:
                return path
            
            # Otherwise, continue the path
            if contradictions[best_neighbor_idx] > 0:
                path.append(best_neighbor)
                current = best_neighbor
            else:
                break
        
        return None
    
    def _create_cycle_object(self, nodes: List[int], cycle_id: int) -> PreferenceCycle:
        """Create a PreferenceCycle object with all associated data."""
        prompts = [self.prompts[i][:500] for i in nodes]  # Truncate for readability
        chosen = [self.chosen_responses[i][:300] for i in nodes]
        rejected = [self.rejected_responses[i][:300] for i in nodes]
        pref_vecs = self.preference_vectors[nodes]
        embeddings = self.prompt_embeddings[nodes]
        
        # Compute H¹ score for this specific cycle
        # This is the "circulation" - how much the preferences fail to close
        pref_dirs = self.preference_directions[nodes]
        
        # Sum of preference rotations around the cycle
        n = len(nodes)
        circulation = 0.0
        for i in range(n):
            j = (i + 1) % n
            # How much does preference at i contradict preference at j?
            circulation += 1.0 - np.dot(pref_dirs[i], pref_dirs[j])
        h1_score = circulation / n
        
        # Classify cycle type based on content analysis
        cycle_type = self._classify_cycle(prompts, chosen, rejected)
        
        return PreferenceCycle(
            cycle_id=cycle_id,
            nodes=nodes,
            prompts=prompts,
            chosen_responses=chosen,
            rejected_responses=rejected,
            preference_vectors=pref_vecs,
            embeddings=embeddings,
            h1_score=h1_score,
            cycle_type=cycle_type
        )
    
    def _classify_cycle(
        self, 
        prompts: List[str], 
        chosen: List[str], 
        rejected: List[str]
    ) -> str:
        """Classify the type of preference cycle based on content."""
        all_text = " ".join(prompts + chosen + rejected).lower()
        
        safety_keywords = ['harm', 'dangerous', 'illegal', 'unsafe', 'kill', 'hurt', 'weapon']
        style_keywords = ['formal', 'casual', 'brief', 'detailed', 'technical', 'simple']
        
        safety_count = sum(1 for kw in safety_keywords if kw in all_text)
        style_count = sum(1 for kw in style_keywords if kw in all_text)
        
        if safety_count >= 2:
            return 'safety'
        elif style_count >= 2:
            return 'style'
        else:
            return 'condorcet'  # General preference cycle
    
    def visualize_cycles(
        self, 
        cycles: List[PreferenceCycle], 
        output_dir: str,
        use_tsne: bool = True
    ):
        """Generate graph and surface visualizations for cycles."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating visualizations in {output_dir}...")
        
        # 1. Overview: All high-risk regions
        self._plot_risk_landscape(output_dir, use_tsne)
        
        # 2. Individual cycle visualizations
        for cycle in cycles:
            self._plot_cycle_graph(cycle, output_dir)
            self._plot_cycle_surface(cycle, output_dir)
        
        # 3. Combined cycle comparison
        if len(cycles) >= 2:
            self._plot_cycle_comparison(cycles, output_dir)
    
    def _plot_risk_landscape(self, output_dir: str, use_tsne: bool = True):
        """Plot the overall harmonic risk landscape."""
        print("  Plotting risk landscape...")
        
        # Reduce dimensions
        if use_tsne and len(self.prompts) <= 5000:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            coords = reducer.fit_transform(self.prompt_embeddings[:5000])
            risks = self.harmonic_risk_scores[:5000]
        else:
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(self.prompt_embeddings)
            risks = self.harmonic_risk_scores
        
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], 
            c=risks, cmap='RdYlBu_r', 
            alpha=0.6, s=10
        )
        plt.colorbar(scatter, label='Harmonic Risk (H¹ contribution)')
        ax.set_title('Preference Space: Harmonic Risk Landscape\n(Red = High inconsistency, potential cycles)')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'risk_landscape.png'), dpi=150)
        plt.close()
    
    def _plot_cycle_graph(self, cycle: PreferenceCycle, output_dir: str):
        """Plot a single cycle as a directed graph."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        G = nx.DiGraph()
        n = len(cycle.nodes)
        
        # Add nodes with labels
        for i, node in enumerate(cycle.nodes):
            label = f"P{i+1}\n{cycle.prompts[i][:50]}..."
            G.add_node(i, label=label)
        
        # Add edges (preference directions)
        for i in range(n):
            j = (i + 1) % n
            G.add_edge(i, j)
        
        # Circular layout
        pos = nx.circular_layout(G)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=2000)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='red', arrows=True, 
                               arrowsize=20, connectionstyle="arc3,rad=0.1")
        
        # Labels
        labels = {i: f"P{i+1}" for i in range(n)}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=12, font_weight='bold')
        
        ax.set_title(f'Preference Cycle {cycle.cycle_id + 1}\nH¹ Score: {cycle.h1_score:.3f}, Type: {cycle.cycle_type}')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cycle_{cycle.cycle_id + 1}_graph.png'), dpi=150)
        plt.close()
    
    def _plot_cycle_surface(self, cycle: PreferenceCycle, output_dir: str):
        """Plot the preference vectors as a 3D surface."""
        fig = plt.figure(figsize=(12, 5))
        
        # Reduce preference vectors to 3D
        pca = PCA(n_components=3)
        pref_3d = pca.fit_transform(cycle.preference_vectors)
        emb_3d = pca.transform(cycle.embeddings)
        
        # Left: Embedding space with preference arrows
        ax1 = fig.add_subplot(121, projection='3d')
        
        n = len(cycle.nodes)
        colors = plt.cm.rainbow(np.linspace(0, 1, n))
        
        for i in range(n):
            ax1.scatter(*emb_3d[i], c=[colors[i]], s=100, label=f'P{i+1}')
            # Arrow showing preference direction
            ax1.quiver(
                emb_3d[i, 0], emb_3d[i, 1], emb_3d[i, 2],
                pref_3d[i, 0], pref_3d[i, 1], pref_3d[i, 2],
                color=colors[i], alpha=0.7, arrow_length_ratio=0.3
            )
        
        ax1.set_title('Embedding Space + Preference Vectors')
        ax1.legend(loc='upper left', fontsize=8)
        
        # Right: Preference vector circulation
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Plot preference vectors as a closed loop
        for i in range(n):
            j = (i + 1) % n
            ax2.plot(
                [pref_3d[i, 0], pref_3d[j, 0]],
                [pref_3d[i, 1], pref_3d[j, 1]],
                [pref_3d[i, 2], pref_3d[j, 2]],
                color='red', linewidth=2
            )
            ax2.scatter(*pref_3d[i], c=[colors[i]], s=100)
        
        ax2.set_title(f'Preference Circulation (H¹ = {cycle.h1_score:.3f})')
        
        plt.suptitle(f'Cycle {cycle.cycle_id + 1}: {cycle.cycle_type.capitalize()} Preferences')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cycle_{cycle.cycle_id + 1}_surface.png'), dpi=150)
        plt.close()
    
    def _plot_cycle_comparison(self, cycles: List[PreferenceCycle], output_dir: str):
        """Compare multiple cycles in a single visualization."""
        fig, axes = plt.subplots(2, min(3, len(cycles)), figsize=(15, 10))
        if len(cycles) < 3:
            axes = axes.reshape(2, -1)
        
        for idx, cycle in enumerate(cycles[:3]):
            # Top row: Graph view
            ax_graph = axes[0, idx]
            G = nx.DiGraph()
            n = len(cycle.nodes)
            for i in range(n):
                G.add_node(i)
                G.add_edge(i, (i + 1) % n)
            
            pos = nx.circular_layout(G)
            nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color='lightblue', node_size=500)
            nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color='red', arrows=True, arrowsize=15)
            nx.draw_networkx_labels(G, pos, {i: f'P{i+1}' for i in range(n)}, ax=ax_graph, font_size=10)
            ax_graph.set_title(f'Cycle {idx+1}: {cycle.cycle_type}\nH¹ = {cycle.h1_score:.3f}')
            ax_graph.axis('off')
            
            # Bottom row: H¹ contribution bar
            ax_bar = axes[1, idx]
            pref_dirs = cycle.preference_vectors / (np.linalg.norm(cycle.preference_vectors, axis=1, keepdims=True) + 1e-8)
            contradictions = []
            for i in range(n):
                j = (i + 1) % n
                contradictions.append(1 - np.dot(pref_dirs[i], pref_dirs[j]))
            
            ax_bar.bar(range(n), contradictions, color='coral')
            ax_bar.set_xlabel('Edge')
            ax_bar.set_ylabel('Local H¹ contribution')
            ax_bar.set_xticks(range(n))
            ax_bar.set_xticklabels([f'{i+1}→{(i+1)%n+1}' for i in range(n)])
        
        plt.suptitle('Preference Cycle Comparison from Anthropic HH-RLHF')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cycle_comparison.png'), dpi=150)
        plt.close()
    
    def export_cycles(self, cycles: List[PreferenceCycle], output_path: str):
        """Export cycles to JSON for paper/analysis."""
        data = {
            'metadata': {
                'dataset': 'anthropic/hh-rlhf',
                'n_samples': len(self.prompts),
                'k_neighbors': self.k_neighbors,
                'n_cycles_found': len(cycles),
                'mean_h1_score': float(np.mean([c.h1_score for c in cycles])) if cycles else 0,
            },
            'cycles': [c.to_dict() for c in cycles]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported {len(cycles)} cycles to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Mine preference cycles from Anthropic HH-RLHF")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to process")
    parser.add_argument("--k-neighbors", type=int, default=10, help="k for k-NN graph")
    parser.add_argument("--min-h1", type=float, default=0.5, help="Minimum H¹ score for cycles")
    parser.add_argument("--max-cycles", type=int, default=10, help="Maximum cycles to find")
    parser.add_argument("--output", type=str, default="cycles", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()
    
    # Initialize miner
    miner = PreferenceCycleMiner(k_neighbors=args.k_neighbors)
    
    # Load data
    miner.load_anthropic_hh(num_samples=args.samples)
    
    # Compute embeddings
    miner.compute_embeddings(batch_size=args.batch_size)
    
    # Build preference graph
    miner.build_preference_graph()
    
    # Find cycles
    cycles = miner.find_cycles(
        min_h1_score=args.min_h1,
        max_cycles=args.max_cycles
    )
    
    # Visualize
    output_dir = os.path.join(
        os.path.dirname(__file__), 
        '..', 'submission', 'figures', args.output
    )
    miner.visualize_cycles(cycles, output_dir)
    
    # Export
    miner.export_cycles(
        cycles, 
        os.path.join(output_dir, 'preference_cycles.json')
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PREFERENCE CYCLE MINING SUMMARY")
    print("="*60)
    print(f"Dataset: Anthropic HH-RLHF ({len(miner.prompts)} samples)")
    print(f"Cycles found: {len(cycles)}")
    print(f"Output: {output_dir}")
    
    if cycles:
        print("\nTop Cycles:")
        for c in sorted(cycles, key=lambda x: -x.h1_score)[:5]:
            print(f"  Cycle {c.cycle_id + 1}: H¹={c.h1_score:.3f}, type={c.cycle_type}, nodes={c.nodes}")
            print(f"    Prompt 1: {c.prompts[0][:80]}...")


if __name__ == "__main__":
    main()
