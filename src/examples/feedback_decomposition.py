"""
Feedback Decomposition in Common Embedding Space

Demonstrates how to combine verbal, ordinal, and pass/fail feedback
in a unified embedding space for preference learning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class FeedbackEmbedder:
    """Embed different feedback types into a common semantic space."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        self.ordinal_anchors = [
            "This is terrible, completely unacceptable.",
            "This is poor, needs significant improvement.",
            "This is acceptable but mediocre.",
            "This is good, minor improvements possible.",
            "This is excellent, no changes needed.",
        ]
        self.ordinal_embeddings = self.encoder.encode(self.ordinal_anchors)
    
    def embed_verbal_feedback(self, text: str) -> np.ndarray:
        """Direct sentence embedding of verbal feedback."""
        return self.encoder.encode(text)
    
    def embed_ordinal_feedback(self, rating: int, max_rating: int = 5) -> np.ndarray:
        """
        Map ordinal rating to embedding space via anchor texts.
        
        Args:
            rating: Integer rating (1 to max_rating)
            max_rating: Maximum possible rating
        
        Returns:
            Embedding vector
        """
        if rating < 1 or rating > max_rating:
            raise ValueError(f"Rating must be between 1 and {max_rating}")
        
        idx = min(rating - 1, len(self.ordinal_anchors) - 1)
        
        if rating <= len(self.ordinal_anchors):
            return self.ordinal_embeddings[idx]
        else:
            frac = (rating - 1) / (max_rating - 1)
            idx_low = int(frac * (len(self.ordinal_anchors) - 1))
            idx_high = min(idx_low + 1, len(self.ordinal_anchors) - 1)
            alpha = frac * (len(self.ordinal_anchors) - 1) - idx_low
            
            return (1 - alpha) * self.ordinal_embeddings[idx_low] + \
                   alpha * self.ordinal_embeddings[idx_high]
    
    def embed_passfail_feedback(self, criterion: str, passed: bool) -> np.ndarray:
        """
        Map pass/fail to embedding with criterion context.
        
        Args:
            criterion: Description of the criterion being evaluated
            passed: Whether the criterion was satisfied
        
        Returns:
            Embedding vector
        """
        if passed:
            text = f"Satisfies criterion: {criterion}"
        else:
            text = f"Fails criterion: {criterion}"
        
        return self.encoder.encode(text)
    
    def compute_preference_vector(
        self,
        state_emb: np.ndarray,
        chosen_emb: np.ndarray,
        rejected_emb: np.ndarray,
        verbal_feedback: Optional[str] = None,
        ordinal_rating: Optional[int] = None,
        passfail: Optional[Dict[str, bool]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Compute weighted preference direction in embedding space.
        
        This vector points from rejected toward chosen,
        modulated by additional feedback signals.
        
        Args:
            state_emb: State/context embedding
            chosen_emb: Chosen response embedding
            rejected_emb: Rejected response embedding
            verbal_feedback: Optional verbal critique
            ordinal_rating: Optional 1-5 rating
            passfail: Optional dict of {"criterion": str, "passed": bool}
            weights: Weighting of different feedback types
        
        Returns:
            Preference vector in embedding space
        """
        if weights is None:
            weights = {
                "base": 1.0,
                "verbal": 0.5,
                "ordinal": 0.3,
                "passfail": 0.2,
            }
        
        base_pref = chosen_emb - rejected_emb
        base_norm = np.linalg.norm(base_pref) + 1e-8
        base_dir = base_pref / base_norm
        
        pref_vector = weights["base"] * base_pref
        
        if verbal_feedback:
            verbal_emb = self.embed_verbal_feedback(verbal_feedback)
            verbal_component = np.dot(verbal_emb, base_pref) / base_norm
            pref_vector += weights["verbal"] * verbal_component * base_dir
        
        if ordinal_rating is not None:
            ordinal_emb = self.embed_ordinal_feedback(ordinal_rating)
            rating_scale = (ordinal_rating - 3) / 2
            pref_vector += weights["ordinal"] * rating_scale * base_pref
        
        if passfail:
            for criterion, passed in passfail.items():
                pf_emb = self.embed_passfail_feedback(criterion, passed)
                pf_scale = 1.0 if passed else -0.5
                pref_vector += weights["passfail"] * pf_scale * (pf_emb - state_emb)
        
        return pref_vector


class HodgeFeedbackDecomposer:
    """Apply Hodge decomposition to preference vector field."""
    
    def __init__(self, k_neighbors: int = 5):
        self.k_neighbors = k_neighbors
    
    def decompose_feedback_field(
        self,
        states: List[np.ndarray],
        preference_vectors: List[np.ndarray],
    ) -> Dict:
        """
        Apply Hodge decomposition to preference vector field.
        
        Args:
            states: List of state embeddings
            preference_vectors: List of preference vectors at each state
        
        Returns:
            dict with gradient, harmonic, and h1_magnitude
        """
        states_array = np.array(states)
        prefs_array = np.array(preference_vectors)
        
        nn = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(states)))
        nn.fit(states_array)
        
        edges = []
        edge_weights = []
        
        for i, state in enumerate(states_array):
            neighbors = nn.kneighbors([state], return_distance=False)[0]
            for j in neighbors:
                if i != j:
                    edges.append((i, j))
                    edge_dir = states_array[j] - states_array[i]
                    edge_norm = np.linalg.norm(edge_dir) + 1e-8
                    weight = np.dot(preference_vectors[i], edge_dir) / edge_norm
                    edge_weights.append(weight)
        
        n_vertices = len(states)
        n_edges = len(edges)
        
        B = np.zeros((n_edges, n_vertices))
        for i, (src, dst) in enumerate(edges):
            B[i, src] = -1
            B[i, dst] = 1
        
        r = np.array(edge_weights)
        
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
            "edges": edges,
            "edge_weights": edge_weights,
        }


class WritingAssistantExample:
    """Example: AI writing assistant with mixed feedback."""
    
    def __init__(self):
        self.embedder = FeedbackEmbedder()
        self.decomposer = HodgeFeedbackDecomposer(k_neighbors=3)
        
        self.examples = [
            {
                "state": "Draft email to friend about weekend plans",
                "chosen": "Hey! Want to grab coffee this weekend? Let me know!",
                "rejected": "Dear Friend, I would like to formally invite you to join me for coffee this weekend. Please advise your availability.",
                "verbal": "Too formal for a friend",
                "ordinal": 2,
                "passfail": {"grammatically_correct": True, "appropriate_tone": False},
            },
            {
                "state": "Draft email to professor about deadline extension",
                "chosen": "Dear Professor Smith, I am writing to request a brief extension on the assignment due to unforeseen circumstances. Would a 2-day extension be possible?",
                "rejected": "Hey Prof, can I get more time on the homework? Thanks!",
                "verbal": "Too casual for academic context",
                "ordinal": 2,
                "passfail": {"grammatically_correct": True, "appropriate_tone": False},
            },
            {
                "state": "Draft thank you note to colleague",
                "chosen": "Thanks so much for your help with the presentation! I really appreciate your insights.",
                "rejected": "Thank you for your assistance.",
                "verbal": "Good balance of warmth and professionalism",
                "ordinal": 4,
                "passfail": {"grammatically_correct": True, "appropriate_tone": True},
            },
            {
                "state": "Draft apology email to client",
                "chosen": "I sincerely apologize for the delay. We are working to resolve this immediately and will keep you updated.",
                "rejected": "Sorry about that! We'll fix it soon.",
                "verbal": "Professional and takes responsibility",
                "ordinal": 5,
                "passfail": {"grammatically_correct": True, "appropriate_tone": True},
            },
        ]
    
    def compute_all_preferences(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute state embeddings and preference vectors for all examples."""
        states = []
        preferences = []
        
        for ex in self.examples:
            state_emb = self.embedder.encoder.encode(ex["state"])
            chosen_emb = self.embedder.encoder.encode(ex["chosen"])
            rejected_emb = self.embedder.encoder.encode(ex["rejected"])
            
            pref_vec = self.embedder.compute_preference_vector(
                state_emb=state_emb,
                chosen_emb=chosen_emb,
                rejected_emb=rejected_emb,
                verbal_feedback=ex["verbal"],
                ordinal_rating=ex["ordinal"],
                passfail=ex["passfail"],
            )
            
            states.append(state_emb)
            preferences.append(pref_vec)
        
        return states, preferences
    
    def visualize_embedding_space(self, states: List[np.ndarray], 
                                  preferences: List[np.ndarray],
                                  decomp: Dict,
                                  save_path: str = None):
        """Visualize 2D PCA projection of embedding space with preference vectors."""
        states_array = np.array(states)
        prefs_array = np.array(preferences)
        
        pca = PCA(n_components=2)
        states_2d = pca.fit_transform(states_array)
        prefs_2d = pca.transform(prefs_array) - pca.transform(np.zeros_like(prefs_array))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax = axes[0]
        ax.set_title("Preference Vectors (Combined Feedback)", fontsize=14, fontweight='bold')
        ax.scatter(states_2d[:, 0], states_2d[:, 1], c='blue', s=100, alpha=0.6, label='States')
        
        for i, (state_2d, pref_2d) in enumerate(zip(states_2d, prefs_2d)):
            rating = self.examples[i]["ordinal"]
            color = plt.cm.RdYlGn(rating / 5.0)
            
            ax.arrow(state_2d[0], state_2d[1], pref_2d[0], pref_2d[1],
                    head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=0.7)
            ax.text(state_2d[0], state_2d[1] - 0.1, f"Ex{i+1}", 
                   fontsize=9, ha='center')
        
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.set_title(f"Hodge Decomposition (H¹={decomp['h1_magnitude']:.3f})", 
                    fontsize=14, fontweight='bold')
        
        gradient_norm = np.linalg.norm(decomp['gradient'])
        harmonic_norm = np.linalg.norm(decomp['harmonic'])
        
        ax.bar(['Gradient\n(Learnable)', 'Harmonic\n(Cycle)'], 
              [gradient_norm, harmonic_norm],
              color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel("L2 Norm")
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def run_example():
    """Run the feedback decomposition example."""
    print("=" * 80)
    print("FEEDBACK DECOMPOSITION EXAMPLE")
    print("=" * 80)
    
    example = WritingAssistantExample()
    
    print("\n1. Computing embeddings and preference vectors...")
    print("-" * 80)
    states, preferences = example.compute_all_preferences()
    print(f"✓ Processed {len(states)} examples")
    print(f"  Embedding dimension: {states[0].shape[0]}")
    print(f"  Preference vector dimension: {preferences[0].shape[0]}")
    
    print("\n2. Performing Hodge decomposition...")
    print("-" * 80)
    decomp = example.decomposer.decompose_feedback_field(states, preferences)
    
    print(f"  H¹ magnitude: {decomp['h1_magnitude']:.4f}")
    print(f"  Gradient norm: {np.linalg.norm(decomp['gradient']):.4f}")
    print(f"  Harmonic norm: {np.linalg.norm(decomp['harmonic']):.4f}")
    
    if decomp['h1_magnitude'] < 0.1:
        print("\n  ✓ Low H¹: Feedback is mostly consistent")
    else:
        print("\n  ⚠ High H¹: Significant inconsistency in feedback")
    
    print("\n3. Example breakdowns:")
    print("-" * 80)
    for i, ex in enumerate(example.examples):
        print(f"\nExample {i+1}: {ex['state']}")
        print(f"  Verbal: '{ex['verbal']}'")
        print(f"  Rating: {ex['ordinal']}/5")
        print(f"  Pass/Fail: {ex['passfail']}")
        print(f"  Preference norm: {np.linalg.norm(preferences[i]):.3f}")
    
    print("\n4. Generating visualization...")
    print("-" * 80)
    fig = example.visualize_embedding_space(
        states, 
        preferences, 
        decomp,
        save_path="../../figures/examples/feedback_decomposition.png"
    )
    print("✓ Saved to figures/examples/feedback_decomposition.png")
    
    plt.show()
    
    return example, states, preferences, decomp


if __name__ == "__main__":
    example, states, preferences, decomposition = run_example()
