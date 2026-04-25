"""
Hodge Critic: Topological Reward Learning from Natural Language Feedback

The Hodge Critic processes human feedback (rankings + verbal critiques) and:
1. Embeds feedback into a high-dimensional vector field
2. Applies Hodge decomposition to separate consistent (gradient) from inconsistent (curl)
3. Provides topological gradients for policy alignment

Key insight: Natural language is trivially embeddable, making this approach practical.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import lsqr
import warnings


@dataclass
class FeedbackItem:
    """A single piece of human feedback on a state-action pair."""
    state_text: str                     # Scene description
    action_text: str                    # Action taken
    next_state_text: Optional[str]      # Resulting scene
    rank: float                         # 0-1 helpfulness/progress
    critique: Optional[str] = None      # Verbal explanation
    evaluator_id: Optional[str] = None  # For multi-evaluator analysis
    
    def to_embedding_text(self) -> str:
        """Combine into single text for embedding."""
        parts = [f"State: {self.state_text}", f"Action: {self.action_text}"]
        if self.next_state_text:
            parts.append(f"Result: {self.next_state_text}")
        if self.critique:
            parts.append(f"Critique: {self.critique}")
        return " | ".join(parts)


@dataclass
class CondorcetCycle:
    """A detected Condorcet cycle (preference inconsistency)."""
    cycle_indices: List[int]            # Indices of items forming the cycle
    cycle_items: List[str]              # Text descriptions of items in cycle
    circulation: float                  # Magnitude of the cycle (non-zero = Condorcet)
    
    def __str__(self) -> str:
        items_str = " > ".join(self.cycle_items[:3]) + " > ..."
        return f"Condorcet({items_str}, circulation={self.circulation:.3f})"


@dataclass
class TopologicalGradient:
    """Result of Hodge decomposition on feedback.

    REVISED: The harmonic component is split into genuine (cross-context value
    tensions like helpfulness-vs-harmlessness) and exploitable (within-context
    transitivity violations). Only exploitable harmonic should be discarded.

    Components:
    - gradient_component: Transitive consensus (Borda count) — always use
    - curl_component: Local cyclic inconsistencies in 3-cliques — discard
    - harmonic_component: ALL global cycles (genuine + exploitable)
    - genuine_harmonic: Cross-context value tensions — PRESERVE
    - exploitable_harmonic: Within-context transitivity violations — DISCARD
    """
    gradient_component: np.ndarray      # ∇φ: transitive consensus
    curl_component: np.ndarray          # δψ: local cyclic inconsistencies
    harmonic_component: np.ndarray      # h: ALL global cycles
    h1_magnitude: float                 # ||harmonic||: magnitude of global cycles
    genuine_harmonic: np.ndarray = None       # Cross-context value tensions — PRESERVE
    exploitable_harmonic: np.ndarray = None   # Within-context violations — DISCARD
    exploit_fraction: float = 0.0              # Fraction of harmonic that is exploitable
    condorcet_cycles: List[CondorcetCycle] = None

    def __post_init__(self):
        if self.condorcet_cycles is None:
            self.condorcet_cycles = []
        if self.genuine_harmonic is None:
            self.genuine_harmonic = self.harmonic_component.copy()
        if self.exploitable_harmonic is None:
            self.exploitable_harmonic = np.zeros_like(self.harmonic_component)

    def get_transitive_direction(self) -> np.ndarray:
        """Return gradient + genuine harmonic for training.

        REVISED: Now includes genuine harmonic (cross-context value tensions)
        which represent real helpfulness-vs-harmlessness tradeoffs, not noise.
        Only exploitable within-context cycles are excluded.
        """
        return self.gradient_component + self.genuine_harmonic

    def get_gradient_only(self) -> np.ndarray:
        """Return ONLY the gradient component (most conservative)."""
        return self.gradient_component.copy()

    def get_exploit_correction(self) -> np.ndarray:
        """Return only the exploitable harmonic for dampening in SGPO."""
        return self.exploitable_harmonic.copy()

    def get_clean_direction(self) -> np.ndarray:
        """DEPRECATED: Use get_transitive_direction() instead."""
        import warnings
        warnings.warn(
            "get_clean_direction() is deprecated. "
            "Use get_transitive_direction() which returns gradient + genuine harmonic.",
            DeprecationWarning
        )
        return self.gradient_component + self.harmonic_component

    def has_condorcet_cycles(self) -> bool:
        """Check if any Condorcet cycles were detected."""
        return len(self.condorcet_cycles) > 0

    def get_cycle_summary(self) -> str:
        """Get human-readable summary of detected cycles."""
        if not self.condorcet_cycles:
            return "No Condorcet cycles detected (preferences are consistent)"
        genuine_pct = (1.0 - self.exploit_fraction) * 100
        return (
            f"{len(self.condorcet_cycles)} Condorcet cycles detected "
            f"(H¹={self.h1_magnitude:.3f}, {genuine_pct:.0f}% genuine)"
        )

    @property
    def reliability_score(self) -> float:
        """
        Reliability = (||gradient||² + ||genuine_harmonic||²) / ||total||²

        REVISED: Genuine harmonic (cross-context tensions) is counted as
        reliable signal. Only exploitable harmonic and curl reduce reliability.
        """
        gradient_energy = np.sum(self.gradient_component ** 2)
        genuine_energy = np.sum(self.genuine_harmonic ** 2)
        curl_energy = np.sum(self.curl_component ** 2)
        exploit_energy = np.sum(self.exploitable_harmonic ** 2)
        total_energy = gradient_energy + genuine_energy + curl_energy + exploit_energy

        if total_energy < 1e-10:
            return 1.0
        return (gradient_energy + genuine_energy) / total_energy

    @property
    def cyclic_residual(self) -> float:
        """Fraction of energy in curl + exploitable harmonic."""
        return 1.0 - self.reliability_score


@dataclass 
class ManifoldPoint:
    """A point on the reward manifold with its local geometry.
    
    NOTE: This is used for continuous embedding space (Module 2 domain).
    Do not conflate with discrete Hodge decomposition (Module 1 domain).
    """
    embedding: np.ndarray
    feedback_items: List[FeedbackItem] = field(default_factory=list)
    local_gradient: Optional[np.ndarray] = None
    # REMOVED: curvature_estimate - curl ≠ curvature, this was a categorical error
    # Curvature is a Riemannian concept; curl is a combinatorial operator
    local_cyclic_inconsistency: float = 0.0  # Magnitude of curl at this point
    is_danger_zone: bool = False  # Near a dangerous region (use conformal_safety.py)
    is_cliff: bool = False  # Sharp discontinuity in value


class HodgeCritic:
    """
    The Hodge Critic learns reward manifold geometry from feedback.
    
    Pipeline:
    1. Collect feedback items (state, action, rank, critique)
    2. Embed all items into high-dimensional space
    3. Construct simplicial complex from trajectory graph
    4. Compute Hodge decomposition to get clean gradients
    5. Use gradients to guide policy optimization
    """
    
    def __init__(
        self,
        embedding_model: Any,
        embed_dim: Optional[int] = None,
        similarity_threshold: float = 0.8,
    ):
        """
        Args:
            embedding_model: A model with .encode(texts) -> np.ndarray
            embed_dim: Embedding dimension (auto-detected if None)
            similarity_threshold: Cosine similarity to consider states "connected"
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        
        # Auto-detect embedding dimension
        if embed_dim is None:
            test = embedding_model.encode(["test"])
            self.embed_dim = test.shape[-1]
        else:
            self.embed_dim = embed_dim
        
        # Storage
        self.feedback_items: List[FeedbackItem] = []
        self.embeddings: Optional[np.ndarray] = None
        self.manifold_points: List[ManifoldPoint] = []
        
        # Graph structure (built from trajectories)
        self.adjacency: Optional[csr_matrix] = None
        self.edge_weights: Dict[Tuple[int, int], float] = {}
        
        # Hodge decomposition results
        self._gradient_field: Optional[np.ndarray] = None
        self._curl_field: Optional[np.ndarray] = None
        self._harmonic_field: Optional[np.ndarray] = None
        self._h1_magnitude: float = 0.0
    
    def add_feedback(self, item: FeedbackItem):
        """Add a single feedback item."""
        self.feedback_items.append(item)
        self._invalidate_cache()
    
    def add_feedback_batch(self, items: List[FeedbackItem]):
        """Add multiple feedback items."""
        self.feedback_items.extend(items)
        self._invalidate_cache()
    
    def add_trajectory_feedback(
        self,
        trajectory: List[Dict],
        ranks: List[float],
        critiques: Optional[List[str]] = None,
        evaluator_id: Optional[str] = None,
    ):
        """
        Add feedback for an entire trajectory.
        
        Args:
            trajectory: List of {"state": str, "action": str, "next_state": str}
            ranks: Per-transition rankings (same length as trajectory)
            critiques: Optional verbal feedback per transition
            evaluator_id: ID of the human evaluator
        """
        critiques = critiques or [None] * len(trajectory)
        
        for t, rank, critique in zip(trajectory, ranks, critiques):
            item = FeedbackItem(
                state_text=t["state"],
                action_text=t["action"],
                next_state_text=t.get("next_state"),
                rank=rank,
                critique=critique,
                evaluator_id=evaluator_id,
            )
            self.feedback_items.append(item)
        
        self._invalidate_cache()
    
    def _invalidate_cache(self):
        """Clear cached computations when data changes."""
        self.embeddings = None
        self.adjacency = None
        self._gradient_field = None
        self._curl_field = None
        self._harmonic_field = None
    
    def _compute_embeddings(self):
        """Embed all feedback items."""
        if self.embeddings is not None:
            return
        
        texts = [item.to_embedding_text() for item in self.feedback_items]
        self.embeddings = self.embedding_model.encode(texts)
        
        # Create manifold points
        self.manifold_points = [
            ManifoldPoint(embedding=self.embeddings[i], feedback_items=[item])
            for i, item in enumerate(self.feedback_items)
        ]
    
    def add_comparison(
        self,
        state_a: str,
        state_b: str,
        preference: float,
        action_a: str = "compare",
        action_b: str = "compare",
    ):
        """
        Add a direct pairwise comparison (edge flow).
        
        Args:
            state_a: First state
            state_b: Second state
            preference: Strength of preference for B over A (positive = B > A)
                        This directly sets the edge flow Y_{ab} = preference.
        """
        # Add items if they don't exist (using dummy ranks for node storage)
        # We need to track them to build the graph
        
        # Helper to find or add item
        def get_or_create_idx(text, act):
            for i, item in enumerate(self.feedback_items):
                if item.state_text == text and item.action_text == act:
                    return i
            # Create new
            item = FeedbackItem(text, act, None, 0.0) # Rank 0 dummy
            self.feedback_items.append(item)
            return len(self.feedback_items) - 1
            
        idx_a = get_or_create_idx(state_a, action_a)
        idx_b = get_or_create_idx(state_b, action_b)
        
        # Store explicit edge flow
        # We need to make sure _build_graph respects this
        if not hasattr(self, '_explicit_edges'):
            self._explicit_edges = {}
        
        self._explicit_edges[(idx_a, idx_b)] = preference
        self._invalidate_cache()

    def _build_graph(self):
        """
        Build the trajectory graph as a simplicial complex.
        
        Edges connect:
        1. Sequential states in trajectories (explicit)
        2. Similar states across trajectories (implicit, via embedding similarity)
        3. Explicit comparisons
        """
        self._compute_embeddings()
        
        n = len(self.feedback_items)
        adj = lil_matrix((n, n))
        
        # 1. Connect explicit comparisons first (highest priority)
        if hasattr(self, '_explicit_edges'):
            for (i, j), weight in self._explicit_edges.items():
                adj[i, j] = 1
                adj[j, i] = 1
                self.edge_weights[(i, j)] = weight
        
        # 2. Connect sequential states (from trajectory structure)
        for i in range(n - 1):
            curr = self.feedback_items[i]
            next_item = self.feedback_items[i + 1]
            
            # Check if they're from the same trajectory (next_state matches next state)
            if curr.next_state_text and curr.next_state_text == next_item.state_text:
                if (i, i+1) not in self.edge_weights: # Don't overwrite explicit
                    adj[i, i + 1] = 1
                    adj[i + 1, i] = 1
                    # Edge weight = rank difference (flow direction)
                    self.edge_weights[(i, i + 1)] = next_item.rank - curr.rank
        
        # 3. Connect similar states via cosine similarity
        if n > 1:
            # Normalize embeddings
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            normalized = self.embeddings / (norms + 1e-8)
            
            # Compute pairwise similarities
            similarities = normalized @ normalized.T
            
            # Connect highly similar states
            for i in range(n):
                for j in range(i + 1, n):
                    if similarities[i, j] > self.similarity_threshold:
                        if adj[i, j] == 0:  # Don't override trajectory/explicit edges
                            adj[i, j] = 0.5  # Weaker connection
                            adj[j, i] = 0.5
                            # Infer flow from rank difference
                            if (i, j) not in self.edge_weights and (j, i) not in self.edge_weights:
                                self.edge_weights[(i, j)] = self.feedback_items[j].rank - self.feedback_items[i].rank
        
        self.adjacency = adj.tocsr()
    
    def compute_hodge_decomposition(self) -> TopologicalGradient:
        """
        Apply Hodge decomposition to the feedback vector field.
        
        The vector field X on the graph is defined by rank differences.
        We decompose: X = ∇φ + ∇×ψ + h
        
        For a graph (0-simplices = nodes, 1-simplices = edges):
        - Gradient: φ is a node potential, ∇φ flows from low to high potential
        - Curl: Cycles in the graph where ranks form a loop
        - Harmonic: Global structure not captured by potential or curl
        """
        self._build_graph()
        
        n = len(self.feedback_items)
        if n < 2:
            return TopologicalGradient(
                gradient_component=np.zeros(self.embed_dim),
                curl_component=np.zeros(self.embed_dim),
                harmonic_component=np.zeros(self.embed_dim),
                h1_magnitude=0.0,
            )
        
        # Build boundary operators for the simplicial complex
        # d0: edges -> nodes (gradient operator)
        # d1: triangles -> edges (curl operator)
        
        # 1. Construct 0-boundary operator (d0)
        # Shape: (n_edges, n_nodes)
        rows, cols, data = [], [], []
        edge_list = list(self.edge_weights.keys())
        edge_to_idx = {edge: i for i, edge in enumerate(edge_list)}
        
        for i, (u, v) in enumerate(edge_list):
            rows.extend([i, i])
            cols.extend([u, v])
            data.extend([-1, 1]) # v - u
            
        d0 = csr_matrix((data, (rows, cols)), shape=(len(edge_list), n))
        
        # 2. Construct 1-boundary operator (d1)
        # Find triangles (2-simplices)
        triangles = []
        
        # To find triangles, iterate over edges (u,v) and check for common neighbor w
        # such that (v,w) and (w,u) exist.
        # Note: We treat the graph as undirected for triangle finding, 
        # but orientations matter for the operator.
        adj_set = set()
        for u, v in edge_list:
            adj_set.add(tuple(sorted((u, v))))
            
        for idx, (u, v) in enumerate(edge_list):
            # Find common neighbors
            u_neighbors = set(self.adjacency.getrow(u).indices)
            v_neighbors = set(self.adjacency.getrow(v).indices)
            common = u_neighbors.intersection(v_neighbors)
            
            for w in common:
                # Check if we have a triangle u->v->w->u (or any permutation)
                # We normalize triangle storage as sorted tuple to avoid duplicates
                tri = tuple(sorted((u, v, w)))
                # Only add if edges exist (which they do by definition of common)
                # But we need to ensure we don't double count
                if w > v: # Force order to avoid duplicates
                    triangles.append((u, v, w))

        # Build d1 matrix: Shape (n_edges, n_triangles)
        if triangles:
            t_rows, t_cols, t_data = [], [], []
            for t_idx, (i, j, k) in enumerate(triangles):
                # Triangle [i,j,k] has boundary [j,k] - [i,k] + [i,j]
                # We need to map these edges to our edge indices and signs
                
                # Edges in the triangle (sorted order for lookup)
                edges_in_tri = [
                    (i, j), (j, k), (i, k)
                ]
                
                # Check orientation against our edge_list
                for u, v in edges_in_tri:
                    # Look for edge (u,v) or (v,u) in our edge list
                    # Since edge_list contains directed edges from trajectories,
                    # we need to be careful. 
                    # If (u,v) is in edge_list: sign +1
                    # If (v,u) is in edge_list: sign -1
                    # Wait, standard d1 definition:
                    # d1([i,j,k]) = [j,k] - [i,k] + [i,j]
                    
                    # Let's check which index this edge corresponds to
                    edge_idx = -1
                    sign = 0
                    
                    if (u, v) in edge_to_idx:
                        edge_idx = edge_to_idx[(u, v)]
                        sign = 1
                    elif (v, u) in edge_to_idx:
                        edge_idx = edge_to_idx[(v, u)]
                        sign = -1
                        
                    if edge_idx != -1:
                        # Correct signs for standard boundary operator
                        if (u,v) == (j,k): term_sign = 1
                        elif (u,v) == (i,k): term_sign = -1
                        elif (u,v) == (i,j): term_sign = 1
                        else: term_sign = 1 # Should match one of above if logic holds
                        
                        t_rows.append(edge_idx)
                        t_cols.append(t_idx)
                        t_data.append(sign * term_sign)

            d1 = csr_matrix((t_data, (t_rows, t_cols)), shape=(len(edge_list), len(triangles)))
        else:
            d1 = csr_matrix((len(edge_list), 0))

        # 3. Solve Hodge Decomposition
        # Y = d0(s) + d1*(v) + h
        # Y is the vector of edge weights (rank differences)
        Y = np.array([self.edge_weights[tuple(edge)] for edge in edge_list])
        
        # a) Gradient Component: Y_g = d0 s
        # Minimize ||Y - d0 s||^2 -> Normal eq: d0.T d0 s = d0.T Y
        # d0.T d0 is the graph Laplacian L0
        L0 = d0.T @ d0
        divergence = d0.T @ Y
        
        try:
            # s = potential on nodes
            s_potential = lsqr(L0, divergence)[0]
            Y_grad = d0 @ s_potential
        except:
            Y_grad = np.zeros_like(Y)
            s_potential = np.zeros(n)

        # b) Curl Component: Y_c = d1* v (if d1 exists)
        # Minimize ||Y - d1* v||^2? No, we decompose the residual.
        # Actually, if we want orthogonal decomposition:
        # Y_grad is projection onto im(d0)
        # Y_curl is projection onto im(d1*)? No, im(d1*) is orthogonal to ker(d1)
        # Let's use the property: Y = Y_g + Y_c + Y_h
        # Y_g = proj_im_d0 (Y)
        # Y_curl = proj_im_d1 (Y)? No curl is usually dual.
        # Standard: C1 = im(d0) + im(d1*) + harm
        
        # We already have Y_grad = proj_im_d0(Y).
        # Residual = Y - Y_grad = Y_curl + Y_h
        residual = Y - Y_grad
        
        Y_curl = np.zeros_like(Y)
        if d1.shape[1] > 0:
            # We want Y_curl \in im(d1). wait, curl is rotational.
            # In standard vector calculus, curl is rot.
            # Here, cyclic component is ker(d0.T).
            # Triangles are boundaries of 2-cells.
            # If we want to capture "local" rotation, we project onto im(d1)? No d1 maps tri->edge.
            # So a flow in im(d1) is a sum of triangle boundaries. 
            # Yes, that's "curl" in the sense of bounding a surface.
            
            # Minimize ||residual - d1 v||^2 -> Normal eq: d1.T d1 v = d1.T residual
            L1_down = d1.T @ d1
            if L1_down.shape[0] > 0:
                curl_potential = lsqr(L1_down, d1.T @ residual)[0]
                Y_curl = d1 @ curl_potential
        
        # c) Harmonic Component
        # Y_h = Y - Y_grad - Y_curl
        Y_harm = Y - Y_grad - Y_curl
        
        # 4. Map back to embedding space for visualization/gradients
        # We need to turn these scalar edge flows back into vector fields in R^d
        
        gradient_field = np.zeros(self.embed_dim)
        curl_field = np.zeros(self.embed_dim)
        harmonic_field = np.zeros(self.embed_dim)
        
        for idx, (u, v) in enumerate(edge_list):
            edge_vec = self.embeddings[v] - self.embeddings[u]
            edge_len = np.linalg.norm(edge_vec) + 1e-8
            edge_dir = edge_vec / edge_len
            
            # Weighted average of edge flows directions
            gradient_field += Y_grad[idx] * edge_dir
            curl_field += Y_curl[idx] * edge_dir
            harmonic_field += Y_harm[idx] * edge_dir
            
        if len(edge_list) > 0:
            gradient_field /= len(edge_list)
            curl_field /= len(edge_list)
            harmonic_field /= len(edge_list)

        # H1 magnitude is norm of harmonic component (global holes)
        # Note: Previous code used curl for H1, but rigorous def is harmonic
        h1_magnitude = np.linalg.norm(Y_harm)
        
        # Cache
        self._gradient_field = gradient_field
        self._curl_field = curl_field
        self._harmonic_field = harmonic_field
        self._h1_magnitude = h1_magnitude
        
        # Detect Condorcet cycles (preference loops with non-zero circulation)
        # We can use the Harmonic flow to guide this
        condorcet_cycles = self._detect_condorcet_cycles(s_potential)
        
        return TopologicalGradient(
            gradient_component=gradient_field,
            curl_component=curl_field,
            harmonic_component=harmonic_field,
            h1_magnitude=h1_magnitude,
            condorcet_cycles=condorcet_cycles,
        )
    
    def _detect_condorcet_cycles(self, potentials: np.ndarray) -> List[CondorcetCycle]:
        """
        Detect Condorcet cycles in the preference graph.
        
        A Condorcet cycle occurs when A > B > C > A (preference is cyclic).
        We detect this by finding cycles where the sum of edge weights
        (rank differences) is non-zero.
        
        Args:
            potentials: Node potentials from Hodge decomposition
            
        Returns:
            List of detected Condorcet cycles
        """
        cycles = []
        n = len(self.feedback_items)
        
        if n < 3:
            return cycles
        
        # Build directed preference graph from rank differences
        # Edge (i,j) exists if item i is connected to item j
        # Edge weight = rank[j] - rank[i] (positive = j preferred)
        
        # Find simple cycles using DFS (limit to small cycles for efficiency)
        MAX_CYCLE_LENGTH = 5
        visited_cycles = set()
        
        def find_cycles_from(start: int, path: List[int], visited: set):
            """DFS to find cycles starting from a node."""
            if len(path) > MAX_CYCLE_LENGTH:
                return
            
            current = path[-1]
            
            # Check neighbors via adjacency
            if self.adjacency is None:
                return
                
            row = self.adjacency.getrow(current).toarray().flatten()
            neighbors = np.where(row > 0)[0]
            
            for neighbor in neighbors:
                if neighbor == start and len(path) >= 3:
                    # Found a cycle back to start
                    cycle_key = tuple(sorted(path))
                    if cycle_key not in visited_cycles:
                        visited_cycles.add(cycle_key)
                        
                        # Compute circulation (sum of rank differences around cycle)
                        circulation = 0.0
                        for i in range(len(path)):
                            u = path[i]
                            v = path[(i + 1) % len(path)] if i < len(path) - 1 else start
                            circulation += self.feedback_items[v].rank - self.feedback_items[u].rank
                        
                        # Non-zero circulation indicates Condorcet paradox
                        if abs(circulation) > 0.01:
                            cycle_items = [
                                self.feedback_items[idx].action_text[:30] 
                                for idx in path
                            ]
                            cycles.append(CondorcetCycle(
                                cycle_indices=path.copy(),
                                cycle_items=cycle_items,
                                circulation=circulation,
                            ))
                
                elif neighbor not in visited and neighbor != start:
                    visited.add(neighbor)
                    path.append(neighbor)
                    find_cycles_from(start, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        # Search from each node (limit for efficiency)
        for start in range(min(n, 20)):  # Limit starting nodes
            find_cycles_from(start, [start], {start})
        
        # Sort by circulation magnitude
        cycles.sort(key=lambda c: abs(c.circulation), reverse=True)
        
        return cycles[:10]  # Return top 10 cycles
    
    def get_topological_gradient_at(self, state_text: str) -> np.ndarray:
        """
        Get the Hodge gradient direction at a given state.
        
        This is the "clean" reward direction with inconsistencies removed.
        """
        if self._gradient_field is None:
            self.compute_hodge_decomposition()
        
        # Embed the query state
        query_embedding = self.embedding_model.encode([state_text])[0]
        
        # Find nearest neighbors in the manifold
        if self.embeddings is None or len(self.embeddings) == 0:
            return self._gradient_field  # Return global gradient
        
        # Cosine similarity to find relevant local structure
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query_embedding)
        similarities = (self.embeddings @ query_embedding) / (norms.flatten() * query_norm + 1e-8)
        
        # Weight the gradient by similarity
        weights = np.maximum(similarities, 0)
        weights = weights / (weights.sum() + 1e-8)
        
        # Local gradient is weighted combination
        local_gradient = self._gradient_field.copy()
        
        # Add contribution from local rank structure
        for i, w in enumerate(weights):
            if w > 0.1:  # Only consider significant neighbors
                local_contrib = self.feedback_items[i].rank * self.embeddings[i]
                local_contrib = local_contrib / (np.linalg.norm(local_contrib) + 1e-8)
                local_gradient += w * local_contrib
        
        return local_gradient / (np.linalg.norm(local_gradient) + 1e-8)
    
    def get_local_geometry(self, state_text: str) -> Dict[str, float]:
        """
        Get local geometric properties for adaptive trust regions.
        
        Returns:
            Dict containing:
            - curvature: Estimated local curvature
            - h1_magnitude: Global inconsistency
            - black_hole_proximity: 0-1 score of nearness to black holes
        """
        if self._h1_magnitude is None:
            self.compute_hodge_decomposition()
            
        embedding = self.embedding_model.encode([state_text])[0]
        
        # Estimate black hole proximity
        proximity = 0.0
        for point in self.manifold_points:
            if point.is_black_hole:
                dist = np.linalg.norm(embedding - point.embedding)
                proximity = max(proximity, np.exp(-dist))
        
        # Estimate local curvature via SVD of neighborhood
        # High tail variance = high curvature/noise
        curvature = 0.0
        if self.embeddings is not None and len(self.embeddings) > 5:
            # Find k nearest neighbors
            dists = np.linalg.norm(self.embeddings - embedding, axis=1)
            nn_indices = np.argsort(dists)[:10]  # k=10
            
            if len(nn_indices) >= 5:
                neighbors = self.embeddings[nn_indices]
                # Center
                centered = neighbors - np.mean(neighbors, axis=0)
                # SVD
                try:
                    _, s, _ = np.linalg.svd(centered, full_matrices=False)
                    # Curvature proxy: ratio of non-dominant singular values
                    # If locally linear (flat), s[0] dominates.
                    if s[0] > 1e-6:
                        curvature = np.sum(s[1:]) / np.sum(s)
                except np.linalg.LinAlgError:
                    pass

        return {
            "curvature": float(curvature),
            "h1_magnitude": self._h1_magnitude,
            "black_hole_proximity": proximity,
        }

    def score_action(
        self,
        state_text: str,
        action_text: str,
        action_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """
        Score an action by its alignment with the Hodge gradient.
        
        Higher score = action moves along the "clean" reward direction.
        """
        gradient = self.get_topological_gradient_at(state_text)
        
        if action_embedding is None:
            action_embedding = self.embedding_model.encode([action_text])[0]
        
        # Cosine similarity between action direction and gradient
        action_norm = np.linalg.norm(action_embedding)
        gradient_norm = np.linalg.norm(gradient)
        
        if action_norm < 1e-8 or gradient_norm < 1e-8:
            return 0.0
        
        alignment = np.dot(action_embedding, gradient) / (action_norm * gradient_norm)
        
        return float(alignment)
    
    def rank_actions(
        self,
        state_text: str,
        actions: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Rank multiple actions by their topological gradient alignment.
        
        Returns:
            List of (action, score) tuples, sorted by score descending.
        """
        action_embeddings = self.embedding_model.encode(actions)
        
        scores = []
        for action, embedding in zip(actions, action_embeddings):
            score = self.score_action(state_text, action, embedding)
            scores.append((action, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def get_consistency_report(self) -> Dict:
        """
        Generate a report on feedback consistency.
        
        Returns:
            Dict with H¹ magnitude, detected cycles, outlier evaluators, etc.
        """
        if self._h1_magnitude is None:
            self.compute_hodge_decomposition()
        
        report = {
            "h1_magnitude": self._h1_magnitude,
            "is_consistent": self._h1_magnitude < 0.1,
            "total_feedback_items": len(self.feedback_items),
            "unique_evaluators": len(set(
                item.evaluator_id for item in self.feedback_items 
                if item.evaluator_id
            )),
        }
        
        # Detect evaluator disagreement
        if report["unique_evaluators"] > 1:
            evaluator_ranks = {}
            for item in self.feedback_items:
                if item.evaluator_id:
                    if item.evaluator_id not in evaluator_ranks:
                        evaluator_ranks[item.evaluator_id] = []
                    evaluator_ranks[item.evaluator_id].append(item.rank)
            
            # Compute variance across evaluators
            mean_ranks = {k: np.mean(v) for k, v in evaluator_ranks.items()}
            overall_variance = np.var(list(mean_ranks.values()))
            report["evaluator_variance"] = float(overall_variance)
            report["evaluator_means"] = mean_ranks
        
        return report
    
    def identify_black_holes(self, threshold: float = -0.5) -> List[ManifoldPoint]:
        """
        Identify regions with consistently negative feedback (black holes).
        
        These are states where most feedback is strongly negative.
        """
        self._compute_embeddings()
        
        black_holes = []
        for point in self.manifold_points:
            avg_rank = np.mean([item.rank for item in point.feedback_items])
            if avg_rank < threshold:
                point.is_black_hole = True
                black_holes.append(point)
        
        return black_holes
    
    def identify_cliffs(self, gradient_threshold: float = 0.8) -> List[Tuple[int, int]]:
        """
        Identify edges with steep rank gradients (cliffs).
        
        Returns:
            List of (from_idx, to_idx) tuples representing cliff edges.
        """
        cliffs = []
        for (i, j), weight in self.edge_weights.items():
            if abs(weight) > gradient_threshold:
                cliffs.append((i, j))
                self.manifold_points[i].is_cliff = True
                self.manifold_points[j].is_cliff = True
        
        return cliffs

    # =========================================================================
    # SCHWARZSCHILD DETECTION: Identifying Black Holes from Preference Data
    # =========================================================================
    
    def schwarzschild_detection(
        self,
        sink_threshold: float = 0.5,
        cliff_threshold: float = 0.8,
        explicit_catastrophic: Optional[List[int]] = None,
        alpha: float = 1.0,
        beta: float = 0.5,
    ) -> Dict[int, float]:
        """
        Comprehensive black hole detection using three complementary methods.
        
        This implements the "Schwarzschild Detection Algorithm" from Section 3.1.5:
        1. Outlier Sink Detection (topological)
        2. Cliff Detection (gradient magnitude)
        3. Explicit Catastrophic Markers
        
        Args:
            sink_threshold: Minimum inflow/outflow ratio to flag as sink
            cliff_threshold: Minimum gradient magnitude for cliff detection
            explicit_catastrophic: List of indices explicitly marked catastrophic
            alpha: Weight for cliff magnitude in Schwarzschild radius
            beta: Weight for negative density in Schwarzschild radius
            
        Returns:
            Dict mapping black hole indices to their Schwarzschild radii
        """
        self._build_graph()
        
        black_holes: Dict[int, float] = {}
        n = len(self.feedback_items)
        
        # Method 1: Outlier Sink Detection
        sink_candidates = self._detect_sinks(sink_threshold)
        for idx, inflow in sink_candidates:
            # Schwarzschild radius proportional to inflow magnitude
            r_s = alpha * inflow
            black_holes[idx] = r_s
            self.manifold_points[idx].is_black_hole = True
        
        # Method 2: Cliff Detection
        cliff_candidates = self._detect_cliff_targets(cliff_threshold)
        for idx, cliff_magnitude in cliff_candidates:
            r_s = alpha * cliff_magnitude
            if idx in black_holes:
                black_holes[idx] = max(black_holes[idx], r_s)
            else:
                black_holes[idx] = r_s
                self.manifold_points[idx].is_black_hole = True
        
        # Method 3: Explicit Catastrophic Markers
        if explicit_catastrophic:
            for idx in explicit_catastrophic:
                if 0 <= idx < n:
                    # Explicit markers get maximum Schwarzschild radius
                    r_s = alpha * 2.0 + beta * 1.0  # Large fixed radius
                    if idx in black_holes:
                        black_holes[idx] = max(black_holes[idx], r_s)
                    else:
                        black_holes[idx] = r_s
                    self.manifold_points[idx].is_black_hole = True
        
        # Add negative density contribution to all detected black holes
        for idx in black_holes:
            neg_density = self._compute_negative_density(idx)
            black_holes[idx] += beta * neg_density
        
        return black_holes
    
    def _detect_sinks(self, threshold: float) -> List[Tuple[int, float]]:
        """
        Detect vertices that act as sinks in the preference flow.
        
        A sink has many edges pointing toward it (low rank) and few pointing away.
        This corresponds to negative divergence in the gradient field.
        
        Returns:
            List of (vertex_index, inflow_magnitude) tuples
        """
        n = len(self.feedback_items)
        inflow = np.zeros(n)
        outflow = np.zeros(n)
        
        for (i, j), weight in self.edge_weights.items():
            if weight > 0:  # j is preferred over i → flow from i to j
                outflow[i] += weight
                inflow[j] += weight
            else:  # i is preferred over j → flow from j to i
                outflow[j] += abs(weight)
                inflow[i] += abs(weight)
        
        sinks = []
        for i in range(n):
            # Sink condition: high inflow, low outflow
            if inflow[i] > threshold and (outflow[i] < 0.1 or inflow[i] / (outflow[i] + 1e-8) > 3):
                sinks.append((i, inflow[i]))
        
        return sinks
    
    def _detect_cliff_targets(self, threshold: float) -> List[Tuple[int, float]]:
        """
        Detect vertices reachable only via steep cliff edges.
        
        A cliff is an edge with extreme gradient (large rank drop).
        The target of a cliff edge is a candidate black hole.
        
        Returns:
            List of (vertex_index, max_cliff_magnitude) tuples
        """
        cliff_targets: Dict[int, float] = {}
        
        for (i, j), weight in self.edge_weights.items():
            # Cliff: large negative weight (steep drop from i to j)
            if weight < -threshold:
                cliff_mag = abs(weight)
                if j in cliff_targets:
                    cliff_targets[j] = max(cliff_targets[j], cliff_mag)
                else:
                    cliff_targets[j] = cliff_mag
        
        return list(cliff_targets.items())
    
    def _compute_negative_density(self, idx: int, radius: int = 2) -> float:
        """
        Compute density of negative feedback in neighborhood of a vertex.
        
        Returns:
            Float in [0, 1] representing proportion of low-ranked neighbors
        """
        if self.adjacency is None:
            return 0.0
        
        # BFS to find neighbors within radius
        visited = {idx}
        frontier = [idx]
        
        for _ in range(radius):
            next_frontier = []
            for node in frontier:
                row = self.adjacency.getrow(node).toarray().flatten()
                neighbors = np.where(row > 0)[0]
                for n in neighbors:
                    if n not in visited:
                        visited.add(n)
                        next_frontier.append(n)
            frontier = next_frontier
        
        if len(visited) <= 1:
            return 0.0
        
        # Compute proportion of low-ranked items in neighborhood
        low_rank_count = sum(
            1 for v in visited 
            if self.feedback_items[v].rank < 0.3
        )
        
        return low_rank_count / len(visited)
    
    def add_catastrophic_feedback(
        self,
        state_text: str,
        action_text: str,
        critique: Optional[str] = None,
        severity: float = 1.0,
    ) -> int:
        """
        Add explicitly catastrophic feedback (black hole marker).
        
        This is the recommended way to mark responses as "forbidden" rather
        than just "bad". Creates a feedback item with negative rank.
        
        Args:
            state_text: The state/prompt
            action_text: The catastrophic response
            critique: Optional explanation of why this is catastrophic
            severity: How catastrophic (0-1, higher = worse)
            
        Returns:
            Index of the added feedback item
        """
        # Use negative rank to signal catastrophic
        item = FeedbackItem(
            state_text=state_text,
            action_text=action_text,
            next_state_text=None,
            rank=-severity,  # Negative rank signals catastrophic
            critique=critique or "CATASTROPHIC: This response is forbidden.",
            evaluator_id="safety_system",
        )
        self.feedback_items.append(item)
        self._invalidate_cache()
        
        return len(self.feedback_items) - 1
    
    def get_safety_metric_at(
        self, 
        state_text: str, 
        black_holes: Dict[int, float],
        failure_clusters: Optional[List[Dict]] = None
    ) -> float:
        """
        Compute the safety metric g(s) at a given state.
        
        g(s) = 1 + sum_{b in B} alpha / d(s, b)^k
        
        Supports both point-based black holes and semantic failure clusters.
        
        Args:
            state_text: The query state
            black_holes: Dict mapping point indices to radii (from schwarzschild_detection)
            failure_clusters: Optional list of cluster dicts (from cluster_failure_modes)
                              Each dict has {'centroid': np.array, 'radius': float}
            
        Returns:
            The metric value (higher = more dangerous)
        """
        if not black_holes and not failure_clusters:
            return 1.0
        
        query_embedding = self.embedding_model.encode([state_text])[0]
        self._compute_embeddings()
        
        metric = 1.0
        k = 2  # Schwarzschild exponent
        
        # Contribution from individual black hole points
        if black_holes:
            for bh_idx, r_s in black_holes.items():
                bh_embedding = self.embeddings[bh_idx]
                dist = np.linalg.norm(query_embedding - bh_embedding)
                
                if dist < r_s:
                    return float('inf')
                metric += r_s / (dist ** k)
        
        # Contribution from semantic failure clusters (Supermassive Black Holes)
        if failure_clusters:
            for cluster in failure_clusters:
                centroid = cluster['centroid']
                r_c = cluster['radius']
                dist = np.linalg.norm(query_embedding - centroid)
                
                # Check if inside cluster event horizon
                if dist < r_c:
                    return float('inf')
                
                # Repulsion from cluster center
                # We treat the cluster as a "supermassive" black hole
                # The effective mass is proportional to the cluster radius
                metric += (r_c * 2.0) / (dist ** k)
        
        return metric

    def cluster_failure_modes(
        self, 
        black_holes: Dict[int, float],
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Group individual black holes into Semantic Failure Clusters.
        
        This aggregates scattered failure points (e.g., specific deceptive responses)
        into coherent regions (e.g., "The Deception Cluster") with a unified
        event horizon.
        
        Args:
            black_holes: Dict mapping black hole indices to their individual radii
            similarity_threshold: Cosine similarity to link failures
            
        Returns:
            List of cluster dictionaries:
            {
                'centroid': np.ndarray,      # Center of the failure mode
                'radius': float,             # Radius of the event horizon
                'member_indices': List[int], # Original points
                'name': str                  # Auto-generated ID
            }
        """
        if not black_holes:
            return []
            
        self._compute_embeddings()
        
        # Convert dict to lists for indexing
        bh_indices = list(black_holes.keys())
        bh_radii = list(black_holes.values())
        bh_embeddings = self.embeddings[bh_indices]
        n_bh = len(bh_indices)
        
        # 1. Build adjacency for black holes based on similarity
        # Normalize for cosine similarity
        norms = np.linalg.norm(bh_embeddings, axis=1, keepdims=True)
        normalized = bh_embeddings / (norms + 1e-8)
        sim_matrix = normalized @ normalized.T
        
        # 2. Find connected components (clusters)
        visited = set()
        clusters = []
        
        for i in range(n_bh):
            if i in visited:
                continue
                
            # BFS to find component
            component_indices_in_list = [i]  # Indices into bh_indices list
            queue = [i]
            visited.add(i)
            
            while queue:
                curr = queue.pop(0)
                # Find neighbors in the black hole set
                for j in range(n_bh):
                    if j not in visited and sim_matrix[curr, j] > similarity_threshold:
                        visited.add(j)
                        queue.append(j)
                        component_indices_in_list.append(j)
            
            # 3. Compute cluster properties
            member_original_indices = [bh_indices[idx] for idx in component_indices_in_list]
            member_embeddings = bh_embeddings[component_indices_in_list]
            
            # Centroid
            centroid = np.mean(member_embeddings, axis=0)
            
            # Radius: Must enclose all individual event horizons
            # R_cluster = max(dist(centroid, member) + r_member) + margin
            max_extent = 0.0
            
            for list_idx, original_idx in zip(component_indices_in_list, member_original_indices):
                embedding = bh_embeddings[list_idx]
                r_member = black_holes[original_idx]
                dist_to_centroid = np.linalg.norm(embedding - centroid)
                
                extent = dist_to_centroid + r_member
                if extent > max_extent:
                    max_extent = extent
            
            # Add small epsilon margin for robustness
            cluster_radius = max_extent * 1.05
            
            clusters.append({
                'centroid': centroid,
                'radius': cluster_radius,
                'member_indices': member_original_indices,
                'name': f"Cluster_{len(clusters)+1}"
            })
            
        return clusters


class GeoDPOLoss:
    """
    Geodesic Direct Preference Optimization loss.
    
    Instead of standard DPO:
        L = -log σ(β(r(y_w) - r(y_l)))
    
    We use:
        L = -log σ(β · ⟨∇φ, Δe⟩)
    
    Where:
        - ∇φ is the Hodge gradient (clean reward direction)
        - Δe = e(y_w) - e(y_l) is embedding difference
    """
    
    def __init__(
        self,
        hodge_critic: HodgeCritic,
        beta: float = 0.1,
    ):
        self.critic = hodge_critic
        self.beta = beta
    
    def compute_loss(
        self,
        state_text: str,
        preferred_action: str,
        dispreferred_action: str,
    ) -> float:
        """
        Compute GeoDPO loss for a preference pair.
        
        Args:
            state_text: The state where preference was expressed
            preferred_action: The better action (y_w)
            dispreferred_action: The worse action (y_l)
        
        Returns:
            Loss value (lower is better alignment)
        """
        import torch
        import torch.nn.functional as F
        
        # Get the Hodge gradient at this state
        gradient = self.critic.get_topological_gradient_at(state_text)
        
        # Embed the actions
        embeddings = self.critic.embedding_model.encode([preferred_action, dispreferred_action])
        delta_e = embeddings[0] - embeddings[1]
        
        # Compute alignment
        alignment = np.dot(gradient, delta_e)
        alignment = alignment / (np.linalg.norm(gradient) * np.linalg.norm(delta_e) + 1e-8)
        
        # DPO-style loss
        logit = self.beta * alignment
        loss = -np.log(1 / (1 + np.exp(-logit)) + 1e-8)
        
        return float(loss)
    
    def compute_batch_loss(
        self,
        batch: List[Dict],
    ) -> float:
        """
        Compute average GeoDPO loss over a batch.
        
        Args:
            batch: List of {"state": str, "preferred": str, "dispreferred": str}
        
        Returns:
            Average loss
        """
        losses = []
        for item in batch:
            loss = self.compute_loss(
                item["state"],
                item["preferred"],
                item["dispreferred"],
            )
            losses.append(loss)
        
        return np.mean(losses)
