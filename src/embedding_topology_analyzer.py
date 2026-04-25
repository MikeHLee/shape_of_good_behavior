"""
Embedding Topology Analyzer: Interpretability for Semantic RL

This module provides interpretable analysis of embedding space topology for
understanding reward manifold structure, consistency, and safety properties.

Key Features:
1. **Hodge Decomposition Visualization**: Separate gradient (learnable), curl (noise),
   and harmonic (global structure) components
2. **Topological Feature Extraction**: Persistent homology-inspired features
3. **Cluster Semantic Analysis**: Automatic labeling of embedding regions
4. **Trajectory Flow Analysis**: Visualize policy behavior on the manifold
5. **Consistency Diagnostics**: Condorcet cycle detection and H¹ analysis
6. **Safety Region Mapping**: Black holes, cliffs, and safe corridors

Mathematical Background:
- Hodge decomposition: ω = dφ + δψ + h (exact + coexact + harmonic)
- H¹ ≠ 0 indicates cyclic inconsistencies (Condorcet paradoxes)
- Riemannian metric encodes safety via geodesic distance to black holes
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from collections import defaultdict
import warnings


@dataclass
class TopologicalFeatures:
    """Extracted topological features from an embedding space."""
    # Basic statistics
    n_points: int
    embedding_dim: int
    
    # Hodge decomposition magnitudes
    gradient_magnitude: float      # ||∇φ|| - learnable component
    curl_magnitude: float          # ||∇×ψ|| - inconsistency/noise
    harmonic_magnitude: float      # ||h|| - global structure
    h1_cohomology: float           # H¹ magnitude (0 = consistent)
    
    # Connectivity
    n_connected_components: int    # H⁰ dimension
    graph_density: float           # Edge density
    avg_degree: float              # Average node degree
    
    # Curvature statistics
    mean_curvature: float
    max_curvature: float
    curvature_variance: float
    
    # Safety statistics
    n_black_holes: int
    n_cliffs: int
    safe_region_fraction: float
    
    # Cluster statistics
    n_clusters: int
    cluster_purity: float          # How well-separated clusters are
    
    def summary(self) -> str:
        """Human-readable summary of topological features."""
        lines = [
            "=" * 60,
            "EMBEDDING TOPOLOGY ANALYSIS",
            "=" * 60,
            f"Points: {self.n_points} in ℝ^{self.embedding_dim}",
            "",
            "HODGE DECOMPOSITION:",
            f"  Gradient (learnable):    {self.gradient_magnitude:.4f}",
            f"  Curl (inconsistency):    {self.curl_magnitude:.4f}",
            f"  Harmonic (global):       {self.harmonic_magnitude:.4f}",
            f"  H¹ Cohomology:           {self.h1_cohomology:.4f}",
            "",
            "CONSISTENCY STATUS:",
        ]
        
        if self.h1_cohomology < 0.05:
            lines.append("  ✓ Preferences are CONSISTENT (H¹ ≈ 0)")
        elif self.h1_cohomology < 0.2:
            lines.append("  ⚠ Minor inconsistencies detected")
        else:
            lines.append("  ✗ SIGNIFICANT inconsistencies (possible Condorcet cycles)")
        
        lines.extend([
            "",
            "CONNECTIVITY:",
            f"  Connected components:    {self.n_connected_components}",
            f"  Graph density:           {self.graph_density:.4f}",
            f"  Average degree:          {self.avg_degree:.2f}",
            "",
            "GEOMETRY:",
            f"  Mean curvature:          {self.mean_curvature:.4f}",
            f"  Max curvature:           {self.max_curvature:.4f}",
            f"  Curvature variance:      {self.curvature_variance:.4f}",
            "",
            "SAFETY:",
            f"  Black holes detected:    {self.n_black_holes}",
            f"  Cliffs detected:         {self.n_cliffs}",
            f"  Safe region fraction:    {self.safe_region_fraction:.2%}",
            "",
            "CLUSTERING:",
            f"  Number of clusters:      {self.n_clusters}",
            f"  Cluster purity:          {self.cluster_purity:.4f}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


@dataclass
class InterpretableRegion:
    """A semantically labeled region of the embedding space."""
    cluster_id: int
    centroid: np.ndarray
    radius: float                   # Approximate region size
    n_points: int
    
    # Semantic labels (extracted from text)
    keywords: List[str]
    label: str                      # Auto-generated label
    
    # Reward statistics
    mean_reward: float
    reward_variance: float
    is_black_hole: bool
    is_cliff_region: bool
    
    # Flow statistics
    gradient_direction: np.ndarray  # Average Hodge gradient in region
    flow_coherence: float           # How aligned gradients are (0-1)
    
    def __str__(self) -> str:
        safety = ""
        if self.is_black_hole:
            safety = " [BLACK HOLE]"
        elif self.is_cliff_region:
            safety = " [CLIFF]"
        return f"Region '{self.label}' ({self.n_points} pts, reward={self.mean_reward:.2f}){safety}"


@dataclass
class TrajectoryAnalysis:
    """Analysis of a trajectory through the embedding space."""
    trajectory_id: str
    n_steps: int
    
    # Path statistics
    total_length: float             # Geodesic length
    euclidean_length: float         # Straight-line length
    tortuosity: float               # total/euclidean (1 = straight)
    
    # Reward progression
    cumulative_reward: float
    reward_trend: str               # "increasing", "decreasing", "oscillating"
    
    # Hodge alignment
    mean_gradient_alignment: float  # Average alignment with ∇φ
    curl_exposure: float            # How much trajectory crosses curl regions
    
    # Safety
    min_black_hole_distance: float
    n_cliff_crossings: int
    safety_score: float             # 0-1, higher = safer
    
    # Regions visited
    regions_visited: List[str]
    
    def summary(self) -> str:
        lines = [
            f"Trajectory: {self.trajectory_id} ({self.n_steps} steps)",
            f"  Length: {self.total_length:.2f} (tortuosity: {self.tortuosity:.2f})",
            f"  Cumulative Reward: {self.cumulative_reward:.2f} ({self.reward_trend})",
            f"  Gradient Alignment: {self.mean_gradient_alignment:.2f}",
            f"  Safety Score: {self.safety_score:.2f}",
            f"  Regions: {' → '.join(self.regions_visited[:5])}{'...' if len(self.regions_visited) > 5 else ''}",
        ]
        return "\n".join(lines)


class EmbeddingTopologyAnalyzer:
    """
    Main analyzer for semantic RL embedding spaces.
    
    Usage:
        analyzer = EmbeddingTopologyAnalyzer(embedding_model)
        analyzer.fit(states, actions, rewards, texts)
        features = analyzer.extract_features()
        print(features.summary())
        
        regions = analyzer.get_interpretable_regions()
        trajectory_analysis = analyzer.analyze_trajectory(trajectory)
    """
    
    def __init__(
        self,
        embedding_model: Any,
        n_clusters: int = 8,
        similarity_threshold: float = 0.7,
        black_hole_threshold: float = -0.3,
        cliff_threshold: float = 0.5,
    ):
        """
        Args:
            embedding_model: Model with .encode(texts) -> np.ndarray
            n_clusters: Number of semantic clusters to identify
            similarity_threshold: Cosine similarity for graph edges
            black_hole_threshold: Reward below this marks black holes
            cliff_threshold: Gradient magnitude above this marks cliffs
        """
        self.embedding_model = embedding_model
        self.n_clusters = n_clusters
        self.similarity_threshold = similarity_threshold
        self.black_hole_threshold = black_hole_threshold
        self.cliff_threshold = cliff_threshold
        
        # Data storage
        self.embeddings: Optional[np.ndarray] = None
        self.texts: List[str] = []
        self.rewards: List[float] = []
        self.actions: List[str] = []
        
        # Computed structures
        self.adjacency: Optional[csr_matrix] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.hodge_gradient: Optional[np.ndarray] = None
        self.hodge_curl: Optional[np.ndarray] = None
        self.hodge_harmonic: Optional[np.ndarray] = None
        self.local_curvatures: Optional[np.ndarray] = None
        
        # Regions
        self.regions: List[InterpretableRegion] = []
        self.black_hole_indices: List[int] = []
        self.cliff_indices: List[int] = []
    
    def fit(
        self,
        states: List[np.ndarray],
        actions: List[str],
        rewards: List[float],
        texts: List[str],
    ):
        """
        Fit the analyzer to trajectory data.
        
        Args:
            states: State embeddings (or will be computed from texts)
            actions: Action texts
            rewards: Scalar rewards
            texts: State descriptions (for semantic clustering)
        """
        self.texts = texts
        self.rewards = rewards
        self.actions = actions
        
        # Compute embeddings if not provided
        if len(states) > 0 and states[0] is not None:
            self.embeddings = np.array(states)
        else:
            self.embeddings = self.embedding_model.encode(texts)
        
        # Build graph structure
        self._build_similarity_graph()
        
        # Compute Hodge decomposition
        self._compute_hodge_decomposition()
        
        # Compute local curvatures
        self._compute_local_curvatures()
        
        # Cluster and label regions
        self._cluster_and_label()
        
        # Identify safety regions
        self._identify_safety_regions()
        
        return self
    
    def _build_similarity_graph(self):
        """Build similarity graph from embeddings."""
        n = len(self.embeddings)
        if n < 2:
            self.adjacency = csr_matrix((n, n))
            return
        
        # Normalize embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized = self.embeddings / (norms + 1e-8)
        
        # Compute pairwise cosine similarities
        similarities = normalized @ normalized.T
        
        # Build adjacency matrix (threshold + sequential connections)
        adj = lil_matrix((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Similarity-based edge
                if similarities[i, j] > self.similarity_threshold:
                    adj[i, j] = similarities[i, j]
                    adj[j, i] = similarities[i, j]
            
            # Sequential edge (trajectory structure)
            if i < n - 1:
                adj[i, i + 1] = max(adj[i, i + 1], 0.5)
                adj[i + 1, i] = max(adj[i + 1, i], 0.5)
        
        self.adjacency = adj.tocsr()
    
    def _compute_hodge_decomposition(self):
        """
        Compute Hodge decomposition of the reward field using boundary operators.
        
        Decomposes reward differences into:
        - Gradient: ∇φ (consistent, learnable direction)
        - Curl: ∇×ψ (inconsistencies, cycles, local triangles)
        - Harmonic: h (global topological structure, holes)
        """
        n = len(self.embeddings)
        if n < 2:
            self.hodge_gradient = np.zeros(self.embeddings.shape[1] if n > 0 else 1)
            self.hodge_curl = np.zeros_like(self.hodge_gradient)
            self.hodge_harmonic = np.zeros_like(self.hodge_gradient)
            return
        
        embed_dim = self.embeddings.shape[1]
        
        # 1. Construct 0-boundary operator (d0): edges -> nodes
        edge_list = []
        rows, cols = self.adjacency.nonzero()
        # Only take i < j to strictly define edges once
        for i, j in zip(rows, cols):
            if i < j:
                edge_list.append((i, j))
        
        edge_to_idx = {edge: i for i, edge in enumerate(edge_list)}
        n_edges = len(edge_list)
        
        if n_edges == 0:
            return

        # d0 matrix construction
        d0_rows, d0_cols, d0_data = [], [], []
        for i, (u, v) in enumerate(edge_list):
            d0_rows.extend([i, i])
            d0_cols.extend([u, v])
            d0_data.extend([-1, 1]) # v - u
            
        d0 = csr_matrix((d0_data, (d0_rows, d0_cols)), shape=(n_edges, n))
        
        # 2. Construct 1-boundary operator (d1): triangles -> edges
        triangles = []
        
        # Find triangles
        for idx, (u, v) in enumerate(edge_list):
            u_neighbors = set(self.adjacency.getrow(u).indices)
            v_neighbors = set(self.adjacency.getrow(v).indices)
            common = u_neighbors.intersection(v_neighbors)
            
            for w in common:
                if w > v: # Avoid duplicates, force sorted order u < v < w
                    triangles.append((u, v, w))
        
        if triangles:
            d1_rows, d1_cols, d1_data = [], [], []
            for t_idx, (i, j, k) in enumerate(triangles):
                # Triangle [i,j,k] boundary = [j,k] - [i,k] + [i,j]
                edges_in_tri = [(i, j), (j, k), (i, k)]
                
                for u, v in edges_in_tri:
                    sign = 0
                    edge_idx = -1
                    
                    if (u, v) in edge_to_idx:
                        edge_idx = edge_to_idx[(u, v)]
                        sign = 1
                    elif (v, u) in edge_to_idx:
                        edge_idx = edge_to_idx[(v, u)]
                        sign = -1
                        
                    if edge_idx != -1:
                        # Boundary signs
                        if (u,v) == (j,k): term_sign = 1
                        elif (u,v) == (i,k): term_sign = -1
                        elif (u,v) == (i,j): term_sign = 1
                        else: term_sign = 1
                        
                        d1_rows.append(edge_idx)
                        d1_cols.append(t_idx)
                        d1_data.append(sign * term_sign)
            
            d1 = csr_matrix((d1_data, (d1_rows, d1_cols)), shape=(n_edges, len(triangles)))
        else:
            d1 = csr_matrix((n_edges, 0))

        # 3. Solve Hodge Decomposition
        # Y is the vector of edge flows (reward differences)
        Y = np.zeros(n_edges)
        for i, (u, v) in enumerate(edge_list):
            Y[i] = self.rewards[v] - self.rewards[u]
            
        # a) Gradient Component: Y_g = d0 s
        # L0 = d0.T @ d0
        # Normal eq: d0.T d0 s = d0.T Y
        try:
            from scipy.sparse.linalg import lsqr
            L0 = d0.T @ d0
            divergence = d0.T @ Y
            s_potential = lsqr(L0, divergence)[0]
            Y_grad = d0 @ s_potential
        except Exception:
            Y_grad = np.zeros_like(Y)
        
        # b) Curl Component
        # residual = Y - Y_grad
        residual = Y - Y_grad
        Y_curl = np.zeros_like(Y)
        
        if d1.shape[1] > 0:
            L1_down = d1.T @ d1
            if L1_down.shape[0] > 0:
                try:
                    curl_potential = lsqr(L1_down, d1.T @ residual)[0]
                    Y_curl = d1 @ curl_potential
                except Exception:
                    pass
        
        # c) Harmonic Component
        Y_harm = Y - Y_grad - Y_curl
        
        # 4. Map back to embedding space vector fields
        gradient_field = np.zeros(embed_dim)
        curl_field = np.zeros(embed_dim)
        harmonic_field = np.zeros(embed_dim)
        
        for i, (u, v) in enumerate(edge_list):
            edge_vec = self.embeddings[v] - self.embeddings[u]
            edge_norm = np.linalg.norm(edge_vec) + 1e-8
            edge_dir = edge_vec / edge_norm
            
            gradient_field += Y_grad[i] * edge_dir
            curl_field += Y_curl[i] * edge_dir
            harmonic_field += Y_harm[i] * edge_dir
            
        if n_edges > 0:
            gradient_field /= n_edges
            curl_field /= n_edges
            harmonic_field /= n_edges
            
        self.hodge_gradient = gradient_field
        self.hodge_curl = curl_field
        self.hodge_harmonic = harmonic_field
    
    def _compute_local_curvatures(self):
        """
        Estimate local curvature at each point via neighborhood PCA.
        
        High curvature = nonlinear manifold region
        Low curvature = locally flat region
        """
        n = len(self.embeddings)
        self.local_curvatures = np.zeros(n)
        
        if n < 5:
            return
        
        for i in range(n):
            # Find k nearest neighbors
            dists = np.linalg.norm(self.embeddings - self.embeddings[i], axis=1)
            k = min(10, n - 1)
            nn_indices = np.argsort(dists)[1:k + 1]  # Exclude self
            
            if len(nn_indices) < 3:
                continue
            
            # Local PCA
            neighbors = self.embeddings[nn_indices]
            centered = neighbors - np.mean(neighbors, axis=0)
            
            try:
                _, s, _ = np.linalg.svd(centered, full_matrices=False)
                # Curvature proxy: fraction of variance in non-dominant directions
                if s[0] > 1e-8:
                    self.local_curvatures[i] = np.sum(s[1:]) / np.sum(s)
            except np.linalg.LinAlgError:
                pass
    
    def _cluster_and_label(self):
        """Cluster embeddings and generate semantic labels."""
        n = len(self.embeddings)
        
        if n < self.n_clusters:
            self.cluster_labels = np.zeros(n, dtype=int)
            self.regions = []
            return
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Cluster embeddings
            n_clusters = min(self.n_clusters, n)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_labels = kmeans.fit_predict(self.embeddings)
            
            # Extract keywords for each cluster
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            
            try:
                tfidf_matrix = vectorizer.fit_transform(self.texts)
                feature_names = np.array(vectorizer.get_feature_names_out())
            except Exception:
                feature_names = np.array([])
                tfidf_matrix = None
            
            # Build regions
            self.regions = []
            for cluster_id in range(n_clusters):
                mask = self.cluster_labels == cluster_id
                cluster_indices = np.where(mask)[0]
                
                if len(cluster_indices) == 0:
                    continue
                
                cluster_embeddings = self.embeddings[mask]
                cluster_rewards = np.array(self.rewards)[mask]
                
                # Centroid and radius
                centroid = np.mean(cluster_embeddings, axis=0)
                dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                radius = np.max(dists) if len(dists) > 0 else 0.0
                
                # Keywords
                keywords = []
                if tfidf_matrix is not None and len(feature_names) > 0:
                    try:
                        cluster_tfidf = np.mean(tfidf_matrix[cluster_indices].toarray(), axis=0)
                        top_indices = cluster_tfidf.argsort()[-3:][::-1]
                        keywords = feature_names[top_indices].tolist()
                    except Exception:
                        keywords = [f"cluster_{cluster_id}"]
                
                label = ", ".join(keywords).upper() if keywords else f"REGION_{cluster_id}"
                
                # Gradient direction in this region
                local_gradients = []
                for idx in cluster_indices:
                    if idx < len(self.embeddings) - 1:
                        grad = self.embeddings[idx + 1] - self.embeddings[idx]
                        if np.linalg.norm(grad) > 1e-8:
                            local_gradients.append(grad / np.linalg.norm(grad))
                
                if local_gradients:
                    avg_gradient = np.mean(local_gradients, axis=0)
                    flow_coherence = np.linalg.norm(avg_gradient)
                else:
                    avg_gradient = np.zeros(self.embeddings.shape[1])
                    flow_coherence = 0.0
                
                region = InterpretableRegion(
                    cluster_id=cluster_id,
                    centroid=centroid,
                    radius=radius,
                    n_points=len(cluster_indices),
                    keywords=keywords,
                    label=label,
                    mean_reward=float(np.mean(cluster_rewards)),
                    reward_variance=float(np.var(cluster_rewards)),
                    is_black_hole=float(np.mean(cluster_rewards)) < self.black_hole_threshold,
                    is_cliff_region=False,  # Updated in _identify_safety_regions
                    gradient_direction=avg_gradient,
                    flow_coherence=flow_coherence,
                )
                self.regions.append(region)
                
        except ImportError:
            # Fallback if sklearn not available
            self.cluster_labels = np.zeros(n, dtype=int)
            self.regions = []
    
    def _identify_safety_regions(self):
        """Identify black holes and cliffs in the embedding space."""
        n = len(self.embeddings)
        
        # Black holes: consistently low reward
        self.black_hole_indices = []
        for i, r in enumerate(self.rewards):
            if r < self.black_hole_threshold:
                self.black_hole_indices.append(i)
        
        # Cliffs: high reward gradient magnitude
        self.cliff_indices = []
        for i in range(n - 1):
            grad_mag = abs(self.rewards[i + 1] - self.rewards[i])
            if grad_mag > self.cliff_threshold:
                self.cliff_indices.append(i)
        
        # Update regions
        for region in self.regions:
            region_indices = np.where(self.cluster_labels == region.cluster_id)[0]
            cliff_count = sum(1 for idx in region_indices if idx in self.cliff_indices)
            region.is_cliff_region = cliff_count > len(region_indices) * 0.3
    
    def extract_features(self) -> TopologicalFeatures:
        """Extract comprehensive topological features."""
        n = len(self.embeddings)
        embed_dim = self.embeddings.shape[1] if n > 0 else 0
        
        # Hodge magnitudes
        gradient_mag = np.linalg.norm(self.hodge_gradient) if self.hodge_gradient is not None else 0.0
        curl_mag = np.linalg.norm(self.hodge_curl) if self.hodge_curl is not None else 0.0
        harmonic_mag = np.linalg.norm(self.hodge_harmonic) if self.hodge_harmonic is not None else 0.0
        
        # H¹ = curl magnitude (measures inconsistency)
        h1 = curl_mag
        
        # Connectivity
        if self.adjacency is not None and n > 0:
            degrees = np.array(self.adjacency.sum(axis=1)).flatten()
            avg_degree = np.mean(degrees)
            max_edges = n * (n - 1) / 2
            actual_edges = self.adjacency.nnz / 2  # Undirected
            density = actual_edges / max_edges if max_edges > 0 else 0.0
            
            # Connected components (approximate via Laplacian spectrum)
            try:
                L = np.diag(degrees) - self.adjacency.toarray()
                eigenvalues = np.linalg.eigvalsh(L)
                n_components = np.sum(eigenvalues < 1e-6)
            except Exception:
                n_components = 1
        else:
            avg_degree = 0.0
            density = 0.0
            n_components = 1
        
        # Curvature statistics
        if self.local_curvatures is not None and len(self.local_curvatures) > 0:
            mean_curv = float(np.mean(self.local_curvatures))
            max_curv = float(np.max(self.local_curvatures))
            var_curv = float(np.var(self.local_curvatures))
        else:
            mean_curv = max_curv = var_curv = 0.0
        
        # Safety
        n_black_holes = len(self.black_hole_indices)
        n_cliffs = len(self.cliff_indices)
        safe_fraction = 1.0 - (n_black_holes + n_cliffs) / n if n > 0 else 1.0
        
        # Cluster purity (how well-separated clusters are)
        if self.cluster_labels is not None and len(self.regions) > 1:
            # Use silhouette score approximation
            try:
                from sklearn.metrics import silhouette_score
                purity = silhouette_score(self.embeddings, self.cluster_labels)
                purity = (purity + 1) / 2  # Normalize to 0-1
            except Exception:
                purity = 0.5
        else:
            purity = 1.0
        
        return TopologicalFeatures(
            n_points=n,
            embedding_dim=embed_dim,
            gradient_magnitude=gradient_mag,
            curl_magnitude=curl_mag,
            harmonic_magnitude=harmonic_mag,
            h1_cohomology=h1,
            n_connected_components=n_components,
            graph_density=density,
            avg_degree=avg_degree,
            mean_curvature=mean_curv,
            max_curvature=max_curv,
            curvature_variance=var_curv,
            n_black_holes=n_black_holes,
            n_cliffs=n_cliffs,
            safe_region_fraction=safe_fraction,
            n_clusters=len(self.regions),
            cluster_purity=purity,
        )
    
    def get_interpretable_regions(self) -> List[InterpretableRegion]:
        """Get list of semantically labeled regions."""
        return self.regions
    
    def analyze_trajectory(
        self,
        trajectory_indices: List[int],
        trajectory_id: str = "trajectory",
    ) -> TrajectoryAnalysis:
        """
        Analyze a trajectory through the embedding space.
        
        Args:
            trajectory_indices: Indices of points in the trajectory
            trajectory_id: Identifier for the trajectory
        """
        n_steps = len(trajectory_indices)
        
        if n_steps < 2:
            return TrajectoryAnalysis(
                trajectory_id=trajectory_id,
                n_steps=n_steps,
                total_length=0.0,
                euclidean_length=0.0,
                tortuosity=1.0,
                cumulative_reward=sum(self.rewards[i] for i in trajectory_indices),
                reward_trend="flat",
                mean_gradient_alignment=0.0,
                curl_exposure=0.0,
                min_black_hole_distance=float('inf'),
                n_cliff_crossings=0,
                safety_score=1.0,
                regions_visited=[],
            )
        
        # Path lengths
        total_length = 0.0
        for i in range(len(trajectory_indices) - 1):
            idx1, idx2 = trajectory_indices[i], trajectory_indices[i + 1]
            total_length += np.linalg.norm(self.embeddings[idx2] - self.embeddings[idx1])
        
        euclidean_length = np.linalg.norm(
            self.embeddings[trajectory_indices[-1]] - self.embeddings[trajectory_indices[0]]
        )
        tortuosity = total_length / (euclidean_length + 1e-8)
        
        # Reward analysis
        traj_rewards = [self.rewards[i] for i in trajectory_indices]
        cumulative_reward = sum(traj_rewards)
        
        # Determine trend
        if len(traj_rewards) >= 3:
            first_half = np.mean(traj_rewards[:len(traj_rewards)//2])
            second_half = np.mean(traj_rewards[len(traj_rewards)//2:])
            if second_half > first_half + 0.1:
                trend = "increasing"
            elif second_half < first_half - 0.1:
                trend = "decreasing"
            else:
                # Check oscillation
                diffs = np.diff(traj_rewards)
                sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                trend = "oscillating" if sign_changes > len(diffs) * 0.3 else "flat"
        else:
            trend = "flat"
        
        # Gradient alignment
        alignments = []
        if self.hodge_gradient is not None and np.linalg.norm(self.hodge_gradient) > 1e-8:
            grad_normalized = self.hodge_gradient / np.linalg.norm(self.hodge_gradient)
            for i in range(len(trajectory_indices) - 1):
                idx1, idx2 = trajectory_indices[i], trajectory_indices[i + 1]
                step_dir = self.embeddings[idx2] - self.embeddings[idx1]
                step_norm = np.linalg.norm(step_dir)
                if step_norm > 1e-8:
                    step_dir = step_dir / step_norm
                    alignments.append(np.dot(step_dir, grad_normalized))
        
        mean_alignment = np.mean(alignments) if alignments else 0.0
        
        # Curl exposure
        curl_exposure = 0.0
        if self.hodge_curl is not None:
            curl_mag = np.linalg.norm(self.hodge_curl)
            curl_exposure = curl_mag  # Simplified: global curl exposure
        
        # Safety metrics
        min_bh_dist = float('inf')
        for idx in trajectory_indices:
            for bh_idx in self.black_hole_indices:
                dist = np.linalg.norm(self.embeddings[idx] - self.embeddings[bh_idx])
                min_bh_dist = min(min_bh_dist, dist)
        
        n_cliff_crossings = sum(1 for idx in trajectory_indices if idx in self.cliff_indices)
        
        # Safety score
        safety_score = 1.0
        if min_bh_dist < 1.0:
            safety_score -= 0.3 * (1.0 - min_bh_dist)
        safety_score -= 0.1 * n_cliff_crossings / n_steps
        safety_score = max(0.0, safety_score)
        
        # Regions visited
        regions_visited = []
        if self.cluster_labels is not None:
            for idx in trajectory_indices:
                cluster_id = self.cluster_labels[idx]
                region = next((r for r in self.regions if r.cluster_id == cluster_id), None)
                if region and (not regions_visited or regions_visited[-1] != region.label):
                    regions_visited.append(region.label)
        
        return TrajectoryAnalysis(
            trajectory_id=trajectory_id,
            n_steps=n_steps,
            total_length=total_length,
            euclidean_length=euclidean_length,
            tortuosity=tortuosity,
            cumulative_reward=cumulative_reward,
            reward_trend=trend,
            mean_gradient_alignment=mean_alignment,
            curl_exposure=curl_exposure,
            min_black_hole_distance=min_bh_dist,
            n_cliff_crossings=n_cliff_crossings,
            safety_score=safety_score,
            regions_visited=regions_visited,
        )
    
    def get_hodge_decomposition_at(self, state_idx: int) -> Dict[str, np.ndarray]:
        """
        Get local Hodge decomposition at a specific state.
        
        Returns gradient, curl, and harmonic directions for interpretability.
        """
        return {
            "gradient": self.hodge_gradient if self.hodge_gradient is not None else np.zeros(1),
            "curl": self.hodge_curl if self.hodge_curl is not None else np.zeros(1),
            "harmonic": self.hodge_harmonic if self.hodge_harmonic is not None else np.zeros(1),
            "local_curvature": self.local_curvatures[state_idx] if self.local_curvatures is not None else 0.0,
        }
    
    def explain_state(self, state_idx: int) -> str:
        """
        Generate human-readable explanation of a state's position in the manifold.
        """
        if state_idx >= len(self.embeddings):
            return "Invalid state index"
        
        lines = [f"STATE {state_idx} ANALYSIS", "-" * 40]
        
        # Text
        if state_idx < len(self.texts):
            lines.append(f"Text: {self.texts[state_idx][:100]}...")
        
        # Reward
        if state_idx < len(self.rewards):
            lines.append(f"Reward: {self.rewards[state_idx]:.3f}")
        
        # Cluster/Region
        if self.cluster_labels is not None:
            cluster_id = self.cluster_labels[state_idx]
            region = next((r for r in self.regions if r.cluster_id == cluster_id), None)
            if region:
                lines.append(f"Region: {region.label}")
                lines.append(f"  Mean reward in region: {region.mean_reward:.3f}")
                lines.append(f"  Flow coherence: {region.flow_coherence:.3f}")
        
        # Curvature
        if self.local_curvatures is not None:
            curv = self.local_curvatures[state_idx]
            curv_desc = "flat" if curv < 0.1 else "curved" if curv < 0.3 else "highly curved"
            lines.append(f"Local geometry: {curv_desc} (κ={curv:.3f})")
        
        # Safety
        safety_warnings = []
        if state_idx in self.black_hole_indices:
            safety_warnings.append("⚠️ BLACK HOLE (low reward region)")
        if state_idx in self.cliff_indices:
            safety_warnings.append("⚠️ CLIFF (steep gradient)")
        
        if safety_warnings:
            lines.extend(safety_warnings)
        else:
            lines.append("✓ Safe region")
        
        # Gradient alignment advice
        if self.hodge_gradient is not None and state_idx < len(self.embeddings) - 1:
            next_dir = self.embeddings[state_idx + 1] - self.embeddings[state_idx]
            next_norm = np.linalg.norm(next_dir)
            grad_norm = np.linalg.norm(self.hodge_gradient)
            
            if next_norm > 1e-8 and grad_norm > 1e-8:
                alignment = np.dot(next_dir / next_norm, self.hodge_gradient / grad_norm)
                if alignment > 0.5:
                    lines.append("→ Trajectory aligned with reward gradient (good)")
                elif alignment < -0.5:
                    lines.append("← Trajectory against reward gradient (suboptimal)")
                else:
                    lines.append("↔ Trajectory orthogonal to gradient (neutral)")
        
        return "\n".join(lines)


def demo_topology_analysis():
    """Demonstrate the embedding topology analyzer."""
    print("=" * 60)
    print("EMBEDDING TOPOLOGY ANALYZER DEMO")
    print("=" * 60)
    
    # Mock embedding model
    class MockEmbedder:
        def encode(self, texts):
            np.random.seed(42)
            embeddings = []
            for text in texts:
                np.random.seed(hash(text) % (2**32))
                emb = np.random.randn(64)
                embeddings.append(emb / np.linalg.norm(emb))
            return np.array(embeddings)
    
    # Create sample trajectory data
    texts = [
        "Agent enters the room and sees a door",
        "Agent moves towards the door",
        "Agent opens the door",
        "Agent enters a dark corridor",
        "Agent hears a noise behind",
        "Agent runs forward quickly",
        "Agent finds a treasure chest",
        "Agent opens the chest",
        "Agent collects the gold coins",
        "Agent exits through the far door",
        "Agent falls into a trap",  # Black hole
        "Agent struggles to escape",  # Black hole
        "Agent finds a secret passage",
        "Agent returns to safety",
        "Agent completes the quest",
    ]
    
    rewards = [0.1, 0.2, 0.3, 0.1, -0.1, 0.0, 0.5, 0.6, 0.8, 0.4, -0.5, -0.6, 0.3, 0.5, 1.0]
    actions = [f"action_{i}" for i in range(len(texts))]
    
    # Initialize analyzer
    embedder = MockEmbedder()
    analyzer = EmbeddingTopologyAnalyzer(
        embedding_model=embedder,
        n_clusters=4,
        black_hole_threshold=-0.3,
    )
    
    # Fit to data
    embeddings = embedder.encode(texts)
    analyzer.fit(
        states=list(embeddings),
        actions=actions,
        rewards=rewards,
        texts=texts,
    )
    
    # Extract and print features
    features = analyzer.extract_features()
    print(features.summary())
    
    # Print regions
    print("\nINTERPRETABLE REGIONS:")
    print("-" * 40)
    for region in analyzer.get_interpretable_regions():
        print(f"  {region}")
    
    # Analyze full trajectory
    print("\nTRAJECTORY ANALYSIS:")
    print("-" * 40)
    traj_analysis = analyzer.analyze_trajectory(list(range(len(texts))), "demo_trajectory")
    print(traj_analysis.summary())
    
    # Explain specific states
    print("\nSTATE EXPLANATIONS:")
    print("-" * 40)
    for idx in [0, 6, 10, 14]:  # Start, treasure, trap, end
        print(analyzer.explain_state(idx))
        print()
    
    print("Demo complete!")


if __name__ == "__main__":
    demo_topology_analysis()
