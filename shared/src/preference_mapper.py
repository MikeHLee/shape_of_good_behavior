"""Map counterfactual pairs into preference space, embedding space, and danger regions.

Produces outputs for all three tracks:
- Track 1: Preference edges for Hodge decomposition
- Track 2: DangerRegionSpec objects for SGPO conformal safety
- Track 3: EmbeddingPair objects with constitutional gradients
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .counterfactual_gen import CounterfactualPair

logger = logging.getLogger(__name__)


@dataclass
class PreferenceEdge:
    """A single preference comparison for Hodge decomposition (Track 1)."""

    item_a: int  # Index of option A
    item_b: int  # Index of option B
    probability: float  # P(A preferred over B), in [0, 1]
    category: str = ""


@dataclass
class EmbeddingPair:
    """Exploit/ideal embeddings with constitutional gradient (Track 3)."""

    exploit_embed: np.ndarray  # Embedding of exploit text
    ideal_embed: np.ndarray  # Embedding of ideal text
    context_embed: np.ndarray  # Embedding of context/prompt
    constitutional_gradient: np.ndarray  # ideal_embed - exploit_embed
    category: str = ""
    principles_violated: List[str] = field(default_factory=list)


@dataclass
class DangerRegionSpec:
    """Danger region specification derived from exploit clusters (Track 2).

    Compatible with conformal_safety.py's DangerRegion and
    learned_danger_boundary.py's training interface.
    """

    center: np.ndarray
    radius: float
    severity: float  # Higher = more dangerous
    category: str = ""
    n_exploits: int = 0


@dataclass
class MappingResult:
    """Complete output from preference mapping."""

    # Track 1: preference edges as (item_a, item_b, probability) tuples
    preference_edges: List[Tuple[int, int, float]]
    n_items: int

    # Track 3: embedding pairs with gradients
    embedding_pairs: List[EmbeddingPair]

    # Track 2: danger region specifications
    danger_regions: List[DangerRegionSpec]

    # Constitutional gradient analysis
    constitutional_gradients: Dict[str, np.ndarray]  # principle → mean gradient

    # Reduced embeddings for RM state vectors
    exploit_embeddings_reduced: np.ndarray  # (n_pairs, reduced_dim)
    ideal_embeddings_reduced: np.ndarray  # (n_pairs, reduced_dim)

    # Raw embeddings
    exploit_embeddings: np.ndarray  # (n_pairs, embed_dim)
    ideal_embeddings: np.ndarray  # (n_pairs, embed_dim)


class PreferenceMapper:
    """Map counterfactual pairs to preference space, embedding space, and danger regions."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._item_texts: Dict[str, int] = {}
        self._next_id = 0
        self._embed_model = None

    def _get_embed_model(self):
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "Install sentence-transformers: pip install sentence-transformers"
                )
            self._embed_model = SentenceTransformer(self.config.embed_model)
        return self._embed_model

    def _register_item(self, text: str) -> int:
        """Assign a unique integer ID to a text (compatible with hodge_utils)."""
        key = text[:200]  # Truncate for dedup key
        if key not in self._item_texts:
            self._item_texts[key] = self._next_id
            self._next_id += 1
        return self._item_texts[key]

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Compute sentence embeddings for a list of texts."""
        model = self._get_embed_model()
        return model.encode(texts, show_progress_bar=True, batch_size=32)

    def map_pairs(
        self, pairs: List[CounterfactualPair]
    ) -> MappingResult:
        """Map counterfactual pairs to all three track representations.

        Returns a MappingResult with preference edges, embedding pairs,
        and danger regions.
        """
        logger.info(f"Mapping {len(pairs)} counterfactual pairs...")

        # Collect all texts for batch embedding
        exploit_texts = [p.exploit_text for p in pairs]
        ideal_texts = [p.ideal_text for p in pairs]
        context_texts = [p.context_text for p in pairs]

        # Compute embeddings
        logger.info("Computing embeddings...")
        all_texts = exploit_texts + ideal_texts + context_texts
        all_embeds = self._embed_texts(all_texts)
        n = len(pairs)
        exploit_embeds = all_embeds[:n]
        ideal_embeds = all_embeds[n : 2 * n]
        context_embeds = all_embeds[2 * n :]

        # Build preference edges (Track 1)
        preference_edges: List[Tuple[int, int, float]] = []
        for i, pair in enumerate(pairs):
            exploit_id = self._register_item(pair.exploit_text)
            ideal_id = self._register_item(pair.ideal_text)
            # ideal is preferred over exploit with high probability
            prob = pair.confidence if pair.confidence > 0.5 else 0.95
            preference_edges.append((ideal_id, exploit_id, prob))

        # Build embedding pairs (Track 3)
        embedding_pairs: List[EmbeddingPair] = []
        for i, pair in enumerate(pairs):
            gradient = ideal_embeds[i] - exploit_embeds[i]
            embedding_pairs.append(
                EmbeddingPair(
                    exploit_embed=exploit_embeds[i],
                    ideal_embed=ideal_embeds[i],
                    context_embed=context_embeds[i],
                    constitutional_gradient=gradient,
                    category=pair.exploit_category,
                    principles_violated=pair.principles_violated,
                )
            )

        # Add cross-pair preferences via embedding similarity
        # This creates shared items across pairs, producing cycles for Hodge analysis
        similarity_edges = self._compute_similarity_preferences(
            pairs, ideal_embeds, exploit_embeds
        )
        preference_edges.extend(similarity_edges)

        # Add cross-category preference tensions
        cross_edges = self._compute_cross_category_preferences(
            pairs, ideal_embeds, exploit_embeds
        )
        preference_edges.extend(cross_edges)

        # PCA reduction for RM state vectors
        from sklearn.decomposition import PCA

        combined = np.vstack([exploit_embeds, ideal_embeds])
        pca = PCA(n_components=self.config.reduced_dim, random_state=self.config.seed)
        combined_reduced = pca.fit_transform(combined)
        exploit_reduced = combined_reduced[:n]
        ideal_reduced = combined_reduced[n:]

        # Extract danger regions (Track 2)
        danger_regions = self._extract_danger_regions(
            pairs, exploit_reduced
        )

        # Compute constitutional gradients (Track 3)
        constitutional_gradients = self._compute_constitutional_gradients(
            pairs, exploit_embeds, ideal_embeds
        )

        logger.info(
            f"Mapped: {len(preference_edges)} preference edges, "
            f"{len(embedding_pairs)} embedding pairs, "
            f"{len(danger_regions)} danger regions"
        )

        return MappingResult(
            preference_edges=preference_edges,
            n_items=self._next_id,
            embedding_pairs=embedding_pairs,
            danger_regions=danger_regions,
            constitutional_gradients=constitutional_gradients,
            exploit_embeddings_reduced=exploit_reduced,
            ideal_embeddings_reduced=ideal_reduced,
            exploit_embeddings=exploit_embeds,
            ideal_embeddings=ideal_embeds,
        )

    def _compute_similarity_preferences(
        self,
        pairs: List[CounterfactualPair],
        ideal_embeds: np.ndarray,
        exploit_embeds: np.ndarray,
        k_neighbors: int = 5,
        max_edges: int = 5000,
    ) -> List[Tuple[int, int, float]]:
        """Create cross-pair preference edges using embedding nearest neighbors.

        For Hodge decomposition to detect cyclic inconsistencies, the preference
        graph needs shared items across comparisons (i.e., item A appears in
        multiple edges). Pure (chosen, rejected) pairs from HH-RLHF are disjoint.

        Strategy: for each ideal response, find its k nearest-neighbor ideals
        and compare them via cosine similarity. Also compare each ideal against
        nearby exploits to create potential cycles (ideal_i > exploit_j but
        ideal_j > exploit_i → cycle if these conflict).
        """
        from sklearn.neighbors import NearestNeighbors

        n = len(pairs)
        k = min(k_neighbors, n - 1)
        if k < 2:
            return []

        edges: List[Tuple[int, int, float]] = []

        # Find nearest neighbors among ideal embeddings
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(ideal_embeds)
        distances, indices = nn.kneighbors(ideal_embeds)

        for i in range(n):
            ideal_id_i = self._register_item(pairs[i].ideal_text)
            exploit_id_i = self._register_item(pairs[i].exploit_text)

            for j_pos in range(1, k + 1):  # skip self at position 0
                j = indices[i, j_pos]
                ideal_id_j = self._register_item(pairs[j].ideal_text)
                exploit_id_j = self._register_item(pairs[j].exploit_text)

                # Edge 1: ideal_i vs ideal_j (similar ideals → near 0.5, creates cycles)
                cosine_dist = distances[i, j_pos]
                sim = 1.0 - cosine_dist
                # Near-identical ideals → 0.5 (ambiguous); dissimilar → more decisive
                prob_ii = 0.5 + 0.2 * (sim - 0.5)  # Compresses around 0.5
                if ideal_id_i != ideal_id_j:
                    edges.append((ideal_id_i, ideal_id_j, np.clip(prob_ii, 0.1, 0.9)))

                # Edge 2: ideal_i vs exploit_j (cross-pair: should ideal_i beat exploit_j?)
                sim_ie = np.dot(ideal_embeds[i], exploit_embeds[j]) / (
                    np.linalg.norm(ideal_embeds[i]) * np.linalg.norm(exploit_embeds[j]) + 1e-8
                )
                # If ideal_i is semantically close to exploit_j, preference is weaker
                prob_ie = 0.7 + 0.2 * (1.0 - sim_ie)  # Higher sim → lower preference
                if ideal_id_i != exploit_id_j:
                    edges.append((ideal_id_i, exploit_id_j, np.clip(prob_ie, 0.55, 0.95)))

            if len(edges) >= max_edges:
                break

        logger.info(f"Added {len(edges)} similarity-based preference edges (k={k})")
        return edges

    def _compute_cross_category_preferences(
        self,
        pairs: List[CounterfactualPair],
        ideal_embeds: np.ndarray,
        exploit_embeds: np.ndarray,
    ) -> List[Tuple[int, int, float]]:
        """Compute cross-principle tension edges via embedding similarity.

        When ideals from different categories conflict, this creates cyclic
        preferences that Hodge decomposition will detect as H1.
        """
        edges: List[Tuple[int, int, float]] = []
        categories = {}
        for i, pair in enumerate(pairs):
            categories.setdefault(pair.exploit_category, []).append(i)

        cat_list = list(categories.keys())
        for ci in range(len(cat_list)):
            for cj in range(ci + 1, len(cat_list)):
                indices_i = categories[cat_list[ci]][:10]  # Sample up to 10
                indices_j = categories[cat_list[cj]][:10]

                for ii in indices_i:
                    for jj in indices_j:
                        # Compare ideal_i vs ideal_j via cosine similarity
                        sim = np.dot(ideal_embeds[ii], ideal_embeds[jj]) / (
                            np.linalg.norm(ideal_embeds[ii])
                            * np.linalg.norm(ideal_embeds[jj])
                            + 1e-8
                        )
                        # Low similarity → potential tension → closer to 0.5 preference
                        prob = 0.5 + 0.3 * sim  # Maps [-1,1] → [0.2, 0.8]
                        id_i = self._register_item(pairs[ii].ideal_text)
                        id_j = self._register_item(pairs[jj].ideal_text)
                        if id_i != id_j:
                            edges.append((id_i, id_j, prob))

        logger.info(f"Added {len(edges)} cross-category preference edges")
        return edges

    def _compute_constitutional_gradients(
        self,
        pairs: List[CounterfactualPair],
        exploit_embeds: np.ndarray,
        ideal_embeds: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute per-principle mean gradient vectors in embedding space."""
        principle_gradients: Dict[str, List[np.ndarray]] = {}

        for i, pair in enumerate(pairs):
            gradient = ideal_embeds[i] - exploit_embeds[i]
            for principle in pair.principles_violated:
                principle_gradients.setdefault(principle, []).append(gradient)

            # Also aggregate by category
            cat_key = f"category:{pair.exploit_category}"
            principle_gradients.setdefault(cat_key, []).append(gradient)

        return {
            k: np.mean(v, axis=0)
            for k, v in principle_gradients.items()
            if len(v) > 0
        }

    def _extract_danger_regions(
        self,
        pairs: List[CounterfactualPair],
        exploit_embeds_reduced: np.ndarray,
    ) -> List[DangerRegionSpec]:
        """Cluster exploit embeddings per category to define danger regions.

        Output is compatible with conformal_safety.py's DangerRegion and
        learned_danger_boundary.py's training interface.
        """
        from sklearn.cluster import DBSCAN

        categories: Dict[str, List[int]] = {}
        for i, pair in enumerate(pairs):
            categories.setdefault(pair.exploit_category, []).append(i)

        danger_regions: List[DangerRegionSpec] = []

        for cat, indices in categories.items():
            if len(indices) < self.config.danger_min_cluster_size:
                continue

            cat_embeds = exploit_embeds_reduced[indices]

            # Use DBSCAN to find clusters within this category
            clustering = DBSCAN(
                eps=self.config.danger_cluster_eps,
                min_samples=self.config.danger_min_cluster_size,
            ).fit(cat_embeds)

            labels = clustering.labels_
            unique_labels = set(labels) - {-1}  # Exclude noise

            if not unique_labels:
                # No DBSCAN clusters found — use category centroid as single region
                center = np.mean(cat_embeds, axis=0)
                distances = np.linalg.norm(cat_embeds - center, axis=1)
                radius = float(np.percentile(distances, 90)) + self.config.danger_margin

                danger_regions.append(
                    DangerRegionSpec(
                        center=center,
                        radius=radius,
                        severity=1.0,
                        category=cat,
                        n_exploits=len(indices),
                    )
                )
            else:
                for label in unique_labels:
                    mask = labels == label
                    cluster_embeds = cat_embeds[mask]
                    center = np.mean(cluster_embeds, axis=0)
                    distances = np.linalg.norm(cluster_embeds - center, axis=1)
                    radius = (
                        float(np.max(distances)) + self.config.danger_margin
                    )

                    danger_regions.append(
                        DangerRegionSpec(
                            center=center,
                            radius=radius,
                            severity=float(np.sum(mask)) / len(indices),
                            category=f"{cat}_cluster{label}",
                            n_exploits=int(np.sum(mask)),
                        )
                    )

        logger.info(
            f"Extracted {len(danger_regions)} danger regions from "
            f"{len(categories)} categories"
        )
        return danger_regions
