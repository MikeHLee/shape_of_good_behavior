"""
Context-Conditional Hodge Critic: Discriminating Valid vs Invalid Preference Cycles

Extends the HodgeCritic to distinguish between:
1. Valid contextual cycles: A > B in context C1, B > A in context C2 (rock-paper-scissors)
2. Invalid intransitive cycles: A > B > C > A within the same context (true bias)

Key insight: We compute H¹ both marginally (ignoring context) and conditionally (within each context).
Only the conditional H¹ represents truly problematic inconsistencies.

Mathematical Framework:
- Marginal H¹: Computed on full preference graph (may include valid contextual variation)
- Conditional H¹: Computed within each context group (captures true inconsistencies)
- Invalid H¹ = Conditional H¹ (this is what we want to filter)
- Valid H¹ = Marginal H¹ - Conditional H¹ (contextual variation we preserve)

Integration with SGPO_ANIS:
- Experiment A: Pre-filter preferences using threshold on conditional H¹
- Experiment C: Compute omega_invalid in advantage and discount only that component
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import lsqr
from collections import defaultdict
import torch
import torch.nn as nn


@dataclass
class ContextualFeedbackItem:
    """Feedback item with context information for conditional H¹ analysis."""
    state_text: str
    action_text: str
    next_state_text: Optional[str]
    rank: float
    context_id: str  # Required: identifies the context group
    critique: Optional[str] = None
    evaluator_id: Optional[str] = None
    
    # For preference pairs
    chosen_text: Optional[str] = None
    rejected_text: Optional[str] = None
    
    def to_embedding_text(self) -> str:
        """Combine into single text for embedding."""
        parts = [f"State: {self.state_text}", f"Action: {self.action_text}"]
        if self.next_state_text:
            parts.append(f"Result: {self.next_state_text}")
        if self.critique:
            parts.append(f"Critique: {self.critique}")
        return " | ".join(parts)
    
    def get_context_embedding_text(self) -> str:
        """Text for context embedding (typically the prompt/state)."""
        return self.state_text


@dataclass
class ConditionalH1Result:
    """Result of context-conditional H¹ analysis."""
    marginal_h1: float  # H¹ computed ignoring context
    conditional_h1: float  # H¹ computed within contexts (invalid cycles)
    valid_contextual_h1: float  # marginal - conditional (valid variation)
    per_context_h1: Dict[str, float]  # H¹ breakdown by context
    n_contexts: int
    n_items_per_context: Dict[str, int]
    
    def get_invalid_ratio(self) -> float:
        """Ratio of H¹ that is invalid (conditional / marginal)."""
        if self.marginal_h1 < 1e-8:
            return 0.0
        return self.conditional_h1 / self.marginal_h1
    
    def __str__(self) -> str:
        return (
            f"ConditionalH1Result(\n"
            f"  marginal_h1={self.marginal_h1:.4f},\n"
            f"  conditional_h1={self.conditional_h1:.4f} (invalid),\n"
            f"  valid_contextual_h1={self.valid_contextual_h1:.4f},\n"
            f"  n_contexts={self.n_contexts},\n"
            f"  invalid_ratio={self.get_invalid_ratio():.2%}\n"
            f")"
        )


class ContextConditionalHodgeCritic:
    """
    Hodge Critic with context-conditional H¹ discrimination.
    
    This critic distinguishes between:
    - Valid contextual cycles (preserved in training)
    - Invalid intransitive cycles (filtered from training)
    
    Usage:
        critic = ContextConditionalHodgeCritic(embedding_model)
        critic.add_contextual_feedback(items)
        
        # Compute conditional H¹
        result = critic.compute_conditional_h1()
        
        # Get harmonic values for SGPO advantage computation
        omega_invalid = critic.harmonic_given_context(states, actions, contexts)
    """
    
    def __init__(
        self,
        embedding_model: Any,
        embed_dim: Optional[int] = None,
        similarity_threshold: float = 0.8,
        device: torch.device = None,
    ):
        """
        Args:
            embedding_model: Model with .encode(texts) -> np.ndarray
            embed_dim: Embedding dimension (auto-detected if None)
            similarity_threshold: Cosine similarity for state connections
            device: PyTorch device for neural network components
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.device = device or torch.device("cpu")
        
        # Auto-detect embedding dimension
        if embed_dim is None:
            test = embedding_model.encode(["test"])
            self.embed_dim = test.shape[-1]
        else:
            self.embed_dim = embed_dim
        
        # Storage
        self.feedback_items: List[ContextualFeedbackItem] = []
        self.embeddings: Optional[np.ndarray] = None
        self.context_embeddings: Optional[np.ndarray] = None
        
        # Context groupings
        self.context_groups: Dict[str, List[int]] = defaultdict(list)
        
        # Cached results
        self._marginal_h1: Optional[float] = None
        self._conditional_h1: Optional[float] = None
        self._per_context_h1: Optional[Dict[str, float]] = None
        self._harmonic_fields: Optional[Dict[str, np.ndarray]] = None
        
        # Neural network for continuous harmonic estimation
        self._harmonic_net: Optional[nn.Module] = None
        self._context_encoder: Optional[nn.Module] = None
    
    def add_contextual_feedback(self, items: List[ContextualFeedbackItem]):
        """Add feedback items with context information."""
        for item in items:
            idx = len(self.feedback_items)
            self.feedback_items.append(item)
            self.context_groups[item.context_id].append(idx)
        self._invalidate_cache()
    
    def add_preference_pair(
        self,
        context_id: str,
        prompt: str,
        chosen: str,
        rejected: str,
        evaluator_id: Optional[str] = None,
    ):
        """Add a preference pair (common format in RLHF datasets)."""
        item = ContextualFeedbackItem(
            state_text=prompt,
            action_text="preference",
            next_state_text=None,
            rank=1.0,  # Chosen preferred
            context_id=context_id,
            evaluator_id=evaluator_id,
            chosen_text=chosen,
            rejected_text=rejected,
        )
        self.add_contextual_feedback([item])
    
    def _invalidate_cache(self):
        """Clear cached computations."""
        self.embeddings = None
        self.context_embeddings = None
        self._marginal_h1 = None
        self._conditional_h1 = None
        self._per_context_h1 = None
        self._harmonic_fields = None
    
    def _compute_embeddings(self):
        """Embed all feedback items and contexts."""
        if self.embeddings is not None:
            return
        
        # Embed feedback items
        texts = [item.to_embedding_text() for item in self.feedback_items]
        self.embeddings = self.embedding_model.encode(texts)
        
        # Embed contexts
        context_texts = [item.get_context_embedding_text() for item in self.feedback_items]
        self.context_embeddings = self.embedding_model.encode(context_texts)
    
    def _compute_h1_for_subset(
        self,
        indices: List[int],
        embeddings: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute H¹ for a subset of items.
        
        Returns:
            h1_magnitude: Scalar H¹ value
            harmonic_field: Vector field in embedding space
        """
        n = len(indices)
        if n < 2:
            return 0.0, np.zeros(self.embed_dim)
        
        # Build adjacency for this subset
        subset_embeddings = embeddings[indices]
        
        # Normalize embeddings
        norms = np.linalg.norm(subset_embeddings, axis=1, keepdims=True)
        normalized = subset_embeddings / (norms + 1e-8)
        
        # Compute pairwise similarities
        similarities = normalized @ normalized.T
        
        # Build edges and weights
        edge_list = []
        edge_weights = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[i, j] > self.similarity_threshold:
                    edge_list.append((i, j))
                    # Weight = rank difference
                    rank_i = self.feedback_items[indices[i]].rank
                    rank_j = self.feedback_items[indices[j]].rank
                    edge_weights[(i, j)] = rank_j - rank_i
        
        if len(edge_list) < 1:
            return 0.0, np.zeros(self.embed_dim)
        
        # Build boundary operator d0
        rows, cols, data = [], [], []
        for idx, (u, v) in enumerate(edge_list):
            rows.extend([idx, idx])
            cols.extend([u, v])
            data.extend([-1, 1])
        
        d0 = csr_matrix((data, (rows, cols)), shape=(len(edge_list), n))
        
        # Edge flow vector
        Y = np.array([edge_weights[tuple(edge)] for edge in edge_list])
        
        # Gradient component
        L0 = d0.T @ d0
        divergence = d0.T @ Y
        
        try:
            s_potential = lsqr(L0, divergence)[0]
            Y_grad = d0 @ s_potential
        except:
            Y_grad = np.zeros_like(Y)
        
        # Harmonic = residual (simplified: no curl for small graphs)
        Y_harm = Y - Y_grad
        h1_magnitude = np.linalg.norm(Y_harm)
        
        # Map back to embedding space
        harmonic_field = np.zeros(self.embed_dim)
        for idx, (u, v) in enumerate(edge_list):
            edge_vec = subset_embeddings[v] - subset_embeddings[u]
            edge_len = np.linalg.norm(edge_vec) + 1e-8
            edge_dir = edge_vec / edge_len
            harmonic_field += Y_harm[idx] * edge_dir
        
        if len(edge_list) > 0:
            harmonic_field /= len(edge_list)
        
        return h1_magnitude, harmonic_field
    
    def compute_conditional_h1(self) -> ConditionalH1Result:
        """
        Compute context-conditional H¹ analysis.
        
        Returns:
            ConditionalH1Result with marginal, conditional, and per-context H¹
        """
        self._compute_embeddings()
        
        n = len(self.feedback_items)
        if n < 2:
            return ConditionalH1Result(
                marginal_h1=0.0,
                conditional_h1=0.0,
                valid_contextual_h1=0.0,
                per_context_h1={},
                n_contexts=0,
                n_items_per_context={},
            )
        
        # 1. Compute marginal H¹ (all items, ignoring context)
        all_indices = list(range(n))
        marginal_h1, marginal_harmonic = self._compute_h1_for_subset(
            all_indices, self.embeddings
        )
        
        # 2. Compute conditional H¹ (within each context)
        per_context_h1 = {}
        context_harmonics = {}
        n_items_per_context = {}
        
        for context_id, indices in self.context_groups.items():
            if len(indices) >= 2:
                h1, harmonic = self._compute_h1_for_subset(indices, self.embeddings)
                per_context_h1[context_id] = h1
                context_harmonics[context_id] = harmonic
            else:
                per_context_h1[context_id] = 0.0
                context_harmonics[context_id] = np.zeros(self.embed_dim)
            n_items_per_context[context_id] = len(indices)
        
        # Conditional H¹ = weighted average of per-context H¹
        total_items = sum(n_items_per_context.values())
        conditional_h1 = sum(
            h1 * n_items_per_context[ctx] / total_items
            for ctx, h1 in per_context_h1.items()
        )
        
        # Valid contextual H¹ = variation explained by context
        valid_contextual_h1 = max(0, marginal_h1 - conditional_h1)
        
        # Cache results
        self._marginal_h1 = marginal_h1
        self._conditional_h1 = conditional_h1
        self._per_context_h1 = per_context_h1
        self._harmonic_fields = context_harmonics
        
        return ConditionalH1Result(
            marginal_h1=marginal_h1,
            conditional_h1=conditional_h1,
            valid_contextual_h1=valid_contextual_h1,
            per_context_h1=per_context_h1,
            n_contexts=len(self.context_groups),
            n_items_per_context=n_items_per_context,
        )
    
    def filter_invalid_cycles(
        self,
        threshold: float = 0.8,
    ) -> List[ContextualFeedbackItem]:
        """
        Filter out items contributing to invalid (within-context) cycles.
        
        Args:
            threshold: H¹ threshold; contexts with H¹ > threshold get filtered
            
        Returns:
            Filtered list of feedback items (invalid cycles removed)
        """
        if self._per_context_h1 is None:
            self.compute_conditional_h1()
        
        # Keep items from contexts with low H¹
        filtered_items = []
        for item in self.feedback_items:
            context_h1 = self._per_context_h1.get(item.context_id, 0.0)
            if context_h1 <= threshold:
                filtered_items.append(item)
        
        return filtered_items
    
    def get_filtered_indices(self, threshold: float = 0.8) -> List[int]:
        """
        Get indices of items that pass the H¹ filter.
        
        Args:
            threshold: H¹ threshold for filtering
            
        Returns:
            List of indices into self.feedback_items that pass filter
        """
        if self._per_context_h1 is None:
            self.compute_conditional_h1()
        
        return [
            i for i, item in enumerate(self.feedback_items)
            if self._per_context_h1.get(item.context_id, 0.0) <= threshold
        ]
    
    # =========================================================================
    # Neural Network Interface for SGPO Integration
    # =========================================================================
    
    def build_harmonic_networks(self, hidden_dim: int = 64):
        """
        Build neural networks for continuous harmonic estimation.
        
        These networks are trained to predict the harmonic component
        at any state/context, enabling use in SGPO advantage computation.
        """
        # Context encoder: context_embedding -> context_features
        self._context_encoder = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(self.device)
        
        # Harmonic network: (state_embedding, context_features) -> omega
        self._harmonic_net = nn.Sequential(
            nn.Linear(self.embed_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        ).to(self.device)
        
        # Train on computed harmonic values
        self._train_harmonic_networks()
    
    def _train_harmonic_networks(self, n_epochs: int = 100, lr: float = 1e-3):
        """Train harmonic networks on computed H¹ values."""
        if self._per_context_h1 is None:
            self.compute_conditional_h1()
        
        if self._harmonic_net is None:
            return
        
        self._compute_embeddings()
        
        if self.embeddings is None or len(self.embeddings) == 0:
            return
        
        # Prepare training data (detached from any computation graph)
        states = torch.tensor(self.embeddings.copy(), dtype=torch.float32, device=self.device)
        contexts = torch.tensor(self.context_embeddings.copy(), dtype=torch.float32, device=self.device)
        
        # Target: per-context H¹ for each item
        targets = torch.tensor(
            [self._per_context_h1.get(item.context_id, 0.0) for item in self.feedback_items],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            list(self._context_encoder.parameters()) + list(self._harmonic_net.parameters()),
            lr=lr,
        )
        
        # Training loop
        self._context_encoder.train()
        self._harmonic_net.train()
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            context_features = self._context_encoder(contexts)
            combined = torch.cat([states, context_features], dim=-1)
            predictions = self._harmonic_net(combined)
            
            loss = nn.MSELoss()(predictions, targets)
            loss.backward()
            optimizer.step()
        
        self._context_encoder.eval()
        self._harmonic_net.eval()
    
    def harmonic_given_context(
        self,
        states: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        contexts: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get harmonic (invalid cycle) component for SGPO advantage.
        
        This returns ω_invalid to be subtracted from TD error:
            A = (r + γV' - V - ω_invalid) / √g
        
        Args:
            states: State embeddings [batch, embed_dim]
            actions: Action embeddings [batch, action_dim] (unused, for interface)
            contexts: Context embeddings [batch, embed_dim]
            
        Returns:
            omega_invalid: Invalid harmonic component [batch, 1]
        """
        if self._harmonic_net is None:
            self.build_harmonic_networks()
        
        # Convert to tensors
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
        if isinstance(contexts, np.ndarray):
            contexts = torch.tensor(contexts, dtype=torch.float32, device=self.device)
        
        # Compute harmonic
        with torch.no_grad():
            context_features = self._context_encoder(contexts)
            combined = torch.cat([states, context_features], dim=-1)
            omega = self._harmonic_net(combined)
        
        return omega
    
    def value(self, states: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Placeholder value function (to be replaced with actual critic).
        
        For full integration, combine with a trained value network.
        """
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        return np.zeros(len(states))
    
    def harmonic(
        self,
        states: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Get marginal harmonic component (for comparison with conditional).
        
        This is the TOTAL harmonic, including valid contextual variation.
        Use harmonic_given_context() for just the invalid component.
        """
        # Return cached marginal if available
        if self._marginal_h1 is not None:
            return np.full(len(states) if hasattr(states, '__len__') else 1, self._marginal_h1)
        return np.zeros(len(states) if hasattr(states, '__len__') else 1)


# =============================================================================
# Utility Functions for Experiment Integration
# =============================================================================

def load_hh_rlhf_with_context(
    num_samples: int = 1000,
    context_strategy: str = "prompt_hash",
) -> List[ContextualFeedbackItem]:
    """
    Load Anthropic HH-RLHF dataset with context identifiers.
    
    Args:
        num_samples: Number of samples to load
        context_strategy: How to assign context IDs
            - "prompt_hash": Hash of prompt text (exact match)
            - "prompt_prefix": First 100 chars of prompt
            - "topic_cluster": Cluster by topic embedding (requires extra processing)
            
    Returns:
        List of ContextualFeedbackItem
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    dataset = load_dataset("anthropic/hh-rlhf", split="train")
    
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    items = []
    for example in dataset:
        try:
            chosen = example["chosen"]
            rejected = example["rejected"]
            
            # Extract prompt (everything before last Assistant response)
            prompt = chosen.rpartition("\n\nAssistant:")[0]
            chosen_resp = chosen.rpartition("\n\nAssistant:")[2].strip()
            rejected_resp = rejected.rpartition("\n\nAssistant:")[2].strip()
            
            if not prompt or not chosen_resp or not rejected_resp:
                continue
            
            # Assign context ID based on strategy
            if context_strategy == "prompt_hash":
                context_id = str(hash(prompt) % 10000)
            elif context_strategy == "prompt_prefix":
                context_id = prompt[:100]
            else:
                context_id = str(hash(prompt) % 10000)
            
            item = ContextualFeedbackItem(
                state_text=prompt,
                action_text="response",
                next_state_text=None,
                rank=1.0,
                context_id=context_id,
                chosen_text=chosen_resp,
                rejected_text=rejected_resp,
            )
            items.append(item)
            
        except Exception:
            continue
    
    return items


def create_normalized_train_sets(
    items: List[ContextualFeedbackItem],
    critic: ContextConditionalHodgeCritic,
    threshold: float = 0.8,
) -> Tuple[List[ContextualFeedbackItem], List[ContextualFeedbackItem]]:
    """
    Create normalized training sets for Experiment A.
    
    Returns two sets of EQUAL SIZE:
    1. raw_set: Random sample from all items
    2. filtered_set: Items passing H¹ threshold (then sampled to match size)
    
    Args:
        items: All feedback items
        critic: Trained context-conditional critic
        threshold: H¹ threshold for filtering
        
    Returns:
        (raw_set, filtered_set) of equal length
    """
    # Get filtered indices
    filtered_indices = critic.get_filtered_indices(threshold)
    filtered_items = [items[i] for i in filtered_indices]
    
    # Normalize sizes
    n_filtered = len(filtered_items)
    
    if n_filtered == 0:
        print("Warning: No items pass filter. Returning all items for both sets.")
        return items, items
    
    # Sample raw set to match filtered size
    import random
    raw_indices = random.sample(range(len(items)), min(n_filtered, len(items)))
    raw_items = [items[i] for i in raw_indices]
    
    return raw_items, filtered_items
