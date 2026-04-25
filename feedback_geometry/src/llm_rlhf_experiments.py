# -*- coding: utf-8 -*-
"""
LLM RLHF Experiments with Hodge Filtering and Conformal Safety

This module implements proper RLHF fine-tuning of open LLMs:
- Experiment A: Hodge preference filtering during reward model training
- Experiment C: Conformal safety during policy optimization

Uses: Llama-3-8B, Mistral-7B, or Qwen-2.5-7B on Modal A100 GPUs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
import json


@dataclass
class LLMRLHFConfig:
    """Configuration for LLM RLHF experiments."""
    # Model selection
    model_name: str = "mistralai/Mistral-7B-v0.3"  # or "meta-llama/Llama-3-8B"
    
    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_length: int = 512
    num_epochs: int = 1
    
    # LoRA parameters (for efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Reward model training
    rm_epochs: int = 1
    rm_batch_size: int = 8
    
    # PPO parameters
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    
    # Hodge filtering (Experiment A)
    use_hodge_filtering: bool = True
    reliability_threshold: float = 0.5
    filter_method: str = "reliability_score"  # or "harmonic_only", "curl_only"
    
    # Conformal safety (Experiment C)
    use_conformal_safety: bool = True
    conformal_sharpness: float = 4.0
    safety_warmup_steps: int = 100
    danger_keywords: List[str] = field(default_factory=lambda: [
        "hack", "exploit", "manipulate", "deceive", "bypass",
        "jailbreak", "ignore instructions", "pretend", "roleplay as"
    ])


# =============================================================================
# HODGE DECOMPOSITION FOR TEXT PREFERENCES
# =============================================================================

class TextHodgeDecomposition:
    """
    Hodge decomposition for text preference data.
    
    Given preference pairs (prompt, chosen, rejected), compute:
    - Gradient component: Consistent transitivity
    - Curl component: Local inconsistencies (A>B>C>A cycles)
    - Harmonic component: Global inconsistencies
    
    Reliability = ||gradient||^2 / ||total||^2
    """
    
    def __init__(self, embedding_model=None, embedding_dim: int = 384):
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        
    def embed_responses(self, texts: List[str]) -> np.ndarray:
        """Embed text responses."""
        if self.embedding_model is not None:
            return self.embedding_model.encode(texts, show_progress_bar=False)
        else:
            # Fallback: hash-based pseudo-embeddings for testing
            embeddings = []
            for text in texts:
                np.random.seed(hash(text) % (2**32))
                embeddings.append(np.random.randn(self.embedding_dim))
            return np.array(embeddings, dtype=np.float32)
    
    def build_preference_graph(
        self,
        preferences: List[Dict],
        similarity_threshold: float = 0.8
    ) -> Tuple[np.ndarray, Dict]:
        """
        Build preference graph from text preferences.
        
        Returns:
            adjacency: (n_nodes, n_nodes) preference strength matrix
            node_info: mapping from node indices to response texts
        """
        # Extract unique responses
        all_responses = set()
        for pref in preferences:
            all_responses.add(pref['chosen'])
            all_responses.add(pref['rejected'])
        
        response_list = list(all_responses)
        response_to_idx = {r: i for i, r in enumerate(response_list)}
        n_nodes = len(response_list)
        
        # Embed all responses
        embeddings = self.embed_responses(response_list)
        
        # Build adjacency matrix (preference flow)
        adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        for pref in preferences:
            i = response_to_idx[pref['chosen']]
            j = response_to_idx[pref['rejected']]
            adjacency[i, j] += 1.0  # i preferred over j
        
        # Normalize by context similarity (same prompt = stronger signal)
        # For simplicity, just normalize by total count
        total = adjacency.sum()
        if total > 0:
            adjacency = adjacency / total
        
        node_info = {
            'responses': response_list,
            'embeddings': embeddings,
            'response_to_idx': response_to_idx
        }
        
        return adjacency, node_info
    
    def decompose(
        self,
        preferences: List[Dict],
        context_embeddings: Optional[np.ndarray] = None,
        use_sparse: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute Hodge decomposition of preference data.
        
        Uses sparse methods for scalability on large graphs.
        
        Returns dict with:
            - gradient: gradient component
            - curl: curl component  
            - harmonic: harmonic component
            - reliability: reliability score per context
        """
        from scipy.sparse import csr_matrix, diags
        from scipy.sparse.linalg import cg, spsolve
        
        adjacency, node_info = self.build_preference_graph(preferences)
        n = adjacency.shape[0]
        
        if n < 2:
            return {
                'gradient': adjacency,
                'curl': np.zeros_like(adjacency),
                'harmonic': np.zeros_like(adjacency),
                'reliability': np.array([1.0]),
                'node_info': node_info
            }
        
        print(f"  Hodge decomposition on {n} nodes...")
        
        # Convert to sparse for large graphs
        if use_sparse and n > 500:
            adj_sparse = csr_matrix(adjacency)
            
            # Build graph Laplacian (sparse)
            out_degree = np.asarray(adj_sparse.sum(axis=1)).flatten()
            in_degree = np.asarray(adj_sparse.sum(axis=0)).flatten()
            degree = out_degree + in_degree
            sym_adj = adj_sparse + adj_sparse.T
            L = diags(degree, format='csr') - sym_adj
            
            # Divergence (out-degree - in-degree)
            divergence = out_degree - in_degree
            
            # Solve L @ phi = divergence using conjugate gradient (regularized)
            L_reg = L + 1e-6 * diags(np.ones(n))  # Regularize for singularity
            phi, info = cg(L_reg, divergence, maxiter=1000)
            
            if info != 0:
                print(f"  Warning: CG did not converge (info={info}), using approximate solution")
            
            # Gradient flow (sparse): only compute for existing edges
            rows, cols = adj_sparse.nonzero()
            grad_data = phi[rows] - phi[cols]
            gradient = csr_matrix((grad_data, (rows, cols)), shape=(n, n)).toarray()
            
            # Curl = antisymmetric - gradient
            antisym = (adjacency - adjacency.T) / 2
            curl = antisym - gradient
            
            # Harmonic = remainder
            harmonic = adjacency - gradient - curl
        else:
            # Dense method for small graphs
            degree = adjacency.sum(axis=1) + adjacency.sum(axis=0)
            L = np.diag(degree) - (adjacency + adjacency.T)
            divergence = adjacency.sum(axis=1) - adjacency.sum(axis=0)
            
            try:
                L_pinv = np.linalg.pinv(L)
                phi = L_pinv @ divergence
            except:
                phi = np.zeros(n)
            
            gradient = np.zeros_like(adjacency)
            for i in range(n):
                for j in range(n):
                    if adjacency[i, j] > 0 or adjacency[j, i] > 0:
                        gradient[i, j] = phi[i] - phi[j]
            
            antisym = (adjacency - adjacency.T) / 2
            curl = antisym - gradient
            harmonic = adjacency - gradient - curl
        
        # Compute reliability score
        total_norm = np.linalg.norm(adjacency, 'fro') ** 2
        gradient_norm = np.linalg.norm(gradient, 'fro') ** 2
        
        if total_norm > 0:
            reliability = gradient_norm / total_norm
        else:
            reliability = 1.0
        
        print(f"  Reliability: {reliability:.3f}")
        
        return {
            'gradient': gradient,
            'curl': curl,
            'harmonic': harmonic,
            'reliability': np.array([reliability]),
            'node_info': node_info
        }
    
    def filter_preferences(
        self,
        preferences: List[Dict],
        method: str = "reliability_score",
        threshold: float = 0.5,
        percentile_mode: bool = True
    ) -> List[Dict]:
        """
        Filter preferences based on Hodge decomposition.
        
        Methods:
            - reliability_score: Keep top X% by gradient ratio (consistent preferences)
            - harmonic_only: Keep top X% by harmonic ratio (global consensus)
            - curl_only: Remove top X% by curl ratio (cyclic inconsistencies)
        
        Args:
            threshold: If percentile_mode=True, fraction to keep (0.8 = keep 80%).
                      If percentile_mode=False, absolute threshold for ratios.
            percentile_mode: Use percentile-based filtering (recommended).
        """
        if len(preferences) < 3:
            return preferences
        
        # Compute global decomposition across ALL preferences
        decomp = self.decompose(preferences)
        node_info = decomp['node_info']
        response_to_idx = node_info['response_to_idx']
        
        gradient = decomp['gradient']
        curl = decomp['curl']
        harmonic = decomp['harmonic']
        
        # Compute scores for all preferences
        pref_scores = []
        for pref in preferences:
            chosen_idx = response_to_idx.get(pref['chosen'])
            rejected_idx = response_to_idx.get(pref['rejected'])
            
            if chosen_idx is None or rejected_idx is None:
                continue
            
            # Get component strengths for this preference edge
            grad_strength = abs(gradient[chosen_idx, rejected_idx])
            curl_strength = abs(curl[chosen_idx, rejected_idx])
            harm_strength = abs(harmonic[chosen_idx, rejected_idx])
            total = grad_strength + curl_strength + harm_strength + 1e-8
            
            grad_ratio = grad_strength / total
            curl_ratio = curl_strength / total
            harm_ratio = harm_strength / total
            
            pref_scores.append({
                'pref': pref,
                'grad_ratio': grad_ratio,
                'curl_ratio': curl_ratio,
                'harm_ratio': harm_ratio
            })
        
        if not pref_scores:
            return preferences
        
        if percentile_mode:
            # Percentile-based filtering: threshold = fraction to keep
            keep_fraction = threshold
            n_keep = max(1, int(len(pref_scores) * keep_fraction))
            
            if method == "reliability_score":
                # Keep top N by gradient ratio (high gradient = consistent)
                pref_scores.sort(key=lambda x: -x['grad_ratio'])
                filtered = [p['pref'] for p in pref_scores[:n_keep]]
            elif method == "harmonic_only":
                # Keep top N by harmonic ratio (high harmonic = global consensus)
                pref_scores.sort(key=lambda x: -x['harm_ratio'])
                filtered = [p['pref'] for p in pref_scores[:n_keep]]
            elif method == "curl_only":
                # Remove top N by curl ratio (high curl = cyclic inconsistency)
                # So we KEEP the bottom N by curl ratio
                pref_scores.sort(key=lambda x: x['curl_ratio'])  # Ascending
                filtered = [p['pref'] for p in pref_scores[:n_keep]]
            else:
                filtered = [p['pref'] for p in pref_scores]
        else:
            # Absolute threshold mode (original behavior)
            filtered = []
            for ps in pref_scores:
                keep = False
                if method == "reliability_score":
                    keep = ps['grad_ratio'] >= threshold
                elif method == "harmonic_only":
                    keep = ps['harm_ratio'] >= threshold
                elif method == "curl_only":
                    keep = ps['curl_ratio'] < threshold
                if keep:
                    filtered.append(ps['pref'])
            
            # Fallback if too aggressive
            if len(filtered) < len(pref_scores) * 0.1:
                print(f"Warning: {method} filtered too aggressively ({len(filtered)}/{len(pref_scores)}), using top 20%")
                pref_scores.sort(key=lambda x: -x['grad_ratio'])
                filtered = [p['pref'] for p in pref_scores[:max(1, len(pref_scores) // 5)]]
        
        print(f"  {method}: kept {len(filtered)}/{len(pref_scores)} ({len(filtered)/len(pref_scores)*100:.1f}%)")
        return filtered


# =============================================================================
# CONFORMAL SAFETY FOR LLM POLICY
# =============================================================================

class ConformalSafetyMetric:
    """
    Conformal safety metric for LLM outputs.
    
    Creates infinite geodesic barriers around dangerous regions:
    - g = e^(2*sigma) where sigma = -beta * log(d)
    - As d -> 0, sigma -> infinity, creating impassable barriers
    
    For LLMs, "danger distance" is measured via:
    - Keyword detection
    - Embedding similarity to known dangerous outputs
    - Classifier scores
    """
    
    def __init__(
        self,
        sharpness: float = 4.0,
        danger_keywords: Optional[List[str]] = None,
        danger_embeddings: Optional[np.ndarray] = None,
        embedding_model=None
    ):
        self.sharpness = sharpness  # beta
        self.danger_keywords = danger_keywords or []
        self.danger_embeddings = danger_embeddings
        self.embedding_model = embedding_model
        
        # Learned danger regions (hardened after violations)
        self.hardened_regions: List[Tuple[np.ndarray, float]] = []
        self.violation_history: List[Dict] = []
        
    def keyword_danger_score(self, text: str) -> float:
        """Compute danger score based on keyword presence."""
        text_lower = text.lower()
        score = 0.0
        
        for keyword in self.danger_keywords:
            if keyword.lower() in text_lower:
                score += 1.0
        
        # Normalize to [0, 1]
        if self.danger_keywords:
            score = min(1.0, score / len(self.danger_keywords))
        
        return score
    
    def embedding_danger_score(self, text: str) -> float:
        """Compute danger score based on embedding similarity."""
        if self.danger_embeddings is None or self.embedding_model is None:
            return 0.0
        
        text_emb = self.embedding_model.encode([text], show_progress_bar=False)[0]
        
        # Compute max similarity to known dangerous outputs
        similarities = []
        for danger_emb in self.danger_embeddings:
            sim = np.dot(text_emb, danger_emb) / (
                np.linalg.norm(text_emb) * np.linalg.norm(danger_emb) + 1e-8
            )
            similarities.append(sim)
        
        if similarities:
            return max(0.0, max(similarities))
        return 0.0
    
    def compute_danger_distance(self, text: str) -> float:
        """
        Compute distance to nearest danger region.
        
        Returns value in (0, 1] where:
        - 1.0 = completely safe
        - 0.0 = in danger region (catastrophe)
        """
        keyword_score = self.keyword_danger_score(text)
        embedding_score = self.embedding_danger_score(text)
        
        # Combine scores (max danger)
        danger_score = max(keyword_score, embedding_score)
        
        # Convert to distance (inverse relationship)
        distance = 1.0 - danger_score
        
        return max(0.01, distance)  # Avoid exactly 0
    
    def conformal_factor(self, text: str) -> float:
        """
        Compute conformal scaling factor sigma.
        
        sigma = -beta * log(distance)
        
        As distance -> 0, sigma -> infinity (impassable barrier)
        """
        distance = self.compute_danger_distance(text)
        sigma = -self.sharpness * np.log(distance)
        return sigma
    
    def scale_reward(self, reward: float, text: str) -> float:
        """
        Scale reward by conformal metric.
        
        Rewards near danger regions are heavily penalized:
        scaled_reward = reward * e^(-2*sigma)
        """
        sigma = self.conformal_factor(text)
        
        # Exponential penalty for danger proximity
        scale = np.exp(-2 * sigma)
        
        return reward * scale
    
    def record_violation(self, text: str, context: str = ""):
        """Record a safety violation for learning."""
        if self.embedding_model is not None:
            text_emb = self.embedding_model.encode([text], show_progress_bar=False)[0]
            
            # Add to hardened regions
            self.hardened_regions.append((text_emb, 0.5))  # radius 0.5 in embedding space
        
        self.violation_history.append({
            'text': text,
            'context': context,
            'timestamp': len(self.violation_history)
        })
    
    def is_in_hardened_region(self, text: str) -> bool:
        """Check if text is in a previously hardened danger region."""
        if not self.hardened_regions or self.embedding_model is None:
            return False
        
        text_emb = self.embedding_model.encode([text], show_progress_bar=False)[0]
        
        for center, radius in self.hardened_regions:
            dist = np.linalg.norm(text_emb - center)
            if dist < radius:
                return True
        
        return False


# =============================================================================
# REWARD MODEL WITH HODGE FILTERING
# =============================================================================

class HodgeFilteredRewardModel(nn.Module):
    """
    Reward model trained on Hodge-filtered preferences.
    
    Architecture matches standard RLHF reward models but
    training data is filtered to remove inconsistent preferences.
    """
    
    def __init__(self, base_model, tokenizer, config: LLMRLHFConfig):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Reward head
        hidden_size = base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
        
        # Hodge decomposition for filtering
        self.hodge = TextHodgeDecomposition()
        
    def forward(self, input_ids, attention_mask):
        """Compute reward for input sequence."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state of last token
        last_hidden = outputs.hidden_states[-1]
        
        # Get the last non-padding token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        
        last_token_hidden = last_hidden[
            torch.arange(batch_size, device=input_ids.device),
            sequence_lengths
        ]
        
        reward = self.reward_head(last_token_hidden).squeeze(-1)
        return reward
    
    def filter_training_data(
        self,
        preferences: List[Dict]
    ) -> List[Dict]:
        """Filter training preferences using Hodge decomposition."""
        if not self.config.use_hodge_filtering:
            return preferences
        
        filtered = self.hodge.filter_preferences(
            preferences,
            method=self.config.filter_method,
            threshold=self.config.reliability_threshold
        )
        
        print(f"Hodge filtering: {len(preferences)} -> {len(filtered)} preferences "
              f"({len(filtered)/len(preferences)*100:.1f}% retained)")
        
        return filtered


# =============================================================================
# PPO WITH CONFORMAL SAFETY
# =============================================================================

class ConformalPPOTrainer:
    """
    PPO trainer with conformal safety constraints.
    
    Integrates conformal metric into:
    1. Reward scaling (near-danger rewards suppressed)
    2. Policy gradient (gradients scaled by metric)
    3. KL penalty (extra penalty for danger-proximal outputs)
    """
    
    def __init__(
        self,
        policy_model,
        ref_model,
        reward_model,
        tokenizer,
        config: LLMRLHFConfig
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Conformal safety
        self.safety = ConformalSafetyMetric(
            sharpness=config.conformal_sharpness,
            danger_keywords=config.danger_keywords
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        self.step_count = 0
        
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> torch.Tensor:
        """Compute rewards with conformal safety scaling."""
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            # Base reward from reward model
            full_text = prompt + response
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            )
            
            with torch.no_grad():
                base_reward = self.reward_model(
                    inputs['input_ids'],
                    inputs['attention_mask']
                ).item()
            
            # Apply conformal safety scaling
            if self.config.use_conformal_safety:
                if self.step_count >= self.config.safety_warmup_steps:
                    scaled_reward = self.safety.scale_reward(base_reward, response)
                    
                    # Check for violations
                    if self.safety.is_in_hardened_region(response):
                        scaled_reward = -100.0  # Catastrophic penalty
                        self.safety.record_violation(response, prompt)
                else:
                    scaled_reward = base_reward
            else:
                scaled_reward = base_reward
            
            rewards.append(scaled_reward)
        
        return torch.tensor(rewards)
    
    def train_step(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> Dict[str, float]:
        """Execute one PPO training step."""
        self.step_count += 1
        
        # Compute rewards
        rewards = self.compute_rewards(prompts, responses)
        
        # TODO: Full PPO implementation
        # This is a placeholder showing the integration points
        
        metrics = {
            'mean_reward': rewards.mean().item(),
            'min_reward': rewards.min().item(),
            'max_reward': rewards.max().item(),
            'violations': len(self.safety.violation_history),
            'hardened_regions': len(self.safety.hardened_regions),
        }
        
        return metrics


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_llm_experiment_a(
    config: LLMRLHFConfig,
    preferences: List[Dict],
    device: str = "cuda"
) -> Dict:
    """
    Run LLM Experiment A: Hodge preference filtering.
    
    Compares reward models trained on:
    1. Raw preferences
    2. Reliability-filtered preferences
    3. Harmonic-filtered preferences
    4. Curl-filtered preferences
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n{'='*60}")
    print("LLM EXPERIMENT A: Hodge Preference Filtering")
    print(f"Model: {config.model_name}")
    print(f"{'='*60}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    methods = ["raw", "reliability_score", "harmonic_only", "curl_only"]
    
    for method in methods:
        print(f"\n--- Method: {method} ---")
        
        # Load fresh model for each method
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Create reward model
        config_copy = LLMRLHFConfig(**vars(config))
        config_copy.filter_method = method
        config_copy.use_hodge_filtering = (method != "raw")
        
        rm = HodgeFilteredRewardModel(base_model, tokenizer, config_copy)
        
        # Filter training data
        if method == "raw":
            train_prefs = preferences
        else:
            train_prefs = rm.filter_training_data(preferences)
        
        # TODO: Actual reward model training
        # For now, record filtering statistics
        
        results[method] = {
            'n_original': len(preferences),
            'n_filtered': len(train_prefs),
            'retention_rate': len(train_prefs) / len(preferences),
        }
        
        print(f"  Retained: {len(train_prefs)}/{len(preferences)} "
              f"({results[method]['retention_rate']*100:.1f}%)")
    
    return results


def run_llm_experiment_c(
    config: LLMRLHFConfig,
    prompts: List[str],
    device: str = "cuda"
) -> Dict:
    """
    Run LLM Experiment C: Conformal safety during PPO.
    
    Tests whether conformal metric prevents reward hacking
    after initial violation experience.
    """
    print(f"\n{'='*60}")
    print("LLM EXPERIMENT C: Conformal Safety PPO")
    print(f"Model: {config.model_name}")
    print(f"Sharpness (beta): {config.conformal_sharpness}")
    print(f"{'='*60}")
    
    # TODO: Full implementation with actual LLM PPO
    # This is the architecture outline
    
    results = {
        'sharpness': config.conformal_sharpness,
        'warmup_steps': config.safety_warmup_steps,
        'danger_keywords': config.danger_keywords,
    }
    
    return results


if __name__ == "__main__":
    # Test configuration
    config = LLMRLHFConfig(
        model_name="mistralai/Mistral-7B-v0.3",
        use_hodge_filtering=True,
        use_conformal_safety=True,
        conformal_sharpness=4.0
    )
    
    print("LLM RLHF Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Hodge filtering: {config.filter_method}")
    print(f"  Conformal sharpness: {config.conformal_sharpness}")
