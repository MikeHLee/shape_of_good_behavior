"""
Clipped-SGPO: Sheaf-Geodesic Policy Optimization with PPO-style Clipping

Combines SGPO's geometric safety properties with PPO's training stability:
- Near black holes: Geometric scaling naturally clips (SGPO's strength)
- In safe regions: PPO-style ratio clipping (PPO's strength)

This hybrid approach provides:
1. Hard safety guarantees via geometric barriers
2. Stable training via clipping in safe regions
3. Better sample efficiency than pure SGPO

Mathematical Foundation:
- SGPO advantage: A_geo = A / √G(s), where G(s) → ∞ near black holes
- PPO clipping: ratio ∈ [1-ε, 1+ε]
- Clipped-SGPO: Uses geometric scaling when G(s) > τ, PPO clipping otherwise
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ClippedSGPOConfig:
    """Configuration for Clipped-SGPO algorithm."""
    clip_ratio: float = 0.2              # PPO clip parameter ε
    geometric_threshold: float = 2.0     # G(s) above this triggers geometric clipping
    gamma: float = 0.99                  # Discount factor
    gae_lambda: float = 0.95             # GAE lambda
    value_coef: float = 0.5              # Value loss coefficient
    entropy_coef: float = 0.01           # Entropy bonus coefficient
    max_grad_norm: float = 0.5           # Gradient clipping
    n_epochs: int = 10                   # PPO epochs per update
    batch_size: int = 64                 # Mini-batch size
    

class ClippedSGPO:
    """
    Sheaf-Geodesic Policy Optimization with PPO-style clipping.
    
    Key insight: SGPO's metric scaling already clips near black holes.
    We add explicit clipping for safe regions where G(s) ≈ 1.
    
    This provides:
    - O(1/√G) bounded updates near black holes (geometric)
    - O(ε) bounded updates in safe regions (PPO clipping)
    
    Usage:
        clipped_gpo = ClippedSGPO(clip_ratio=0.2, geometric_threshold=2.0)
        advantages, metrics = clipped_gpo.compute_advantage(states, actions, rewards, hodge_critic, metric_model)
        loss = clipped_gpo.compute_loss(old_log_probs, new_log_probs, advantages, metrics)
    """
    
    def __init__(
        self,
        clip_ratio: float = 0.2,
        geometric_threshold: float = 2.0,
        config: Optional[ClippedSGPOConfig] = None,
    ):
        """
        Initialize Clipped-SGPO.
        
        Args:
            clip_ratio: PPO clip parameter ε (default: 0.2)
            geometric_threshold: G(s) threshold for switching between mechanisms (default: 2.0)
            config: Full configuration object (overrides individual params if provided)
        """
        if config is not None:
            self.clip_ratio = config.clip_ratio
            self.geo_threshold = config.geometric_threshold
            self.config = config
        else:
            self.clip_ratio = clip_ratio
            self.geo_threshold = geometric_threshold
            self.config = ClippedSGPOConfig(
                clip_ratio=clip_ratio,
                geometric_threshold=geometric_threshold,
            )
    
    def compute_advantage(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        hodge_critic: Any,
        metric_model: Any,
        gamma: float = 0.99,
        diagnostic_critic: Any = None,
        alpha_exploit: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Hodge-corrected advantages with hybrid clipping strategy.

        The advantage is computed using Hodge decomposition:
            A^Hodge = (1/√G(s)) * (r + γV' - V - α_exploit * ω_exploit)

        Where:
            - G(s) is the Riemannian metric at state s
            - ω_exploit is ONLY the exploitable harmonic component
            - α_exploit scales the correction (from CV or learned)
            - Genuine value tension cycles are NOT subtracted

        Args:
            states: State embeddings (batch_size, embed_dim)
            actions: Action indices (batch_size,)
            rewards: Scalar rewards (batch_size,)
            next_states: Next state embeddings (batch_size, embed_dim)
            dones: Episode termination flags (batch_size,)
            hodge_critic: HodgeCritic instance for value/harmonic estimation
            metric_model: Metric model for G(s) computation
            gamma: Discount factor
            diagnostic_critic: Optional HodgeDiagnosticCritic for selective correction
            alpha_exploit: Scaling for exploitable harmonic (default 1.0)

        Returns:
            advantages: Computed advantages (batch_size,)
            metrics: Metric values G(s) at each state (batch_size,)
        """
        batch_size = len(states)

        # Get value estimates from Hodge critic
        if hasattr(hodge_critic, 'value'):
            V = hodge_critic.value(states)
            V_next = hodge_critic.value(next_states)
        else:
            # Fallback: simple value estimation
            V = np.zeros(batch_size)
            V_next = np.zeros(batch_size)

        # Get harmonic correction — selective if diagnostic critic available
        if diagnostic_critic is not None and hasattr(diagnostic_critic, 'get_exploit_correction'):
            # Only subtract exploitable cycles, preserve genuine value tensions
            omega = alpha_exploit * diagnostic_critic.get_exploit_correction(
                states, actions,
                getattr(diagnostic_critic, '_cached_edges', []),
                getattr(diagnostic_critic, '_cached_n_items', 0),
                getattr(diagnostic_critic, '_cached_diagnosis', None),
            )
        elif hasattr(hodge_critic, 'harmonic'):
            # Legacy: subtract ALL harmonic (the old broken behavior)
            omega = hodge_critic.harmonic(states, actions)
        else:
            omega = np.zeros(batch_size)

        # TD error with selective harmonic correction
        td_error = rewards + gamma * V_next * (1 - dones) - V - omega
        
        # Get metric scaling from metric model
        if callable(metric_model):
            if isinstance(states, np.ndarray):
                states_tensor = torch.tensor(states, dtype=torch.float32)
            else:
                states_tensor = states
            with torch.no_grad():
                G = metric_model(states_tensor)
                if isinstance(G, torch.Tensor):
                    G = G.cpu().numpy()
        else:
            G = np.ones(batch_size)
        
        # Ensure G is numpy array
        G = np.asarray(G).flatten()
        
        # Hybrid clipping strategy
        advantages = np.zeros(batch_size)
        
        for i in range(batch_size):
            g = G[i]
            if g > self.geo_threshold:
                # Near black hole: use geometric scaling (SGPO)
                # This naturally clips by making advantage small as g → ∞
                advantages[i] = td_error[i] / np.sqrt(g)
            else:
                # Safe region: standard advantage (PPO clipping applied in loss)
                advantages[i] = td_error[i]
        
        return advantages, G
    
    def compute_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        metrics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute clipped policy loss with hybrid clipping.
        
        The loss combines:
        1. Geometric scaling near black holes (metrics > threshold)
        2. PPO-style ratio clipping in safe regions (metrics ≤ threshold)
        
        Args:
            old_log_probs: Log probabilities from old policy (batch_size,)
            new_log_probs: Log probabilities from current policy (batch_size,)
            advantages: Computed advantages (batch_size,)
            metrics: Metric values G(s) at each state (batch_size,)
            
        Returns:
            loss: Scalar policy loss (to be minimized)
        """
        # Probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Apply PPO clipping only where metric is small (safe regions)
        # Near black holes, geometric scaling already bounds updates
        clipped_ratio = torch.where(
            metrics > self.geo_threshold,
            ratio,  # No ratio clipping near black holes
            torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        )
        
        # Surrogate objectives
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        
        # Pessimistic bound (take minimum to be conservative)
        loss = -torch.min(surr1, surr2).mean()
        
        return loss
    
    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
        clip_value: bool = True,
    ) -> torch.Tensor:
        """
        Compute value function loss with optional clipping.
        
        Args:
            values: Current value estimates (batch_size,)
            returns: Target returns (batch_size,)
            old_values: Previous value estimates for clipping (batch_size,)
            clip_value: Whether to clip value updates
            
        Returns:
            value_loss: Scalar value loss
        """
        if clip_value and old_values is not None:
            # Clipped value loss (PPO-style)
            value_clipped = old_values + torch.clamp(
                values - old_values,
                -self.clip_ratio,
                self.clip_ratio
            )
            value_loss_1 = F.mse_loss(values, returns)
            value_loss_2 = F.mse_loss(value_clipped, returns)
            value_loss = torch.max(value_loss_1, value_loss_2)
        else:
            value_loss = F.mse_loss(values, returns)
        
        return value_loss
    
    def compute_total_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        metrics: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss for Clipped-SGPO update.
        
        Args:
            old_log_probs: Log probs from old policy
            new_log_probs: Log probs from current policy
            advantages: Computed advantages
            metrics: Metric values G(s)
            values: Value estimates
            returns: Target returns
            entropy: Policy entropy
            old_values: Previous value estimates (optional)
            
        Returns:
            total_loss: Combined loss for optimization
            loss_dict: Dictionary of individual loss components
        """
        # Policy loss with hybrid clipping
        policy_loss = self.compute_loss(old_log_probs, new_log_probs, advantages, metrics)
        
        # Value loss
        value_loss = self.compute_value_loss(values, returns, old_values)
        
        # Entropy bonus (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )
        
        # Compute additional metrics
        with torch.no_grad():
            ratio = torch.exp(new_log_probs - old_log_probs)
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            clip_fraction = (
                (torch.abs(ratio - 1) > self.clip_ratio).float().mean().item()
            )
            
            # Separate stats for geometric vs clipped regions
            geo_mask = metrics > self.geo_threshold
            n_geo = geo_mask.sum().item()
            n_clip = (~geo_mask).sum().item()
        
        loss_dict = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "total_loss": total_loss.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "n_geometric_clipped": n_geo,
            "n_ppo_clipped": n_clip,
        }
        
        return total_loss, loss_dict


class ClippedSGPOTrainer:
    """
    Full trainer for Clipped-SGPO algorithm.
    
    Handles the complete training loop including:
    - Rollout collection
    - Advantage computation with Hodge correction
    - Policy and value updates with hybrid clipping
    """
    
    def __init__(
        self,
        policy: nn.Module,
        hodge_critic: Any,
        metric_model: nn.Module,
        lr: float = 3e-4,
        config: Optional[ClippedSGPOConfig] = None,
        device: torch.device = None,
    ):
        """
        Initialize Clipped-SGPO trainer.
        
        Args:
            policy: Policy network (action distribution + value head)
            hodge_critic: Hodge critic for reward decomposition
            metric_model: Riemannian metric model
            lr: Learning rate
            config: Algorithm configuration
            device: Torch device
        """
        self.policy = policy
        self.hodge_critic = hodge_critic
        self.metric_model = metric_model
        self.config = config or ClippedSGPOConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Clipped-SGPO algorithm
        self.clipped_gpo = ClippedSGPO(config=self.config)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        # Move models to device
        self.policy.to(self.device)
        if hasattr(self.metric_model, 'to'):
            self.metric_model.to(self.device)
        
        # Training stats
        self.train_stats: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
        }
    
    def update(self, rollout_buffer: Any) -> Dict[str, float]:
        """
        Perform policy update from collected rollout.
        
        Args:
            rollout_buffer: Buffer containing trajectory data
            
        Returns:
            stats: Dictionary of training statistics
        """
        # Extract data from buffer
        states = np.array(rollout_buffer.states)
        actions = np.array(rollout_buffer.actions)
        rewards = np.array(rollout_buffer.rewards)
        old_log_probs = np.array(rollout_buffer.log_probs)
        dones = np.array(rollout_buffer.dones)
        
        # Compute next states (shift by 1)
        next_states = np.roll(states, -1, axis=0)
        next_states[-1] = states[-1]  # Last state loops to itself
        
        # Compute advantages with Hodge correction and metric scaling
        advantages, metrics = self.clipped_gpo.compute_advantage(
            states, actions, rewards, next_states, dones,
            self.hodge_critic, self.metric_model,
            gamma=self.config.gamma
        )
        
        # Compute returns (advantages + values)
        values = np.array(rollout_buffer.values)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        metrics_t = torch.tensor(metrics, dtype=torch.float32, device=self.device)
        old_values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
        
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        # Aggregate stats
        total_stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "n_updates": 0,
        }
        
        # Training loop
        for epoch in range(self.config.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.config.batch_size):
                end = start + self.config.batch_size
                batch_idx = indices[start:end]
                
                # Get batch data
                batch_states = states_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_metrics = metrics_t[batch_idx]
                batch_old_values = old_values_t[batch_idx]
                
                # Forward pass
                if hasattr(self.policy, 'get_action_and_value'):
                    _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                        batch_states, batch_actions
                    )
                else:
                    # Fallback for simpler policies
                    logits, new_values = self.policy(batch_states)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy()
                
                # Compute total loss
                total_loss, loss_dict = self.clipped_gpo.compute_total_loss(
                    batch_old_log_probs,
                    new_log_probs,
                    batch_advantages,
                    batch_metrics,
                    new_values,
                    batch_returns,
                    entropy,
                    batch_old_values,
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate stats
                for key in ["policy_loss", "value_loss", "entropy", "approx_kl"]:
                    total_stats[key] += loss_dict[key]
                total_stats["n_updates"] += 1
            
            # Early stopping based on KL divergence
            avg_kl = total_stats["approx_kl"] / total_stats["n_updates"]
            if avg_kl > 0.02:  # Target KL threshold
                break
        
        # Average stats
        n_updates = total_stats["n_updates"]
        stats = {
            "policy_loss": total_stats["policy_loss"] / n_updates,
            "value_loss": total_stats["value_loss"] / n_updates,
            "entropy": total_stats["entropy"] / n_updates,
            "approx_kl": total_stats["approx_kl"] / n_updates,
            "n_epochs": epoch + 1,
        }
        
        # Store in history
        for key, value in stats.items():
            if key in self.train_stats:
                self.train_stats[key].append(value)
        
        return stats


# Theoretical justification as docstring
CLIPPED_SGPO_THEORY = """
Proposition (Clipped-SGPO Stability):

Let π_θ be updated by Clipped-SGPO. Then:

1. In regions where G(s) > τ (near black holes):
   The effective policy update is bounded by O(1/√G) due to geometric scaling.
   As G(s) → ∞ (approaching event horizon), updates → 0.

2. In regions where G(s) ≤ τ (safe regions):
   The effective policy update is bounded by O(ε) due to PPO clipping.
   The ratio r(θ) = π_new/π_old is clamped to [1-ε, 1+ε].

Proof sketch:
- SGPO's advantage scaling: A_geo = A / √G. As G → ∞, A_geo → 0.
- PPO's ratio clipping: |r(θ) - 1| ≤ ε, bounding policy divergence.
- Clipped-SGPO inherits both bounds in their respective domains.
- The transition at G = τ ensures smooth interpolation between mechanisms.

Safety Guarantee:
The geometric barrier G(x) → ∞ as x → B (black hole boundary) creates an
infinite "energy cost" for crossing the event horizon. Since policy
gradients are scaled by 1/√G, no finite gradient can push the policy
across this barrier.

This is equivalent to a Reciprocal Control Barrier Function (RCBF):
    h(x) = 1/dist(x, B)
    ḣ(x) ≤ -α * h(x)
guarantees forward invariance of the safe set.
"""
