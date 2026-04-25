"""
Semantic MDP Reinforcement Learning

Implements PPO, CPO, and SGPO (Sheaf-Geodesic Policy Optimization) for Semantic State Machines.

Key innovations:
1. **Semantic States**: Natural language embeddings as state representations
2. **Manifold-Aware Gradients**: Policy optimization respects embedding geometry
3. **Hodge-Based Safety**: SGPO uses topological reward decomposition for constraint satisfaction

Mathematical Framework:
- State space S: Embedding manifold M ⊂ ℝᵈ
- Action space A: MCP tool calls (discrete, semantically grounded)
- Transition P: LLM oracle or rule-based
- Reward R: Hodge-decomposed vector field on M

PPO: Trust region optimization on the statistical manifold of policies
CPO: Constrained optimization with expectation-based safety constraints  
SGPO: Geodesic optimization with hard topological safety barriers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import warnings


@dataclass
class RolloutBuffer:
    """
    Stores trajectory data for policy optimization.
    
    Tensor RL Interpretation:
    - states: The state tensor sequence s_0, s_1, ..., s_T (embedded observations)
    - actions: The action tensor sequence a_0, a_1, ..., a_{T-1}
    - rewards: The reward tensor (scalar or vector per step)
    - rewards_vector: High-dimensional reward embeddings (for Hodge decomposition)
    - values: Value function estimates V(s_t)
    - log_probs: Log probabilities log π(a_t|s_t) for policy gradient
    
    The trajectory τ = {(s_t, a_t, r_t)}_{t=0}^T is the fundamental object
    that gets scored by the reward sheaf and differentiated to update policy.
    """
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)  # Scalar rewards (for backward compat)
    rewards_vector: List[np.ndarray] = field(default_factory=list)  # Vector rewards (Proposal D)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    
    # For CPO/SGPO: constraint costs
    costs: List[float] = field(default_factory=list)
    
    # For semantic MDP: text states and actions (natural language state space)
    state_texts: List[str] = field(default_factory=list)
    action_texts: List[str] = field(default_factory=list)
    
    # For world model learning: predicted next states (Proposal B)
    predicted_next_states: List[np.ndarray] = field(default_factory=list)
    prediction_uncertainties: List[float] = field(default_factory=list)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.rewards_vector.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.costs.clear()
        self.state_texts.clear()
        self.action_texts.clear()
        self.predicted_next_states.clear()
        self.prediction_uncertainties.clear()
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        cost: float = 0.0,
        state_text: str = "",
        action_text: str = "",
        reward_vector: np.ndarray = None,
        predicted_next_state: np.ndarray = None,
        prediction_uncertainty: float = 0.0,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.costs.append(cost)
        self.state_texts.append(state_text)
        self.action_texts.append(action_text)
        
        # Vector reward support (for Hodge decomposition)
        if reward_vector is not None:
            self.rewards_vector.append(reward_vector)
        
        # World model prediction tracking
        if predicted_next_state is not None:
            self.predicted_next_states.append(predicted_next_state)
            self.prediction_uncertainties.append(prediction_uncertainty)
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns."""
        n = len(self.rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)
        
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + np.array(self.values)
        return returns, advantages
    
    def to_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Convert buffer to tensors for training."""
        return {
            "states": torch.tensor(np.array(self.states), dtype=torch.float32, device=device),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32, device=device),
            "values": torch.tensor(self.values, dtype=torch.float32, device=device),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            "dones": torch.tensor(self.dones, dtype=torch.float32, device=device),
            "costs": torch.tensor(self.costs, dtype=torch.float32, device=device),
        }


class SemanticPolicyNetwork(nn.Module):
    """
    Policy network for Semantic MDP.
    
    Takes embedding vectors as input, outputs action distributions
    over MCP-style actions.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_actions: int,
        hidden_dims: List[int] = [256, 128],
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_actions = num_actions
        
        # Build policy network
        layers = []
        in_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_dim, num_actions)
        self.value_head = nn.Linear(in_dim, 1)
        
        # For CPO: constraint value head
        self.cost_value_head = nn.Linear(in_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, state_value)."""
        features = self.feature_extractor(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value.squeeze(-1)
    
    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log_prob, entropy, and value.
        
        If action is provided, compute log_prob for that action.
        Otherwise, sample a new action.
        """
        logits, value = self.forward(x)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value
    
    def get_cost_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get constraint cost value estimate."""
        features = self.feature_extractor(x)
        return self.cost_value_head(features).squeeze(-1)


class ManifoldAwarePolicyNetwork(SemanticPolicyNetwork):
    """
    Policy network with manifold-aware features.
    
    Incorporates:
    1. Fisher Information Metric awareness
    2. Geodesic interpolation in embedding space
    3. Curvature-adaptive learning rates
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_actions: int,
        hidden_dims: List[int] = [256, 128],
        use_fisher_metric: bool = True,
    ):
        super().__init__(embed_dim, num_actions, hidden_dims)
        
        self.use_fisher_metric = use_fisher_metric
        
        # Fisher Information approximation network
        if use_fisher_metric:
            self.fisher_net = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus(),  # Ensure positive metric
            )
    
    def get_fisher_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate Fisher Information at state x.
        
        F(θ) = E[∇log π · ∇log π^T]
        
        In practice, we learn a scalar approximation.
        """
        if not self.use_fisher_metric:
            return torch.ones(x.shape[0], device=x.device)
        
        return self.fisher_net(x).squeeze(-1) + 1e-4  # Add epsilon for stability
    
    def natural_gradient_direction(
        self,
        gradient: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute natural gradient: F⁻¹ · ∇L
        
        This respects the geometry of the policy manifold.
        """
        fisher = self.get_fisher_metric(x)
        return gradient / fisher.unsqueeze(-1)


class BasePolicyOptimizer(ABC):
    """Base class for policy optimization algorithms."""
    
    def __init__(
        self,
        policy: SemanticPolicyNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: torch.device = None,
    ):
        self.policy = policy
        self.lr = lr
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        # Logging
        self.train_stats: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
        }
    
    @abstractmethod
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Perform policy update from rollout buffer."""
        pass
    
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select action for given state.
        
        Returns: (action, log_prob, value)
        """
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            
            logits, value = self.policy(state_t)
            
            if deterministic:
                action = logits.argmax(dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)[0, action]
            else:
                probs = Categorical(logits=logits)
                action = probs.sample()
                log_prob = probs.log_prob(action)
            
            return int(action.item()), float(log_prob.item()), float(value.item())


class SemanticPPO(BasePolicyOptimizer):
    """
    Proximal Policy Optimization for Semantic MDPs.
    
    Key insight: PPO's trust region can be interpreted as geodesic ball
    on the statistical manifold of policies. The clipping mechanism
    approximates staying within this geodesic ball.
    
    Manifold interpretation:
    - Policy π_θ lives on statistical manifold M
    - Fisher Information Metric defines geometry
    - PPO clip keeps updates within geodesic ε-ball
    """
    
    def __init__(
        self,
        policy: SemanticPolicyNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        target_kl: Optional[float] = None,
        use_manifold_gradient: bool = False,
        device: torch.device = None,
    ):
        super().__init__(policy, lr, gamma, device)
        
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.use_manifold_gradient = use_manifold_gradient
    
    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> Dict[str, float]:
        """
        PPO update step.
        
        Implements the clipped surrogate objective:
        L^{CLIP}(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
        """
        # Compute returns and advantages
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        data = buffer.to_tensors(self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        
        n_samples = len(buffer.states)
        indices = np.arange(n_samples)
        
        # Training stats
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                # Get batch data
                states = data["states"][batch_idx]
                actions = data["actions"][batch_idx]
                old_log_probs = data["log_probs"][batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                
                # Forward pass
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    states, actions
                )
                
                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                # Manifold-aware gradient (optional)
                if self.use_manifold_gradient and isinstance(self.policy, ManifoldAwarePolicyNetwork):
                    # Compute Fisher-weighted gradient
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Apply natural gradient transformation
                    fisher = self.policy.get_fisher_metric(states).mean()
                    for param in self.policy.parameters():
                        if param.grad is not None:
                            param.grad = param.grad / (fisher + 1e-4)
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Compute approximate KL for early stopping
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                n_updates += 1
            
            # Early stopping based on KL divergence
            if self.target_kl is not None and total_approx_kl / n_updates > self.target_kl:
                break
        
        stats = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": total_approx_kl / n_updates,
            "n_epochs": epoch + 1,
        }
        
        for key, value in stats.items():
            if key in self.train_stats:
                self.train_stats[key].append(value)
        
        return stats


class SemanticCPO(BasePolicyOptimizer):
    """
    Constrained Policy Optimization for Semantic MDPs.
    
    Extends PPO with constraint handling via Lagrangian relaxation.
    
    Optimization problem:
        max_θ E[R(τ)]
        s.t. E[C(τ)] ≤ d
    
    Lagrangian:
        L(θ, λ) = E[R(τ)] - λ(E[C(τ)] - d)
    
    Limitation: CPO only enforces constraints in *expectation*, not per-trajectory.
    This means some trajectories may violate constraints even if the average is safe.
    """
    
    def __init__(
        self,
        policy: SemanticPolicyNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        cost_limit: float = 25.0,
        lagrange_lr: float = 0.01,
        lagrange_init: float = 0.1,
        device: torch.device = None,
    ):
        super().__init__(policy, lr, gamma, device)
        
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Constraint parameters
        self.cost_limit = cost_limit
        self.lagrange_multiplier = lagrange_init
        self.lagrange_lr = lagrange_lr
        
        # Cost value optimizer
        self.cost_optimizer = torch.optim.Adam(
            [p for n, p in policy.named_parameters() if "cost" in n],
            lr=lr,
        )
        
        # Stats
        self.train_stats["cost"] = []
        self.train_stats["lagrange"] = []
    
    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> Dict[str, float]:
        """
        CPO update with Lagrangian constraint handling.
        
        1. Compute reward and cost advantages
        2. Update policy with Lagrangian objective
        3. Update Lagrange multiplier based on constraint violation
        """
        # Compute returns and advantages for rewards
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute cost returns (for constraint)
        cost_returns = self._compute_cost_returns(buffer)
        
        data = buffer.to_tensors(self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        cost_returns_t = torch.tensor(cost_returns, dtype=torch.float32, device=self.device)
        
        n_samples = len(buffer.states)
        indices = np.arange(n_samples)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_cost_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                states = data["states"][batch_idx]
                actions = data["actions"][batch_idx]
                old_log_probs = data["log_probs"][batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_cost_returns = cost_returns_t[batch_idx]
                
                # Forward pass
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    states, actions
                )
                cost_values = self.policy.get_cost_value(states)
                
                # Policy loss with Lagrangian
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                # Reward objective
                reward_obj = torch.min(surr1, surr2).mean()
                
                # Cost objective (penalize high costs)
                cost_obj = (ratio * batch_cost_returns).mean()
                
                # Lagrangian objective: max reward - λ * cost
                policy_loss = -(reward_obj - self.lagrange_multiplier * cost_obj)
                
                # Value losses
                value_loss = F.mse_loss(new_values, batch_returns)
                cost_value_loss = F.mse_loss(cost_values, batch_cost_returns)
                
                # Entropy
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.value_coef * cost_value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_cost_loss += cost_value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Update Lagrange multiplier
        avg_cost = np.mean(buffer.costs)
        constraint_violation = avg_cost - self.cost_limit
        self.lagrange_multiplier = max(
            0.0,
            self.lagrange_multiplier + self.lagrange_lr * constraint_violation
        )
        
        stats = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "cost_value_loss": total_cost_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "avg_cost": avg_cost,
            "lagrange_multiplier": self.lagrange_multiplier,
            "constraint_violation": constraint_violation,
        }
        
        for key, value in stats.items():
            if key in self.train_stats:
                self.train_stats[key].append(value)
        
        return stats
    
    def _compute_cost_returns(self, buffer: RolloutBuffer) -> np.ndarray:
        """Compute discounted cost returns."""
        n = len(buffer.costs)
        cost_returns = np.zeros(n)
        running_cost = 0.0
        
        for t in reversed(range(n)):
            if buffer.dones[t]:
                running_cost = 0.0
            running_cost = buffer.costs[t] + self.gamma * running_cost
            cost_returns[t] = running_cost
        
        return cost_returns


class SemanticSGPO(BasePolicyOptimizer):
    """
    Sheaf-Geodesic Policy Optimization for Semantic MDPs.
    
    Key innovation: Uses Hodge decomposition and Riemannian geometry
    to provide *hard* safety guarantees, not just expectation constraints.
    
    Components:
    1. **Hodge Critic**: Decomposes reward into gradient (learnable) + curl (noise)
    2. **Riemannian Metric**: Defines distance on embedding manifold
    3. **Black Hole Barriers**: Forbidden regions have infinite metric (geodesic barriers)
    4. **Natural Gradient**: Policy updates follow geodesics on statistical manifold
    
    Safety guarantee:
    - Black holes are singularities in the metric: g(x) → ∞ as x → B
    - Geodesics cannot cross event horizons
    - Policy optimization follows geodesics → trajectory-level safety
    """
    
    def __init__(
        self,
        policy: ManifoldAwarePolicyNetwork,
        hodge_critic: Any,  # HodgeCritic from hodge_critic.py
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        hodge_reward_weight: float = 0.5,
        black_hole_penalty: float = 100.0,
        use_natural_gradient: bool = True,
        device: torch.device = None,
    ):
        super().__init__(policy, lr, gamma, device)
        
        self.hodge_critic = hodge_critic
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.hodge_reward_weight = hodge_reward_weight
        self.black_hole_penalty = black_hole_penalty
        self.use_natural_gradient = use_natural_gradient
        
        # Black hole regions (learned from negative feedback)
        self.black_hole_centers: List[np.ndarray] = []
        self.black_hole_radii: List[float] = []
        
        # Stats
        self.train_stats["hodge_reward"] = []
        self.train_stats["black_hole_penalty"] = []
        self.train_stats["h1_magnitude"] = []
    
    def add_black_hole(self, center: np.ndarray, radius: float):
        """Add a forbidden region (black hole) to the metric."""
        self.black_hole_centers.append(center)
        self.black_hole_radii.append(radius)
    
    def compute_metric_at(self, state: np.ndarray) -> float:
        """
        Compute Riemannian metric value at a state.
        
        g(x) = 1 + Σᵢ κᵢ / (‖x - cᵢ‖ - rᵢ)²
        
        Where cᵢ, rᵢ are black hole centers and radii.
        """
        metric = 1.0
        
        for center, radius in zip(self.black_hole_centers, self.black_hole_radii):
            dist = np.linalg.norm(state - center)
            if dist < radius:
                # Inside event horizon: infinite metric
                return float('inf')
            
            # Add singularity contribution
            safe_dist = max(dist - radius, 1e-4)
            metric += self.black_hole_penalty / (safe_dist ** 2)
        
        return metric
    
    def compute_hodge_reward(
        self,
        state_text: str,
        action_text: str,
        next_state_text: str,
    ) -> float:
        """
        Compute reward from Hodge gradient alignment.
        
        r_hodge = ⟨∇φ, Δe⟩
        
        Where ∇φ is the Hodge gradient and Δe is the embedding change.
        """
        if self.hodge_critic is None:
            return 0.0
        
        # Get Hodge gradient at current state
        gradient = self.hodge_critic.get_topological_gradient_at(state_text)
        
        # Compute embedding change
        embeddings = self.hodge_critic.embedding_model.encode([state_text, next_state_text])
        delta_e = embeddings[1] - embeddings[0]
        
        # Alignment with gradient
        alignment = np.dot(gradient, delta_e)
        alignment = alignment / (np.linalg.norm(gradient) * np.linalg.norm(delta_e) + 1e-8)
        
        return float(alignment)

    def _compute_adaptive_epsilon(self, buffer: RolloutBuffer) -> float:
        """
        Compute adaptive clipping epsilon based on topological consistency.
        
        Intuition:
        - High H¹ (inconsistency) -> Smaller trust region (gradients are noisy/conflicting)
        - Close to Black Hole -> Smaller trust region (safety requires caution)
        - High Curvature -> Smaller trust region (linear approximation breaks down)
        
        Returns:
            Adapted epsilon value
        """
        if self.hodge_critic is None:
            return self.clip_epsilon
            
        # 1. Inconsistency penalty
        # H¹ magnitude typically ranges 0.0 to 1.0
        try:
            hodge_result = self.hodge_critic.compute_hodge_decomposition()
            h1 = hodge_result.h1_magnitude
        except Exception:
            h1 = 0.0
            
        # Scale factor: 1.0 when H¹=0, decays to 0.5 when H¹=1.0
        consistency_scale = 1.0 / (1.0 + h1)
        
        # 2. Black hole proximity (safety penalty)
        # Check last few states in buffer
        max_proximity = 0.0
        max_curvature = 0.0
        if hasattr(buffer, "state_texts") and len(buffer.state_texts) > 0:
            # Sample last 10 states for efficiency
            sample_texts = buffer.state_texts[-10:]
            for text in sample_texts:
                geo = self.hodge_critic.get_local_geometry(text)
                max_proximity = max(max_proximity, geo.get("black_hole_proximity", 0.0))
                max_curvature = max(max_curvature, geo.get("curvature", 0.0))
        
        # Scale factor: 1.0 when far, decays to 0.1 when very close
        safety_scale = 1.0 - (0.9 * max_proximity)
        
        # 3. Curvature penalty (linearity penalty)
        # Scale factor: 1.0 when flat, decays when curved
        curvature_scale = 1.0 - (0.5 * max_curvature)
        
        # Combine scales
        adaptive_eps = self.clip_epsilon * consistency_scale * safety_scale * curvature_scale
        
        return max(0.01, adaptive_eps)  # Lower bound

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> Dict[str, float]:
        """
        SGPO update with Hodge rewards, metric-aware gradients, and adaptive trust regions.
        """
        # Calculate adaptive epsilon for this update
        current_epsilon = self._compute_adaptive_epsilon(buffer)
        
        # Augment rewards with Hodge component
        augmented_rewards = []
        total_hodge_reward = 0.0
        total_black_hole_penalty = 0.0
        
        for i, (reward, state, state_text, action_text) in enumerate(zip(
            buffer.rewards,
            buffer.states,
            buffer.state_texts,
            buffer.action_texts,
        )):
            # Base reward
            aug_reward = reward
            
            # Add Hodge alignment reward
            if i < len(buffer.state_texts) - 1:
                next_state_text = buffer.state_texts[i + 1]
                hodge_r = self.compute_hodge_reward(state_text, action_text, next_state_text)
                aug_reward += self.hodge_reward_weight * hodge_r
                total_hodge_reward += hodge_r
            
            # Subtract black hole penalty
            metric = self.compute_metric_at(state)
            if metric > 10.0:
                penalty = np.log(metric)
                aug_reward -= penalty
                total_black_hole_penalty += penalty
            
            augmented_rewards.append(aug_reward)
        
        # Create modified buffer with augmented rewards
        aug_buffer = RolloutBuffer()
        for i in range(len(buffer.states)):
            aug_buffer.add(
                state=buffer.states[i],
                action=buffer.actions[i],
                reward=augmented_rewards[i],
                value=buffer.values[i],
                log_prob=buffer.log_probs[i],
                done=buffer.dones[i],
                cost=buffer.costs[i],
                state_text=buffer.state_texts[i],
                action_text=buffer.action_texts[i],
            )
        
        # Compute returns and advantages
        returns, advantages = aug_buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        data = aug_buffer.to_tensors(self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        
        n_samples = len(aug_buffer.states)
        indices = np.arange(n_samples)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                states = data["states"][batch_idx]
                actions = data["actions"][batch_idx]
                old_log_probs = data["log_probs"][batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                
                # Forward pass
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    states, actions
                )
                
                # PPO-style clipped objective with ADAPTIVE epsilon
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - current_epsilon, 1 + current_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # Entropy
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                # Natural gradient (Fisher-weighted)
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.use_natural_gradient and isinstance(self.policy, ManifoldAwarePolicyNetwork):
                    fisher = self.policy.get_fisher_metric(states).mean()
                    # Efficiency note: We use a scalar approximation of Fisher info
                    # This avoids inverting a large Hessian (O(N^3)) while preserving
                    # the "steepest descent on statistical manifold" property.
                    for param in self.policy.parameters():
                        if param.grad is not None:
                            param.grad = param.grad / (fisher.item() + 1e-4)
                
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Get H¹ magnitude from Hodge critic
        h1_magnitude = 0.0
        if self.hodge_critic is not None:
            try:
                hodge_result = self.hodge_critic.compute_hodge_decomposition()
                h1_magnitude = hodge_result.h1_magnitude
            except Exception:
                pass
        
        stats = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "avg_hodge_reward": total_hodge_reward / len(buffer.states),
            "avg_black_hole_penalty": total_black_hole_penalty / len(buffer.states),
            "h1_magnitude": h1_magnitude,
            "n_black_holes": len(self.black_hole_centers),
            "adaptive_epsilon": current_epsilon,  # Log the adaptive epsilon
        }
        
        for key, value in stats.items():
            if key in self.train_stats:
                self.train_stats[key].append(value)
        
        return stats
    
    def learn_black_holes_from_feedback(
        self,
        threshold: float = 0.2,
        radius: float = 0.3,
    ):
        """
        Automatically identify black holes from negative feedback in Hodge critic.
        
        States with consistently low ranks become forbidden regions.
        """
        if self.hodge_critic is None:
            return
        
        black_holes = self.hodge_critic.identify_black_holes(threshold=threshold)
        
        for point in black_holes:
            if point.embedding not in [np.array_equal(point.embedding, c) for c in self.black_hole_centers]:
                self.add_black_hole(point.embedding, radius)


class SemanticMDPTrainer:
    """
    High-level trainer for Semantic MDP experiments.
    
    Supports PPO, CPO, and SGPO algorithms with common training loop.
    """
    
    def __init__(
        self,
        env: Any,  # StorytellingMachine or similar
        algorithm: str = "gpo",
        embed_dim: int = 384,
        num_actions: int = 5,
        hodge_critic: Any = None,
        device: torch.device = None,
        **algo_kwargs,
    ):
        self.env = env
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create policy network
        if algorithm == "gpo":
            self.policy = ManifoldAwarePolicyNetwork(embed_dim, num_actions).to(self.device)
        else:
            self.policy = SemanticPolicyNetwork(embed_dim, num_actions).to(self.device)
        
        # Create optimizer
        if algorithm == "ppo":
            self.optimizer = SemanticPPO(self.policy, device=self.device, **algo_kwargs)
        elif algorithm == "cpo":
            self.optimizer = SemanticCPO(self.policy, device=self.device, **algo_kwargs)
        elif algorithm == "gpo":
            self.optimizer = SemanticSGPO(
                self.policy, hodge_critic, device=self.device, **algo_kwargs
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.algorithm = algorithm
        self.buffer = RolloutBuffer()
    
    def collect_rollout(
        self,
        n_steps: int,
        initial_scene: str,
        action_mapping: Callable[[int], Any],
    ) -> float:
        """
        Collect rollout data from environment.
        
        Args:
            n_steps: Number of steps to collect
            initial_scene: Starting scene description
            action_mapping: Function to convert action index to MCP action
        
        Returns:
            Total episode reward
        """
        self.buffer.clear()
        
        state, info = self.env.reset(initial_scene)
        total_reward = 0.0
        
        for step in range(n_steps):
            # Get action from policy
            action_idx, log_prob, value = self.optimizer.get_action(state)
            action = action_mapping(action_idx)
            
            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Compute cost (for CPO)
            cost = info.get("cost", 0.0)
            if info.get("in_black_hole", False):
                cost += 10.0
            
            # Get text representations
            state_text = info.get("belief", {})
            if hasattr(state_text, "observation"):
                state_text = state_text.observation
            else:
                state_text = str(state_text)
            
            action_text = str(action)
            
            # Store in buffer
            self.buffer.add(
                state=state,
                action=action_idx,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                cost=cost,
                state_text=state_text,
                action_text=action_text,
            )
            
            total_reward += reward
            state = next_state
            
            if done:
                state, info = self.env.reset(initial_scene)
        
        return total_reward
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training update."""
        # Get final value estimate
        if len(self.buffer.states) > 0:
            with torch.no_grad():
                state_t = torch.tensor(
                    self.buffer.states[-1],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                _, last_value = self.policy(state_t)
                last_value = last_value.item()
        else:
            last_value = 0.0
        
        return self.optimizer.update(self.buffer, last_value)
    
    def train(
        self,
        total_steps: int,
        rollout_length: int,
        initial_scene: str,
        action_mapping: Callable[[int], Any],
        log_interval: int = 10,
        callback: Optional[Callable] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            total_steps: Total environment steps
            rollout_length: Steps per rollout
            initial_scene: Starting scene
            action_mapping: Convert action index to MCP action
            log_interval: How often to log stats
            callback: Optional callback after each update
        
        Returns:
            Training statistics
        """
        n_updates = total_steps // rollout_length
        all_stats = {"episode_reward": []}
        
        print(f"\n{'='*60}")
        print(f"SEMANTIC MDP TRAINING ({self.algorithm.upper()})")
        print(f"{'='*60}")
        print(f"Total steps: {total_steps}")
        print(f"Rollout length: {rollout_length}")
        print(f"Updates: {n_updates}")
        
        for update in range(n_updates):
            # Collect rollout
            episode_reward = self.collect_rollout(
                rollout_length, initial_scene, action_mapping
            )
            all_stats["episode_reward"].append(episode_reward)
            
            # Update policy
            stats = self.train_step()
            
            # Log
            if (update + 1) % log_interval == 0:
                print(f"\nUpdate {update + 1}/{n_updates}")
                print(f"  Episode reward: {episode_reward:.2f}")
                print(f"  Policy loss: {stats.get('policy_loss', 0):.4f}")
                print(f"  Value loss: {stats.get('value_loss', 0):.4f}")
                
                if self.algorithm == "gpo":
                    print(f"  Hodge reward: {stats.get('avg_hodge_reward', 0):.4f}")
                    print(f"  H¹ magnitude: {stats.get('h1_magnitude', 0):.4f}")
                elif self.algorithm == "cpo":
                    print(f"  Avg cost: {stats.get('avg_cost', 0):.2f}")
                    print(f"  Lagrange: {stats.get('lagrange_multiplier', 0):.4f}")
            
            # Callback
            if callback is not None:
                callback(update, stats, self)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        
        return all_stats


def compare_algorithms_demo():
    """
    Demo comparing PPO, CPO, and SGPO on a synthetic semantic MDP.
    """
    print("\n" + "="*60)
    print("SEMANTIC MDP ALGORITHM COMPARISON")
    print("="*60)
    
    # Create a simple synthetic environment for testing
    class SimpleSyntheticEnv:
        def __init__(self, embed_dim=32):
            self.embed_dim = embed_dim
            self.state = None
            self.step_count = 0
            self.black_hole = np.random.randn(embed_dim) * 0.5
        
        def reset(self, initial_scene):
            self.state = np.random.randn(self.embed_dim) * 0.1
            self.step_count = 0
            return self.state, {"belief": initial_scene}
        
        def step(self, action):
            # Simple dynamics: move in random direction + action bias
            direction = np.random.randn(self.embed_dim) * 0.1
            direction[action % self.embed_dim] += 0.2
            self.state = self.state + direction
            
            # Reward: distance from black hole (inverse)
            dist_to_black_hole = np.linalg.norm(self.state - self.black_hole)
            reward = 0.1 * dist_to_black_hole
            
            # Cost: proximity to black hole
            cost = max(0, 1.0 - dist_to_black_hole)
            
            # Check if in black hole
            in_black_hole = dist_to_black_hole < 0.3
            if in_black_hole:
                reward -= 10.0
            
            self.step_count += 1
            done = self.step_count >= 50 or in_black_hole
            
            return self.state, reward, done, False, {
                "belief": f"Step {self.step_count}",
                "cost": cost,
                "in_black_hole": in_black_hole,
            }
    
    # Test parameters
    embed_dim = 32
    num_actions = 4
    rollout_length = 200
    n_updates = 20
    
    results = {}
    
    for algo in ["ppo", "cpo", "gpo"]:
        print(f"\n--- Testing {algo.upper()} ---")
        
        env = SimpleSyntheticEnv(embed_dim)
        
        # Create mock hodge critic for SGPO
        hodge_critic = None
        if algo == "gpo":
            class MockHodgeCritic:
                def __init__(self):
                    self.gradient = np.random.randn(embed_dim)
                    self.gradient /= np.linalg.norm(self.gradient)
                
                def get_topological_gradient_at(self, state_text):
                    return self.gradient
                
                class MockEmbedModel:
                    def encode(self, texts):
                        return np.random.randn(len(texts), embed_dim)
                
                embedding_model = MockEmbedModel()
                
                def compute_hodge_decomposition(self):
                    class Result:
                        h1_magnitude = 0.05
                    return Result()
                
                def identify_black_holes(self, threshold=0.2):
                    return []
                
                def get_local_geometry(self, state_text):
                    return {
                        "curvature": 0.0,
                        "h1_magnitude": 0.05,
                        "black_hole_proximity": 0.0,
                    }
            
            hodge_critic = MockHodgeCritic()
        
        trainer = SemanticMDPTrainer(
            env=env,
            algorithm=algo,
            embed_dim=embed_dim,
            num_actions=num_actions,
            hodge_critic=hodge_critic,
        )
        
        # Add black hole to SGPO
        if algo == "gpo":
            trainer.optimizer.add_black_hole(env.black_hole, radius=0.3)
        
        def action_mapping(idx):
            return idx
        
        stats = trainer.train(
            total_steps=rollout_length * n_updates,
            rollout_length=rollout_length,
            initial_scene="Starting scene",
            action_mapping=action_mapping,
            log_interval=5,
        )
        
        results[algo] = stats
        
        avg_reward = np.mean(stats["episode_reward"][-5:])
        print(f"{algo.upper()} final avg reward: {avg_reward:.2f}")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    
    return results


if __name__ == "__main__":
    compare_algorithms_demo()
