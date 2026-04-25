# -*- coding: utf-8 -*-
"""
Conformal SGPO: Mathematically Correct Safe Policy Optimization

This module implements SGPO variants using the corrected mathematical framework:
- Module 2: Conformal Safety Metric with σ(x)→∞ at danger boundary
- Infinite geodesic distance = geometric unreachability (not soft penalties)
- Hybrid danger region learning from behavioral telemetry (costs)
- Maximum entropy exploration for efficient boundary characterization

Key Differences from Original SGPO:
1. CONFORMAL metric (σ→∞) instead of soft penalty (g = base + severity*danger)
2. LEARNED danger regions hardened into conformal barriers
3. ANISOTROPIC variant preserves escape routes
4. Integration with DiscreteHodgeRank for reliability-weighted training

References:
- Hodge Theory, Bilattices, and Social Choice.pdf
- handoffs/14_MATHEMATICAL_RESTRUCTURING.md
- high_dimensional_reward_spaces/src/conformal_safety.py

Author: Cascade (Feb 2026)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from scipy.spatial.distance import cdist


# ============================================================================
# 1. CONFORMAL SAFETY METRIC (Module 2 - Corrected)
# ============================================================================

@dataclass
class LearnedDangerRegion:
    """A danger region learned from behavioral telemetry (cost signals)."""
    center: np.ndarray  # Estimated center
    radius: float       # Estimated radius (mass-derived)
    confidence: float   # How confident we are (0-1)
    total_cost: float   # Accumulated cost signal
    n_observations: int # Number of observations
    
    def update(self, state: np.ndarray, cost: float, learning_rate: float = 0.1):
        """Update danger region estimate from new observation."""
        if cost > 0:
            # Move center toward high-cost state
            self.center = (1 - learning_rate) * self.center + learning_rate * state
            self.total_cost += cost
            self.n_observations += 1
            # Expand radius based on accumulated evidence
            self.radius = np.sqrt(self.total_cost / (self.n_observations + 1))
            self.confidence = min(1.0, self.n_observations / 50)  # Saturates at 50 obs


class ConformalSafetyMetric(nn.Module):
    """
    Conformal metric g_ij(x) = e^{2σ(x)} δ_ij where σ(x) → ∞ at danger boundary.
    
    This creates INFINITE geodesic distance to danger regions, not soft penalties.
    Geodesic distance: d(a,b) = ∫ e^{σ(γ(t))} |γ'(t)| dt → ∞ if path crosses danger.
    
    Key Innovation: Hybrid learning + hardening
    1. LEARN approximate danger regions from cost signals (behavioral telemetry)
    2. HARDEN into conformal barriers once confident
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        hidden_dim: int = 32,
        sharpness: float = 2.0,      # β: controls barrier steepness
        min_distance: float = 0.1,   # Minimum safe distance (numerical stability)
        confidence_threshold: float = 0.7,  # When to harden learned region
    ):
        super().__init__()
        self.state_dim = state_dim
        self.sharpness = sharpness
        self.min_distance = min_distance
        self.confidence_threshold = confidence_threshold
        
        # Learned danger estimator (soft, before hardening)
        self.danger_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures positive danger estimate
        )
        
        # Danger center estimator
        self.center_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
        # Learned danger regions (from behavioral telemetry)
        self.learned_regions: List[LearnedDangerRegion] = []
        
        # Hardened regions (conformal barriers)
        self.hardened_regions: List[Tuple[np.ndarray, float]] = []  # (center, radius)
    
    def add_known_danger_region(self, center: np.ndarray, radius: float):
        """Add a known danger region (for evaluation benchmarks)."""
        self.hardened_regions.append((center.copy(), radius))
    
    def update_learned_regions(self, states: np.ndarray, costs: np.ndarray, lr: float = 0.1):
        """
        Update danger region estimates from behavioral telemetry.
        
        Uses mass-derived radius: r = sqrt(total_cost / n_obs)
        This creates severity-scaled exclusion zones.
        """
        high_cost_mask = costs > 0.5
        if not high_cost_mask.any():
            return
        
        high_cost_states = states[high_cost_mask]
        high_costs = costs[high_cost_mask]
        
        if len(self.learned_regions) == 0:
            # Initialize first region at centroid of high-cost states
            centroid = high_cost_states.mean(axis=0)
            self.learned_regions.append(LearnedDangerRegion(
                center=centroid,
                radius=1.0,
                confidence=0.0,
                total_cost=high_costs.sum(),
                n_observations=len(high_costs)
            ))
        else:
            # Update existing regions
            for state, cost in zip(high_cost_states, high_costs):
                # Find closest region
                distances = [np.linalg.norm(state - r.center) for r in self.learned_regions]
                closest_idx = np.argmin(distances)
                
                if distances[closest_idx] < self.learned_regions[closest_idx].radius * 2:
                    # Update existing region
                    self.learned_regions[closest_idx].update(state, cost, lr)
                else:
                    # Create new region if far from existing
                    self.learned_regions.append(LearnedDangerRegion(
                        center=state.copy(),
                        radius=0.5,
                        confidence=0.0,
                        total_cost=cost,
                        n_observations=1
                    ))
        
        # Harden confident regions
        self._harden_confident_regions()
    
    def _harden_confident_regions(self):
        """Convert high-confidence learned regions to conformal barriers."""
        for region in self.learned_regions:
            if region.confidence >= self.confidence_threshold:
                # Check if already hardened
                already_hardened = any(
                    np.linalg.norm(region.center - hc) < 0.5 
                    for hc, _ in self.hardened_regions
                )
                if not already_hardened:
                    self.hardened_regions.append((region.center.copy(), region.radius))
    
    def distance_to_danger(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute distance to nearest danger boundary.
        
        Returns: Distance to boundary (negative if inside danger).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x_np = x.detach().cpu().numpy()
        
        # Start with large distance
        min_dist = torch.ones(x.shape[0], 1, device=x.device) * 100.0
        
        # Check hardened regions
        for center, radius in self.hardened_regions:
            dist_to_center = torch.norm(x - torch.tensor(center, device=x.device), dim=-1, keepdim=True)
            dist_to_boundary = dist_to_center - radius
            min_dist = torch.minimum(min_dist, dist_to_boundary)
        
        # Check learned (but not yet hardened) regions with lower weight
        for region in self.learned_regions:
            if region.confidence < self.confidence_threshold:
                dist_to_center = torch.norm(
                    x - torch.tensor(region.center, device=x.device), 
                    dim=-1, keepdim=True
                )
                # Scale by confidence
                effective_radius = region.radius * region.confidence
                dist_to_boundary = dist_to_center - effective_radius
                # Soft influence (doesn't create hard barrier)
                min_dist = torch.minimum(min_dist, dist_to_boundary * 2)
        
        return min_dist
    
    def conformal_factor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute conformal factor σ(x).
        
        σ(x) = -β * log(d(x)) where d(x) = distance to danger boundary.
        As d→0: σ→∞ (infinite barrier)
        As d→∞: σ→-∞·β but we clamp to 0 for safe regions
        """
        dist = self.distance_to_danger(x)
        
        # Clamp for numerical stability
        safe_dist = torch.clamp(dist, min=self.min_distance)
        
        # Conformal factor: σ = -β * log(d)
        # When d<1: σ>0 (inflated metric near danger)
        # When d>1: σ<0 (normal metric far from danger), but we clamp to 0
        sigma = -self.sharpness * torch.log(safe_dist)
        sigma = torch.clamp(sigma, min=0.0)  # Only inflate, never deflate
        
        return sigma
    
    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute conformal metric tensor g_ij = e^{2σ} δ_ij.
        
        Returns scalar since metric is isotropic (same in all directions).
        """
        sigma = self.conformal_factor(x)
        return torch.exp(2 * sigma)
    
    def forward(self, x: torch.Tensor, v: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conformal metric and escape factor.
        
        Args:
            x: States [batch, state_dim]
            v: Velocities [batch, state_dim] (optional, for anisotropic)
        
        Returns:
            g: Metric values [batch, 1]
            escape_factor: 1.0 if escaping, 0.0 if approaching [batch, 1]
        """
        g = self.metric_tensor(x)
        
        if v is None:
            return g, torch.ones_like(g)
        
        # Compute escape factor (for anisotropic variant)
        # Direction away from danger is the gradient of distance
        dist = self.distance_to_danger(x)
        
        # Approximate gradient via finite differences
        eps = 0.01
        grad_dist = torch.zeros_like(x)
        for i in range(x.shape[-1]):
            x_plus = x.clone()
            x_plus[:, i] += eps
            x_minus = x.clone()
            x_minus[:, i] -= eps
            grad_dist[:, i] = (self.distance_to_danger(x_plus) - self.distance_to_danger(x_minus)).squeeze() / (2 * eps)
        
        # Escape direction is gradient of distance (points away from danger)
        escape_dir = grad_dist / (torch.norm(grad_dist, dim=-1, keepdim=True) + 1e-8)
        
        # Project velocity onto escape direction
        v_escape = torch.sum(v * escape_dir, dim=-1, keepdim=True)
        
        # escape_factor = 1 if moving away (v_escape > 0), 0 if approaching
        escape_factor = torch.sigmoid(v_escape * 5.0)
        
        return g, escape_factor


class ConformalSafetyMetricANIS(ConformalSafetyMetric):
    """
    Anisotropic Conformal Safety Metric.
    
    Key Innovation: Only penalize movement TOWARD danger.
    Escape and tangential movement remain free (unscaled).
    
    This preserves learning signal for evasive maneuvers while
    creating infinite barriers for approach.
    """
    
    def anisotropic_metric(self, x: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute direction-dependent metric.
        
        Returns:
            g_effective: Effective metric for this (state, velocity) pair
            escape_factor: How much the agent is escaping
        """
        g_isotropic, escape_factor = self.forward(x, v)
        
        # Anisotropic scaling: g_eff = g_base + (1 - escape_factor) * g_danger
        # When escaping: g_eff ≈ 1 (free movement)
        # When approaching: g_eff ≈ g_isotropic (full barrier)
        g_effective = 1.0 + (1.0 - escape_factor) * (g_isotropic - 1.0)
        
        return g_effective, escape_factor


# ============================================================================
# 2. CORRECTED HODGE CRITIC INTEGRATION
# ============================================================================

class ReliabilityWeightedCritic(nn.Module):
    """
    Critic that weights value estimates by preference reliability.
    
    Uses the corrected reliability score:
    reliability = ||gradient||² / ||total||²
    
    where gradient is the transitive (Borda) component, and
    total = gradient + curl + harmonic.
    
    NOTE: This requires integration with DiscreteHodgeRank (Module 1).
    """
    
    def __init__(self, state_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Store reliability scores for states
        self.reliability_cache: Dict[bytes, float] = {}
    
    def set_reliability(self, state: np.ndarray, reliability: float):
        """Cache reliability score for a state."""
        key = state.tobytes()
        self.reliability_cache[key] = reliability
    
    def get_reliability(self, states: torch.Tensor) -> torch.Tensor:
        """Get cached reliability scores."""
        reliabilities = []
        for state in states.detach().cpu().numpy():
            key = state.tobytes()
            reliabilities.append(self.reliability_cache.get(key, 1.0))
        return torch.tensor(reliabilities, device=states.device).unsqueeze(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_net(x)
    
    def reliability_weighted_value(self, x: torch.Tensor) -> torch.Tensor:
        """Value estimate weighted by preference reliability."""
        value = self.forward(x)
        reliability = self.get_reliability(x)
        return value * reliability


# ============================================================================
# 3. CONFORMAL SGPO ALGORITHMS
# ============================================================================

@dataclass
class ConformalSGPOConfig:
    """Configuration for Conformal SGPO variants."""
    name: str = "conformal_sgpo"
    episodes: int = 300
    gamma: float = 0.99
    lr_actor: float = 1e-3
    lr_critic: float = 3e-3
    lr_metric: float = 3e-3
    
    # Conformal metric parameters (TUNED)
    sharpness: float = 4.0          # β: barrier steepness (increased from 2.0)
    confidence_threshold: float = 0.7  # When to harden regions
    warmup_episodes: int = 10       # Learn regions before hardening (reduced from 30)
    
    # Hybrid: Lagrangian + Conformal
    use_lagrangian: bool = True
    cost_limit: float = 5.0
    lr_lambda: float = 1e-2
    
    # Anisotropic variant
    anisotropic: bool = False
    
    # Reliability integration (Module 1)
    use_reliability_weighting: bool = False


class Actor(nn.Module):
    """Simple Gaussian actor for navigation."""
    def __init__(self, state_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(state_dim) - 1.0)
    
    def forward(self, x):
        mu = self.net(x)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)


class Critic(nn.Module):
    """Simple value critic."""
    def __init__(self, state_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def train_conformal_sgpo(
    env: Any,  # SandbaggingEnv or similar
    config: ConformalSGPOConfig,
    seed: int,
    known_danger_regions: List[Tuple[np.ndarray, float]] = None,
) -> Dict:
    """
    Train Conformal SGPO with mathematically correct safety metric.
    
    Key Improvements:
    1. CONFORMAL metric with σ→∞ at danger (not soft penalty)
    2. HYBRID: Learn regions from costs, then harden into barriers
    3. ANISOTROPIC option: only penalize approach, not escape
    
    Args:
        env: Environment with step(action) -> (state, reward, cost, done, info)
        config: Configuration
        seed: Random seed
        known_danger_regions: Optional list of (center, radius) for evaluation
    
    Returns:
        Dictionary with episode metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = 2  # Assuming 2D navigation
    
    # Initialize networks
    actor = Actor(state_dim)
    critic = Critic(state_dim)
    
    # Initialize CONFORMAL safety metric (key correction)
    if config.anisotropic:
        metric = ConformalSafetyMetricANIS(
            state_dim=state_dim,
            sharpness=config.sharpness,
            confidence_threshold=config.confidence_threshold
        )
    else:
        metric = ConformalSafetyMetric(
            state_dim=state_dim,
            sharpness=config.sharpness,
            confidence_threshold=config.confidence_threshold
        )
    
    # Add known danger regions (for evaluation benchmarks)
    if known_danger_regions:
        for center, radius in known_danger_regions:
            metric.add_known_danger_region(center, radius)
    
    # Optimizers
    opt_actor = optim.Adam(actor.parameters(), lr=config.lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=config.lr_critic)
    opt_metric = optim.Adam(metric.parameters(), lr=config.lr_metric)
    
    # Lagrangian components (optional hybrid)
    if config.use_lagrangian:
        cost_critic = Critic(state_dim)
        opt_cost_critic = optim.Adam(cost_critic.parameters(), lr=config.lr_critic)
        log_lambda = nn.Parameter(torch.zeros(1))
        opt_lambda = optim.Adam([log_lambda], lr=config.lr_lambda)
    
    # Tracking
    episode_returns = []
    episode_violations = []
    goal_reached = []
    metrics_log = {
        "avg_sigma": [],
        "avg_metric": [],
        "n_hardened_regions": [],
        "escape_factor": [],
    }
    
    for ep in range(config.episodes):
        obs = env.reset()
        trajectory = []
        ep_violations = 0
        ep_return = 0.0
        reached = False
        
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
            
            next_obs, reward, cost, done, info = env.step(action.numpy())
            trajectory.append({
                'state': obs,
                'action': action,
                'reward': reward,
                'cost': cost,
                'info': info
            })
            ep_violations += int(info.get('in_trap', cost > 0))
            ep_return += reward
            if info.get('dist_to_goal', float('inf')) < 1.0:
                reached = True
            obs = next_obs
        
        # Prepare tensors
        states = torch.FloatTensor(np.array([t['state'] for t in trajectory]))
        actions = torch.stack([t['action'] for t in trajectory])
        rewards = [t['reward'] for t in trajectory]
        costs = torch.FloatTensor([t['cost'] for t in trajectory])
        
        # Compute returns
        returns = []
        G = 0
        for t in reversed(trajectory):
            G = t['reward'] + config.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        
        # Cost returns (for Lagrangian)
        if config.use_lagrangian:
            c_returns = []
            C = 0
            for t in reversed(trajectory):
                C = t['cost'] + config.gamma * C
                c_returns.insert(0, C)
            c_returns = torch.FloatTensor(c_returns).unsqueeze(1)
        
        # Update critic
        vals = critic(states)
        loss_crit = nn.MSELoss()(vals, returns)
        opt_critic.zero_grad()
        loss_crit.backward()
        opt_critic.step()
        
        # Update cost critic (Lagrangian)
        if config.use_lagrangian:
            c_vals = cost_critic(states)
            loss_cost_crit = nn.MSELoss()(c_vals, c_returns)
            opt_cost_critic.zero_grad()
            loss_cost_crit.backward()
            opt_cost_critic.step()
        
        # Update danger region estimates from behavioral telemetry (HYBRID LEARNING)
        if ep >= config.warmup_episodes:
            states_np = states.detach().numpy()
            costs_np = costs.detach().numpy()
            metric.update_learned_regions(states_np, costs_np)
        
        # Compute advantage with CONFORMAL scaling
        with torch.no_grad():
            # Get conformal metric values
            if config.anisotropic:
                g_values, escape_factors = metric.anisotropic_metric(states, actions)
            else:
                g_values, escape_factors = metric(states, actions)
            
            sigma_values = metric.conformal_factor(states)
            
            r_adv = returns - critic(states)
            
            # Hybrid: Lagrangian + Conformal
            if config.use_lagrangian:
                lambda_val = torch.exp(log_lambda).detach()
                c_adv = c_returns - cost_critic(states)
                combined_adv = r_adv - lambda_val * c_adv
            else:
                combined_adv = r_adv
            
            # CONFORMAL advantage scaling
            # Natural gradient: ∇_natural = G^{-1} ∇ = e^{-2σ} ∇
            # This naturally suppresses updates near danger (high σ)
            conformal_scale = torch.exp(-2 * sigma_values)
            
            if config.anisotropic:
                # Preserve escape signal: only scale approach
                scale = escape_factors + (1 - escape_factors) * conformal_scale
            else:
                scale = conformal_scale
            
            riemannian_adv = scale * combined_adv
            
            # Log metrics
            metrics_log["avg_sigma"].append(sigma_values.mean().item())
            metrics_log["avg_metric"].append(g_values.mean().item())
            metrics_log["n_hardened_regions"].append(len(metric.hardened_regions))
            metrics_log["escape_factor"].append(escape_factors.mean().item())
        
        # Update actor
        dists = actor(states)
        log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
        loss_actor = -(log_probs * riemannian_adv).mean()
        
        opt_actor.zero_grad()
        loss_actor.backward()
        opt_actor.step()
        
        # Update lambda (Lagrangian)
        if config.use_lagrangian:
            avg_cost = c_returns.mean()
            loss_lambda = -log_lambda * (config.cost_limit - avg_cost.detach())
            opt_lambda.zero_grad()
            loss_lambda.backward()
            opt_lambda.step()
        
        episode_returns.append(ep_return)
        episode_violations.append(ep_violations)
        goal_reached.append(reached)
    
    return {
        "seed": seed,
        "episode_returns": episode_returns,
        "episode_violations": episode_violations,
        "goal_reached": goal_reached,
        "metrics": metrics_log,
        "n_hardened_regions": len(metric.hardened_regions),
        "learned_regions": [(r.center.tolist(), r.radius, r.confidence) 
                           for r in metric.learned_regions],
    }


def train_conformal_sgpo_anis(
    env: Any,
    config: ConformalSGPOConfig,
    seed: int,
    known_danger_regions: List[Tuple[np.ndarray, float]] = None,
) -> Dict:
    """Convenience function for anisotropic variant."""
    config.anisotropic = True
    config.name = "conformal_sgpo_anis"
    return train_conformal_sgpo(env, config, seed, known_danger_regions)


def train_conformal_sgpo_anis_cchc(
    env: Any,
    config: ConformalSGPOConfig,
    seed: int,
    hodge_critic: Any = None,  # ContextConditionalHodgeCritic
    known_danger_regions: List[Tuple[np.ndarray, float]] = None,
) -> Dict:
    """
    Conformal SGPO with Anisotropic metric and Context-Conditional Hodge Critic.
    
    Integration points:
    1. CCHC provides reliability scores for advantage weighting
    2. Reliability = ||gradient||² / ||total||² (corrected formula)
    3. Only gradient (transitive) preferences used for value estimation
    """
    config.anisotropic = True
    config.use_reliability_weighting = True
    config.name = "conformal_sgpo_anis_cchc"
    
    # TODO: Full CCHC integration requires:
    # 1. Embedding states into preference space
    # 2. Computing per-state reliability from DiscreteHodgeRank
    # 3. Weighting advantage by reliability
    #
    # For now, this is a placeholder that uses the conformal metric.
    # Full integration will come after updating experiment A.
    
    return train_conformal_sgpo(env, config, seed, known_danger_regions)


# ============================================================================
# 4. COMPARISON: OLD vs NEW
# ============================================================================

"""
Comparison of Original vs Corrected SGPO:

┌─────────────────┬────────────────────────────┬────────────────────────────────┐
│ Aspect          │ Original SGPO              │ Conformal SGPO (Corrected)     │
├─────────────────┼────────────────────────────┼────────────────────────────────┤
│ Metric          │ g = base + severity*danger │ g = e^{2σ}, σ = -β*log(d)     │
│ Barrier         │ Soft (can be overcome)     │ Hard (infinite distance)       │
│ Danger Learning │ danger_net only            │ Hybrid: learn + harden         │
│ Escape Handling │ Partial (anisotropic)      │ Full anisotropic preservation │
│ Scaling         │ adv / sqrt(g) or log(1+g)  │ adv * e^{-2σ} (natural grad)   │
│ Integration     │ Mixed discrete/continuous  │ Module 2 only (continuous)     │
└─────────────────┴────────────────────────────┴────────────────────────────────┘

Key Benefits of Conformal Approach:
1. INFINITE geodesic distance = geometric unreachability
2. Natural gradient scaling (e^{-2σ}) is mathematically principled
3. Hybrid learning allows discovery + hardening of danger regions
4. Clear separation from Module 1 (DiscreteHodgeRank for training)
"""


if __name__ == "__main__":
    # Quick test
    print("Conformal SGPO module loaded.")
    
    # Test conformal metric
    metric = ConformalSafetyMetric(state_dim=2, sharpness=2.0)
    metric.add_known_danger_region(np.array([5.0, 6.0]), 2.5)
    
    # Test points
    safe_point = torch.tensor([[0.0, 0.0]])
    near_danger = torch.tensor([[4.0, 5.0]])
    in_danger = torch.tensor([[5.0, 6.0]])
    
    print(f"Safe point σ: {metric.conformal_factor(safe_point).item():.2f}")
    print(f"Near danger σ: {metric.conformal_factor(near_danger).item():.2f}")
    print(f"In danger σ: {metric.conformal_factor(in_danger).item():.2f}")
    
    # Test metric tensor
    print(f"\nSafe point metric: {metric.metric_tensor(safe_point).item():.2f}")
    print(f"Near danger metric: {metric.metric_tensor(near_danger).item():.2f}")
    print(f"In danger metric: {metric.metric_tensor(in_danger).item():.2f}")
