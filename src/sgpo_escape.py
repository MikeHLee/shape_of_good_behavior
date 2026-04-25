"""
EscapeSGPO: Sheaf-Geodesic Policy Optimization with Black Hole Escape Mechanisms

An enhanced variant of ClippedSGPO designed to prevent policy freezing near known
black holes (forbidden regions) while maintaining safety guarantees.

Key improvements over ClippedSGPO:
1. Soft Singularities: Saturating metric function prevents infinite values
2. Repulsive Gradients: Active signal pushing policy away from danger
3. Adaptive Thresholds: Smooth transition zone instead of hard cutoff
4. Entropy Boost: Increased exploration incentive near boundaries

Mathematical Foundation:
- Metric saturation: g(x) = g_smooth(x) + Σᵢ min(strength_i / dist^α, g_max)
- Repulsive bonus: r_repel = β / (dist_to_nearest + ε)
- Adaptive scaling: smooth interpolation between [τ_soft, τ_hard]
- Dynamic entropy: H_coef = H_base * (1 + γ * proximity_factor)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class HodgeCriticInterface:
    """
    Simple Hodge critic interface for EscapeSGPO.
    
    Provides value estimation and harmonic component (cyclic preference correction).
    """
    
    def __init__(
        self,
        value_net: Optional[nn.Module] = None,
        embed_dim: int = 384,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cpu")
        self.embed_dim = embed_dim
        
        if value_net is not None:
            self.value_net = value_net.to(self.device)
        else:
            self.value_net = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            ).to(self.device)
        
        self.harmonic_net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)
        
        self._h1_magnitude = 0.0
    
    def value(self, states: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Estimate value function V(s)."""
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            values = self.value_net(states).squeeze(-1)
        
        return values.cpu().numpy()
    
    def harmonic(
        self,
        states: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Estimate harmonic component ω·v (cyclic preference correction)."""
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            omega = self.harmonic_net(states).squeeze(-1)
        
        return omega.cpu().numpy()


@dataclass
class EscapeSGPOConfig:
    """Configuration for EscapeSGPO algorithm."""
    # PPO-style clipping
    clip_ratio: float = 0.2
    
    # Adaptive threshold parameters (smooth transition zone)
    soft_threshold: float = 1.5       # Start dampening here
    hard_threshold: float = 10.0      # Maximum dampening (but not zero)
    
    # Singularity softening
    singularity_power: float = 1.5    # Slower divergence than default 2.0
    max_singularity_contribution: float = 1e4  # Saturation limit
    
    # Repulsive gradient injection
    repulsion_strength: float = 0.1   # β in repulsive bonus
    repulsion_epsilon: float = 0.1    # ε to prevent division by zero
    
    # Entropy boost near boundaries
    base_entropy_coef: float = 0.05   # Higher than standard 0.01
    entropy_boost_factor: float = 5.0 # Multiplier when near danger
    
    # Standard RL parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    
    # Learning rates
    lr: float = 3e-4
    metric_lr: float = 1e-4
    metric_update_freq: int = 5
    
    # CPO initialization
    cpo_cost_threshold: float = 0.5
    horizon_scale: float = 1.0
    
    # Adaptive discovery
    adaptive_discovery: bool = True
    discovery_threshold: float = 0.9
    max_singularities: int = 20


class AnisotropicSingularityMetric(nn.Module):
    """
    Anisotropic metric with DIRECTIONAL singularities.
    
    Key innovation: The metric only diverges in the direction TOWARD the black hole,
    not in tangential or escape directions. This creates a "one-way membrane" that:
    - Blocks approach to dangerous regions
    - Allows free movement tangentially (orbiting)
    - Allows free escape (backing out)
    
    Mathematically:
        g(x, v) = g_base * |v|² + (strength / dist^α) * |v_toward|²
    
    Where v_toward = max(0, -v · n̂) is the velocity component toward the black hole.
    
    This preserves the learning signal for escape maneuvers while blocking
    approaches to dangerous regions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        min_metric: float = 1.0,
        max_singularity_contribution: float = 1e4,
        singularity_power: float = 1.5,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.min_metric = min_metric
        self.max_contribution = max_singularity_contribution
        self.default_power = singularity_power
        
        # Smooth learned component
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.smooth_net = nn.Sequential(*layers)
        
        # Singularity storage
        self.singularities: List[Dict] = []
        self._centers: List[torch.Tensor] = []
    
    def add_singularity(
        self,
        center: Union[np.ndarray, torch.Tensor],
        radius: float,
        strength: float,
        power: Optional[float] = None,
        trainable: bool = False,
    ) -> int:
        """Add an anisotropic singularity (directional black hole)."""
        if isinstance(center, np.ndarray):
            center = torch.tensor(center, dtype=torch.float32)
        
        power = power if power is not None else self.default_power
        
        self.singularities.append({
            'center': center,
            'radius': radius,
            'strength': strength,
            'power': power,
            'trainable': trainable,
        })
        self._centers.append(center)
        
        return len(self.singularities) - 1
    
    def compute_directional_metric(
        self,
        x: torch.Tensor,
        v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anisotropic metric considering velocity direction.
        
        Args:
            x: State positions (batch_size, input_dim)
            v: Velocity vectors (batch_size, input_dim). If None, returns isotropic component.
            
        Returns:
            g_isotropic: Base metric without directional component
            g_directional: Additional metric for toward-singularity movement
        """
        single_input = x.dim() == 1
        if single_input:
            x = x.unsqueeze(0)
        
        device = x.device
        batch_size = x.shape[0]
        
        # Base smooth component
        g_base = F.softplus(self.smooth_net(x).squeeze(-1)) + self.min_metric
        
        # Directional component (only applies to toward-singularity velocity)
        g_directional = torch.zeros(batch_size, device=device)
        
        for sing in self.singularities:
            center = sing['center'].to(device)
            radius = sing['radius']
            strength = sing['strength']
            power = sing['power']
            
            # Vector from x toward singularity center
            to_center = center.unsqueeze(0) - x  # (batch, dim)
            dist = torch.norm(to_center, dim=-1)  # (batch,)
            
            # Unit vector toward singularity
            n_hat = to_center / (dist.unsqueeze(-1) + 1e-8)  # (batch, dim)
            
            # Distance to event horizon
            safe_dist = torch.clamp(dist - radius, min=1e-3)
            
            # Singularity contribution (saturating)
            contribution = strength / (safe_dist ** power + 1e-6)
            contribution = torch.clamp(contribution, max=self.max_contribution)
            
            g_directional = g_directional + contribution
        
        if single_input:
            return g_base.squeeze(0), g_directional.squeeze(0)
        
        return g_base, g_directional
    
    def forward(
        self,
        x: torch.Tensor,
        v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute effective metric for state x and optional velocity v.
        
        If v is provided, uses anisotropic scaling (toward-direction only).
        If v is None, returns base metric (for backwards compatibility).
        """
        g_base, g_dir = self.compute_directional_metric(x, v)
        
        if v is None:
            # No velocity provided - return isotropic metric
            return g_base + g_dir
        
        # Compute toward-component of velocity
        device = x.device
        if x.dim() == 1:
            x = x.unsqueeze(0)
            v = v.unsqueeze(0)
        
        batch_size = x.shape[0]
        toward_ratio = torch.zeros(batch_size, device=device)
        
        for sing in self.singularities:
            center = sing['center'].to(device)
            to_center = center.unsqueeze(0) - x
            dist = torch.norm(to_center, dim=-1)
            n_hat = to_center / (dist.unsqueeze(-1) + 1e-8)
            
            # Velocity component toward singularity (positive = approaching)
            v_toward = torch.sum(v * n_hat, dim=-1)
            v_toward_positive = torch.clamp(v_toward, min=0)  # Only penalize approach
            v_norm = torch.norm(v, dim=-1) + 1e-8
            
            # Ratio of toward-velocity to total velocity
            toward_ratio = torch.max(toward_ratio, v_toward_positive / v_norm)
        
        # Effective metric: base + directional * toward_ratio²
        # This makes escape nearly free, approach very expensive
        g_effective = g_base + g_dir * (toward_ratio ** 2)
        
        return g_effective
    
    def compute_escape_advantage_scale(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        soft_threshold: float = 1.5,
        hard_threshold: float = 10.0,
    ) -> torch.Tensor:
        """
        Compute advantage scaling that preserves escape signals.
        
        Key insight: For escape velocities (moving away from black holes),
        we should NOT scale down the advantage, allowing the agent to learn
        escape behaviors even when very close to singularities.
        
        Returns scale in [min_scale, 1.0] where:
        - 1.0 for escape velocities (preserves full learning signal)
        - Reduced for approach velocities (dampens dangerous moves)
        """
        device = x.device
        if x.dim() == 1:
            x = x.unsqueeze(0)
            v = v.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        g_base, g_dir = self.compute_directional_metric(x, v)
        
        # Compute escape factor for each sample
        escape_factors = torch.ones(batch_size, device=device)
        
        for sing in self.singularities:
            center = sing['center'].to(device)
            to_center = center.unsqueeze(0) - x
            dist = torch.norm(to_center, dim=-1)
            n_hat = to_center / (dist.unsqueeze(-1) + 1e-8)
            
            # Velocity toward singularity (positive = approaching, negative = escaping)
            v_toward = torch.sum(v * n_hat, dim=-1)
            
            # Escape factor: 1.0 if escaping, reduced if approaching
            # Smooth transition using sigmoid
            escape_factor = torch.sigmoid(-v_toward * 5)  # High when escaping
            escape_factors = escape_factors * escape_factor
        
        # Compute base scaling from isotropic metric
        g_total = g_base + g_dir
        
        # Adaptive threshold scaling
        scale = torch.ones_like(g_total)
        
        safe_mask = g_total <= soft_threshold
        danger_mask = g_total >= hard_threshold
        trans_mask = ~safe_mask & ~danger_mask
        
        scale[safe_mask] = 1.0
        min_scale = 1.0 / np.sqrt(hard_threshold)
        scale[danger_mask] = min_scale
        
        if trans_mask.any():
            t = (g_total[trans_mask] - soft_threshold) / (hard_threshold - soft_threshold)
            scale[trans_mask] = 1.0 - t * (1.0 - min_scale)
        
        # Blend: for escaping, use full scale (1.0); for approaching, use computed scale
        final_scale = escape_factors * 1.0 + (1 - escape_factors) * scale
        
        return final_scale
    
    def distance_to_nearest_singularity(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distance to nearest black hole horizon."""
        if len(self.singularities) == 0:
            batch_size = x.shape[0] if x.dim() > 1 else 1
            return (
                torch.full((batch_size,), float('inf'), device=x.device),
                torch.zeros(batch_size, dtype=torch.long, device=x.device)
            )
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        distances = []
        for sing in self.singularities:
            center = sing['center'].to(x.device)
            dist = torch.norm(x - center.unsqueeze(0), dim=-1) - sing['radius']
            distances.append(dist)
        
        distances = torch.stack(distances, dim=1)
        min_distances, nearest_idx = distances.min(dim=1)
        
        return min_distances, nearest_idx
    
    def is_safe(self, x: torch.Tensor, threshold: float = 100.0) -> torch.Tensor:
        """Check if points are in safe regions."""
        g_base, g_dir = self.compute_directional_metric(x, None)
        return (g_base + g_dir) < threshold


class SoftSingularityMetric(nn.Module):
    """
    Metric model with soft (saturating) ISOTROPIC singularities.
    
    Unlike the standard metric model that returns infinity inside black holes,
    this model uses a saturating function that allows gradients to flow while
    still creating strong barriers.
    
    g(x) = g_base + Σᵢ min(strength_i / (dist - radius)^α, g_max)
    
    Note: This is the isotropic version. For directional singularities that
    preserve escape routes, use AnisotropicSingularityMetric.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        min_metric: float = 1.0,
        max_singularity_contribution: float = 1e4,
        singularity_power: float = 1.5,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.min_metric = min_metric
        self.max_contribution = max_singularity_contribution
        self.default_power = singularity_power
        
        # Smooth learned component
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.smooth_net = nn.Sequential(*layers)
        
        # Singularity storage
        self.singularities: List[Dict] = []
        self._centers: List[torch.Tensor] = []
    
    def add_singularity(
        self,
        center: Union[np.ndarray, torch.Tensor],
        radius: float,
        strength: float,
        power: Optional[float] = None,
        trainable: bool = False,
    ) -> int:
        """Add a soft singularity (black hole with saturation)."""
        if isinstance(center, np.ndarray):
            center = torch.tensor(center, dtype=torch.float32)
        
        power = power if power is not None else self.default_power
        
        self.singularities.append({
            'center': center,
            'radius': radius,
            'strength': strength,
            'power': power,
            'trainable': trainable,
        })
        self._centers.append(center)
        
        return len(self.singularities) - 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute soft metric at points x.
        
        Returns saturated values instead of infinity near singularities.
        """
        single_input = x.dim() == 1
        if single_input:
            x = x.unsqueeze(0)
        
        device = x.device
        
        # Smooth component (always positive via softplus)
        g = F.softplus(self.smooth_net(x).squeeze(-1)) + self.min_metric
        
        # Add soft singularity contributions
        for sing in self.singularities:
            center = sing['center'].to(device)
            radius = sing['radius']
            strength = sing['strength']
            power = sing['power']
            
            # Distance to center
            dist = torch.norm(x - center.unsqueeze(0), dim=-1)
            
            # Distance to event horizon (clamped to small positive)
            safe_dist = torch.clamp(dist - radius, min=1e-3)
            
            # Saturating contribution
            contribution = strength / (safe_dist ** power + 1e-6)
            contribution = torch.clamp(contribution, max=self.max_contribution)
            
            g = g + contribution
        
        if single_input:
            return g.squeeze(0)
        return g
    
    def distance_to_nearest_singularity(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distance to nearest black hole horizon."""
        if len(self.singularities) == 0:
            batch_size = x.shape[0] if x.dim() > 1 else 1
            return (
                torch.full((batch_size,), float('inf'), device=x.device),
                torch.zeros(batch_size, dtype=torch.long, device=x.device)
            )
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        distances = []
        for sing in self.singularities:
            center = sing['center'].to(x.device)
            dist = torch.norm(x - center.unsqueeze(0), dim=-1) - sing['radius']
            distances.append(dist)
        
        distances = torch.stack(distances, dim=1)
        min_distances, nearest_idx = distances.min(dim=1)
        
        return min_distances, nearest_idx
    
    def is_safe(self, x: torch.Tensor, threshold: float = 100.0) -> torch.Tensor:
        """Check if points are in safe regions."""
        g = self.forward(x)
        return g < threshold
    
    def compute_loss(
        self,
        states: torch.Tensor,
        costs: torch.Tensor,
        cost_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Train metric to correlate with cost."""
        g = self.forward(states)
        
        # Target: high metric where cost is high
        target = torch.where(
            costs > cost_threshold,
            torch.tensor(10.0, device=states.device),
            torch.tensor(1.0, device=states.device),
        )
        
        return F.mse_loss(torch.clamp(g, max=100.0), target)


class EscapeSGPO:
    """
    Sheaf-Geodesic Policy Optimization with escape mechanisms.
    
    Prevents policy freezing near black holes through:
    1. Soft singularities (no infinite values)
    2. Repulsive gradients (active avoidance signal)
    3. Adaptive thresholds (smooth dampening)
    4. Entropy boost (exploration near danger)
    """
    
    def __init__(self, config: Optional[EscapeSGPOConfig] = None):
        self.config = config or EscapeSGPOConfig()
    
    def compute_repulsive_bonus(
        self,
        states: Union[np.ndarray, torch.Tensor],
        metric_model: SoftSingularityMetric,
    ) -> np.ndarray:
        """
        Compute repulsive bonus for moving away from singularities.
        
        r_repel = β / (dist_to_nearest + ε)
        
        This provides a positive learning signal even when the advantage
        would otherwise be dampened near black holes.
        """
        if isinstance(states, np.ndarray):
            states_t = torch.tensor(states, dtype=torch.float32)
        else:
            states_t = states
        
        with torch.no_grad():
            distances, _ = metric_model.distance_to_nearest_singularity(states_t)
            
            # Repulsive bonus inversely proportional to distance
            bonus = self.config.repulsion_strength / (
                distances + self.config.repulsion_epsilon
            )
            
            # Clamp to reasonable range
            bonus = torch.clamp(bonus, max=1.0)
        
        if isinstance(bonus, torch.Tensor):
            return bonus.cpu().numpy()
        return bonus
    
    def compute_adaptive_scale(self, g: float) -> float:
        """
        Compute adaptive scaling factor with smooth transition.
        
        - g <= soft_threshold: scale = 1.0 (no dampening)
        - g >= hard_threshold: scale = 1/√hard_threshold (bounded minimum)
        - Between: smooth interpolation
        """
        soft = self.config.soft_threshold
        hard = self.config.hard_threshold
        
        if g <= soft:
            return 1.0
        elif g >= hard:
            # Bounded minimum scale (not zero!)
            return 1.0 / np.sqrt(hard)
        else:
            # Smooth cosine interpolation
            t = (g - soft) / (hard - soft)
            # Interpolate in log space for smoother behavior
            log_scale_start = 0.0  # log(1.0)
            log_scale_end = -0.5 * np.log(hard)
            log_scale = log_scale_start + t * (log_scale_end - log_scale_start)
            return np.exp(log_scale)
    
    def compute_advantage(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        hodge_critic: Any,
        metric_model: SoftSingularityMetric,
        gamma: float = 0.99,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages with escape mechanisms.
        
        A_escape = scale(G) * (r + γV' - V - ω) + r_repel
        
        Where:
        - scale(G) uses adaptive thresholding (smooth, bounded)
        - r_repel is the repulsive bonus from nearby singularities
        """
        batch_size = len(states)
        
        # Get value estimates
        if hasattr(hodge_critic, 'value'):
            V = hodge_critic.value(states)
            V_next = hodge_critic.value(next_states)
        else:
            V = np.zeros(batch_size)
            V_next = np.zeros(batch_size)
        
        # Get harmonic component
        if hasattr(hodge_critic, 'harmonic'):
            omega = hodge_critic.harmonic(states, actions)
        else:
            omega = np.zeros(batch_size)
        
        # TD error with harmonic correction
        td_error = rewards + gamma * V_next * (1 - dones) - V - omega
        
        # Get metric values
        states_t = torch.tensor(states, dtype=torch.float32)
        with torch.no_grad():
            G = metric_model(states_t)
            if isinstance(G, torch.Tensor):
                G = G.cpu().numpy()
        G = np.asarray(G).flatten()
        
        # Compute repulsive bonus
        repulsive_bonus = self.compute_repulsive_bonus(states, metric_model)
        
        # Compute advantages with adaptive scaling
        advantages = np.zeros(batch_size)
        for i in range(batch_size):
            scale = self.compute_adaptive_scale(G[i])
            advantages[i] = scale * td_error[i] + repulsive_bonus[i]
        
        return advantages, G
    
    def compute_adaptive_entropy_coef(
        self,
        metrics: torch.Tensor,
    ) -> float:
        """
        Compute adaptive entropy coefficient based on proximity to danger.
        
        H_coef = H_base * (1 + boost_factor * proximity_ratio)
        """
        proximity_ratio = (metrics > self.config.soft_threshold).float().mean()
        
        adaptive_coef = self.config.base_entropy_coef * (
            1 + self.config.entropy_boost_factor * proximity_ratio.item()
        )
        
        return adaptive_coef
    
    def compute_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        metrics: torch.Tensor,
    ) -> torch.Tensor:
        """Compute policy loss with PPO clipping in safe regions."""
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clipping only in safe regions
        clipped_ratio = torch.where(
            metrics > self.config.hard_threshold,
            ratio,  # No additional clipping near black holes (already scaled)
            torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
        )
        
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        
        return -torch.min(surr1, surr2).mean()
    
    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute value function loss."""
        if old_values is not None:
            value_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.clip_ratio,
                self.config.clip_ratio
            )
            loss1 = F.mse_loss(values, returns)
            loss2 = F.mse_loss(value_clipped, returns)
            return torch.max(loss1, loss2)
        return F.mse_loss(values, returns)
    
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
        """Compute total loss with adaptive entropy."""
        policy_loss = self.compute_loss(
            old_log_probs, new_log_probs, advantages, metrics
        )
        value_loss = self.compute_value_loss(values, returns, old_values)
        
        # Adaptive entropy coefficient
        entropy_coef = self.compute_adaptive_entropy_coef(metrics)
        entropy_loss = -entropy.mean()
        
        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            + entropy_coef * entropy_loss
        )
        
        # Stats
        with torch.no_grad():
            ratio = torch.exp(new_log_probs - old_log_probs)
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            
            n_near_danger = (metrics > self.config.soft_threshold).sum().item()
            n_in_danger = (metrics > self.config.hard_threshold).sum().item()
        
        loss_dict = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "entropy_coef": entropy_coef,
            "total_loss": total_loss.item(),
            "approx_kl": approx_kl,
            "n_near_danger": n_near_danger,
            "n_in_danger": n_in_danger,
        }
        
        return total_loss, loss_dict


class EscapeSGPOTrainer:
    """
    Full trainer for EscapeSGPO algorithm.
    
    Handles complete training loop with escape mechanisms.
    """
    
    def __init__(
        self,
        model: nn.Module,
        hodge_critic: Any,
        metric_model: SoftSingularityMetric,
        config: Optional[EscapeSGPOConfig] = None,
        device: torch.device = None,
    ):
        self.model = model
        self.hodge_critic = hodge_critic
        self.metric_model = metric_model
        self.config = config or EscapeSGPOConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        self.metric_model.to(self.device)
        
        self.escape_gpo = EscapeSGPO(config=self.config)
        
        self.policy_optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr
        )
        self.metric_optimizer = torch.optim.Adam(
            metric_model.parameters(), lr=self.config.metric_lr
        )
        
        self.update_count = 0
        self.train_stats: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "entropy_coef": [],
            "metric_loss": [],
            "n_black_holes": [],
            "n_near_danger": [],
        }
    
    @classmethod
    def from_black_holes(
        cls,
        model: nn.Module,
        black_holes: List[Dict],
        embed_dim: int,
        config: Optional[EscapeSGPOConfig] = None,
        device: torch.device = None,
    ) -> "EscapeSGPOTrainer":
        """
        Factory: initialize from known black hole locations.
        
        Args:
            model: Policy network
            black_holes: List of dicts with 'center', 'radius', 'strength' keys
            embed_dim: Embedding dimension
            config: Configuration
            device: Torch device
        """
        config = config or EscapeSGPOConfig()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create soft metric model
        metric_model = SoftSingularityMetric(
            input_dim=embed_dim,
            max_singularity_contribution=config.max_singularity_contribution,
            singularity_power=config.singularity_power,
        )
        
        # Add black holes
        for bh in black_holes:
            metric_model.add_singularity(
                center=bh['center'],
                radius=bh['radius'],
                strength=bh.get('strength', 1.0),
                power=bh.get('power', config.singularity_power),
            )
        
        print(f"[EscapeSGPO] Initialized with {len(black_holes)} soft singularities")
        
        # Use local HodgeCriticInterface
        hodge_critic = HodgeCriticInterface(embed_dim=embed_dim, device=device)
        
        return cls(
            model=model,
            hodge_critic=hodge_critic,
            metric_model=metric_model,
            config=config,
            device=device,
        )
    
    def add_black_hole(
        self,
        center: np.ndarray,
        radius: float,
        strength: float = 1.0,
    ):
        """Manually add a black hole."""
        self.metric_model.add_singularity(
            center=center,
            radius=radius,
            strength=strength,
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform single training step."""
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)
        dones = batch["dones"].to(self.device)
        costs = batch.get("costs", torch.zeros_like(rewards)).to(self.device)
        
        # Approximate next states
        next_states = torch.roll(states, -1, dims=0)
        next_states[-1] = states[-1]
        
        # Compute advantages with escape mechanisms
        advantages_np, metrics_np = self.escape_gpo.compute_advantage(
            states.cpu().numpy(),
            actions.cpu().numpy(),
            rewards.cpu().numpy(),
            next_states.cpu().numpy(),
            dones.cpu().numpy(),
            self.hodge_critic,
            self.metric_model,
            gamma=self.config.gamma,
        )
        
        # Get value estimates for returns
        with torch.no_grad():
            _, values = self.model(states)
            if values.dim() > 1:
                values = values.squeeze(-1)
        
        returns = torch.tensor(advantages_np, dtype=torch.float32, device=self.device) + values
        advantages = torch.tensor(advantages_np, dtype=torch.float32, device=self.device)
        metrics_t = torch.tensor(metrics_np, dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        logits, new_values = self.model(states)
        if new_values.dim() > 1:
            new_values = new_values.squeeze(-1)
        
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Compute total loss
        total_loss, loss_dict = self.escape_gpo.compute_total_loss(
            old_log_probs,
            new_log_probs,
            advantages,
            metrics_t,
            new_values,
            returns,
            entropy,
            values,
        )
        
        # Policy update
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.policy_optimizer.step()
        
        # Metric update
        metric_loss = 0.0
        if self.update_count % self.config.metric_update_freq == 0:
            metric_loss = self.metric_model.compute_loss(states, costs)
            self.metric_optimizer.zero_grad()
            metric_loss.backward()
            self.metric_optimizer.step()
            metric_loss = metric_loss.item()
        
        self.update_count += 1
        
        # Compile stats
        stats = {
            "policy_loss": loss_dict["policy_loss"],
            "value_loss": loss_dict["value_loss"],
            "entropy": loss_dict["entropy"],
            "entropy_coef": loss_dict["entropy_coef"],
            "approx_kl": loss_dict["approx_kl"],
            "metric_loss": metric_loss,
            "n_black_holes": len(self.metric_model.singularities),
            "n_near_danger": loss_dict["n_near_danger"],
            "n_in_danger": loss_dict["n_in_danger"],
        }
        
        # Store history
        for key in self.train_stats:
            if key in stats:
                self.train_stats[key].append(stats[key])
        
        return stats
    
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Select action for given state."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            
            logits, value = self.model(state_t)
            if value.dim() > 1:
                value = value.squeeze(-1)
            
            if deterministic:
                action = logits.argmax(dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)[0, action]
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return int(action.item()), float(log_prob.item()), float(value.item())
    
    def evaluate_escape_capability(
        self,
        states: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate how well the policy can escape from near-danger regions.
        """
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            metrics = self.metric_model(states_t).cpu().numpy()
            distances, _ = self.metric_model.distance_to_nearest_singularity(states_t)
            distances = distances.cpu().numpy()
            
            # Compute repulsive bonuses
            repulsive = self.escape_gpo.compute_repulsive_bonus(states, self.metric_model)
        
        return {
            "mean_metric": float(np.mean(metrics)),
            "max_metric": float(np.max(metrics)),
            "min_distance_to_danger": float(np.min(distances)),
            "mean_distance_to_danger": float(np.mean(distances)),
            "mean_repulsive_bonus": float(np.mean(repulsive)),
            "pct_near_danger": float((metrics > self.config.soft_threshold).mean() * 100),
            "pct_in_danger": float((metrics > self.config.hard_threshold).mean() * 100),
        }
    
    def save(self, path: str):
        """Save trainer state."""
        state = {
            "model_state": self.model.state_dict(),
            "metric_model_state": self.metric_model.state_dict(),
            "policy_optimizer_state": self.policy_optimizer.state_dict(),
            "metric_optimizer_state": self.metric_optimizer.state_dict(),
            "config": self.config,
            "update_count": self.update_count,
            "train_stats": self.train_stats,
            "singularities": self.metric_model.singularities,
        }
        torch.save(state, path)
        print(f"Saved EscapeSGPO trainer to {path}")
    
    def load(self, path: str):
        """Load trainer state."""
        state = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(state["model_state"])
        self.metric_model.load_state_dict(state["metric_model_state"], strict=False)
        self.policy_optimizer.load_state_dict(state["policy_optimizer_state"])
        self.metric_optimizer.load_state_dict(state["metric_optimizer_state"])
        self.update_count = state["update_count"]
        self.train_stats = state["train_stats"]
        
        print(f"Loaded EscapeSGPO trainer from {path}")
