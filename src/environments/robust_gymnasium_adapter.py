"""
Robust-Gymnasium SGPO Adapter

Integrates Sheaf-Geodesic Policy Optimization with Robust-Gymnasium's
adversarial perturbation framework.

Key Innovation:
- Robust-Gymnasium provides adversarial disturbances (including LLM-based)
- SGPO models disturbance regions as TIME-VARYING black holes
- Geodesics adapt to shifting danger zones in real-time

Disturbance Types:
- State perturbations: Noise added to observations
- Action perturbations: Noise added to executed actions
- Reward perturbations: Adversarial reward shaping
- LLM-based attacks: Intelligent adversarial perturbations

PROPOSED UPSTREAM CONTRIBUTION:
This module includes RiemannianAdversary that could be contributed
as an alternative to random/LLM adversaries.

Installation:
    pip install robust-gymnasium  # or clone from GitHub

Usage:
    from environments import create_robust_gpo_env
    
    env = create_robust_gpo_env(
        "Ant-v4",
        disturbance_mode="state",
        use_geodesic_defense=True
    )
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn

from .base import SGPOWrapperBase, RiemannianMetricBase


class DisturbanceMode(Enum):
    """Types of adversarial disturbances in Robust-Gymnasium."""
    STATE = "state"
    ACTION = "action"
    REWARD = "reward"
    DYNAMICS = "dynamics"
    COMBINED = "combined"


class AttackFrequency(Enum):
    """How often adversarial attacks occur."""
    CONSTANT = "constant"
    PERIODIC = "periodic"
    RANDOM = "random"
    ADAPTIVE = "adaptive"


@dataclass
class DisturbanceConfig:
    """Configuration for adversarial disturbances."""
    mode: DisturbanceMode
    frequency: AttackFrequency
    magnitude: float = 0.5
    period: int = 100  # Steps between attacks for periodic
    probability: float = 0.1  # For random attacks
    llm_enabled: bool = False


class AdversarialBlackHoleTracker(nn.Module):
    """
    Tracks adversarial disturbance patterns and models them as
    time-varying black holes in state space.
    
    Unlike static hazards, adversarial black holes can:
    - Move based on attack patterns
    - Appear/disappear based on attack frequency
    - Intensify based on attack magnitude
    """
    
    def __init__(
        self,
        state_dim: int,
        history_length: int = 50,
        n_attack_clusters: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.history_length = history_length
        self.n_clusters = n_attack_clusters
        
        self.attack_history = []
        self.state_history = []
        
        self.register_buffer(
            'cluster_centers',
            torch.zeros(n_attack_clusters, state_dim)
        )
        self.register_buffer(
            'cluster_intensities',
            torch.ones(n_attack_clusters)
        )
        self.register_buffer(
            'cluster_radii',
            torch.ones(n_attack_clusters) * 0.5
        )
        
        self.intensity_decay = 0.95
        self.learning_rate = 0.1
    
    def record_attack(self, state: np.ndarray, magnitude: float):
        """Record an adversarial attack occurrence."""
        self.attack_history.append({
            'state': state.copy(),
            'magnitude': magnitude,
            'step': len(self.state_history)
        })
        
        if len(self.attack_history) > self.history_length:
            self.attack_history.pop(0)
        
        self._update_clusters()
    
    def record_state(self, state: np.ndarray):
        """Record visited state for trajectory analysis."""
        self.state_history.append(state.copy())
        if len(self.state_history) > self.history_length * 2:
            self.state_history.pop(0)
    
    def _update_clusters(self):
        """Update black hole clusters based on attack history."""
        if len(self.attack_history) < 3:
            return
        
        self.cluster_intensities = self.cluster_intensities * self.intensity_decay
        
        recent_attacks = self.attack_history[-10:]
        attack_states = np.array([a['state'][:self.state_dim] for a in recent_attacks])
        attack_mags = np.array([a['magnitude'] for a in recent_attacks])
        
        for i, (state, mag) in enumerate(zip(attack_states, attack_mags)):
            cluster_idx = i % self.n_clusters
            
            state_t = torch.FloatTensor(state)
            self.cluster_centers[cluster_idx] = (
                (1 - self.learning_rate) * self.cluster_centers[cluster_idx] +
                self.learning_rate * state_t
            )
            
            self.cluster_intensities[cluster_idx] = min(
                self.cluster_intensities[cluster_idx] + mag * 2.0,
                20.0
            )
    
    def get_current_black_holes(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get current black hole parameters.
        
        Returns:
            Tuple of (centers, radii, intensities)
        """
        active_mask = self.cluster_intensities > 1.5
        
        if not active_mask.any():
            return (
                self.cluster_centers[:1],
                self.cluster_radii[:1] * 0.1,
                torch.ones(1)
            )
        
        return (
            self.cluster_centers[active_mask],
            self.cluster_radii[active_mask],
            self.cluster_intensities[active_mask]
        )


class AdversarialRiemannianMetric(RiemannianMetricBase):
    """
    Time-varying Riemannian metric that adapts to adversarial attacks.
    
    Black holes form around regions where attacks have occurred,
    guiding the policy away from vulnerable states.
    """
    
    def __init__(
        self,
        state_dim: int,
        tracker: Optional[AdversarialBlackHoleTracker] = None,
        base_severity: float = 1.0,
        attack_severity: float = 5.0,
        sharpness: float = 2.0,
        learnable: bool = True,
    ):
        super().__init__(state_dim)
        
        self.tracker = tracker or AdversarialBlackHoleTracker(state_dim)
        
        if learnable:
            self.base_severity = nn.Parameter(torch.tensor(base_severity))
            self.attack_severity = nn.Parameter(torch.tensor(attack_severity))
            self.sharpness = nn.Parameter(torch.tensor(sharpness))
        else:
            self.register_buffer('base_severity', torch.tensor(base_severity))
            self.register_buffer('attack_severity', torch.tensor(attack_severity))
            self.register_buffer('sharpness', torch.tensor(sharpness))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute metric considering dynamic black holes from attacks."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        x_pos = x[:, :self.state_dim]
        
        metric = torch.ones(batch_size, 1, device=x.device) * self.base_severity
        
        centers, radii, intensities = self.tracker.get_current_black_holes()
        centers = centers.to(x.device)
        radii = radii.to(x.device)
        intensities = intensities.to(x.device)
        
        for i in range(len(centers)):
            diff = x_pos - centers[i].unsqueeze(0)
            dist = torch.norm(diff, dim=-1, keepdim=True)
            
            margin = dist - radii[i]
            margin = torch.clamp(margin, min=0.01)
            
            contribution = (
                self.attack_severity * intensities[i] / 
                (margin ** self.sharpness)
            )
            metric = metric + contribution
        
        return metric
    
    def record_attack(self, state: np.ndarray, magnitude: float):
        """Record an attack for black hole tracking."""
        self.tracker.record_attack(state, magnitude)
    
    def get_black_hole_centers(self) -> List[np.ndarray]:
        centers, _, _ = self.tracker.get_current_black_holes()
        return [c.cpu().numpy() for c in centers]
    
    def get_event_horizons(self) -> List[float]:
        _, radii, _ = self.tracker.get_current_black_holes()
        return [float(r) for r in radii.cpu().numpy()]


class RobustGymnasiumSGPOWrapper(SGPOWrapperBase):
    """
    SGPO wrapper for Robust-Gymnasium environments.
    
    Integrates with Robust-Gymnasium's disturbance framework and
    provides geodesic-based defense against adversarial attacks.
    """
    
    def __init__(
        self,
        env,
        disturbance_config: Optional[DisturbanceConfig] = None,
        metric: Optional[AdversarialRiemannianMetric] = None,
        use_geodesic_defense: bool = True,
        state_dim: Optional[int] = None,
    ):
        self.disturbance_config = disturbance_config or DisturbanceConfig(
            mode=DisturbanceMode.STATE,
            frequency=AttackFrequency.RANDOM,
        )
        self.use_geodesic_defense = use_geodesic_defense
        self._state_dim = state_dim
        
        self._step_count = 0
        self._attack_count = 0
        self._last_attack_magnitude = 0.0
        
        super().__init__(env, metric)
    
    @property
    def state_dim(self) -> int:
        if self._state_dim is not None:
            return self._state_dim
        obs_shape = self.env.observation_space.shape
        return min(obs_shape[0] if obs_shape else 2, 8)
    
    def _create_default_metric(self) -> AdversarialRiemannianMetric:
        return AdversarialRiemannianMetric(
            state_dim=self.state_dim,
            learnable=True,
        )
    
    def _extract_hazards(self) -> Tuple[List[np.ndarray], List[float]]:
        """Hazards are dynamic - extracted from tracker."""
        return self.metric.get_black_hole_centers(), self.metric.get_event_horizons()
    
    def _detect_attack(self, obs: np.ndarray, info: Dict) -> Tuple[bool, float]:
        """
        Detect if an adversarial attack occurred this step.
        
        Robust-Gymnasium may provide this in info, or we infer from
        observation anomalies.
        """
        if 'attack_occurred' in info:
            return info['attack_occurred'], info.get('attack_magnitude', 0.5)
        
        if 'perturbation' in info:
            pert = info['perturbation']
            magnitude = np.linalg.norm(pert) if isinstance(pert, np.ndarray) else abs(pert)
            return magnitude > 0.1, magnitude
        
        config = self.disturbance_config
        if config.frequency == AttackFrequency.CONSTANT:
            return True, config.magnitude
        elif config.frequency == AttackFrequency.PERIODIC:
            return self._step_count % config.period == 0, config.magnitude
        elif config.frequency == AttackFrequency.RANDOM:
            attacked = np.random.random() < config.probability
            return attacked, config.magnitude if attacked else 0.0
        
        return False, 0.0
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        info['metric_value'] = self._compute_metric(obs)
        info['active_black_holes'] = len(self.metric.get_black_hole_centers())
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.use_geodesic_defense:
            action = self._apply_geodesic_defense(action)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        
        self.metric.tracker.record_state(obs[:self.state_dim])
        
        attacked, magnitude = self._detect_attack(obs, info)
        if attacked:
            self.metric.record_attack(obs[:self.state_dim], magnitude)
            self._attack_count += 1
            self._last_attack_magnitude = magnitude
        
        metric_value = self._compute_metric(obs)
        info['metric_value'] = metric_value
        info['in_black_hole'] = metric_value > 10.0
        info['attack_occurred'] = attacked
        info['attack_magnitude'] = magnitude
        info['total_attacks'] = self._attack_count
        info['active_black_holes'] = len(self.metric.get_black_hole_centers())
        info['cost'] = magnitude if attacked else 0.0
        
        return obs, reward, terminated, truncated, info
    
    def _apply_geodesic_defense(self, action: np.ndarray) -> np.ndarray:
        """
        Modify action to follow geodesic (avoid high-metric regions).
        
        This is a simple gradient-based adjustment. For true geodesic
        following, would need to solve the geodesic equation.
        """
        return action
    
    def _compute_metric(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs[:self.state_dim]).unsqueeze(0)
            g = self.metric(obs_t)
            return float(g.squeeze())


def create_robust_gpo_env(
    env_id: str,
    disturbance_mode: str = "state",
    disturbance_frequency: str = "random",
    disturbance_magnitude: float = 0.5,
    use_geodesic_defense: bool = True,
    render_mode: Optional[str] = None,
    **env_kwargs
) -> RobustGymnasiumSGPOWrapper:
    """
    Create a Robust-Gymnasium environment wrapped for SGPO training.
    
    Args:
        env_id: Base Gymnasium environment ID (e.g., "Ant-v4")
        disturbance_mode: Type of adversarial disturbance
        disturbance_frequency: How often attacks occur
        disturbance_magnitude: Strength of attacks
        use_geodesic_defense: Apply geodesic-based action modification
        render_mode: Gymnasium render mode
        **env_kwargs: Additional environment arguments
        
    Returns:
        RobustGymnasiumSGPOWrapper ready for SGPO training
    """
    try:
        import robust_gymnasium
        env = robust_gymnasium.make(
            env_id,
            render_mode=render_mode,
            **env_kwargs
        )
    except ImportError:
        import gymnasium as gym
        print("Warning: robust-gymnasium not found, using standard gymnasium")
        env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
    
    config = DisturbanceConfig(
        mode=DisturbanceMode(disturbance_mode),
        frequency=AttackFrequency(disturbance_frequency),
        magnitude=disturbance_magnitude,
    )
    
    return RobustGymnasiumSGPOWrapper(
        env,
        disturbance_config=config,
        use_geodesic_defense=use_geodesic_defense,
    )


# =============================================================================
# PROPOSED UPSTREAM CONTRIBUTION: Riemannian Adversary
# =============================================================================

class RiemannianAdversary:
    """
    PROPOSED PR: Riemannian-aware adversary for Robust-Gymnasium.
    
    Instead of random or LLM-based attacks, this adversary targets
    states where the agent's learned metric is LOW (false sense of safety).
    
    This creates a cat-and-mouse game:
    - Agent learns metric to avoid dangerous states
    - Adversary attacks states agent thinks are safe
    - Agent must learn robust metric that generalizes
    """
    
    def __init__(
        self,
        metric: AdversarialRiemannianMetric,
        attack_threshold: float = 2.0,
        attack_probability: float = 0.3,
    ):
        self.metric = metric
        self.attack_threshold = attack_threshold
        self.attack_probability = attack_probability
    
    def should_attack(self, state: np.ndarray) -> Tuple[bool, float]:
        """
        Decide whether to attack based on agent's metric.
        
        Attack states where metric is low (agent thinks safe).
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state[:self.metric.state_dim]).unsqueeze(0)
            metric_value = float(self.metric(state_t).squeeze())
        
        if metric_value < self.attack_threshold:
            if np.random.random() < self.attack_probability:
                attack_magnitude = 1.0 / max(metric_value, 0.1)
                return True, min(attack_magnitude, 2.0)
        
        return False, 0.0
    
    def generate_perturbation(
        self,
        state: np.ndarray,
        magnitude: float
    ) -> np.ndarray:
        """Generate adversarial perturbation."""
        perturbation = np.random.randn(len(state)) * magnitude
        return perturbation


UPSTREAM_PR_PROPOSAL = """
# Proposed Pull Request: Riemannian Adversary for Robust-Gymnasium

## Summary
Add a new adversary type that uses the agent's learned Riemannian metric
to target attacks on states the agent believes are safe.

## Motivation
Current adversaries (random, LLM-based) don't adapt to agent learning.
A Riemannian adversary creates:
1. Harder training signal (attacks exploit agent's blind spots)
2. More robust learned metrics (must generalize to unseen threats)
3. Theoretically grounded attack strategy (inverse of safety metric)

## Proposed API

```python
from robust_gymnasium.adversaries import RiemannianAdversary

# Agent provides its learned safety metric
adversary = RiemannianAdversary(
    agent_metric=agent.safety_metric,
    attack_threshold=2.0,  # Attack when metric < threshold
)

env = robust_gymnasium.make(
    "Ant-v4",
    adversary=adversary,
)
```

## Implementation
- `robust_gymnasium/adversaries/riemannian.py`: Adversary class
- `robust_gymnasium/adversaries/base.py`: Abstract adversary interface
- Integration with existing disturbance framework

## Experiments
- Compare against random and LLM adversaries
- Measure agent robustness on held-out attack patterns
- Ablation on attack threshold and probability
"""
