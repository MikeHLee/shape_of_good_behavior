"""
Base Classes for SGPO Environment Adapters

Provides abstract interfaces for:
- RiemannianMetricBase: Learns metric tensor g(x) that → ∞ near black holes
- SGPOWrapperBase: Wraps any Gymnasium env with geodesic-aware observations
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class RiemannianMetricBase(nn.Module, ABC):
    """
    Abstract base class for Riemannian metrics in SGPO.
    
    The metric g(x) encodes the geometry of the state space:
    - g(x) = 1: Flat space (no danger)
    - g(x) → ∞: Singularity (black hole / forbidden region)
    
    Geodesics (shortest paths in this geometry) naturally curve
    around high-metric regions, providing safety guarantees.
    """
    
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute metric factor g(x) at state x.
        
        Args:
            x: State tensor of shape (batch, state_dim) or (state_dim,)
            
        Returns:
            Metric factor g(x) of shape (batch, 1) or (1,)
            Higher values indicate more dangerous regions.
        """
        pass
    
    @abstractmethod
    def get_black_hole_centers(self) -> List[np.ndarray]:
        """Return list of black hole center positions."""
        pass
    
    @abstractmethod
    def get_event_horizons(self) -> List[float]:
        """Return list of event horizon radii for each black hole."""
        pass
    
    def geodesic_distance(self, x1: torch.Tensor, x2: torch.Tensor, 
                          n_steps: int = 10) -> torch.Tensor:
        """
        Approximate geodesic distance between two points.
        
        Uses numerical integration along straight line (first-order approx).
        For true geodesics, would need to solve geodesic equation.
        """
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
            
        t = torch.linspace(0, 1, n_steps).to(x1.device)
        path = x1.unsqueeze(1) + t.view(1, -1, 1) * (x2 - x1).unsqueeze(1)
        
        metrics = []
        for i in range(n_steps):
            g = self.forward(path[:, i, :])
            metrics.append(g)
        metrics = torch.stack(metrics, dim=1)
        
        euclidean_dist = torch.norm(x2 - x1, dim=-1, keepdim=True)
        avg_metric = metrics.mean(dim=1)
        
        return euclidean_dist * torch.sqrt(avg_metric)


class SGPOWrapperBase(ABC):
    """
    Abstract base class for SGPO environment wrappers.
    
    Wraps a Gymnasium environment to:
    1. Extract hazard/black hole information
    2. Compute Riemannian metric at each step
    3. Provide geodesic-adjusted advantages for policy optimization
    """
    
    def __init__(self, env, metric: Optional[RiemannianMetricBase] = None):
        self.env = env
        self._metric = metric
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    @property
    def metric(self) -> RiemannianMetricBase:
        if self._metric is None:
            self._metric = self._create_default_metric()
        return self._metric
    
    @metric.setter
    def metric(self, value: RiemannianMetricBase):
        self._metric = value
    
    @abstractmethod
    def _create_default_metric(self) -> RiemannianMetricBase:
        """Create default metric based on environment hazards."""
        pass
    
    @abstractmethod
    def _extract_hazards(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract hazard positions and radii from environment.
        
        Returns:
            Tuple of (hazard_centers, hazard_radii)
        """
        pass
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        info['metric_value'] = self._compute_metric(obs)
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        metric_value = self._compute_metric(obs)
        info['metric_value'] = metric_value
        info['in_black_hole'] = metric_value > 100.0
        
        if 'cost' not in info:
            info['cost'] = 1.0 if info['in_black_hole'] else 0.0
            
        return obs, reward, terminated, truncated, info
    
    def _compute_metric(self, obs: np.ndarray) -> float:
        """Compute metric value at current observation."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            g = self.metric(obs_t[:, :self.metric.state_dim])
            return float(g.squeeze())
    
    def compute_riemannian_advantage(
        self, 
        advantages: torch.Tensor, 
        states: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform standard advantages to Riemannian advantages.
        
        The key insight: divide by sqrt(g) to reduce advantage magnitude
        in dangerous regions, making the policy more conservative there.
        
        Args:
            advantages: Standard TD advantages (batch,)
            states: State observations (batch, state_dim)
            
        Returns:
            Riemannian-adjusted advantages
        """
        with torch.no_grad():
            g = self.metric(states[:, :self.metric.state_dim])
            riemannian_adv = advantages / torch.sqrt(g.squeeze())
        return riemannian_adv
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    def close(self):
        return self.env.close()
    
    def __getattr__(self, name):
        return getattr(self.env, name)
