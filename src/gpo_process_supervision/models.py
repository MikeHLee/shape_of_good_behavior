"""
Neural network models for SGPO with process supervision.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Dict, Optional

from environment import AnomalyType
from feedback import TrajectoryFeedback, AnomalyCandidate


class Actor(nn.Module):
    """Policy network that outputs action distribution."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs: torch.Tensor) -> Normal:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h = self.net(obs)
        mu = torch.tanh(self.mu(h))
        std = torch.exp(self.log_std.clamp(-20, 2))
        return Normal(mu, std)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.forward(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.squeeze(0), log_prob.squeeze(0)


class Critic(nn.Module):
    """Value network."""
    
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs).squeeze(-1)


class StepValueNetwork(nn.Module):
    """Network that predicts per-step value embedding v(s,a)."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class OutcomeNetwork(nn.Module):
    """Network that predicts trajectory-level outcome embedding."""
    
    def __init__(self, obs_dim: int, hidden: int = 64, embed_dim: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(obs_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, embed_dim)
    
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)
        _, (h_n, _) = self.encoder(trajectory)
        return self.head(h_n[-1])


class LearnedAggregator(nn.Module):
    """Learns the aggregation operator ⊕ for compositionality checking."""
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, step_embeddings: torch.Tensor) -> torch.Tensor:
        """Aggregate step embeddings into trajectory embedding.
        
        Args:
            step_embeddings: (batch, seq_len, embed_dim) or (seq_len, embed_dim)
        """
        if step_embeddings.dim() == 2:
            step_embeddings = step_embeddings.unsqueeze(0)
        
        attn_out, _ = self.attention(step_embeddings, step_embeddings, step_embeddings)
        attn_out = self.norm(attn_out + step_embeddings)
        pooled = attn_out.mean(dim=1)
        return self.output(pooled)


class AnomalyAwareRewardLearning(nn.Module):
    """Full anomaly-aware reward learning system.
    
    Implements compositionality residual from Appendix D:
    Δ(τ) = R_outcome(τ) - ⊕ v(s_t, a_t)
    """
    
    def __init__(self, obs_dim: int, act_dim: int, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.outcome_model = OutcomeNetwork(obs_dim, embed_dim=embed_dim)
        self.step_model = StepValueNetwork(obs_dim, act_dim, embed_dim=embed_dim)
        self.aggregator = LearnedAggregator(embed_dim)
        
        # Thresholds for anomaly detection
        self.wormhole_threshold = 1.0
        self.cliff_threshold = -0.5
        self.plateau_min_length = 5
    
    def compute_step_embeddings(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute embeddings for each step."""
        step_embeds = []
        for i in range(len(actions)):
            embed = self.step_model(observations[i], actions[i])
            step_embeds.append(embed)
        return torch.stack(step_embeds).squeeze(1)
    
    def compute_residual(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute compositionality residual Δ(τ)."""
        r_outcome = self.outcome_model(observations)
        
        step_embeds = self.compute_step_embeddings(observations[:-1], actions)
        r_process = self.aggregator(step_embeds)
        
        return r_outcome - r_process
    
    def compute_step_impacts(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute marginal impact of each step (leave-one-out)."""
        step_embeds = self.compute_step_embeddings(observations[:-1], actions)
        full_aggregate = self.aggregator(step_embeds)
        
        impacts = []
        for t in range(len(actions)):
            mask = torch.ones(len(actions), dtype=torch.bool, device=actions.device)
            mask[t] = False
            reduced_embeds = step_embeds[mask]
            
            if len(reduced_embeds) > 0:
                reduced_aggregate = self.aggregator(reduced_embeds)
                impact = torch.norm(full_aggregate - reduced_aggregate)
            else:
                impact = torch.norm(full_aggregate)
            impacts.append(impact)
        
        return torch.stack(impacts)
    
    def detect_anomalies(
        self,
        feedback: TrajectoryFeedback,
        device: torch.device,
    ) -> List[AnomalyCandidate]:
        """Detect anomalies from trajectory feedback."""
        anomalies = []
        
        obs = torch.tensor(feedback.observations, dtype=torch.float32, device=device)
        act = torch.tensor(feedback.actions, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            residual = self.compute_residual(obs, act)
            residual_norm = torch.norm(residual).item()
            residual_mean = residual.mean().item()
        
        # Wormhole: outcome >> process
        if residual_norm > self.wormhole_threshold and residual_mean > 0:
            anomalies.append(AnomalyCandidate(
                anomaly_type=AnomalyType.WORMHOLE,
                trajectory_id=feedback.id,
                confidence=min(1.0, residual_norm / 2.0),
                residual=residual_norm,
                description=f"Outcome exceeds process (residual={residual_norm:.2f})",
            ))
        
        # Cliff: from feedback discontinuities
        for sf in feedback.step_feedback:
            if sf.is_discontinuity and sf.anomaly_flag == AnomalyType.CLIFF:
                anomalies.append(AnomalyCandidate(
                    anomaly_type=AnomalyType.CLIFF,
                    trajectory_id=feedback.id,
                    step_range=(sf.step_idx, sf.step_idx + 1),
                    location=feedback.path[sf.step_idx] if sf.step_idx < len(feedback.path) else None,
                    confidence=0.9,
                    description=f"Cliff at step {sf.step_idx}",
                ))
        
        # Plateau: extended no-progress region
        if feedback.plateau_range is not None:
            start, end = feedback.plateau_range
            if end - start >= self.plateau_min_length:
                anomalies.append(AnomalyCandidate(
                    anomaly_type=AnomalyType.PLATEAU,
                    trajectory_id=feedback.id,
                    step_range=(start, end),
                    confidence=min(1.0, (end - start) / 20.0),
                    description=f"Plateau from step {start} to {end}",
                ))
        
        return anomalies


class AnomalyAwareMetric(nn.Module):
    """Riemannian metric that inflates near dangerous regions.
    
    Implements the conformal factor φ(x) ≈ 1/dist(x, B)^α
    from the safety guarantees.
    """
    
    def __init__(
        self,
        black_holes: List[Dict],
        base_metric: float = 1.0,
        severity: float = 10.0,
        sharpness: float = 2.0,
    ):
        super().__init__()
        
        centers = [bh['center'] for bh in black_holes]
        radii = [bh['radius'] for bh in black_holes]
        
        self.register_buffer(
            'bh_centers',
            torch.tensor(np.array(centers), dtype=torch.float32)
        )
        self.register_buffer(
            'bh_radii',
            torch.tensor(radii, dtype=torch.float32)
        )
        
        self.base_metric = nn.Parameter(torch.tensor(base_metric))
        self.severity = nn.Parameter(torch.tensor(severity))
        self.sharpness = nn.Parameter(torch.tensor(sharpness))
        
        # Dynamically detected cliffs
        self.cliff_centers: List[torch.Tensor] = []
        self.cliff_severities: List[float] = []
    
    def add_cliff(self, center: np.ndarray, severity: float = 5.0):
        """Add a detected cliff to the metric."""
        self.cliff_centers.append(torch.tensor(center, dtype=torch.float32))
        self.cliff_severities.append(severity)
    
    def clear_cliffs(self):
        """Clear all detected cliffs."""
        self.cliff_centers = []
        self.cliff_severities = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor g(x) at position x."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        pos = x[:, :2]
        batch_size = pos.shape[0]
        device = x.device
        
        # Base metric
        g = torch.ones(batch_size, 1, device=device) * F.softplus(self.base_metric)
        
        # Black hole contributions (Schwarzschild-like)
        for i in range(len(self.bh_radii)):
            center = self.bh_centers[i].to(device)
            radius = self.bh_radii[i]
            
            dist = torch.norm(pos - center.unsqueeze(0), dim=1, keepdim=True)
            event_horizon = radius * 0.8
            margin = torch.clamp(dist - event_horizon, min=0.01)
            
            schwarzschild = F.softplus(self.severity) / (margin ** F.softplus(self.sharpness))
            g = g + schwarzschild
        
        # Cliff contributions
        for center, sev in zip(self.cliff_centers, self.cliff_severities):
            center = center.to(device)
            dist = torch.norm(pos - center.unsqueeze(0), dim=1, keepdim=True)
            cliff_contribution = sev / (dist + 0.1) ** 2
            g = g + cliff_contribution
        
        return g
    
    def get_metric_field(
        self,
        x_range: tuple = (-0.5, 3.0),
        y_range: tuple = (-0.5, 3.0),
        resolution: int = 50,
        device: torch.device = None,
    ) -> tuple:
        """Compute metric field over a grid for visualization."""
        if device is None:
            device = self.bh_centers.device
        
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(resolution):
            for j in range(resolution):
                pos = torch.tensor(
                    [X[i, j], Y[i, j], 0, 0],
                    dtype=torch.float32,
                    device=device
                )
                with torch.no_grad():
                    Z[i, j] = np.log(self.forward(pos).item() + 1)
        
        return X, Y, Z
