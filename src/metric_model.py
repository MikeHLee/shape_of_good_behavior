"""
Pre-Initialized Metric Model for SGPO

Riemannian metric model with pre-initialized singularities from CPO constraints.

The metric is a sum of:
1. Learned smooth component (neural network)
2. Fixed singularities from CPO constraints (black holes)

Mathematical Foundation:
g(x) = g_smooth(x) + Σᵢ strength_i / dist(x, cᵢ)^αᵢ

Where:
- g_smooth is a learned positive definite metric (neural network with softplus output)
- cᵢ are black hole centers (from CPO constraints)
- αᵢ are singularity powers (typically 2)

This provides:
- Hard safety guarantees: geodesics cannot cross singularities (infinite distance)
- Soft learning: smooth component can adapt to reward landscape
- Fast initialization: known dangers are pre-encoded
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Singularity:
    """A singularity in the Riemannian metric (black hole)."""
    center: torch.Tensor    # Center coordinates
    radius: float           # Event horizon radius
    strength: float         # Scaling factor for metric contribution
    power: float = 2.0      # Exponent in 1/dist^power
    trainable: bool = False # Whether center/strength can be learned
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "center": self.center.cpu().numpy().tolist(),
            "radius": self.radius,
            "strength": self.strength,
            "power": self.power,
            "trainable": self.trainable,
        }
    
    @classmethod
    def from_dict(cls, d: Dict, device: torch.device = None) -> "Singularity":
        """Create from dictionary."""
        center = torch.tensor(d["center"], dtype=torch.float32)
        if device is not None:
            center = center.to(device)
        return cls(
            center=center,
            radius=d["radius"],
            strength=d["strength"],
            power=d.get("power", 2.0),
            trainable=d.get("trainable", False),
        )


class SmoothMetricNetwork(nn.Module):
    """
    Neural network for learning smooth component of metric.
    
    Outputs a positive scalar representing local metric scaling.
    Uses softplus to ensure positivity.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        min_metric: float = 1.0,
    ):
        """
        Initialize smooth metric network.
        
        Args:
            input_dim: Dimension of state embeddings
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            min_metric: Minimum metric value (baseline)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.min_metric = min_metric
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Softplus())  # Ensure positive output
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth metric component.
        
        Args:
            x: State embeddings (batch_size, input_dim)
            
        Returns:
            Metric values (batch_size,) with minimum value = min_metric
        """
        raw = self.network(x)
        return raw.squeeze(-1) + self.min_metric


class PreInitializedMetricModel(nn.Module):
    """
    Riemannian metric model with pre-initialized singularities.
    
    The metric is a sum of:
    1. Learned smooth component (neural network)
    2. Fixed singularities from CPO constraints
    
    g(x) = g_smooth(x) + Σᵢ strength_i / dist(x, cᵢ)^αᵢ
    
    Usage:
        model = PreInitializedMetricModel(input_dim=384)
        model.add_singularity(center, radius, strength)
        
        metric_values = model(states)  # Forward pass
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        min_metric: float = 1.0,
        singularity_influence_radius: float = 3.0,
    ):
        """
        Initialize pre-initialized metric model.
        
        Args:
            input_dim: Dimension of state embeddings
            hidden_dim: Hidden dimension for smooth network
            num_layers: Number of hidden layers
            min_metric: Minimum metric value (baseline)
            singularity_influence_radius: Singularities only contribute within
                                          this multiple of their radius
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.min_metric = min_metric
        self.singularity_influence = singularity_influence_radius
        
        # Learned smooth component
        self.smooth_net = SmoothMetricNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            min_metric=min_metric,
        )
        
        # Pre-initialized singularities (from CPO)
        self.singularities: List[Singularity] = []
        
        # For trainable singularities
        self._trainable_centers = nn.ParameterList()
        self._trainable_strengths = nn.ParameterList()
    
    def add_singularity(
        self,
        center: Union[np.ndarray, torch.Tensor, List],
        radius: float,
        strength: float,
        power: float = 2.0,
        trainable: bool = False,
    ) -> int:
        """
        Add a pre-known singularity (black hole).
        
        Args:
            center: Center coordinates of the singularity
            radius: Event horizon radius
            strength: Metric contribution scaling
            power: Exponent in 1/dist^power
            trainable: Whether this singularity can be fine-tuned
            
        Returns:
            Index of the added singularity
        """
        # Convert to tensor
        if isinstance(center, np.ndarray):
            center_tensor = torch.tensor(center, dtype=torch.float32)
        elif isinstance(center, list):
            center_tensor = torch.tensor(center, dtype=torch.float32)
        else:
            center_tensor = center.float()
        
        if trainable:
            # Make center and strength learnable
            center_param = nn.Parameter(center_tensor)
            strength_param = nn.Parameter(torch.tensor([strength]))
            self._trainable_centers.append(center_param)
            self._trainable_strengths.append(strength_param)
            
            self.singularities.append(Singularity(
                center=center_param,
                radius=radius,
                strength=strength,  # Will use strength_param in forward
                power=power,
                trainable=True,
            ))
        else:
            self.register_buffer(
                f"singularity_center_{len(self.singularities)}",
                center_tensor
            )
            self.singularities.append(Singularity(
                center=center_tensor,
                radius=radius,
                strength=strength,
                power=power,
                trainable=False,
            ))
        
        return len(self.singularities) - 1
    
    def remove_singularity(self, index: int) -> bool:
        """
        Remove a singularity by index.
        
        Args:
            index: Index of singularity to remove
            
        Returns:
            True if removed, False if index invalid
        """
        if 0 <= index < len(self.singularities):
            sing = self.singularities.pop(index)
            if sing.trainable:
                # Remove from parameter lists (complex, avoid if possible)
                pass
            return True
        return False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute metric at points x.
        
        g(x) = g_smooth(x) + Σᵢ strength_i / dist(x, cᵢ)^αᵢ
        
        Args:
            x: State embeddings (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Metric values (batch_size,) or scalar
        """
        # Handle single point
        single_input = x.dim() == 1
        if single_input:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        device = x.device
        
        # Smooth component
        g = self.smooth_net(x)  # (batch_size,)
        
        # Add singularity contributions
        for i, sing in enumerate(self.singularities):
            center = sing.center.to(device)
            
            # Distance from each point to singularity center
            dist = torch.norm(x - center.unsqueeze(0), dim=-1)  # (batch_size,)
            
            # Get actual strength (may be parameter for trainable)
            if sing.trainable and i < len(self._trainable_strengths):
                strength = self._trainable_strengths[i].abs()  # Ensure positive
            else:
                strength = sing.strength
            
            # Singularity contribution: strength / dist^power
            # Clamp dist to avoid division by zero
            safe_dist = torch.clamp(dist - sing.radius, min=1e-6)
            
            # Only contribute near the singularity (within influence radius)
            influence_radius = self.singularity_influence * sing.radius
            mask = (dist < influence_radius).float()
            
            contribution = mask * strength / (safe_dist ** sing.power)
            
            # Handle points inside event horizon (infinite metric)
            inside = dist < sing.radius
            contribution = torch.where(
                inside,
                torch.tensor(float('inf'), device=device),
                contribution
            )
            
            g = g + contribution
        
        if single_input:
            return g.squeeze(0)
        return g
    
    def compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of metric ∇g(x).
        
        Useful for geodesic computation and natural gradients.
        
        Args:
            x: State embeddings (batch_size, input_dim)
            
        Returns:
            Metric gradients (batch_size, input_dim)
        """
        x = x.requires_grad_(True)
        g = self.forward(x)
        
        # Compute gradient
        grad = torch.autograd.grad(
            g.sum(), x,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return grad
    
    def geodesic_distance_approx(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Approximate geodesic distance between two points.
        
        Uses trapezoidal integration along straight line path.
        True geodesic would curve around high-metric regions.
        
        Args:
            x1: Start point (input_dim,)
            x2: End point (input_dim,)
            num_steps: Integration steps
            
        Returns:
            Approximate geodesic distance (scalar)
        """
        # Linear interpolation
        t = torch.linspace(0, 1, num_steps, device=x1.device)
        path = x1.unsqueeze(0) + t.unsqueeze(1) * (x2 - x1).unsqueeze(0)
        
        # Metric along path
        g_path = self.forward(path)  # (num_steps,)
        
        # Path length element
        dl = torch.norm(x2 - x1) / num_steps
        
        # Integrate sqrt(g) * dl using trapezoidal rule
        sqrt_g = torch.sqrt(g_path)
        distance = dl * (sqrt_g[:-1] + sqrt_g[1:]).sum() / 2
        
        return distance
    
    def is_safe(self, x: torch.Tensor, threshold: float = 100.0) -> torch.Tensor:
        """
        Check if points are in safe regions.
        
        Args:
            x: State embeddings (batch_size, input_dim)
            threshold: Metric above this is considered unsafe
            
        Returns:
            Boolean tensor (batch_size,) - True if safe
        """
        g = self.forward(x)
        return g < threshold
    
    def distance_to_nearest_singularity(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distance to nearest singularity event horizon.
        
        Args:
            x: State embeddings (batch_size, input_dim)
            
        Returns:
            min_distances: Distance to nearest horizon (batch_size,)
            nearest_idx: Index of nearest singularity (batch_size,)
        """
        if len(self.singularities) == 0:
            return (
                torch.full((x.shape[0],), float('inf'), device=x.device),
                torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            )
        
        distances = []
        for sing in self.singularities:
            center = sing.center.to(x.device)
            dist = torch.norm(x - center.unsqueeze(0), dim=-1) - sing.radius
            distances.append(dist)
        
        distances = torch.stack(distances, dim=1)  # (batch_size, n_singularities)
        min_distances, nearest_idx = distances.min(dim=1)
        
        return min_distances, nearest_idx
    
    def compute_loss(
        self,
        states: torch.Tensor,
        costs: torch.Tensor,
        cost_threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute loss for training the smooth metric component.
        
        The metric should be high where costs are high.
        
        Args:
            states: State embeddings (batch_size, input_dim)
            costs: Cost values (batch_size,)
            cost_threshold: Costs above this should have high metric
            
        Returns:
            Loss scalar
        """
        g = self.forward(states)
        
        # Target: high metric for high cost
        # Use soft target based on cost
        target_g = 1.0 + 10.0 * F.relu(costs - cost_threshold)
        
        # MSE loss
        loss = F.mse_loss(g, target_g)
        
        # Regularization: smooth metric shouldn't be too high in safe regions
        safe_mask = costs < cost_threshold
        if safe_mask.any():
            safe_penalty = F.relu(g[safe_mask] - 2.0).mean()
            loss = loss + 0.1 * safe_penalty
        
        return loss
    
    def save(self, path: str):
        """Save model state including singularities."""
        state = {
            "model_state": self.state_dict(),
            "input_dim": self.input_dim,
            "min_metric": self.min_metric,
            "singularity_influence": self.singularity_influence,
            "singularities": [s.to_dict() for s in self.singularities],
        }
        torch.save(state, path)
    
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "PreInitializedMetricModel":
        """Load model from saved state."""
        state = torch.load(path, map_location=device)
        
        model = cls(
            input_dim=state["input_dim"],
            min_metric=state.get("min_metric", 1.0),
            singularity_influence_radius=state.get("singularity_influence", 3.0),
        )
        
        # Restore singularities
        for s_dict in state.get("singularities", []):
            sing = Singularity.from_dict(s_dict, device)
            model.singularities.append(sing)
            if not sing.trainable:
                model.register_buffer(
                    f"singularity_center_{len(model.singularities)-1}",
                    sing.center
                )
        
        # Load state dict (may need to be careful about buffer names)
        try:
            model.load_state_dict(state["model_state"], strict=False)
        except:
            pass  # Singularity buffers may have different names
        
        if device is not None:
            model.to(device)
        
        return model
    
    def __repr__(self) -> str:
        n_sing = len(self.singularities)
        n_train = sum(1 for s in self.singularities if s.trainable)
        return (
            f"PreInitializedMetricModel("
            f"input_dim={self.input_dim}, "
            f"singularities={n_sing} ({n_train} trainable))"
        )


class AdaptiveMetricModel(PreInitializedMetricModel):
    """
    Extension that can learn new singularities during training.
    
    In addition to pre-initialized black holes, this model can
    discover new dangerous regions and add singularities dynamically.
    """
    
    def __init__(
        self,
        input_dim: int,
        max_singularities: int = 20,
        discovery_threshold: float = 0.9,
        **kwargs,
    ):
        """
        Initialize adaptive metric model.
        
        Args:
            input_dim: State embedding dimension
            max_singularities: Maximum number of singularities to track
            discovery_threshold: Cost threshold for discovering new black holes
            **kwargs: Additional args for PreInitializedMetricModel
        """
        super().__init__(input_dim, **kwargs)
        
        self.max_singularities = max_singularities
        self.discovery_threshold = discovery_threshold
        
        # Track candidate singularities
        self.candidate_centers: List[np.ndarray] = []
        self.candidate_counts: List[int] = []
        self.discovery_min_count = 5  # Minimum observations before adding
    
    def update_candidates(
        self,
        states: np.ndarray,
        costs: np.ndarray,
        cluster_radius: float = 0.3,
    ):
        """
        Update candidate singularities based on high-cost observations.
        
        Args:
            states: Observed state embeddings
            costs: Observed costs
            cluster_radius: Radius for clustering candidates
        """
        high_cost_mask = costs > self.discovery_threshold
        if not high_cost_mask.any():
            return
        
        dangerous_states = states[high_cost_mask]
        
        for state in dangerous_states:
            # Check if near existing candidate
            added_to_existing = False
            for i, center in enumerate(self.candidate_centers):
                if np.linalg.norm(state - center) < cluster_radius:
                    # Update center as running average
                    n = self.candidate_counts[i]
                    self.candidate_centers[i] = (n * center + state) / (n + 1)
                    self.candidate_counts[i] += 1
                    added_to_existing = True
                    break
            
            if not added_to_existing:
                self.candidate_centers.append(state.copy())
                self.candidate_counts.append(1)
        
        # Promote candidates with enough observations
        self._promote_candidates()
    
    def _promote_candidates(self):
        """Promote well-observed candidates to full singularities."""
        if len(self.singularities) >= self.max_singularities:
            return
        
        promoted = []
        for i, (center, count) in enumerate(zip(self.candidate_centers, self.candidate_counts)):
            if count >= self.discovery_min_count:
                # Check not too close to existing singularity
                too_close = False
                for sing in self.singularities:
                    if np.linalg.norm(center - sing.center.cpu().numpy()) < 0.5:
                        too_close = True
                        break
                
                if not too_close:
                    self.add_singularity(
                        center=center,
                        radius=0.2,
                        strength=1.0,
                        trainable=True,  # Learned singularities are trainable
                    )
                    promoted.append(i)
        
        # Remove promoted candidates
        for i in sorted(promoted, reverse=True):
            self.candidate_centers.pop(i)
            self.candidate_counts.pop(i)
