"""
Learned Implicit Danger Boundaries for SGPO

[UPGRADE: January 2026 - Post-Modal Experiments]
This module was added to replace spherical black holes with learned implicit surfaces.
Key insight: Spherical assumptions fail in high-D (curse of dimensionality).
The neural network learns arbitrary danger region shapes from data.
See Section 3.3 (Learned Implicit Danger Boundaries) and Equation 7 in paper.

Instead of using spherical black holes (which are trivially avoided in high-D),
we learn an implicit surface from data. The danger region is defined by:

    d(x) < 0 → inside dangerous region
    d(x) = 0 → at the "event horizon" (level set)
    d(x) > 0 → in safe region

The metric g(x) = 1 + σ / |d(x)|^α creates an infinite barrier at the boundary,
ensuring agents cannot cross into dangerous territory.

Key Advantages:
1. Learned boundaries adapt to actual danger distribution (not assumed spheres)
2. Level set can represent any shape (non-convex, disjoint regions)
3. Works with any source of danger labels (safety classifiers, Safety-Gym, human feedback)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class BoundaryConfig:
    """Configuration for learned danger boundary."""
    embed_dim: int
    hidden_dim: int = 256
    n_layers: int = 3
    dropout: float = 0.1
    strength: float = 100.0      # σ in g(x) = 1 + σ/|d(x)|^α
    alpha: float = 1.5           # α in the metric formula
    margin: float = 0.1          # Safety margin for boundary
    min_distance: float = 1e-3   # Clamp to prevent numerical overflow


if HAS_TORCH:
    class LearnedDangerBoundary(nn.Module):
        """
        Implicit surface for danger region - NOT a sphere.
        
        d(x) < 0 → inside dangerous region
        d(x) = 0 → at the "event horizon" (level set)
        d(x) > 0 → in safe region
        
        The network learns to output a signed distance-like value,
        negative for dangerous states, positive for safe states.
        """
        
        def __init__(
            self,
            embed_dim: int,
            hidden_dim: int = 256,
            n_layers: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.hidden_dim = hidden_dim
            
            layers = []
            in_dim = embed_dim
            for i in range(n_layers - 1):
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                in_dim = hidden_dim
            
            layers.append(nn.Linear(hidden_dim, 1))
            self.net = nn.Sequential(*layers)
            
            self._init_weights()
        
        def _init_weights(self):
            """Initialize weights for stable training."""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Returns signed distance: negative = inside danger, positive = safe.
            
            Args:
                x: State embeddings (batch_size, embed_dim)
                
            Returns:
                Signed distance values (batch_size, 1)
            """
            return self.net(x)
        
        def is_dangerous(self, x: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
            """
            Check if states are in dangerous region.
            
            Network trained with BCEWithLogitsLoss: positive logits = dangerous.
            
            Args:
                x: State embeddings (batch_size, embed_dim)
                margin: Safety margin (positive = more conservative)
                
            Returns:
                Boolean mask (batch_size,)
            """
            logits = self.forward(x).squeeze(-1)
            return logits > -margin  # Positive logits = dangerous
        
        def metric(
            self,
            x: torch.Tensor,
            strength: float = 100.0,
            alpha: float = 1.5,
            min_distance: float = 1e-3,
        ) -> torch.Tensor:
            """
            Compute Riemannian metric that diverges at the learned boundary.
            
            g(x) = 1 + σ / |d(x)|^α
            
            As d(x) → 0 (approaching boundary), g(x) → ∞
            
            Args:
                x: State embeddings (batch_size, embed_dim)
                strength: σ parameter (controls barrier strength)
                alpha: Power in denominator (≥1 for infinite barrier)
                min_distance: Clamp to prevent numerical overflow
                
            Returns:
                Metric values (batch_size, 1)
            """
            d = self.forward(x)
            safe_d = torch.clamp(d.abs(), min=min_distance)
            return 1.0 + strength / (safe_d ** alpha)
        
        def anisotropic_metric(
            self,
            x: torch.Tensor,
            strength: float = 100.0,
            alpha: float = 1.5,
            min_distance: float = 1e-3,
            anisotropy_ratio: float = 0.1,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute anisotropic metric that preserves escape routes.
            
            The metric is high in the direction toward danger (gradient of d),
            but lower in perpendicular directions, allowing escape.
            
            Args:
                x: State embeddings (batch_size, embed_dim)
                strength: σ parameter
                alpha: Power in denominator
                min_distance: Clamp for numerical stability
                anisotropy_ratio: Ratio of perpendicular to parallel metric
                
            Returns:
                Tuple of (metric_parallel, metric_perpendicular)
            """
            x.requires_grad_(True)
            d = self.forward(x)
            
            grad_d = torch.autograd.grad(
                d.sum(), x, create_graph=True, retain_graph=True
            )[0]
            
            safe_d = torch.clamp(d.abs(), min=min_distance)
            metric_parallel = 1.0 + strength / (safe_d ** alpha)
            metric_perpendicular = 1.0 + anisotropy_ratio * strength / (safe_d ** alpha)
            
            return metric_parallel, metric_perpendicular, grad_d
        
        def geodesic_cost(
            self,
            trajectory: torch.Tensor,
            strength: float = 100.0,
            alpha: float = 1.5,
        ) -> torch.Tensor:
            """
            Compute geodesic cost along a trajectory.
            
            Cost = ∫ √(g(x)) ds where ds is the Euclidean step length.
            
            Args:
                trajectory: Sequence of states (seq_len, embed_dim)
                strength: Metric strength parameter
                alpha: Metric power parameter
                
            Returns:
                Total geodesic cost (scalar)
            """
            if trajectory.dim() == 2:
                trajectory = trajectory.unsqueeze(0)  # Add batch dim
            
            batch_size, seq_len, embed_dim = trajectory.shape
            
            metrics = self.metric(
                trajectory.view(-1, embed_dim),
                strength=strength,
                alpha=alpha,
            ).view(batch_size, seq_len, 1)
            
            steps = trajectory[:, 1:] - trajectory[:, :-1]
            step_lengths = torch.norm(steps, dim=-1, keepdim=True)
            
            avg_metrics = (metrics[:, 1:] + metrics[:, :-1]) / 2
            
            geodesic_costs = (torch.sqrt(avg_metrics) * step_lengths).sum(dim=1)
            
            return geodesic_costs.squeeze()
        
        def train_from_labels(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor,
            n_epochs: int = 100,
            batch_size: int = 64,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            pos_weight: Optional[float] = None,
            verbose: bool = True,
        ) -> Dict[str, List[float]]:
            """
            Train boundary from binary labels.
            
            The network is trained as a binary classifier, then the logits
            are used as signed distance (positive logits → dangerous, d < 0).
            
            Args:
                embeddings: State embeddings (N, embed_dim)
                labels: Binary labels (N,), 1 = dangerous, 0 = safe
                n_epochs: Number of training epochs
                batch_size: Batch size
                lr: Learning rate
                weight_decay: L2 regularization
                pos_weight: Weight for positive class (dangerous)
                verbose: Print training progress
                
            Returns:
                Dictionary with training history
            """
            embeddings = embeddings.float()
            labels = labels.float().view(-1, 1)
            
            dataset = TensorDataset(embeddings, labels)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            if pos_weight is None:
                n_dangerous = labels.sum().item()
                n_safe = len(labels) - n_dangerous
                pos_weight = n_safe / max(n_dangerous, 1)
            
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
            optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
            
            history = {"loss": [], "accuracy": []}
            
            for epoch in range(n_epochs):
                total_loss = 0.0
                correct = 0
                total = 0
                
                for batch_emb, batch_labels in loader:
                    optimizer.zero_grad()
                    
                    logits = self.forward(batch_emb)
                    
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * len(batch_emb)
                    
                    preds = (logits > 0).float()
                    correct += (preds == batch_labels).sum().item()
                    total += len(batch_labels)
                
                scheduler.step()
                
                avg_loss = total_loss / len(embeddings)
                accuracy = correct / total
                history["loss"].append(avg_loss)
                history["accuracy"].append(accuracy)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
            
            return history
        
        def get_signed_distance(self, x: torch.Tensor) -> torch.Tensor:
            """
            Get signed distance for visualization/analysis.
            
            Positive logits from classifier → dangerous → d < 0
            Negative logits from classifier → safe → d > 0
            
            Returns:
                Signed distance (positive = safe, negative = dangerous)
            """
            logits = self.forward(x)
            return -logits.squeeze(-1)


class LearnedDangerBoundaryNumpy:
    """
    NumPy-only version for environments without PyTorch.
    Uses a simpler RBF-based approach.
    """
    
    def __init__(
        self,
        embed_dim: int,
        n_centers: int = 50,
        strength: float = 100.0,
        alpha: float = 1.5,
    ):
        self.embed_dim = embed_dim
        self.n_centers = n_centers
        self.strength = strength
        self.alpha = alpha
        
        self.centers = None
        self.weights = None
        self.bias = 0.0
        self.rbf_scale = 1.0
    
    def _rbf_kernel(self, x: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute RBF kernel values."""
        dists = np.linalg.norm(
            x[:, np.newaxis, :] - centers[np.newaxis, :, :],
            axis=-1
        )
        return np.exp(-dists**2 / (2 * self.rbf_scale**2))
    
    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        regularization: float = 1e-3,
    ):
        """
        Fit the boundary using RBF interpolation.
        
        Args:
            embeddings: State embeddings (N, embed_dim)
            labels: Binary labels (N,), 1 = dangerous, 0 = safe
        """
        n_samples = len(embeddings)
        
        if n_samples <= self.n_centers:
            self.centers = embeddings.copy()
        else:
            indices = np.random.choice(n_samples, self.n_centers, replace=False)
            self.centers = embeddings[indices]
        
        dists = np.linalg.norm(
            self.centers[:, np.newaxis, :] - self.centers[np.newaxis, :, :],
            axis=-1
        )
        self.rbf_scale = np.median(dists[dists > 0])
        
        K = self._rbf_kernel(embeddings, self.centers)
        
        targets = 2 * labels - 1
        
        KtK = K.T @ K + regularization * np.eye(len(self.centers))
        self.weights = np.linalg.solve(KtK, K.T @ targets)
        self.bias = np.mean(targets - K @ self.weights)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute signed distance (positive = dangerous, negative = safe).
        """
        x = np.atleast_2d(x)
        K = self._rbf_kernel(x, self.centers)
        logits = K @ self.weights + self.bias
        return -logits
    
    def is_dangerous(self, x: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Check if states are dangerous."""
        d = self(x)
        return d < margin
    
    def metric(self, x: np.ndarray) -> np.ndarray:
        """Compute metric value at states."""
        d = self(x)
        safe_d = np.maximum(np.abs(d), 1e-3)
        return 1.0 + self.strength / (safe_d ** self.alpha)


class SafetyClassifierBoundary:
    """
    Wrapper to use a pre-trained safety classifier as a danger boundary.
    
    This allows using existing safety classifiers (e.g., from HuggingFace)
    as the oracle for defining dangerous regions.
    """
    
    def __init__(
        self,
        classifier: Callable[[List[str]], List[Dict]],
        encoder: Callable[[List[str]], np.ndarray],
        danger_label: str = "unsafe",
        threshold: float = 0.5,
        strength: float = 100.0,
        alpha: float = 1.5,
    ):
        """
        Args:
            classifier: Function that takes text and returns classification
            encoder: Function that encodes text to embeddings
            danger_label: Label indicating dangerous content
            threshold: Probability threshold for danger classification
            strength: Metric strength parameter
            alpha: Metric power parameter
        """
        self.classifier = classifier
        self.encoder = encoder
        self.danger_label = danger_label
        self.threshold = threshold
        self.strength = strength
        self.alpha = alpha
        
        self.boundary = None
    
    def build_from_texts(
        self,
        texts: List[str],
        embed_dim: int,
        hidden_dim: int = 256,
    ) -> "LearnedDangerBoundary":
        """
        Build a learned boundary from text samples.
        
        Args:
            texts: List of text samples to classify
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension for boundary network
            
        Returns:
            Trained LearnedDangerBoundary
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for LearnedDangerBoundary")
        
        embeddings = self.encoder(texts)
        
        classifications = self.classifier(texts)
        labels = np.array([
            1 if c.get("label") == self.danger_label and c.get("score", 0) >= self.threshold
            else 0
            for c in classifications
        ])
        
        self.boundary = LearnedDangerBoundary(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )
        
        self.boundary.train_from_labels(
            torch.tensor(embeddings),
            torch.tensor(labels),
        )
        
        return self.boundary


class SafetyGymBoundary:
    """
    Build danger boundary from Safety-Gym cost function.
    
    Safety-Gym provides C(s) cost at each state. States with
    cost > threshold are considered inside the danger region.
    """
    
    def __init__(
        self,
        env,
        cost_threshold: float = 0.0,
        n_samples: int = 10000,
        strength: float = 100.0,
        alpha: float = 1.5,
    ):
        """
        Args:
            env: Safety-Gym environment
            cost_threshold: Cost above this is dangerous
            n_samples: Number of states to sample
            strength: Metric strength
            alpha: Metric power
        """
        self.env = env
        self.cost_threshold = cost_threshold
        self.n_samples = n_samples
        self.strength = strength
        self.alpha = alpha
        
        self.boundary = None
    
    def build_boundary(
        self,
        hidden_dim: int = 256,
        n_epochs: int = 100,
        verbose: bool = True,
    ) -> "LearnedDangerBoundary":
        """
        Sample environment and build boundary.
        
        Returns:
            Trained LearnedDangerBoundary
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for LearnedDangerBoundary")
        
        states = []
        costs = []
        
        obs = self.env.reset()
        for _ in range(self.n_samples):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            
            states.append(obs.copy())
            costs.append(info.get("cost", 0.0))
            
            if done:
                obs = self.env.reset()
        
        states = np.array(states)
        labels = (np.array(costs) > self.cost_threshold).astype(np.float32)
        
        embed_dim = states.shape[1]
        self.boundary = LearnedDangerBoundary(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )
        
        self.boundary.train_from_labels(
            torch.tensor(states),
            torch.tensor(labels),
            n_epochs=n_epochs,
            verbose=verbose,
        )
        
        return self.boundary


def create_synthetic_boundary_data(
    embed_dim: int = 768,
    n_samples: int = 5000,
    n_danger_regions: int = 3,
    danger_radius_range: Tuple[float, float] = (0.5, 2.0),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic labeled data with known dangerous regions.
    
    Useful for testing the learned boundary approach without
    needing a real safety classifier or Safety-Gym.
    
    Args:
        embed_dim: Embedding dimension
        n_samples: Number of samples to generate
        n_danger_regions: Number of danger "blobs"
        danger_radius_range: Range of danger region radii
        seed: Random seed
        
    Returns:
        Tuple of (embeddings, labels)
    """
    np.random.seed(seed)
    
    danger_centers = np.random.randn(n_danger_regions, embed_dim) * 2
    danger_radii = np.random.uniform(
        danger_radius_range[0],
        danger_radius_range[1],
        n_danger_regions
    )
    
    embeddings = np.random.randn(n_samples, embed_dim)
    
    labels = np.zeros(n_samples)
    for center, radius in zip(danger_centers, danger_radii):
        dists = np.linalg.norm(embeddings - center, axis=1)
        labels = np.logical_or(labels, dists < radius)
    
    labels = labels.astype(np.float32)
    
    return embeddings, labels, danger_centers, danger_radii


if __name__ == "__main__":
    print("Testing LearnedDangerBoundary...")
    
    embeddings, labels, centers, radii = create_synthetic_boundary_data(
        embed_dim=64,
        n_samples=2000,
        n_danger_regions=3,
    )
    
    print(f"Data: {len(embeddings)} samples, {labels.sum():.0f} dangerous")
    
    if HAS_TORCH:
        boundary = LearnedDangerBoundary(embed_dim=64, hidden_dim=128)
        
        history = boundary.train_from_labels(
            torch.tensor(embeddings),
            torch.tensor(labels),
            n_epochs=50,
            verbose=True,
        )
        
        with torch.no_grad():
            test_emb = torch.tensor(embeddings[:100])
            test_labels = labels[:100]
            
            preds = boundary.is_dangerous(test_emb).numpy()
            accuracy = (preds == test_labels).mean()
            print(f"\nTest accuracy: {accuracy:.4f}")
            
            metrics = boundary.metric(test_emb).numpy()
            print(f"Metric range: [{metrics.min():.2f}, {metrics.max():.2f}]")
            
            traj = torch.tensor(embeddings[:20]).float()
            cost = boundary.geodesic_cost(traj)
            print(f"Geodesic cost of trajectory: {cost.item():.2f}")
    else:
        print("PyTorch not available, testing NumPy version...")
        boundary = LearnedDangerBoundaryNumpy(embed_dim=64)
        boundary.fit(embeddings, labels)
        
        preds = boundary.is_dangerous(embeddings[:100])
        accuracy = (preds == labels[:100]).mean()
        print(f"Test accuracy: {accuracy:.4f}")
    
    print("\nDone!")
