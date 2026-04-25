"""
Safety-Gymnasium SGPO Adapter

Integrates Sheaf-Geodesic Policy Optimization with Safety-Gymnasium environments.
Maps hazards (vases, gremlins, etc.) to Riemannian singularities (black holes).

Key Innovation:
- Traditional CPO uses Lagrangian constraints (reactive)
- SGPO uses geometric curvature (proactive - geodesics naturally avoid hazards)

Installation:
    pip install safety-gymnasium

Usage:
    from environments import create_safety_gpo_env
    
    env = create_safety_gpo_env("SafetyPointGoal1-v0")
    obs, info = env.reset()
    
    # info['metric_value'] gives g(x) - high near hazards
    # Use env.compute_riemannian_advantage() in training
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from .base import SGPOWrapperBase, RiemannianMetricBase

try:
    import safety_gymnasium
    SAFETY_GYM_AVAILABLE = True
except ImportError:
    SAFETY_GYM_AVAILABLE = False


class MultiHazardRiemannianMetric(RiemannianMetricBase):
    """
    Riemannian metric for environments with multiple hazards.
    
    Each hazard creates a gravitational well in the metric space.
    The metric is the sum of contributions from all hazards:
    
        g(x) = 1 + Σᵢ severity / (||x - cᵢ|| - rᵢ)^sharpness
    
    This creates Schwarzschild-like singularities around each hazard.
    """
    
    def __init__(
        self,
        state_dim: int,
        hazard_centers: List[np.ndarray],
        hazard_radii: List[float],
        event_horizon_factor: float = 0.8,
        severity: float = 5.0,
        sharpness: float = 2.0,
        learnable: bool = True,
    ):
        """
        Args:
            state_dim: Dimension of state space (typically 2 for position)
            hazard_centers: List of (x, y, ...) positions for each hazard
            hazard_radii: Radius of each hazard
            event_horizon_factor: Fraction of radius that is "event horizon"
            severity: Strength of metric singularity (learnable)
            sharpness: How quickly metric grows near hazard (learnable)
            learnable: If True, severity and sharpness are nn.Parameters
        """
        super().__init__(state_dim)
        
        self.n_hazards = len(hazard_centers)
        self.register_buffer(
            'hazard_centers', 
            torch.FloatTensor(np.array(hazard_centers)[:, :state_dim])
        )
        self.register_buffer(
            'hazard_radii',
            torch.FloatTensor(hazard_radii)
        )
        self.event_horizon_factor = event_horizon_factor
        
        if learnable:
            self.severity = nn.Parameter(torch.tensor(severity))
            self.sharpness = nn.Parameter(torch.tensor(sharpness))
        else:
            self.register_buffer('severity', torch.tensor(severity))
            self.register_buffer('sharpness', torch.tensor(sharpness))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute metric factor at state x."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        x_pos = x[:, :self.state_dim]
        
        metric = torch.ones(batch_size, 1, device=x.device)
        
        for i in range(self.n_hazards):
            center = self.hazard_centers[i]
            radius = self.hazard_radii[i]
            event_horizon = radius * self.event_horizon_factor
            
            diff = x_pos - center.unsqueeze(0)
            dist = torch.norm(diff, dim=-1, keepdim=True)
            
            margin = dist - event_horizon
            margin = torch.clamp(margin, min=0.01)
            
            contribution = self.severity / (margin ** self.sharpness)
            metric = metric + contribution
        
        return metric
    
    def get_black_hole_centers(self) -> List[np.ndarray]:
        return [c.cpu().numpy() for c in self.hazard_centers]
    
    def get_event_horizons(self) -> List[float]:
        return [
            float(r * self.event_horizon_factor) 
            for r in self.hazard_radii.cpu().numpy()
        ]


class SafetyGymnasiumSGPOWrapper(SGPOWrapperBase):
    """
    SGPO wrapper for Safety-Gymnasium environments.
    
    Automatically extracts hazard information from the environment
    and constructs appropriate Riemannian metrics.
    
    Supported environments:
    - SafetyPointGoal{0,1,2}-v0
    - SafetyCarGoal{0,1,2}-v0  
    - SafetyAntGoal{0,1,2}-v0
    - SafetyPointButton{0,1,2}-v0
    - SafetyPointCircle{0,1,2}-v0
    - etc.
    """
    
    POSITION_INDICES = {
        'point': slice(0, 2),
        'car': slice(0, 2),
        'racecar': slice(0, 2),
        'ant': slice(0, 2),
        'doggo': slice(0, 2),
    }
    
    def __init__(
        self,
        env,
        metric: Optional[MultiHazardRiemannianMetric] = None,
        position_slice: Optional[slice] = None,
        event_horizon_factor: float = 0.8,
    ):
        """
        Args:
            env: Safety-Gymnasium environment
            metric: Optional pre-configured metric (auto-created if None)
            position_slice: Slice for extracting position from observation
            event_horizon_factor: How much of hazard radius is "event horizon"
        """
        self.event_horizon_factor = event_horizon_factor
        self._position_slice = position_slice
        super().__init__(env, metric)
        
    def _detect_agent_type(self) -> str:
        """Detect agent type from environment spec."""
        env_id = self.env.spec.id.lower() if self.env.spec else ""
        for agent in ['point', 'car', 'racecar', 'ant', 'doggo']:
            if agent in env_id:
                return agent
        return 'point'
    
    @property
    def position_slice(self) -> slice:
        if self._position_slice is not None:
            return self._position_slice
        agent_type = self._detect_agent_type()
        return self.POSITION_INDICES.get(agent_type, slice(0, 2))
    
    def _extract_hazards(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract hazard positions and sizes from Safety-Gymnasium environment.
        
        Safety-Gymnasium stores hazards in env.task.hazards (or similar).
        """
        centers = []
        radii = []
        
        task = getattr(self.env, 'task', None)
        if task is None:
            task = getattr(self.env.unwrapped, 'task', None)
        
        if task is None:
            return centers, radii
        
        hazard_types = ['hazards', 'vases', 'gremlins', 'pillars']
        
        for hazard_type in hazard_types:
            hazard_obj = getattr(task, hazard_type, None)
            if hazard_obj is None:
                continue
                
            positions = getattr(hazard_obj, 'pos', None)
            if positions is None:
                positions = getattr(hazard_obj, 'positions', [])
            
            size = getattr(hazard_obj, 'size', 0.2)
            if isinstance(size, (list, np.ndarray)):
                sizes = size
            else:
                sizes = [size] * len(positions)
            
            for pos, sz in zip(positions, sizes):
                centers.append(np.array(pos)[:2])
                radii.append(float(sz))
        
        if not centers:
            centers.append(np.array([0.0, 0.0]))
            radii.append(0.2)
        
        return centers, radii
    
    def _create_default_metric(self) -> MultiHazardRiemannianMetric:
        """Create metric from environment hazards."""
        centers, radii = self._extract_hazards()
        
        pos_dim = self.position_slice.stop - (self.position_slice.start or 0)
        
        return MultiHazardRiemannianMetric(
            state_dim=pos_dim,
            hazard_centers=centers,
            hazard_radii=radii,
            event_horizon_factor=self.event_horizon_factor,
            severity=5.0,
            sharpness=2.0,
            learnable=True,
        )
    
    def _compute_metric(self, obs: np.ndarray) -> float:
        """Compute metric at agent's current position."""
        with torch.no_grad():
            pos = obs[self.position_slice]
            pos_t = torch.FloatTensor(pos).unsqueeze(0)
            g = self.metric(pos_t)
            return float(g.squeeze())
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with additional SGPO information."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        metric_value = self._compute_metric(obs)
        info['metric_value'] = metric_value
        info['in_black_hole'] = metric_value > 50.0
        
        pos = obs[self.position_slice]
        min_dist = float('inf')
        for center in self.metric.get_black_hole_centers():
            dist = np.linalg.norm(pos - center)
            min_dist = min(min_dist, dist)
        info['min_dist_to_hazard'] = min_dist
        
        return obs, reward, terminated, truncated, info
    
    def visualize_metric_field(
        self,
        ax=None,
        xlim: Tuple[float, float] = (-3, 3),
        ylim: Tuple[float, float] = (-3, 3),
        resolution: int = 100,
        max_metric: float = 50.0,
    ):
        """Visualize the learned Riemannian metric as a heatmap."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        x = np.linspace(xlim[0], xlim[1], resolution)
        y = np.linspace(ylim[0], ylim[1], resolution)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.flatten(), Y.flatten()], axis=1)
        
        with torch.no_grad():
            points_t = torch.FloatTensor(points)
            metric_values = self.metric(points_t).numpy().flatten()
        
        metric_values = np.clip(metric_values, 1, max_metric)
        Z = metric_values.reshape(X.shape)
        
        im = ax.contourf(X, Y, Z, levels=20, cmap='hot_r')
        plt.colorbar(im, ax=ax, label='g(x) - Metric Factor')
        
        for center, eh in zip(
            self.metric.get_black_hole_centers(),
            self.metric.get_event_horizons()
        ):
            circle = plt.Circle(center[:2], eh, fill=False, color='black', linewidth=2)
            ax.add_patch(circle)
            ax.scatter(center[0], center[1], c='black', s=50, marker='x')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Riemannian Metric Field (Black Holes)')
        ax.set_aspect('equal')
        
        return ax


def create_safety_gpo_env(
    env_id: str,
    event_horizon_factor: float = 0.8,
    render_mode: Optional[str] = None,
    **env_kwargs
) -> SafetyGymnasiumSGPOWrapper:
    """
    Create a Safety-Gymnasium environment wrapped for SGPO training.
    
    Args:
        env_id: Safety-Gymnasium environment ID (e.g., "SafetyPointGoal1-v0")
        event_horizon_factor: Fraction of hazard radius as event horizon
        render_mode: Gymnasium render mode
        **env_kwargs: Additional environment arguments
        
    Returns:
        SafetyGymnasiumSGPOWrapper ready for SGPO training
        
    Example:
        env = create_safety_gpo_env("SafetyPointGoal1-v0")
        obs, info = env.reset()
        print(f"Metric at start: {info['metric_value']}")
    """
    if not SAFETY_GYM_AVAILABLE:
        raise ImportError(
            "safety-gymnasium not installed. Run: pip install safety-gymnasium"
        )
    
    env = safety_gymnasium.make(env_id, render_mode=render_mode, **env_kwargs)
    
    return SafetyGymnasiumSGPOWrapper(
        env,
        event_horizon_factor=event_horizon_factor,
    )


class SGPOTrainer:
    """
    Trainer for Sheaf-Geodesic Policy Optimization on Safety-Gymnasium.
    
    Implements the core SGPO algorithm:
    1. Collect trajectories with metric information
    2. Compute Riemannian advantages (A/√g)
    3. Update policy with geodesic-aware gradients
    4. Update metric to track empirical danger
    """
    
    def __init__(
        self,
        env: SafetyGymnasiumSGPOWrapper,
        actor: nn.Module,
        critic: nn.Module,
        actor_lr: float = 1e-3,
        critic_lr: float = 3e-3,
        metric_lr: float = 1e-2,
        gamma: float = 0.99,
        device: str = 'cpu',
    ):
        self.env = env
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.metric = env.metric.to(device)
        self.device = device
        
        self.opt_actor = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.opt_critic = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.opt_metric = torch.optim.Adam(
            [p for p in self.metric.parameters() if p.requires_grad], 
            lr=metric_lr
        )
        self.gamma = gamma
        
        self.trajectory_history = []
        self.metrics_history = {
            'returns': [],
            'costs': [],
            'violations': [],
            'goal_reached': [],
        }
    
    def collect_episode(self) -> Dict[str, Any]:
        """Collect one episode of experience."""
        obs, info = self.env.reset()
        
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'costs': [],
            'metric_values': [],
            'dones': [],
        }
        
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).to(self.device)
            
            with torch.no_grad():
                dist = self.actor(obs_t)
                action = dist.sample()
            
            next_obs, reward, terminated, truncated, info = self.env.step(
                action.cpu().numpy()
            )
            done = terminated or truncated
            
            trajectory['states'].append(obs)
            trajectory['actions'].append(action.cpu())
            trajectory['rewards'].append(reward)
            trajectory['costs'].append(info.get('cost', 0.0))
            trajectory['metric_values'].append(info['metric_value'])
            trajectory['dones'].append(done)
            
            obs = next_obs
        
        return trajectory
    
    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)
    
    def train_episode(self) -> Dict[str, float]:
        """Train for one episode."""
        traj = self.collect_episode()
        
        states = torch.FloatTensor(np.array(traj['states'])).to(self.device)
        actions = torch.stack(traj['actions']).to(self.device)
        returns = self.compute_returns(traj['rewards']).to(self.device)
        costs = torch.FloatTensor(traj['costs']).to(self.device)
        
        values = self.critic(states).squeeze()
        critic_loss = nn.MSELoss()(values, returns)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()
        
        pos_dim = self.env.position_slice.stop - (self.env.position_slice.start or 0)
        positions = states[:, self.env.position_slice]
        g_predicted = self.metric(positions).squeeze()
        g_target = 1.0 + costs * 10.0
        metric_loss = nn.MSELoss()(g_predicted, g_target)
        self.opt_metric.zero_grad()
        metric_loss.backward()
        self.opt_metric.step()
        
        with torch.no_grad():
            g_values = self.metric(positions).squeeze()
            advantages = returns - values.detach()
            riemannian_adv = advantages / torch.sqrt(g_values)
        
        dist = self.actor(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        actor_loss = -(log_probs * riemannian_adv).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()
        
        episode_return = sum(traj['rewards'])
        episode_cost = sum(traj['costs'])
        violations = sum(1 for c in traj['costs'] if c > 0)
        
        self.metrics_history['returns'].append(episode_return)
        self.metrics_history['costs'].append(episode_cost)
        self.metrics_history['violations'].append(violations)
        
        return {
            'return': episode_return,
            'cost': episode_cost,
            'violations': violations,
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'metric_loss': metric_loss.item(),
        }
    
    def train(self, n_episodes: int, log_interval: int = 10) -> Dict[str, List]:
        """Train for multiple episodes."""
        for ep in range(n_episodes):
            metrics = self.train_episode()
            
            if (ep + 1) % log_interval == 0:
                recent_returns = self.metrics_history['returns'][-log_interval:]
                recent_violations = self.metrics_history['violations'][-log_interval:]
                print(
                    f"Episode {ep+1}: "
                    f"Return={np.mean(recent_returns):.2f}, "
                    f"Violations={np.mean(recent_violations):.2f}"
                )
        
        return self.metrics_history
