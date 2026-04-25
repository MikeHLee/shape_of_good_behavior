"""
SGPO Trainer with process supervision feedback integration.
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from collections import defaultdict

from environment import AnomalyNavigationEnv, AnomalyType
from feedback import (
    TrajectoryFeedback,
    AnomalyCandidate,
    ProcessSupervisor,
)
from models import (
    Actor,
    Critic,
    AnomalyAwareRewardLearning,
    AnomalyAwareMetric,
)


@dataclass
class TrainingConfig:
    """Configuration for SGPO training."""
    gamma: float = 0.99
    lam: float = 0.97
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    metric_lr: float = 1e-2
    reward_lr: float = 1e-3
    train_iters: int = 5
    
    # Reward shaping
    goal_bonus: float = 10.0
    hazard_penalty: float = -20.0
    wormhole_penalty: float = -5.0
    
    # Metric update
    metric_update_freq: int = 10
    cliff_severity: float = 5.0


@dataclass
class TrainingState:
    """Current training state."""
    episode: int = 0
    total_steps: int = 0
    history: Dict[str, List] = field(default_factory=lambda: defaultdict(list))
    detected_anomalies: List[AnomalyCandidate] = field(default_factory=list)
    pending_feedback: List[str] = field(default_factory=list)


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lam: float = 0.97,
) -> tuple:
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    values = list(values) + [0]
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


class SGPOTrainer:
    """Sheaf-Geodesic Policy Optimization trainer with process supervision.
    
    Supports two modes:
    1. Automatic: Uses simulated feedback for rapid iteration
    2. Interactive: Pauses for human feedback via Streamlit UI
    """
    
    def __init__(
        self,
        env: AnomalyNavigationEnv,
        config: TrainingConfig = None,
        device: torch.device = None,
        interactive: bool = False,
    ):
        self.env = env
        self.config = config or TrainingConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.interactive = interactive
        
        # Initialize models
        self.actor = Actor(env.obs_dim, env.act_dim).to(self.device)
        self.critic = Critic(env.obs_dim).to(self.device)
        self.metric = AnomalyAwareMetric(env.black_holes).to(self.device)
        self.reward_learner = AnomalyAwareRewardLearning(
            env.obs_dim, env.act_dim
        ).to(self.device)
        
        # Optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.metric_optim = torch.optim.Adam(self.metric.parameters(), lr=self.config.metric_lr)
        self.reward_optim = torch.optim.Adam(self.reward_learner.parameters(), lr=self.config.reward_lr)
        
        # Feedback supervisor
        self.supervisor = ProcessSupervisor(env, interactive=interactive)
        
        # State
        self.state = TrainingState()
        
        # Callbacks
        self._on_episode_end: Optional[Callable] = None
        self._on_feedback_needed: Optional[Callable] = None
    
    def set_callbacks(
        self,
        on_episode_end: Callable = None,
        on_feedback_needed: Callable = None,
    ):
        """Set callbacks for UI integration."""
        self._on_episode_end = on_episode_end
        self._on_feedback_needed = on_feedback_needed
    
    def collect_trajectory(self, deterministic: bool = False) -> Dict:
        """Collect a single trajectory."""
        obs = self.env.reset()
        observations = [obs]
        actions = []
        log_probs = []
        values = []
        done = False
        final_info = {}
        
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                action, log_prob = self.actor.get_action(obs_t, deterministic)
                value = self.critic(obs_t)
            
            action_np = action.cpu().numpy()
            obs, _, done, info = self.env.step(action_np)
            
            observations.append(obs)
            actions.append(action_np)
            log_probs.append(log_prob.item())
            values.append(value.item())
            final_info = info
        
        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'log_probs': log_probs,
            'values': values,
            'path': self.env.get_trajectory(),
            'step_anomalies': self.env.get_step_anomalies(),
            'reached_goal': final_info.get('reached_goal', False),
            'used_wormhole': final_info.get('used_wormhole', False),
            'final_info': final_info,
        }
    
    def compute_rewards(
        self,
        traj_data: Dict,
        feedback: Optional[TrajectoryFeedback] = None,
    ) -> List[float]:
        """Compute rewards from trajectory and feedback."""
        T = len(traj_data['actions'])
        rewards = [0.0] * T
        
        # Use step-level feedback if available
        if feedback and feedback.step_feedback:
            for sf in feedback.step_feedback:
                if sf.step_idx < T:
                    rewards[sf.step_idx] = sf.quality - 0.5
        
        # Terminal bonuses/penalties
        if traj_data['reached_goal']:
            rewards[-1] += self.config.goal_bonus
            if traj_data['used_wormhole']:
                rewards[-1] += self.config.wormhole_penalty
        
        # Hazard penalties
        for i, anomaly in enumerate(traj_data['step_anomalies']):
            if i < T:
                if anomaly == AnomalyType.BLACK_HOLE:
                    rewards[i] += self.config.hazard_penalty
                elif anomaly == AnomalyType.CLIFF:
                    rewards[i] += self.config.hazard_penalty * 0.5
        
        return rewards
    
    def train_episode_ppo(self, traj_data: Dict, rewards: List[float]) -> Dict:
        """Standard PPO update."""
        observations = torch.tensor(
            traj_data['observations'][:-1],
            dtype=torch.float32,
            device=self.device
        )
        actions = torch.tensor(
            traj_data['actions'],
            dtype=torch.float32,
            device=self.device
        )
        old_log_probs = torch.tensor(
            traj_data['log_probs'],
            dtype=torch.float32,
            device=self.device
        )
        values = traj_data['values']
        
        # GAE
        advantages, returns = compute_gae(
            rewards, values, self.config.gamma, self.config.lam
        )
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # PPO updates
        for _ in range(self.config.train_iters):
            dist = self.actor(observations)
            new_log_probs = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
            ) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            critic_loss = F.mse_loss(self.critic(observations), returns)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
        }
    
    def train_episode_gpo(self, traj_data: Dict, rewards: List[float]) -> Dict:
        """SGPO update with Riemannian metric."""
        observations = torch.tensor(
            traj_data['observations'][:-1],
            dtype=torch.float32,
            device=self.device
        )
        actions = torch.tensor(
            traj_data['actions'],
            dtype=torch.float32,
            device=self.device
        )
        old_log_probs = torch.tensor(
            traj_data['log_probs'],
            dtype=torch.float32,
            device=self.device
        )
        values = traj_data['values']
        
        # GAE
        advantages, returns = compute_gae(
            rewards, values, self.config.gamma, self.config.lam
        )
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Riemannian scaling: A_geo = A / sqrt(g(x))
        with torch.no_grad():
            g = self.metric(observations).squeeze()
        
        riemannian_advantages = advantages / torch.sqrt(g + 1e-8)
        riemannian_advantages = (
            (riemannian_advantages - riemannian_advantages.mean()) /
            (riemannian_advantages.std() + 1e-8)
        )
        
        # PPO updates with Riemannian advantages
        for _ in range(self.config.train_iters):
            dist = self.actor(observations)
            new_log_probs = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * riemannian_advantages
            surr2 = torch.clamp(
                ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
            ) * riemannian_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            critic_loss = F.mse_loss(self.critic(observations), returns)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
        
        # Update metric based on hazard encounters
        cost_tensor = torch.tensor(
            [1.0 if a != AnomalyType.NONE else 0.0 for a in traj_data['step_anomalies']],
            dtype=torch.float32,
            device=self.device
        )
        if len(cost_tensor) > 0:
            pred_risk = self.metric(observations).squeeze()
            # Trim to match lengths
            min_len = min(len(pred_risk), len(cost_tensor))
            pred_risk = pred_risk[:min_len]
            cost_tensor = cost_tensor[:min_len]
            
            metric_target = 1.0 + 10.0 * cost_tensor
            metric_loss = F.mse_loss(pred_risk, metric_target)
            
            self.metric_optim.zero_grad()
            metric_loss.backward()
            self.metric_optim.step()
        else:
            metric_loss = torch.tensor(0.0)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'metric_loss': metric_loss.item(),
        }
    
    def process_feedback(self, feedback: TrajectoryFeedback):
        """Process feedback and update models/metric."""
        # Detect anomalies
        anomalies = self.reward_learner.detect_anomalies(feedback, self.device)
        self.state.detected_anomalies.extend(anomalies)
        
        # Update metric with detected cliffs
        for anom in anomalies:
            if anom.anomaly_type == AnomalyType.CLIFF and anom.location is not None:
                self.metric.add_cliff(anom.location, self.config.cliff_severity)
        
        return anomalies
    
    def run_episode(self, use_gpo: bool = True) -> Dict:
        """Run a single training episode."""
        # Collect trajectory
        traj_data = self.collect_trajectory()
        
        # Get feedback
        if self.interactive:
            traj_id = self.supervisor.create_pending_feedback(traj_data)
            self.state.pending_feedback.append(traj_id)
            
            if self._on_feedback_needed:
                self._on_feedback_needed(traj_id, traj_data)
            
            # Return partial result - training continues after feedback
            return {
                'status': 'awaiting_feedback',
                'trajectory_id': traj_id,
                'traj_data': traj_data,
            }
        else:
            feedback = self.supervisor.evaluate_trajectory_auto(traj_data)
        
        # Compute rewards
        rewards = self.compute_rewards(traj_data, feedback)
        
        # Train
        if use_gpo:
            losses = self.train_episode_gpo(traj_data, rewards)
        else:
            losses = self.train_episode_ppo(traj_data, rewards)
        
        # Process feedback for anomaly detection
        anomalies = self.process_feedback(feedback)
        
        # Update state
        self.state.episode += 1
        self.state.total_steps += len(traj_data['actions'])
        
        result = {
            'status': 'completed',
            'episode': self.state.episode,
            'return': sum(rewards),
            'steps': len(traj_data['actions']),
            'reached_goal': traj_data['reached_goal'],
            'used_wormhole': traj_data['used_wormhole'],
            'anomalies_detected': len(anomalies),
            **losses,
        }
        
        # Update history
        for k, v in result.items():
            if isinstance(v, (int, float, bool)):
                self.state.history[k].append(v)
        
        if self._on_episode_end:
            self._on_episode_end(result, traj_data, feedback)
        
        return result
    
    def continue_after_feedback(
        self,
        traj_id: str,
        traj_data: Dict,
        use_gpo: bool = True,
    ) -> Dict:
        """Continue training after receiving interactive feedback."""
        feedback = self.supervisor.finalize_feedback(traj_id)
        
        if traj_id in self.state.pending_feedback:
            self.state.pending_feedback.remove(traj_id)
        
        # Compute rewards
        rewards = self.compute_rewards(traj_data, feedback)
        
        # Train
        if use_gpo:
            losses = self.train_episode_gpo(traj_data, rewards)
        else:
            losses = self.train_episode_ppo(traj_data, rewards)
        
        # Process feedback
        anomalies = self.process_feedback(feedback)
        
        # Update state
        self.state.episode += 1
        self.state.total_steps += len(traj_data['actions'])
        
        result = {
            'status': 'completed',
            'episode': self.state.episode,
            'return': sum(rewards),
            'steps': len(traj_data['actions']),
            'reached_goal': traj_data['reached_goal'],
            'used_wormhole': traj_data['used_wormhole'],
            'anomalies_detected': len(anomalies),
            **losses,
        }
        
        for k, v in result.items():
            if isinstance(v, (int, float, bool)):
                self.state.history[k].append(v)
        
        if self._on_episode_end:
            self._on_episode_end(result, traj_data, feedback)
        
        return result
    
    def train(
        self,
        n_episodes: int,
        use_gpo: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """Train for n episodes (automatic mode only)."""
        if self.interactive:
            raise ValueError("Use run_episode() for interactive mode")
        
        for ep in range(n_episodes):
            result = self.run_episode(use_gpo=use_gpo)
            
            if verbose and (ep + 1) % 20 == 0:
                recent = min(20, len(self.state.history['return']))
                avg_return = np.mean(self.state.history['return'][-recent:])
                goal_rate = np.mean(self.state.history['reached_goal'][-recent:])
                wormhole_rate = np.mean(self.state.history['used_wormhole'][-recent:])
                
                print(
                    f"Episode {ep+1}: "
                    f"Return={avg_return:.1f}, "
                    f"Goals={goal_rate:.0%}, "
                    f"Wormholes={wormhole_rate:.0%}, "
                    f"Anomalies={len(self.state.detected_anomalies)}"
                )
        
        return {
            'history': dict(self.state.history),
            'detected_anomalies': self.state.detected_anomalies,
            'total_episodes': self.state.episode,
        }
    
    def get_state_dict(self) -> Dict:
        """Get state dict for saving."""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'metric': self.metric.state_dict(),
            'reward_learner': self.reward_learner.state_dict(),
            'state': {
                'episode': self.state.episode,
                'total_steps': self.state.total_steps,
                'history': dict(self.state.history),
            },
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dict."""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.metric.load_state_dict(state_dict['metric'])
        self.reward_learner.load_state_dict(state_dict['reward_learner'])
        
        self.state.episode = state_dict['state']['episode']
        self.state.total_steps = state_dict['state']['total_steps']
        self.state.history = defaultdict(list, state_dict['state']['history'])
