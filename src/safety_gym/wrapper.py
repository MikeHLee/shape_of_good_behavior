"""
Gym wrapper that adds topological safety metrics to any environment.

This is the main interface for using the Safety Gym library.
"""

import gym
import numpy as np
from typing import Dict, Any, Optional, List
from .topological_space import TopologicalSpace
from .continuous_space import ContinuousControlSpace
from .discrete_space import DiscreteNavigationSpace


class TopologicalSafetyWrapper(gym.Wrapper):
    """
    Wraps any Gym environment to add topological safety metrics.
    
    This wrapper:
    1. Tracks harmonic risk at each state
    2. Identifies black hole regions from failures
    3. Computes safety metrics (proximity, violations)
    4. Provides topological gradients for policy optimization
    
    Usage:
        env = gym.make("HalfCheetah-v3")
        safe_env = TopologicalSafetyWrapper(env, space_type="continuous")
        
        obs = safe_env.reset()
        done = False
        while not done:
            action = policy(obs)
            obs, reward, done, info = safe_env.step(action)
            print(f"Harmonic risk: {info['harmonic_risk']:.3f}")
            print(f"Is safe: {info['is_safe']}")
    """
    
    def __init__(
        self,
        env: gym.Env,
        space_type: str = "continuous",
        track_trajectories: bool = True,
        **space_kwargs
    ):
        """
        Initialize topological safety wrapper.
        
        Args:
            env: Gym environment to wrap
            space_type: "continuous", "discrete", or "image"
            track_trajectories: Whether to store full trajectories
            **space_kwargs: Additional arguments for topological space
        """
        super().__init__(env)
        
        # Create appropriate topological space
        if space_type == "continuous":
            state_dim = env.observation_space.shape[0]
            self.topo_space = ContinuousControlSpace(
                env_name=env.spec.id if hasattr(env, 'spec') else "unknown",
                state_dim=state_dim,
                **space_kwargs
            )
        elif space_type == "discrete":
            self.topo_space = DiscreteNavigationSpace(**space_kwargs)
        else:
            raise ValueError(f"Unknown space type: {space_type}")
        
        self.space_type = space_type
        self.track_trajectories = track_trajectories
        
        # Episode tracking
        self.current_trajectory = []
        self.all_trajectories = []
        self.failed_trajectories = []
        
        # Metrics tracking
        self.episode_metrics = {
            'harmonic_risk': [],
            'black_hole_proximity': [],
            'safety_violations': 0,
            'trajectory_shift': 0.0,
        }
        
        self.cumulative_metrics = {
            'total_episodes': 0,
            'total_violations': 0,
            'mean_risk': 0.0,
            'max_risk': 0.0,
        }
    
    def reset(self, **kwargs):
        """Reset environment and episode tracking."""
        obs = self.env.reset(**kwargs)
        
        # Store previous trajectory if it exists
        if self.current_trajectory:
            self.all_trajectories.append({
                'states': [t['state'] for t in self.current_trajectory],
                'actions': [t['action'] for t in self.current_trajectory],
                'rewards': [t['reward'] for t in self.current_trajectory],
                'metrics': self.episode_metrics.copy(),
            })
        
        # Reset episode tracking
        self.current_trajectory = []
        self.episode_metrics = {
            'harmonic_risk': [],
            'black_hole_proximity': [],
            'safety_violations': 0,
            'trajectory_shift': 0.0,
        }
        
        return obs
    
    def step(self, action):
        """
        Step environment and compute topological metrics.
        
        Returns:
            obs: Next observation
            reward: Environment reward
            done: Episode termination flag
            info: Dictionary with additional metrics including:
                - harmonic_risk: H¹ cohomology risk at current state
                - is_safe: Whether state is in safe region
                - black_hole_proximity: Distance to nearest black hole
                - safety_score: Combined safety metric
        """
        obs, reward, done, info = self.env.step(action)
        
        # Compute topological metrics if we have topology data
        if len(self.topo_space.topology_data['embeddings']) > 0:
            harmonic_risk = self.topo_space.compute_harmonic_risk(obs)
            is_safe = self.topo_space.is_safe(obs)
            bh_proximity = self.topo_space.compute_black_hole_proximity(obs)
            
            # Combined safety score (higher = safer)
            safety_score = (1.0 - harmonic_risk) + min(bh_proximity / 10.0, 1.0)
            
            # Track metrics
            self.episode_metrics['harmonic_risk'].append(harmonic_risk)
            self.episode_metrics['black_hole_proximity'].append(bh_proximity)
            if not is_safe:
                self.episode_metrics['safety_violations'] += 1
            
            # Add to info dict
            info['harmonic_risk'] = harmonic_risk
            info['is_safe'] = is_safe
            info['black_hole_proximity'] = bh_proximity
            info['safety_score'] = safety_score
        else:
            # No topology data yet
            info['harmonic_risk'] = 0.5
            info['is_safe'] = True
            info['black_hole_proximity'] = float('inf')
            info['safety_score'] = 0.5
        
        # Track trajectory
        if self.track_trajectories:
            self.current_trajectory.append({
                'state': obs.copy() if isinstance(obs, np.ndarray) else obs,
                'action': action.copy() if isinstance(action, np.ndarray) else action,
                'reward': reward,
                'info': info.copy(),
            })
        
        # If episode failed (negative reward or explicit failure), mark trajectory
        if done and (reward < 0 or info.get('failure', False)):
            self.failed_trajectories.append({
                'states': [t['state'] for t in self.current_trajectory],
                'final_reward': reward,
            })
        
        # Update cumulative metrics
        if done:
            self.cumulative_metrics['total_episodes'] += 1
            self.cumulative_metrics['total_violations'] += self.episode_metrics['safety_violations']
            if self.episode_metrics['harmonic_risk']:
                mean_risk = np.mean(self.episode_metrics['harmonic_risk'])
                max_risk = np.max(self.episode_metrics['harmonic_risk'])
                self.cumulative_metrics['mean_risk'] = mean_risk
                self.cumulative_metrics['max_risk'] = max(
                    self.cumulative_metrics['max_risk'], max_risk
                )
        
        return obs, reward, done, info
    
    def add_topology_sample(self, state: Any, risk: float):
        """
        Add a state to the topology database.
        
        Args:
            state: State to add
            risk: Harmonic risk at this state
        """
        self.topo_space.add_topology_sample(state, risk)
    
    def mine_topology_from_random_exploration(self, n_steps: int = 1000):
        """
        Build topology database from random exploration.
        
        Args:
            n_steps: Number of random steps to take
        """
        print(f"Mining topology from {n_steps} random steps...")
        
        obs = self.reset()
        for step in range(n_steps):
            action = self.env.action_space.sample()
            next_obs, reward, done, info = self.env.step(action)
            
            # Estimate risk (high if negative reward or failure)
            if reward < 0 or info.get('failure', False):
                risk = 0.9
            else:
                risk = 0.1
            
            self.add_topology_sample(next_obs, risk)
            
            if done:
                obs = self.reset()
            else:
                obs = next_obs
            
            if (step + 1) % 100 == 0:
                print(f"  {step + 1}/{n_steps} steps completed")
        
        print(f"Topology mining complete: {len(self.topo_space.topology_data['states'])} samples")
    
    def identify_black_holes_from_failures(self, **kwargs):
        """
        Identify black hole regions from failed trajectories.
        
        Args:
            **kwargs: Arguments for identify_black_holes
        """
        if not self.failed_trajectories:
            print("No failed trajectories to analyze")
            return
        
        print(f"Identifying black holes from {len(self.failed_trajectories)} failed trajectories...")
        
        if self.space_type == "continuous":
            self.topo_space.identify_black_holes_from_trajectories(
                self.failed_trajectories, **kwargs
            )
        else:
            # For discrete spaces, extract failure states directly
            failed_states = []
            for traj in self.failed_trajectories:
                failed_states.extend(traj['states'][-10:])
            self.topo_space.identify_black_holes(failed_states, **kwargs)
        
        print(f"Identified {len(self.topo_space.black_hole_regions)} black hole regions")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of safety metrics.
        
        Returns:
            summary: Dictionary with metric statistics
        """
        return {
            'cumulative': self.cumulative_metrics.copy(),
            'topology': self.topo_space.get_topology_summary(),
            'current_episode': {
                'mean_risk': np.mean(self.episode_metrics['harmonic_risk']) if self.episode_metrics['harmonic_risk'] else 0.0,
                'violations': self.episode_metrics['safety_violations'],
            }
        }
    
    def compute_topological_reward_shaping(
        self,
        state: Any,
        base_reward: float,
        risk_penalty: float = 1.0,
        proximity_bonus: float = 0.1,
    ) -> float:
        """
        Compute reward shaping based on topological safety.
        
        This can be used to augment the environment reward with safety bonuses.
        
        Args:
            state: Current state
            base_reward: Original environment reward
            risk_penalty: Penalty weight for high-risk states
            proximity_bonus: Bonus weight for staying far from black holes
        
        Returns:
            shaped_reward: Modified reward with topological shaping
        """
        harmonic_risk = self.topo_space.compute_harmonic_risk(state)
        bh_proximity = self.topo_space.compute_black_hole_proximity(state)
        
        # Penalty for high risk
        risk_term = -risk_penalty * harmonic_risk
        
        # Bonus for staying far from black holes
        proximity_term = proximity_bonus * min(bh_proximity / 10.0, 1.0)
        
        shaped_reward = base_reward + risk_term + proximity_term
        
        return shaped_reward
    
    def save_topology(self, path: str):
        """
        Save topology database to file.
        
        Args:
            path: Path to save topology data
        """
        import pickle
        
        data = {
            'topology_data': self.topo_space.topology_data,
            'black_hole_regions': self.topo_space.black_hole_regions,
            'space_type': self.space_type,
            'cumulative_metrics': self.cumulative_metrics,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Topology saved to {path}")
    
    def load_topology(self, path: str):
        """
        Load topology database from file.
        
        Args:
            path: Path to load topology data from
        """
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.topo_space.topology_data = data['topology_data']
        self.topo_space.black_hole_regions = data['black_hole_regions']
        self.cumulative_metrics = data.get('cumulative_metrics', self.cumulative_metrics)
        
        print(f"Topology loaded from {path}")
        print(f"  {len(self.topo_space.topology_data['states'])} samples")
        print(f"  {len(self.topo_space.black_hole_regions)} black holes")
