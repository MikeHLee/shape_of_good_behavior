#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Proximal Policy Optimization (PPO) Implementation

This script demonstrates a Proximal Policy Optimization implementation, a state-of-the-art
policy gradient method used in reinforcement learning. PPO is particularly important as it's
the algorithm used in Reinforcement Learning from Human Feedback (RLHF) for training LLMs.

Before running, make sure to:
1. Activate the project venv at the project root
2. Install the required packages: pip install numpy matplotlib torch gym
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
from typing import List, Tuple, Dict, Any
import time

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """
    Combined actor-critic network for PPO.
    The actor outputs action probabilities, while the critic estimates state values.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Initialize the actor-critic network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Size of the hidden layers
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state
            
        Returns:
            Tuple: Action probabilities and state value
        """
        features = self.feature_layer(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def act(self, state):
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Tuple: Selected action, log probability, and state value
        """
        state = torch.FloatTensor(state).to(device)
        action_probs, state_value = self.forward(state)
        
        # Create a categorical distribution over action probabilities
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), state_value
    
    def evaluate(self, state, action):
        """
        Evaluate an action in the given state.
        
        Args:
            state: Input state
            action: Action to evaluate
            
        Returns:
            Tuple: Log probability, state value, and entropy
        """
        action_probs, state_value = self.forward(state)
        
        # Create a categorical distribution over action probabilities
        dist = Categorical(action_probs)
        
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_log_probs, state_value, dist_entropy


class PPOMemory:
    """
    Memory buffer for storing trajectories experienced by a PPO agent.
    """
    
    def __init__(self):
        """
        Initialize the memory buffer.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, reward, next_state, log_prob, value, done):
        """
        Add a new experience to memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            log_prob: Log probability of the action
            value: Value estimate
            done: Whether the episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """
        Clear the memory buffer.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def get_batch(self):
        """
        Get all stored experiences as a batch.
        
        Returns:
            Tuple: Batches of states, actions, old_log_probs, values, rewards, dones
        """
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        values = torch.FloatTensor(np.array(self.values)).to(device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(device)
        dones = torch.FloatTensor(np.array(self.dones)).to(device)
        
        return states, actions, old_log_probs, values, rewards, dones


class PPO:
    """
    Proximal Policy Optimization algorithm implementation.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64,
                 lr: float = 3e-4, gamma: float = 0.99, eps_clip: float = 0.2,
                 K_epochs: int = 10, entropy_coef: float = 0.01, value_coef: float = 0.5,
                 gae_lambda: float = 0.95):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Size of the hidden layers
            lr (float): Learning rate
            gamma (float): Discount factor
            eps_clip (float): PPO clipping parameter
            K_epochs (int): Number of epochs to update the policy
            entropy_coef (float): Entropy coefficient for exploration
            value_coef (float): Value loss coefficient
            gae_lambda (float): GAE lambda parameter
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gae_lambda = gae_lambda
        
        # Initialize actor-critic network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize memory
        self.memory = PPOMemory()
    
    def select_action(self, state):
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            int: Selected action
        """
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state)
        
        return action, log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for the next state
            
        Returns:
            Tuple: Returns and advantages
        """
        returns = []
        advantages = []
        gae = 0
        
        # Add the next value to the values list for easier computation
        all_values = values.clone()
        all_values = torch.cat((all_values, next_value.unsqueeze(0)))
        
        # Compute returns and advantages in reverse order
        for i in reversed(range(len(rewards))):
            # Calculate TD target and TD error
            delta = rewards[i] + self.gamma * all_values[i+1] * (1 - dones[i]) - all_values[i]
            
            # Update GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            
            # Prepend to maintain correct order
            advantages.insert(0, gae)
            returns.insert(0, gae + all_values[i])
        
        return torch.FloatTensor(returns).to(device), torch.FloatTensor(advantages).to(device)
    
    def update(self, next_value):
        """
        Update the policy using the collected experiences.
        
        Args:
            next_value: Value estimate for the next state
        """
        # Get batch data from memory
        states, actions, old_log_probs, values, rewards, dones = self.memory.get_batch()
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate actions
            log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            
            # Remove extra dimension from state_values
            state_values = state_values.squeeze(-1)
            
            # Calculate ratios
            ratios = torch.exp(log_probs - old_log_probs.detach())
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Calculate policy, value, and entropy losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(state_values, returns.detach())
            entropy_loss = -dist_entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear memory
        self.memory.clear()


def train_ppo(env, agent: PPO, max_episodes: int = 1000, max_timesteps: int = 500,
             update_timestep: int = 4000, log_interval: int = 20) -> List[float]:
    """
    Train the PPO agent.
    
    Args:
        env: The environment
        agent (PPO): The PPO agent
        max_episodes (int): Maximum number of episodes
        max_timesteps (int): Maximum timesteps per episode
        update_timestep (int): Timesteps between policy updates
        log_interval (int): Episodes between logging
        
    Returns:
        List[float]: Episode rewards
    """
    # Logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    episode_rewards = []
    
    # Training loop
    for i_episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(max_timesteps):
            timestep += 1
            
            # Select action
            action, log_prob, val = agent.select_action(state)
            
            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience in memory
            agent.memory.add(state, action, reward, next_state, log_prob, val, done)
            
            state = next_state
            episode_reward += reward
            
            # Update if its time
            if timestep % update_timestep == 0:
                # Get value estimate for the next state
                _, _, next_value = agent.select_action(next_state)
                next_value = torch.FloatTensor([next_value]).to(device)
                
                # Update the policy
                agent.update(next_value)
                timestep = 0
            
            if done:
                break
        
        # Logging
        running_reward += episode_reward
        episode_rewards.append(episode_reward)
        avg_length += t
        
        if i_episode % log_interval == 0:
            avg_length = avg_length / log_interval
            running_reward = running_reward / log_interval
            
            print(f'Episode {i_episode}\tAvg length: {avg_length:.2f}\tAvg reward: {running_reward:.2f}')
            
            running_reward = 0
            avg_length = 0
    
    return episode_rewards


def plot_rewards(rewards: List[float]):
    """
    Plot the rewards during training.
    
    Args:
        rewards (List[float]): List of episode rewards
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.plot(np.convolve(rewards, np.ones(100) / 100, mode='same'), 'r-')
    plt.title('PPO Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()


def test_agent(env, agent: PPO, n_episodes: int = 5):
    """
    Test the trained agent.
    
    Args:
        env: The environment
        agent (PPO): The trained PPO agent
        n_episodes (int): Number of episodes to test
    """
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        print(f"\nEpisode {i_episode}:")
        
        while not done:
            # Select action
            action, _, _ = agent.select_action(state)
            
            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Render the environment
            env.render()
            time.sleep(0.01)
            
            state = next_state
            episode_reward += reward
            
            print(f"Action: {action}, Reward: {reward:.2f}")
        
        print(f"Episode {i_episode} Reward: {episode_reward:.2f}")


def main():
    """
    Main function to run the PPO algorithm on the LunarLander environment.
    """
    print("\n=== Proximal Policy Optimization (PPO) Implementation ===\n")
    
    # Create environment
    env_name = "LunarLander-v2"
    env = gym.make(env_name, render_mode=None)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create PPO agent
    agent = PPO(state_dim, action_dim, hidden_dim=64, lr=3e-4, gamma=0.99,
                eps_clip=0.2, K_epochs=10, entropy_coef=0.01, value_coef=0.5,
                gae_lambda=0.95)
    
    # Train the agent
    print("\nTraining the PPO agent...")
    rewards = train_ppo(env, agent, max_episodes=300, max_timesteps=1000,
                       update_timestep=4000, log_interval=20)
    
    # Plot the rewards
    plot_rewards(rewards)
    
    # Test the trained agent
    print("\nTesting the trained agent...")
    test_env = gym.make(env_name, render_mode='human')
    test_agent(test_env, agent, n_episodes=3)
    
    # Close the environments
    env.close()
    test_env.close()


if __name__ == "__main__":
    main()
