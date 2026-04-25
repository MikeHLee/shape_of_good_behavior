#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Q-Network (DQN) Implementation

This script demonstrates a Deep Q-Network implementation, a reinforcement learning
algorithm that uses neural networks to approximate the Q-function. It includes
a CartPole environment and visualizations of the learning process.

Before running, make sure to:
1. Activate the project venv at the project root
2. Install the required packages: pip install numpy matplotlib torch gym
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Any
import time

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    A buffer for storing and sampling experiences for experience replay.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size (int): Size of the batch to sample
            
        Returns:
            List[Experience]: Batch of experiences
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            int: Current size of the buffer
        """
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Neural network for approximating the Q-function.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        """
        Initialize the Q-network.
        
        Args:
            state_size (int): Dimension of the state space
            action_size (int): Dimension of the action space
            hidden_size (int): Size of the hidden layers
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """
    Agent implementing the Deep Q-Network algorithm.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64,
                 lr: float = 1e-3, gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 buffer_size: int = 10000, batch_size: int = 64,
                 update_every: int = 4, tau: float = 1e-3):
        """
        Initialize the DQN agent.
        
        Args:
            state_size (int): Dimension of the state space
            action_size (int): Dimension of the action space
            hidden_size (int): Size of the hidden layers in the Q-network
            lr (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Decay rate for exploration
            buffer_size (int): Size of the replay buffer
            batch_size (int): Batch size for training
            update_every (int): How often to update the network
            tau (float): Soft update parameter
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.tau = tau
        
        # Q-Networks (online and target)
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """
        Update the agent's knowledge based on the new experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
    
    def act(self, state, eval_mode: bool = False):
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state
            eval_mode (bool): Whether to use evaluation mode (no exploration)
            
        Returns:
            int: Selected action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Set the network to evaluation mode
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        
        # Set the network back to training mode
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())
    
    def learn(self, experiences: List[Experience]):
        """
        Update the Q-network based on a batch of experiences.
        
        Args:
            experiences (List[Experience]): Batch of experiences
        """
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Get max predicted Q values for next states from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def soft_update(self):
        """
        Soft update of the target network's weights.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


def train_dqn(env, agent: DQNAgent, n_episodes: int = 1000, max_t: int = 1000, 
              print_every: int = 100) -> List[float]:
    """
    Train the DQN agent.
    
    Args:
        env: The environment
        agent (DQNAgent): The DQN agent
        n_episodes (int): Number of episodes to train
        max_t (int): Maximum number of timesteps per episode
        print_every (int): How often to print the progress
        
    Returns:
        List[float]: Scores for each episode
    """
    scores = []
    
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
        
        scores.append(score)
        
        # Print progress
        if i_episode % print_every == 0:
            print(f'Episode {i_episode}/{n_episodes} | Average Score: {np.mean(scores[-100:]):.2f}')
    
    return scores


def plot_scores(scores: List[float]):
    """
    Plot the scores during training.
    
    Args:
        scores (List[float]): List of scores for each episode
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(scores)), np.convolve(scores, np.ones(100) / 100, mode='same'), 'r-')
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title('DQN Training Progress')
    plt.grid(True)
    plt.show()


def test_agent(env, agent: DQNAgent, n_episodes: int = 5, render: bool = True):
    """
    Test the trained agent.
    
    Args:
        env: The environment
        agent (DQNAgent): The trained DQN agent
        n_episodes (int): Number of episodes to test
        render (bool): Whether to render the environment
    """
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        done = False
        
        print(f"\nEpisode {i_episode}:")
        
        while not done:
            action = agent.act(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if render:
                env.render()
                time.sleep(0.02)
            
            print(f"Action: {action}, Reward: {reward:.2f}")
            
            state = next_state
            score += reward
        
        print(f"Episode {i_episode} Score: {score}")


def main():
    """
    Main function to run the DQN algorithm on the CartPole environment.
    """
    print("\n=== Deep Q-Network (DQN) Implementation ===\n")
    
    # Create environment
    env = gym.make('CartPole-v1', render_mode=None)
    
    # Get state and action dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Create DQN agent
    agent = DQNAgent(state_size=state_size, action_size=action_size,
                     hidden_size=64, lr=5e-4, gamma=0.99,
                     epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                     buffer_size=10000, batch_size=64,
                     update_every=4, tau=1e-3)
    
    # Train the agent
    print("\nTraining the DQN agent...")
    scores = train_dqn(env, agent, n_episodes=500, max_t=1000, print_every=100)
    
    # Plot the scores
    plot_scores(scores)
    
    # Test the trained agent
    print("\nTesting the trained agent...")
    test_env = gym.make('CartPole-v1', render_mode='human')
    test_agent(test_env, agent, n_episodes=3)
    
    # Close the environments
    env.close()
    test_env.close()


if __name__ == "__main__":
    main()
