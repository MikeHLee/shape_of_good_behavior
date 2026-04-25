#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic Q-Learning Implementation

This script demonstrates a simple implementation of the Q-learning algorithm,
a fundamental reinforcement learning technique. It includes a grid world environment
and visualizations of the learning process.

Before running, make sure to:
1. Activate the project venv at the project root
2. Install the required packages: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, List, Dict, Any
import time


class GridWorldEnv:
    """
    A simple grid world environment for reinforcement learning experiments.
    The agent navigates a grid with obstacles and a goal state.
    """
    
    def __init__(self, size: int = 5, obstacles: List[Tuple[int, int]] = None, goal: Tuple[int, int] = None):
        """
        Initialize the grid world environment.
        
        Args:
            size (int): Size of the grid (size x size)
            obstacles (List[Tuple[int, int]]): List of obstacle positions
            goal (Tuple[int, int]): Position of the goal state
        """
        self.size = size
        self.obstacles = obstacles or [(1, 1), (2, 2), (3, 1)]
        self.goal = goal or (size - 1, size - 1)
        self.reset()
        
        # Define action space: 0=up, 1=right, 2=down, 3=left
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['Up', 'Right', 'Down', 'Left']
        
    def reset(self) -> Tuple[int, int]:
        """
        Reset the environment to the initial state.
        
        Returns:
            Tuple[int, int]: Initial state (position)
        """
        self.state = (0, 0)  # Start at top-left corner
        return self.state
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment based on the chosen action.
        
        Args:
            action (int): Action index (0=up, 1=right, 2=down, 3=left)
            
        Returns:
            Tuple[Tuple[int, int], float, bool, Dict[str, Any]]: Next state, reward, done flag, info dict
        """
        # Get action direction
        direction = self.actions[action]
        
        # Calculate new position
        new_row = self.state[0] + direction[0]
        new_col = self.state[1] + direction[1]
        new_state = (new_row, new_col)
        
        # Check if the move is valid
        if (0 <= new_row < self.size and 0 <= new_col < self.size and 
                new_state not in self.obstacles):
            self.state = new_state
        
        # Calculate reward
        if self.state == self.goal:
            reward = 1.0
            done = True
        elif self.state in self.obstacles:
            reward = -1.0
            done = False
        else:
            reward = -0.01  # Small negative reward for each step
            done = False
            
        info = {}
        return self.state, reward, done, info
    
    def render(self, q_table=None) -> None:
        """
        Render the grid world environment.
        
        Args:
            q_table (np.ndarray, optional): Q-table to visualize the policy
        """
        grid = np.zeros((self.size, self.size))
        
        # Mark obstacles
        for obs in self.obstacles:
            if 0 <= obs[0] < self.size and 0 <= obs[1] < self.size:
                grid[obs] = -1
        
        # Mark goal
        grid[self.goal] = 2
        
        # Mark current position
        grid[self.state] = 1
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Display the grid
        cmap = plt.cm.colors.ListedColormap(['white', 'green', 'gold', 'red'])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(grid, cmap=cmap, norm=norm)
        
        # Draw grid lines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, self.size, 1))
        ax.set_yticks(np.arange(-0.5, self.size, 1))
        
        # Hide tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Add cell coordinates
        for i in range(self.size):
            for j in range(self.size):
                ax.text(j, i, f'({i},{j})', ha='center', va='center', color='black')
        
        # If Q-table is provided, visualize the policy
        if q_table is not None:
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) in self.obstacles or (i, j) == self.goal:
                        continue
                        
                    # Get the best action for this state
                    state_idx = i * self.size + j
                    best_action = np.argmax(q_table[state_idx])
                    
                    # Draw an arrow indicating the best action
                    if best_action == 0:  # Up
                        ax.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
                    elif best_action == 1:  # Right
                        ax.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
                    elif best_action == 2:  # Down
                        ax.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
                    elif best_action == 3:  # Left
                        ax.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        
        plt.title('Grid World Environment')
        plt.show()


def q_learning(env: GridWorldEnv, episodes: int = 1000, alpha: float = 0.1, gamma: float = 0.99, 
               epsilon: float = 0.1, decay_rate: float = 0.01) -> np.ndarray:
    """
    Implement the Q-learning algorithm.
    
    Args:
        env (GridWorldEnv): The environment
        episodes (int): Number of episodes to train
        alpha (float): Learning rate
        gamma (float): Discount factor
        epsilon (float): Exploration rate
        decay_rate (float): Epsilon decay rate
        
    Returns:
        np.ndarray: Learned Q-table
    """
    # Initialize Q-table with zeros
    num_states = env.size * env.size
    num_actions = len(env.actions)
    q_table = np.zeros((num_states, num_actions))
    
    # Initialize lists to store metrics
    rewards_per_episode = []
    steps_per_episode = []
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        state_idx = state[0] * env.size + state[1]
        done = False
        total_reward = 0
        steps = 0
        
        # Decay epsilon over time
        current_epsilon = epsilon * (1 / (1 + decay_rate * episode))
        
        while not done:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < current_epsilon:
                action = random.randint(0, num_actions - 1)  # Explore
            else:
                action = np.argmax(q_table[state_idx])  # Exploit
            
            # Take action and observe next state and reward
            next_state, reward, done, _ = env.step(action)
            next_state_idx = next_state[0] * env.size + next_state[1]
            
            # Q-learning update rule
            best_next_action = np.argmax(q_table[next_state_idx])
            td_target = reward + gamma * q_table[next_state_idx][best_next_action]
            td_error = td_target - q_table[state_idx][action]
            q_table[state_idx][action] += alpha * td_error
            
            # Update state
            state = next_state
            state_idx = next_state_idx
            total_reward += reward
            steps += 1
        
        # Store metrics
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}/{episodes}, Reward: {total_reward:.2f}, Steps: {steps}")
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(steps_per_episode)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.show()
    
    return q_table


def print_policy(env: GridWorldEnv, q_table: np.ndarray) -> None:
    """
    Print the learned policy in a human-readable format.
    
    Args:
        env (GridWorldEnv): The environment
        q_table (np.ndarray): Learned Q-table
    """
    print("\nLearned Policy:")
    print("-" * 50)
    
    for i in range(env.size):
        for j in range(env.size):
            state_idx = i * env.size + j
            if (i, j) in env.obstacles:
                print("  OBSTACLE  ", end="")
            elif (i, j) == env.goal:
                print("    GOAL    ", end="")
            else:
                best_action = np.argmax(q_table[state_idx])
                print(f"  {env.action_names[best_action]:<8}  ", end="")
        print("\n")


def main():
    """
    Main function to run the Q-learning algorithm on the grid world environment.
    """
    print("\n=== Q-Learning in Grid World Environment ===\n")
    
    # Create environment
    env = GridWorldEnv(size=5)
    
    # Visualize initial environment
    print("Initial Environment:")
    env.render()
    
    # Run Q-learning algorithm
    print("\nTraining Q-learning agent...")
    q_table = q_learning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1, decay_rate=0.01)
    
    # Print the learned policy
    print_policy(env, q_table)
    
    # Visualize final policy
    print("\nFinal Policy:")
    env.render(q_table)
    
    # Test the learned policy
    print("\nTesting learned policy...")
    state = env.reset()
    env.render(q_table)
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        state_idx = state[0] * env.size + state[1]
        action = np.argmax(q_table[state_idx])
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"Step {steps}: Action = {env.action_names[action]}, New State = {state}, Reward = {reward:.2f}")
        time.sleep(0.5)  # Pause to observe the agent's behavior
        env.render(q_table)
        
        if steps > 100:  # Safety check to prevent infinite loops
            print("Maximum steps reached. Stopping.")
            break
    
    print(f"\nTest completed: Total Reward = {total_reward:.2f}, Steps = {steps}")


if __name__ == "__main__":
    main()
