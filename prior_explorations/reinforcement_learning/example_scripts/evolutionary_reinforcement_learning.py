#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evolutionary Reinforcement Learning Implementation

This script demonstrates an evolutionary approach to reinforcement learning, where a population
of models is evolved over generations. The implementation includes model generation, training,
filtering, improvement, and testing with model persistence.

Before running, make sure to:
1. Activate the project venv at the project root
2. Install the required packages: pip install numpy matplotlib torch gym pickle
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import random
import os
import pickle
import time
from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    """
    Neural network for the policy in reinforcement learning.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [64, 64]):
        """
        Initialize the policy network.
        
        Args:
            input_dim (int): Dimension of the input (state)
            output_dim (int): Dimension of the output (action)
            hidden_dims (List[int]): Dimensions of the hidden layers
        """
        super(PolicyNetwork, self).__init__()
        
        # Build the network architecture dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # For discrete action spaces, we'll use Softmax
        # For continuous action spaces, we'd need a different approach
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor (state)
            
        Returns:
            torch.Tensor: Output tensor (action probabilities)
        """
        return self.network(x)


class Agent:
    """
    Agent that uses a policy network to interact with an environment.
    """
    
    def __init__(self, policy_network: PolicyNetwork):
        """
        Initialize the agent.
        
        Args:
            policy_network (PolicyNetwork): The policy network
        """
        self.policy_network = policy_network
        self.fitness = 0.0  # Initialize fitness score
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action based on the current state.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            int: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
        
        # Sample action from the probability distribution
        action = torch.multinomial(action_probs, 1).item()
        return action
    
    def evaluate(self, env, n_episodes: int = 5) -> float:
        """
        Evaluate the agent's performance in the environment.
        
        Args:
            env: The environment
            n_episodes (int): Number of episodes to evaluate
            
        Returns:
            float: Average reward over n_episodes
        """
        total_rewards = []
        
        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        self.fitness = avg_reward  # Update fitness score
        return avg_reward


class EvolutionaryRL:
    """
    Implementation of Evolutionary Reinforcement Learning.
    """
    
    def __init__(self, env_name: str, population_size: int = 50, elite_size: int = 5,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.5,
                 hidden_dims: List[int] = [64, 64], model_dir: str = "models"):
        """
        Initialize the Evolutionary RL system.
        
        Args:
            env_name (str): Name of the environment
            population_size (int): Size of the population
            elite_size (int): Number of elite individuals to keep
            mutation_rate (float): Probability of mutation
            crossover_rate (float): Probability of crossover
            hidden_dims (List[int]): Dimensions of hidden layers in the policy network
            model_dir (str): Directory to save models
        """
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.hidden_dims = hidden_dims
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Get state and action dimensions from the environment
        self.state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.is_discrete = True
        else:
            self.action_dim = self.env.action_space.shape[0]
            self.is_discrete = False
        
        # Initialize population
        self.population = self.initialize_population()
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def initialize_population(self) -> List[Agent]:
        """
        Initialize a population of agents with random policy networks.
        
        Returns:
            List[Agent]: Population of agents
        """
        population = []
        
        for _ in range(self.population_size):
            # Create a policy network with random weights
            policy_network = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(device)
            
            # Initialize with random weights
            for param in policy_network.parameters():
                param.data.normal_(0, 0.1)  # Initialize with small random values
            
            # Create an agent with this policy network
            agent = Agent(policy_network)
            population.append(agent)
        
        return population
    
    def evaluate_population(self, n_episodes: int = 3) -> None:
        """
        Evaluate all agents in the population.
        
        Args:
            n_episodes (int): Number of episodes to evaluate each agent
        """
        for i, agent in enumerate(self.population):
            fitness = agent.evaluate(self.env, n_episodes)
            print(f"Agent {i+1}/{self.population_size}: Fitness = {fitness:.2f}")
    
    def select_parents(self, n_parents: int) -> List[Agent]:
        """
        Select parents for reproduction using tournament selection.
        
        Args:
            n_parents (int): Number of parents to select
            
        Returns:
            List[Agent]: Selected parents
        """
        parents = []
        tournament_size = max(3, self.population_size // 10)
        
        for _ in range(n_parents):
            # Randomly select individuals for the tournament
            tournament = random.sample(self.population, tournament_size)
            
            # Select the best individual from the tournament
            winner = max(tournament, key=lambda agent: agent.fitness)
            parents.append(winner)
        
        return parents
    
    def crossover(self, parent1: Agent, parent2: Agent) -> Agent:
        """
        Perform crossover between two parents to create a child.
        
        Args:
            parent1 (Agent): First parent
            parent2 (Agent): Second parent
            
        Returns:
            Agent: Child agent
        """
        # Create a new policy network for the child
        child_network = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(device)
        
        # Get the state dictionaries of the parents' networks
        parent1_dict = parent1.policy_network.state_dict()
        parent2_dict = parent2.policy_network.state_dict()
        child_dict = child_network.state_dict()
        
        # Perform crossover for each parameter
        for param_name in child_dict:
            # Randomly choose whether to inherit from parent1 or parent2
            if random.random() < 0.5:
                child_dict[param_name] = parent1_dict[param_name].clone()
            else:
                child_dict[param_name] = parent2_dict[param_name].clone()
        
        # Load the new state dictionary into the child's network
        child_network.load_state_dict(child_dict)
        
        # Create and return a new agent with the child network
        return Agent(child_network)
    
    def mutate(self, agent: Agent) -> Agent:
        """
        Mutate an agent's policy network.
        
        Args:
            agent (Agent): Agent to mutate
            
        Returns:
            Agent: Mutated agent
        """
        # Create a deep copy of the agent to avoid modifying the original
        mutated_agent = Agent(deepcopy(agent.policy_network))
        
        # Get the state dictionary of the network
        state_dict = mutated_agent.policy_network.state_dict()
        
        # Mutate each parameter with probability mutation_rate
        for param_name in state_dict:
            if random.random() < self.mutation_rate:
                # Add Gaussian noise to the parameter
                noise = torch.randn_like(state_dict[param_name]) * 0.1
                state_dict[param_name] += noise
        
        # Load the mutated state dictionary back into the network
        mutated_agent.policy_network.load_state_dict(state_dict)
        
        return mutated_agent
    
    def evolve(self) -> None:
        """
        Evolve the population to the next generation.
        """
        # Sort the population by fitness in descending order
        sorted_population = sorted(self.population, key=lambda agent: agent.fitness, reverse=True)
        
        # Keep the elite individuals
        elite = sorted_population[:self.elite_size]
        
        # Create the new population starting with the elite
        new_population = elite.copy()
        
        # Fill the rest of the population with children from crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents
            parents = self.select_parents(2)
            
            # Perform crossover with probability crossover_rate
            if random.random() < self.crossover_rate:
                child = self.crossover(parents[0], parents[1])
            else:
                # If no crossover, just clone one of the parents
                child = Agent(deepcopy(random.choice(parents).policy_network))
            
            # Perform mutation with probability mutation_rate
            child = self.mutate(child)
            
            # Add the child to the new population
            new_population.append(child)
        
        # Update the population
        self.population = new_population
        self.generation += 1
    
    def train(self, n_generations: int, eval_episodes: int = 3) -> None:
        """
        Train the population for a specified number of generations.
        
        Args:
            n_generations (int): Number of generations to train
            eval_episodes (int): Number of episodes to evaluate each agent
        """
        for gen in range(n_generations):
            print(f"\nGeneration {self.generation + 1}/{self.generation + n_generations}:")
            
            # Evaluate the population
            self.evaluate_population(eval_episodes)
            
            # Record statistics
            fitnesses = [agent.fitness for agent in self.population]
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            print(f"Best Fitness: {best_fitness:.2f}, Average Fitness: {avg_fitness:.2f}")
            
            # Save the best model of this generation
            best_agent = max(self.population, key=lambda agent: agent.fitness)
            self.save_model(best_agent, f"gen_{self.generation}_best")
            
            # Evolve to the next generation (except for the last iteration)
            if gen < n_generations - 1:
                self.evolve()
    
    def save_model(self, agent: Agent, name: str) -> None:
        """
        Save an agent's policy network to disk.
        
        Args:
            agent (Agent): Agent to save
            name (str): Name for the saved model
        """
        model_path = os.path.join(self.model_dir, f"{name}.pkl")
        
        # Save the model state dictionary
        torch.save(agent.policy_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, name: str) -> Agent:
        """
        Load an agent's policy network from disk.
        
        Args:
            name (str): Name of the saved model
            
        Returns:
            Agent: Agent with the loaded policy network
        """
        model_path = os.path.join(self.model_dir, f"{name}.pkl")
        
        # Create a new policy network
        policy_network = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(device)
        
        # Load the state dictionary
        policy_network.load_state_dict(torch.load(model_path))
        
        # Create and return an agent with the loaded network
        return Agent(policy_network)
    
    def test_best_agent(self, render: bool = True, n_episodes: int = 5) -> None:
        """
        Test the best agent in the population.
        
        Args:
            render (bool): Whether to render the environment
            n_episodes (int): Number of episodes to test
        """
        # Find the best agent
        best_agent = max(self.population, key=lambda agent: agent.fitness)
        
        # Create a test environment with rendering if specified
        if render:
            test_env = gym.make(self.env_name, render_mode='human')
        else:
            test_env = self.env
        
        for episode in range(n_episodes):
            state, _ = test_env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            print(f"\nTesting Episode {episode + 1}/{n_episodes}:")
            
            while not done:
                action = best_agent.select_action(state)
                next_state, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
                step += 1
                
                if render:
                    time.sleep(0.01)  # Slow down rendering
                    print(f"Step {step}: Action = {action}, Reward = {reward:.2f}")
            
            print(f"Episode {episode + 1} Reward: {episode_reward:.2f}")
        
        if render and test_env != self.env:
            test_env.close()
    
    def plot_training_history(self) -> None:
        """
        Plot the training history (best and average fitness over generations).
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.generation + 1), self.best_fitness_history, label='Best Fitness')
        plt.plot(range(1, self.generation + 1), self.avg_fitness_history, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Average Reward)')
        plt.title('Evolutionary Reinforcement Learning Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_training_state(self, filename: str = "erl_training_state") -> None:
        """
        Save the entire training state to disk.
        
        Args:
            filename (str): Name for the saved state
        """
        state_path = os.path.join(self.model_dir, f"{filename}.pkl")
        
        # Create a dictionary with all the necessary information
        state = {
            'generation': self.generation,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'population': [agent.policy_network.state_dict() for agent in self.population],
            'env_name': self.env_name,
            'population_size': self.population_size,
            'elite_size': self.elite_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'hidden_dims': self.hidden_dims
        }
        
        # Save the state
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Training state saved to {state_path}")
    
    def load_training_state(self, filename: str = "erl_training_state") -> None:
        """
        Load the entire training state from disk.
        
        Args:
            filename (str): Name of the saved state
        """
        state_path = os.path.join(self.model_dir, f"{filename}.pkl")
        
        # Load the state
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        
        # Restore the environment if it's different
        if state['env_name'] != self.env_name:
            self.env_name = state['env_name']
            self.env = gym.make(self.env_name)
            
            # Update state and action dimensions
            self.state_dim = self.env.observation_space.shape[0]
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                self.action_dim = self.env.action_space.n
                self.is_discrete = True
            else:
                self.action_dim = self.env.action_space.shape[0]
                self.is_discrete = False
        
        # Restore other parameters
        self.population_size = state['population_size']
        self.elite_size = state['elite_size']
        self.mutation_rate = state['mutation_rate']
        self.crossover_rate = state['crossover_rate']
        self.hidden_dims = state['hidden_dims']
        
        # Restore training history
        self.generation = state['generation']
        self.best_fitness_history = state['best_fitness_history']
        self.avg_fitness_history = state['avg_fitness_history']
        
        # Restore population
        self.population = []
        for policy_state_dict in state['population']:
            policy_network = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(device)
            policy_network.load_state_dict(policy_state_dict)
            self.population.append(Agent(policy_network))
        
        print(f"Training state loaded from {state_path}")
        print(f"Restored at generation {self.generation} with population size {len(self.population)}")


def main():
    """
    Main function to run the Evolutionary Reinforcement Learning algorithm.
    """
    print("\n=== Evolutionary Reinforcement Learning Implementation ===\n")
    
    # Create the ERL system
    env_name = "CartPole-v1"
    erl = EvolutionaryRL(
        env_name=env_name,
        population_size=50,
        elite_size=5,
        mutation_rate=0.1,
        crossover_rate=0.7,
        hidden_dims=[32, 32],
        model_dir="erl_models"
    )
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {erl.state_dim}, Action dimension: {erl.action_dim}")
    print(f"Population size: {erl.population_size}, Elite size: {erl.elite_size}")
    print(f"Mutation rate: {erl.mutation_rate}, Crossover rate: {erl.crossover_rate}")
    
    # Train for a few generations
    print("\nTraining the population...")
    erl.train(n_generations=10, eval_episodes=3)
    
    # Plot the training history
    erl.plot_training_history()
    
    # Save the training state
    erl.save_training_state()
    
    # Test the best agent
    print("\nTesting the best agent...")
    erl.test_best_agent(render=True, n_episodes=3)
    
    # Example of resuming training from a saved state
    print("\nSaving and loading training state demonstration:")
    
    # Create a new ERL instance
    new_erl = EvolutionaryRL(
        env_name=env_name,
        population_size=50,  # These parameters will be overridden by the loaded state
        elite_size=5,
        mutation_rate=0.1,
        crossover_rate=0.7,
        hidden_dims=[32, 32],
        model_dir="erl_models"
    )
    
    # Load the saved training state
    new_erl.load_training_state()
    
    # Continue training for a few more generations
    print("\nContinuing training from the loaded state...")
    new_erl.train(n_generations=5, eval_episodes=3)
    
    # Plot the updated training history
    new_erl.plot_training_history()
    
    # Test the best agent after additional training
    print("\nTesting the best agent after additional training...")
    new_erl.test_best_agent(render=True, n_episodes=3)
    
    # Close the environment
    erl.env.close()


if __name__ == "__main__":
    main()
