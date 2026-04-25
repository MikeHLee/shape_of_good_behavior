"""
METHODOLOGICAL VERIFICATION: Detecting Cyclic Preferences via H¹ Cohomology

Theoretical Background:
The Condorcet Paradox occurs when preferences are cyclic: A > B > C > A.
In reinforcement learning, this manifests when the reward for traversing a loop
is always positive, creating a contradiction for scalar value functions.

Key Insight:
On a circle S¹, if reward r(θ) is always positive for clockwise motion:
  - The path integral ∮ r(θ) dθ ≠ 0
  - This is the H¹ cohomology obstruction
  - No smooth scalar function V(θ) can satisfy V'(θ) = r(θ) AND V(0) = V(2π)

Hodge Decomposition:
Any 1-form (like reward) on S¹ decomposes as:
  r = dV + ω
where:
  - dV = exact part (gradient of periodic potential V)
  - ω = harmonic part (constant on S¹, captures H¹)

The harmonic coefficient ω = (1/2π) ∮ r dθ is precisely the H¹ invariant.

This script serves as a UNIT TEST for the HodgeCritic's ability to detect
these topological obstructions in a controlled setting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
import json
import os
from pathlib import Path

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "condorcet"
FIGURES_DIR = SCRIPT_DIR.parent / "figures" / "experiments"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. ENVIRONMENT: The Condorcet Ring
# ============================================================================

class CondorcetRingEnv:
    """
    Continuous circular state space representing cyclic preferences.
    
    State: Angle θ ∈ [-π, π] on the unit circle
    Action: Angular velocity v ∈ [-1, 1]
    Reward: Positive for clockwise motion (positive velocity)
    """
    def __init__(self, base_reward=0.5, noise_std=0.1):
        self.theta = 0.0
        self.max_steps = 100
        self.dt = 0.1
        self.base_reward = base_reward
        self.noise_std = noise_std
        self.step_count = 0

    def reset(self):
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array([np.sin(self.theta), np.cos(self.theta)], dtype=np.float32)

    def step(self, action):
        velocity = float(np.clip(action, -1.0, 1.0))
        delta_theta = velocity * self.dt
        self.theta += delta_theta
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        
        reward = self.base_reward * velocity + np.random.normal(0, self.noise_std)
        
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        return self._get_obs(), reward, done, {'theta': self.theta, 'velocity': velocity}

    def compute_h1_ground_truth(self):
        return self.base_reward

# ============================================================================
# 2. BASELINE: PPO with Scalar Value Function
# ============================================================================

class ScalarCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================================
# 3. PROPOSED: SGPO with Hodge Decomposition
# ============================================================================

class HodgeCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.potential_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.harmonic_coeff = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        potential = self.potential_net(x)
        harmonic = self.harmonic_coeff.expand(x.shape[0], 1)
        return potential, harmonic
    
    def predict_reward(self, state, next_state, velocity):
        V_curr, _ = self.forward(state)
        V_next, omega = self.forward(next_state)
        dV = V_next - V_curr
        harmonic_contribution = omega * velocity.unsqueeze(-1)
        return dV + harmonic_contribution

# ============================================================================
# 4. SHARED: Actor Policy
# ============================================================================

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1) - 0.5)

    def forward(self, x):
        mu = self.net(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

# ============================================================================
# 5. TRAINING FUNCTIONS
# ============================================================================

def train_ppo(env, episodes=500, gamma=0.99):
    actor = Actor()
    critic = ScalarCritic()
    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)

    returns_history = []
    value_losses = []

    for ep in range(episodes):
        obs = env.reset()
        trajectory = []
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
            
            next_obs, reward, done, info = env.step(action.item())
            trajectory.append((obs, action.item(), reward, info['velocity']))
            obs = next_obs

        returns = []
        G = 0
        for _, _, r, _ in reversed(trajectory):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        states = torch.FloatTensor(np.array([t[0] for t in trajectory]))
        actions = torch.FloatTensor([t[1] for t in trajectory])

        values = critic(states).squeeze()
        critic_loss = nn.MSELoss()(values, returns)
        
        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()
        value_losses.append(critic_loss.item())

        with torch.no_grad():
            baselines = critic(states).squeeze()
        advantages = returns - baselines

        dists = actor(states)
        log_probs = dists.log_prob(actions.unsqueeze(1)).squeeze()
        actor_loss = -(log_probs * advantages).mean()

        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()

        returns_history.append(sum([t[2] for t in trajectory]))

    return actor, critic, returns_history, value_losses

def train_gpo(env, episodes=500, gamma=0.99):
    actor = Actor()
    critic = HodgeCritic()
    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)

    returns_history = []
    harmonic_history = []

    for ep in range(episodes):
        obs = env.reset()
        trajectory = []
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
            
            next_obs, reward, done, info = env.step(action.item())
            trajectory.append((obs, next_obs, action.item(), reward, info['velocity']))
            obs = next_obs

        states = torch.FloatTensor(np.array([t[0] for t in trajectory]))
        next_states = torch.FloatTensor(np.array([t[1] for t in trajectory]))
        actions = torch.FloatTensor([t[2] for t in trajectory])
        rewards = torch.FloatTensor([t[3] for t in trajectory])
        velocities = torch.FloatTensor([t[4] for t in trajectory])

        predicted_rewards = critic.predict_reward(states, next_states, velocities).squeeze()
        critic_loss = nn.MSELoss()(predicted_rewards, rewards)
        
        V_curr, _ = critic(states)
        orthogonality_loss = 0.01 * V_curr.mean().pow(2)
        total_critic_loss = critic_loss + orthogonality_loss
        
        opt_critic.zero_grad()
        total_critic_loss.backward()
        opt_critic.step()

        with torch.no_grad():
            V_curr, omega = critic(states)
            V_next, _ = critic(next_states)
            expected_reward = (V_next - V_curr).squeeze() + omega.squeeze() * velocities
            advantages = rewards - expected_reward

        dists = actor(states)
        log_probs = dists.log_prob(actions.unsqueeze(1)).squeeze()
        actor_loss = -(log_probs * advantages).mean()

        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()

        returns_history.append(rewards.sum().item())
        harmonic_history.append(critic.harmonic_coeff.item())

    return actor, critic, returns_history, harmonic_history

def compute_empirical_h1(env, policy, num_cycles=10):
    total_reward = 0
    total_angular_motion = 0
    
    for _ in range(num_cycles):
        obs = env.reset()
        done = False
        cycle_reward = 0
        cycle_motion = 0
        
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist = policy(obs_t)
                action = dist.sample()
            
            obs, reward, done, info = env.step(action.item())
            cycle_reward += reward
            cycle_motion += abs(info['velocity'] * env.dt)
        
        total_reward += cycle_reward
        total_angular_motion += cycle_motion
    
    # H1 is the harmonic coefficient ω such that Reward ≈ ω * velocity
    # Sum(Reward) ≈ ω * Sum(velocity)
    # Sum(velocity) = Sum(velocity * dt) / dt = total_angular_motion / dt
    # Therefore: ω ≈ Sum(Reward) * dt / total_angular_motion
    h1_empirical = (total_reward * env.dt) / total_angular_motion
    return h1_empirical

# ============================================================================
# 6. MAIN EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("METHODOLOGICAL VERIFICATION: CONDORCET CYCLE")
    print("Verifying H¹ Cohomology Detection on S¹ Topology")
    print("=" * 60)
    
    env = CondorcetRingEnv(base_reward=0.5, noise_std=0.05)
    h1_ground_truth = env.compute_h1_ground_truth()
    print(f"Ground Truth H¹ = {h1_ground_truth:.4f}")
    
    print("\nTraining PPO (Scalar Value Function)...")
    ppo_actor, ppo_critic, ppo_returns, ppo_losses = train_ppo(env, episodes=400)
    
    print("Training SGPO (Hodge Decomposition)...")
    gpo_actor, gpo_critic, gpo_returns, gpo_harmonic = train_gpo(env, episodes=400)
    
    print("\nComputing Empirical H¹...")
    h1_ppo = compute_empirical_h1(env, ppo_actor)
    h1_gpo = compute_empirical_h1(env, gpo_actor)
    h1_learned = gpo_critic.harmonic_coeff.item()
    
    print(f"\nH¹ Results:")
    print(f"  Ground Truth:     {h1_ground_truth:.4f}")
    print(f"  Learned (SGPO ω):  {h1_learned:.4f}")
    print(f"  Empirical (PPO):  {h1_ppo:.4f}")
    print(f"  Empirical (SGPO):  {h1_gpo:.4f}")

    metrics_data = {
        "ground_truth_h1": float(h1_ground_truth),
        "learned_h1": float(h1_learned),
        "empirical_h1_ppo": float(h1_ppo),
        "empirical_h1_gpo": float(h1_gpo),
        "ppo_returns": ppo_returns,
        "gpo_returns": gpo_returns,
        "gpo_harmonic_history": gpo_harmonic,
        "ppo_loss_history": ppo_losses
    }
    
    with open(RESULTS_DIR / 'condorcet_metrics.json', 'w') as f:
        json.dump(metrics_data, f)
    print(f"Metrics saved to {RESULTS_DIR / 'condorcet_metrics.json'}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Learning Curves
    ax1 = axes[0, 0]
    window = 20
    ppo_smooth = np.convolve(ppo_returns, np.ones(window)/window, mode='valid')
    gpo_smooth = np.convolve(gpo_returns, np.ones(window)/window, mode='valid')
    ax1.plot(ppo_smooth, label='PPO (Scalar)', color='red', alpha=0.8)
    ax1.plot(gpo_smooth, label='SGPO (Hodge)', color='blue', alpha=0.8)
    ax1.set_title('Learning Curves')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Harmonic Coefficient Evolution
    ax2 = axes[0, 1]
    ax2.plot(gpo_harmonic, color='green', label='Learned ω')
    ax2.axhline(y=h1_ground_truth, color='black', linestyle='--', label=f'True H¹ = {h1_ground_truth}')
    ax2.set_title('Harmonic Coefficient (H¹) Learning')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('ω')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Value Functions
    ax3 = axes[0, 2]
    theta_range = np.linspace(-np.pi, np.pi, 100)
    obs_batch = torch.FloatTensor(np.stack([np.sin(theta_range), np.cos(theta_range)], axis=1))
    
    with torch.no_grad():
        ppo_values = ppo_critic(obs_batch).numpy().flatten()
        gpo_potentials, gpo_omegas = gpo_critic(obs_batch)
        gpo_potentials = gpo_potentials.numpy().flatten()
    
    ax3.plot(theta_range, ppo_values, color='red', label='PPO V(θ)', linewidth=2)
    ax3.plot(theta_range, gpo_potentials, color='blue', linestyle='--', label='SGPO V(θ) (potential)', linewidth=2)
    ax3.set_title('Value Functions on S¹')
    ax3.set_xlabel('θ')
    ax3.set_ylabel('V(θ)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: PPO Critic Loss
    ax4 = axes[1, 0]
    ax4.plot(ppo_losses, color='red', alpha=0.7)
    ax4.set_title('PPO Critic Loss')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('MSE Loss')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Policy Comparison
    ax5 = axes[1, 1]
    with torch.no_grad():
        ppo_actions = ppo_actor(obs_batch).mean.numpy().flatten()
        gpo_actions = gpo_actor(obs_batch).mean.numpy().flatten()
    
    ax5.plot(theta_range, ppo_actions, color='red', label='PPO π(θ)', linewidth=2)
    ax5.plot(theta_range, gpo_actions, color='blue', label='SGPO π(θ)', linewidth=2)
    ax5.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax5.set_title('Policy (Mean Action) by State')
    ax5.set_xlabel('θ')
    ax5.set_ylabel('Mean Velocity')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: H¹ Summary
    ax6 = axes[1, 2]
    methods = ['Ground\nTruth', 'Learned\n(SGPO)', 'PPO', 'SGPO']
    values = [h1_ground_truth, h1_learned, h1_ppo, h1_gpo]
    colors = ['black', 'green', 'red', 'blue']
    ax6.bar(methods, values, color=colors, alpha=0.7)
    ax6.axhline(y=h1_ground_truth, color='black', linestyle='--', alpha=0.5)
    ax6.set_title('H¹ Cohomology Comparison')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'condorcet_results.png', dpi=150)
    print(f"Plot saved to {FIGURES_DIR / 'condorcet_results.png'}")
