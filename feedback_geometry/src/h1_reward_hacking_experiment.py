"""
H¹-Exploitable Reward Hacking Experiment

Core Thesis: When human feedback contains cyclic inconsistencies (H¹ ≠ 0),
standard RLHF reward models create exploitable gaps that enable reward hacking.

This experiment demonstrates:
1. Cyclic preferences in feedback data create H¹ ≠ 0
2. Standard reward models trained on this data have exploitable regions
3. Agents can "hack" the reward by entering these cyclic regions
4. Hodge-filtered training prevents exploitation

Environment: Navigation with a "preference trap"
- Region A: Safe path (moderate reward)
- Region B: Trap path (cyclic preferences - some evaluators love it, some hate it)
- The H¹ component of feedback on Region B allows reward hacking
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt

# Import experiment framework
from experiment_framework import (
    ExperimentConfig, SeedResult, ExperimentRunner,
    compute_statistics, compare_methods, MethodComparison
)

# ============================================================================
# 1. PREFERENCE DATA WITH CONTROLLED H¹
# ============================================================================

@dataclass
class PreferencePair:
    """A single preference comparison."""
    state: np.ndarray
    response_a: int  # Action index
    response_b: int
    preference: float  # 1.0 = prefer A, 0.0 = prefer B, 0.5 = tie
    evaluator_id: int


class PreferenceDataset:
    """
    Generates preference data with controlled H¹ (cyclic inconsistency).
    
    The key insight: In certain regions (the "trap"), evaluators disagree
    cyclically. Some prefer action sequence A > B > C > A.
    This creates H¹ ≠ 0 in that region's preference graph.
    """
    
    def __init__(
        self,
        n_evaluators: int = 5,
        n_actions: int = 4,
        trap_center: np.ndarray = np.array([5.0, 5.0]),
        trap_radius: float = 2.0,
        h1_magnitude: float = 0.5,  # Controls strength of cycling
        noise_std: float = 0.1
    ):
        self.n_evaluators = n_evaluators
        self.n_actions = n_actions
        self.trap_center = trap_center
        self.trap_radius = trap_radius
        self.h1_magnitude = h1_magnitude
        self.noise_std = noise_std
        
        # Ground truth preferences (outside trap)
        # Action 0 > 1 > 2 > 3 (consistent ranking)
        self.base_rewards = np.array([1.0, 0.75, 0.5, 0.25])
        
        # Per-evaluator "bias" that creates cycling in trap region
        # Evaluator i prefers action (i % n_actions) over others
        self.evaluator_biases = np.zeros((n_evaluators, n_actions))
        for i in range(n_evaluators):
            favorite = i % n_actions
            self.evaluator_biases[i, favorite] = h1_magnitude
    
    def in_trap(self, state: np.ndarray) -> bool:
        """Check if state is in the trap region."""
        return np.linalg.norm(state - self.trap_center) < self.trap_radius
    
    def get_true_reward(self, state: np.ndarray, action: int) -> float:
        """Ground truth reward (unknown to the agent)."""
        if self.in_trap(state):
            # Trap region: looks good but is actually bad
            return -1.0  # Catastrophic
        else:
            return self.base_rewards[action]
    
    def get_evaluator_preference(
        self,
        state: np.ndarray,
        action_a: int,
        action_b: int,
        evaluator_id: int
    ) -> float:
        """
        Simulates a human evaluator's preference.
        
        Outside trap: Consistent preferences based on base_rewards
        Inside trap: Cyclic preferences due to evaluator biases
        """
        if self.in_trap(state):
            # In trap: evaluator-specific biases create cycling
            score_a = self.base_rewards[action_a] + self.evaluator_biases[evaluator_id, action_a]
            score_b = self.base_rewards[action_b] + self.evaluator_biases[evaluator_id, action_b]
        else:
            # Outside trap: consistent preferences
            score_a = self.base_rewards[action_a]
            score_b = self.base_rewards[action_b]
        
        # Add noise
        score_a += np.random.normal(0, self.noise_std)
        score_b += np.random.normal(0, self.noise_std)
        
        # Bradley-Terry probability
        prob_a = 1.0 / (1.0 + np.exp(-(score_a - score_b)))
        return prob_a
    
    def generate_dataset(
        self,
        n_pairs: int = 1000,
        trap_fraction: float = 0.3
    ) -> List[PreferencePair]:
        """Generate preference dataset with controlled trap fraction."""
        pairs = []
        
        n_trap = int(n_pairs * trap_fraction)
        n_safe = n_pairs - n_trap
        
        # Generate safe region pairs
        for _ in range(n_safe):
            # Random state outside trap
            while True:
                state = np.random.uniform(0, 10, size=2)
                if not self.in_trap(state):
                    break
            
            action_a = np.random.randint(self.n_actions)
            action_b = np.random.randint(self.n_actions)
            while action_b == action_a:
                action_b = np.random.randint(self.n_actions)
            
            evaluator = np.random.randint(self.n_evaluators)
            pref = self.get_evaluator_preference(state, action_a, action_b, evaluator)
            
            pairs.append(PreferencePair(state, action_a, action_b, pref, evaluator))
        
        # Generate trap region pairs
        for _ in range(n_trap):
            # Random state inside trap
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, self.trap_radius)
            state = self.trap_center + radius * np.array([np.cos(angle), np.sin(angle)])
            
            action_a = np.random.randint(self.n_actions)
            action_b = np.random.randint(self.n_actions)
            while action_b == action_a:
                action_b = np.random.randint(self.n_actions)
            
            evaluator = np.random.randint(self.n_evaluators)
            pref = self.get_evaluator_preference(state, action_a, action_b, evaluator)
            
            pairs.append(PreferencePair(state, action_a, action_b, pref, evaluator))
        
        np.random.shuffle(pairs)
        return pairs
    
    def compute_h1(self, pairs: List[PreferencePair], region: str = "all") -> float:
        """
        Compute H¹ magnitude for a set of preference pairs.
        
        H¹ is computed via Hodge decomposition of the preference graph.
        Non-zero H¹ indicates cyclic preferences that cannot be represented
        by a scalar reward function.
        """
        # Filter pairs by region
        if region == "trap":
            pairs = [p for p in pairs if self.in_trap(p.state)]
        elif region == "safe":
            pairs = [p for p in pairs if not self.in_trap(p.state)]
        
        if len(pairs) < 10:
            return 0.0
        
        # Build preference graph
        # Nodes: actions, Edges: comparisons, Weights: preference strengths
        n = self.n_actions
        edge_weights = np.zeros((n, n))
        edge_counts = np.zeros((n, n))
        
        for p in pairs:
            a, b = p.response_a, p.response_b
            weight = 2 * p.preference - 1  # Convert to [-1, 1]
            edge_weights[a, b] += weight
            edge_weights[b, a] -= weight
            edge_counts[a, b] += 1
            edge_counts[b, a] += 1
        
        # Average weights
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_weights = np.where(edge_counts > 0, edge_weights / edge_counts, 0)
        
        # Hodge decomposition via graph Laplacian
        # The harmonic component is what remains after projecting out the gradient
        
        # Build incidence matrix (edges × vertices)
        edges = []
        weights = []
        for i in range(n):
            for j in range(i+1, n):
                if edge_counts[i, j] > 0:
                    edges.append((i, j, avg_weights[i, j]))
        
        if len(edges) < 3:
            return 0.0
        
        m = len(edges)
        B = np.zeros((m, n))  # Incidence matrix
        w = np.zeros(m)       # Edge weights
        
        for idx, (i, j, weight) in enumerate(edges):
            B[idx, i] = 1
            B[idx, j] = -1
            w[idx] = weight
        
        # Hodge decomposition: w = B @ potential + harmonic
        # potential = (B^T B)^+ B^T w
        # harmonic = w - B @ potential
        
        BtB = B.T @ B
        # Add regularization for pseudo-inverse
        BtB_reg = BtB + 1e-6 * np.eye(n)
        potential = np.linalg.solve(BtB_reg, B.T @ w)
        
        exact_component = B @ potential
        harmonic_component = w - exact_component
        
        # H¹ magnitude is the norm of the harmonic component
        h1 = np.linalg.norm(harmonic_component) / np.sqrt(m)
        
        return float(h1)


# ============================================================================
# 2. REWARD MODELS
# ============================================================================

class StandardRewardModel(nn.Module):
    """Standard RLHF reward model (scalar output)."""
    
    def __init__(self, state_dim: int = 2, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_actions, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.n_actions = n_actions
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # One-hot encode action
        action_onehot = torch.zeros(state.shape[0], self.n_actions, device=state.device)
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        x = torch.cat([state, action_onehot], dim=1)
        return self.net(x)
    
    def get_reward(self, state: np.ndarray, action: int) -> float:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_t = torch.LongTensor([action])
            return self.forward(state_t, action_t).item()


class HodgeFilteredRewardModel(nn.Module):
    """
    Reward model trained on Hodge-filtered preferences.
    
    Before training, we apply Hodge decomposition to the preference data
    and train only on the exact (gradient) component, discarding the
    harmonic (cyclic) component.
    """
    
    def __init__(self, state_dim: int = 2, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_actions, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.n_actions = n_actions
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_onehot = torch.zeros(state.shape[0], self.n_actions, device=state.device)
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        x = torch.cat([state, action_onehot], dim=1)
        return self.net(x)
    
    def get_reward(self, state: np.ndarray, action: int) -> float:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_t = torch.LongTensor([action])
            return self.forward(state_t, action_t).item()


def train_reward_model(
    model: nn.Module,
    pairs: List[PreferencePair],
    epochs: int = 100,
    lr: float = 1e-3,
    hodge_filter: bool = False
) -> List[float]:
    """Train reward model on preference pairs."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    # Convert to tensors
    states = torch.FloatTensor(np.array([p.state for p in pairs]))
    actions_a = torch.LongTensor([p.response_a for p in pairs])
    actions_b = torch.LongTensor([p.response_b for p in pairs])
    preferences = torch.FloatTensor([p.preference for p in pairs])
    
    # If Hodge filtering, adjust preferences to remove harmonic component
    if hodge_filter:
        preferences = _hodge_filter_preferences(pairs, preferences)
    
    for epoch in range(epochs):
        # Bradley-Terry loss
        r_a = model(states, actions_a).squeeze()
        r_b = model(states, actions_b).squeeze()
        
        # P(a > b) = sigmoid(r_a - r_b)
        logits = r_a - r_b
        loss = nn.BCEWithLogitsLoss()(logits, preferences)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return losses


def _hodge_filter_preferences(
    pairs: List[PreferencePair],
    preferences: torch.Tensor
) -> torch.Tensor:
    """
    Apply Hodge decomposition to filter out cyclic component.
    Returns adjusted preferences that are representable by a scalar potential.
    """
    # Group pairs by state region
    # For each region, compute local Hodge decomposition
    # Project preferences onto exact subspace
    
    # Simplified version: compute global preference graph and project
    n_actions = max(max(p.response_a, p.response_b) for p in pairs) + 1
    n_pairs = len(pairs)
    
    # Build preference matrix
    pref_matrix = torch.zeros(n_actions, n_actions)
    count_matrix = torch.zeros(n_actions, n_actions)
    
    for i, p in enumerate(pairs):
        a, b = p.response_a, p.response_b
        w = 2 * preferences[i] - 1  # Convert to [-1, 1]
        pref_matrix[a, b] += w
        pref_matrix[b, a] -= w
        count_matrix[a, b] += 1
        count_matrix[b, a] += 1
    
    # Average
    avg_pref = pref_matrix / (count_matrix + 1e-6)
    
    # Compute potential via least squares
    # potential[i] - potential[j] ≈ avg_pref[i, j]
    edges = []
    for i in range(n_actions):
        for j in range(i+1, n_actions):
            if count_matrix[i, j] > 0:
                edges.append((i, j, avg_pref[i, j].item()))
    
    if len(edges) < 2:
        return preferences
    
    m = len(edges)
    B = np.zeros((m, n_actions))
    w = np.zeros(m)
    
    for idx, (i, j, weight) in enumerate(edges):
        B[idx, i] = 1
        B[idx, j] = -1
        w[idx] = weight
    
    # Solve for potential
    BtB = B.T @ B + 1e-6 * np.eye(n_actions)
    potential = np.linalg.solve(BtB, B.T @ w)
    
    # Project each preference onto exact component
    filtered_prefs = preferences.clone()
    for i, p in enumerate(pairs):
        a, b = p.response_a, p.response_b
        exact_diff = potential[a] - potential[b]
        # Convert back to [0, 1] probability
        filtered_prefs[i] = torch.sigmoid(torch.tensor(exact_diff))
    
    return filtered_prefs


# ============================================================================
# 3. ENVIRONMENT
# ============================================================================

class PreferenceTrapEnv:
    """
    Navigation environment with a "preference trap".
    
    The trap is a region where:
    - Human feedback is inconsistent (H¹ ≠ 0)
    - Standard reward models learn a positive reward (exploitation)
    - True outcome is catastrophic
    """
    
    def __init__(
        self,
        trap_center: np.ndarray = np.array([5.0, 5.0]),
        trap_radius: float = 2.0,
        goal: np.ndarray = np.array([9.0, 9.0]),
        max_steps: int = 50
    ):
        self.trap_center = trap_center
        self.trap_radius = trap_radius
        self.goal = goal
        self.max_steps = max_steps
        self.dt = 0.5
        
        self.state = None
        self.step_count = 0
    
    def reset(self) -> np.ndarray:
        self.state = np.array([0.5, 0.5])
        self.step_count = 0
        return self.state.copy()
    
    def in_trap(self, state: np.ndarray = None) -> bool:
        if state is None:
            state = self.state
        return np.linalg.norm(state - self.trap_center) < self.trap_radius
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Returns: (next_state, true_reward, done, info)
        Note: true_reward is the actual outcome, not what the reward model predicts
        """
        velocity = np.clip(action, -1.0, 1.0)
        self.state = self.state + velocity * self.dt
        self.state = np.clip(self.state, 0, 10)
        
        self.step_count += 1
        
        # True reward (ground truth)
        dist_to_goal = np.linalg.norm(self.state - self.goal)
        in_trap = self.in_trap()
        
        if in_trap:
            true_reward = -10.0  # Catastrophic
            done = True
        elif dist_to_goal < 1.0:
            true_reward = 10.0  # Goal reached
            done = True
        elif self.step_count >= self.max_steps:
            true_reward = -dist_to_goal / 10.0  # Penalty for not reaching goal
            done = True
        else:
            true_reward = -0.01  # Small step penalty
            done = False
        
        info = {
            'in_trap': in_trap,
            'dist_to_goal': dist_to_goal,
            'dist_to_trap': np.linalg.norm(self.state - self.trap_center) - self.trap_radius
        }
        
        return self.state.copy(), true_reward, done, info


# ============================================================================
# 4. POLICIES
# ============================================================================

class Actor(nn.Module):
    def __init__(self, state_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )
        self.log_std = nn.Parameter(torch.zeros(2) - 1.0)
    
    def forward(self, x):
        mu = self.net(x)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)


class Critic(nn.Module):
    def __init__(self, state_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# 5. TRAINING FUNCTIONS
# ============================================================================

def train_policy_with_reward_model(
    env: PreferenceTrapEnv,
    reward_model: nn.Module,
    episodes: int = 200,
    gamma: float = 0.99
) -> Tuple[Actor, List[float], List[int], List[bool]]:
    """
    Train a policy using a learned reward model.
    
    This simulates standard RLHF: train on predicted rewards, evaluate on true rewards.
    """
    actor = Actor()
    critic = Critic()
    opt_actor = optim.Adam(actor.parameters(), lr=1e-3)
    opt_critic = optim.Adam(critic.parameters(), lr=3e-3)
    
    episode_true_returns = []
    episode_violations = []
    goal_reached = []
    
    for ep in range(episodes):
        obs = env.reset()
        trajectory = []
        ep_true_return = 0.0
        ep_violations = 0
        reached_goal = False
        
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
            
            next_obs, true_reward, done, info = env.step(action.numpy())
            
            # Get predicted reward from learned model
            # Convert continuous action to discrete action index for reward model
            action_idx = int(np.clip((action.numpy()[0] + 1) * 2, 0, 3))
            predicted_reward = reward_model.get_reward(obs, action_idx)
            
            trajectory.append((obs, action, predicted_reward, true_reward))
            ep_true_return += true_reward
            ep_violations += int(info['in_trap'])
            if info['dist_to_goal'] < 1.0:
                reached_goal = True
            
            obs = next_obs
        
        # Update policy using PREDICTED rewards (this is what RLHF does)
        states = torch.FloatTensor(np.array([t[0] for t in trajectory]))
        actions = torch.stack([t[1] for t in trajectory])
        predicted_rewards = torch.FloatTensor([t[2] for t in trajectory])
        
        # Compute returns using predicted rewards
        returns = []
        G = 0
        for _, _, r, _ in reversed(trajectory):
            G = r + gamma * G  # Using predicted reward
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        
        # Update critic
        vals = critic(states)
        loss_crit = nn.MSELoss()(vals, returns)
        opt_critic.zero_grad()
        loss_crit.backward()
        opt_critic.step()
        
        # Update actor
        with torch.no_grad():
            baselines = critic(states).squeeze()
        advantages = returns.squeeze() - baselines
        
        dists = actor(states)
        log_probs = dists.log_prob(actions).sum(dim=1)
        loss_actor = -(log_probs * advantages).mean()
        
        opt_actor.zero_grad()
        loss_actor.backward()
        opt_actor.step()
        
        episode_true_returns.append(ep_true_return)
        episode_violations.append(ep_violations)
        goal_reached.append(reached_goal)
    
    return actor, episode_true_returns, episode_violations, goal_reached


# ============================================================================
# 6. MAIN EXPERIMENT
# ============================================================================

def run_h1_exploitation_experiment(
    seed: int,
    h1_magnitude: float = 0.5,
    n_preference_pairs: int = 500,
    trap_fraction: float = 0.3,
    policy_episodes: int = 200
) -> Dict:
    """Run a single seed of the H¹ exploitation experiment."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Generate preference data with controlled H¹
    pref_dataset = PreferenceDataset(
        n_evaluators=5,
        n_actions=4,
        h1_magnitude=h1_magnitude
    )
    pairs = pref_dataset.generate_dataset(n_preference_pairs, trap_fraction)
    
    # 2. Compute H¹ in different regions
    h1_all = pref_dataset.compute_h1(pairs, region="all")
    h1_trap = pref_dataset.compute_h1(pairs, region="trap")
    h1_safe = pref_dataset.compute_h1(pairs, region="safe")
    
    # 3. Train standard reward model
    standard_rm = StandardRewardModel()
    standard_losses = train_reward_model(standard_rm, pairs, epochs=100, hodge_filter=False)
    
    # 4. Train Hodge-filtered reward model
    hodge_rm = HodgeFilteredRewardModel()
    hodge_losses = train_reward_model(hodge_rm, pairs, epochs=100, hodge_filter=True)
    
    # 5. Train policies using each reward model
    env = PreferenceTrapEnv()
    
    _, standard_returns, standard_violations, standard_goals = train_policy_with_reward_model(
        env, standard_rm, episodes=policy_episodes
    )
    
    _, hodge_returns, hodge_violations, hodge_goals = train_policy_with_reward_model(
        env, hodge_rm, episodes=policy_episodes
    )
    
    # 6. Compute exploitation metrics
    results = {
        "seed": seed,
        "h1_magnitude_param": h1_magnitude,
        "h1_computed": {
            "all": h1_all,
            "trap": h1_trap,
            "safe": h1_safe
        },
        "standard_policy": {
            "total_violations": sum(standard_violations),
            "final_violations": sum(standard_violations[-50:]),
            "exploitation_rate": sum(standard_violations) / policy_episodes,
            "final_return": float(np.mean(standard_returns[-50:])),
            "goal_rate": float(np.mean(standard_goals[-50:]))
        },
        "hodge_policy": {
            "total_violations": sum(hodge_violations),
            "final_violations": sum(hodge_violations[-50:]),
            "exploitation_rate": sum(hodge_violations) / policy_episodes,
            "final_return": float(np.mean(hodge_returns[-50:])),
            "goal_rate": float(np.mean(hodge_goals[-50:]))
        },
        "exploitation_prevented": sum(standard_violations) - sum(hodge_violations)
    }
    
    return results


def run_full_experiment(
    num_seeds: int = 50,
    h1_magnitudes: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    output_dir: str = "results/h1_exploitation"
):
    """Run full H¹ exploitation experiment across seeds and H¹ magnitudes."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for h1_mag in h1_magnitudes:
        print(f"\n{'='*60}")
        print(f"H¹ Magnitude: {h1_mag}")
        print(f"{'='*60}")
        
        mag_results = []
        for seed in range(num_seeds):
            if (seed + 1) % 10 == 0:
                print(f"  Seed {seed+1}/{num_seeds}...")
            
            result = run_h1_exploitation_experiment(
                seed=seed,
                h1_magnitude=h1_mag
            )
            mag_results.append(result)
        
        # Aggregate results for this H¹ magnitude
        standard_violations = [r["standard_policy"]["total_violations"] for r in mag_results]
        hodge_violations = [r["hodge_policy"]["total_violations"] for r in mag_results]
        h1_computed = [r["h1_computed"]["trap"] for r in mag_results]
        
        aggregate = {
            "h1_magnitude_param": h1_mag,
            "h1_computed_mean": float(np.mean(h1_computed)),
            "h1_computed_std": float(np.std(h1_computed)),
            "standard_violations": compute_statistics(standard_violations, "violations").to_dict(),
            "hodge_violations": compute_statistics(hodge_violations, "violations").to_dict(),
            "comparison": compare_methods(standard_violations, hodge_violations, "Standard", "Hodge"),
            "per_seed": mag_results
        }
        
        all_results.append(aggregate)
        
        # Print summary
        print(f"  H¹ (computed): {aggregate['h1_computed_mean']:.3f} ± {aggregate['h1_computed_std']:.3f}")
        print(f"  Standard violations: {aggregate['standard_violations']['mean']:.1f} ± {aggregate['standard_violations']['std']:.1f}")
        print(f"  Hodge violations: {aggregate['hodge_violations']['mean']:.1f} ± {aggregate['hodge_violations']['std']:.1f}")
        print(f"  Effect size (Cohen's d): {aggregate['comparison']['cohens_d']:.2f} ({aggregate['comparison']['effect_size']})")
    
    # Save all results
    final_output = {
        "experiment": "H1_Exploitable_Reward_Hacking",
        "num_seeds": num_seeds,
        "h1_magnitudes": h1_magnitudes,
        "results": all_results
    }
    
    with open(output_path / "h1_exploitation_results.json", 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"\nResults saved to {output_path / 'h1_exploitation_results.json'}")
    
    # Generate plots
    _plot_results(all_results, output_path)
    
    return final_output


def _plot_results(results: List[Dict], output_path: Path):
    """Generate visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    h1_params = [r["h1_magnitude_param"] for r in results]
    h1_computed = [r["h1_computed_mean"] for r in results]
    standard_mean = [r["standard_violations"]["mean"] for r in results]
    standard_ci = [(r["standard_violations"]["ci_95"][1] - r["standard_violations"]["ci_95"][0])/2 for r in results]
    hodge_mean = [r["hodge_violations"]["mean"] for r in results]
    hodge_ci = [(r["hodge_violations"]["ci_95"][1] - r["hodge_violations"]["ci_95"][0])/2 for r in results]
    
    # Plot 1: H¹ vs Violations
    ax1 = axes[0, 0]
    ax1.errorbar(h1_computed, standard_mean, yerr=standard_ci, 
                 marker='o', label='Standard RLHF', color='red', capsize=3)
    ax1.errorbar(h1_computed, hodge_mean, yerr=hodge_ci,
                 marker='s', label='Hodge-Filtered', color='blue', capsize=3)
    ax1.set_xlabel('H¹ Magnitude (computed)')
    ax1.set_ylabel('Total Violations')
    ax1.set_title('H¹ Enables Reward Hacking')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Exploitation prevented
    ax2 = axes[0, 1]
    prevented = [s - h for s, h in zip(standard_mean, hodge_mean)]
    ax2.bar(range(len(h1_params)), prevented, color='green', alpha=0.7)
    ax2.set_xticks(range(len(h1_params)))
    ax2.set_xticklabels([f'{h:.2f}' for h in h1_params])
    ax2.set_xlabel('H¹ Magnitude (parameter)')
    ax2.set_ylabel('Violations Prevented')
    ax2.set_title('Hodge Filtering Prevents Exploitation')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Effect size vs H¹
    ax3 = axes[1, 0]
    effect_sizes = [abs(r["comparison"]["cohens_d"]) for r in results]
    ax3.bar(range(len(h1_params)), effect_sizes, color='purple', alpha=0.7)
    ax3.axhline(y=0.8, color='black', linestyle='--', label="Large effect threshold")
    ax3.set_xticks(range(len(h1_params)))
    ax3.set_xticklabels([f'{h:.2f}' for h in h1_params])
    ax3.set_xlabel('H¹ Magnitude (parameter)')
    ax3.set_ylabel("|Cohen's d|")
    ax3.set_title('Effect Size of Hodge Filtering')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Correlation
    ax4 = axes[1, 1]
    # Flatten per-seed data for correlation
    all_h1 = []
    all_violations = []
    for r in results:
        for seed_data in r["per_seed"]:
            all_h1.append(seed_data["h1_computed"]["trap"])
            all_violations.append(seed_data["standard_policy"]["total_violations"])
    
    ax4.scatter(all_h1, all_violations, alpha=0.3, s=20)
    
    # Fit line
    slope, intercept, r_value, p_value, _ = stats.linregress(all_h1, all_violations)
    x_line = np.linspace(min(all_h1), max(all_h1), 100)
    ax4.plot(x_line, slope * x_line + intercept, 'r-', 
             label=f'r={r_value:.2f}, p={p_value:.3f}')
    ax4.set_xlabel('H¹ (per-seed)')
    ax4.set_ylabel('Violations (Standard RLHF)')
    ax4.set_title('Correlation: H¹ → Exploitation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'h1_exploitation_plots.png', dpi=150)
    print(f"Plots saved to {output_path / 'h1_exploitation_plots.png'}")


# ============================================================================
# 7. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="H¹ Exploitation Experiment")
    parser.add_argument("--seeds", type=int, default=50, help="Number of seeds")
    parser.add_argument("--quick", action="store_true", help="Quick test with 5 seeds")
    args = parser.parse_args()
    
    num_seeds = 5 if args.quick else args.seeds
    
    print("="*60)
    print("H¹-EXPLOITABLE REWARD HACKING EXPERIMENT")
    print("="*60)
    print(f"\nThesis: Cyclic preferences (H¹ ≠ 0) in human feedback enable")
    print("reward hacking. Hodge-filtered training prevents exploitation.")
    print(f"\nRunning with {num_seeds} seeds...")
    
    results = run_full_experiment(
        num_seeds=num_seeds,
        h1_magnitudes=[0.0, 0.25, 0.5, 0.75, 1.0],
        output_dir="../../results/h1_exploitation"
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
