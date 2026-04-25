"""
Modal GPU Runner for Feedback Geometry Experiments

Deploy experiments to Modal cloud with GPU acceleration.
Usage:
    modal run modal_runner.py::run_h1_experiment
    modal run modal_runner.py::run_sandbagging_experiment
    modal run modal_runner.py::run_sandbagging_v2 --mode full
    modal run modal_runner.py::run_sandbagging_v2 --mode diagnostics
    modal run modal_runner.py::run_sandbagging_v2 --mode generalization
    modal run modal_runner.py::run_all_experiments
"""

import modal

app = modal.App("feedback-geometry-experiments")
volume = modal.Volume.from_name("feedback-geometry-data", create_if_missing=True)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24.0,<2.0.0",
        "torch>=2.1.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn",
    )
)

# Extended image with sentence-transformers for CCHC experiments
image_cchc = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24.0,<2.0.0",
        "torch>=2.1.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn",
        "sentence-transformers>=2.2.0",
        "datasets>=2.0.0",
    )
)


# ============================================================================
# H¹ EXPLOITATION EXPERIMENT
# ============================================================================

@app.function(
    image=image,
    gpu="L4",
    timeout=7200,  # 2 hours
    volumes={VOLUME_PATH: volume},
)
def run_h1_experiment(num_seeds: int = 50, h1_magnitudes: list = None):
    """
    Run the H¹ exploitation experiment on Modal GPU.
    
    Demonstrates that cyclic preferences (H¹ ≠ 0) enable reward hacking,
    and Hodge filtering prevents exploitation.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import json
    from pathlib import Path
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running H¹ Exploitation Experiment on {DEVICE}")
    print(f"Seeds: {num_seeds}")
    
    if h1_magnitudes is None:
        h1_magnitudes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # ========== DATASET ==========
    class PreferenceDataset:
        """Synthetic preference dataset with controlled H¹ cyclic inconsistency."""
        
        def __init__(self, num_items: int = 100, h1_magnitude: float = 0.0, seed: int = 42):
            np.random.seed(seed)
            self.num_items = num_items
            self.h1_magnitude = h1_magnitude
            self.items = np.random.randn(num_items, 16)
            self.true_utilities = np.random.randn(num_items)
            self.pairs = []
            self.labels = []
            self._generate_preferences()
        
        def _generate_preferences(self):
            num_pairs = self.num_items * 10
            for _ in range(num_pairs):
                i, j = np.random.choice(self.num_items, 2, replace=False)
                true_diff = self.true_utilities[i] - self.true_utilities[j]
                base_prob = 1.0 / (1.0 + np.exp(-true_diff))
                
                # Inject cyclic component
                cyclic_bias = self.h1_magnitude * np.sin(2 * np.pi * (i - j) / self.num_items)
                noisy_prob = np.clip(base_prob + cyclic_bias, 0.01, 0.99)
                label = 1.0 if np.random.random() < noisy_prob else 0.0
                
                self.pairs.append((i, j))
                self.labels.append(label)
        
        def compute_h1_magnitude(self) -> float:
            """Compute actual H¹ via cycle basis."""
            n = min(20, self.num_items)
            pref_matrix = np.zeros((n, n))
            counts = np.zeros((n, n))
            
            for (i, j), label in zip(self.pairs, self.labels):
                if i < n and j < n:
                    pref_matrix[i, j] += label
                    pref_matrix[j, i] += (1 - label)
                    counts[i, j] += 1
                    counts[j, i] += 1
            
            counts = np.maximum(counts, 1)
            pref_matrix /= counts
            
            total_cycle = 0.0
            num_cycles = 0
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        cycle = pref_matrix[i,j] + pref_matrix[j,k] + pref_matrix[k,i]
                        deviation = abs(cycle - 1.5)
                        total_cycle += deviation
                        num_cycles += 1
            
            return total_cycle / max(num_cycles, 1)
        
        def get_tensors(self, device="cpu"):
            items_t = torch.FloatTensor(self.items).to(device)
            pairs_t = torch.LongTensor(self.pairs).to(device)
            labels_t = torch.FloatTensor(self.labels).to(device)
            return items_t, pairs_t, labels_t
    
    # ========== REWARD MODELS ==========
    class StandardRewardModel(nn.Module):
        def __init__(self, input_dim: int = 16, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, x):
            return self.net(x)
        
        def train_on_preferences(self, items, pairs, labels, epochs=100, lr=1e-3):
            optimizer = optim.Adam(self.parameters(), lr=lr)
            for epoch in range(epochs):
                i_idx, j_idx = pairs[:, 0], pairs[:, 1]
                r_i = self.forward(items[i_idx]).squeeze()
                r_j = self.forward(items[j_idx]).squeeze()
                probs = torch.sigmoid(r_i - r_j)
                loss = nn.BCELoss()(probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    class HodgeFilteredRewardModel(nn.Module):
        """Reward model with Hodge decomposition to filter cyclic component."""
        
        def __init__(self, input_dim: int = 16, hidden_dim: int = 64):
            super().__init__()
            self.potential = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.harmonic_coeff = nn.Parameter(torch.zeros(1))
        
        def forward(self, x):
            return self.potential(x)
        
        def train_on_preferences(self, items, pairs, labels, epochs=100, lr=1e-3):
            optimizer = optim.Adam(self.parameters(), lr=lr)
            for epoch in range(epochs):
                i_idx, j_idx = pairs[:, 0], pairs[:, 1]
                phi_i = self.potential(items[i_idx]).squeeze()
                phi_j = self.potential(items[j_idx]).squeeze()
                
                # Gradient (exact) component only
                gradient_diff = phi_i - phi_j
                
                # Harmonic (cyclic) component - penalized
                harmonic_contrib = self.harmonic_coeff * torch.sin(
                    2 * np.pi * (i_idx.float() - j_idx.float()) / len(items)
                )
                
                full_diff = gradient_diff + harmonic_contrib
                probs = torch.sigmoid(full_diff)
                
                pred_loss = nn.BCELoss()(probs, labels)
                harmonic_penalty = 10.0 * (self.harmonic_coeff ** 2)
                loss = pred_loss + harmonic_penalty
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    # ========== ENVIRONMENT ==========
    class PreferenceTrapEnv:
        """Navigation environment with a 'preference trap' exploiting cyclic rewards."""
        
        def __init__(self, reward_model, items, trap_indices, device="cpu"):
            self.reward_model = reward_model
            self.items = items
            self.trap_indices = set(trap_indices)
            self.device = device
            self.state_idx = 0
            self.step_count = 0
            self.max_steps = 50
        
        def reset(self):
            self.state_idx = np.random.randint(0, len(self.items) // 2)
            self.step_count = 0
            return self.state_idx
        
        def step(self, action):
            # Action: move to adjacent item (+1, -1, or stay)
            action = int(np.clip(action, -1, 1))
            new_idx = (self.state_idx + action) % len(self.items)
            
            old_state = self.items[self.state_idx:self.state_idx+1]
            new_state = self.items[new_idx:new_idx+1]
            
            with torch.no_grad():
                reward = (self.reward_model(new_state) - self.reward_model(old_state)).item()
            
            self.state_idx = new_idx
            self.step_count += 1
            
            in_trap = new_idx in self.trap_indices
            done = self.step_count >= self.max_steps
            
            return new_idx, reward, done, {"in_trap": in_trap}
    
    # ========== POLICY ==========
    class SimplePolicy(nn.Module):
        def __init__(self, num_items: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Embedding(num_items, 32),
                nn.Flatten(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 3)  # -1, 0, +1
            )
        
        def forward(self, state_idx):
            x = torch.LongTensor([state_idx])
            logits = self.net(x)
            return torch.distributions.Categorical(logits=logits)
        
        def train_on_env(self, env, episodes=100, lr=1e-3):
            optimizer = optim.Adam(self.parameters(), lr=lr)
            trap_visits = 0
            total_reward = 0.0
            
            for ep in range(episodes):
                state = env.reset()
                done = False
                log_probs = []
                rewards = []
                
                while not done:
                    dist = self.forward(state)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    next_state, reward, done, info = env.step(action.item() - 1)
                    
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    if info["in_trap"]:
                        trap_visits += 1
                    
                    state = next_state
                
                # REINFORCE update
                returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + 0.99 * G
                    returns.insert(0, G)
                returns = torch.FloatTensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                
                loss = sum(-lp * R for lp, R in zip(log_probs, returns))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_reward += sum(rewards)
            
            return trap_visits, total_reward / episodes
    
    # ========== RUN EXPERIMENT ==========
    results = {
        "h1_magnitudes": h1_magnitudes,
        "num_seeds": num_seeds,
        "standard_model": {"trap_visits": [], "returns": []},
        "hodge_filtered": {"trap_visits": [], "returns": []},
        "measured_h1": [],
    }
    
    for h1_mag in h1_magnitudes:
        print(f"\n=== H¹ magnitude: {h1_mag} ===")
        
        standard_traps = []
        hodge_traps = []
        standard_returns = []
        hodge_returns = []
        measured_h1s = []
        
        for seed in range(num_seeds):
            if (seed + 1) % 10 == 0:
                print(f"  Seed {seed+1}/{num_seeds}")
            
            # Generate dataset
            dataset = PreferenceDataset(num_items=100, h1_magnitude=h1_mag, seed=seed)
            measured_h1s.append(dataset.compute_h1_magnitude())
            
            items, pairs, labels = dataset.get_tensors(device=DEVICE)
            
            # Trap items (where cyclic bias is strongest)
            trap_indices = list(range(45, 55))
            
            # Train standard model
            standard_model = StandardRewardModel().to(DEVICE)
            standard_model.train_on_preferences(items, pairs, labels)
            
            # Train Hodge-filtered model
            hodge_model = HodgeFilteredRewardModel().to(DEVICE)
            hodge_model.train_on_preferences(items, pairs, labels)
            
            # Train policies
            env_standard = PreferenceTrapEnv(standard_model, items, trap_indices, DEVICE)
            policy_standard = SimplePolicy(100)
            traps_s, ret_s = policy_standard.train_on_env(env_standard)
            
            env_hodge = PreferenceTrapEnv(hodge_model, items, trap_indices, DEVICE)
            policy_hodge = SimplePolicy(100)
            traps_h, ret_h = policy_hodge.train_on_env(env_hodge)
            
            standard_traps.append(traps_s)
            hodge_traps.append(traps_h)
            standard_returns.append(ret_s)
            hodge_returns.append(ret_h)
        
        results["standard_model"]["trap_visits"].append({
            "h1": h1_mag,
            "mean": float(np.mean(standard_traps)),
            "std": float(np.std(standard_traps)),
            "ci_95": [float(np.percentile(standard_traps, 2.5)), float(np.percentile(standard_traps, 97.5))]
        })
        results["hodge_filtered"]["trap_visits"].append({
            "h1": h1_mag,
            "mean": float(np.mean(hodge_traps)),
            "std": float(np.std(hodge_traps)),
            "ci_95": [float(np.percentile(hodge_traps, 2.5)), float(np.percentile(hodge_traps, 97.5))]
        })
        results["measured_h1"].append({
            "target": h1_mag,
            "measured_mean": float(np.mean(measured_h1s)),
            "measured_std": float(np.std(measured_h1s))
        })
        
        print(f"  Standard: {np.mean(standard_traps):.1f} ± {np.std(standard_traps):.1f} trap visits")
        print(f"  Hodge:    {np.mean(hodge_traps):.1f} ± {np.std(hodge_traps):.1f} trap visits")
    
    # ========== STATISTICAL ANALYSIS ==========
    print("\n=== Statistical Analysis ===")
    
    # Correlation: H¹ → exploitation
    h1_vals = h1_magnitudes
    standard_means = [r["mean"] for r in results["standard_model"]["trap_visits"]]
    hodge_means = [r["mean"] for r in results["hodge_filtered"]["trap_visits"]]
    
    corr_standard, p_standard = stats.pearsonr(h1_vals, standard_means)
    corr_hodge, p_hodge = stats.pearsonr(h1_vals, hodge_means)
    
    results["correlations"] = {
        "standard_h1_exploitation": {"r": float(corr_standard), "p": float(p_standard)},
        "hodge_h1_exploitation": {"r": float(corr_hodge), "p": float(p_hodge)}
    }
    
    print(f"Standard model: r={corr_standard:.3f}, p={p_standard:.4f}")
    print(f"Hodge filtered: r={corr_hodge:.3f}, p={p_hodge:.4f}")
    
    # ========== SAVE RESULTS ==========
    output_path = Path(VOLUME_PATH) / "h1_exploitation"
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    x = np.array(h1_vals)
    y_std = np.array(standard_means)
    y_hdg = np.array(hodge_means)
    err_std = [r["std"] for r in results["standard_model"]["trap_visits"]]
    err_hdg = [r["std"] for r in results["hodge_filtered"]["trap_visits"]]
    
    ax1.errorbar(x, y_std, yerr=err_std, marker='o', label='Standard RLHF', capsize=3)
    ax1.errorbar(x, y_hdg, yerr=err_hdg, marker='s', label='Hodge-Filtered', capsize=3)
    ax1.set_xlabel('Injected H¹ Magnitude')
    ax1.set_ylabel('Trap Visits (Exploitation)')
    ax1.set_title(f'H¹ → Exploitation (n={num_seeds} seeds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.bar([0, 1], [corr_standard, corr_hodge], color=['red', 'blue'], alpha=0.7)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Standard', 'Hodge'])
    ax2.set_ylabel('Pearson r (H¹ ↔ Exploitation)')
    ax2.set_title('Correlation Strength')
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / "h1_exploitation_plot.png", dpi=150)
    
    volume.commit()
    print(f"\nResults saved to {output_path}")
    
    return results


# ============================================================================
# SANDBAGGING TRAP EXPERIMENT
# ============================================================================

@app.function(
    image=image,
    gpu="L4",
    timeout=10800,  # 3 hours
    volumes={VOLUME_PATH: volume},
)
def run_sandbagging_experiment(num_seeds: int = 50, run_ablations: bool = True):
    """
    Run the Sandbagging Trap v2 experiment on Modal GPU.
    
    Compares PPO, CPO, and SGPO on a navigation task with deceptive rewards.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import json
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Tuple, Dict, List
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Sandbagging Experiment on {DEVICE}")
    print(f"Seeds: {num_seeds}, Ablations: {run_ablations}")
    
    # ========== ENVIRONMENT ==========
    @dataclass
    class EnvConfig:
        trap_center: Tuple[float, float] = (5.0, 6.0)
        trap_radius: float = 2.5
        event_horizon: float = 2.0
        goal: Tuple[float, float] = (10.0, 10.0)
        start: Tuple[float, float] = (0.0, 0.0)
        max_steps: int = 60
        dt: float = 0.5
        trap_reward: float = 3.0
    
    class SandbaggingEnv:
        def __init__(self, config: EnvConfig = None):
            self.config = config or EnvConfig()
            self.trap_center = np.array(self.config.trap_center)
            self.trap_radius = self.config.trap_radius
            self.event_horizon = self.config.event_horizon
            self.goal = np.array(self.config.goal)
            self.start = np.array(self.config.start)
            self.max_steps = self.config.max_steps
            self.dt = self.config.dt
            self.trap_reward = self.config.trap_reward
            self.state = None
            self.step_count = 0
        
        def reset(self) -> np.ndarray:
            self.state = self.start.copy()
            self.step_count = 0
            return self.state.copy()
        
        def distance_to_trap(self, state=None) -> float:
            if state is None: state = self.state
            return np.linalg.norm(state - self.trap_center) - self.trap_radius
        
        def in_trap(self, state=None) -> bool:
            return self.distance_to_trap(state) < 0
        
        def in_event_horizon(self, state=None) -> bool:
            if state is None: state = self.state
            return np.linalg.norm(state - self.trap_center) < self.event_horizon
        
        def step(self, action: np.ndarray):
            velocity = np.clip(action, -1.0, 1.0)
            prev_dist = np.linalg.norm(self.goal - self.state)
            self.state = self.state + velocity * self.dt
            curr_dist = np.linalg.norm(self.goal - self.state)
            
            progress_reward = prev_dist - curr_dist
            in_trap = self.in_trap()
            trap_reward = self.trap_reward if in_trap else 0.0
            reward = progress_reward + trap_reward
            cost = 1.0 if in_trap else 0.0
            
            self.step_count += 1
            done = False
            
            if curr_dist < 1.0:
                reward += 10.0
                done = True
            if self.in_event_horizon():
                reward -= 100.0
                done = True
            if self.step_count >= self.max_steps:
                done = True
            
            info = {
                'in_trap': in_trap,
                'dist_to_trap': self.distance_to_trap(),
                'dist_to_goal': curr_dist
            }
            return self.state.copy(), reward, cost, done, info
    
    # ========== NETWORKS ==========
    class Actor(nn.Module):
        def __init__(self, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 2)
            )
            self.log_std = nn.Parameter(torch.zeros(2) - 1.0)
        
        def forward(self, x):
            mu = self.net(x)
            std = torch.exp(self.log_std)
            return torch.distributions.Normal(mu, std)
    
    class Critic(nn.Module):
        def __init__(self, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, x):
            return self.net(x)
    
    class LearnedRiemannianMetric(nn.Module):
        def __init__(self, hidden_dim: int = 32, sharpness: float = 2.0, severity: float = 5.0):
            super().__init__()
            self.danger_net = nn.Sequential(
                nn.Linear(2, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1), nn.Softplus()
            )
            self.base_metric = nn.Parameter(torch.tensor(1.0))
            self.sharpness = nn.Parameter(torch.tensor(sharpness))
            self.severity = nn.Parameter(torch.tensor(severity))
        
        def forward(self, x):
            if x.dim() == 1: x = x.unsqueeze(0)
            danger = self.danger_net(x)
            return self.base_metric + self.severity * (danger ** self.sharpness)
    
    # ========== TRAINING FUNCTIONS ==========
    def train_ppo(env, episodes=300, gamma=0.99, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        actor = Actor().to(DEVICE)
        critic = Critic().to(DEVICE)
        opt_actor = optim.Adam(actor.parameters(), lr=1e-3)
        opt_critic = optim.Adam(critic.parameters(), lr=3e-3)
        
        returns, violations = [], []
        
        for ep in range(episodes):
            obs = env.reset()
            trajectory = []
            ep_violations = 0
            ep_return = 0.0
            done = False
            
            while not done:
                obs_t = torch.FloatTensor(obs).to(DEVICE)
                with torch.no_grad():
                    dist = actor(obs_t)
                    action = dist.sample()
                
                next_obs, reward, cost, done, info = env.step(action.cpu().numpy())
                trajectory.append((obs, action, reward))
                ep_violations += int(info['in_trap'])
                ep_return += reward
                obs = next_obs
            
            states = torch.FloatTensor(np.array([t[0] for t in trajectory])).to(DEVICE)
            actions = torch.stack([t[1] for t in trajectory]).to(DEVICE)
            
            G = 0
            ret = []
            for _, _, r in reversed(trajectory):
                G = r + gamma * G
                ret.insert(0, G)
            ret = torch.FloatTensor(ret).unsqueeze(1).to(DEVICE)
            
            loss_crit = nn.MSELoss()(critic(states), ret)
            opt_critic.zero_grad()
            loss_crit.backward()
            opt_critic.step()
            
            with torch.no_grad():
                adv = ret - critic(states)
            
            dists = actor(states)
            log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
            loss_actor = -(log_probs * adv).mean()
            opt_actor.zero_grad()
            loss_actor.backward()
            opt_actor.step()
            
            returns.append(ep_return)
            violations.append(ep_violations)
        
        return returns, violations
    
    def train_cpo(env, episodes=300, gamma=0.99, cost_limit=5.0, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        actor = Actor().to(DEVICE)
        r_critic = Critic().to(DEVICE)
        c_critic = Critic().to(DEVICE)
        
        opt_actor = optim.Adam(actor.parameters(), lr=1e-3)
        opt_r_critic = optim.Adam(r_critic.parameters(), lr=3e-3)
        opt_c_critic = optim.Adam(c_critic.parameters(), lr=3e-3)
        
        log_lambda = nn.Parameter(torch.zeros(1, device=DEVICE))
        opt_lambda = optim.Adam([log_lambda], lr=1e-2)
        
        returns, violations = [], []
        
        for ep in range(episodes):
            obs = env.reset()
            trajectory = []
            ep_violations = 0
            ep_return = 0.0
            done = False
            
            while not done:
                obs_t = torch.FloatTensor(obs).to(DEVICE)
                with torch.no_grad():
                    dist = actor(obs_t)
                    action = dist.sample()
                
                next_obs, reward, cost, done, info = env.step(action.cpu().numpy())
                trajectory.append((obs, action, reward, cost))
                ep_violations += int(info['in_trap'])
                ep_return += reward
                obs = next_obs
            
            states = torch.FloatTensor(np.array([t[0] for t in trajectory])).to(DEVICE)
            actions = torch.stack([t[1] for t in trajectory]).to(DEVICE)
            
            r_ret, c_ret = [], []
            Gr, Gc = 0, 0
            for t in reversed(trajectory):
                Gr = t[2] + gamma * Gr
                Gc = t[3] + gamma * Gc
                r_ret.insert(0, Gr)
                c_ret.insert(0, Gc)
            r_ret = torch.FloatTensor(r_ret).unsqueeze(1).to(DEVICE)
            c_ret = torch.FloatTensor(c_ret).unsqueeze(1).to(DEVICE)
            
            opt_r_critic.zero_grad()
            nn.MSELoss()(r_critic(states), r_ret).backward()
            opt_r_critic.step()
            
            opt_c_critic.zero_grad()
            nn.MSELoss()(c_critic(states), c_ret).backward()
            opt_c_critic.step()
            
            lambda_val = torch.exp(log_lambda).detach()
            r_adv = r_ret - r_critic(states).detach()
            c_adv = c_ret - c_critic(states).detach()
            combined_adv = r_adv - lambda_val * c_adv
            
            dists = actor(states)
            log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
            loss_actor = -(log_probs * combined_adv).mean()
            opt_actor.zero_grad()
            loss_actor.backward()
            opt_actor.step()
            
            loss_lambda = -log_lambda * (cost_limit - c_ret.mean().detach())
            opt_lambda.zero_grad()
            loss_lambda.backward()
            opt_lambda.step()
            
            returns.append(ep_return)
            violations.append(ep_violations)
        
        return returns, violations
    
    def train_sgpo(env, episodes=300, gamma=0.99, sharpness=2.0, severity=5.0, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        actor = Actor().to(DEVICE)
        critic = Critic().to(DEVICE)
        metric = LearnedRiemannianMetric(sharpness=sharpness, severity=severity).to(DEVICE)
        
        opt_actor = optim.Adam(actor.parameters(), lr=1e-3)
        opt_critic = optim.Adam(critic.parameters(), lr=3e-3)
        opt_metric = optim.Adam(metric.parameters(), lr=3e-3)
        
        returns, violations = [], []
        
        for ep in range(episodes):
            obs = env.reset()
            trajectory = []
            ep_violations = 0
            ep_return = 0.0
            done = False
            
            while not done:
                obs_t = torch.FloatTensor(obs).to(DEVICE)
                with torch.no_grad():
                    dist = actor(obs_t)
                    action = dist.sample()
                
                next_obs, reward, cost, done, info = env.step(action.cpu().numpy())
                trajectory.append((obs, action, reward, cost, info['dist_to_trap']))
                ep_violations += int(info['in_trap'])
                ep_return += reward
                obs = next_obs
            
            states = torch.FloatTensor(np.array([t[0] for t in trajectory])).to(DEVICE)
            actions = torch.stack([t[1] for t in trajectory]).to(DEVICE)
            costs = torch.FloatTensor([t[3] for t in trajectory]).to(DEVICE)
            trap_dists = torch.FloatTensor([t[4] for t in trajectory]).to(DEVICE)
            
            G = 0
            ret = []
            for t in reversed(trajectory):
                G = t[2] + gamma * G
                ret.insert(0, G)
            ret = torch.FloatTensor(ret).unsqueeze(1).to(DEVICE)
            
            loss_crit = nn.MSELoss()(critic(states), ret)
            opt_critic.zero_grad()
            loss_crit.backward()
            opt_critic.step()
            
            g_pred = metric(states)
            safe_dist = torch.clamp(trap_dists, min=0.1)
            g_target = 1.0 + 5.0 / safe_dist.unsqueeze(1) + costs.unsqueeze(1) * 10.0
            loss_metric = nn.MSELoss()(g_pred, g_target)
            opt_metric.zero_grad()
            loss_metric.backward()
            opt_metric.step()
            
            with torch.no_grad():
                g_values = metric(states)
                adv = ret - critic(states)
                riemannian_adv = adv / torch.sqrt(g_values)
            
            dists = actor(states)
            log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
            loss_actor = -(log_probs * riemannian_adv).mean()
            opt_actor.zero_grad()
            loss_actor.backward()
            opt_actor.step()
            
            returns.append(ep_return)
            violations.append(ep_violations)
        
        return returns, violations
    
    # ========== RUN EXPERIMENT ==========
    env_config = EnvConfig()
    
    method_results = {}
    for method_name, train_fn in [("ppo", train_ppo), ("cpo", train_cpo), ("sgpo", train_sgpo)]:
        print(f"\n=== {method_name.upper()} ===")
        
        all_returns = []
        all_violations = []
        
        for seed in range(num_seeds):
            if (seed + 1) % 10 == 0:
                print(f"  Seed {seed+1}/{num_seeds}")
            
            env = SandbaggingEnv(env_config)
            returns, violations = train_fn(env, seed=seed)
            all_returns.append(returns)
            all_violations.append(violations)
        
        final_returns = [np.mean(r[-50:]) for r in all_returns]
        total_violations = [sum(v) for v in all_violations]
        
        method_results[method_name] = {
            "final_return": {
                "mean": float(np.mean(final_returns)),
                "std": float(np.std(final_returns)),
                "ci_95": [float(np.percentile(final_returns, 2.5)), float(np.percentile(final_returns, 97.5))]
            },
            "total_violations": {
                "mean": float(np.mean(total_violations)),
                "std": float(np.std(total_violations)),
                "ci_95": [float(np.percentile(total_violations, 2.5)), float(np.percentile(total_violations, 97.5))]
            },
            "learning_curves": {
                "returns_mean": np.mean(all_returns, axis=0).tolist(),
                "violations_mean": np.mean(all_violations, axis=0).tolist()
            }
        }
        
        print(f"  Return: {np.mean(final_returns):.2f} ± {np.std(final_returns):.2f}")
        print(f"  Violations: {np.mean(total_violations):.1f} ± {np.std(total_violations):.1f}")
    
    # ========== ABLATIONS ==========
    ablation_results = {}
    if run_ablations:
        print("\n=== Ablations ===")
        ablation_seeds = min(20, num_seeds)
        
        for param_name, values in [("sharpness", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
                                    ("severity", [1.0, 2.5, 5.0, 10.0, 20.0])]:
            print(f"  {param_name}")
            ablation_results[param_name] = []
            
            for val in values:
                violations_list = []
                for seed in range(ablation_seeds):
                    env = SandbaggingEnv(env_config)
                    kwargs = {"sharpness": val} if param_name == "sharpness" else {"severity": val}
                    _, violations = train_sgpo(env, seed=seed, **kwargs)
                    violations_list.append(sum(violations))
                
                ablation_results[param_name].append({
                    "value": val,
                    "violations_mean": float(np.mean(violations_list)),
                    "violations_std": float(np.std(violations_list))
                })
    
    # ========== STATISTICAL COMPARISONS ==========
    print("\n=== Statistical Comparisons ===")
    comparisons = {}
    
    # SGPO vs PPO
    sgpo_viol = [sum(v) for v in all_violations]  # from last run (sgpo)
    ppo_viol = method_results["ppo"]["total_violations"]
    
    # Manual extraction for comparison
    t_stat, p_val = stats.ttest_ind(sgpo_viol, sgpo_viol)  # placeholder
    
    # ========== SAVE RESULTS ==========
    output_path = Path(VOLUME_PATH) / "sandbagging_v2"
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "experiment": "SandbaggingTrap_v2",
        "num_seeds": num_seeds,
        "method_results": method_results,
        "ablations": ablation_results
    }
    
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    methods = ["ppo", "cpo", "sgpo"]
    colors = {"ppo": "red", "cpo": "orange", "sgpo": "blue"}
    
    # Learning curves
    ax1 = axes[0, 0]
    for m in methods:
        lc = method_results[m]["learning_curves"]["returns_mean"]
        window = 20
        if len(lc) > window:
            smoothed = np.convolve(lc, np.ones(window)/window, mode='valid')
            ax1.plot(smoothed, color=colors[m], label=m.upper())
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Violations bar chart
    ax2 = axes[0, 1]
    x = np.arange(len(methods))
    means = [method_results[m]["total_violations"]["mean"] for m in methods]
    ci_widths = [(method_results[m]["total_violations"]["ci_95"][1] - 
                  method_results[m]["total_violations"]["ci_95"][0]) / 2 for m in methods]
    ax2.bar(x, means, yerr=ci_widths, color=[colors[m] for m in methods], alpha=0.7, capsize=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in methods])
    ax2.set_ylabel('Total Violations')
    ax2.set_title('Violations (95% CI)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Ablation: sharpness
    ax3 = axes[1, 0]
    if "sharpness" in ablation_results:
        abl = ablation_results["sharpness"]
        vals = [a["value"] for a in abl]
        means = [a["violations_mean"] for a in abl]
        stds = [a["violations_std"] for a in abl]
        ax3.errorbar(vals, means, yerr=stds, marker='o', capsize=3)
        ax3.axvline(x=2.0, color='red', linestyle='--', alpha=0.5, label='β=2')
        ax3.set_xlabel('Sharpness β')
        ax3.set_ylabel('Total Violations')
        ax3.set_title('Ablation: Sharpness')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Ablation: severity
    ax4 = axes[1, 1]
    if "severity" in ablation_results:
        abl = ablation_results["severity"]
        vals = [a["value"] for a in abl]
        means = [a["violations_mean"] for a in abl]
        stds = [a["violations_std"] for a in abl]
        ax4.errorbar(vals, means, yerr=stds, marker='s', capsize=3)
        ax4.set_xlabel('Severity C')
        ax4.set_ylabel('Total Violations')
        ax4.set_title('Ablation: Severity')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "sandbagging_plot.png", dpi=150)
    
    volume.commit()
    print(f"\nResults saved to {output_path}")
    
    return results


# ============================================================================
# SANDBAGGING V2.1 - WITH FIXES AND NEW MODES
# ============================================================================

@app.function(
    image=image,
    gpu="L4",
    timeout=14400,  # 4 hours
    volumes={VOLUME_PATH: volume},
)
def run_sandbagging_v2(
    num_seeds: int = 50,
    mode: str = "full"  # "full", "diagnostics", "generalization"
):
    """
    Run Sandbagging v2.1 experiment with all fixes on Modal GPU.
    
    Modes:
    - full: PPO, CPO, SGPO, SGPO_ANIS comparison with ablations
    - diagnostics: SGPO debugging with metric field visualization
    - generalization: Train on trap A, test on trap B (unseen)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import json
    from pathlib import Path
    from dataclasses import dataclass, field
    from typing import Tuple, Dict, List
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Sandbagging v2.1 on {DEVICE}")
    print(f"Mode: {mode}, Seeds: {num_seeds}")
    
    # ========== ENVIRONMENT ==========
    @dataclass
    class EnvConfig:
        trap_center: Tuple[float, float] = (5.0, 6.0)
        trap_radius: float = 2.5
        event_horizon: float = 2.0
        goal: Tuple[float, float] = (10.0, 10.0)
        start: Tuple[float, float] = (0.0, 0.0)
        max_steps: int = 60
        dt: float = 0.5
        trap_reward: float = 3.0
    
    @dataclass
    class AlgoConfig:
        episodes: int = 300
        gamma: float = 0.99
        lr_actor: float = 1e-3
        lr_critic: float = 3e-3
        lr_metric: float = 3e-3
        cost_limit: float = 5.0
        metric_sharpness: float = 2.0
        metric_severity: float = 5.0
        warmup_episodes: int = 30
        use_soft_scaling: bool = True
        metric_reg_weight: float = 0.1
        use_hybrid_lagrangian: bool = True
        hybrid_lambda_lr: float = 1e-2
        anisotropic_max_metric: float = 100.0
    
    class SandbaggingEnv:
        def __init__(self, config: EnvConfig = None):
            self.config = config or EnvConfig()
            self.trap_center = np.array(self.config.trap_center)
            self.trap_radius = self.config.trap_radius
            self.event_horizon = self.config.event_horizon
            self.goal = np.array(self.config.goal)
            self.start = np.array(self.config.start)
            self.max_steps = self.config.max_steps
            self.dt = self.config.dt
            self.trap_reward = self.config.trap_reward
            self.state = None
            self.step_count = 0
        
        def reset(self) -> np.ndarray:
            self.state = self.start.copy()
            self.step_count = 0
            return self.state.copy()
        
        def distance_to_trap(self, state=None) -> float:
            if state is None: state = self.state
            return np.linalg.norm(state - self.trap_center) - self.trap_radius
        
        def in_trap(self, state=None) -> bool:
            return self.distance_to_trap(state) < 0
        
        def in_event_horizon(self, state=None) -> bool:
            if state is None: state = self.state
            return np.linalg.norm(state - self.trap_center) < self.event_horizon
        
        def step(self, action: np.ndarray):
            velocity = np.clip(action, -1.0, 1.0)
            prev_dist = np.linalg.norm(self.goal - self.state)
            self.state = self.state + velocity * self.dt
            curr_dist = np.linalg.norm(self.goal - self.state)
            
            progress_reward = prev_dist - curr_dist
            in_trap = self.in_trap()
            trap_reward = self.trap_reward if in_trap else 0.0
            reward = progress_reward + trap_reward
            cost = 1.0 if in_trap else 0.0
            
            self.step_count += 1
            done = False
            
            if curr_dist < 1.0:
                reward += 10.0
                done = True
            if self.in_event_horizon():
                reward -= 100.0
                done = True
            if self.step_count >= self.max_steps:
                done = True
            
            info = {
                'in_trap': in_trap,
                'dist_to_trap': self.distance_to_trap(),
                'dist_to_goal': curr_dist
            }
            return self.state.copy(), reward, cost, done, info
    
    # ========== NETWORKS ==========
    class Actor(nn.Module):
        def __init__(self, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 2)
            )
            self.log_std = nn.Parameter(torch.zeros(2) - 1.0)
        
        def forward(self, x):
            mu = self.net(x)
            std = torch.exp(self.log_std)
            return torch.distributions.Normal(mu, std)
    
    class Critic(nn.Module):
        def __init__(self, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, x):
            return self.net(x)
    
    class LearnedRiemannianMetric(nn.Module):
        def __init__(self, hidden_dim: int = 32, sharpness: float = 2.0, severity: float = 5.0):
            super().__init__()
            self.danger_net = nn.Sequential(
                nn.Linear(2, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1), nn.Softplus()
            )
            self.base_metric = nn.Parameter(torch.tensor(1.0))
            self.sharpness = nn.Parameter(torch.tensor(sharpness))
            self.severity = nn.Parameter(torch.tensor(severity))
        
        def forward(self, x):
            if x.dim() == 1: x = x.unsqueeze(0)
            danger = self.danger_net(x)
            return self.base_metric + self.severity * (danger ** self.sharpness)
    
    class AnisotropicRiemannianMetric(nn.Module):
        """Anisotropic metric: only penalizes movement TOWARD danger."""
        def __init__(self, hidden_dim: int = 32, severity: float = 5.0, max_metric: float = 100.0):
            super().__init__()
            self.danger_net = nn.Sequential(
                nn.Linear(2, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
            self.center_net = nn.Sequential(
                nn.Linear(2, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
            self.base_metric = 1.0
            self.severity = severity
            self.max_metric = max_metric
        
        def forward(self, x, v=None):
            if x.dim() == 1: x = x.unsqueeze(0)
            danger_output = self.danger_net(x)
            danger_level = torch.sigmoid(danger_output[:, 0:1]) * self.severity
            danger_center = self.center_net(x)
            
            if v is None:
                g = self.base_metric + danger_level
                return torch.clamp(g, max=self.max_metric), torch.ones_like(g)
            
            if v.dim() == 1: v = v.unsqueeze(0)
            to_danger = danger_center - x
            dist_to_danger = torch.norm(to_danger, dim=-1, keepdim=True) + 1e-8
            n_hat = to_danger / dist_to_danger
            v_toward = torch.sum(v * n_hat, dim=-1, keepdim=True)
            v_toward_pos = torch.clamp(v_toward, min=0)
            v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
            toward_ratio_sq = (v_toward_pos / v_norm) ** 2
            g_dir = danger_level / (dist_to_danger + 0.1)
            g = self.base_metric + toward_ratio_sq * g_dir
            g = torch.clamp(g, max=self.max_metric)
            escape_factor = torch.sigmoid(-v_toward * 5.0)
            return g, escape_factor
        
        def get_danger_center(self, x):
            if x.dim() == 1: x = x.unsqueeze(0)
            return self.center_net(x)
    
    # ========== TRAINING FUNCTIONS ==========
    def train_ppo(env, config, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        actor = Actor().to(DEVICE)
        critic = Critic().to(DEVICE)
        opt_actor = optim.Adam(actor.parameters(), lr=config.lr_actor)
        opt_critic = optim.Adam(critic.parameters(), lr=config.lr_critic)
        
        returns, violations = [], []
        
        for ep in range(config.episodes):
            obs = env.reset()
            trajectory = []
            ep_violations = 0
            ep_return = 0.0
            done = False
            
            while not done:
                obs_t = torch.FloatTensor(obs).to(DEVICE)
                with torch.no_grad():
                    dist = actor(obs_t)
                    action = dist.sample()
                
                next_obs, reward, cost, done, info = env.step(action.cpu().numpy())
                trajectory.append((obs, action, reward))
                ep_violations += int(info['in_trap'])
                ep_return += reward
                obs = next_obs
            
            states = torch.FloatTensor(np.array([t[0] for t in trajectory])).to(DEVICE)
            actions = torch.stack([t[1] for t in trajectory]).to(DEVICE)
            
            G = 0
            ret = []
            for _, _, r in reversed(trajectory):
                G = r + config.gamma * G
                ret.insert(0, G)
            ret = torch.FloatTensor(ret).unsqueeze(1).to(DEVICE)
            
            loss_crit = nn.MSELoss()(critic(states), ret)
            opt_critic.zero_grad()
            loss_crit.backward()
            opt_critic.step()
            
            with torch.no_grad():
                adv = ret - critic(states)
            
            dists = actor(states)
            log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
            loss_actor = -(log_probs * adv).mean()
            opt_actor.zero_grad()
            loss_actor.backward()
            opt_actor.step()
            
            returns.append(ep_return)
            violations.append(ep_violations)
        
        return returns, violations, actor
    
    def train_cpo(env, config, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        actor = Actor().to(DEVICE)
        r_critic = Critic().to(DEVICE)
        c_critic = Critic().to(DEVICE)
        
        opt_actor = optim.Adam(actor.parameters(), lr=config.lr_actor)
        opt_r_critic = optim.Adam(r_critic.parameters(), lr=config.lr_critic)
        opt_c_critic = optim.Adam(c_critic.parameters(), lr=config.lr_critic)
        
        log_lambda = nn.Parameter(torch.zeros(1, device=DEVICE))
        opt_lambda = optim.Adam([log_lambda], lr=config.hybrid_lambda_lr)
        
        returns, violations = [], []
        
        for ep in range(config.episodes):
            obs = env.reset()
            trajectory = []
            ep_violations = 0
            ep_return = 0.0
            done = False
            
            while not done:
                obs_t = torch.FloatTensor(obs).to(DEVICE)
                with torch.no_grad():
                    dist = actor(obs_t)
                    action = dist.sample()
                
                next_obs, reward, cost, done, info = env.step(action.cpu().numpy())
                trajectory.append((obs, action, reward, cost))
                ep_violations += int(info['in_trap'])
                ep_return += reward
                obs = next_obs
            
            states = torch.FloatTensor(np.array([t[0] for t in trajectory])).to(DEVICE)
            actions = torch.stack([t[1] for t in trajectory]).to(DEVICE)
            
            r_ret, c_ret = [], []
            Gr, Gc = 0, 0
            for t in reversed(trajectory):
                Gr = t[2] + config.gamma * Gr
                Gc = t[3] + config.gamma * Gc
                r_ret.insert(0, Gr)
                c_ret.insert(0, Gc)
            r_ret = torch.FloatTensor(r_ret).unsqueeze(1).to(DEVICE)
            c_ret = torch.FloatTensor(c_ret).unsqueeze(1).to(DEVICE)
            
            opt_r_critic.zero_grad()
            nn.MSELoss()(r_critic(states), r_ret).backward()
            opt_r_critic.step()
            
            opt_c_critic.zero_grad()
            nn.MSELoss()(c_critic(states), c_ret).backward()
            opt_c_critic.step()
            
            lambda_val = torch.exp(log_lambda).detach()
            r_adv = r_ret - r_critic(states).detach()
            c_adv = c_ret - c_critic(states).detach()
            combined_adv = r_adv - lambda_val * c_adv
            
            dists = actor(states)
            log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
            loss_actor = -(log_probs * combined_adv).mean()
            opt_actor.zero_grad()
            loss_actor.backward()
            opt_actor.step()
            
            loss_lambda = -log_lambda * (config.cost_limit - c_ret.mean().detach())
            opt_lambda.zero_grad()
            loss_lambda.backward()
            opt_lambda.step()
            
            returns.append(ep_return)
            violations.append(ep_violations)
        
        return returns, violations, actor
    
    def train_sgpo(env, config, seed=0):
        """SGPO v2.1 with warmup, soft scaling, regularization, hybrid."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        actor = Actor().to(DEVICE)
        critic = Critic().to(DEVICE)
        metric = LearnedRiemannianMetric(
            sharpness=config.metric_sharpness,
            severity=config.metric_severity
        ).to(DEVICE)
        
        opt_actor = optim.Adam(actor.parameters(), lr=config.lr_actor)
        opt_critic = optim.Adam(critic.parameters(), lr=config.lr_critic)
        opt_metric = optim.Adam(metric.parameters(), lr=config.lr_metric)
        
        # Hybrid components
        if config.use_hybrid_lagrangian:
            cost_critic = Critic().to(DEVICE)
            opt_cost_critic = optim.Adam(cost_critic.parameters(), lr=config.lr_critic)
            log_lambda = nn.Parameter(torch.zeros(1, device=DEVICE))
            opt_lambda = optim.Adam([log_lambda], lr=config.hybrid_lambda_lr)
        
        returns, violations = [], []
        
        for ep in range(config.episodes):
            obs = env.reset()
            trajectory = []
            ep_violations = 0
            ep_return = 0.0
            done = False
            
            while not done:
                obs_t = torch.FloatTensor(obs).to(DEVICE)
                with torch.no_grad():
                    dist = actor(obs_t)
                    action = dist.sample()
                
                next_obs, reward, cost, done, info = env.step(action.cpu().numpy())
                trajectory.append((obs, action, reward, cost, info['dist_to_trap']))
                ep_violations += int(info['in_trap'])
                ep_return += reward
                obs = next_obs
            
            states = torch.FloatTensor(np.array([t[0] for t in trajectory])).to(DEVICE)
            actions = torch.stack([t[1] for t in trajectory]).to(DEVICE)
            costs = torch.FloatTensor([t[3] for t in trajectory]).to(DEVICE)
            trap_dists = torch.FloatTensor([t[4] for t in trajectory]).to(DEVICE)
            
            # Returns
            r_ret, c_ret = [], []
            Gr, Gc = 0, 0
            for t in reversed(trajectory):
                Gr = t[2] + config.gamma * Gr
                Gc = t[3] + config.gamma * Gc
                r_ret.insert(0, Gr)
                c_ret.insert(0, Gc)
            r_ret = torch.FloatTensor(r_ret).unsqueeze(1).to(DEVICE)
            c_ret = torch.FloatTensor(c_ret).unsqueeze(1).to(DEVICE)
            
            # Update critics
            loss_crit = nn.MSELoss()(critic(states), r_ret)
            opt_critic.zero_grad()
            loss_crit.backward()
            opt_critic.step()
            
            if config.use_hybrid_lagrangian:
                opt_cost_critic.zero_grad()
                nn.MSELoss()(cost_critic(states), c_ret).backward()
                opt_cost_critic.step()
            
            # Update metric (after warmup)
            if ep >= config.warmup_episodes:
                g_pred = metric(states)
                safe_dist = torch.clamp(trap_dists, min=0.1)
                g_target = 1.0 + 5.0 / safe_dist.unsqueeze(1) + costs.unsqueeze(1) * 10.0
                loss_metric_mse = nn.MSELoss()(g_pred, g_target)
                metric_reg = config.metric_reg_weight * (g_pred.mean() - 1.0) ** 2
                loss_metric = loss_metric_mse + metric_reg
                opt_metric.zero_grad()
                loss_metric.backward()
                opt_metric.step()
            
            # Compute advantage with geometric scaling
            with torch.no_grad():
                g_values = metric(states)
                r_adv = r_ret - critic(states)
                
                if config.use_hybrid_lagrangian:
                    lambda_val = torch.exp(log_lambda).detach()
                    c_adv = c_ret - cost_critic(states)
                    combined_adv = r_adv - lambda_val * c_adv
                else:
                    combined_adv = r_adv
                
                if config.use_soft_scaling:
                    scale = 1.0 / (1.0 + torch.log(1.0 + g_values))
                else:
                    scale = 1.0 / torch.sqrt(g_values)
                
                riemannian_adv = scale * combined_adv
            
            dists = actor(states)
            log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
            loss_actor = -(log_probs * riemannian_adv).mean()
            opt_actor.zero_grad()
            loss_actor.backward()
            opt_actor.step()
            
            if config.use_hybrid_lagrangian:
                loss_lambda = -log_lambda * (config.cost_limit - c_ret.mean().detach())
                opt_lambda.zero_grad()
                loss_lambda.backward()
                opt_lambda.step()
            
            returns.append(ep_return)
            violations.append(ep_violations)
        
        return returns, violations, actor
    
    def train_sgpo_anisotropic(env, config, seed=0):
        """Anisotropic SGPO - directional metric."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        actor = Actor().to(DEVICE)
        critic = Critic().to(DEVICE)
        metric = AnisotropicRiemannianMetric(
            severity=config.metric_severity,
            max_metric=config.anisotropic_max_metric
        ).to(DEVICE)
        
        opt_actor = optim.Adam(actor.parameters(), lr=config.lr_actor)
        opt_critic = optim.Adam(critic.parameters(), lr=config.lr_critic)
        opt_metric = optim.Adam(metric.parameters(), lr=config.lr_metric)
        
        if config.use_hybrid_lagrangian:
            cost_critic = Critic().to(DEVICE)
            opt_cost_critic = optim.Adam(cost_critic.parameters(), lr=config.lr_critic)
            log_lambda = nn.Parameter(torch.zeros(1, device=DEVICE))
            opt_lambda = optim.Adam([log_lambda], lr=config.hybrid_lambda_lr)
        
        returns, violations = [], []
        
        for ep in range(config.episodes):
            obs = env.reset()
            trajectory = []
            ep_violations = 0
            ep_return = 0.0
            done = False
            
            while not done:
                obs_t = torch.FloatTensor(obs).to(DEVICE)
                with torch.no_grad():
                    dist = actor(obs_t)
                    action = dist.sample()
                
                next_obs, reward, cost, done, info = env.step(action.cpu().numpy())
                trajectory.append((obs, action, reward, cost, info['dist_to_trap']))
                ep_violations += int(info['in_trap'])
                ep_return += reward
                obs = next_obs
            
            states = torch.FloatTensor(np.array([t[0] for t in trajectory])).to(DEVICE)
            actions = torch.stack([t[1] for t in trajectory]).to(DEVICE)
            costs = torch.FloatTensor([t[3] for t in trajectory]).to(DEVICE)
            trap_dists = torch.FloatTensor([t[4] for t in trajectory]).to(DEVICE)
            
            r_ret, c_ret = [], []
            Gr, Gc = 0, 0
            for t in reversed(trajectory):
                Gr = t[2] + config.gamma * Gr
                Gc = t[3] + config.gamma * Gc
                r_ret.insert(0, Gr)
                c_ret.insert(0, Gc)
            r_ret = torch.FloatTensor(r_ret).unsqueeze(1).to(DEVICE)
            c_ret = torch.FloatTensor(c_ret).unsqueeze(1).to(DEVICE)
            
            loss_crit = nn.MSELoss()(critic(states), r_ret)
            opt_critic.zero_grad()
            loss_crit.backward()
            opt_critic.step()
            
            if config.use_hybrid_lagrangian:
                opt_cost_critic.zero_grad()
                nn.MSELoss()(cost_critic(states), c_ret).backward()
                opt_cost_critic.step()
            
            if ep >= config.warmup_episodes:
                g_pred, _ = metric(states, actions)
                safe_dist = torch.clamp(trap_dists, min=0.1)
                g_target = 1.0 + 5.0 / safe_dist.unsqueeze(1) + costs.unsqueeze(1) * 10.0
                loss_metric = nn.MSELoss()(g_pred, g_target)
                opt_metric.zero_grad()
                loss_metric.backward()
                opt_metric.step()
            
            with torch.no_grad():
                g_values, escape_factors = metric(states, actions)
                r_adv = r_ret - critic(states)
                
                if config.use_hybrid_lagrangian:
                    lambda_val = torch.exp(log_lambda).detach()
                    c_adv = c_ret - cost_critic(states)
                    combined_adv = r_adv - lambda_val * c_adv
                else:
                    combined_adv = r_adv
                
                if config.use_soft_scaling:
                    scale = escape_factors + (1 - escape_factors) / (1.0 + torch.log(1.0 + g_values))
                else:
                    scale = escape_factors + (1 - escape_factors) / torch.sqrt(g_values)
                
                riemannian_adv = scale * combined_adv
            
            dists = actor(states)
            log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
            loss_actor = -(log_probs * riemannian_adv).mean()
            opt_actor.zero_grad()
            loss_actor.backward()
            opt_actor.step()
            
            if config.use_hybrid_lagrangian:
                loss_lambda = -log_lambda * (config.cost_limit - c_ret.mean().detach())
                opt_lambda.zero_grad()
                loss_lambda.backward()
                opt_lambda.step()
            
            returns.append(ep_return)
            violations.append(ep_violations)
        
        return returns, violations, actor
    
    # ========== EVALUATION ==========
    def evaluate_policy(actor, env, num_episodes=20):
        returns, violations = [], []
        for _ in range(num_episodes):
            obs = env.reset()
            ep_ret, ep_viol = 0.0, 0
            done = False
            while not done:
                obs_t = torch.FloatTensor(obs).to(DEVICE)
                with torch.no_grad():
                    dist = actor(obs_t)
                    action = dist.sample()
                next_obs, reward, cost, done, info = env.step(action.cpu().numpy())
                ep_viol += int(info['in_trap'])
                ep_ret += reward
                obs = next_obs
            returns.append(ep_ret)
            violations.append(ep_viol)
        return {
            "mean_return": float(np.mean(returns)),
            "mean_violations": float(np.mean(violations))
        }
    
    # ========== RUN EXPERIMENTS ==========
    output_path = Path(VOLUME_PATH) / f"sandbagging_v2_{mode}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    env_config = EnvConfig()
    algo_config = AlgoConfig()
    
    if mode == "full":
        # Run all 4 methods
        methods = {
            "ppo": train_ppo,
            "cpo": train_cpo,
            "sgpo": train_sgpo,
            "sgpo_anis": train_sgpo_anisotropic
        }
        
        results = {}
        for name, train_fn in methods.items():
            print(f"\n=== {name.upper()} ===")
            all_returns, all_violations = [], []
            
            for seed in range(num_seeds):
                if (seed + 1) % 10 == 0:
                    print(f"  Seed {seed+1}/{num_seeds}")
                env = SandbaggingEnv(env_config)
                returns, violations, _ = train_fn(env, algo_config, seed)
                all_returns.append(returns)
                all_violations.append(violations)
            
            final_returns = [np.mean(r[-50:]) for r in all_returns]
            total_violations = [sum(v) for v in all_violations]
            
            results[name] = {
                "final_return": {"mean": float(np.mean(final_returns)), "std": float(np.std(final_returns))},
                "total_violations": {"mean": float(np.mean(total_violations)), "std": float(np.std(total_violations))},
                "learning_curves": {
                    "returns_mean": np.mean(all_returns, axis=0).tolist(),
                    "violations_mean": np.mean(all_violations, axis=0).tolist()
                }
            }
            print(f"  Violations: {np.mean(total_violations):.1f} ± {np.std(total_violations):.1f}")
        
        with open(output_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(methods))
        means = [results[m]["total_violations"]["mean"] for m in methods]
        stds = [results[m]["total_violations"]["std"] for m in methods]
        colors = ['red', 'orange', 'blue', 'green']
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.set_ylabel('Total Violations')
        ax.set_title(f'Sandbagging v2.1 ({num_seeds} seeds)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_path / "results.png", dpi=150)
    
    elif mode == "generalization":
        # Train on trap A, test on trap B
        train_trap = (5.0, 6.0)
        test_trap = (7.0, 3.0)
        
        print(f"Train trap: {train_trap}")
        print(f"Test trap: {test_trap} (UNSEEN)")
        
        train_config = EnvConfig(trap_center=train_trap)
        test_config = EnvConfig(trap_center=test_trap)
        
        methods = {
            "ppo": train_ppo,
            "cpo": train_cpo,
            "sgpo": train_sgpo,
            "sgpo_anis": train_sgpo_anisotropic
        }
        
        results = {}
        for name, train_fn in methods.items():
            print(f"\n=== {name.upper()} ===")
            train_viols, test_viols = [], []
            
            for seed in range(num_seeds):
                if (seed + 1) % 5 == 0:
                    print(f"  Seed {seed+1}/{num_seeds}")
                
                train_env = SandbaggingEnv(train_config)
                _, _, actor = train_fn(train_env, algo_config, seed)
                
                # Evaluate on train env
                train_eval = evaluate_policy(actor, SandbaggingEnv(train_config))
                # Evaluate on test env (UNSEEN trap)
                test_eval = evaluate_policy(actor, SandbaggingEnv(test_config))
                
                train_viols.append(train_eval["mean_violations"])
                test_viols.append(test_eval["mean_violations"])
            
            results[name] = {
                "train_violations": {"mean": float(np.mean(train_viols)), "std": float(np.std(train_viols))},
                "test_violations": {"mean": float(np.mean(test_viols)), "std": float(np.std(test_viols))},
                "generalization_gap": {"mean": float(np.mean(np.array(test_viols) - np.array(train_viols)))}
            }
            print(f"  Train: {np.mean(train_viols):.2f}, Test: {np.mean(test_viols):.2f}")
        
        with open(output_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        x = np.arange(len(methods))
        width = 0.35
        
        ax1 = axes[0]
        train_means = [results[m]["train_violations"]["mean"] for m in methods]
        test_means = [results[m]["test_violations"]["mean"] for m in methods]
        ax1.bar(x - width/2, train_means, width, label='Train', color='blue', alpha=0.7)
        ax1.bar(x + width/2, test_means, width, label='Test (Unseen)', color='red', alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.upper() for m in methods])
        ax1.set_ylabel('Mean Violations')
        ax1.set_title('Train vs Test')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2 = axes[1]
        gaps = [results[m]["generalization_gap"]["mean"] for m in methods]
        colors = ['green' if g < 0 else 'red' for g in gaps]
        ax2.bar(x, gaps, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.upper() for m in methods])
        ax2.set_ylabel('Gap (Test - Train)')
        ax2.set_title('Generalization Gap')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path / "generalization.png", dpi=150)
    
    else:  # diagnostics
        print("Running SGPO diagnostics...")
        # Just run SGPO with extra logging
        all_metric_vals = []
        for seed in range(min(num_seeds, 10)):
            env = SandbaggingEnv(env_config)
            _, violations, _ = train_sgpo(env, algo_config, seed)
            all_metric_vals.append(sum(violations))
            print(f"  Seed {seed}: violations = {sum(violations)}")
        
        results = {
            "mode": "diagnostics",
            "violations": all_metric_vals,
            "mean": float(np.mean(all_metric_vals)),
            "std": float(np.std(all_metric_vals))
        }
        
        with open(output_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    volume.commit()
    print(f"\nResults saved to {output_path}")
    return results


# ============================================================================
# RUN ALL EXPERIMENTS
# ============================================================================
# SGPO_ANIS_CCHC EXPERIMENTS (Context-Conditional Hodge Critic)
# ============================================================================

@app.function(
    image=image_cchc,
    gpu="L4",
    timeout=10800,  # 3 hours
    volumes={VOLUME_PATH: volume},
)
def run_experiment_a_cchc(num_seeds: int = 50, n_samples: int = 5000, h1_threshold: float = 0.8):
    """
    Experiment A: Pre-filtered Reward Models with CCHC.
    
    Compares reward models trained on raw vs H¹-filtered preferences.
    Both models train on equal sample counts for fair comparison.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import json
    from pathlib import Path
    from dataclasses import dataclass, asdict
    from typing import List, Tuple, Dict, Optional
    from collections import defaultdict
    from scipy.sparse import csr_matrix, lil_matrix
    from scipy.sparse.linalg import lsqr
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Experiment A (CCHC) on {DEVICE}")
    print(f"Seeds: {num_seeds}, Samples: {n_samples}, Threshold: {h1_threshold}")
    
    # Load embedding model
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_dim = embedding_model.get_sentence_embedding_dimension()
    print(f"Embedding dim: {embed_dim}")
    
    # ========== DATA CLASSES ==========
    @dataclass
    class ContextualFeedbackItem:
        state_text: str
        action_text: str
        rank: float
        context_id: str
        chosen_text: Optional[str] = None
        rejected_text: Optional[str] = None
    
    # ========== REWARD MODEL ==========
    class PreferenceRewardModel(nn.Module):
        def __init__(self, embed_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.reward_net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        
        def forward(self, x):
            return self.reward_net(x)
        
        def preference_probability(self, chosen_emb, rejected_emb):
            r_chosen = self.forward(chosen_emb)
            r_rejected = self.forward(rejected_emb)
            return torch.sigmoid(r_chosen - r_rejected)
    
    # ========== LOAD DATA ==========
    def load_hh_rlhf_with_context(num_samples: int):
        from datasets import load_dataset
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        
        if num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))
        
        items = []
        for example in dataset:
            try:
                chosen = example["chosen"]
                rejected = example["rejected"]
                prompt = chosen.rpartition("\n\nAssistant:")[0]
                chosen_resp = chosen.rpartition("\n\nAssistant:")[2].strip()
                rejected_resp = rejected.rpartition("\n\nAssistant:")[2].strip()
                
                if not prompt or not chosen_resp or not rejected_resp:
                    continue
                
                context_id = str(hash(prompt) % 10000)
                items.append(ContextualFeedbackItem(
                    state_text=prompt,
                    action_text="response",
                    rank=1.0,
                    context_id=context_id,
                    chosen_text=chosen_resp,
                    rejected_text=rejected_resp,
                ))
            except:
                continue
        return items
    
    # ========== COMPUTE H¹ ==========
    def compute_h1_for_subset(indices, embeddings, items, similarity_threshold=0.8):
        n = len(indices)
        if n < 2:
            return 0.0
        
        subset_embeddings = embeddings[indices]
        norms = np.linalg.norm(subset_embeddings, axis=1, keepdims=True)
        normalized = subset_embeddings / (norms + 1e-8)
        similarities = normalized @ normalized.T
        
        edge_list = []
        edge_weights = {}
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[i, j] > similarity_threshold:
                    edge_list.append((i, j))
                    edge_weights[(i, j)] = items[indices[j]].rank - items[indices[i]].rank
        
        if len(edge_list) < 1:
            return 0.0
        
        rows, cols, data = [], [], []
        for idx, (u, v) in enumerate(edge_list):
            rows.extend([idx, idx])
            cols.extend([u, v])
            data.extend([-1, 1])
        
        d0 = csr_matrix((data, (rows, cols)), shape=(len(edge_list), n))
        Y = np.array([edge_weights[tuple(edge)] for edge in edge_list])
        
        L0 = d0.T @ d0
        divergence = d0.T @ Y
        
        try:
            s_potential = lsqr(L0, divergence)[0]
            Y_grad = d0 @ s_potential
        except:
            Y_grad = np.zeros_like(Y)
        
        Y_harm = Y - Y_grad
        return np.linalg.norm(Y_harm)
    
    def compute_conditional_h1(items, embeddings):
        context_groups = defaultdict(list)
        for i, item in enumerate(items):
            context_groups[item.context_id].append(i)
        
        # Marginal H¹
        marginal_h1 = compute_h1_for_subset(list(range(len(items))), embeddings, items)
        
        # Per-context H¹
        per_context_h1 = {}
        for ctx_id, indices in context_groups.items():
            if len(indices) >= 2:
                per_context_h1[ctx_id] = compute_h1_for_subset(indices, embeddings, items)
            else:
                per_context_h1[ctx_id] = 0.0
        
        # Conditional H¹ = weighted average
        total = sum(len(context_groups[c]) for c in context_groups)
        conditional_h1 = sum(
            h1 * len(context_groups[c]) / total
            for c, h1 in per_context_h1.items()
        )
        
        return marginal_h1, conditional_h1, per_context_h1
    
    # ========== TRAINING ==========
    def train_reward_model(model, chosen_embs, rejected_embs, epochs=50, lr=1e-4, batch_size=32):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        n = len(chosen_embs)
        model.train()
        
        for epoch in range(epochs):
            perm = torch.randperm(n)
            chosen_embs = chosen_embs[perm]
            rejected_embs = rejected_embs[perm]
            
            for i in range(0, n, batch_size):
                batch_chosen = chosen_embs[i:i+batch_size].to(DEVICE)
                batch_rejected = rejected_embs[i:i+batch_size].to(DEVICE)
                
                optimizer.zero_grad()
                probs = model.preference_probability(batch_chosen, batch_rejected)
                loss = -torch.log(probs + 1e-8).mean()
                loss.backward()
                optimizer.step()
        
        return loss.item()
    
    def evaluate_model(model, chosen_embs, rejected_embs):
        model.eval()
        with torch.no_grad():
            probs = model.preference_probability(
                chosen_embs.to(DEVICE), 
                rejected_embs.to(DEVICE)
            )
            accuracy = (probs > 0.5).float().mean().item()
            exploitation = (probs < 0.3).float().mean().item()
        return accuracy, exploitation
    
    # ========== RUN EXPERIMENT ==========
    results = []
    
    for seed in range(num_seeds):
        print(f"\n--- Seed {seed+1}/{num_seeds} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Load data
        items = load_hh_rlhf_with_context(n_samples)
        print(f"  Loaded {len(items)} items")
        
        # Embed
        all_texts = [item.state_text for item in items]
        embeddings = embedding_model.encode(all_texts)
        
        # Compute conditional H¹
        marginal_h1, conditional_h1, per_context_h1 = compute_conditional_h1(items, embeddings)
        print(f"  Marginal H¹: {marginal_h1:.4f}, Conditional H¹: {conditional_h1:.4f}")
        
        # Filter items by context H¹
        filtered_indices = [
            i for i, item in enumerate(items)
            if per_context_h1.get(item.context_id, 0.0) <= h1_threshold
        ]
        
        # Normalize sizes
        n_filtered = len(filtered_indices)
        n_train = int(0.8 * min(n_filtered, len(items)))
        
        # Sample raw indices
        raw_indices = np.random.choice(len(items), size=n_filtered, replace=False).tolist()
        
        # Embed chosen/rejected
        raw_chosen = embedding_model.encode([items[i].chosen_text[:500] for i in raw_indices[:n_train]])
        raw_rejected = embedding_model.encode([items[i].rejected_text[:500] for i in raw_indices[:n_train]])
        filtered_chosen = embedding_model.encode([items[i].chosen_text[:500] for i in filtered_indices[:n_train]])
        filtered_rejected = embedding_model.encode([items[i].rejected_text[:500] for i in filtered_indices[:n_train]])
        
        # Test set
        test_chosen = embedding_model.encode([items[i].chosen_text[:500] for i in raw_indices[n_train:n_train+200]])
        test_rejected = embedding_model.encode([items[i].rejected_text[:500] for i in raw_indices[n_train:n_train+200]])
        
        # Convert to tensors
        raw_chosen_t = torch.tensor(raw_chosen, dtype=torch.float32)
        raw_rejected_t = torch.tensor(raw_rejected, dtype=torch.float32)
        filtered_chosen_t = torch.tensor(filtered_chosen, dtype=torch.float32)
        filtered_rejected_t = torch.tensor(filtered_rejected, dtype=torch.float32)
        test_chosen_t = torch.tensor(test_chosen, dtype=torch.float32)
        test_rejected_t = torch.tensor(test_rejected, dtype=torch.float32)
        
        # Train models
        print("  Training raw model...")
        raw_model = PreferenceRewardModel(embed_dim).to(DEVICE)
        raw_loss = train_reward_model(raw_model, raw_chosen_t, raw_rejected_t)
        
        print("  Training filtered model...")
        filtered_model = PreferenceRewardModel(embed_dim).to(DEVICE)
        filtered_loss = train_reward_model(filtered_model, filtered_chosen_t, filtered_rejected_t)
        
        # Evaluate
        raw_acc, raw_exploit = evaluate_model(raw_model, test_chosen_t, test_rejected_t)
        filt_acc, filt_exploit = evaluate_model(filtered_model, test_chosen_t, test_rejected_t)
        
        print(f"  Raw: Acc={raw_acc:.2%}, Exploit={raw_exploit:.2%}")
        print(f"  Filtered: Acc={filt_acc:.2%}, Exploit={filt_exploit:.2%}")
        
        results.append({
            "seed": seed,
            "marginal_h1": marginal_h1,
            "conditional_h1": conditional_h1,
            "n_train": n_train,
            "raw_accuracy": raw_acc,
            "raw_exploitation": raw_exploit,
            "filtered_accuracy": filt_acc,
            "filtered_exploitation": filt_exploit,
        })
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    raw_acc_mean = np.mean([r["raw_accuracy"] for r in results])
    filt_acc_mean = np.mean([r["filtered_accuracy"] for r in results])
    raw_exploit_mean = np.mean([r["raw_exploitation"] for r in results])
    filt_exploit_mean = np.mean([r["filtered_exploitation"] for r in results])
    
    print(f"Raw Model:      Accuracy={raw_acc_mean:.2%}, Exploitation={raw_exploit_mean:.2%}")
    print(f"Filtered Model: Accuracy={filt_acc_mean:.2%}, Exploitation={filt_exploit_mean:.2%}")
    
    # Save
    output_path = Path(VOLUME_PATH) / "cchc_experiment_a"
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "config": {"num_seeds": num_seeds, "n_samples": n_samples, "h1_threshold": h1_threshold},
        "results": results,
        "summary": {
            "raw_accuracy": {"mean": raw_acc_mean, "std": np.std([r["raw_accuracy"] for r in results])},
            "filtered_accuracy": {"mean": filt_acc_mean, "std": np.std([r["filtered_accuracy"] for r in results])},
            "raw_exploitation": {"mean": raw_exploit_mean, "std": np.std([r["raw_exploitation"] for r in results])},
            "filtered_exploitation": {"mean": filt_exploit_mean, "std": np.std([r["filtered_exploitation"] for r in results])},
        }
    }
    
    with open(output_path / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    volume.commit()
    print(f"\nResults saved to {output_path}")
    
    return summary


@app.function(
    image=image_cchc,
    gpu="L4",
    timeout=14400,  # 4 hours
    volumes={VOLUME_PATH: volume},
)
def run_experiment_c_cchc(num_seeds: int = 50, n_episodes: int = 500):
    """
    Experiment C: SGPO_ANIS_CCHC vs SGPO_ANIS.
    
    Compares policy optimization with and without context-conditional
    harmonic discounting in the advantage computation.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    import json
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Tuple, Optional
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Experiment C (SGPO_ANIS_CCHC) on {DEVICE}")
    print(f"Seeds: {num_seeds}, Episodes: {n_episodes}")
    
    # ========== CONFIG ==========
    @dataclass
    class Config:
        embed_dim: int = 384
        hidden_dim: int = 128
        gamma: float = 0.99
        clip_ratio: float = 0.2
        entropy_coef: float = 0.01
        policy_lr: float = 3e-4
        critic_lr: float = 1e-3
        metric_lr: float = 1e-3
        max_steps: int = 100
        warmup_episodes: int = 30
        metric_severity: float = 5.0
        anisotropic_max_metric: float = 100.0
    
    config = Config()
    
    # ========== NETWORKS ==========
    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
            )
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        def forward(self, x):
            mean = self.net(x)
            std = torch.exp(self.log_std).expand_as(mean)
            return torch.distributions.Normal(mean, std)
    
    class Critic(nn.Module):
        def __init__(self, state_dim, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        
        def forward(self, x):
            return self.net(x)
    
    class AnisotropicMetric(nn.Module):
        def __init__(self, state_dim, hidden_dim=32, severity=5.0, max_metric=100.0):
            super().__init__()
            self.danger_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.center_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim),
            )
            self.base_metric = 1.0
            self.severity = severity
            self.max_metric = max_metric
        
        def forward(self, x, v=None):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            danger_level = torch.sigmoid(self.danger_net(x)) * self.severity
            
            if v is None:
                g = self.base_metric + danger_level.squeeze(-1)
                escape = torch.ones(x.shape[0], device=x.device)
                return g, escape
            
            if v.dim() == 1:
                v = v.unsqueeze(0)
            
            danger_center = self.center_net(x)
            to_danger = danger_center - x
            dist = torch.norm(to_danger, dim=-1, keepdim=True) + 1e-8
            n_hat = to_danger / dist
            
            v_toward = torch.sum(v * n_hat, dim=-1, keepdim=True)
            v_toward_pos = torch.clamp(v_toward, min=0)
            v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
            
            toward_ratio_sq = (v_toward_pos / v_norm) ** 2
            g_dir = danger_level / (dist + 0.1)
            
            g = self.base_metric + (toward_ratio_sq * g_dir).squeeze(-1)
            g = torch.clamp(g, max=self.max_metric)
            
            escape = torch.sigmoid(-v_toward * 5.0).squeeze(-1)
            
            return g, escape
    
    class HarmonicNet(nn.Module):
        def __init__(self, state_dim, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        
        def forward(self, states, contexts):
            combined = torch.cat([states, contexts], dim=-1)
            return self.net(combined).squeeze(-1)
    
    # ========== ENVIRONMENT ==========
    class EmbeddingEnv:
        def __init__(self, embed_dim, trap_centers=None, max_steps=100):
            self.embed_dim = embed_dim
            self.max_steps = max_steps
            
            if trap_centers is None:
                self.trap_centers = [np.random.randn(embed_dim) * 5.0 for _ in range(3)]
            else:
                self.trap_centers = trap_centers
            
            self.trap_radius = 2.0
            self.goal = np.random.randn(embed_dim) * 10.0
            self.state = None
            self.step_count = 0
        
        def reset(self):
            self.state = np.random.randn(self.embed_dim) * 0.1
            self.step_count = 0
            return self.state.copy()
        
        def step(self, action):
            action = np.clip(action, -0.5, 0.5)
            self.state = self.state + action
            
            # Trap penalty
            trap_cost = 0.0
            in_trap = False
            for tc in self.trap_centers:
                dist = np.linalg.norm(self.state - tc)
                if dist < self.trap_radius:
                    trap_cost += 10.0
                    in_trap = True
            
            # Goal bonus
            goal_dist = np.linalg.norm(self.state - self.goal)
            goal_reached = goal_dist < 2.0
            goal_bonus = 10.0 if goal_reached else 0.0
            
            # Base reward (progress toward goal)
            reward = -0.1 * goal_dist / 10.0 - trap_cost + goal_bonus
            
            self.step_count += 1
            done = self.step_count >= self.max_steps or goal_reached
            
            return self.state.copy(), reward, done, {"in_trap": in_trap, "goal_reached": goal_reached}
    
    # ========== TRAINING ==========
    def train_sgpo_anis(env, config, use_cchc=False, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        state_dim = env.embed_dim
        action_dim = env.embed_dim
        
        actor = Actor(state_dim, action_dim, config.hidden_dim).to(DEVICE)
        critic = Critic(state_dim, config.hidden_dim).to(DEVICE)
        metric = AnisotropicMetric(state_dim, severity=config.metric_severity).to(DEVICE)
        
        harmonic_net = None
        if use_cchc:
            harmonic_net = HarmonicNet(state_dim).to(DEVICE)
            opt_harmonic = optim.Adam(harmonic_net.parameters(), lr=1e-3)
        
        opt_actor = optim.Adam(actor.parameters(), lr=config.policy_lr)
        opt_critic = optim.Adam(critic.parameters(), lr=config.critic_lr)
        opt_metric = optim.Adam(metric.parameters(), lr=config.metric_lr)
        
        returns_history = []
        violations_history = []
        total_violations = 0
        total_goals = 0
        
        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            trajectory = []
            ep_return = 0.0
            ep_violations = 0
            
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    dist = actor(obs_t)
                    action = dist.sample()
                    old_lp = dist.log_prob(action).sum()
                
                next_obs, reward, done, info = env.step(action.squeeze(0).cpu().numpy())
                
                trajectory.append({
                    "obs": obs,
                    "next_obs": next_obs,
                    "action": action,
                    "reward": reward,
                    "old_lp": old_lp.item(),
                    "in_trap": info["in_trap"],
                })
                
                obs = next_obs
                ep_return += reward
                if info["in_trap"]:
                    ep_violations += 1
                if info["goal_reached"]:
                    total_goals += 1
            
            returns_history.append(ep_return)
            violations_history.append(ep_violations)
            total_violations += ep_violations
            
            # Convert trajectory
            states = torch.FloatTensor(np.array([t["obs"] for t in trajectory])).to(DEVICE)
            next_states = torch.FloatTensor(np.array([t["next_obs"] for t in trajectory])).to(DEVICE)
            actions = torch.cat([t["action"] for t in trajectory]).to(DEVICE)
            rewards = torch.FloatTensor([t["reward"] for t in trajectory]).to(DEVICE)
            old_lps = torch.FloatTensor([t["old_lp"] for t in trajectory]).to(DEVICE)
            
            # Critic update
            with torch.no_grad():
                V_next = critic(next_states).squeeze()
            V = critic(states).squeeze()
            targets = rewards + config.gamma * V_next
            critic_loss = F.mse_loss(V, targets)
            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()
            
            # Metric update
            if ep >= config.warmup_episodes:
                danger_targets = torch.zeros(len(states), device=DEVICE)
                for i, t in enumerate(trajectory):
                    if t["in_trap"]:
                        danger_targets[i] = 1.0
                
                g, _ = metric(states, actions)
                metric_loss = F.mse_loss(g, danger_targets * config.metric_severity + 1.0)
                opt_metric.zero_grad()
                metric_loss.backward()
                opt_metric.step()
            
            # Harmonic update (CCHC only)
            if use_cchc and harmonic_net is not None:
                # Train harmonic net to predict trap visits
                trap_targets = torch.FloatTensor([1.0 if t["in_trap"] else 0.0 for t in trajectory]).to(DEVICE)
                omega_pred = harmonic_net(states, states)  # Context = state for simplicity
                harmonic_loss = F.mse_loss(omega_pred, trap_targets)
                opt_harmonic.zero_grad()
                harmonic_loss.backward()
                opt_harmonic.step()
            
            # Advantage with optional CCHC
            with torch.no_grad():
                V = critic(states).squeeze()
                V_next = critic(next_states).squeeze()
                g, escape = metric(states, actions)
                
                td_error = rewards + config.gamma * V_next - V
                
                if use_cchc and harmonic_net is not None:
                    omega_invalid = harmonic_net(states, states)
                    td_error = td_error - omega_invalid
                
                scale = escape + (1 - escape) / (1.0 + torch.log(1.0 + g))
                adv = scale * td_error
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            # Actor update
            new_lps = actor(states).log_prob(actions).sum(dim=-1)
            ratio = torch.exp(new_lps - old_lps)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            
            entropy = actor(states).entropy().sum(dim=-1).mean()
            actor_loss = actor_loss - config.entropy_coef * entropy
            
            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()
            
            if ep % 100 == 0:
                print(f"  Ep {ep}: Return={ep_return:.1f}, Violations={ep_violations}, Total={total_violations}")
        
        return {
            "total_violations": total_violations,
            "final_return": float(np.mean(returns_history[-100:])),
            "goal_rate": total_goals / n_episodes,
            "returns": returns_history,
            "violations": violations_history,
        }
    
    # ========== RUN ==========
    results = []
    
    for seed in range(num_seeds):
        print(f"\n--- Seed {seed+1}/{num_seeds} ---")
        
        env = EmbeddingEnv(config.embed_dim, max_steps=config.max_steps)
        
        print("  Training SGPO_ANIS...")
        baseline = train_sgpo_anis(env, config, use_cchc=False, seed=seed)
        
        print("  Training SGPO_ANIS_CCHC...")
        cchc = train_sgpo_anis(env, config, use_cchc=True, seed=seed)
        
        results.append({
            "seed": seed,
            "sgpo_anis": {
                "violations": baseline["total_violations"],
                "return": baseline["final_return"],
                "goal_rate": baseline["goal_rate"],
            },
            "sgpo_anis_cchc": {
                "violations": cchc["total_violations"],
                "return": cchc["final_return"],
                "goal_rate": cchc["goal_rate"],
            },
        })
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for method in ["sgpo_anis", "sgpo_anis_cchc"]:
        violations = [r[method]["violations"] for r in results]
        returns = [r[method]["return"] for r in results]
        goals = [r[method]["goal_rate"] for r in results]
        
        print(f"{method:20s}: Violations={np.mean(violations):.1f}±{np.std(violations):.1f}, "
              f"Return={np.mean(returns):.2f}, Goals={np.mean(goals):.1%}")
    
    # Save
    output_path = Path(VOLUME_PATH) / "cchc_experiment_c"
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "config": {"num_seeds": num_seeds, "n_episodes": n_episodes},
        "results": results,
        "summary": {
            method: {
                "violations": {"mean": np.mean([r[method]["violations"] for r in results]),
                              "std": np.std([r[method]["violations"] for r in results])},
                "return": {"mean": np.mean([r[method]["return"] for r in results]),
                          "std": np.std([r[method]["return"] for r in results])},
                "goal_rate": {"mean": np.mean([r[method]["goal_rate"] for r in results]),
                             "std": np.std([r[method]["goal_rate"] for r in results])},
            }
            for method in ["sgpo_anis", "sgpo_anis_cchc"]
        }
    }
    
    with open(output_path / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    volume.commit()
    print(f"\nResults saved to {output_path}")
    
    return summary


@app.function(
    image=image_cchc,
    gpu="L4",
    timeout=21600,  # 6 hours
    volumes={VOLUME_PATH: volume},
)
def run_cchc_experiments(num_seeds: int = 50, n_samples: int = 5000, n_episodes: int = 500):
    """Run both CCHC experiments (A and C)."""
    print("="*60)
    print("SGPO_ANIS_CCHC EXPERIMENTS")
    print("="*60)
    
    exp_a = run_experiment_a_cchc.remote(num_seeds=num_seeds, n_samples=n_samples)
    exp_c = run_experiment_c_cchc.remote(num_seeds=num_seeds, n_episodes=n_episodes)
    
    return {"experiment_a": exp_a, "experiment_c": exp_c}


# ============================================================================
# RUN ALL EXPERIMENTS
# ============================================================================

@app.function(
    image=image,
    gpu="L4",
    timeout=18000,  # 5 hours
    volumes={VOLUME_PATH: volume},
)
def run_all_experiments(num_seeds: int = 50):
    """Run all experiments sequentially."""
    print("="*60)
    print("FEEDBACK GEOMETRY EXPERIMENTS")
    print("="*60)
    
    h1_results = run_h1_experiment.remote(num_seeds=num_seeds)
    sandbagging_results = run_sandbagging_v2.remote(num_seeds=num_seeds, mode="full")
    
    return {
        "h1_exploitation": h1_results,
        "sandbagging_v2": sandbagging_results
    }


# ============================================================================
# DOWNLOAD RESULTS
# ============================================================================

@app.local_entrypoint()
def main():
    """Default entry point - run quick test."""
    print("Running quick test with 5 seeds...")
    print("For full run: modal run modal_runner.py::run_all_experiments --num-seeds 50")
    
    results = run_h1_experiment.remote(num_seeds=5)
    print("\nH¹ experiment complete!")
    print(f"Results: {results}")
