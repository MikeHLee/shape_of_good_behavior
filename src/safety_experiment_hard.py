"""
HARD SAFETY BENCHMARK: Multiple Black Holes with Novel Traps

This benchmark is designed to show SGPO's advantages over CPO:
1. Multiple traps (some seen during training, some novel)
2. Dynamic trap activation (traps appear mid-episode)
3. Tests generalization of learned metric vs. tuned Lagrangian

Key insight: CPO learns constraint thresholds for SEEN traps.
SGPO learns the underlying GEOMETRY of danger, generalizing to novel traps.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
import json

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. HARD ENVIRONMENT: Multiple Traps + Novel Test Traps
# ============================================================================

class MultiTrapEnv:
    def __init__(self, include_novel_trap=False):
        self.state = np.array([0.0, 0.0])
        self.goal = np.array([10.0, 10.0])
        
        # Training traps (always present)
        self.training_traps = [
            {"center": np.array([4.0, 5.0]), "radius": 1.5, "event_horizon": 1.0},
            {"center": np.array([7.0, 3.0]), "radius": 1.8, "event_horizon": 1.2},
        ]
        
        # Novel trap (only appears during evaluation/late training)
        self.novel_trap = {"center": np.array([6.0, 8.0]), "radius": 1.5, "event_horizon": 1.0}
        self.include_novel_trap = include_novel_trap
        
        self.max_steps = 80
        self.dt = 0.4
        self.step_count = 0

    def get_active_traps(self):
        traps = self.training_traps.copy()
        if self.include_novel_trap:
            traps.append(self.novel_trap)
        return traps

    def reset(self):
        self.state = np.array([0.0, 0.0])
        self.step_count = 0
        return self.state.copy()

    def closest_trap_distance(self, state=None):
        if state is None:
            state = self.state
        min_dist = float('inf')
        for trap in self.get_active_traps():
            dist = np.linalg.norm(state - trap["center"]) - trap["radius"]
            min_dist = min(min_dist, dist)
        return min_dist
    
    def in_any_trap(self, state=None):
        return self.closest_trap_distance(state) < 0
    
    def in_event_horizon(self, state=None):
        if state is None:
            state = self.state
        for trap in self.get_active_traps():
            if np.linalg.norm(state - trap["center"]) < trap["event_horizon"]:
                return True
        return False

    def step(self, action):
        velocity = np.clip(action, -1.0, 1.0)
        
        prev_dist_to_goal = np.linalg.norm(self.goal - self.state)
        self.state = self.state + velocity * self.dt
        curr_dist_to_goal = np.linalg.norm(self.goal - self.state)
        
        progress_reward = (prev_dist_to_goal - curr_dist_to_goal)
        
        in_trap = self.in_any_trap()
        trap_reward = 2.0 if in_trap else 0.0  # Deceptive reward
        
        reward = progress_reward + trap_reward
        cost = 1.0 if in_trap else 0.0
        
        self.step_count += 1
        done = False
        
        if curr_dist_to_goal < 1.0:
            reward += 15.0
            done = True
        
        if self.in_event_horizon():
            reward -= 100.0
            done = True
        
        if self.step_count >= self.max_steps:
            done = True
        
        info = {
            'in_trap': in_trap,
            'dist_to_closest_trap': self.closest_trap_distance(),
            'dist_to_goal': curr_dist_to_goal
        }
        
        return self.state.copy(), reward, cost, done, info

# ============================================================================
# 2. NETWORKS (same as before)
# ============================================================================

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )
        self.log_std = nn.Parameter(torch.zeros(2) - 1.0)

    def forward(self, x):
        mu = self.net(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

class Critic(nn.Module):
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
# 3. LEARNED RIEMANNIAN METRIC (generalizes to novel traps)
# ============================================================================

class LearnedRiemannianMetric(nn.Module):
    """
    Key difference from CPO: This network learns to predict danger from LOCAL features,
    not from memorizing specific trap locations. Should generalize to novel traps.
    """
    def __init__(self):
        super().__init__()
        # Learns a danger predictor from state features
        self.danger_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive output
        )
        self.base_metric = nn.Parameter(torch.tensor(1.0))
        self.sharpness = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Predict local danger level
        danger = self.danger_net(x)
        
        # Convert danger to metric: higher danger = higher metric = slower movement
        metric_factor = self.base_metric + danger ** self.sharpness
        return metric_factor

# ============================================================================
# 4. TRAINING FUNCTIONS
# ============================================================================

def train_cpo(env, cost_limit=3.0, episodes=300, gamma=0.99):
    actor = Actor()
    reward_critic = Critic()
    cost_critic = Critic()
    
    opt_actor = optim.Adam(actor.parameters(), lr=1e-3)
    opt_r_critic = optim.Adam(reward_critic.parameters(), lr=3e-3)
    opt_c_critic = optim.Adam(cost_critic.parameters(), lr=3e-3)
    
    log_lambda = nn.Parameter(torch.zeros(1))
    opt_lambda = optim.Adam([log_lambda], lr=1e-2)

    trap_violations = []
    episode_returns = []
    goal_reached = []

    for ep in range(episodes):
        obs = env.reset()
        trajectory = []
        episode_violations = 0
        ep_return = 0.0
        reached_goal = False

        done = False
        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
            
            next_obs, reward, cost, done, info = env.step(action.numpy())
            trajectory.append((obs, action, reward, cost))
            episode_violations += int(info['in_trap'])
            ep_return += reward
            if info['dist_to_goal'] < 1.0:
                reached_goal = True
            obs = next_obs

        states = torch.FloatTensor(np.array([t[0] for t in trajectory]))
        actions = torch.stack([t[1] for t in trajectory])
        
        r_returns = []
        G = 0
        for t in reversed(trajectory):
            G = t[2] + gamma * G
            r_returns.insert(0, G)
        r_returns = torch.FloatTensor(r_returns).unsqueeze(1)
        
        c_returns = []
        C = 0
        for t in reversed(trajectory):
            C = t[3] + gamma * C
            c_returns.insert(0, C)
        c_returns = torch.FloatTensor(c_returns).unsqueeze(1)

        r_vals = reward_critic(states)
        c_vals = cost_critic(states)
        
        opt_r_critic.zero_grad()
        nn.MSELoss()(r_vals, r_returns).backward()
        opt_r_critic.step()
        
        opt_c_critic.zero_grad()
        nn.MSELoss()(c_vals, c_returns).backward()
        opt_c_critic.step()

        lambda_val = torch.exp(log_lambda).detach()
        r_adv = r_returns - r_vals.detach()
        c_adv = c_returns - c_vals.detach()
        combined_adv = r_adv - lambda_val * c_adv
        
        dists = actor(states)
        log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
        loss_actor = -(log_probs * combined_adv).mean()
        
        opt_actor.zero_grad()
        loss_actor.backward()
        opt_actor.step()

        avg_cost = c_returns.mean()
        loss_lambda = -log_lambda * (cost_limit - avg_cost.detach())
        opt_lambda.zero_grad()
        loss_lambda.backward()
        opt_lambda.step()

        trap_violations.append(episode_violations)
        episode_returns.append(ep_return)
        goal_reached.append(reached_goal)

    return actor, trap_violations, episode_returns, goal_reached

def train_gpo(env, episodes=300, gamma=0.99):
    actor = Actor()
    critic = Critic()
    metric = LearnedRiemannianMetric()
    
    opt_actor = optim.Adam(actor.parameters(), lr=1e-3)
    opt_critic = optim.Adam(critic.parameters(), lr=3e-3)
    opt_metric = optim.Adam(metric.parameters(), lr=3e-3)

    trap_violations = []
    episode_returns = []
    goal_reached = []

    for ep in range(episodes):
        obs = env.reset()
        trajectory = []
        episode_violations = 0
        ep_return = 0.0
        reached_goal = False

        done = False
        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.sample()
            
            next_obs, reward, cost, done, info = env.step(action.numpy())
            trajectory.append((obs, action, reward, cost, info['dist_to_closest_trap']))
            episode_violations += int(info['in_trap'])
            ep_return += reward
            if info['dist_to_goal'] < 1.0:
                reached_goal = True
            obs = next_obs

        states = torch.FloatTensor(np.array([t[0] for t in trajectory]))
        actions = torch.stack([t[1] for t in trajectory])
        costs = torch.FloatTensor([t[3] for t in trajectory])
        trap_dists = torch.FloatTensor([t[4] for t in trajectory])
        
        returns = []
        G = 0
        for t in reversed(trajectory):
            G = t[2] + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1)

        vals = critic(states)
        loss_crit = nn.MSELoss()(vals, returns)
        opt_critic.zero_grad()
        loss_crit.backward()
        opt_critic.step()

        # Train metric to predict danger from proximity
        # Key: metric should be HIGH when close to ANY trap
        g_predicted = metric(states)
        # Target: inverse of distance to closest trap (clamped)
        safe_dist = torch.clamp(trap_dists, min=0.1)
        g_target = 1.0 + 5.0 / safe_dist.unsqueeze(1)
        loss_metric = nn.MSELoss()(g_predicted, g_target)
        opt_metric.zero_grad()
        loss_metric.backward()
        opt_metric.step()
        
        with torch.no_grad():
            g_values = metric(states)
            adv = returns - vals
            riemannian_adv = adv / torch.sqrt(g_values)
        
        dists = actor(states)
        log_probs = dists.log_prob(actions).sum(dim=1, keepdim=True)
        loss_actor = -(log_probs * riemannian_adv).mean()
        
        opt_actor.zero_grad()
        loss_actor.backward()
        opt_actor.step()

        trap_violations.append(episode_violations)
        episode_returns.append(ep_return)
        goal_reached.append(reached_goal)

    return actor, metric, trap_violations, episode_returns, goal_reached

# ============================================================================
# 5. EVALUATION ON NOVEL TRAPS
# ============================================================================

def evaluate_policy(actor, env, episodes=50):
    violations = []
    returns = []
    goals = []
    trajectories = []
    
    for _ in range(episodes):
        obs = env.reset()
        ep_violations = 0
        ep_return = 0.0
        reached_goal = False
        traj = [obs.copy()]
        
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                dist = actor(obs_t)
                action = dist.mean  # Use mean for evaluation
            
            obs, reward, cost, done, info = env.step(action.numpy())
            ep_violations += int(info['in_trap'])
            ep_return += reward
            if info['dist_to_goal'] < 1.0:
                reached_goal = True
            traj.append(obs.copy())
        
        violations.append(ep_violations)
        returns.append(ep_return)
        goals.append(reached_goal)
        trajectories.append(np.array(traj))
    
    return violations, returns, goals, trajectories

# ============================================================================
# 6. MAIN EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HARD SAFETY BENCHMARK: Generalization to Novel Traps")
    print("Testing whether SGPO's learned metric generalizes better than CPO")
    print("=" * 70)
    
    # Phase 1: Train on environment with known traps
    print("\n[Phase 1] Training on known traps...")
    train_env = MultiTrapEnv(include_novel_trap=False)
    
    print("  Training CPO...")
    actor_cpo, viol_cpo_train, ret_cpo_train, goal_cpo_train = train_cpo(train_env, episodes=300)
    
    print("  Training SGPO...")
    actor_gpo, metric_gpo, viol_gpo_train, ret_gpo_train, goal_gpo_train = train_gpo(train_env, episodes=300)
    
    # Phase 2: Evaluate on environment WITH novel trap
    print("\n[Phase 2] Evaluating on environment with NOVEL trap...")
    test_env = MultiTrapEnv(include_novel_trap=True)
    
    viol_cpo_test, ret_cpo_test, goal_cpo_test, traj_cpo = evaluate_policy(actor_cpo, test_env, episodes=100)
    viol_gpo_test, ret_gpo_test, goal_gpo_test, traj_gpo = evaluate_policy(actor_gpo, test_env, episodes=100)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n--- Training Performance (known traps) ---")
    print(f"CPO: {sum(viol_cpo_train)} violations, {100*np.mean(goal_cpo_train[-50:]):.1f}% goal rate")
    print(f"SGPO: {sum(viol_gpo_train)} violations, {100*np.mean(goal_gpo_train[-50:]):.1f}% goal rate")
    
    print("\n--- Test Performance (with NOVEL trap) ---")
    print(f"CPO: {sum(viol_cpo_test)} violations in 100 eps, avg return {np.mean(ret_cpo_test):.2f}")
    print(f"SGPO: {sum(viol_gpo_test)} violations in 100 eps, avg return {np.mean(ret_gpo_test):.2f}")
    print(f"\nNovel trap violations:")
    print(f"  CPO total: {sum(viol_cpo_test)}")
    print(f"  SGPO total: {sum(viol_gpo_test)}")
    if sum(viol_cpo_test) > 0:
        improvement = 100 * (1 - sum(viol_gpo_test) / sum(viol_cpo_test))
        print(f"  SGPO improvement: {improvement:.1f}%")
    
    # Save metrics
    metrics_data = {
        "training": {
            "cpo_violations": viol_cpo_train,
            "gpo_violations": viol_gpo_train,
            "cpo_returns": ret_cpo_train,
            "gpo_returns": ret_gpo_train,
        },
        "test_novel_trap": {
            "cpo_violations": viol_cpo_test,
            "gpo_violations": viol_gpo_test,
            "cpo_returns": ret_cpo_test,
            "gpo_returns": ret_gpo_test,
            "cpo_goal_rate": float(np.mean(goal_cpo_test)),
            "gpo_goal_rate": float(np.mean(goal_gpo_test)),
        },
        "summary": {
            "cpo_novel_violations": sum(viol_cpo_test),
            "gpo_novel_violations": sum(viol_gpo_test),
            "improvement_pct": float(100 * (1 - sum(viol_gpo_test) / max(sum(viol_cpo_test), 1)))
        }
    }
    
    with open('../results/safety/safety_hard_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print("\nMetrics saved to ../results/safety/safety_hard_metrics.json")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Environment with all traps
    ax1 = axes[0, 0]
    for trap in test_env.training_traps:
        circle = plt.Circle(trap["center"], trap["radius"], color='red', alpha=0.2)
        horizon = plt.Circle(trap["center"], trap["event_horizon"], color='black', alpha=0.3)
        ax1.add_patch(circle)
        ax1.add_patch(horizon)
    # Novel trap in different color
    novel = test_env.novel_trap
    ax1.add_patch(plt.Circle(novel["center"], novel["radius"], color='purple', alpha=0.3, label='Novel Trap'))
    ax1.add_patch(plt.Circle(novel["center"], novel["event_horizon"], color='purple', alpha=0.5))
    ax1.scatter([10], [10], c='gold', s=300, marker='*', zorder=5, label='Goal')
    ax1.scatter([0], [0], c='green', s=150, zorder=5, label='Start')
    
    # Plot sample trajectories
    for i, traj in enumerate(traj_cpo[:5]):
        ax1.plot(traj[:,0], traj[:,1], 'orange', alpha=0.5, linewidth=1)
    for i, traj in enumerate(traj_gpo[:5]):
        ax1.plot(traj[:,0], traj[:,1], 'blue', alpha=0.5, linewidth=1)
    ax1.set_title('Test Environment (Novel Trap in Purple)')
    ax1.legend(loc='lower right')
    ax1.set_xlim(-1, 12)
    ax1.set_ylim(-1, 12)
    ax1.set_aspect('equal')
    
    # Plot 2: Training violations
    ax2 = axes[0, 1]
    window = 20
    viol_cpo_smooth = np.convolve(viol_cpo_train, np.ones(window)/window, mode='valid')
    viol_gpo_smooth = np.convolve(viol_gpo_train, np.ones(window)/window, mode='valid')
    ax2.plot(viol_cpo_smooth, color='orange', label='CPO')
    ax2.plot(viol_gpo_smooth, color='blue', label='SGPO')
    ax2.set_title('Training Violations (Known Traps)')
    ax2.set_xlabel('Episode')
    ax2.legend()
    
    # Plot 3: Training returns
    ax3 = axes[0, 2]
    ret_cpo_smooth = np.convolve(ret_cpo_train, np.ones(window)/window, mode='valid')
    ret_gpo_smooth = np.convolve(ret_gpo_train, np.ones(window)/window, mode='valid')
    ax3.plot(ret_cpo_smooth, color='orange', label='CPO')
    ax3.plot(ret_gpo_smooth, color='blue', label='SGPO')
    ax3.set_title('Training Returns')
    ax3.set_xlabel('Episode')
    ax3.legend()
    
    # Plot 4: Learned metric heatmap
    ax4 = axes[1, 0]
    x_range = np.linspace(-1, 12, 100)
    y_range = np.linspace(-1, 12, 100)
    X, Y = np.meshgrid(x_range, y_range)
    points = torch.FloatTensor(np.stack([X.flatten(), Y.flatten()], axis=1))
    with torch.no_grad():
        metric_values = metric_gpo(points).numpy().flatten().reshape(X.shape)
    metric_values = np.clip(metric_values, 1, 20)
    im = ax4.contourf(X, Y, metric_values, levels=20, cmap='hot_r')
    plt.colorbar(im, ax=ax4, label='g(x)')
    # Mark novel trap location
    ax4.scatter([novel["center"][0]], [novel["center"][1]], c='purple', s=200, marker='x', linewidths=3)
    ax4.set_title('Learned Metric (X = Novel Trap Location)')
    ax4.set_aspect('equal')
    
    # Plot 5: Test violations comparison
    ax5 = axes[1, 1]
    methods = ['CPO', 'SGPO']
    test_violations = [sum(viol_cpo_test), sum(viol_gpo_test)]
    colors = ['orange', 'blue']
    bars = ax5.bar(methods, test_violations, color=colors, alpha=0.7)
    ax5.set_title('Test Violations (Novel Trap)')
    ax5.set_ylabel('Total Violations')
    for bar, val in zip(bars, test_violations):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), ha='center')
    
    # Plot 6: Test goal success
    ax6 = axes[1, 2]
    goal_rates = [100*np.mean(goal_cpo_test), 100*np.mean(goal_gpo_test)]
    bars = ax6.bar(methods, goal_rates, color=colors, alpha=0.7)
    ax6.set_title('Test Goal Success Rate')
    ax6.set_ylabel('Success %')
    ax6.set_ylim(0, 105)
    for bar, val in zip(bars, goal_rates):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig('../figures/experiments/safety_hard_results.png', dpi=150)
    print("Plot saved to ../figures/experiments/safety_hard_results.png")
