"""
Implicit Boundary Experiment for SGPO

Tests learned implicit danger boundaries vs spherical black holes.

Key insight: In high-D embedding space, point singularities are trivially avoided.
Real dangerous regions have complex, non-convex boundaries that must be LEARNED.

This experiment:
1. Creates synthetic dangerous regions (non-spherical)
2. Trains LearnedDangerBoundary to find the boundary
3. Compares agent behavior with learned vs spherical boundaries
4. Measures: boundary accuracy, geodesic cost, safety violations
"""

import modal

app = modal.App("implicit-boundary-gpo")
volume = modal.Volume.from_name("geodpo-data", create_if_missing=True)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy>=1.24.0,<2.0.0",
        "torch>=2.1.0",
        "matplotlib>=3.7.0",
        "seaborn",
        "scikit-learn",
    )
)


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
def implicit_boundary_experiment(
    embed_dim: int = 128,
    n_danger_samples: int = 5000,
    n_episodes: int = 500,
    seed: int = 42,
):
    """
    Compare learned implicit boundaries vs spherical black holes.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.distributions import Normal
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import json
    import os
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Implicit Boundary SGPO on {DEVICE}")
    print(f"Config: d={embed_dim}, samples={n_danger_samples}, episodes={n_episodes}")
    
    # =========================================================================
    # LEARNED DANGER BOUNDARY (copied inline for Modal)
    # =========================================================================
    
    class LearnedDangerBoundary(nn.Module):
        """Implicit surface for danger region."""
        
        def __init__(self, embed_dim: int, hidden_dim: int = 256, n_layers: int = 3):
            super().__init__()
            layers = []
            in_dim = embed_dim
            for i in range(n_layers - 1):
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ])
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, 1))
            self.net = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.net(x)
        
        def is_dangerous(self, x, margin=0.0):
            d = self.forward(x).squeeze(-1)
            return d > margin  # Positive logits = dangerous
        
        def metric(self, x, strength=100.0, alpha=1.5):
            d = self.forward(x)
            signed_dist = -d  # Convert: positive logits → negative distance
            safe_d = torch.clamp(signed_dist.abs(), min=1e-3)
            return 1.0 + strength / (safe_d ** alpha)
        
        def train_boundary(self, embeddings, labels, n_epochs=100, lr=1e-3):
            optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
            n_pos = labels.sum().item()
            n_neg = len(labels) - n_pos
            pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(embeddings.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                logits = self.forward(embeddings)
                loss = criterion(logits, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 20 == 0:
                    preds = (logits > 0).float().squeeze()
                    acc = (preds == labels).float().mean().item()
                    print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.4f}")
    
    # =========================================================================
    # CREATE NON-SPHERICAL DANGER REGIONS
    # =========================================================================
    
    def create_amorphous_danger_regions(embed_dim, n_samples):
        """
        Create danger regions that are NOT spherical.
        Uses multiple overlapping ellipsoids with random orientations.
        """
        n_regions = 4
        
        # Random region centers
        centers = np.random.randn(n_regions, embed_dim) * 3
        
        # Random orientations (rotation matrices)
        rotations = []
        for _ in range(n_regions):
            H = np.random.randn(embed_dim, embed_dim)
            Q, _ = np.linalg.qr(H)
            rotations.append(Q)
        
        # Random axis scales (elongated ellipsoids)
        scales = []
        for _ in range(n_regions):
            s = np.random.uniform(0.5, 3.0, embed_dim)
            s[:embed_dim//4] *= 3  # Make some axes much longer
            scales.append(s)
        
        # Sample points
        embeddings = np.random.randn(n_samples, embed_dim) * 5
        
        # Check which points are inside any ellipsoid
        labels = np.zeros(n_samples)
        for c, R, s in zip(centers, rotations, scales):
            # Transform to ellipsoid coordinates
            diff = embeddings - c
            rotated = diff @ R
            scaled = rotated / s
            dist_sq = (scaled ** 2).sum(axis=1)
            inside = dist_sq < 1.0
            labels = np.logical_or(labels, inside)
        
        return embeddings.astype(np.float32), labels.astype(np.float32), centers, scales
    
    # =========================================================================
    # ENVIRONMENT WITH LEARNED BOUNDARY
    # =========================================================================
    
    class ImplicitBoundaryEnv:
        """Environment using learned implicit danger boundary."""
        
        def __init__(self, boundary_model, embed_dim, goal):
            self.boundary = boundary_model
            self.embed_dim = embed_dim
            self.goal = goal
            self.max_steps = 100
            self.state = None
            self.step_count = 0
        
        def reset(self, start=None):
            if start is not None:
                self.state = start.copy()
            else:
                self.state = np.random.randn(self.embed_dim).astype(np.float32) * 2
            self.step_count = 0
            return self.state.copy()
        
        def step(self, action):
            move = np.clip(action, -0.5, 0.5)
            new_state = self.state + move
            step_size = np.linalg.norm(move)
            
            # Goal reward
            old_dist = np.linalg.norm(self.state - self.goal)
            new_dist = np.linalg.norm(new_state - self.goal)
            goal_reward = (old_dist - new_dist) * 10.0
            
            # Geodesic cost from learned boundary
            with torch.no_grad():
                state_t = torch.tensor(new_state).unsqueeze(0).float().to(DEVICE)
                g = self.boundary.metric(state_t).item()
            
            geodesic_cost = np.sqrt(g) * step_size
            
            # Safety violation check
            with torch.no_grad():
                is_danger = self.boundary.is_dangerous(state_t).item()
            
            reward = goal_reward - geodesic_cost
            if is_danger:
                reward -= 50.0  # Heavy penalty for violation
            
            self.state = new_state.astype(np.float32)
            self.step_count += 1
            done = self.step_count >= self.max_steps or new_dist < 0.5
            
            return self.state.copy(), reward, done, {
                'g': g,
                'geodesic_cost': geodesic_cost,
                'is_dangerous': is_danger,
                'goal_dist': new_dist,
            }
    
    # =========================================================================
    # SPHERICAL BASELINE ENVIRONMENT
    # =========================================================================
    
    class SphericalBlackHoleEnv:
        """Baseline with spherical black holes at region centers."""
        
        def __init__(self, centers, radius, embed_dim, goal, strength=100.0):
            self.centers = centers
            self.radius = radius
            self.strength = strength
            self.embed_dim = embed_dim
            self.goal = goal
            self.max_steps = 100
            self.state = None
            self.step_count = 0
        
        def compute_metric(self, state):
            g = 1.0
            for c in self.centers:
                dist = np.linalg.norm(state - c)
                safe_dist = max(dist - self.radius, 0.1)
                g += self.strength / (safe_dist ** 1.5)
            return min(g, 10000.0)
        
        def is_inside_blackhole(self, state):
            for c in self.centers:
                if np.linalg.norm(state - c) < self.radius:
                    return True
            return False
        
        def reset(self, start=None):
            if start is not None:
                self.state = start.copy()
            else:
                self.state = np.random.randn(self.embed_dim).astype(np.float32) * 2
            self.step_count = 0
            return self.state.copy()
        
        def step(self, action):
            move = np.clip(action, -0.5, 0.5)
            new_state = self.state + move
            step_size = np.linalg.norm(move)
            
            old_dist = np.linalg.norm(self.state - self.goal)
            new_dist = np.linalg.norm(new_state - self.goal)
            goal_reward = (old_dist - new_dist) * 10.0
            
            g = self.compute_metric(new_state)
            geodesic_cost = np.sqrt(g) * step_size
            
            is_danger = self.is_inside_blackhole(new_state)
            
            reward = goal_reward - geodesic_cost
            if is_danger:
                reward -= 50.0
            
            self.state = new_state.astype(np.float32)
            self.step_count += 1
            done = self.step_count >= self.max_steps or new_dist < 0.5
            
            return self.state.copy(), reward, done, {
                'g': g,
                'geodesic_cost': geodesic_cost,
                'is_dangerous': is_danger,
                'goal_dist': new_dist,
            }
    
    # =========================================================================
    # POLICY NETWORK
    # =========================================================================
    
    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim, action_dim, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
            self.mean = nn.Linear(hidden, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        def forward(self, x):
            h = self.net(x)
            mean = self.mean(h)
            std = torch.exp(self.log_std.clamp(-5, 2))
            return Normal(mean, std)
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    def train_policy(env, policy, n_episodes, name):
        optimizer = optim.Adam(policy.parameters(), lr=3e-4)
        
        episode_rewards = []
        episode_violations = []
        episode_geodesic_costs = []
        
        for ep in range(n_episodes):
            state = env.reset()
            log_probs = []
            rewards = []
            violations = 0
            total_geo_cost = 0
            
            done = False
            while not done:
                state_t = torch.tensor(state).float().unsqueeze(0).to(DEVICE)
                dist = policy(state_t)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                
                next_state, reward, done, info = env.step(action.cpu().numpy().flatten())
                
                log_probs.append(log_prob)
                rewards.append(reward)
                violations += int(info['is_dangerous'])
                total_geo_cost += info['geodesic_cost']
                
                state = next_state
            
            # REINFORCE update
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            returns = torch.tensor(returns).float().to(DEVICE)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            policy_loss = 0
            for lp, R in zip(log_probs, returns):
                policy_loss -= lp * R
            
            optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            episode_rewards.append(sum(rewards))
            episode_violations.append(violations)
            episode_geodesic_costs.append(total_geo_cost)
            
            if (ep + 1) % 50 == 0:
                avg_r = np.mean(episode_rewards[-50:])
                avg_v = np.mean(episode_violations[-50:])
                avg_g = np.mean(episode_geodesic_costs[-50:])
                print(f"  [{name}] Ep {ep+1}: reward={avg_r:.2f}, violations={avg_v:.2f}, geo_cost={avg_g:.2f}")
        
        return {
            'rewards': episode_rewards,
            'violations': episode_violations,
            'geodesic_costs': episode_geodesic_costs,
        }
    
    # =========================================================================
    # RUN EXPERIMENT
    # =========================================================================
    
    print("\n" + "="*60)
    print("PHASE 1: Creating non-spherical danger regions")
    print("="*60)
    
    embeddings, labels, centers, scales = create_amorphous_danger_regions(
        embed_dim, n_danger_samples
    )
    n_dangerous = labels.sum()
    print(f"Created {len(embeddings)} samples, {n_dangerous:.0f} dangerous ({100*n_dangerous/len(embeddings):.1f}%)")
    
    print("\n" + "="*60)
    print("PHASE 2: Training learned danger boundary")
    print("="*60)
    
    boundary = LearnedDangerBoundary(embed_dim, hidden_dim=256).to(DEVICE)
    emb_t = torch.tensor(embeddings).float().to(DEVICE)
    lab_t = torch.tensor(labels).float().to(DEVICE)
    boundary.train_boundary(emb_t, lab_t, n_epochs=100)
    
    # Evaluate boundary
    with torch.no_grad():
        preds = (boundary.forward(emb_t) > 0).float().squeeze()
        acc = (preds == lab_t).float().mean().item()
        prec = precision_score(labels, preds.cpu().numpy())
        rec = recall_score(labels, preds.cpu().numpy())
    print(f"\nBoundary accuracy: {acc:.4f}, precision: {prec:.4f}, recall: {rec:.4f}")
    
    print("\n" + "="*60)
    print("PHASE 3: Training policies")
    print("="*60)
    
    # Goal is outside all danger regions
    goal = np.random.randn(embed_dim).astype(np.float32) * 5
    
    # Env with learned boundary
    env_learned = ImplicitBoundaryEnv(boundary, embed_dim, goal)
    policy_learned = PolicyNetwork(embed_dim, embed_dim).to(DEVICE)
    
    print("\nTraining with LEARNED boundary:")
    results_learned = train_policy(env_learned, policy_learned, n_episodes, "Learned")
    
    # Env with spherical black holes
    avg_radius = np.mean([np.mean(s) for s in scales])
    env_spherical = SphericalBlackHoleEnv(centers, avg_radius, embed_dim, goal)
    policy_spherical = PolicyNetwork(embed_dim, embed_dim).to(DEVICE)
    
    print("\nTraining with SPHERICAL black holes:")
    results_spherical = train_policy(env_spherical, policy_spherical, n_episodes, "Spherical")
    
    print("\n" + "="*60)
    print("PHASE 4: Evaluation")
    print("="*60)
    
    def evaluate_policy(env, policy, n_eval=50):
        total_rewards = []
        total_violations = []
        reached_goal = 0
        
        for _ in range(n_eval):
            state = env.reset()
            ep_reward = 0
            ep_violations = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    state_t = torch.tensor(state).float().unsqueeze(0).to(DEVICE)
                    dist = policy(state_t)
                    action = dist.mean  # Deterministic for eval
                
                state, reward, done, info = env.step(action.cpu().numpy().flatten())
                ep_reward += reward
                ep_violations += int(info['is_dangerous'])
                
                if info['goal_dist'] < 0.5:
                    reached_goal += 1
            
            total_rewards.append(ep_reward)
            total_violations.append(ep_violations)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'mean_violations': np.mean(total_violations),
            'goal_rate': reached_goal / n_eval,
        }
    
    eval_learned = evaluate_policy(env_learned, policy_learned)
    eval_spherical = evaluate_policy(env_spherical, policy_spherical)
    
    print(f"\nLEARNED BOUNDARY:")
    print(f"  Mean reward: {eval_learned['mean_reward']:.2f}")
    print(f"  Mean violations: {eval_learned['mean_violations']:.2f}")
    print(f"  Goal rate: {eval_learned['goal_rate']:.2%}")
    
    print(f"\nSPHERICAL BLACK HOLES:")
    print(f"  Mean reward: {eval_spherical['mean_reward']:.2f}")
    print(f"  Mean violations: {eval_spherical['mean_violations']:.2f}")
    print(f"  Goal rate: {eval_spherical['goal_rate']:.2%}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    results = {
        'config': {
            'embed_dim': embed_dim,
            'n_danger_samples': n_danger_samples,
            'n_episodes': n_episodes,
            'seed': seed,
        },
        'boundary_metrics': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
        },
        'training': {
            'learned': {
                'rewards': results_learned['rewards'],
                'violations': results_learned['violations'],
                'geodesic_costs': results_learned['geodesic_costs'],
            },
            'spherical': {
                'rewards': results_spherical['rewards'],
                'violations': results_spherical['violations'],
                'geodesic_costs': results_spherical['geodesic_costs'],
            },
        },
        'evaluation': {
            'learned': eval_learned,
            'spherical': eval_spherical,
        },
    }
    
    os.makedirs(f"{VOLUME_PATH}/implicit_boundary", exist_ok=True)
    with open(f"{VOLUME_PATH}/implicit_boundary/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    window = 20
    
    def smooth(x):
        return np.convolve(x, np.ones(window)/window, mode='valid')
    
    ax = axes[0]
    ax.plot(smooth(results_learned['rewards']), label='Learned', color='blue')
    ax.plot(smooth(results_spherical['rewards']), label='Spherical', color='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Rewards')
    ax.legend()
    
    ax = axes[1]
    ax.plot(smooth(results_learned['violations']), label='Learned', color='blue')
    ax.plot(smooth(results_spherical['violations']), label='Spherical', color='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Violations')
    ax.set_title('Safety Violations')
    ax.legend()
    
    ax = axes[2]
    ax.plot(smooth(results_learned['geodesic_costs']), label='Learned', color='blue')
    ax.plot(smooth(results_spherical['geodesic_costs']), label='Spherical', color='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Geodesic Cost')
    ax.set_title('Geodesic Cost')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{VOLUME_PATH}/implicit_boundary/training_curves.png", dpi=150)
    
    volume.commit()
    print(f"\nResults saved to {VOLUME_PATH}/implicit_boundary/")
    
    return results


@app.local_entrypoint()
def main():
    result = implicit_boundary_experiment.remote(
        embed_dim=128,
        n_danger_samples=5000,
        n_episodes=500,
        seed=42,
    )
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Boundary accuracy: {result['boundary_metrics']['accuracy']:.4f}")
    print(f"Learned - violations: {result['evaluation']['learned']['mean_violations']:.2f}")
    print(f"Spherical - violations: {result['evaluation']['spherical']['mean_violations']:.2f}")
