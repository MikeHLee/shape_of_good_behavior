"""
Geodesic Distance Penalty Experiment

KEY FIX: Instead of just penalizing being near black holes, we penalize
GEODESIC STEP LENGTH: √g(x) * ||step||

This makes the agent learn that detouring around black holes is CHEAPER
than going straight through, because geodesic distance is shorter.
"""

import modal

app = modal.App("geodesic-distance-gpo")
volume = modal.Volume.from_name("geodpo-data", create_if_missing=True)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy>=1.24.0,<2.0.0",
        "torch>=2.1.0",
        "matplotlib>=3.7.0",
        "seaborn",
    )
)


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
def geodesic_distance_experiment(embed_dim: int = 768, episodes: int = 1000):
    """
    TRUE Sheaf-Geodesic Policy Optimization using geodesic distance penalty.
    
    Key innovation: reward -= √g(x) * ||step||
    
    This makes geodesically long paths (through black holes) expensive,
    forcing the agent to learn detours that are geodesically shorter.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.distributions import Normal
    import json
    import shutil
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Geodesic Distance SGPO on {DEVICE}")
    print(f"Config: d={embed_dim}, episodes={episodes}")
    
    class GeodesicStyleEnv:
        """
        Environment with GEODESIC DISTANCE penalty.
        
        Key change: Step cost = √g(x) * ||action||
        Agent must learn that detouring is geodesically shorter.
        """
        def __init__(self, embed_dim=768):
            self.embed_dim = embed_dim
            self.max_steps = 100
            self.step_count = 0
            
            # Create 3 orthogonal archetypes
            H = np.random.randn(embed_dim, embed_dim)
            Q, _ = np.linalg.qr(H)
            self.archetypes = {
                'Concise': Q[:, 0] * 10.0,
                'Empathy': Q[:, 1] * 10.0,
                'Detail': Q[:, 2] * 10.0
            }
            self.state = np.zeros(embed_dim)
            
            # Black holes at MIDPOINTS (original setup - but now agent should detour!)
            self.black_holes = []
            arch_list = list(self.archetypes.values())
            for i in range(3):
                midpoint = (arch_list[i] + arch_list[(i+1)%3]) / 2.0
                self.black_holes.append({
                    'center': midpoint,
                    'radius': 2.0,
                    'strength': 100.0  # Strong singularity
                })
        
        def compute_metric(self, state):
            """
            Compute Riemannian metric g(x) at state.
            g(x) = 1 + Σᵢ strength_i / dist(x, center_i)^α
            """
            g = 1.0  # Base metric
            
            for bh in self.black_holes:
                dist = np.linalg.norm(state - bh['center'])
                # Soft singularity: saturates but still gets very large
                safe_dist = max(dist - bh['radius'], 0.1)
                contribution = bh['strength'] / (safe_dist ** 1.5)
                contribution = min(contribution, 10000.0)  # Cap at 10000
                g += contribution
            
            return g
        
        def reset(self):
            start_arch = list(self.archetypes.keys())[np.random.randint(0, 3)]
            self.state = self.archetypes[start_arch] + np.random.randn(self.embed_dim) * 0.1
            self.step_count = 0
            return self.state.copy()
        
        def get_preference_vector(self, state=None):
            if state is None: state = self.state
            distances = {n: np.linalg.norm(state - p) for n, p in self.archetypes.items()}
            archetype = min(distances, key=distances.get)
            transitions = {'Concise': 'Empathy', 'Empathy': 'Detail', 'Detail': 'Concise'}
            target = self.archetypes[transitions[archetype]]
            direction = target - state
            norm = np.linalg.norm(direction)
            return direction / norm if norm > 0 else direction
        
        def step(self, action):
            move = np.clip(action, -1.0, 1.0)  # Larger action space for detours
            step_size = np.linalg.norm(move)
            
            # Preference reward (for following cycle)
            pref_dir = self.get_preference_vector()
            base_reward = float(np.dot(move, pref_dir)) * 5.0
            
            # KEY: Geodesic distance penalty = √g(x) * ||step||
            # This makes paths through high-metric regions expensive!
            new_state = self.state + move
            g = self.compute_metric(new_state)
            geodesic_cost = np.sqrt(g) * step_size
            
            # Total reward: preference alignment - geodesic cost
            reward = base_reward - geodesic_cost
            
            self.state = new_state
            self.step_count += 1
            done = self.step_count >= self.max_steps
            
            return self.state.copy(), reward, done, {
                'g': g, 
                'geodesic_cost': geodesic_cost,
                'step_size': step_size
            }
        
        def compute_h1_ground_truth(self):
            v1, v2, v3 = self.archetypes['Concise'], self.archetypes['Empathy'], self.archetypes['Detail']
            return sum(10.0 * np.linalg.norm(e - s) for s, e in [(v1, v2), (v2, v3), (v3, v1)])
    
    class EuclideanStyleEnv:
        """
        Control: Same environment but with EUCLIDEAN distance penalty.
        No metric - just penalizes step size uniformly.
        """
        def __init__(self, embed_dim=768):
            self.embed_dim = embed_dim
            self.max_steps = 100
            self.step_count = 0
            
            H = np.random.randn(embed_dim, embed_dim)
            Q, _ = np.linalg.qr(H)
            self.archetypes = {
                'Concise': Q[:, 0] * 10.0,
                'Empathy': Q[:, 1] * 10.0,
                'Detail': Q[:, 2] * 10.0
            }
            self.state = np.zeros(embed_dim)
            
            self.black_holes = []
            arch_list = list(self.archetypes.values())
            for i in range(3):
                midpoint = (arch_list[i] + arch_list[(i+1)%3]) / 2.0
                self.black_holes.append({
                    'center': midpoint,
                    'radius': 2.0,
                    'strength': 100.0
                })
        
        def reset(self):
            start_arch = list(self.archetypes.keys())[np.random.randint(0, 3)]
            self.state = self.archetypes[start_arch] + np.random.randn(self.embed_dim) * 0.1
            self.step_count = 0
            return self.state.copy()
        
        def get_preference_vector(self, state=None):
            if state is None: state = self.state
            distances = {n: np.linalg.norm(state - p) for n, p in self.archetypes.items()}
            archetype = min(distances, key=distances.get)
            transitions = {'Concise': 'Empathy', 'Empathy': 'Detail', 'Detail': 'Concise'}
            target = self.archetypes[transitions[archetype]]
            direction = target - state
            norm = np.linalg.norm(direction)
            return direction / norm if norm > 0 else direction
        
        def compute_black_hole_cost(self, state):
            """Traditional: penalize being near black holes."""
            total = 0.0
            for bh in self.black_holes:
                dist = np.linalg.norm(state - bh['center'])
                if dist < bh['radius']:
                    total += bh['strength'] * 10.0
                elif dist < bh['radius'] * 2:
                    proximity = 1.0 - (dist - bh['radius']) / bh['radius']
                    total += bh['strength'] * (proximity ** 2)
            return total
        
        def step(self, action):
            move = np.clip(action, -1.0, 1.0)
            step_size = np.linalg.norm(move)
            
            pref_dir = self.get_preference_vector()
            base_reward = float(np.dot(move, pref_dir)) * 5.0
            
            new_state = self.state + move
            
            # Euclidean: uniform step cost + black hole penalty
            euclidean_cost = step_size  # Just step size, no metric
            bh_cost = self.compute_black_hole_cost(new_state)
            
            reward = base_reward - euclidean_cost - bh_cost
            
            self.state = new_state
            self.step_count += 1
            done = self.step_count >= self.max_steps
            
            return self.state.copy(), reward, done, {
                'euclidean_cost': euclidean_cost,
                'bh_cost': bh_cost,
                'step_size': step_size
            }
    
    # Networks
    class Actor(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 256), nn.LayerNorm(256), nn.Tanh(),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.Tanh(),
                nn.Linear(128, d)
            )
            self.log_std = nn.Parameter(torch.ones(1) * -0.5)  # Higher variance for exploration
        def forward(self, x): 
            return Normal(self.net(x), torch.exp(self.log_std).expand_as(self.net(x)))
    
    class Critic(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 256), nn.LayerNorm(256), nn.Tanh(),
                nn.Linear(256, 128), nn.Tanh(),
                nn.Linear(128, 1)
            )
        def forward(self, x): return self.net(x)
    
    def train_agent(env, episodes, name="Agent"):
        actor = Actor(env.embed_dim).to(DEVICE)
        critic = Critic(env.embed_dim).to(DEVICE)
        opt_a = optim.Adam(actor.parameters(), lr=3e-4)
        opt_c = optim.Adam(critic.parameters(), lr=1e-3)
        
        history = {'returns': [], 'metrics': [], 'step_sizes': []}
        
        for ep in range(episodes):
            obs, done, traj, ep_ret = env.reset(), False, [], 0
            ep_g, ep_steps = [], []
            
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    dist = actor(obs_t)
                    action = dist.sample()
                    old_lp = dist.log_prob(action).sum()
                
                next_obs, r, done, info = env.step(action.squeeze(0).cpu().numpy())
                traj.append((obs, action, r, old_lp.item()))
                obs, ep_ret = next_obs, ep_ret + r
                
                ep_g.append(info.get('g', 1.0))
                ep_steps.append(info.get('step_size', 0.0))
            
            history['returns'].append(ep_ret)
            history['metrics'].append(np.mean(ep_g))
            history['step_sizes'].append(np.mean(ep_steps))
            
            # PPO update
            states = torch.FloatTensor(np.array([t[0] for t in traj])).to(DEVICE)
            actions = torch.cat([t[1] for t in traj]).to(DEVICE)
            rewards = torch.FloatTensor([t[2] for t in traj]).to(DEVICE)
            old_lps = torch.FloatTensor([t[3] for t in traj]).to(DEVICE)
            
            vals = critic(states).squeeze()
            opt_c.zero_grad()
            nn.MSELoss()(vals, rewards).backward()
            opt_c.step()
            
            adv = rewards - vals.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            new_lps = actor(states).log_prob(actions).sum(dim=1)
            ratio = torch.exp(new_lps - old_lps)
            loss = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
            
            # Entropy bonus for exploration
            entropy = actor(states).entropy().sum(dim=1).mean()
            loss = loss - 0.02 * entropy
            
            opt_a.zero_grad()
            loss.backward()
            opt_a.step()
            
            if ep % 100 == 0:
                print(f"{name} Ep {ep}: Ret={ep_ret:.1f}, g={np.mean(ep_g):.2f}, step={np.mean(ep_steps):.3f}")
        
        return history
    
    # Run experiments
    geo_env = GeodesicStyleEnv(embed_dim)
    euc_env = EuclideanStyleEnv(embed_dim)
    h1_truth = geo_env.compute_h1_ground_truth()
    
    print(f"\nH1 Truth: {h1_truth:.2f}, Black Holes: {len(geo_env.black_holes)}")
    print("\nGEODESIC DISTANCE EXPERIMENT")
    print("Geodesic SGPO: cost = √g(x) * ||step|| (should learn to detour)")
    print("Euclidean PPO: cost = ||step|| + bh_penalty (takes straight paths)")
    print()
    
    print("="*60 + "\nTraining Euclidean PPO (control)...\n" + "="*60)
    euc_hist = train_agent(euc_env, episodes, "EuclideanPPO")
    
    print("\n" + "="*60 + "\nTraining Geodesic SGPO...\n" + "="*60)
    geo_hist = train_agent(geo_env, episodes, "GeodesicSGPO")
    
    # Analysis
    euc_mean, euc_final = np.mean(euc_hist['returns']), np.mean(euc_hist['returns'][-100:])
    geo_mean, geo_final = np.mean(geo_hist['returns']), np.mean(geo_hist['returns'][-100:])
    
    # Key metric: Does geodesic SGPO learn to avoid high-g regions?
    euc_g_final = np.mean(euc_hist['metrics'][-100:])
    geo_g_final = np.mean(geo_hist['metrics'][-100:])
    
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"Euclidean PPO: Return={euc_final:.1f}, Avg g={euc_g_final:.2f}")
    print(f"Geodesic SGPO:  Return={geo_final:.1f}, Avg g={geo_g_final:.2f}")
    print(f"\nKey insight: If Geodesic SGPO has LOWER avg g, it learned to detour!")
    print(f"g improvement: {euc_g_final - geo_g_final:.2f} (positive = learned to avoid black holes)")
    
    results = {
        'h1_truth': float(h1_truth),
        'euc_returns': euc_hist['returns'],
        'euc_metrics': euc_hist['metrics'],
        'geo_returns': geo_hist['returns'],
        'geo_metrics': geo_hist['metrics'],
        'config': {'embed_dim': embed_dim, 'episodes': episodes},
        'summary': {
            'euc_mean': euc_mean, 'euc_final': euc_final, 'euc_g_final': euc_g_final,
            'geo_mean': geo_mean, 'geo_final': geo_final, 'geo_g_final': geo_g_final,
            'g_improvement': euc_g_final - geo_g_final,
        }
    }
    
    with open("results_geodesic.json", "w") as f: json.dump(results, f, indent=2)
    shutil.copy("results_geodesic.json", f"{VOLUME_PATH}/geodesic_distance_results.json")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(euc_hist['returns'], label='Euclidean PPO', alpha=0.7)
    axes[0, 0].plot(geo_hist['returns'], label='Geodesic SGPO', alpha=0.7)
    axes[0, 0].set_title("Returns"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(euc_hist['metrics'], label='Euclidean PPO', alpha=0.7)
    axes[0, 1].plot(geo_hist['metrics'], label='Geodesic SGPO', alpha=0.7)
    axes[0, 1].set_title("Average Metric g(x) (Lower = Avoiding Black Holes)")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    
    # Moving averages for clearer trends
    window = 50
    euc_ma = np.convolve(euc_hist['metrics'], np.ones(window)/window, mode='valid')
    geo_ma = np.convolve(geo_hist['metrics'], np.ones(window)/window, mode='valid')
    axes[1, 0].plot(euc_ma, label='Euclidean PPO', alpha=0.7)
    axes[1, 0].plot(geo_ma, label='Geodesic SGPO', alpha=0.7)
    axes[1, 0].set_title(f"Metric g(x) - {window}-Episode Moving Average")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(euc_hist['step_sizes'], label='Euclidean PPO', alpha=0.7)
    axes[1, 1].plot(geo_hist['step_sizes'], label='Geodesic SGPO', alpha=0.7)
    axes[1, 1].set_title("Average Step Size"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("geodesic_plot.png", dpi=150)
    shutil.copy("geodesic_plot.png", f"{VOLUME_PATH}/geodesic_distance_results.png")
    volume.commit()
    
    print(f"\nSaved to {VOLUME_PATH}/geodesic_distance_results.json")
    return results['summary']
