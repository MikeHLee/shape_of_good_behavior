"""
Hyperplane Barrier Experiment

KEY FIX: In high dimensions, point singularities are too easy to miss.
Instead, we create HYPERPLANE BARRIERS between archetypes.

A hyperplane barrier makes ANY path between two archetypes pass through
the danger zone, forcing the agent to learn detours through the third dimension.
"""

import modal

app = modal.App("hyperplane-barrier-gpo")
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
def hyperplane_barrier_experiment(embed_dim: int = 768, episodes: int = 1000):
    """
    Geodesic navigation with HYPERPLANE barriers.
    
    Key insight: Point singularities in high-D are easy to miss.
    Hyperplane barriers create walls that MUST be navigated around.
    
    Geometry:
    - 3 archetypes at orthogonal directions
    - Barrier planes perpendicular to each archetype-to-archetype path
    - Agent must detour through "third dimension" to avoid barriers
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
    print(f"Running Hyperplane Barrier SGPO on {DEVICE}")
    print(f"Config: d={embed_dim}, episodes={episodes}")
    
    class HyperplaneBarrierEnv:
        """
        Environment with HYPERPLANE BARRIERS between archetypes.
        
        Barrier metric: g(x) increases based on distance to barrier PLANE,
        not distance to a point. This creates walls the agent can't ignore.
        """
        def __init__(self, embed_dim=768, barrier_strength=500.0, barrier_thickness=3.0):
            self.embed_dim = embed_dim
            self.max_steps = 100
            self.step_count = 0
            self.barrier_strength = barrier_strength
            self.barrier_thickness = barrier_thickness
            
            # Create 3 orthogonal archetypes
            H = np.random.randn(embed_dim, embed_dim)
            Q, _ = np.linalg.qr(H)
            self.Q = Q  # Store for barrier calculations
            
            self.archetypes = {
                'Concise': Q[:, 0] * 10.0,
                'Empathy': Q[:, 1] * 10.0,
                'Detail': Q[:, 2] * 10.0
            }
            self.state = np.zeros(embed_dim)
            
            # Barriers: hyperplanes perpendicular to archetype-to-archetype directions
            # Barrier i blocks the direct path from archetype i to archetype (i+1)%3
            arch_list = list(self.archetypes.values())
            self.barriers = []
            for i in range(3):
                a1, a2 = arch_list[i], arch_list[(i+1)%3]
                midpoint = (a1 + a2) / 2.0
                normal = (a2 - a1) / np.linalg.norm(a2 - a1)  # Direction from a1 to a2
                
                self.barriers.append({
                    'point': midpoint,      # Point on the hyperplane
                    'normal': normal,       # Normal vector (perpendicular to plane)
                    'extent': 8.0,          # Barrier extends this far from midpoint
                })
            
            print(f"Created {len(self.barriers)} hyperplane barriers")
        
        def compute_barrier_metric(self, state):
            """
            Compute metric based on distance to barrier HYPERPLANES.
            
            For each barrier:
            - Project state onto plane normal to get signed distance
            - If |signed_dist| < thickness AND within extent, add to metric
            
            This creates "walls" that span across dimensions.
            """
            g = 1.0  # Base metric
            
            for barrier in self.barriers:
                point = barrier['point']
                normal = barrier['normal']
                extent = barrier['extent']
                
                # Signed distance to hyperplane
                to_point = state - point
                signed_dist = np.dot(to_point, normal)
                
                # Distance along hyperplane (to check if within extent)
                parallel_component = to_point - signed_dist * normal
                parallel_dist = np.linalg.norm(parallel_component)
                
                # Barrier contribution: high if close to plane AND within extent
                if parallel_dist < extent:
                    # Exponential falloff with distance to plane
                    dist_to_plane = abs(signed_dist)
                    if dist_to_plane < self.barrier_thickness:
                        # Inside barrier zone
                        penetration = 1.0 - dist_to_plane / self.barrier_thickness
                        contribution = self.barrier_strength * (penetration ** 2)
                        
                        # Also scale by how centered we are (stronger in middle of barrier)
                        center_factor = 1.0 - (parallel_dist / extent) ** 2
                        contribution *= max(center_factor, 0.1)
                        
                        g += contribution
            
            return g
        
        def reset(self):
            start_arch = list(self.archetypes.keys())[np.random.randint(0, 3)]
            self.state = self.archetypes[start_arch] + np.random.randn(self.embed_dim) * 0.5
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
            move = np.clip(action, -1.0, 1.0)
            step_size = np.linalg.norm(move)
            
            # Preference reward
            pref_dir = self.get_preference_vector()
            base_reward = float(np.dot(move, pref_dir)) * 5.0
            
            # NEW position
            new_state = self.state + move
            
            # Geodesic cost: √g(x) * ||step||
            g = self.compute_barrier_metric(new_state)
            geodesic_cost = np.sqrt(g) * step_size
            
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
    
    class NaiveEnv:
        """Control: Same geometry but with simple proximity penalty (not hyperplane)."""
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
            
            # Point-based black holes (will be easy to miss)
            arch_list = list(self.archetypes.values())
            self.black_holes = []
            for i in range(3):
                midpoint = (arch_list[i] + arch_list[(i+1)%3]) / 2.0
                self.black_holes.append({
                    'center': midpoint,
                    'radius': 2.0,
                    'strength': 100.0
                })
        
        def compute_metric(self, state):
            g = 1.0
            for bh in self.black_holes:
                dist = np.linalg.norm(state - bh['center'])
                safe_dist = max(dist - bh['radius'], 0.1)
                contribution = bh['strength'] / (safe_dist ** 1.5)
                g += min(contribution, 10000.0)
            return g
        
        def reset(self):
            start_arch = list(self.archetypes.keys())[np.random.randint(0, 3)]
            self.state = self.archetypes[start_arch] + np.random.randn(self.embed_dim) * 0.5
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
            move = np.clip(action, -1.0, 1.0)
            step_size = np.linalg.norm(move)
            pref_dir = self.get_preference_vector()
            base_reward = float(np.dot(move, pref_dir)) * 5.0
            
            new_state = self.state + move
            g = self.compute_metric(new_state)
            geodesic_cost = np.sqrt(g) * step_size
            
            reward = base_reward - geodesic_cost
            self.state = new_state
            self.step_count += 1
            done = self.step_count >= self.max_steps
            
            return self.state.copy(), reward, done, {'g': g, 'geodesic_cost': geodesic_cost, 'step_size': step_size}
    
    # Networks
    class Actor(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 256), nn.LayerNorm(256), nn.Tanh(),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.Tanh(),
                nn.Linear(128, d)
            )
            self.log_std = nn.Parameter(torch.ones(1) * -0.5)
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
        
        history = {'returns': [], 'metrics': [], 'geodesic_costs': []}
        
        for ep in range(episodes):
            obs, done, traj = env.reset(), False, []
            ep_ret, ep_g, ep_geo = 0, [], []
            
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    dist = actor(obs_t)
                    action = dist.sample()
                    old_lp = dist.log_prob(action).sum()
                
                next_obs, r, done, info = env.step(action.squeeze(0).cpu().numpy())
                traj.append((obs, action, r, old_lp.item()))
                obs, ep_ret = next_obs, ep_ret + r
                ep_g.append(info['g'])
                ep_geo.append(info['geodesic_cost'])
            
            history['returns'].append(ep_ret)
            history['metrics'].append(np.mean(ep_g))
            history['geodesic_costs'].append(np.sum(ep_geo))
            
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
            entropy = actor(states).entropy().sum(dim=1).mean()
            loss = loss - 0.02 * entropy
            
            opt_a.zero_grad()
            loss.backward()
            opt_a.step()
            
            if ep % 100 == 0:
                print(f"{name} Ep {ep}: Ret={ep_ret:.1f}, g={np.mean(ep_g):.2f}, GeoCost={np.sum(ep_geo):.1f}")
        
        return history
    
    # Run experiments
    print("\n" + "="*70)
    print("HYPERPLANE BARRIER vs POINT SINGULARITY EXPERIMENT")
    print("="*70)
    print("\nHyperplane barriers create WALLS that can't be ignored in high-D.")
    print("Point singularities are easy to miss by chance.\n")
    
    barrier_env = HyperplaneBarrierEnv(embed_dim)
    naive_env = NaiveEnv(embed_dim)
    h1_truth = barrier_env.compute_h1_ground_truth()
    
    print(f"H1 Truth: {h1_truth:.2f}")
    print(f"Embed dim: {embed_dim}")
    
    print("\n" + "="*60 + "\nTraining with POINT singularities (control)...\n" + "="*60)
    naive_hist = train_agent(naive_env, episodes, "PointSing")
    
    print("\n" + "="*60 + "\nTraining with HYPERPLANE barriers...\n" + "="*60)
    barrier_hist = train_agent(barrier_env, episodes, "Hyperplane")
    
    # Analysis
    naive_g_final = np.mean(naive_hist['metrics'][-100:])
    barrier_g_final = np.mean(barrier_hist['metrics'][-100:])
    naive_geo_final = np.mean(naive_hist['geodesic_costs'][-100:])
    barrier_geo_final = np.mean(barrier_hist['geodesic_costs'][-100:])
    
    print(f"\n{'='*70}\nRESULTS\n{'='*70}")
    print(f"Point Singularities:   Avg g={naive_g_final:.2f}, Total GeoCost={naive_geo_final:.1f}")
    print(f"Hyperplane Barriers:   Avg g={barrier_g_final:.2f}, Total GeoCost={barrier_geo_final:.1f}")
    print(f"\nKey: Higher g = agent encounters barriers more")
    print(f"If barrier agent has LOWER g, it learned to detour!")
    print(f"g difference: {naive_g_final - barrier_g_final:.2f}")
    
    results = {
        'h1_truth': float(h1_truth),
        'naive_returns': naive_hist['returns'],
        'naive_metrics': naive_hist['metrics'],
        'naive_geocosts': naive_hist['geodesic_costs'],
        'barrier_returns': barrier_hist['returns'],
        'barrier_metrics': barrier_hist['metrics'],
        'barrier_geocosts': barrier_hist['geodesic_costs'],
        'config': {'embed_dim': embed_dim, 'episodes': episodes},
        'summary': {
            'naive_g_final': naive_g_final,
            'barrier_g_final': barrier_g_final,
            'naive_geo_final': naive_geo_final,
            'barrier_geo_final': barrier_geo_final,
            'g_difference': naive_g_final - barrier_g_final,
        }
    }
    
    with open("results_hyperplane.json", "w") as f: json.dump(results, f, indent=2)
    shutil.copy("results_hyperplane.json", f"{VOLUME_PATH}/hyperplane_barrier_results.json")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(naive_hist['returns'], label='Point Singularities', alpha=0.7)
    axes[0, 0].plot(barrier_hist['returns'], label='Hyperplane Barriers', alpha=0.7)
    axes[0, 0].set_title("Returns"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(naive_hist['metrics'], label='Point Singularities', alpha=0.7)
    axes[0, 1].plot(barrier_hist['metrics'], label='Hyperplane Barriers', alpha=0.7)
    axes[0, 1].set_title("Average Metric g(x)"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    
    window = 50
    naive_ma = np.convolve(naive_hist['metrics'], np.ones(window)/window, mode='valid')
    barrier_ma = np.convolve(barrier_hist['metrics'], np.ones(window)/window, mode='valid')
    axes[1, 0].plot(naive_ma, label='Point Singularities', alpha=0.7)
    axes[1, 0].plot(barrier_ma, label='Hyperplane Barriers', alpha=0.7)
    axes[1, 0].set_title(f"Metric g(x) - {window}-ep MA"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(naive_hist['geodesic_costs'], label='Point Singularities', alpha=0.7)
    axes[1, 1].plot(barrier_hist['geodesic_costs'], label='Hyperplane Barriers', alpha=0.7)
    axes[1, 1].set_title("Total Geodesic Cost per Episode"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("hyperplane_plot.png", dpi=150)
    shutil.copy("hyperplane_plot.png", f"{VOLUME_PATH}/hyperplane_barrier_results.png")
    volume.commit()
    
    print(f"\nSaved to {VOLUME_PATH}/hyperplane_barrier_results.json")
    return results['summary']
