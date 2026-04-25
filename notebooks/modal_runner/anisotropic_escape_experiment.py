"""
Anisotropic EscapeSGPO Experiment - Directional Singularities

This experiment tests the anisotropic metric that only penalizes movement
TOWARD black holes, preserving escape routes and learning signals.
"""

import modal

app = modal.App("anisotropic-escape-gpo")
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
def anisotropic_style_cycle(embed_dim: int = 768, episodes: int = 1000):
    """
    Anisotropic EscapeSGPO on High-Dim Style Cycle.
    
    Key innovation: Metric only diverges in direction TOWARD black holes.
    Escape and tangential movement remain free.
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
    print(f"Running Anisotropic EscapeSGPO on {DEVICE}")
    print(f"Config: d={embed_dim}, episodes={episodes}")
    
    # Environment (same as before)
    class HighDimStyleEnvWithBlackHoles:
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
                self.black_holes.append({
                    'center': (arch_list[i] + arch_list[(i+1)%3]) / 2.0,
                    'radius': 2.0,
                    'strength': 10.0
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
            total_cost = 0.0
            for bh in self.black_holes:
                dist = np.linalg.norm(state - bh['center'])
                if dist < bh['radius']:
                    total_cost += bh['strength'] * 10.0
                elif dist < bh['radius'] * 2:
                    proximity = 1.0 - (dist - bh['radius']) / bh['radius']
                    total_cost += bh['strength'] * (proximity ** 2)
            return total_cost
        
        def step(self, action):
            move = np.clip(action, -0.5, 0.5)
            pref_dir = self.get_preference_vector()
            base_reward = float(np.dot(move, pref_dir)) * 10.0
            new_state = self.state + move
            bh_cost = self.compute_black_hole_cost(new_state)
            center = sum(self.archetypes.values()) / 3.0
            if np.linalg.norm(new_state - center) > 20.0: bh_cost += 1.0
            reward = base_reward - bh_cost
            self.state = new_state
            self.step_count += 1
            done = self.step_count >= self.max_steps
            return self.state.copy(), reward, done, {'bh_cost': bh_cost}
        
        def compute_h1_ground_truth(self):
            v1, v2, v3 = self.archetypes['Concise'], self.archetypes['Empathy'], self.archetypes['Detail']
            return sum(10.0 * np.linalg.norm(e - s) for s, e in [(v1, v2), (v2, v3), (v3, v1)])
    
    # Networks
    class Actor(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 256), nn.LayerNorm(256), nn.Tanh(),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.Tanh(),
                nn.Linear(128, d)
            )
            self.log_std = nn.Parameter(torch.ones(1) * -1.0)
        def forward(self, x): return Normal(self.net(x), torch.exp(self.log_std).expand_as(self.net(x)))
    
    class ScalarCritic(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 256), nn.LayerNorm(256), nn.Tanh(),
                nn.Linear(256, 128), nn.Tanh(),
                nn.Linear(128, 1)
            )
        def forward(self, x): return self.net(x)
    
    class AnisotropicSGPOCritic(nn.Module):
        """Critic with ANISOTROPIC metric - directional singularities."""
        def __init__(self, d, black_holes):
            super().__init__()
            self.potential = nn.Sequential(
                nn.Linear(d, 256), nn.LayerNorm(256), nn.Tanh(),
                nn.Linear(256, 128), nn.Tanh(),
                nn.Linear(128, 1)
            )
            self.skew = nn.Parameter(torch.randn(d, d) * 0.01)
            self.metric_net = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
            self.black_holes = black_holes
            self.bh_centers = [torch.FloatTensor(bh['center']) for bh in black_holes]
            self.bh_radii = [bh['radius'] for bh in black_holes]
            self.bh_strengths = [bh['strength'] for bh in black_holes]
        
        def compute_anisotropic_metric(self, x, v):
            """
            Compute directional metric: g(x,v) = g_base + g_dir * (v_toward / |v|)^2
            Only penalizes movement TOWARD black holes.
            """
            device = x.device
            g_base = torch.relu(self.metric_net(x)) + 1.0
            
            # Compute toward-component ratio for each black hole
            toward_ratio_sq = torch.zeros(x.shape[0], device=device)
            
            for center, radius, strength in zip(self.bh_centers, self.bh_radii, self.bh_strengths):
                center = center.to(device)
                to_center = center.unsqueeze(0) - x
                dist = torch.norm(to_center, dim=-1)
                n_hat = to_center / (dist.unsqueeze(-1) + 1e-8)
                
                # Velocity component toward singularity
                v_toward = torch.sum(v * n_hat, dim=-1)
                v_toward_pos = torch.clamp(v_toward, min=0)  # Only penalize approach
                v_norm = torch.norm(v, dim=-1) + 1e-8
                
                # Singularity contribution (saturating)
                safe_dist = torch.clamp(dist - radius, min=1e-3)
                g_dir = strength / (safe_dist ** 1.5 + 1e-6)
                g_dir = torch.clamp(g_dir, max=1000.0)
                
                # Anisotropic scaling: only applies to toward-component
                toward_ratio_sq = torch.max(toward_ratio_sq, (v_toward_pos / v_norm) ** 2 * g_dir)
            
            return g_base.squeeze(-1) + toward_ratio_sq
        
        def compute_escape_factor(self, x, v):
            """
            Compute escape factor: 1.0 if escaping, 0.0 if approaching.
            Used to preserve learning signal for escape maneuvers.
            """
            device = x.device
            escape_factor = torch.ones(x.shape[0], device=device)
            
            for center in self.bh_centers:
                center = center.to(device)
                to_center = center.unsqueeze(0) - x
                dist = torch.norm(to_center, dim=-1)
                n_hat = to_center / (dist.unsqueeze(-1) + 1e-8)
                
                # Negative v_toward means escaping
                v_toward = torch.sum(v * n_hat, dim=-1)
                escape_factor = escape_factor * torch.sigmoid(-v_toward * 5)
            
            return escape_factor
        
        def forward(self, x, v):
            """Returns: value, harmonic, metric, escape_factor."""
            W = self.skew - self.skew.t()
            V = self.potential(x)
            omega = torch.matmul(x, W)
            g = self.compute_anisotropic_metric(x, v)
            escape = self.compute_escape_factor(x, v)
            return V, omega, g, escape
    
    def train_ppo(env, episodes):
        actor = Actor(env.embed_dim).to(DEVICE)
        critic = ScalarCritic(env.embed_dim).to(DEVICE)
        opt_a = optim.Adam(actor.parameters(), lr=3e-4)
        opt_c = optim.Adam(critic.parameters(), lr=1e-3)
        history = {'returns': [], 'bh_costs': []}
        
        for ep in range(episodes):
            obs, done, traj, ep_ret, ep_bh = env.reset(), False, [], 0, 0
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    dist = actor(obs_t)
                    action = dist.sample()
                    old_lp = dist.log_prob(action).sum()
                next_obs, r, done, info = env.step(action.squeeze(0).cpu().numpy())
                traj.append((obs, action, r, old_lp.item()))
                obs, ep_ret, ep_bh = next_obs, ep_ret + r, ep_bh + info.get('bh_cost', 0)
            
            history['returns'].append(ep_ret)
            history['bh_costs'].append(ep_bh)
            
            states = torch.FloatTensor(np.array([t[0] for t in traj])).to(DEVICE)
            actions = torch.cat([t[1] for t in traj]).to(DEVICE)
            rewards = torch.FloatTensor([t[2] for t in traj]).to(DEVICE)
            old_lps = torch.FloatTensor([t[3] for t in traj]).to(DEVICE)
            
            vals = critic(states).squeeze()
            opt_c.zero_grad(); nn.MSELoss()(vals, rewards).backward(); opt_c.step()
            adv = rewards - vals.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            new_lps = actor(states).log_prob(actions).sum(dim=1)
            ratio = torch.exp(new_lps - old_lps)
            loss = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
            opt_a.zero_grad(); loss.backward(); opt_a.step()
            
            if ep % 100 == 0: print(f"PPO Ep {ep}: Ret={ep_ret:.1f}, BH={ep_bh:.1f}")
        return history
    
    def train_anisotropic_gpo(env, episodes):
        actor = Actor(env.embed_dim).to(DEVICE)
        critic = AnisotropicSGPOCritic(env.embed_dim, env.black_holes).to(DEVICE)
        opt_a = optim.Adam(actor.parameters(), lr=3e-4)
        opt_c = optim.Adam(critic.parameters(), lr=1e-3)
        history = {'returns': [], 'bh_costs': [], 'escape_ratios': [], 'metrics': []}
        
        for ep in range(episodes):
            obs, done, traj, ep_ret, ep_bh = env.reset(), False, [], 0, 0
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    dist = actor(obs_t)
                    action = dist.sample()
                    old_lp = dist.log_prob(action).sum()
                next_obs, r, done, info = env.step(action.squeeze(0).cpu().numpy())
                traj.append((obs, next_obs, action, r, old_lp.item()))
                obs, ep_ret, ep_bh = next_obs, ep_ret + r, ep_bh + info.get('bh_cost', 0)
            
            history['returns'].append(ep_ret)
            history['bh_costs'].append(ep_bh)
            
            states = torch.FloatTensor(np.array([t[0] for t in traj])).to(DEVICE)
            next_states = torch.FloatTensor(np.array([t[1] for t in traj])).to(DEVICE)
            actions = torch.cat([t[2] for t in traj]).to(DEVICE)
            rewards = torch.FloatTensor([t[3] for t in traj]).to(DEVICE)
            old_lps = torch.FloatTensor([t[4] for t in traj]).to(DEVICE)
            
            # Critic update
            V, omega, g, escape = critic(states, actions)
            V_next, _, _, _ = critic(next_states, actions)
            dV = (V_next - V).squeeze()
            omega_contrib = (omega * actions).sum(dim=1)
            pred = dV + omega_contrib
            opt_c.zero_grad(); nn.MSELoss()(pred, rewards).backward(); opt_c.step()
            
            # Compute ANISOTROPIC advantages
            with torch.no_grad():
                V, omega, g, escape = critic(states, actions)
                V_next, _, _, _ = critic(next_states, actions)
                td_error = rewards - V.squeeze() + V_next.squeeze() - (omega * actions).sum(dim=1)
                
                # Directional scaling: full signal for escapes, dampened for approaches
                scale = escape * 1.0 + (1 - escape) / torch.sqrt(g)
                adv = scale * td_error
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                
                history['escape_ratios'].append(escape.mean().item())
                history['metrics'].append(g.mean().item())
            
            # Actor update
            new_lps = actor(states).log_prob(actions).sum(dim=1)
            ratio = torch.exp(new_lps - old_lps)
            loss = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
            entropy = actor(states).entropy().sum(dim=1).mean()
            loss = loss - 0.01 * entropy
            opt_a.zero_grad(); loss.backward(); opt_a.step()
            
            if ep % 100 == 0:
                print(f"AnisSGPO Ep {ep}: Ret={ep_ret:.1f}, BH={ep_bh:.1f}, "
                      f"Escape={escape.mean():.3f}, g={g.mean():.2f}")
        return history
    
    # Run experiments
    env = HighDimStyleEnvWithBlackHoles(embed_dim)
    h1_truth = env.compute_h1_ground_truth()
    print(f"\nH1 Truth: {h1_truth:.2f}, Black Holes: {len(env.black_holes)}\n")
    
    print("="*60 + "\nTraining PPO...\n" + "="*60)
    ppo_hist = train_ppo(env, episodes)
    
    print("\n" + "="*60 + "\nTraining Anisotropic SGPO...\n" + "="*60)
    anis_hist = train_anisotropic_gpo(env, episodes)
    
    # Analysis
    ppo_mean, ppo_final = np.mean(ppo_hist['returns']), np.mean(ppo_hist['returns'][-100:])
    anis_mean, anis_final = np.mean(anis_hist['returns']), np.mean(anis_hist['returns'][-100:])
    
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"PPO:           Mean={ppo_mean:.1f}, Final100={ppo_final:.1f}")
    print(f"AnisotropicSGPO: Mean={anis_mean:.1f}, Final100={anis_final:.1f}")
    print(f"Improvement: {(anis_final - ppo_final):.1f} ({100*(anis_final/ppo_final - 1):.1f}%)")
    
    # Save results
    results = {
        'h1_truth': float(h1_truth),
        'ppo_returns': ppo_hist['returns'],
        'ppo_bh_costs': ppo_hist['bh_costs'],
        'anis_returns': anis_hist['returns'],
        'anis_bh_costs': anis_hist['bh_costs'],
        'anis_escape_ratios': anis_hist['escape_ratios'],
        'anis_metrics': anis_hist['metrics'],
        'config': {'embed_dim': embed_dim, 'episodes': episodes},
        'summary': {
            'ppo_mean': ppo_mean, 'ppo_final': ppo_final,
            'anis_mean': anis_mean, 'anis_final': anis_final,
            'improvement': anis_final - ppo_final,
        }
    }
    
    with open("results_anisotropic.json", "w") as f: json.dump(results, f, indent=2)
    shutil.copy("results_anisotropic.json", f"{VOLUME_PATH}/anisotropic_style_cycle.json")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(ppo_hist['returns'], label='PPO', alpha=0.7)
    axes[0, 0].plot(anis_hist['returns'], label='Anisotropic SGPO', alpha=0.7)
    axes[0, 0].axhline(y=h1_truth, color='k', linestyle='--', label='H¹ Truth', alpha=0.5)
    axes[0, 0].set_title(f"Returns (d={embed_dim})"); axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Return"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(ppo_hist['bh_costs'], label='PPO', alpha=0.7)
    axes[0, 1].plot(anis_hist['bh_costs'], label='Anisotropic SGPO', alpha=0.7)
    axes[0, 1].set_title("Black Hole Costs"); axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Cost"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].plot(anis_hist['escape_ratios'], color='green', alpha=0.7)
    axes[1, 0].set_title("Anisotropic SGPO: Escape Ratio"); axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Escape Factor"); axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(anis_hist['metrics'], color='purple', alpha=0.7)
    axes[1, 1].set_title("Anisotropic SGPO: Mean Metric"); axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("g(x,v)"); axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("anisotropic_plot.png", dpi=150)
    shutil.copy("anisotropic_plot.png", f"{VOLUME_PATH}/anisotropic_style_cycle.png")
    volume.commit()
    
    print(f"\nSaved to {VOLUME_PATH}/anisotropic_style_cycle.json")
    return {
        "ppo_mean": ppo_mean, "ppo_final": ppo_final,
        "anis_mean": anis_mean, "anis_final": anis_final,
        "improvement": anis_final - ppo_final,
    }
