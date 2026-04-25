"""
Corrected Anisotropic EscapeSGPO Experiment

Key fix: Black holes are placed OFF the optimal path, so agents can:
1. Follow the preference cycle without hitting black holes
2. Accidentally wander near black holes due to exploration
3. Demonstrate escape capability with anisotropic metric
"""

import modal

app = modal.App("corrected-anisotropic-gpo")
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
def corrected_anisotropic_experiment(embed_dim: int = 768, episodes: int = 1000):
    """
    Corrected environment: Black holes OFF the optimal path.
    
    This tests whether anisotropic SGPO can:
    1. Navigate the preference cycle efficiently
    2. Avoid accidentally entering black holes
    3. Escape if it gets too close
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
    print(f"Running CORRECTED Anisotropic EscapeSGPO on {DEVICE}")
    print(f"Config: d={embed_dim}, episodes={episodes}")
    
    class CorrectedStyleEnv:
        """
        Key fix: Black holes placed AWAY from archetype transition paths.
        Agents can follow preference cycle without hitting black holes.
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
            
            # BLACK HOLES: Place them PERPENDICULAR to the cycle, not on it
            # This allows agents to navigate the cycle while avoiding danger
            center = sum(self.archetypes.values()) / 3.0
            
            # Place black holes at 45-degree angles from archetypes
            # (between the "spokes" rather than on them)
            self.black_holes = []
            arch_list = list(self.archetypes.values())
            for i in range(3):
                # Midpoint between archetypes (on the cycle)
                midpoint = (arch_list[i] + arch_list[(i+1)%3]) / 2.0
                
                # Perpendicular direction (away from cycle)
                to_mid = midpoint - center
                # Rotate 90 degrees in a random 2D plane
                perp_dir = Q[:, (i+3) % embed_dim] * np.linalg.norm(to_mid)
                
                # Black hole location: off the cycle
                bh_center = center + perp_dir * 0.7
                
                self.black_holes.append({
                    'center': bh_center,
                    'radius': 1.5,  # Smaller radius
                    'strength': 20.0  # Higher strength
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
            """Higher cost for being inside black holes."""
            total_cost = 0.0
            for bh in self.black_holes:
                dist = np.linalg.norm(state - bh['center'])
                if dist < bh['radius']:
                    # Inside: very high cost
                    total_cost += bh['strength'] * 20.0
                elif dist < bh['radius'] * 2:
                    # Near: increasing cost
                    proximity = 1.0 - (dist - bh['radius']) / bh['radius']
                    total_cost += bh['strength'] * (proximity ** 2)
            return total_cost
        
        def step(self, action):
            move = np.clip(action, -0.5, 0.5)
            pref_dir = self.get_preference_vector()
            
            # Base reward for following preference cycle
            base_reward = float(np.dot(move, pref_dir)) * 10.0
            
            # Cost from black holes (now avoidable)
            new_state = self.state + move
            bh_cost = self.compute_black_hole_cost(new_state)
            
            # Small penalty for straying too far from center
            center = sum(self.archetypes.values()) / 3.0
            if np.linalg.norm(new_state - center) > 25.0:
                bh_cost += 0.5
            
            reward = base_reward - bh_cost
            
            self.state = new_state
            self.step_count += 1
            done = self.step_count >= self.max_steps
            
            return self.state.copy(), reward, done, {'bh_cost': bh_cost}
        
        def compute_h1_ground_truth(self):
            v1, v2, v3 = self.archetypes['Concise'], self.archetypes['Empathy'], self.archetypes['Detail']
            return sum(10.0 * np.linalg.norm(e - s) for s, e in [(v1, v2), (v2, v3), (v3, v1)])
    
    # Networks (same as before)
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
            device = x.device
            g_base = torch.relu(self.metric_net(x)) + 1.0
            toward_ratio_sq = torch.zeros(x.shape[0], device=device)
            
            for center, radius, strength in zip(self.bh_centers, self.bh_radii, self.bh_strengths):
                center = center.to(device)
                to_center = center.unsqueeze(0) - x
                dist = torch.norm(to_center, dim=-1)
                n_hat = to_center / (dist.unsqueeze(-1) + 1e-8)
                
                v_toward = torch.sum(v * n_hat, dim=-1)
                v_toward_pos = torch.clamp(v_toward, min=0)
                v_norm = torch.norm(v, dim=-1) + 1e-8
                
                safe_dist = torch.clamp(dist - radius, min=1e-3)
                g_dir = strength / (safe_dist ** 1.5 + 1e-6)
                g_dir = torch.clamp(g_dir, max=1000.0)
                
                toward_ratio_sq = torch.max(toward_ratio_sq, (v_toward_pos / v_norm) ** 2 * g_dir)
            
            return g_base.squeeze(-1) + toward_ratio_sq
        
        def compute_escape_factor(self, x, v):
            device = x.device
            escape_factor = torch.ones(x.shape[0], device=device)
            
            for center in self.bh_centers:
                center = center.to(device)
                to_center = center.unsqueeze(0) - x
                dist = torch.norm(to_center, dim=-1)
                n_hat = to_center / (dist.unsqueeze(-1) + 1e-8)
                v_toward = torch.sum(v * n_hat, dim=-1)
                escape_factor = escape_factor * torch.sigmoid(-v_toward * 5)
            
            return escape_factor
        
        def forward(self, x, v):
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
            
            V, omega, g, escape = critic(states, actions)
            V_next, _, _, _ = critic(next_states, actions)
            dV = (V_next - V).squeeze()
            omega_contrib = (omega * actions).sum(dim=1)
            pred = dV + omega_contrib
            opt_c.zero_grad(); nn.MSELoss()(pred, rewards).backward(); opt_c.step()
            
            with torch.no_grad():
                V, omega, g, escape = critic(states, actions)
                V_next, _, _, _ = critic(next_states, actions)
                td_error = rewards - V.squeeze() + V_next.squeeze() - (omega * actions).sum(dim=1)
                scale = escape * 1.0 + (1 - escape) / torch.sqrt(g)
                adv = scale * td_error
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                history['escape_ratios'].append(escape.mean().item())
                history['metrics'].append(g.mean().item())
            
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
    env = CorrectedStyleEnv(embed_dim)
    h1_truth = env.compute_h1_ground_truth()
    print(f"\nH1 Truth: {h1_truth:.2f}, Black Holes: {len(env.black_holes)}")
    print("BLACK HOLES NOW OFF THE OPTIMAL PATH - agents can avoid them!\n")
    
    print("="*60 + "\nTraining PPO...\n" + "="*60)
    ppo_hist = train_ppo(env, episodes)
    
    print("\n" + "="*60 + "\nTraining Anisotropic SGPO...\n" + "="*60)
    anis_hist = train_anisotropic_gpo(env, episodes)
    
    ppo_mean, ppo_final = np.mean(ppo_hist['returns']), np.mean(ppo_hist['returns'][-100:])
    anis_mean, anis_final = np.mean(anis_hist['returns']), np.mean(anis_hist['returns'][-100:])
    ppo_bh_mean, ppo_bh_final = np.mean(ppo_hist['bh_costs']), np.mean(ppo_hist['bh_costs'][-100:])
    anis_bh_mean, anis_bh_final = np.mean(anis_hist['bh_costs']), np.mean(anis_hist['bh_costs'][-100:])
    
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"PPO:           Return={ppo_final:.1f}, BH_Cost={ppo_bh_final:.1f}")
    print(f"AnisotropicSGPO: Return={anis_final:.1f}, BH_Cost={anis_bh_final:.1f}")
    print(f"Return Improvement: {(anis_final - ppo_final):.1f}")
    print(f"Safety Improvement: {(ppo_bh_final - anis_bh_final):.1f} (lower is better)")
    
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
            'ppo_bh_mean': ppo_bh_mean, 'ppo_bh_final': ppo_bh_final,
            'anis_mean': anis_mean, 'anis_final': anis_final,
            'anis_bh_mean': anis_bh_mean, 'anis_bh_final': anis_bh_final,
            'return_improvement': anis_final - ppo_final,
            'safety_improvement': ppo_bh_final - anis_bh_final,
        }
    }
    
    with open("results_corrected.json", "w") as f: json.dump(results, f, indent=2)
    shutil.copy("results_corrected.json", f"{VOLUME_PATH}/corrected_anisotropic.json")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(ppo_hist['returns'], label='PPO', alpha=0.7)
    axes[0, 0].plot(anis_hist['returns'], label='Anisotropic SGPO', alpha=0.7)
    axes[0, 0].axhline(y=h1_truth, color='k', linestyle='--', label='H¹ Truth', alpha=0.5)
    axes[0, 0].set_title("Returns"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(ppo_hist['bh_costs'], label='PPO', alpha=0.7)
    axes[0, 1].plot(anis_hist['bh_costs'], label='Anisotropic SGPO', alpha=0.7)
    axes[0, 1].set_title("Black Hole Costs (Lower=Better)"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].plot(anis_hist['escape_ratios'], color='green', alpha=0.7)
    axes[1, 0].set_title("Escape Ratio"); axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(anis_hist['metrics'], color='purple', alpha=0.7)
    axes[1, 1].set_title("Mean Metric g(x,v)"); axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("corrected_plot.png", dpi=150)
    shutil.copy("corrected_plot.png", f"{VOLUME_PATH}/corrected_anisotropic.png")
    volume.commit()
    
    print(f"\nSaved to {VOLUME_PATH}/corrected_anisotropic.json")
    return results['summary']
