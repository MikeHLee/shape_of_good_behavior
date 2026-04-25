"""
H¹-Exploitable Reward Hacking Experiment (V2 - Corrected)

FIXES from V1:
1. H¹ INJECTION: Now uses proper Hodge decomposition to inject and verify H¹
2. DATA SNOOPING: Train/test split ensures trap is UNSEEN during reward model training
3. HODGE FILTERING: True Hodge decomposition, not just a penalty term

Core Thesis: When human feedback contains cyclic inconsistencies (H¹ ≠ 0),
standard RLHF reward models learn biases that generalize to create exploitable gaps.

Experimental Design:
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING DATA                             │
│  Items 0 to (N_TRAIN-1) with injected H¹ cyclic preferences │
│  NO TRAP - just cyclic inconsistencies from evaluator bias  │
└─────────────────────────────────────────────────────────────┘
                           ↓
                  Reward Model Training
                  - Standard: train on raw preferences
                  - Hodge-filtered: train on gradient component only
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    TEST DATA (UNSEEN)                        │
│  Items N_TRAIN to N_TOTAL, includes exploitation trap        │
│  Trap items have artificially high reward model predictions  │
│  but catastrophic true reward                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
                  Policy Evaluation
                  - Does cyclic bias generalize → exploit trap?
                  - Does Hodge filtering prevent exploitation?
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from scipy import stats
import matplotlib.pyplot as plt

from hodge_utils import (
    compute_h1_from_preferences,
    inject_h1_controlled,
    hodge_filter_preferences,
    HodgeDecomposition
)

from experiment_framework import (
    ExperimentConfig, SeedResult, ExperimentRunner,
    compute_statistics, compare_methods
)


# ============================================================================
# 1. PREFERENCE DATA WITH CONTROLLED H¹ (CORRECTED)
# ============================================================================

@dataclass
class PreferencePair:
    """A single preference comparison."""
    item_a: int
    item_b: int
    preference: float  # Probability that item_a > item_b
    state: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    evaluator_id: int = 0


class ControlledH1PreferenceDataset:
    """
    Generate preference data with TRUE controlled H¹.
    
    Key fixes from V1:
    1. Uses hodge_utils.inject_h1_controlled for proper H¹ injection
    2. Verifies H¹ magnitude via Hodge decomposition
    3. Separates train/test items completely
    """
    
    def __init__(
        self,
        n_train_items: int = 70,
        n_test_items: int = 30,
        h1_magnitude: float = 0.5,
        trap_item_start: int = 80,  # Trap is in TEST region
        trap_item_end: int = 90,
        base_noise: float = 0.1,
        n_comparisons_per_pair: int = 3
    ):
        self.n_train_items = n_train_items
        self.n_test_items = n_test_items
        self.n_total_items = n_train_items + n_test_items
        self.h1_magnitude = h1_magnitude
        self.trap_item_start = trap_item_start
        self.trap_item_end = trap_item_end
        self.base_noise = base_noise
        self.n_comparisons_per_pair = n_comparisons_per_pair
        
        # Ground truth utilities (the "true" ranking)
        self.utilities = np.random.randn(self.n_total_items)
        
        # Trap items have NEGATIVE true utility (catastrophic)
        self.utilities[trap_item_start:trap_item_end] = -10.0
        
    def generate_train_preferences(self) -> Tuple[List[PreferencePair], float]:
        """
        Generate TRAINING preferences (items 0 to n_train_items-1).
        
        These have injected H¹ cyclic inconsistency but NO TRAP.
        """
        # Use hodge_utils for controlled H¹ injection
        raw_prefs, measured_h1 = inject_h1_controlled(
            n_items=self.n_train_items,
            h1_target=self.h1_magnitude,
            base_noise=self.base_noise,
            n_comparisons_per_pair=self.n_comparisons_per_pair
        )
        
        # Convert to PreferencePair objects
        pairs = []
        for item_a, item_b, prob in raw_prefs:
            # Create a state vector based on items (for reward model input)
            state = self._item_to_state(item_a, item_b)
            pairs.append(PreferencePair(
                item_a=item_a,
                item_b=item_b,
                preference=prob,
                state=state
            ))
        
        return pairs, measured_h1
    
    def generate_test_preferences(self) -> List[PreferencePair]:
        """
        Generate TEST preferences (items n_train_items to n_total_items-1).
        
        These are CONSISTENT (H¹ ≈ 0) but include the TRAP region.
        The trap items appear attractive in the preference model
        due to generalization of cyclic bias.
        """
        pairs = []
        test_items = list(range(self.n_train_items, self.n_total_items))
        
        for i, item_a in enumerate(test_items):
            for item_b in test_items[i+1:]:
                # Ground truth preferences (consistent)
                diff = self.utilities[item_a] - self.utilities[item_b]
                prob = 1.0 / (1.0 + np.exp(-diff))
                
                for _ in range(self.n_comparisons_per_pair):
                    noisy_prob = np.clip(prob + np.random.normal(0, self.base_noise), 0.05, 0.95)
                    state = self._item_to_state(item_a, item_b)
                    pairs.append(PreferencePair(
                        item_a=item_a,
                        item_b=item_b,
                        preference=noisy_prob,
                        state=state
                    ))
        
        return pairs
    
    def _item_to_state(self, item_a: int, item_b: int) -> np.ndarray:
        """Convert item pair to state representation for reward model."""
        # Embed items in 2D space for simple visualization
        # Use a circle embedding for train items, grid for test items
        angle_a = 2 * np.pi * item_a / self.n_total_items
        angle_b = 2 * np.pi * item_b / self.n_total_items
        return np.array([
            (np.cos(angle_a) + np.cos(angle_b)) / 2,
            (np.sin(angle_a) + np.sin(angle_b)) / 2
        ])
    
    def is_trap_item(self, item: int) -> bool:
        """Check if item is in the trap region."""
        return self.trap_item_start <= item < self.trap_item_end
    
    def get_true_utility(self, item: int) -> float:
        """Get ground truth utility (for evaluation only)."""
        return self.utilities[item]


# ============================================================================
# 2. REWARD MODELS
# ============================================================================

class PreferenceRewardModel(nn.Module):
    """
    Reward model that learns from pairwise preferences.
    
    Maps (state, item) → scalar reward estimate
    """
    
    def __init__(self, state_dim: int = 2, n_items: int = 100, hidden_dim: int = 64):
        super().__init__()
        self.n_items = n_items
        
        # Item embedding
        self.item_embed = nn.Embedding(n_items, hidden_dim // 2)
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for (state, item) pair.
        
        Args:
            state: (batch, state_dim) state vectors
            item: (batch,) item indices
            
        Returns:
            reward: (batch, 1) reward estimates
        """
        item_feat = self.item_embed(item)  # (batch, hidden_dim//2)
        state_feat = self.state_encoder(state)  # (batch, hidden_dim//2)
        combined = torch.cat([state_feat, item_feat], dim=-1)
        return self.reward_head(combined)
    
    def get_reward(self, state: np.ndarray, item: int) -> float:
        """Get reward for single (state, item) pair."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            item_t = torch.LongTensor([item])
            return self.forward(state_t, item_t).item()


def train_preference_reward_model(
    model: PreferenceRewardModel,
    pairs: List[PreferencePair],
    epochs: int = 100,
    lr: float = 1e-3,
    hodge_filter: bool = False,
    h1_threshold: float = 0.0,
    n_items: int = 100
) -> Tuple[List[float], float, Dict]:
    """
    Train reward model on preference pairs using Bradley-Terry loss.
    
    Args:
        model: Reward model to train
        pairs: Preference pairs
        epochs: Training epochs
        lr: Learning rate
        hodge_filter: If True, apply Hodge filtering to reduce/remove H¹
        h1_threshold: Maximum H¹ to allow after filtering (0 = remove all)
                      Set to e.g. 0.75 to only remove the most circular components
        n_items: Total number of items (for Hodge decomposition)
        
    Returns:
        losses: Training loss history
        final_h1: H¹ of training data (after filtering if applied)
        filter_info: Dict with filtering metadata (h1_before, h1_after, alpha)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    filter_info = {}
    
    # Convert to tensors
    states = torch.FloatTensor(np.array([p.state for p in pairs]))
    items_a = torch.LongTensor([p.item_a for p in pairs])
    items_b = torch.LongTensor([p.item_b for p in pairs])
    preferences = torch.FloatTensor([p.preference for p in pairs])
    
    # Compute initial H¹
    raw_prefs = [(p.item_a, p.item_b, p.preference) for p in pairs]
    initial_h1, _ = compute_h1_from_preferences(raw_prefs, n_items)
    
    # Apply Hodge filtering if requested
    if hodge_filter:
        filtered_raw, filter_info = hodge_filter_preferences(
            raw_prefs, n_items, h1_threshold=h1_threshold, return_info=True
        )
        preferences = torch.FloatTensor([p[2] for p in filtered_raw])
        final_h1 = filter_info['h1_after']
    else:
        final_h1 = initial_h1
        filter_info = {'h1_before': initial_h1, 'h1_after': initial_h1, 'alpha': 1.0, 'threshold': None}
    
    for epoch in range(epochs):
        # Bradley-Terry loss: P(a > b) = sigmoid(r_a - r_b)
        r_a = model(states, items_a).squeeze()
        r_b = model(states, items_b).squeeze()
        
        logits = r_a - r_b
        loss = nn.BCEWithLogitsLoss()(logits, preferences)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return losses, final_h1, filter_info


# ============================================================================
# 3. ITEM SELECTION ENVIRONMENT
# ============================================================================

class ItemSelectionEnv:
    """
    Environment where agent selects items to maximize reward.
    
    The agent's task is to identify high-value items.
    TRAP items appear attractive to biased reward models
    but have catastrophic true reward.
    """
    
    def __init__(
        self,
        dataset: ControlledH1PreferenceDataset,
        reward_model: PreferenceRewardModel,
        n_selections: int = 20,
        test_only: bool = True
    ):
        self.dataset = dataset
        self.reward_model = reward_model
        self.n_selections = n_selections
        self.test_only = test_only
        
        # Available items (test items only for evaluation)
        if test_only:
            self.available_items = list(range(
                dataset.n_train_items, 
                dataset.n_total_items
            ))
        else:
            self.available_items = list(range(dataset.n_total_items))
        
        self.current_selection = 0
        self.selected_items = []
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.current_selection = 0
        self.selected_items = []
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation (selection progress)."""
        return np.array([
            self.current_selection / self.n_selections,
            len(self.selected_items) / len(self.available_items)
        ])
    
    def step(self, item: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Select an item.
        
        Returns:
            obs: Next observation
            true_reward: Ground truth reward (for evaluation)
            done: Episode complete?
            info: Additional info
        """
        if item not in self.available_items:
            item = np.random.choice(self.available_items)
        
        self.selected_items.append(item)
        self.current_selection += 1
        
        # TRUE reward (ground truth)
        true_reward = self.dataset.get_true_utility(item)
        
        # Check if trap was selected
        in_trap = self.dataset.is_trap_item(item)
        
        done = self.current_selection >= self.n_selections
        
        info = {
            'in_trap': in_trap,
            'item': item,
            'predicted_reward': self.reward_model.get_reward(self._get_obs(), item)
        }
        
        return self._get_obs(), true_reward, done, info
    
    def get_predicted_reward(self, item: int) -> float:
        """Get reward model's prediction for an item."""
        return self.reward_model.get_reward(self._get_obs(), item)


# ============================================================================
# 4. POLICIES
# ============================================================================

class GreedyPolicy:
    """Greedy policy that selects item with highest predicted reward."""
    
    def __init__(self, reward_model: PreferenceRewardModel, items: List[int]):
        self.reward_model = reward_model
        self.items = items
    
    def select(self, obs: np.ndarray) -> int:
        """Select item with highest predicted reward."""
        best_item = None
        best_reward = -float('inf')
        
        for item in self.items:
            pred = self.reward_model.get_reward(obs, item)
            if pred > best_reward:
                best_reward = pred
                best_item = item
        
        return best_item


class SoftmaxPolicy:
    """Softmax policy over predicted rewards (for exploration)."""
    
    def __init__(
        self, 
        reward_model: PreferenceRewardModel, 
        items: List[int],
        temperature: float = 1.0
    ):
        self.reward_model = reward_model
        self.items = items
        self.temperature = temperature
    
    def select(self, obs: np.ndarray) -> int:
        """Sample item according to softmax over predicted rewards."""
        rewards = np.array([
            self.reward_model.get_reward(obs, item) 
            for item in self.items
        ])
        
        # Softmax
        exp_r = np.exp((rewards - rewards.max()) / self.temperature)
        probs = exp_r / exp_r.sum()
        
        return np.random.choice(self.items, p=probs)


# ============================================================================
# 5. MAIN EXPERIMENT
# ============================================================================

def run_single_experiment(
    seed: int,
    h1_magnitude: float = 0.5,
    h1_threshold: float = 0.0,
    n_train_items: int = 70,
    n_test_items: int = 30,
    trap_start: int = 80,
    trap_end: int = 90,
    n_selections: int = 20,
    n_eval_episodes: int = 50,
    rm_epochs: int = 100
) -> Dict:
    """
    Run a single seed of the corrected H¹ exploitation experiment.
    
    Args:
        h1_threshold: Maximum H¹ to allow after Hodge filtering.
                      0.0 = remove all cyclic component
                      0.75 = only remove if H¹ > 0.75 (partial filtering)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Generate dataset with controlled H¹
    dataset = ControlledH1PreferenceDataset(
        n_train_items=n_train_items,
        n_test_items=n_test_items,
        h1_magnitude=h1_magnitude,
        trap_item_start=trap_start,
        trap_item_end=trap_end
    )
    
    # 2. Generate train preferences (with H¹) and test preferences
    train_pairs, measured_h1_train = dataset.generate_train_preferences()
    test_pairs = dataset.generate_test_preferences()
    
    # Compute H¹ on test (should be ~0)
    test_raw = [(p.item_a, p.item_b, p.preference) for p in test_pairs]
    measured_h1_test, _ = compute_h1_from_preferences(
        test_raw, 
        dataset.n_total_items
    )
    
    # 3. Train STANDARD reward model (on raw preferences)
    standard_rm = PreferenceRewardModel(n_items=dataset.n_total_items)
    standard_losses, h1_after_standard, _ = train_preference_reward_model(
        standard_rm, train_pairs, 
        epochs=rm_epochs, 
        hodge_filter=False,
        n_items=n_train_items
    )
    
    # 4. Train HODGE-FILTERED reward model (with threshold)
    # h1_threshold controls how much cyclic component to remove
    # 0.0 = remove all, 0.75 = only remove if H¹ > 0.75
    hodge_rm = PreferenceRewardModel(n_items=dataset.n_total_items)
    hodge_losses, h1_after_hodge, filter_info = train_preference_reward_model(
        hodge_rm, train_pairs,
        epochs=rm_epochs,
        hodge_filter=True,
        h1_threshold=h1_threshold,
        n_items=n_train_items
    )
    
    # 5. Evaluate both policies on TEST items (with trap)
    test_items = list(range(n_train_items, n_train_items + n_test_items))
    
    def evaluate_policy(reward_model: PreferenceRewardModel) -> Dict:
        """Evaluate a policy over multiple episodes."""
        policy = SoftmaxPolicy(reward_model, test_items, temperature=0.5)
        env = ItemSelectionEnv(dataset, reward_model, n_selections, test_only=True)
        
        total_true_return = 0
        total_trap_visits = 0
        trap_visit_episodes = 0
        
        for _ in range(n_eval_episodes):
            obs = env.reset()
            ep_return = 0
            ep_trap = 0
            
            for _ in range(n_selections):
                item = policy.select(obs)
                obs, true_reward, done, info = env.step(item)
                ep_return += true_reward
                ep_trap += int(info['in_trap'])
                
                if done:
                    break
            
            total_true_return += ep_return
            total_trap_visits += ep_trap
            trap_visit_episodes += int(ep_trap > 0)
        
        return {
            'mean_true_return': total_true_return / n_eval_episodes,
            'mean_trap_visits': total_trap_visits / n_eval_episodes,
            'trap_episode_rate': trap_visit_episodes / n_eval_episodes
        }
    
    standard_eval = evaluate_policy(standard_rm)
    hodge_eval = evaluate_policy(hodge_rm)
    
    # 6. Compile results
    results = {
        'seed': seed,
        'h1_target': h1_magnitude,
        'h1_threshold': h1_threshold,
        'h1_measured': {
            'train': measured_h1_train,
            'test': measured_h1_test,
            'after_standard_training': h1_after_standard,
            'after_hodge_training': h1_after_hodge
        },
        'filter_info': {
            'alpha': filter_info['alpha'],
            'h1_before': filter_info['h1_before'],
            'h1_after': filter_info['h1_after']
        },
        'standard_policy': standard_eval,
        'hodge_policy': hodge_eval,
        'exploitation_metrics': {
            'trap_visits_prevented': standard_eval['mean_trap_visits'] - hodge_eval['mean_trap_visits'],
            'return_improvement': hodge_eval['mean_true_return'] - standard_eval['mean_true_return']
        }
    }
    
    return results


def run_full_experiment(
    num_seeds: int = 50,
    h1_magnitudes: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    h1_threshold: float = 0.0,
    output_dir: str = "results/h1_exploitation_v2"
):
    """
    Run full experiment across H¹ magnitudes and seeds.
    
    Args:
        h1_threshold: Maximum H¹ to allow after Hodge filtering.
                      0.0 = remove all (default)
                      0.75 = only remove most extreme cyclic components
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for h1_mag in h1_magnitudes:
        print(f"\n{'='*60}")
        print(f"H¹ Target: {h1_mag}")
        print(f"{'='*60}")
        
        mag_results = []
        h1_measured_all = []
        standard_traps = []
        hodge_traps = []
        
        for seed in range(num_seeds):
            if (seed + 1) % 10 == 0:
                print(f"  Seed {seed+1}/{num_seeds}...")
            
            result = run_single_experiment(seed=seed, h1_magnitude=h1_mag, h1_threshold=h1_threshold)
            mag_results.append(result)
            
            h1_measured_all.append(result['h1_measured']['train'])
            standard_traps.append(result['standard_policy']['mean_trap_visits'])
            hodge_traps.append(result['hodge_policy']['mean_trap_visits'])
        
        # Aggregate statistics
        aggregate = {
            'h1_target': h1_mag,
            'h1_measured': {
                'mean': float(np.mean(h1_measured_all)),
                'std': float(np.std(h1_measured_all))
            },
            'standard_trap_visits': compute_statistics(standard_traps, 'trap_visits').to_dict(),
            'hodge_trap_visits': compute_statistics(hodge_traps, 'trap_visits').to_dict(),
            'comparison': compare_methods(standard_traps, hodge_traps, 'Standard', 'Hodge'),
            'per_seed': mag_results
        }
        
        all_results.append(aggregate)
        
        # Print summary
        print(f"  H¹ measured: {aggregate['h1_measured']['mean']:.3f} ± {aggregate['h1_measured']['std']:.3f}")
        print(f"  Standard trap visits: {aggregate['standard_trap_visits']['mean']:.2f} ± {aggregate['standard_trap_visits']['std']:.2f}")
        print(f"  Hodge trap visits: {aggregate['hodge_trap_visits']['mean']:.2f} ± {aggregate['hodge_trap_visits']['std']:.2f}")
        print(f"  Effect size: {aggregate['comparison']['cohens_d']:.2f} ({aggregate['comparison']['effect_size']})")
    
    # Compute key correlations
    all_h1 = [r['h1_measured']['train'] for agg in all_results for r in agg['per_seed']]
    all_standard_traps = [r['standard_policy']['mean_trap_visits'] for agg in all_results for r in agg['per_seed']]
    all_hodge_traps = [r['hodge_policy']['mean_trap_visits'] for agg in all_results for r in agg['per_seed']]
    
    h1_standard_corr = stats.pearsonr(all_h1, all_standard_traps)
    h1_hodge_corr = stats.pearsonr(all_h1, all_hodge_traps)
    
    # Final output
    final_output = {
        'experiment': 'H1_Exploitation_V2_Corrected',
        'fixes_applied': [
            'True H¹ injection via Hodge decomposition',
            'Train/test split (trap unseen during training)',
            'True Hodge filtering (not penalty term)'
        ],
        'num_seeds': num_seeds,
        'h1_magnitudes': h1_magnitudes,
        'correlations': {
            'h1_vs_standard_traps': {
                'pearson_r': h1_standard_corr[0],
                'p_value': h1_standard_corr[1]
            },
            'h1_vs_hodge_traps': {
                'pearson_r': h1_hodge_corr[0],
                'p_value': h1_hodge_corr[1]
            }
        },
        'results': all_results
    }
    
    # Save results
    with open(output_path / 'results.json', 'w') as f:
        json.dump(final_output, f, indent=2, default=float)
    
    print(f"\nResults saved to {output_path / 'results.json'}")
    
    # Generate plots
    _plot_results(final_output, output_path)
    
    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    print(f"H¹ ↔ Standard trap visits correlation: r={h1_standard_corr[0]:.3f}, p={h1_standard_corr[1]:.4f}")
    print(f"H¹ ↔ Hodge trap visits correlation: r={h1_hodge_corr[0]:.3f}, p={h1_hodge_corr[1]:.4f}")
    
    if h1_standard_corr[0] > 0.5 and h1_standard_corr[1] < 0.05:
        print("✓ H¹ ENABLES exploitation in standard RLHF (correlation significant)")
    if abs(h1_hodge_corr[0]) < 0.2:
        print("✓ Hodge filtering PREVENTS H¹-based exploitation (no correlation)")
    
    return final_output


def _plot_results(results: Dict, output_path: Path):
    """Generate visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    h1_targets = [r['h1_target'] for r in results['results']]
    h1_measured = [r['h1_measured']['mean'] for r in results['results']]
    standard_means = [r['standard_trap_visits']['mean'] for r in results['results']]
    standard_stds = [r['standard_trap_visits']['std'] for r in results['results']]
    hodge_means = [r['hodge_trap_visits']['mean'] for r in results['results']]
    hodge_stds = [r['hodge_trap_visits']['std'] for r in results['results']]
    
    # Plot 1: H¹ Target vs Measured
    ax1 = axes[0, 0]
    ax1.plot(h1_targets, h1_targets, 'k--', label='Perfect tracking')
    ax1.scatter(h1_targets, h1_measured, c='blue', s=100, zorder=5)
    ax1.set_xlabel('H¹ Target')
    ax1.set_ylabel('H¹ Measured')
    ax1.set_title('H¹ Injection Verification')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: H¹ vs Trap Visits
    ax2 = axes[0, 1]
    ax2.errorbar(h1_measured, standard_means, yerr=standard_stds,
                 marker='o', label='Standard RLHF', color='red', capsize=3)
    ax2.errorbar(h1_measured, hodge_means, yerr=hodge_stds,
                 marker='s', label='Hodge-Filtered', color='blue', capsize=3)
    ax2.set_xlabel('H¹ Magnitude')
    ax2.set_ylabel('Mean Trap Visits')
    ax2.set_title('H¹ → Exploitation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Trap visits prevented
    ax3 = axes[1, 0]
    prevented = [s - h for s, h in zip(standard_means, hodge_means)]
    colors = ['green' if p > 0 else 'red' for p in prevented]
    ax3.bar(range(len(h1_targets)), prevented, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(h1_targets)))
    ax3.set_xticklabels([f'{h:.1f}' for h in h1_targets])
    ax3.set_xlabel('H¹ Target')
    ax3.set_ylabel('Trap Visits Prevented')
    ax3.set_title('Hodge Filtering Benefit')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Scatter correlation
    ax4 = axes[1, 1]
    all_h1 = []
    all_standard = []
    all_hodge = []
    for agg in results['results']:
        for r in agg['per_seed']:
            all_h1.append(r['h1_measured']['train'])
            all_standard.append(r['standard_policy']['mean_trap_visits'])
            all_hodge.append(r['hodge_policy']['mean_trap_visits'])
    
    ax4.scatter(all_h1, all_standard, alpha=0.3, c='red', label='Standard', s=20)
    ax4.scatter(all_h1, all_hodge, alpha=0.3, c='blue', label='Hodge', s=20)
    
    # Fit lines
    z_standard = np.polyfit(all_h1, all_standard, 1)
    z_hodge = np.polyfit(all_h1, all_hodge, 1)
    x_line = np.linspace(min(all_h1), max(all_h1), 100)
    ax4.plot(x_line, np.polyval(z_standard, x_line), 'r-', linewidth=2,
             label=f'Standard (r={results["correlations"]["h1_vs_standard_traps"]["pearson_r"]:.2f})')
    ax4.plot(x_line, np.polyval(z_hodge, x_line), 'b-', linewidth=2,
             label=f'Hodge (r={results["correlations"]["h1_vs_hodge_traps"]["pearson_r"]:.2f})')
    
    ax4.set_xlabel('H¹ Magnitude (per seed)')
    ax4.set_ylabel('Trap Visits')
    ax4.set_title('Per-Seed Correlation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'plots.png', dpi=150)
    print(f"Plots saved to {output_path / 'plots.png'}")


# ============================================================================
# 6. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="H¹ Exploitation Experiment V2 (Corrected)")
    parser.add_argument("--seeds", type=int, default=50, help="Number of seeds")
    parser.add_argument("--quick", action="store_true", help="Quick test with 5 seeds")
    parser.add_argument("--h1-threshold", type=float, default=0.0, 
                        help="H¹ threshold for partial filtering (0=remove all, 0.75=only remove if H¹>0.75)")
    args = parser.parse_args()
    
    num_seeds = 5 if args.quick else args.seeds
    
    print("="*60)
    print("H¹ EXPLOITATION EXPERIMENT V2 (CORRECTED)")
    print("="*60)
    print("\nFixes applied:")
    print("  1. True H¹ injection via Hodge decomposition")
    print("  2. Train/test split (trap unseen during training)")
    print("  3. True Hodge filtering (with threshold-based partial removal)")
    print(f"\nH¹ threshold: {args.h1_threshold} (0=remove all, higher=partial)")
    print(f"Running with {num_seeds} seeds...")
    
    results = run_full_experiment(
        num_seeds=num_seeds,
        h1_magnitudes=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        h1_threshold=args.h1_threshold,
        output_dir="../../results/h1_exploitation_v2"
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
