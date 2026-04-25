# Handoff: SGPO Refinement and Parameter Search

## Status: � Fixes Implemented - Ready for Re-run

**Date**: 2026-02-22 (Updated)  
**Experiment**: `sandbagging_experiment_v2.py`  
**Results**: `results_from_modal/sandbagging_v2/results.json`

---

## Changes Implemented (v2.1)

### Quick Fixes Applied
1. **✅ Warmup Period** — Metric doesn't update for first 30 episodes (`config.warmup_episodes`)
2. **✅ Softer Scaling** — `1/(1 + log(1+g))` instead of `1/sqrt(g)` (`config.use_soft_scaling`)
3. **✅ Metric Regularization** — Prevents collapse with `metric_reg_weight * (g.mean() - 1)²`
4. **✅ Hybrid Lagrangian+Geometric** — Combines CPO's constraint with SGPO's metric (`config.use_hybrid_lagrangian`)

### New Components Added
- **`AnisotropicRiemannianMetric`** — Directional metric that only penalizes movement TOWARD danger (escape is free)
- **`train_sgpo_anisotropic()`** — Training function for anisotropic SGPO
- **`DiagnosticResult`** + `train_sgpo_with_diagnostics()` — Extensive logging for debugging
- **`plot_metric_field()`** — Visualize learned metric over state space
- **`run_diagnostic_experiment()`** — Full diagnostic run with summary plots
- **`GeneralizationEnvConfig`** — Train/test split configuration
- **`run_generalization_experiment()`** — Train on trap A, test on trap B (UNSEEN)

### New CLI Options
```bash
# Full experiment with 4 methods (PPO, CPO, SGPO, SGPO_ANIS)
python sandbagging_experiment_v2.py --seeds 50

# Quick test
python sandbagging_experiment_v2.py --quick

# Diagnostic mode (metric field visualization, learning curves)
python sandbagging_experiment_v2.py --mode diagnostics --seeds 5

# Generalization test (train on trap A, test on trap B)
python sandbagging_experiment_v2.py --mode generalization --seeds 20
```

---

## Executive Summary

SGPO (Sheaf-Geodesic Policy Optimization) shows **worse** safety performance than baselines:

| Method | Total Violations | Final Return |
|--------|-----------------|--------------|
| PPO | 30.1 ± 22.3 | 0.50 ± 7.51 |
| CPO | **23.8 ± 14.1** | **2.08 ± 5.69** |
| SGPO | 46.2 ± 32.0 ❌ | 0.63 ± 5.65 |

**This is the opposite of expected.** SGPO should have the fewest violations.

---

## Issue 1: SGPO Has Highest Violations

### Evidence

From the results:
- **SGPO**: 46.2 ± 32.0 violations (worst)
- **PPO**: 30.1 ± 22.3 violations (middle)
- **CPO**: 23.8 ± 14.1 violations (best)

CPO's Lagrangian constraint optimization outperforms SGPO's geometric approach.

### Potential Causes

1. **Metric Learning Failure**: The `LearnedRiemannianMetric` may not be learning the correct danger field
2. **Metric Target Signal**: We're using `1/dist_to_trap` as target, but we don't know the trap location
3. **Advantage Scaling Issue**: Dividing by `√g` may be too aggressive or not aggressive enough
4. **Warmup Period**: No warmup - metric and policy train simultaneously from episode 1

---

## Issue 2: Same Train/Test Problem as H¹ Experiment

The sandbagging environment has the same issue:
- Trap location is **fixed and known** at (5.0, 6.0)
- Policy trains and evaluates on the **same** trap
- The metric is learning to avoid a **known** obstacle, not generalizing

### Correct Design

```
TRAINING PHASE:
- Train on environment with trap at (5.0, 6.0)
- Learn danger metric from cost signals

TESTING PHASE (UNSEEN):
- Evaluate on environment with trap at DIFFERENT location, e.g. (7.0, 3.0)
- Does the learned metric generalize?
```

---

## Issue 3: Metric Learning Approach

### Current Implementation

```python
class LearnedRiemannianMetric(nn.Module):
    def __init__(self, ...):
        self.danger_net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Softplus()
        )
```

**Problem**: The metric is learned from scratch with only 2D position input. It has no way to identify danger except by memorizing locations.

### Better Approaches

**Option A: Feature-based Metric**
```python
# Input features that generalize:
# - Local reward gradient
# - Recent cost history
# - Distance from known safe regions
def compute_danger_features(state, trajectory_history):
    features = []
    features.append(compute_local_reward_gradient(state))
    features.append(recent_cost_rate(trajectory_history[-10:]))
    features.append(distance_from_start(state))
    return np.array(features)
```

**Option B: Use Existing GPO Variants**

The `high_dimensional_reward_spaces` folder has more sophisticated implementations:
- `AnisotropicSGPOCritic` - directional singularities
- `EscapeSGPO` - preserves escape routes
- Full `GeoDPO` with Hodge decomposition

### Files to Review

```
topics/high_dimensional_reward_spaces/notebooks/modal_runner/
├── anisotropic_escape_experiment.py    # Better metric design
├── geodesic_distance_experiment.py     # Alternative formulation
├── hyperplane_barrier_experiment.py    # Barrier function approach
└── geodpo_experiments.py               # Full implementation (273KB!)
```

---

## Issue 4: Ablation Results Are Inconclusive

### Sharpness Ablation (β)

| β | Violations |
|---|------------|
| 0.5 | 37.3 ± 26.8 |
| 1.0 | 41.7 ± 28.4 |
| 1.5 | 29.8 ± 31.6 |
| 2.0 | 46.0 ± 42.6 |
| 2.5 | 35.5 ± 24.5 |
| 3.0 | 35.1 ± 29.5 |

**No clear trend.** The theoretical prediction (β=2 is optimal) is not supported.

### Severity Ablation (C)

| C | Violations |
|---|------------|
| 1.0 | 36.4 ± 33.2 |
| 2.5 | 33.6 ± 19.4 |
| 5.0 | 46.0 ± 42.6 |
| 10.0 | 53.5 ± 47.1 |
| 20.0 | 49.7 ± 38.3 |

**Higher severity → MORE violations.** This is backwards! Higher metric penalty should mean fewer violations.

**Hypothesis**: The metric penalty is interfering with learning, not helping.

---

## Recommended Investigation Plan

### Phase 1: Diagnose Metric Learning

```python
# Add diagnostic logging
def train_sgpo_with_diagnostics(env, config, seed):
    # ... training loop ...
    
    # Log these every episode:
    diagnostics = {
        "metric_at_trap": metric(trap_center).item(),
        "metric_at_goal": metric(goal).item(),
        "metric_at_start": metric(start).item(),
        "metric_loss": loss_metric.item(),
        "danger_gradient_norm": compute_gradient_norm(metric, states),
    }
    
    # Visualize metric field
    if ep % 50 == 0:
        plot_metric_field(metric, env)
```

**Question to answer**: Is the metric actually learning to be high near the trap?

### Phase 2: Try Existing GPO Variants

```python
# Import from existing codebase
from high_dimensional_reward_spaces.notebooks.modal_runner.anisotropic_escape_experiment import (
    AnisotropicSGPOCritic
)

# This critic has:
# - Direction-aware metric (only penalizes TOWARD danger)
# - Black hole detection
# - Escape route preservation
```

### Phase 3: Proper Train/Test Split

```python
@dataclass
class EnvConfig:
    # Training trap
    train_trap_center: Tuple[float, float] = (5.0, 6.0)
    
    # Testing trap (DIFFERENT LOCATION)
    test_trap_center: Tuple[float, float] = (7.0, 3.0)
    
    # Or: multiple traps, train on subset
    all_traps: List[Tuple[float, float]] = [
        (3.0, 4.0), (5.0, 6.0), (7.0, 3.0), (8.0, 8.0)
    ]
    train_traps: List[int] = [0, 1]  # indices
    test_traps: List[int] = [2, 3]   # UNSEEN
```

### Phase 4: Hyperparameter Search (After Fixing Above)

| Parameter | Search Range | Notes |
|-----------|--------------|-------|
| `lr_metric` | 1e-4 to 1e-2 | Currently 3e-3 |
| `metric_hidden_dim` | 16, 32, 64, 128 | Currently 32 |
| `warmup_episodes` | 0, 10, 30, 50, 100 | No warmup currently |
| `metric_update_freq` | 1, 5, 10 | Every episode currently |
| `advantage_scaling` | 1/√g, 1/g, exp(-g) | Currently 1/√g |

---

## What Success Looks Like

After fixes:

1. **SGPO < CPO < PPO** in violations (on training environment)
2. **SGPO generalizes** to unseen traps (CPO/PPO do not)
3. **Metric field visualization** shows high values near traps
4. **Ablations show clear trends** (e.g., higher β → fewer violations up to a point)

---

## Code Locations

| File | Purpose |
|------|---------|
| `feedback_geometry/src/sandbagging_experiment_v2.py` | Current implementation |
| `high_dimensional_reward_spaces/src/safety_experiment.py` | Original SGPO |
| `high_dimensional_reward_spaces/src/safety_experiment_hard.py` | Multi-trap version |
| `high_dimensional_reward_spaces/notebooks/modal_runner/*.py` | Advanced variants |

---

## Quick Fixes to Try First

1. **Add Warmup**: Don't update metric for first 30 episodes
   ```python
   if ep > config.warmup_episodes:
       opt_metric.step()
   ```

2. **Softer Advantage Scaling**: Use log instead of sqrt
   ```python
   riemannian_adv = adv / (1 + torch.log(1 + g_values))
   ```

3. **Metric Regularization**: Prevent metric from collapsing
   ```python
   metric_reg = 0.1 * (g_predicted.mean() - 1.0) ** 2
   loss_metric = mse + metric_reg
   ```

4. **Use CPO's Lagrangian AS WELL AS metric**:
   ```python
   # Hybrid: Lagrangian constraint + geometric scaling
   combined_adv = (r_adv - lambda * c_adv) / sqrt(g)
   ```

---

## References

- Existing experiments: `topics/high_dimensional_reward_spaces/notebooks/modal_runner/`
- Theory: `topics/high_dimensional_reward_spaces/docs/RESEARCH_PROPOSAL.md`
- Formal guarantees: Memory `e7dac5bb-b537-43f5-80e6-38b1e8d5a4fd`
