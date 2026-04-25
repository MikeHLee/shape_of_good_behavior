# Experimental Setup Analysis: Sandbagging v2.1

## Current Implementation Summary

### 1. Test Environments

**Domain Type**: Synthetic 2D navigation task (NOT real-world data)

**Environment: SandbaggingEnv**
- **State space**: 2D continuous positions (x, y) ∈ [0, 10] × [0, 10]
- **Action space**: 2D continuous velocities, clipped to [-1, 1]
- **Dynamics**: `state_{t+1} = state_t + dt * action_t` where dt=0.5

**Key Locations**:
- Start: (0, 0)
- Goal: (10, 10)
- Trap center: (5, 6) with radius 2.5
- Event horizon: 2.0 (catastrophic boundary)

**Reward Structure**:
```
reward = progress_reward + trap_reward + goal_bonus + catastrophe_penalty
  where:
    progress_reward = distance_to_goal_before - distance_to_goal_after
    trap_reward = +3.0 if in_trap else 0.0  [DECEPTIVE]
    goal_bonus = +10.0 if dist_to_goal < 1.0
    catastrophe_penalty = -100.0 if in_event_horizon
```

**Cost Signal**:
```
cost = 1.0 if in_trap else 0.0
```

### 2. Metrics Definition

**Violations** (per episode):
- Count of steps where agent is inside trap region
- Range: 0 to max_steps (60)
- Aggregation: sum across all steps in episode

**Return** (per episode):
- Cumulative discounted reward: G_t = Σ γ^k * r_{t+k}
- Includes trap reward, progress reward, goal bonus, catastrophe penalty
- Aggregation: mean of final 50 episodes (last 1/6 of training)

**Goal Rate** (per seed):
- Fraction of episodes where agent reaches goal (dist_to_goal < 1.0)
- Range: 0.0 to 1.0
- Aggregation: mean across all episodes

### 3. Current Results (50 seeds, 300 episodes)

| Method | Violations | Return | Goal Rate |
|--------|-----------|--------|-----------|
| PPO | 29.3 ± 22.1 | 0.29 ± 5.28 | 0.0% |
| CPO | 22.8 ± 19.6 | 1.04 ± 6.68 | 0.0% |
| SGPO | 22.6 ± 15.8 | 1.49 ± 6.93 | 0.0% |
| **SGPO_ANIS** | **22.0 ± 16.2** | **0.77 ± 6.13** | **0.5%** |

**Key Observation**: SGPO_ANIS is the only method achieving any goal success (0.5%), suggesting directional metric learning has merit.

---

## Critical Gaps vs. Real-World Alignment

### What This Domain IS:
✓ A clear mathematical formalization of "sandbagging" (deceptive reward)
✓ A test of whether learned metrics can detect danger
✓ A controlled environment for ablation studies
✓ Synthetic but interpretable

### What This Domain IS NOT:
✗ **Real text data**: No HH-RLHF, no semantic MDPs, no actual language model behavior
✗ **High-dimensional**: Only 2D state space (vs. 768-4096D embeddings in real RLHF)
✗ **Multimodal**: Single modality (position), no cross-modal inconsistency
✗ **Safety gym**: No SafetyAnt, SafetyCar, or other complex dynamics
✗ **Semantic**: No learned representations, no ontological structure

### Implications for SGPO_ANIS Tuning:
1. **Promising signal**: The anisotropic metric's directional penalization works in principle
2. **Limited scope**: Success here doesn't guarantee transfer to high-dim RLHF
3. **Next steps**: Should validate on:
   - Higher-dimensional state spaces (e.g., frozen embeddings)
   - Real preference data (HH-RLHF or similar)
   - Safety gym benchmarks (SafetyAnt, SafetyCar)

---

## SGPO_ANIS Architecture (Current)

```python
class AnisotropicRiemannianMetric(nn.Module):
    """
    Directional metric: only penalizes movement TOWARD danger.
    
    Key insight: escape routes should remain low-cost.
    """
    
    def forward(self, x, v=None):
        # Estimate danger center from state
        danger_center = self.center_net(x)
        
        # Compute direction toward danger
        to_danger = danger_center - x
        n_hat = to_danger / ||to_danger||
        
        # Project velocity onto danger direction
        v_toward = v · n_hat
        
        # Only penalize approaching movement
        toward_ratio = max(0, v_toward)^2 / ||v||^2
        
        # Metric scales with danger level × approach ratio
        g = base + danger_level * toward_ratio
        
        # Escape factor: 1.0 if escaping, 0.0 if approaching
        escape_factor = sigmoid(-v_toward * 5.0)
        
        return g, escape_factor
```

**Current Limitations**:
- Danger center is learned but not grounded in actual trap location
- No explicit Hodge decomposition
- No context-conditioned scaling

---

## Proposed Enhancement: Context-Conditioned Hodge Components

Your idea to add context-conditioned Hodge components is excellent. Here's the conceptual framework:

### Current SGPO_ANIS:
```
advantage_scaling = escape_factor + (1 - escape_factor) / sqrt(g)
```

### Proposed with Hodge:
```
# Decompose learned metric into:
# g = g_exact + g_closed  (Hodge decomposition)
#   where g_exact = ∇φ (gradient of potential)
#         g_closed = ∇×ψ (curl/cyclic component)

# Context-conditioned weighting:
λ_context = f_context(state, trajectory_history)

# Modified scaling:
advantage_scaling = escape_factor + (1 - escape_factor) / sqrt(λ_context * g_exact + (1-λ_context) * g_closed)
```

**Benefits**:
1. Explicitly separates "true danger" (exact) from "deceptive cycles" (closed)
2. Context-conditioning allows adaptation to different trap configurations
3. Aligns with sheaf-theoretic framework (H¹ = cyclic components)

---

## Recommendations Before Tuning SGPO_ANIS

### Phase 1: Clarify Domain Scope
- [ ] Decide: Are we optimizing for synthetic sandbagging, or preparing for real RLHF?
- [ ] If synthetic: Focus on ablations (metric architecture, danger center learning)
- [ ] If real RLHF: Should validate on frozen embeddings + HH-RLHF data first

### Phase 2: Implement Hodge Decomposition
- [ ] Add explicit H¹ detection in metric learning
- [ ] Implement context-conditioned weighting
- [ ] Test on current sandbagging domain

### Phase 3: Validate on Realistic Domains
- [ ] Port to frozen embedding space (BERT/GTE on HH-RLHF)
- [ ] Test on Safety Gym (SafetyAnt, SafetyCar)
- [ ] Measure transfer to unseen trap configurations

### Phase 4: Tune SGPO_ANIS
- [ ] Ablate danger center learning (fixed vs. learned)
- [ ] Ablate escape factor scaling (sigmoid slope, threshold)
- [ ] Ablate Hodge weighting (λ_context schedule)

---

## Questions for You

1. **Domain Priority**: Should we focus on perfecting the synthetic sandbagging domain, or pivot to real-world data (HH-RLHF + frozen embeddings)?

2. **Hodge Implementation**: Do you want to implement Hodge decomposition before tuning, or tune SGPO_ANIS first and add Hodge later?

3. **Generalization Testing**: The current generalization experiment (train trap A, test trap B) shows poor transfer. Should we:
   - Improve metric learning to generalize better?
   - Or accept this as a limitation of the synthetic domain?

4. **Baseline Comparison**: Should we compare SGPO_ANIS against:
   - Constrained RL baselines (CPO, PCPO)?
   - Reward model uncertainty methods?
   - Hodge-filtered training (from H¹ exploitation paper)?
