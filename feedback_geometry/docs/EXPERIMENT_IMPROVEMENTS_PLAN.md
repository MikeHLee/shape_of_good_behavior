# Experiment Improvements Plan: Aligning with Correct Mathematical Rigor

**Date**: February 24, 2026  
**Context**: Revising experiments after correcting categorical conflations per `Hodge Theory, Bilattices, and Social Choice.pdf`

---

## Current Experiment Status Summary

| Experiment | Domain | Key Result | Status |
|------------|--------|------------|--------|
| **A: Pre-filtered RM** | HH-RLHF (real text, 384-dim) | **80% exploitation reduction, 30% accuracy boost** | ✅ Excellent |
| **C: SGPO_ANIS_CCHC** | Synthetic 2D navigation | Zero violations (env issue) | ⚠️ Needs env tuning |
| **Sandbagging v2** | Synthetic 2D navigation | SGPO_ANIS shows 0.5% goal rate (only method) | 🔶 Promising, needs tuning |

---

## Critical Mathematical Corrections to Apply

### Issue 1: H¹ Interpretation

**Previous (Incorrect)**:
- Treating "conditional H¹" as the key metric
- Using H¹ ≠ 0 as a threshold-based filter

**Corrected (Per PDF)**:
- **Reliability Score** = ||gradient||² / ||total||² is the correct metric
- H¹ dimension (Betti number β₁) counts independent cycles, doesn't measure magnitude
- **Energy distribution** across gradient/curl/harmonic is what matters

**Impact on Experiment A**:
```python
# OLD: Binary threshold on conditional H¹
if conditional_h1 > 0.8:
    mark_as_invalid(context)

# NEW: Use reliability score (gradient energy ratio)
components = hodge_decompose(preference_graph)
reliability = components.gradient_energy / components.total_energy
if reliability < threshold:  # e.g., < 0.5
    mark_as_unreliable(context)
```

### Issue 2: Module Conflation

**Previous (Incorrect)**:
- SGPO_ANIS uses learned `danger_net` (continuous) + Hodge decomposition (discrete) in same pipeline
- Soft potential penalties (can be overcome)
- No separation between reward filtering and safety enforcement

**Corrected (Per Handoff)**:

| Module | Domain | Purpose | Implementation |
|--------|--------|---------|----------------|
| **Module 1: Discrete HodgeRank** | Preference graph (discrete) | Extract transitive preferences for RM training | `discrete_hodge_rank.py` |
| **Module 2: Conformal Safety** | Latent embedding (continuous) | Geometric barriers via σ(x)→∞ | `conformal_safety.py` |

**Impact on Experiments**:
- Experiment A should use **Module 1** (DiscreteHodgeRank) for preference filtering
- Experiment C/Sandbagging should use **Module 2** (ConformalSafetyMetric) for trap avoidance

### Issue 3: Soft Penalties vs Geometric Barriers

**Previous (Incorrect)**:
```python
# Learned danger metric with soft penalty
metric_factor = base + severity * (danger ** sharpness)
# This can still be overcome with sufficient reward
```

**Corrected (Per Conformal Safety)**:
```python
# Conformal metric where σ(x) → ∞ at danger boundary
def conformal_factor(x):
    d = distance_to_danger_boundary(x)
    if d <= 0:
        return float('inf')  # Inside danger = infinite barrier
    return -sharpness * np.log(d)  # Diverges as d → 0

# Geodesic distance to danger is INFINITE
# Cannot be overcome by any finite reward
```

---

## Experiment-Specific Improvements

### Experiment A: Pre-filtered Reward Models ✅

**Current Results** (already excellent):
- Raw: 60.4% accuracy, 31.8% exploitation
- Filtered: 91.9% accuracy, 6.5% exploitation
- **79.6% exploitation reduction**

**Improvements to Apply**:

1. **Use proper reliability score instead of binary H¹ threshold**
```python
from discrete_hodge_rank import DiscreteHodgeRank, PreferenceGraph

hodge = DiscreteHodgeRank()
for context_id, context_prefs in grouped_preferences.items():
    graph = PreferenceGraph.from_pairwise_comparisons(
        n_items=len(unique_items),
        comparisons=context_prefs
    )
    components = hodge.decompose(graph)
    
    # Use reliability score instead of binary H¹
    if components.reliability_score < 0.5:
        mark_unreliable(context_id)
```

2. **Report gradient/curl/harmonic energy breakdown**
```python
# Add to results.json
{
    "gradient_energy_mean": ...,
    "curl_energy_mean": ...,
    "harmonic_energy_mean": ...,
    "reliability_score_mean": ...
}
```

3. **Correlation analysis**: Plot exploitation rate vs reliability score (not binary H¹)

**Priority**: Medium (already working well, these are refinements)

---

### Experiment C: SGPO_ANIS_CCHC ⚠️

**Current Issue**: Zero violations across all 50 seeds — trap dynamics not working

**Root Causes**:
1. Trap may be unreachable from start (0,0) given dynamics
2. Progress reward dominates trap reward (+3.0 too weak)
3. Embedding space navigation doesn't align with Euclidean trap

**Improvements to Apply**:

1. **Environment Tuning**:
```python
# Option A: Make trap more attractive
trap_reward = 10.0  # Increase from 3.0

# Option B: Place trap on path to goal
trap_center = (4.0, 4.0)  # On diagonal from (0,0) to (10,10)

# Option C: Longer episodes to explore
max_steps = 120  # Increase from 60
```

2. **Replace soft metric with conformal safety**:
```python
from conformal_safety import ConformalSafetyMetric, ConformalPolicyOptimizer

# Define danger region geometrically
metric = ConformalSafetyMetric()
metric.add_danger_region(
    center=np.array([5.0, 6.0]),
    radius=2.5,
    sharpness=2.0
)

# Policy optimizer uses conformal natural gradient
optimizer = ConformalPolicyOptimizer(metric)
safe_grads, metadata = optimizer.compute_safe_update(states, vanilla_grads)
```

3. **Separate Module 1 from Module 2**:
- Module 1 (HodgeRank): Used during reward model training (not policy optimization)
- Module 2 (Conformal): Used during policy optimization (not reward training)

**Priority**: High (need to fix environment first, then apply conformal safety)

---

### Sandbagging v2: SGPO_ANIS 🔶

**Current Results** (SGPO_ANIS is promising):
| Method | Violations | Goal Rate |
|--------|-----------|-----------|
| PPO | 29.3 ± 22.1 | 0.0% |
| CPO | 22.8 ± 19.6 | 0.0% |
| SGPO | 22.6 ± 15.8 | 0.0% |
| **SGPO_ANIS** | **22.0 ± 16.2** | **0.5%** |

**Why SGPO_ANIS is Promising**:
- Only method achieving any goal success
- Lowest violations
- Directional metric (only penalize movement toward danger) is conceptually sound

**Improvements to Apply**:

1. **Replace AnisotropicRiemannianMetric with ConformalSafetyMetric**:

```python
# CURRENT: Learned danger_net with soft penalty
class AnisotropicRiemannianMetric(nn.Module):
    def forward(self, x, v=None):
        danger = self.danger_net(x)
        # Soft penalty that can be overcome
        metric_factor = base + severity * danger

# IMPROVED: Conformal metric with infinite barrier
class ConformalSGPO_ANIS:
    def __init__(self, danger_regions):
        self.metric = ConformalSafetyMetric()
        for center, radius in danger_regions:
            self.metric.add_danger_region(center, radius)
    
    def compute_advantage_scaling(self, states, velocities):
        scales = []
        for state, vel in zip(states, velocities):
            sigma = self.metric.conformal_factor(state)
            if np.isinf(sigma):
                scales.append(0.0)  # No update in danger
            else:
                # Anisotropic: check direction
                grad_sigma = self.metric.conformal_factor_gradient(state)
                toward_danger = np.dot(vel, -grad_sigma)
                
                if toward_danger > 0:  # Moving toward danger
                    scales.append(np.exp(-2 * sigma))  # Strong suppression
                else:  # Escaping
                    scales.append(1.0)  # Normal update
        return np.array(scales)
```

2. **Integrate with Hodge-filtered advantage**:

The context-conditioned Hodge component (CCHC) should be used differently:
- **During training**: Filter preference data using Module 1 (DiscreteHodgeRank)
- **During policy optimization**: Use Module 2 (ConformalSafety) for advantage scaling

```python
class SGPO_ANIS_Corrected:
    def __init__(self, conformal_metric, hodge_filtered_reward_model):
        self.metric = conformal_metric  # Module 2
        self.reward_model = hodge_filtered_reward_model  # Trained on Module 1 output
    
    def compute_advantage(self, states, actions):
        # Reward from Hodge-filtered model (Module 1 applied at training time)
        rewards = self.reward_model(states, actions)
        
        # Advantage scaling from conformal safety (Module 2 applied at inference)
        safety_scales = self.metric.compute_natural_gradient_scale(states)
        
        # Combined: clean reward signal + geometric safety
        return rewards * safety_scales
```

3. **Ablation: Compare conformal vs soft metrics**:
```python
ablation_configs = {
    "soft_isotropic": LearnedRiemannianMetric(),  # Current baseline
    "soft_anisotropic": AnisotropicRiemannianMetric(),  # Current SGPO_ANIS
    "conformal_isotropic": ConformalSafetyMetric(),  # New Module 2
    "conformal_anisotropic": ConformalANIS(),  # Recommended
}
```

**Priority**: High (SGPO_ANIS shows promise, apply conformal safety to improve)

---

## Proposed New Experiment: Full Pipeline Validation

### Experiment D: End-to-End Modular Safe RLHF

**Purpose**: Validate the full 2-module pipeline on realistic data

**Setup**:
1. **Module 1 (Training)**: Use DiscreteHodgeRank on HH-RLHF to train reward model
2. **Module 2 (Optimization)**: Use ConformalSafetyMetric for policy training
3. **Evaluation**: Test on held-out preferences with known dangerous regions

**Domain Options**:

| Option | Description | Complexity | Realism |
|--------|-------------|------------|---------|
| A: Synthetic 2D | Current sandbagging env | Low | Low |
| B: Frozen Embeddings | HH-RLHF with BERT/GTE, 2D projection | Medium | Medium |
| C: Semantic MDP | Full semantic state machine with GPT-4 oracle | High | High |
| D: Safety Gym | SafetyAnt/SafetyCar with reward from HH-RLHF embedding | High | High |

**Recommendation**: Start with Option B (frozen embeddings), then scale to C/D

---

## Implementation Priority

### Phase 1: Fix Environment Issues (1-2 days)
- [ ] Tune Experiment C environment (trap placement, rewards)
- [ ] Verify violations are actually measurable

### Phase 2: Apply Conformal Safety (2-3 days)
- [ ] Replace soft metrics with ConformalSafetyMetric in SGPO_ANIS
- [ ] Run ablation comparing soft vs conformal

### Phase 3: Apply Correct HodgeRank (1-2 days)
- [ ] Update Experiment A to use reliability score instead of binary H¹
- [ ] Report energy breakdown (gradient/curl/harmonic)

### Phase 4: Full Pipeline (3-5 days)
- [ ] Implement Experiment D with modular separation
- [ ] Validate on frozen embeddings from HH-RLHF

---

## Key Metrics to Track (Updated)

| Metric | Old Definition | Corrected Definition |
|--------|---------------|---------------------|
| H¹ magnitude | `norm(harmonic_component)` | `harmonic_energy / total_energy` (fraction) |
| Reliability | Binary threshold on H¹ | `gradient_energy / total_energy` (0-1 continuous) |
| Safety violation | Count of steps in trap | Geodesic distance to danger boundary |
| Exploitation rate | P(reject > choose) < 0.3 | Same (this was correct) |

---

## Questions for Discussion

1. **Domain Priority**: Should we fix the synthetic 2D environment first, or pivot to frozen embeddings on HH-RLHF?

2. **Conformal vs Learned**: The conformal metric requires knowing danger regions a priori. Should we:
   - Use known trap locations (evaluation benchmark)
   - Learn danger regions from cost signals, then apply conformal metric
   - Hybrid: learn approximate regions, then apply conformal barrier

3. **CCHC Integration**: The Context-Conditional Hodge Critic separates marginal/conditional H¹. How does this map to the corrected framework?
   - Marginal H¹ ≈ Total cyclic energy (curl + harmonic)
   - Conditional H¹ ≈ Within-context cycles (should be zero if preferences are consistent)

4. **Scaling**: Experiment A used 5000 samples. Should we scale up for Experiment D?
