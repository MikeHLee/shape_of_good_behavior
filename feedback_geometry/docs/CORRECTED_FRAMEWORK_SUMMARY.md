# Corrected Framework Summary: Modular Safe RLHF

**Date**: February 24, 2026  
**Status**: Implementation complete, ready for testing

---

## Overview

This document summarizes the mathematical corrections and implementation changes made to the Safe RLHF framework based on the reference document "Hodge Theory, Bilattices, and Social Choice.pdf".

---

## New Files Created

| File | Purpose |
|------|---------|
| `src/conformal_sgpo.py` | Corrected SGPO variants with conformal safety metrics |
| `src/corrected_experiments.py` | Experiment A (3 variants) + Experiment C (fixed env) |

---

## Mathematical Corrections Applied

### 1. Hodge Decomposition Interpretation

| Component | OLD Interpretation | CORRECTED Interpretation |
|-----------|-------------------|--------------------------|
| **Gradient (∇φ)** | Part of "clean" signal | **USE FOR TRAINING** - transitive consensus (Borda count) |
| **Curl (δψ)** | Often ignored | **DISCARD** - local cyclic inconsistencies in 3-cliques |
| **Harmonic (h)** | "Global structure" to preserve | **DISCARD** - global Condorcet paradoxes |

### 2. Reliability Score (NEW)

```
reliability = ||gradient||² / ||total||²

where total = gradient + curl + harmonic
```

- **High (→1)**: Preferences nearly transitive, ranking trustworthy
- **Low (→0)**: Preferences cyclic chaos, ranking unreliable

This replaces the old binary H¹ threshold approach.

### 3. Conformal Safety Metric

**OLD (Soft Penalty)**:
```python
g = base + severity * danger^sharpness  # Can be overcome!
```

**NEW (Infinite Barrier)**:
```python
σ(x) = -β * log(distance_to_danger)
g(x) = e^{2σ(x)}

# As distance → 0: σ → ∞, g → ∞
# Geodesic distance to danger = INFINITE
```

### 4. Module Separation

| Module | Domain | Purpose | File |
|--------|--------|---------|------|
| **Module 1** | Discrete (preference graph) | Extract transitive preferences | `DiscreteHodgeRank` |
| **Module 2** | Continuous (latent space) | Geometric safety barriers | `ConformalSafetyMetric` |
| **Module 3** | Diagnostics (future) | Runtime anomaly detection | TBD |

---

## Experiment A: Three Filtering Variants

### Methods

1. **`harmonic_only`** (original baseline)
   - Filter contexts where harmonic ratio > threshold
   - Captures global Condorcet cycles

2. **`curl_only`** (new)
   - Filter contexts where curl ratio > threshold
   - Captures local cycles in 3-cliques

3. **`reliability_score`** (new, recommended)
   - Accept only contexts where reliability ≥ threshold
   - Uses gradient²/(gradient²+curl²+harmonic²)
   - Most mathematically principled

### Expected Results

| Method | What It Filters | Expected Exploitation Reduction |
|--------|----------------|--------------------------------|
| harmonic_only | Global cycles | Moderate (original ~80%) |
| curl_only | Local cycles | Should be significant |
| reliability_score | All cyclic content | Should be highest |

---

## Experiment C: Fixed Environment

### Environment Fixes

| Parameter | OLD | NEW | Rationale |
|-----------|-----|-----|-----------|
| Trap center | (5, 6) | (4, 4) | **On diagonal path to goal** |
| Trap reward | 3.0 | 5.0 | Higher temptation |
| Max steps | 60 | 100 | More exploration time |
| Event horizon | 2.0 | 1.5 | Smaller catastrophic core |

### Methods Compared

1. **PPO** - Baseline (no safety)
2. **CPO** - Lagrangian constraint (expectation-based)
3. **Conformal SGPO** - Isotropic conformal barriers
4. **Conformal SGPO ANIS** - Anisotropic (escape routes free)
5. **Conformal SGPO ANIS CCHC** - With Hodge reliability weighting

### Key Innovation: Hybrid Danger Learning

```python
# Phase 1: LEARN danger regions from behavioral telemetry (costs)
metric.update_learned_regions(states, costs)

# Phase 2: HARDEN into conformal barriers when confident
if region.confidence >= threshold:
    hardened_regions.append((center, radius))

# Phase 3: Apply CONFORMAL metric
σ = -β * log(distance_to_hardened_boundary)
g = e^{2σ}  # Infinite at boundary
```

---

## SGPO Variants Comparison

```
┌─────────────────────┬──────────────────────────┬──────────────────────────────┐
│ Variant             │ Metric                   │ Key Feature                  │
├─────────────────────┼──────────────────────────┼──────────────────────────────┤
│ Original SGPO       │ g = base + severity*d^β  │ Soft penalty (overcome-able) │
│ Conformal SGPO      │ g = e^{-2β·log(d)}       │ Infinite barrier             │
│ Conformal SGPO ANIS │ + anisotropic scaling    │ Escape routes preserved      │
│ Conformal SGPO CCHC │ + reliability weighting  │ Hodge-filtered advantages    │
└─────────────────────┴──────────────────────────┴──────────────────────────────┘
```

---

## Advantage Scaling Formula

**OLD**:
```python
adv_scaled = adv / sqrt(g)  # or / log(1+g)
```

**NEW (Conformal Natural Gradient)**:
```python
# Natural gradient preconditioned by metric inverse
scale = exp(-2 * σ)  # = 1/g

# For anisotropic variant:
scale = escape_factor + (1 - escape_factor) * exp(-2 * σ)

adv_scaled = scale * adv
```

This naturally suppresses updates near danger (high σ) while preserving escape learning signal.

---

## Next Steps

### Phase 1: Validate on Synthetic 2D (Current)
- [ ] Run corrected Experiment C with all methods
- [ ] Verify trap is actually reachable and tempting
- [ ] Compare violation rates across methods

### Phase 2: Validate Experiment A Variants
- [ ] Run three filtering variants on HH-RLHF
- [ ] Compare exploitation rates
- [ ] Verify reliability_score outperforms binary threshold

### Phase 3: Scale to RLHF Embeddings
- [ ] Use frozen embeddings (BERT/GTE) on HH-RLHF
- [ ] Navigate in high-dimensional embedding space
- [ ] Test transfer of learned danger regions

### Phase 4: Publication
- [ ] Update paper with corrected mathematics
- [ ] Generate publication-ready figures
- [ ] Statistical analysis (50+ seeds, 95% CI)

---

## Quick Test Commands

```bash
cd /Users/Michaellee/Documents/Runes/ai_research/topics/feedback_geometry

# Test conformal metric
python -c "
from src.conformal_sgpo import ConformalSafetyMetric
import numpy as np
import torch

metric = ConformalSafetyMetric(state_dim=2, sharpness=2.0)
metric.add_known_danger_region(np.array([4.0, 4.0]), 2.0)

safe = torch.tensor([[0.0, 0.0]])
near = torch.tensor([[3.0, 3.0]])
danger = torch.tensor([[4.0, 4.0]])

print(f'Safe σ: {metric.conformal_factor(safe).item():.2f}')
print(f'Near σ: {metric.conformal_factor(near).item():.2f}')
print(f'Danger σ: {metric.conformal_factor(danger).item():.2f}')
"

# Test Hodge decomposition
python -c "
from src.corrected_experiments import DiscreteHodgeRank

hodge = DiscreteHodgeRank()

# Cyclic preferences: A > B > C > A
comps = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
result = hodge.decompose(3, comps)

print(f'Reliability: {result.reliability_score:.3f}')
print(f'Curl ratio: {result.curl_ratio:.3f}')
print(f'Harmonic ratio: {result.harmonic_ratio:.3f}')
"

# Test fixed environment
python -c "
from src.corrected_experiments import FixedSandbaggingEnv
import numpy as np

env = FixedSandbaggingEnv()
obs = env.reset()

# Move toward trap (on diagonal)
for _ in range(10):
    obs, r, cost, done, info = env.step(np.array([0.5, 0.5]))

print(f'Position: {obs}')
print(f'In trap: {info[\"in_trap\"]}')
print(f'Cost: {cost}')
"
```

---

## References

1. **Hodge Theory, Bilattices, and Social Choice.pdf** - Mathematical foundation
2. **handoffs/14_MATHEMATICAL_RESTRUCTURING.md** - Module separation guide
3. **high_dimensional_reward_spaces/src/discrete_hodge_rank.py** - Module 1 reference
4. **high_dimensional_reward_spaces/src/conformal_safety.py** - Module 2 reference
