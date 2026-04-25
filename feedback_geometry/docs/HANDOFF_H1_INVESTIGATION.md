# Handoff: H¹ Correlation Investigation

## Status: � Fixes Implemented, Validation Needed

**Date**: 2026-02-22 (Updated)  
**Experiment**: `h1_reward_hacking_experiment_v2.py` (NEW)  
**Results**: `results/h1_exploitation_v2/results.json`

---

## Update Summary (2026-02-22)

### Fixes Implemented ✅

1. **H¹ Injection Fixed** (`hodge_utils.py`)
   - Now uses proper Hodge decomposition to inject harmonic component
   - Correlation between target and measured H¹: **r = 0.991**
   - Test: `python3 hodge_utils.py` shows H¹ tracks linearly with target

2. **Train/Test Split Implemented** (`h1_reward_hacking_experiment_v2.py`)
   - Train items: 0-69 (NO TRAP)
   - Test items: 70-99 (includes trap at 80-90)
   - Trap is UNSEEN during reward model training

3. **True Hodge Filtering** (`hodge_utils.hodge_filter_preferences()`)
   - Properly removes harmonic component via decomposition
   - Verified: H¹ drops to 0.0 after filtering

### Current Results (5 seeds, quick test)

```
H¹ Target → Measured | Standard Trap | Hodge Trap
0.0 → 0.11          | 7.18 ± 0.96   | 6.80 ± 1.48
0.2 → 0.42          | 5.45 ± 1.19   | 6.74 ± 1.46
0.4 → 0.80          | 6.52 ± 0.75   | 5.71 ± 0.90
0.6 → 1.16          | 5.82 ± 0.52   | 7.76 ± 1.53
0.8 → 1.46          | 7.00 ± 0.84   | 6.14 ± 1.85
1.0 → 1.69          | 6.14 ± 1.18   | 6.93 ± 0.78

Correlations:
  H¹ ↔ Standard traps: r=-0.05, p=0.79
  H¹ ↔ Hodge traps: r=0.03, p=0.86
```

### Remaining Issue: No H¹→Exploitation Correlation

The H¹ injection now works, but trap visits don't correlate with H¹. Possible causes:

1. **Variance**: 5 seeds is too few; need 50+ seeds for statistical power
2. **Mechanism Gap**: The cyclic bias in training items (0-69) may not transfer to exploitation in test items (70-99)
3. **Reward Model Architecture**: The item embedding approach may not capture the kind of bias that leads to exploitation

### Next Steps

1. **Run full experiment** (50 seeds): `python3 h1_reward_hacking_experiment_v2.py --seeds 50`
2. **If still no correlation**, redesign the exploitation mechanism:
   - Instead of separate train/test items, use train/test REGIONS in continuous state space
   - Make the trap a region where cyclic feedback generalizes (same "style" as training cycles)
3. **Consider alternative metrics**: Instead of trap visits, measure reward model prediction error in cyclic regions

---

## Original Issues (Fixed)

---

## Issue 1: H¹ Injection Not Working

### Evidence

From `results.json`:
```json
"measured_h1": [
  {"target": 0.0, "measured_mean": 1.2436},
  {"target": 0.1, "measured_mean": 1.2457},
  {"target": 0.2, "measured_mean": 1.2451},
  {"target": 0.3, "measured_mean": 1.2451},
  {"target": 0.4, "measured_mean": 1.2472},
  {"target": 0.5, "measured_mean": 1.2488}
]
```

**Problem**: All measured H¹ values are ~1.24 regardless of injected magnitude (0.0-0.5). The "H¹ magnitude" parameter is not actually controlling the cyclic inconsistency.

### Root Cause

In `h1_reward_hacking_experiment.py`, the H¹ injection is:

```python
cyclic_bias = self.h1_magnitude * np.sin(2 * np.pi * (i - j) / self.num_items)
noisy_prob = np.clip(base_prob + cyclic_bias, 0.01, 0.99)
```

But the `compute_h1_magnitude()` function measures something different:
- It computes transitivity violations in triplets
- The base dataset already has ~1.24 transitivity violation from random noise
- The sinusoidal injection gets lost in the noise

### Fix Required

**Option A**: Use proper Hodge decomposition to inject H¹
```python
# Build boundary operator ∂₁: edges → nodes
# Inject harmonic component directly: h ∈ ker(∂₁*) ∩ ker(∂₁)
# This guarantees H¹ ≠ 0 by construction
```

**Option B**: Use synthetic preferences with controlled cycles
```python
# Create explicit Condorcet cycles: A > B > C > A
# Scale the cycle strength with h1_magnitude
# Measure H¹ as the number/strength of such cycles
```

**Option C**: Use real preference data with known H¹
```python
# PrefLib datasets have measured transitivity violations
# Select subsets with varying H¹ levels
```

---

## Issue 2: Data Snooping (Train/Test Overlap)

### Current Design (Flawed)

```
┌─────────────────────────────────────────────────────────┐
│                    SAME DATASET                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Items 0-44  │    │ Items 45-55 │    │ Items 56-99 │  │
│  │   (safe)    │    │   (TRAP)    │    │   (safe)    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘  │
│                           ↓                              │
│                  Reward Model Training                   │
│                           ↓                              │
│                  Policy Training + Evaluation            │
│                  (on SAME trap region!)                  │
└─────────────────────────────────────────────────────────┘
```

**Problem**: The reward model sees preferences involving the trap region during training. The policy is evaluated on the same trap it trained on. This is data snooping.

### Correct Design

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING DATA                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Items 0-69 (with H¹ cyclic inconsistency)       │    │
│  │ NO TRAP - just cyclic preferences               │    │
│  └─────────────────────────────────────────────────┘    │
│                           ↓                              │
│                  Reward Model Training                   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                    TEST DATA (UNSEEN)                    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Items 70-99 (includes exploitation trap)        │    │
│  │ TRAP at items 80-90 (never seen in training!)   │    │
│  └─────────────────────────────────────────────────┘    │
│                           ↓                              │
│                  Policy Evaluation ONLY                  │
└─────────────────────────────────────────────────────────┘
```

**Key**: The trap must be **out-of-distribution**. This tests whether:
- Standard RLHF: Learned cyclic bias generalizes → exploits unseen trap
- Hodge-filtered: No cyclic bias learned → avoids unseen trap

---

## What "Hodge Filtering" Should Mean

### Current Implementation (Insufficient)

```python
# Just adds a penalty on harmonic coefficient
harmonic_penalty = 10.0 * (self.harmonic_coeff ** 2)
loss = prediction_loss + harmonic_penalty
```

This doesn't actually perform Hodge decomposition.

### Correct Implementation

**True Hodge Filtering**:
1. Build preference graph from training data
2. Compute Hodge decomposition: `f = df + δg + h`
   - `df`: Gradient (exact) component → consistent preferences
   - `δg`: Co-gradient component → sinks/sources
   - `h`: Harmonic component → cyclic inconsistency (H¹)
3. **Remove `h`** from the preference signal before training
4. Train reward model on filtered preferences

```python
# Correct approach
from scipy.sparse.linalg import lsqr

def hodge_decompose(preferences, graph):
    # Build boundary operators
    B1 = build_boundary_1(graph)  # edges → nodes
    B2 = build_boundary_2(graph)  # faces → edges (if triangles)
    
    # Solve for potential: preferences ≈ B1^T φ
    phi, _ = lsqr(B1.T, preferences)
    gradient_component = B1.T @ phi
    
    # Harmonic = preferences - gradient - co-gradient
    harmonic = preferences - gradient_component  # simplified
    
    return gradient_component, harmonic

# Train on gradient_component only (H¹ = 0 by construction)
```

---

## Corrected Experiment Design

### Phase 1: Generate Datasets with True H¹ Control

```python
def generate_preference_dataset(n_items, h1_level):
    """
    Generate preferences with controlled H¹.
    
    h1_level = 0: Perfect transitivity (no cycles)
    h1_level = 1: Strong Condorcet cycles
    """
    # Start with ground truth utilities
    utilities = np.random.randn(n_items)
    
    # Generate pairwise preferences
    preferences = []
    for i, j in all_pairs(n_items):
        true_prob = sigmoid(utilities[i] - utilities[j])
        
        # Inject controlled cycles
        if h1_level > 0:
            # Add cyclic perturbation
            cycle_contrib = h1_level * compute_cycle_membership(i, j)
            true_prob = clip(true_prob + cycle_contrib, 0.05, 0.95)
        
        label = bernoulli(true_prob)
        preferences.append((i, j, label))
    
    # Verify H¹ via Hodge decomposition
    measured_h1 = compute_harmonic_norm(preferences)
    assert abs(measured_h1 - h1_level) < 0.1  # Sanity check
    
    return preferences
```

### Phase 2: Train/Test Split

```python
# TRAINING: Items 0-69, NO TRAP
train_items = list(range(70))
train_prefs = filter_preferences(all_prefs, train_items)

# Hodge decomposition on TRAINING data only
gradient_prefs, harmonic_prefs = hodge_decompose(train_prefs)

# Train two models:
# 1. Standard: train on train_prefs (includes H¹)
# 2. Hodge-filtered: train on gradient_prefs (H¹ removed)

# TESTING: Items 70-99, includes UNSEEN TRAP at 80-90
test_items = list(range(70, 100))
trap_items = list(range(80, 90))  # Never in training!
```

### Phase 3: Policy Evaluation on Unseen Trap

```python
# Create environment with UNSEEN trap
env = PreferenceTrapEnv(
    reward_model=trained_model,
    items=all_items,  # Full item space
    trap_indices=trap_items,  # Items 80-90
    start_region=test_items[:5],  # Start in test region
)

# Evaluate: Does the policy exploit the unseen trap?
# - If cyclic bias generalizes: HIGH trap visits
# - If no cyclic bias: LOW trap visits
```

---

## Parameter Search Required

Once the above issues are fixed, systematic parameter search is needed:

| Parameter | Current | Search Range | Notes |
|-----------|---------|--------------|-------|
| `h1_magnitude` | 0.0-0.5 | 0.0-1.0 | After fixing injection |
| `harmonic_penalty` | 10.0 | 0.1-100.0 | May need tuning |
| `train_test_split` | None | 0.5-0.8 | Items for train/test |
| `trap_severity` | hardcoded | 1.0-10.0 | How attractive is trap |
| `num_cycles` | implicit | 1-10 | Explicit Condorcet cycles |

---

## Files to Modify

1. **`h1_reward_hacking_experiment.py`**:
   - Fix `PreferenceDataset` to inject true H¹
   - Implement proper Hodge decomposition
   - Add train/test split for items
   - Move trap to test-only region

2. **`experiment_framework.py`**:
   - Add `TrainTestSplit` configuration
   - Add H¹ verification metrics

3. **NEW: `hodge_utils.py`**:
   - `build_boundary_operators()`
   - `hodge_decompose()`
   - `compute_harmonic_norm()`
   - `verify_h1_level()`

---

## Success Criteria

After fixes, we should see:

1. **H¹ Control**: Measured H¹ tracks injected H¹ (r > 0.9)
2. **Exploitation Correlation**: Standard RLHF trap visits increase with H¹ (r > 0.5, p < 0.05)
3. **Hodge Protection**: Hodge-filtered trap visits uncorrelated with H¹ (|r| < 0.2)
4. **Generalization**: Effect holds on UNSEEN traps (not just in-distribution)

---

## References

- Jiang et al. (2011): "Statistical ranking and combinatorial Hodge theory"
- Hirani (2003): "Discrete Exterior Calculus" (Chapter 4: Hodge decomposition)
- Current code: `topics/high_dimensional_reward_spaces/src/condorcet_experiment.py` (has some Hodge code)
