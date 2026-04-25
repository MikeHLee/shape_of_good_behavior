# Publishable Experimental Design: Context-Conditional H¹ Filtering for RLHF

**Date**: 2026-02-23  
**Status**: Implementation Complete - Ready for Full Runs

---

## Executive Summary

This document describes two experiments for publishable results on Hodge-filtered RLHF:

| Experiment | Focus | Key Innovation |
|------------|-------|----------------|
| **A** | Pre-filtered reward models | Compare raw vs H¹-filtered preference data (normalized sizes) |
| **C** | Context-conditional SGPO | Harmonic-discounted advantage preserving valid contextual cycles |

**Core Insight**: Standard H¹ filtering removes ALL cyclic preferences, but some cycles are **valid** (context-dependent, like rock-paper-scissors). We discriminate:
- **Invalid cycles**: Intransitive within same context → filter/discount
- **Valid cycles**: Context-dependent variation → preserve

---

## Mathematical Framework

### Context-Conditional H¹

Given preferences grouped by context $c \in C$:

$$H^1_{\text{marginal}} = \|Y - Y_{\text{grad}}\|_2 \quad \text{(all data)}$$

$$H^1_{\text{conditional}} = \sum_c \frac{n_c}{N} \|Y_c - Y_c^{\text{grad}}\|_2 \quad \text{(within contexts)}$$

$$H^1_{\text{valid}} = H^1_{\text{marginal}} - H^1_{\text{conditional}}$$

**Interpretation**:
- $H^1_{\text{conditional}}$: True inconsistencies (invalid cycles)
- $H^1_{\text{valid}}$: Contextual variation (valid cycles)

### SGPO_ANIS_HODGE Advantage

Standard SGPO_ANIS:
$$A = \frac{r + \gamma V' - V}{\sqrt{g(x,v)}} \cdot \text{escape}(x,v)$$

With Hodge correction:
$$A = \frac{r + \gamma V' - V - \omega_{\text{invalid}}}{\sqrt{g(x,v)}} \cdot \text{escape}(x,v)$$

Where $\omega_{\text{invalid}}$ is the conditional harmonic component.

---

## Experiment A: Pre-filtered Reward Models

### Hypothesis
Reward models trained on H¹-filtered preferences exhibit lower exploitation rates while maintaining comparable accuracy.

### Design

```
┌─────────────────────────────────────────────────────────────┐
│                    HH-RLHF Dataset                          │
│                    (N samples)                              │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                                         ▼
┌─────────────────┐                    ┌─────────────────────┐
│   Raw Sample    │                    │  Filtered Sample    │
│   (M samples)   │                    │  (H¹ < 0.8)         │
│                 │                    │  (M samples)        │
└─────────────────┘                    └─────────────────────┘
         │                                         │
         ▼                                         ▼
┌─────────────────┐                    ┌─────────────────────┐
│ Reward Model    │                    │ Reward Model        │
│ (Bradley-Terry) │                    │ (Bradley-Terry)     │
└─────────────────┘                    └─────────────────────┘
         │                                         │
         └────────────────────┬────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │   Evaluation    │
                    │   - Accuracy    │
                    │   - Exploitation│
                    │     Rate        │
                    └─────────────────┘
```

### Critical: Normalized Data Sizes
Both models train on **equal sample counts** to ensure fair comparison.

### Metrics
- **Test Accuracy**: P(chosen > rejected) on held-out set
- **Exploitation Rate**: P(model strongly prefers rejected) - indicates reward hacking potential
- **H¹ Reduction**: How much conditional H¹ was removed by filtering

### Run Command
```bash
python src/hodge_filtered_rlhf_experiment.py --experiment A --seeds 50 --n-samples 5000
```

---

## Experiment C: Context-Conditional SGPO

### Hypothesis
SGPO_ANIS_HODGE (with context-conditional harmonic discounting) achieves:
1. Lowest violation rate
2. Highest goal rate
3. Comparable or better returns

### Design

```
┌─────────────────────────────────────────────────────────────┐
│                  ContextConditionalHodgeCritic              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │ Value Net   │  │ Harmonic    │  │ Context Conditioner  │ │
│  │   V(s)      │  │   ω(s,a)    │  │   ω_invalid(s,a,c)   │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│   SGPO_ANIS     │  │ SGPO_ANIS_HODGE │  │  Anisotropic Metric │
│   (baseline)    │  │  (ω_invalid     │  │  g(x,v) directional │
│                 │  │   subtracted)   │  │  safety scaling     │
└─────────────────┘  └─────────────────┘  └─────────────────────┘
```

### Methods Compared

| Method | Harmonic Correction | Anisotropic Metric |
|--------|--------------------|--------------------|
| PPO | ✗ | ✗ |
| CPO | ✗ | ✗ |
| SGPO_ANIS | ✗ | ✓ |
| **SGPO_ANIS_CCHC** | ✓ (context-conditional) | ✓ |

### Advantage Formula

```python
# SGPO_ANIS
td_error = rewards + gamma * V_next - V
scale = escape_factors + (1 - escape_factors) / (1 + log(1 + g))
adv = scale * td_error

# SGPO_ANIS_CCHC (Context-Conditional Hodge Critic)
td_error = rewards + gamma * V_next - V - omega_invalid  # Key difference
scale = escape_factors + (1 - escape_factors) / (1 + log(1 + g))
adv = scale * td_error
```

### Metrics
- **Total Violations**: Cumulative trap/hazard entries
- **Final Return**: Mean return over last 100 episodes
- **Goal Rate**: Percentage of episodes reaching goal

### Run Command
```bash
python src/hodge_filtered_rlhf_experiment.py --experiment C --seeds 50 --n-episodes 500
```

---

## File Structure

```
feedback_geometry/src/
├── context_conditional_hodge_critic.py   # Extended HodgeCritic with context
│   ├── ContextualFeedbackItem            # Feedback with context_id
│   ├── ConditionalH1Result               # Marginal vs conditional H¹
│   ├── ContextConditionalHodgeCritic     # Main critic class
│   │   ├── compute_conditional_h1()      # Discriminate valid/invalid cycles
│   │   ├── filter_invalid_cycles()       # For Experiment A
│   │   └── harmonic_given_context()      # For Experiment C
│   └── load_hh_rlhf_with_context()       # Data loading utility
│
├── hodge_filtered_rlhf_experiment.py     # Unified experiment script
│   ├── ExperimentConfig                  # All hyperparameters
│   ├── PreferenceRewardModel             # Bradley-Terry model
│   ├── PreferenceBasedEnv                # RL environment
│   ├── AnisotropicMetricWithHodge        # Extended metric
│   ├── train_sgpo_anis_hodge()           # Training loop
│   ├── run_experiment_a()                # Pre-filtered reward models
│   └── run_experiment_c()                # Context-conditional SGPO
│
└── hodge_utils.py                        # Core Hodge decomposition
    ├── hodge_filter_preferences()        # Threshold-aware filtering
    ├── compute_conditional_h1()          # Context-conditional H¹
    └── filter_invalid_cycles_only()      # Preserve valid cycles
```

---

## Integration with Existing Infrastructure

### From high_dimensional_reward_spaces

| Component | Source | Used For |
|-----------|--------|----------|
| `HodgeCritic` | `hodge_critic.py` | Base class structure |
| `ClippedSGPO` | `sgpo_clipped.py` | Advantage computation pattern |
| `AnisotropicSGPOCritic` | `anisotropic_escape_experiment.py` | Directional metric |
| `mine_preference_cycles.py` | Existing | Real HH-RLHF loading |

### Key Integration Point

The harmonic subtraction from `sgpo_clipped.py` line 139:
```python
td_error = rewards + gamma * V_next * (1 - dones) - V - omega
```

We extend this to use **context-conditional omega**:
```python
omega_invalid = hodge_critic.harmonic_given_context(states, actions, contexts)
td_error = rewards + gamma * V_next * (1 - dones) - V - omega_invalid
```

---

## Expected Results

### Experiment A
| Model | Accuracy | Exploitation Rate |
|-------|----------|-------------------|
| Raw | ~85% | ~15-20% |
| H¹-Filtered | ~83-85% | ~5-10% |

**Exploitation Reduction**: 40-60%

### Experiment C
| Method | Violations | Return | Goal Rate |
|--------|-----------|--------|-----------|
| PPO | ~30 ± 20 | ~0.3 | 0% |
| CPO | ~23 ± 15 | ~1.0 | 0% |
| SGPO_ANIS | ~22 ± 16 | ~0.8 | 0.5% |
| **SGPO_ANIS_CCHC** | ~18 ± 12 | ~1.2 | 1-2% |

---

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
sentence-transformers>=2.2.0  # For real embeddings
datasets>=2.0.0               # For HH-RLHF
```

---

## Next Steps

1. **Full Experiment Runs**: Execute with 50+ seeds on real HH-RLHF data
2. **Statistical Analysis**: 95% CI, effect sizes (Cohen's d), p-values
3. **Ablation Studies**:
   - H¹ threshold sensitivity (0.5, 0.6, 0.7, 0.8, 0.9)
   - Context grouping strategies (prompt hash, topic cluster, evaluator)
4. **Visualization**: H¹ reduction curves, exploitation vs accuracy trade-off
5. **Paper Writing**: Results → Feedback Geometry paper Section 5
