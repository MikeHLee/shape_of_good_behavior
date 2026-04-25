# Comprehensive Analysis: Hodge Preference Filtering & Conformal Safety

**Date**: February 26, 2026  
**Status**: In Progress

---

## Executive Summary

This document consolidates findings from our RLHF experiments investigating:
1. **Experiment A**: HodgeRank preference filtering for reward model training
2. **Experiment C**: Conformal safety metrics during policy optimization

### Key Results

| Experiment | Method | Key Metric | Finding |
|------------|--------|------------|---------|
| A | Hodge Filtering | Retention rates | Working: 20-100% retention based on method |
| A | Curl Analysis | Energy distribution | 47% curl energy in random graph; needs percentile filtering |
| C | Conformal SGPO | Violations | **19.28 ± 15.8** vs PPO 23.14 ± 15.9 (17% improvement) |
| C | SGPO_ANIS | Goal rate | **0.4%** vs PPO/CPO 0% (non-zero task completion) |

---

## Part 1: Experiment A - Hodge Preference Filtering

### 1.1 Problem Statement

Human preference data contains inconsistencies (cycles A > B > C > A). These create:
- Reward hacking vulnerabilities
- Unstable reward model training
- Exploitation of annotation noise

**Solution**: Use Hodge decomposition to identify and filter inconsistent preferences.

### 1.2 Mathematical Framework

The preference graph decomposes into three orthogonal components:

```
Preference Flow = Gradient + Curl + Harmonic
```

- **Gradient**: Transitive preferences (A > B > C implies A > C) - **KEEP**
- **Curl**: Local cycles (inconsistencies) - **FILTER OUT**
- **Harmonic**: Global Condorcet paradoxes - **DIAGNOSTIC ONLY**

**Reliability Score** = ||Gradient||² / ||Total||²

### 1.3 Implementation Status

#### Fixed Issues
1. **Sparse matrix shape mismatch** - Fixed degree computation in Laplacian construction
2. **Divergence calculation** - Fixed to use pre-computed degree arrays
3. **Curl filtering threshold** - Changed from absolute (0.5) to percentile-based

#### Curl Filtering Bug Analysis

**Problem**: `curl_only` method retained 100% of data despite 47% curl energy globally.

**Root Cause**: Per-edge curl ratios are distributed (mean=0.29, always < 0.5), even when global curl is significant. Absolute threshold filtering doesn't work.

**Solution**: Percentile-based filtering - remove top X% of edges by curl ratio.

```python
# Before (broken):
keep = curl_ratio < 0.5  # Always true!

# After (fixed):
pref_scores.sort(key=lambda x: x['curl_ratio'])  # Ascending
filtered = pref_scores[:n_keep]  # Keep lowest curl
```

### 1.4 LLM Experiment Results (Qwen-2.5-7B, 5K samples)

| Method | Retention | Exploitation Rate |
|--------|-----------|-------------------|
| raw | 100% | 25.6% |
| reliability_score | 20% | 25.6% |
| harmonic_only | 20% | 25.6% |
| curl_only | 100% → 80% (after fix) | TBD |

**Note**: Uniform exploitation rates because current implementation uses simplified reward model (not full fine-tuning). Full fine-tuning experiment planned as "Experiment X".

### 1.5 Computational Tractability

#### Current Run (50K HH-RLHF samples)
- **Nodes**: 71,297 unique responses
- **Edges**: 44,295 preferences
- **Laplacian construction**: 48s
- **CG solve**: 0.01s (converged)
- **Component computation**: In progress

#### Scaling Analysis
| Dataset Size | Nodes | Time (est.) | Memory |
|--------------|-------|-------------|--------|
| 5K | ~10K | 2-3s | 1GB |
| 50K | ~71K | ~60s | 16GB |
| 160K (full) | ~191K | ~5-10min | 64GB+ |

**Conclusion**: Sparse Hodge decomposition is computationally tractable for full HH-RLHF dataset as a pre-training filter step.

---

## Part 2: Experiment C - Conformal Safety

### 2.1 Problem Statement

RL agents exploit reward functions, including safety-constrained rewards. Standard constraint methods (CPO) use expectation-based constraints that allow occasional violations.

**Solution**: Conformal safety metric that creates geometric barriers (infinite geodesic distance to danger).

### 2.2 Key Results (50 seeds, 300 episodes)

| Method | Violations (mean ± std) | Goal Rate |
|--------|------------------------|-----------|
| PPO | 23.14 ± 15.88 | 0.0% |
| CPO | 21.94 ± 12.82 | 0.0% |
| **conformal_sgpo** | **19.28 ± 15.83** | 0.013% |
| conformal_sgpo_anis | 31.16 ± 47.06 | **0.4%** |

### 2.3 Analysis

**Conformal SGPO** shows:
- **17% violation reduction** vs PPO (19.28 vs 23.14)
- **12% violation reduction** vs CPO (19.28 vs 21.94)
- Statistical significance needs larger sample or longer runs

**SGPO_ANIS** shows:
- Higher variance (outliers like seed 4 with 309 violations)
- **Non-zero goal rate** (0.4%) - preserves task learning while attempting safety
- Trade-off: more violations but actually solves the task sometimes

### 2.4 Historical Comparison (sandbagging_v2)

| Method | Violations | Goal Rate |
|--------|-----------|-----------|
| PPO | 30.08 ± 22.3 | ~0% |
| CPO | 23.8 ± 14.1 | ~0% |
| SGPO | 46.2 ± 32.0 | ~0% |

The current conformal methods are competitive with or better than baselines.

---

## Part 3: Next Steps

### Immediate (Methodology Validation)

1. **Complete 50K Hodge analysis** - Running on Modal
2. **Re-run Experiment A with fixed curl filtering** - Percentile-based
3. **Document full tractability analysis** - Time/memory scaling

### Future Experiments

| Experiment | Description | Goal |
|------------|-------------|------|
| **X** | Full LLM fine-tuning with Hodge-filtered data | Demonstrate downstream benefit on reward hacking benchmark |
| **Y** | Conformal safety in LLM PPO/GRPO | Scale conformal methods to language models |
| **Z** | Open SafetyGym benchmarks | Demonstrate on standard safety RL benchmarks |

---

## Appendix: Code Locations

- **Hodge decomposition**: `src/llm_rlhf_experiments.py` (TextHodgeDecomposition class)
- **Conformal safety**: `src/llm_rlhf_experiments.py` (ConformalSafetyMetric class)
- **Modal runners**: `modal_llm_rlhf.py`, `modal_hodge_analysis.py`
- **Results**: `results/corrected_v2/`, `results/llm_rlhf/`

---

## References

1. Jiang et al. "HodgeRank on random graphs" - Hodge decomposition for ranking
2. Anthropic HH-RLHF dataset - Preference data source
3. "Hodge Theory, Bilattices, and Social Choice.pdf" - Mathematical foundations
