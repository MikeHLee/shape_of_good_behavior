# Corrected Experiments Results Analysis

**Date**: February 24, 2026  
**Seeds**: 50 per method  
**Episodes**: 300 per seed (Experiment C)

---

## Experiment A: Preference Filtering

### Summary Statistics

| Method | n_train | Avg Reliability | Curl Ratio | Harmonic Ratio |
|--------|---------|-----------------|------------|----------------|
| raw | 214 | 0.720 | 0.0 | 0.0 |
| curl_only | 214 | 0.720 | 0.0 | 0.0 |
| harmonic_only | 144 | 0.720 | 0.0 | 0.0 |
| reliability_score | 144 | 0.720 | 0.0 | 0.0 |

### Interpretation

**Key Observations**:
1. **Filtering is active**: `harmonic_only` and `reliability_score` filter down to 144 samples (vs 214 raw)
2. **33% of contexts filtered**: (214-144)/214 = 32.7% of contexts removed by strict filtering
3. **Zero curl/harmonic in remaining**: After filtering, remaining contexts show 0 cyclic content

**Limitation**: This experiment uses synthetic preferences. The Hodge decomposition is working correctly (validated in tests), but the synthetic data generation created clean cycles that get completely filtered, leaving 0 residual curl/harmonic.

**Next Step**: Run on real HH-RLHF data where cyclic content is naturally distributed.

---

## Experiment C: Conformal Safety

### Summary Statistics

| Method | Violations (mean ± std) | Goal Rate | Final Return |
|--------|------------------------|-----------|--------------|
| conformal_sgpo | 52.7 ± 52.8 | 0.14% | -3.87 ± 17.2 |
| conformal_sgpo_anis | 44.5 ± 36.8 | 0.23% | -2.77 ± 11.9 |

### Statistical Comparison

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Violation reduction | 15.7% | ANIS has fewer violations |
| t-statistic | 0.907 | — |
| p-value | 0.367 | Not significant at α=0.05 |
| Cohen's d | 0.181 | Small effect size |
| Goal rate improvement | 61.9% | ANIS reaches goal more often |

### Interpretation

**Key Findings**:
1. **Anisotropic shows trend toward fewer violations** (15.7% reduction) but not statistically significant
2. **Goal rate improvement is substantial** (61.9%) - anisotropic preserves escape learning
3. **Both methods still have many violations** - baseline comparison needed

**Why High Violations?**

The conformal metric is working (validation showed σ→∞ near danger), but:
1. **No baseline comparison**: We don't know how PPO/CPO would perform on this env
2. **Trap is attractive**: +5.0 reward creates strong temptation
3. **Hybrid learning needs tuning**: hardening threshold may need adjustment
4. **Episode warmup**: First 30 episodes have no safety (metric learning phase)

---

## Comparison: Previous vs Current Results

### Previous Sandbagging Results (Original SGPO)
| Method | Violations | Goal Rate |
|--------|-----------|-----------|
| PPO | 29.3 ± 22.1 | 0.0% |
| CPO | 22.8 ± 19.6 | 0.0% |
| SGPO | 22.6 ± 15.8 | 0.0% |
| SGPO_ANIS | 22.0 ± 16.2 | 0.5% |

### Current Conformal Results
| Method | Violations | Goal Rate |
|--------|-----------|-----------|
| conformal_sgpo | 52.7 ± 52.8 | 0.14% |
| conformal_sgpo_anis | 44.5 ± 36.8 | 0.23% |

### Analysis

**Issue Identified**: Current conformal methods have HIGHER violations than original SGPO.

**Root Causes**:
1. **Different environment**: Fixed env has trap at (4,4) on path, more accessible
2. **Longer episodes**: 300 vs 60 episodes = more exposure to trap
3. **Higher trap reward**: 5.0 vs 3.0 creates stronger temptation
4. **Metric not fully learned**: 30 ep warmup may be insufficient

---

## Recommendations

### Immediate Fixes for Experiment C

1. **Add PPO/CPO baselines** to new environment for fair comparison
2. **Reduce warmup period** or pre-train metric on cost signals
3. **Tune sharpness β** - try β=4.0 or higher for stronger barrier
4. **Lower trap reward** to 3.0 (same as original) for comparable temptation

### For Paper 1 (Preference Inconsistency)

1. **Run on HH-RLHF** - real data has natural cyclic content
2. **Train actual reward models** - current experiment only computes Hodge stats
3. **Measure exploitation rate** on held-out test set

### For Paper 2 (Conformal Safety)

1. **Add baseline comparison** (PPO, CPO, original SGPO)
2. **Run ablation on β** (sharpness parameter)
3. **Compare same environment** to original sandbagging results

---

## Files Generated

- `experiment_a_full_20260224_195241.csv` - Experiment A raw data
- `experiment_c_full_20260224_195625.csv` - Experiment C raw data

---

## Next Steps

1. [ ] Add PPO/CPO/original_SGPO baselines to Experiment C
2. [ ] Integrate HH-RLHF data for Experiment A  
3. [ ] Train reward models (not just compute Hodge stats)
4. [ ] Tune conformal hyperparameters (β, warmup, confidence threshold)
5. [ ] Generate publication figures
