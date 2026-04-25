# Paper 1: Preference Inconsistency & The Hodge Critic

## Result Interpretation Guide

**Working Title**: "Filtering Cyclic Preferences in RLHF: A Discrete Hodge Theory Approach"

**Core Thesis**: Cyclic inconsistencies in human preference data enable reward hacking. Filtering preferences by their Hodge decomposition reliability score produces more robust reward models.

---

## Key Experiments

### Experiment A: Pre-filtered Reward Models

**Setup**:
- Dataset: Anthropic HH-RLHF (~160k examples, sample 5000 per seed)
- Embeddings: Sentence-Transformers (384-dim)
- 50 seeds for statistical significance

**Methods Compared**:
| Method | Filter Criterion | What It Captures |
|--------|-----------------|------------------|
| `raw` | None | Baseline (all preferences) |
| `harmonic_only` | harmonic_ratio < 0.8 | Global Condorcet cycles |
| `curl_only` | curl_ratio < 0.8 | Local 3-clique cycles |
| `reliability_score` | reliability ≥ 0.5 | Gradient dominance |

---

## Metrics & Interpretation

### Primary Metrics

| Metric | Definition | What It Means |
|--------|-----------|---------------|
| **Exploitation Rate** | P(model prefers rejected) < 0.3 | **KEY METRIC**: Lower = more robust |
| **Accuracy** | P(model prefers chosen) > 0.5 | Basic learning capability |
| **Reliability Score** | ||gradient||² / ||total||² | Preference transitivity |

### Expected Results Matrix

| Comparison | Expected Outcome | Interpretation |
|------------|-----------------|----------------|
| filtered < raw exploitation | ✓ Core thesis validated | Filtering removes hackable cycles |
| harmonic ≈ curl exploitation | Similar contributions | Both cycle types matter |
| reliability < harmonic/curl | Reliability is strictest | Combined filtering is best |
| accuracy similar across all | ✓ Filtering doesn't hurt learning | Quality preserved |

---

## Statistical Analysis Requirements

### Per-Seed Metrics to Collect
```python
{
    "seed": int,
    "method": str,  # "raw", "harmonic_only", "curl_only", "reliability_score"
    
    # Hodge decomposition statistics
    "avg_gradient_energy": float,
    "avg_curl_energy": float,
    "avg_harmonic_energy": float,
    "avg_reliability_score": float,
    
    # Training statistics
    "n_train_samples": int,
    "n_filtered_out": int,
    "filter_rate": float,  # fraction filtered
    
    # Evaluation metrics
    "test_accuracy": float,
    "exploitation_rate": float,
    "exploitation_95ci": (float, float),
}
```

### Statistical Tests

| Test | Purpose | When to Use |
|------|---------|-------------|
| **Welch's t-test** | Compare means | filtered vs raw exploitation |
| **Mann-Whitney U** | Non-parametric | If non-normal distribution |
| **Cohen's d** | Effect size | Report alongside p-values |
| **95% CI** | Uncertainty | All main results |

### Minimum Sample Size
- **50 seeds** per method for publication-quality statistics
- **20 seeds** acceptable for initial validation

---

## Figures to Generate

### Figure 1: Exploitation Rate Comparison (Main Result)
```
Bar chart with 95% CI:
X-axis: Method (raw, harmonic_only, curl_only, reliability_score)
Y-axis: Exploitation Rate (0-0.5)
Error bars: 95% confidence intervals
Annotation: Significance stars (* p<0.05, ** p<0.01, *** p<0.001)
```

### Figure 2: Energy Distribution by Method
```
Stacked bar chart:
X-axis: Method
Y-axis: Energy fraction (0-1)
Colors: Gradient (green), Curl (yellow), Harmonic (red)
Purpose: Show what each filter removes
```

### Figure 3: Reliability vs Exploitation Correlation
```
Scatter plot:
X-axis: Average reliability score per context
Y-axis: Exploitation rate on that context's test pairs
Regression line + R² value
Purpose: Validate reliability → exploitation causation
```

### Figure 4: ROC Curve for Filtering Thresholds
```
ROC curves for each filter type:
X-axis: False positive rate (good preferences filtered)
Y-axis: True positive rate (bad preferences filtered)
Compare: harmonic_only, curl_only, reliability_score
Purpose: Show reliability_score has best AUC
```

---

## Result Interpretation Scenarios

### Scenario A: Strong Positive Result (Expected)
```
reliability_score exploitation: 0.05 ± 0.02
raw exploitation: 0.30 ± 0.04
Effect size (Cohen's d): > 0.8 (large)
```
**Interpretation**: "Filtering by reliability score reduces exploitation by 83%, demonstrating that cyclic preferences are the primary source of reward hacking vulnerability."

### Scenario B: Moderate Positive Result
```
reliability_score exploitation: 0.15 ± 0.03
raw exploitation: 0.30 ± 0.04
Effect size: 0.5-0.8 (medium)
```
**Interpretation**: "Filtering reduces exploitation by 50%, suggesting cyclic preferences contribute significantly but not exclusively to reward hacking."

### Scenario C: Curl vs Harmonic Differential
```
curl_only exploitation: 0.08 ± 0.02
harmonic_only exploitation: 0.20 ± 0.03
```
**Interpretation**: "Local cycles (curl) contribute more to exploitation than global cycles (harmonic), suggesting that 3-clique inconsistencies are the primary attack vector."

### Scenario D: No Significant Difference
```
All methods similar exploitation rates
```
**Interpretation**: "HH-RLHF may have low natural cyclic content (check avg_reliability_score). Consider synthetic dataset with injected cycles for controlled experiment."

---

## Paper Narrative Structure

### Abstract Key Points
1. RLHF reward models are vulnerable to reward hacking
2. We apply discrete Hodge theory to decompose preferences into transitive (gradient) and cyclic (curl + harmonic) components
3. Training only on high-reliability (gradient-dominated) preferences reduces exploitation by X%
4. This provides a principled, mathematically grounded approach to preference data curation

### Introduction
- Problem: Reward hacking in RLHF
- Insight: Cyclic preferences (A > B > C > A) create exploitable reward landscapes
- Contribution: Hodge decomposition identifies and filters cyclic content

### Method
- Discrete Hodge theory on preference graphs
- Three components: gradient (transitive), curl (local cycles), harmonic (global cycles)
- Reliability score: ||gradient||² / ||total||²
- Filtering strategies: harmonic-only, curl-only, reliability threshold

### Results
- Table 1: Exploitation rates by method (50 seeds, 95% CI)
- Figure 1: Bar chart comparison
- Figure 2: Energy distribution
- Figure 3: Reliability-exploitation correlation
- Statistical tests: Welch's t, Cohen's d

### Discussion
- Why curl matters: 3-clique cycles create local reward hacking opportunities
- Why harmonic matters: global Condorcet paradoxes create systemic vulnerabilities
- Reliability score combines both, outperforms individual filters
- Limitations: requires sufficient data per context for decomposition

---

## Code Snippet: Result Collection

```python
def collect_experiment_a_results(preferences, seeds=50):
    """Collect results for Paper 1."""
    results = []
    
    methods = ["raw", "harmonic_only", "curl_only", "reliability_score"]
    
    for seed in range(seeds):
        for method in methods:
            # Run experiment
            result = run_experiment_a_variant(
                preferences=preferences,
                method=method,
                config=FilteringConfig(method=method),
                seed=seed,
            )
            
            results.append({
                "seed": seed,
                "method": method,
                "accuracy": result.accuracy,
                "exploitation_rate": result.exploitation_rate,
                "avg_reliability": result.avg_reliability,
                "avg_curl_ratio": result.avg_curl_ratio,
                "avg_harmonic_ratio": result.avg_harmonic_ratio,
                "n_train": result.n_train,
            })
    
    return pd.DataFrame(results)


def analyze_paper_1_results(df):
    """Generate Paper 1 statistics."""
    summary = df.groupby("method").agg({
        "exploitation_rate": ["mean", "std", "count"],
        "accuracy": ["mean", "std"],
        "avg_reliability": "mean",
    })
    
    # Statistical tests
    raw = df[df.method == "raw"]["exploitation_rate"]
    reliability = df[df.method == "reliability_score"]["exploitation_rate"]
    
    from scipy.stats import ttest_ind, mannwhitneyu
    t_stat, p_value = ttest_ind(raw, reliability, equal_var=False)
    
    # Effect size
    cohens_d = (raw.mean() - reliability.mean()) / np.sqrt(
        (raw.std()**2 + reliability.std()**2) / 2
    )
    
    return {
        "summary": summary,
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "exploitation_reduction": (raw.mean() - reliability.mean()) / raw.mean(),
    }
```

---

## Connection to Paper 2

Paper 1 establishes:
- **Module 1 (DiscreteHodgeRank)** for preference filtering
- Reliability score as the correct metric for preference quality

Paper 2 uses filtered preferences from Paper 1 as input to:
- **Module 2 (ConformalSafety)** for policy optimization
- End-to-end pipeline: clean preferences → robust reward model → safe policy
