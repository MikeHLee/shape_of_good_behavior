# Paper 2: Conformal Safety Metrics & Catastrophic Avoidance

## Result Interpretation Guide

**Working Title**: "Conformal Safety Barriers for Reinforcement Learning: Geometric Guarantees Against Catastrophic Outcomes"

**Core Thesis**: Soft penalty-based safety constraints can be overcome by sufficient reward. Conformal metrics with σ(x)→∞ at danger boundaries create infinite geodesic barriers that provide per-trajectory safety guarantees.

---

## Key Experiments

### Experiment C: Sandbagging Trap Avoidance

**Setup**:
- Environment: 2D navigation with deceptive trap on path to goal
- Trap: (4, 4) with radius 2.0, reward +5.0 (tempting)
- Goal: (10, 10) with reward +10.0
- Event horizon: radius 1.5 (catastrophic)
- 50 seeds, 300 episodes each

**Methods Compared**:
| Method | Safety Mechanism | Key Property |
|--------|-----------------|--------------|
| `ppo` | None | Baseline |
| `cpo` | Lagrangian constraint | Expectation-based |
| `conformal_sgpo` | Isotropic σ→∞ | Infinite barrier |
| `conformal_sgpo_anis` | Anisotropic σ→∞ | Escape routes free |
| `conformal_sgpo_anis_cchc` | + Hodge reliability | Full pipeline |

---

## Metrics & Interpretation

### Primary Metrics

| Metric | Definition | What It Means |
|--------|-----------|---------------|
| **Violations** | Total trap entries | **KEY SAFETY METRIC**: Lower = safer |
| **Goal Rate** | Episodes reaching goal | Task performance |
| **Final Return** | Mean return (last 50 eps) | Learning quality |
| **Catastrophe Rate** | Episodes hitting event horizon | Critical failures |

### Secondary Metrics (Conformal-specific)

| Metric | Definition | Purpose |
|--------|-----------|---------|
| **avg_sigma** | Mean conformal factor | Higher near danger = working |
| **n_hardened_regions** | Learned → hardened regions | Hybrid learning success |
| **escape_factor** | Mean escape direction | Anisotropic effectiveness |

---

## Expected Results Matrix

| Comparison | Expected Outcome | Interpretation |
|------------|-----------------|----------------|
| conformal < cpo violations | ✓ Infinite barriers > Lagrangian | Core thesis validated |
| conformal_anis > conformal goal_rate | Anisotropic preserves escape | Directional metric helps |
| conformal_cchc ≈ conformal_anis | Hodge adds reliability weighting | Marginal improvement expected |
| All conformal: catastrophe_rate ≈ 0 | ✓ Geometric guarantee works | No event horizon crossings |

---

## Statistical Analysis Requirements

### Per-Seed Metrics to Collect
```python
{
    "seed": int,
    "method": str,
    
    # Safety metrics
    "total_violations": int,
    "catastrophe_count": int,
    "violations_per_episode": float,
    
    # Performance metrics
    "goal_rate": float,
    "final_return": float,
    "final_return_std": float,
    
    # Conformal-specific
    "avg_sigma": float,
    "max_sigma": float,
    "n_hardened_regions": int,
    "avg_escape_factor": float,
    
    # Learning dynamics
    "violations_by_episode": List[int],  # For learning curve
    "returns_by_episode": List[float],
}
```

### Statistical Tests

| Test | Purpose | When to Use |
|------|---------|-------------|
| **Welch's t-test** | Compare mean violations | conformal vs cpo |
| **Chi-square** | Catastrophe rate comparison | Binary outcome |
| **Cohen's d** | Effect size | Report with p-values |
| **Bootstrap CI** | Non-normal distributions | Violation counts |

---

## Figures to Generate

### Figure 1: Violations Comparison (Main Result)
```
Bar chart with 95% CI:
X-axis: Method (ppo, cpo, conformal_sgpo, conformal_sgpo_anis, conformal_anis_cchc)
Y-axis: Mean violations per episode
Error bars: 95% confidence intervals
Highlight: Conformal methods should be near zero
```

### Figure 2: Safety-Performance Tradeoff
```
Scatter plot:
X-axis: Mean violations per episode
Y-axis: Goal rate
Each point: One method (averaged over seeds)
Error bars: 95% CI in both dimensions
Pareto frontier highlighted
Purpose: Show conformal achieves both safety AND performance
```

### Figure 3: Learning Curves
```
Line plot with shaded 95% CI:
X-axis: Episode number
Y-axis: Cumulative violations
Lines: Each method
Purpose: Show conformal methods learn to avoid trap faster
```

### Figure 4: Conformal Factor Heatmap
```
2D heatmap over state space:
Color: σ(x) value (log scale)
Overlay: Trap region (circle), Goal (star), Trajectories (sample)
Purpose: Visualize infinite barrier
```

### Figure 5: Anisotropic Escape Preservation
```
Vector field plot:
Background: σ(x) heatmap
Arrows: Gradient direction (escape routes)
Highlight: Arrows pointing away from trap should be longer
Purpose: Show anisotropic metric preserves escape learning
```

---

## Result Interpretation Scenarios

### Scenario A: Strong Positive Result (Expected)
```
conformal_sgpo_anis violations: 2.1 ± 1.5
cpo violations: 22.0 ± 16.0
catastrophe_rate (conformal): 0.0%
catastrophe_rate (cpo): 5.2%
```
**Interpretation**: "Conformal barriers reduce violations by 90% compared to Lagrangian constraints. More importantly, conformal methods achieve zero catastrophic failures, validating the geometric safety guarantee."

### Scenario B: Anisotropic Advantage
```
conformal_sgpo goal_rate: 45%
conformal_sgpo_anis goal_rate: 72%
Both have similar violation rates
```
**Interpretation**: "Anisotropic metrics achieve 60% higher goal rate while maintaining equivalent safety. Preserving escape routes is crucial for task performance."

### Scenario C: Hybrid Learning Success
```
n_hardened_regions: 1.2 ± 0.4 (converges to true trap)
violations in warmup: 15.0
violations post-hardening: 0.5
```
**Interpretation**: "Hybrid learning successfully discovers danger regions from behavioral telemetry and hardens them into conformal barriers, enabling zero-shot deployment."

### Scenario D: Conformal Fails (Diagnostic)
```
conformal violations ≈ cpo violations
```
**Check**: 
- Is trap reachable? (run diagnostic)
- Is sharpness β sufficient? (ablate)
- Is σ actually diverging? (check max_sigma)

---

## Paper Narrative Structure

### Abstract Key Points
1. Current safe RL methods (CPO, etc.) provide expectation-based guarantees that allow rare catastrophes
2. We introduce conformal safety metrics where g(x) = e^{2σ(x)} and σ→∞ at danger boundaries
3. This creates infinite geodesic distance to danger = geometric unreachability
4. Experiments show zero catastrophic failures vs X% for baselines
5. Anisotropic variant preserves escape learning, achieving Y% higher goal rate

### Introduction
- Problem: Soft penalties can be overcome; expectation-based constraints allow rare catastrophes
- Insight: Riemannian geometry provides per-trajectory guarantees via geodesic distance
- Contribution: Conformal metrics with diverging σ create infinite barriers

### Method
- Conformal metric: g_ij = e^{2σ} δ_ij
- Conformal factor: σ(x) = -β log(d(x)) where d = distance to danger
- Natural gradient scaling: ∇_nat = e^{-2σ} ∇_vanilla (automatic suppression near danger)
- Anisotropic variant: only scale movement toward danger
- Hybrid learning: behavioral telemetry → learned regions → hardened barriers

### Results
- Table 1: Violations, goal rate, catastrophe rate by method
- Figure 1: Violations comparison bar chart
- Figure 2: Safety-performance Pareto frontier
- Figure 3: Learning curves
- Figure 4: Conformal factor visualization
- Statistical tests with effect sizes

### Discussion
- Why infinite barriers work: geodesic distance = cost of crossing
- Why anisotropic helps: escape learning signal preserved
- Connection to Control Barrier Functions: conformal ≈ reciprocal CBF
- Limitations: requires known or learnable danger boundary

---

## Key Theoretical Claims to Validate

### Claim 1: Infinite Geodesic Distance
```
d_geo(safe, danger) = ∫ e^{σ(γ(t))} |γ'(t)| dt → ∞

Validation: Compute geodesic distance numerically for sample paths
Expected: Distance diverges as path approaches danger
```

### Claim 2: Natural Gradient Suppression
```
∇_natural = G^{-1} ∇_vanilla = e^{-2σ} ∇_vanilla → 0 as σ → ∞

Validation: Log gradient magnitudes during training
Expected: Gradients near danger should be orders of magnitude smaller
```

### Claim 3: Per-Trajectory Safety
```
Unlike E[cost] ≤ C, conformal provides: ∀τ, τ ∩ Danger = ∅

Validation: Track every trajectory for danger intersection
Expected: Zero intersections for conformal, non-zero for CPO
```

---

## Ablation Studies

### Ablation 1: Sharpness β
```python
beta_values = [0.5, 1.0, 2.0, 4.0, 8.0]
# Expected: Higher β → stronger barrier → fewer violations
# But also: Too high β → learning instability
```

### Ablation 2: Confidence Threshold for Hardening
```python
thresholds = [0.3, 0.5, 0.7, 0.9]
# Expected: Lower threshold → faster hardening → earlier safety
# But also: Too low → false positives
```

### Ablation 3: Isotropic vs Anisotropic
```python
# Compare violations AND goal_rate
# Expected: Similar violations, anisotropic higher goal_rate
```

---

## Code Snippet: Result Collection

```python
def collect_experiment_c_results(config, seeds=50):
    """Collect results for Paper 2."""
    from conformal_sgpo import (
        train_conformal_sgpo,
        train_conformal_sgpo_anis,
        ConformalSGPOConfig
    )
    from corrected_experiments import FixedSandbaggingEnv
    
    results = []
    
    for seed in range(seeds):
        env = FixedSandbaggingEnv()
        trap_center, trap_radius = env.get_trap_info()
        known_regions = [(trap_center, trap_radius)]
        
        for method in config.methods:
            if method == "conformal_sgpo":
                sgpo_config = ConformalSGPOConfig(
                    episodes=config.episodes,
                    sharpness=config.sharpness,
                    anisotropic=False,
                )
                result = train_conformal_sgpo(env, sgpo_config, seed, known_regions)
            
            elif method == "conformal_sgpo_anis":
                sgpo_config = ConformalSGPOConfig(
                    episodes=config.episodes,
                    sharpness=config.sharpness,
                    anisotropic=True,
                )
                result = train_conformal_sgpo_anis(env, sgpo_config, seed, known_regions)
            
            # ... other methods ...
            
            results.append({
                "seed": seed,
                "method": method,
                "total_violations": sum(result["episode_violations"]),
                "goal_rate": np.mean(result["goal_reached"]),
                "final_return": np.mean(result["episode_returns"][-50:]),
                "avg_sigma": np.mean(result["metrics"]["avg_sigma"]),
                "n_hardened_regions": result["n_hardened_regions"],
            })
    
    return pd.DataFrame(results)


def analyze_paper_2_results(df):
    """Generate Paper 2 statistics."""
    summary = df.groupby("method").agg({
        "total_violations": ["mean", "std"],
        "goal_rate": ["mean", "std"],
        "final_return": ["mean", "std"],
    })
    
    # Catastrophe rate
    # (Assuming catastrophe = violation > threshold per episode)
    
    # Statistical tests: conformal vs cpo
    cpo = df[df.method == "cpo"]["total_violations"]
    conformal = df[df.method == "conformal_sgpo_anis"]["total_violations"]
    
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(cpo, conformal, equal_var=False)
    
    cohens_d = (cpo.mean() - conformal.mean()) / np.sqrt(
        (cpo.std()**2 + conformal.std()**2) / 2
    )
    
    return {
        "summary": summary,
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "violation_reduction": (cpo.mean() - conformal.mean()) / cpo.mean(),
    }
```

---

## Connection to Paper 1

Paper 1 provides:
- **Filtered preferences** with high reliability scores
- **Robust reward model** trained on gradient-dominated data

Paper 2 uses:
- **Reward model from Paper 1** for base reward signal
- **Module 2 (ConformalSafety)** for policy optimization
- **Combined narrative**: Clean data → Robust model → Safe policy

### Joint Figure (for Dissertation/Thesis)
```
Pipeline diagram:
[Raw Preferences] → [Hodge Filter (Paper 1)] → [Clean Preferences]
                                                      ↓
[Environment] ← [Safe Policy (Paper 2)] ← [Robust Reward Model]
                                                      ↓
                                         [Conformal Barrier]
```

---

## Venue Considerations

### Paper 2 Target Venues (Safety Focus)
1. **NeurIPS Safe RL Workshop** - Most aligned
2. **ICML** - Theoretical contribution (infinite barriers)
3. **ICLR** - If strong empirical results
4. **RLC (RL Conference)** - Applied focus

### Key Selling Points by Venue
- **NeurIPS**: "First geometric guarantee for per-trajectory safety"
- **ICML**: "Conformal Riemannian metrics for safe optimization"
- **ICLR**: "Zero catastrophic failures in sandbagging experiments"
