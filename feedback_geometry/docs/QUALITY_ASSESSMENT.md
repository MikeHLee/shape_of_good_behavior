# Quality Assessment: Existing Experiments

**Date**: February 2026
**Purpose**: Identify improvements needed for publication-quality experiments

---

## Current Experiments Reviewed

| File | Environment | Quality Issues |
|------|-------------|----------------|
| `safety_experiment.py` | 2D Sandbagging Trap | Single seed, low-dim, known trap location |
| `safety_experiment_hard.py` | Multi-trap with novel trap | Single seed, 2D, no statistical tests |
| `condorcet_experiment.py` | Condorcet Ring (S¹) | Single seed, toy topology |

---

## Critical Quality Issues

### 1. Statistical Rigor (HIGH PRIORITY)

**Problem**: All experiments use single fixed seed (`torch.manual_seed(42)`).

**Impact**: 
- No confidence intervals
- Results may be seed-specific artifacts
- Cannot claim statistical significance

**Fix**:
- Run 50+ seeds per configuration
- Report mean ± std, 95% CI
- Use Welch's t-test or Mann-Whitney U for comparisons
- Report effect sizes (Cohen's d)

---

### 2. Missing H¹ → Reward Hacking Connection (HIGH PRIORITY)

**Problem**: The sandbagging experiments show SGPO avoids traps, but don't demonstrate that **feedback inconsistency enables the reward hacking** in the first place.

**Impact**: The core thesis (H¹ ≠ 0 creates exploitable gaps) is not experimentally validated.

**Fix**: Create "H¹-Exploitable Sandbagging" experiment:
1. Generate feedback with controlled H¹ (cyclic preferences on trap entry)
2. Train reward model on this inconsistent feedback
3. Show standard RLHF exploits the H¹ gap → enters trap
4. Show Hodge-filtered training prevents exploitation → avoids trap
5. Measure correlation: H¹ magnitude vs. exploitation rate

---

### 3. Low-Dimensional Toy Environments (MEDIUM PRIORITY)

**Problem**: 2D navigation doesn't test "high-dimensional geometry" claim.

**Impact**: Reviewers will question generalization to real embeddings.

**Fix** (phased approach):
- **Phase 1 (current paper)**: Keep 2D but add semantic embedding experiment
- **Phase 2 (follow-up)**: Full Safety Gym integration

Semantic embedding experiment:
- State = sentence embedding (384-dim from sentence-transformers)
- Actions = style transitions (Concise ↔ Empathetic ↔ Detailed)
- Trap = deceptive response style that users initially prefer but is harmful

---

### 4. Missing Baselines (MEDIUM PRIORITY)

**Problem**: Only PPO and "CPO" (actually Lagrangian PG). Missing state-of-the-art.

**Impact**: Cannot claim SGPO beats SOTA.

**Fix**: Add baselines:
- **PCPO** (Yang et al., 2020): Projection-based CPO
- **FOCOPS** (Zhang et al., 2020): First-Order CPO
- **CUP** (Yang et al., 2022): Constrained Update Projection
- **TRPO-Lagrangian**: Standard trust-region Lagrangian

Implementation: Use existing SafeRL libraries (safety-gymnasium, safe-rl-kit).

---

### 5. Metric Learning Quality (MEDIUM PRIORITY)

**Problem**: In `safety_experiment.py`, the `RiemannianMetric` is initialized with **known trap location**:
```python
metric = RiemannianMetric(
    trap_center=env.trap_center,  # ← CHEATING
    trap_radius=env.trap_radius,
    event_horizon=env.event_horizon
)
```

**Impact**: Not a fair test of learned metric vs. hand-designed barrier.

**Fix**: Use `LearnedRiemannianMetric` (from `safety_experiment_hard.py`) that learns danger from cost signals without knowing trap location a priori.

---

### 6. Missing Ablations (MEDIUM PRIORITY)

**Problem**: No systematic exploration of hyperparameters.

**Impact**: Cannot understand which components matter.

**Fix**: Add ablation grid:
- **β (sharpness)**: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
- **Horizon radius**: [0.5, 1.0, 2.0]
- **Severity C**: [1.0, 5.0, 10.0]
- **Algorithm components**: SGPO-Full, SGPO-NoHodge, SGPO-FlatMetric

---

## Proposed Experiment Structure

### Experiment 1: H¹-Exploitable Reward Hacking (NEW - PRIORITY)

**Thesis**: When human feedback contains cyclic inconsistencies (H¹ ≠ 0), standard RLHF reward models are exploitable. The agent can "hack" the reward by entering regions where the cycle occurs.

**Setup**:
1. **Feedback generation**: Create preference pairs with controlled H¹
   - Region A: Users prefer "safe but slow"
   - Region B: Users prefer "fast but risky"
   - Transition: Some users prefer A→B, others B→A (creates cycle)
2. **Reward model training**: Train Bradley-Terry on this inconsistent data
3. **Policy training**: Train PPO on the learned reward
4. **Observation**: PPO exploits the cyclic region → oscillates or enters trap
5. **Intervention**: 
   - Hodge-filter the feedback (remove harmonic component)
   - Re-train reward model
   - Show exploitation rate drops

**Metrics**:
- H¹ magnitude of feedback graph
- Exploitation rate (% episodes entering trap)
- Correlation coefficient (H¹ vs exploitation)

---

### Experiment 2: Sandbagging Trap (UPGRADED)

**Current → Upgraded**:
- Single seed → 50 seeds
- Known trap → Learned metric
- PPO/CPO → PPO/CPO/PCPO/SGPO
- No ablations → Full β/C/horizon grid
- No statistics → Mean ± CI, effect sizes, significance tests

**New metrics**:
- Per-episode violation count (not just total)
- Time-to-first-violation
- Pareto frontier (return vs violations)
- Metric learning convergence rate

---

### Experiment 3: Multi-Trap Generalization (UPGRADED)

**Current → Upgraded**:
- Single seed → 50 seeds
- 2 training + 1 novel → 3 training + 2 novel traps
- No cross-validation → K-fold trap holdout

**New metrics**:
- Generalization gap (training vs novel trap violations)
- Metric extrapolation quality (predicted danger at novel locations)

---

## Implementation Priority

| Priority | Experiment | Effort | Deadline |
|----------|------------|--------|----------|
| 1 | H¹-Exploitable Reward Hacking | High | Week 1-2 |
| 2 | Upgraded Sandbagging (multi-seed + stats) | Medium | Week 2-3 |
| 3 | Ablation grid | Low | Week 3 |
| 4 | Multi-Trap Generalization upgrade | Medium | Week 4 |
| 5 | Semantic embedding experiment | High | Week 5-6 |

---

## Code Quality Improvements

1. **Config-driven experiments**: YAML/JSON configs for reproducibility
2. **Logging**: Weights & Biases or TensorBoard integration
3. **Checkpointing**: Save model checkpoints every N episodes
4. **Results format**: Standardized JSON with metadata (seed, config, git hash)
5. **Plotting**: Automated figure generation with confidence bands
