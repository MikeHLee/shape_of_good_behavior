# Methodology Analysis: Theory-Implementation Gap Assessment

## Executive Summary

This document analyzes the gap between the theoretical framework (Sheaf-Theoretic Reward Spaces) and the current toy implementations in the Jupyter notebooks. We identify specific inconsistencies, propose corrections, and outline what's needed to faithfully implement the proposed methods.

---

## 1. Core Theoretical Claims vs. Current Implementation

### 1.1 Sheaf Cohomology for Consistency Detection

**Theory (RESEARCH_PROPOSAL.md)**:
- H¹ ≠ 0 indicates local evaluations cannot be reconciled
- Čech cohomology detects obstructions to gluing local sections
- Cohomology should be computed from restriction maps between scales

**Current Implementation**: 
- **None of the notebooks compute cohomology**
- No restriction maps are learned or applied
- No sheaf structure is actually constructed

**Gap Severity**: HIGH — This is the core theoretical contribution but is entirely missing from experiments.

---

### 1.2 Hodge Decomposition for Cyclic Preferences

**Theory**:
- Rewards decompose into: exact (dφ) + coexact (d*ψ) + harmonic (ω)
- Harmonic forms capture H¹ cohomology (cycles)
- On the circle, constant 1-forms are harmonic and capture the "winding"

**Condorcet_Cycle_Experiment.ipynb Implementation**:
```python
# Current: flux is a SCALAR output, not a 1-form
self.flux_net = nn.Sequential(..., nn.Linear(64, 1))

# Used as: pred_reward = diff_v + fluxes.squeeze()
```

**Issues**:
1. **Flux should be a 1-form** (vector field), not a scalar
2. **No actual Hodge decomposition** — just adding scalar "flux" to TD error
3. **No orthogonality constraint** between exact and harmonic parts
4. **No de Rham cohomology computation** to verify H¹ detection

**Correct Approach**:
```python
# Flux should be a vector field (1-form = dual to tangent vectors)
# On the circle S¹, tangent space is 1D at each point
# Harmonic 1-forms on S¹ are constant (dθ)

# The key insight: ∮ r dθ = ∫∫ (dr/dθ) dθ ≠ 0 for cyclic preferences
# This non-zero integral IS the H¹ obstruction

# Correct: Learn ω ∈ H¹(S¹, ℝ) as a single learnable constant
# (the "winding number" of the preference cycle)
```

---

### 1.3 Black Hole Avoidance via Riemannian Geometry

**Theory (RISK_TOPOLOGY.md)**:
- Black holes are singularities with event horizons
- Metric tensor G(x) → ∞ as x approaches singularity
- Geodesics curve away from singularities
- Policy should follow geodesics, not Euclidean gradients

**AI_Safety_Benchmark.ipynb Implementation**:
```python
# Current: Metric just scales advantage
geodesic_adv = adv / torch.sqrt(g_s)

# The metric is trained to match cost:
target_metric = 1.0 + (costs * 10.0)
```

**Issues**:
1. **Not actual geodesic optimization** — just dividing advantage
2. **Metric is trained on cost signal** — not learned from geometry
3. **No Christoffel symbols** — needed for true geodesic equations
4. **No curvature computation** — theory requires κ → -∞ at black holes
5. **No event horizon enforcement** — trajectories can still enter trap

**Correct Approach**:
```python
# True geodesic policy gradient requires:
# 1. Riemannian gradient: ∇_R f = G^{-1} ∇_E f
# 2. Parallel transport of gradients along trajectory
# 3. Metric distortion that creates "infinite distance" to black hole

# For practical implementation:
# - Conformal metric: g(x) = e^{2σ(x)} I where σ → ∞ near black hole
# - This makes arc length to center infinite
# - Christoffel symbols: Γ^i_jk = δ^i_j ∂_k σ + δ^i_k ∂_j σ - g_jk ∂^i σ
```

---

### 1.4 LLM Style Tuning with Cyclic Preferences

**Theory**:
- Cyclic preferences (A > B > C > A) cannot have consistent scalar reward
- Vector-valued rewards / reward sheaf should handle this
- H¹ ≠ 0 detects the cycle

**LLM_Style_Tuning_Experiment.ipynb Implementation**:
```python
# Current: Flux is trained to match action * reward
target_flux = actions * rewards
flux_loss = nn.MSELoss()(f, target_flux)

# Advantage is dot product of flux and action
adv = (f * actions).sum(dim=1, keepdim=True).detach()
```

**Issues**:
1. **Flux target makes no theoretical sense** — multiplying action by reward
2. **Advantage computation is ad-hoc** — not derived from theory
3. **No cohomology detection** of the preference cycle
4. **Results are chaotic** — both methods behave similarly

---

## 2. Experiment Design Issues

### 2.1 AI Safety Benchmark

**Intended Demonstration**:
- PPO enters trap (ignores risk)
- CPO rides boundary (Lagrangian constraint)
- SGPO curves wide (geodesic avoidance)

**Actual Results** (from screenshot):
- All three algorithms show erratic behavior
- No clear differentiation in path structure
- SGPO doesn't show predicted wide curve

**Root Cause**:
- The "geodesic" modification doesn't actually create geometric avoidance
- The metric is local (per-state) but doesn't affect path planning
- Need: actual trajectory-level geodesic optimization or metric-based value iteration

### 2.2 Condorcet Cycle

**Intended Demonstration**:
- PPO's value function fails on cyclic rewards (gradient vanishes)
- SGPO's Hodge decomposition captures the cycle as H¹
- Flux provides stable driving force

**Actual Results** (from screenshot):
- PPO shows oscillating returns (reasonable)
- SGPO shows higher but still oscillating returns
- Value function plots don't clearly show the expected flat potential + constant flux

**Root Cause**:
- The Hodge critic architecture doesn't enforce the decomposition correctly
- Need: orthogonality loss between exact and harmonic parts
- Need: actual H¹ computation (path integral of reward around cycle)

### 2.3 LLM Style Tuning

**Intended Demonstration**:
- PPO stalls due to cyclic preferences
- SGPO navigates the cycle via vector-valued rewards

**Actual Results** (from screenshot):
- Both trajectories are chaotic and similar
- No clear cycling pattern in SGPO
- Decoded text doesn't show style adaptation

**Root Cause**:
- The reward field is a vector field, but advantage is still scalar
- The policy doesn't receive proper gradient signal from the cycle structure
- Need: vector-valued policy gradient or explicit cycle navigation

---

## 3. Corrected Implementation Strategy

### 3.1 Experiment 1: Condorcet Cycle (Fundamental Test)

**Objective**: Demonstrate that H¹ cohomology detects cyclic preferences.

**Corrected Methodology**:

1. **Environment**: Same circular state space
2. **Reward Structure**: Constant positive reward for clockwise motion = ω = c·dθ
3. **H¹ Computation**: 
   ```
   H¹ = (1/2π) ∮ r(θ) dθ  (path integral around cycle)
   ```
   This should be non-zero for cyclic preferences.

4. **Value Function Analysis**:
   - PPO: V(θ) tries to satisfy V(θ+2π) = V(θ) + ∫r > V(θ) — contradiction!
   - SGPO: V(θ) = potential part (periodic), ω = harmonic part (the constant c)

5. **Hodge Critic Architecture**:
   ```python
   class HodgeCritic:
       def __init__(self):
           self.potential = nn.Sequential(...)  # V: S¹ → ℝ (periodic)
           self.harmonic_coeff = nn.Parameter(torch.zeros(1))  # Single scalar for H¹
       
       def forward(self, theta):
           # dV + ω where ω = harmonic_coeff * dθ
           V = self.potential(theta)
           omega = self.harmonic_coeff  # Constant 1-form on S¹
           return V, omega
   ```

6. **Training**:
   - Fit: r ≈ V(θ') - V(θ) + ω
   - Orthogonality: ∮ V dθ = 0 (potential has zero average)

### 3.2 Experiment 2: Black Hole Avoidance (Safety Test)

**Objective**: Demonstrate geodesic avoidance of dangerous regions.

**Corrected Methodology**:

1. **Environment**: 2D grid with trap region
2. **Metric Tensor**:
   ```python
   def metric(x, black_hole_center, event_horizon):
       r = distance(x, black_hole_center)
       if r < event_horizon:
           return float('inf')
       # Schwarzschild-like metric
       return 1.0 / (1 - event_horizon/r)
   ```

3. **Geodesic Value Iteration**:
   - Instead of Euclidean Bellman: V(s) = max_a [r + γV(s')]
   - Use Riemannian distance: V(s) = max_a [r + γV(s') - d_g(s,s')]
   - Where d_g is geodesic distance (accounts for metric)

4. **Policy Gradient with Riemannian Geometry**:
   ```python
   # Natural policy gradient uses Fisher information (a metric!)
   # We replace Fisher with our learned safety metric
   riemannian_grad = inverse_metric @ euclidean_grad
   ```

5. **Key Metric Properties**:
   - Arc length inside trap → ∞ (agent "can't reach" the trap)
   - Geodesics curve around the trap automatically
   - No explicit constraint needed — geometry enforces safety

### 3.3 Experiment 3: LLM Style Cycling (Application Test)

**Objective**: Show vector-valued rewards handle cyclic preferences.

**Corrected Methodology**:

1. **Reward Space**: Instead of scalar r, use r ∈ ℝ² where:
   - r₁ = agreement with current preference
   - r₂ = orthogonal preference direction

2. **Pareto Optimization**: Policy optimizes on the reward vector frontier

3. **Cycle Detection**:
   - Compute H¹ of the preference graph
   - If H¹ ≠ 0, scalar aggregation is impossible

4. **Navigation Strategy**:
   - Policy should cycle through styles (not converge to one)
   - Track "phase" of the cycle explicitly

---

## 4. Theoretical Gaps Requiring Resolution

### 4.1 Sheaf Construction

**Missing**: How exactly do we construct the reward sheaf from data?

**Needed**:
- Formal definition of open cover for trajectory space
- Algorithm to learn restriction maps from multi-scale feedback
- Computational method for Čech cohomology H¹

### 4.2 Metric Learning

**Missing**: How do we learn the Riemannian metric from human feedback?

**Options**:
1. Supervised: Humans label danger → fit metric to danger labels
2. Implicit: Metric emerges from trajectory preferences
3. Geometric: Use trajectory curvature to infer reward geometry

### 4.3 Sheaf-Geodesic Policy Optimization

**Missing**: Formal algorithm for policy optimization on learned manifold.

**Needed**:
- Connection to natural policy gradient (which uses Fisher metric)
- Algorithm that uses learned safety metric instead of Fisher
- Proof of convergence / safety guarantees

### 4.4 Hodge Theory in RL

**Missing**: Rigorous connection between Hodge decomposition and value functions.

**Needed**:
- Formal statement: "Reward 1-form decomposes as r = dV + ω where ω ∈ H¹"
- When does H¹ ≠ 0 in practice? (cyclic preferences, non-transitive comparisons)
- Algorithm to compute the decomposition from trajectory data

---

## 5. Recommended Immediate Actions

1. **Rewrite Condorcet notebook** with correct Hodge architecture and H¹ computation
2. **Rewrite Safety notebook** with proper metric distortion (at least conformal scaling)
3. **Add explicit H¹ computation** to verify cycle detection
4. **Create synthetic benchmarks** with known ground truth (known cycles, known black holes)
5. **Simplify experiments** — prove one concept clearly before combining

---

## 6. Validation Criteria

For each experiment to be considered "successful":

1. **Condorcet Cycle**:
   - H¹ computation returns non-zero value (≈ average clockwise reward)
   - PPO value function is flat or oscillating (gradient death)
   - SGPO harmonic coefficient matches H¹ value
   - SGPO provides stable policy gradient even with flat V

2. **Black Hole Avoidance**:
   - Metric diverges at trap boundary (numerically large)
   - Geodesic paths visibly curve around trap
   - PPO enters trap, SGPO stays outside with margin
   - Safety margin scales with metric steepness

3. **LLM Style Cycle**:
   - Policy visits all three style regions over time
   - Cycle period is stable (not chaotic)
   - H¹ of preference graph is correctly computed as non-zero
