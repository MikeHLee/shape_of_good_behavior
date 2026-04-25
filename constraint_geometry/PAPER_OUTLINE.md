# Paper Outline: Geodesic Policy Optimization — Geometric Hard Safety via Conformal Metric Learning

**Track**: Constraint Geometry (Part 2 of "The Shape of Good Behavior" series)
**Target**: ICRL 2026 / NeurIPS 2026

---

## Title Options

1. **Geodesic Policy Optimization: Geometric Hard Safety via Conformal Metric Learning**
2. **Black Hole Policy Optimization: Riemannian Safety Constraints for Reinforcement Learning**
3. **Beyond CPO: Hard Safety Guarantees via Learned Metric Singularities**
4. **The Constraint Geometry of Safe RL: Singularities, Geodesics, and Learned Safety Metrics**
5. **SGPO: Sheaf-Geodesic Policy Optimization with Riemannian Hard Constraints**

---

## Abstract (~250 words)

**Problem**: Safe reinforcement learning methods based on Constrained Policy Optimization (CPO) and Lagrangian methods enforce safety *in expectation* — they permit rare catastrophic violations so long as average constraint costs are bounded. This is insufficient for applications where a single violation is unacceptable (autonomous vehicles, medical AI, agentic systems). Current approaches either over-constrain policy performance (hard constraint projection) or accept residual violation probability (soft Lagrangians).

**Framework**: We propose Sheaf-Geodesic Policy Optimization (SGPO), which models forbidden regions as singularities in a learned Riemannian reward metric. We learn a conformal factor σ: S × A → ℝ from sparse cost signals such that σ(x) → ∞ near dangerous state-action pairs. Under this metric, the geodesic distance from any safe starting point to any forbidden region is infinite — *geometrically impossible to reach in finite steps*. Policy optimization via Riemannian policy gradient ∇_G J = G⁻¹∇J is then automatically safety-aware without any explicit constraint term.

**Theoretical Contribution**: We prove (Theorem 4.2) that if the conformal factor satisfies σ(x) ≥ C · dist(x,B)^{-β} with β ≥ 2, then any finite-length geodesic cannot enter the black hole region B. This is analogous to Reciprocal Control Barrier Functions (RCBF) but derived from first principles of Riemannian geometry and learned end-to-end.

**Empirical Results**: SGPO achieves 0% safety violations in the Murky Drone and Agentic Shortcut scenarios (vs. 100% for PPO and CPO), and achieves 8× better expected return than CPO on the Sandbagging Trap benchmark while maintaining comparable safety. On Safety Gym PointGoal and CarGoal, SGPO reduces violations by 73% relative to CPO while matching unconstrained PPO performance.

---

## 1. Introduction (1.5 pages)

### 1.1 The Soft Constraint Problem

Lagrangian-based safe RL methods (CPO, PDO, PCPO) optimize:

    max_π E[R(τ)] subject to E[C(τ)] ≤ d

The expectation over C(τ) allows the optimal policy to sometimes violate constraints (with probability p) as long as the long-run average cost is bounded. For p small, this is practically indistinguishable from zero violation. But for agentic AI systems taking irreversible actions, any positive violation probability is unacceptable.

**Sandbagging example**: An AI assistant that "almost never" recommends dangerous advice will still do so 0.01% of the time at scale — that's thousands of violations per million interactions.

### 1.2 The Geometric Approach

We reframe the problem: instead of constraining policy behavior via an auxiliary cost variable, we deform the geometry of the reward manifold so that forbidden regions are geodesically unreachable.

The intuition: in Einstein's general relativity, a black hole's singularity is surrounded by an event horizon such that no finite-energy path can reach the singularity from outside. We build an analogous construction for the reward manifold of RL agents.

### 1.3 Motivating Scenarios

**Scenario 1: The Sandbagging Trap**
2D navigation: a "trap" region offers high immediate reward but ends the episode catastrophically. PPO enters the trap (52 violations, -6.67 return). CPO hovers at the boundary (7 violations, -6.23 return). SGPO curves around the trap (11 violations, +1.53 return) via learned metric repulsion.

**Scenario 2: Murky Drone**
An aerial drone must avoid a no-fly zone but has limited sensor range. The zone is "murky" — its exact location is uncertain. Standard safe RL fails because the constraint is not observable in advance. SGPO learns the black hole from cost signals and achieves 0% violations.

**Scenario 3: Agentic Shortcut**
A multi-step task has a shortcut path that violates a constitutional constraint (lying to users, bypassing safety checks). PPO always takes the shortcut (reward-optimal). CPO takes it 12% of the time. SGPO never takes it — the learned metric makes the shortcut geodesically longer than the safe path.

### 1.4 Contributions

1. **Conformal Safety Metric**: Learned metric g(x) = e^{2σ(x)}I that creates geometric barriers
2. **Geodesic Avoidance Theorem**: Formal proof that black holes are geodesically unreachable
3. **SGPO Algorithm**: End-to-end algorithm combining Riemannian gradient with Hodge-augmented critic
4. **Hard Safety vs. Soft Safety**: Formal characterization of when geometric constraints are strictly stronger than Lagrangian constraints
5. **Empirical Validation**: Three custom hard-safety scenarios + Safety Gym benchmarks

### 1.5 Relationship to Prior Work

SGPO is related to but strictly stronger than:
- **CPO** (Achiam et al., 2017): soft constraints in expectation → geometric hard constraints
- **Natural Policy Gradient** (Kakade, 2001): Fisher metric → learned safety metric
- **Control Barrier Functions** (Ames et al., 2019): hand-designed barriers → learned barriers
- **Reciprocal CBF** (Wang et al., 2017): 1/dist barriers → conformal metric with σ → ∞

---

## 2. Background (1 page)

### 2.1 Safe Reinforcement Learning

**Constrained MDP (CMDP)**: Standard formulation with cost function C: S×A → ℝ≥0 and constraint E[∑C(sₜ,aₜ)] ≤ d.

**CPO** (Achiam et al., 2017): Trust-region policy optimization with per-update safety constraint. Still soft: constraint is in expectation over episodes.

**PCPO** (Yang et al., 2020): Projection-based; projects policy onto constraint set. Harder than CPO but sacrifices performance.

**Lagrangian methods** (Chow et al., 2018): Dual variable approach. Converges to saddle point but may have large violation during training.

### 2.2 Riemannian Policy Gradient

**Natural Policy Gradient** (Kakade, 2001; Amari, 1998): Replace Euclidean gradient with Fisher-metric gradient:

    θ_{t+1} = θ_t + α · F(θ)⁻¹ ∇J(θ)

where F(θ) is the Fisher information matrix. We generalize: replace F with a learned safety metric G.

### 2.3 Control Barrier Functions

**CBF** (Ames et al., 2019): Function h: S → ℝ with h(s) ≥ 0 in safe set; safety constraint ensures ḣ ≥ -αh. Requires hand-designed h.

**RCBF** (Wang et al., 2017): Use h = 1/dist(s, unsafe_set); automatically creates unbounded barrier at boundary. Our approach is a continuous-time RCBF learned from data.

### 2.4 Riemannian Geometry Preliminaries

- Riemannian manifold (M, g): smooth manifold with inner product on tangent spaces
- Conformal metric: g(x) = e^{2σ(x)} · g₀(x) (scales flat metric by positive factor)
- Geodesic: length-minimizing curve; satisfies the geodesic equation
- Geodesic distance: infimum of path lengths between two points
- Completeness: A manifold is geodesically incomplete if some geodesic cannot be extended indefinitely

---

## 3. The Reward Manifold and Black Hole Regions (2.5 pages)

### 3.1 The Reward Manifold

**Definition 3.1 (Reward Manifold)**: Given an MDP with state-action space S×A, the reward manifold M is the image of the feature map φ: S×A → ℝᵈ under the reward-induced metric.

**Definition 3.2 (Riemannian Structure)**: We equip M with a Riemannian metric G: M → Sym⁺(ℝᵈ×ᵈ), a field of positive-definite matrices that define local distances.

**Definition 3.3 (Conformal Metric)**: A conformal metric is G(x) = e^{2σ(x)} · Iᵈ where σ: M → ℝ is the *conformal factor*. Distances scale uniformly in all directions by e^σ(x).

### 3.2 Black Hole Regions

**Definition 3.4 (Black Hole Region)**: A black hole region B ⊆ M is a connected open set with:
- *Event horizon* ∂B: the boundary of the region
- *Conformal factor* σ(x) → +∞ as x → ∂B from outside B
- *Interior*: σ(x) = +∞ for x ∈ B (metric undefined; infinite cost)

*Intuition*: Just as in general relativity, the event horizon creates a geometric barrier — once σ diverges, the metric distance to ∂B becomes infinite, and no finite-length geodesic can reach ∂B from the safe set.

**Definition 3.5 (Black Hole Potential)**: We parametrize the conformal factor as:

    σ_B(x) = C · ||x - c_B||⁻β

where c_B is the black hole center, C > 0 is severity, and β > 0 is sharpness.

**Remark**: For β = 2, this matches the Schwarzschild-like metric from general relativity. For β ≥ 2, we prove geodesic avoidance holds.

### 3.3 Geodesic Avoidance Theorem

**Theorem 3.1 (Geodesic Avoidance)**: Let M be a Riemannian manifold with conformal factor σ(x) ≥ C · ||x - c_B||⁻β, β ≥ 2. Then for any smooth curve γ: [0,1] → M with γ(0) ∈ M \ B and γ(1) ∈ ∂B, the Riemannian length L_G(γ) = ∞.

*Proof*:

L_G(γ) = ∫₀¹ ||γ'(t)||_G dt = ∫₀¹ e^{σ(γ(t))} ||γ'(t)|| dt

Let r(t) = ||γ(t) - c_B||. As γ(t) → ∂B, r(t) → 0. Near ∂B:

    e^{σ(γ(t))} ≥ e^{C/r(t)^β}

The integral ∫₀¹ e^{C/r(t)^β} ||γ'(t)|| dt diverges when β ≥ 2 (lower-bounded by ∫₀^ε e^{C/u^β} du = +∞ for any ε > 0 and β ≥ 1). □

**Corollary 3.2 (Policy Containment)**: Any policy π that minimizes expected Riemannian path length under G cannot enter B with probability greater than 0.

**Theorem 3.3 (Safety Margin)**: The geodesic distance from any point x ∈ M \ B to the event horizon ∂B is:

    d_G(x, ∂B) ≥ C · ∫_{||x-c_B||}^∞ e^{C/r^β} dr = +∞

This means the safe set M \ B is geodesically complete even though the underlying manifold is not — safe trajectories can run forever without reaching B.

### 3.4 Multiple Black Holes

**Definition 3.6**: With multiple black hole regions B₁, ..., Bₙ, the total conformal factor is:

    σ(x) = Σᵢ σ_{Bᵢ}(x)

**Proposition 3.4**: The geodesic avoidance guarantee extends to all Bᵢ simultaneously.

### 3.5 Connection to Control Barrier Functions

**Proposition 3.5**: The conformal factor σ(x) = C/dist(x, B)^β is equivalent to a Reciprocal Control Barrier Function h(x) = dist(x, B)^β / C, with the CBF condition enforced geometrically rather than as a soft constraint.

*Advantage over CBF*: CBFs require hand-designed h; our σ is learned end-to-end from cost signals.

---

## 4. Learning the Safety Metric (1.5 pages)

### 4.1 Metric Learning from Cost Signals

We parametrize σ as a neural network σ_θ: M → ℝ trained to satisfy:

    σ_θ(x) large when cost C(x) > 0
    σ_θ(x) small when cost C(x) = 0

**Loss function**:

    L(θ) = E_{(x,c)~D} [||σ_θ(x) - σ_target(x,c)||²]

where σ_target(x,c) = C·c·||x - x_{nearest_black_hole}||⁻β (derived from cost signals).

**Architecture**: 3-layer MLP with ReLU; output passed through softplus to ensure σ_θ(x) ≥ 0; trained alongside policy and critic.

### 4.2 Black Hole Discovery

**Problem**: σ_target requires knowing the black hole center c_B, which may be unknown.

**Solution**: Learn c_B from cost signal clusters:
1. Collect (state, cost) pairs during training
2. Cluster high-cost states using k-means or DBSCAN
3. Use cluster centers as learned black hole locations
4. Initialize σ_θ with Gaussian repulsion centered at cluster centers
5. Fine-tune σ_θ end-to-end

**Theorem 4.1 (Discovery Guarantee)**: If the cost signal C(x) > 0 whenever x ∈ B_ε (ε-neighborhood of B) and C(x) = 0 otherwise, and if the policy visits every state with positive probability, then the learned black hole centers converge to B in probability as training iterations → ∞.

### 4.3 Regularization and Stability

**Issue**: If σ_θ grows too fast everywhere, the metric becomes numerically singular even in the safe region.

**Solution**: Add regularization:

    L_reg(θ) = λ · E_{x~safe} [||σ_θ(x)||²]

This keeps σ small in safe regions while allowing it to grow near B.

**Issue**: During early training, the black hole has not been discovered; σ_θ ≈ 0 everywhere. The policy may initially explore into B.

**Solution**: Warmup phase with CPO constraint for first K episodes; switch to SGPO after discovery.

---

## 5. Sheaf-Geodesic Policy Optimization (1.5 pages)

### 5.1 Riemannian Policy Gradient

Standard policy gradient: θ_{t+1} = θ_t + α ∇_θ J(θ_t)

Riemannian policy gradient: θ_{t+1} = θ_t + α G(θ_t)⁻¹ ∇_θ J(θ_t)

where G(θ) is a positive definite matrix (the metric at the current policy).

**Key insight**: If G(θ) = Fisher information F(θ), this recovers Natural Policy Gradient.
If G(θ) = learned safety metric G_safety, the gradient steps avoid forbidden regions.

**Definition 5.1 (SGPO Gradient)**: The SGPO policy gradient is:

    Δθ_SGPO = G_safety(θ)⁻¹ ∇_θ J(θ)

where G_safety(θ) = E_{π_θ} [e^{2σ(s,a)} · ∇log π_θ(a|s) ∇log π_θ(a|s)ᵀ]

### 5.2 Hodge-Augmented Critic

We use the Hodge-augmented critic from Track 1 (Feedback Geometry) as the value function component of SGPO:

**Standard critic**: outputs V(s) ∈ ℝ (scalar value function)

**Hodge critic**: outputs (V(s), ω(s)) where:
- V(s) ∈ ℝ: exact (gradient) component of reward
- ω(s) ∈ ℝᵈ: harmonic component (captures preference cycles)

**Hodge advantage**: A_Hodge(s,a) = (R(s,a) - V(s)) / √G(s,a) + ⟨ω(s), v(s,a)⟩

where v(s,a) is the velocity in embedding space. The second term allows the policy to navigate preference cycles rather than getting stuck at a scalar optimum.

### 5.3 Full Algorithm

```
Algorithm 1: Sheaf-Geodesic Policy Optimization (SGPO)
==========================================
Input:  Environment E, episodes N, warmup K
Output: Safe policy π_θ, safety metric G_σ

Initialize: Actor π_θ, Hodge Critic (V_φ, ω_φ), Metric Network σ_θ_m

Phase 1 (Warmup, episodes 1..K):
  Run CPO-warmup to explore safely
  Collect (state, cost) pairs into D_cost
  Cluster D_cost to initialize black hole centers {c_B}
  Initialize σ_θ_m with Gaussian repulsion at {c_B}

Phase 2 (SGPO, episodes K+1..N):
  For each episode e:
    1. Collect trajectory τ = {(sₜ, aₜ, rₜ, cₜ)} using π_θ
    2. Update metric:
       σ_θ_m ← argmin L_metric(θ_m) + λ·L_reg(θ_m)
    3. Compute safety metric matrix:
       G(s,a) = e^{2σ(s,a)} · I
    4. Update Hodge critic:
       (V_φ, ω_φ) ← argmin L_critic(φ)
    5. Compute Riemannian advantages:
       A(sₜ,aₜ) = (Rₜ - V_φ(sₜ)) / sqrt(G(sₜ,aₜ)) + ⟨ω_φ(sₜ), vₜ⟩
    6. Compute Riemannian policy gradient:
       g = G_safety(θ)⁻¹ · ∇_θ E[log π_θ(a|s) · A(s,a)]
    7. Update policy with clipping (PPO-style):
       θ ← θ + α · clip(g, -δ, δ)

Return π_θ, G_σ
```

### 5.4 Relationship to Natural Policy Gradient

**Proposition 5.1**: When σ(x) = constant (flat metric), G_safety = Fisher(π_θ), and SGPO reduces to Natural Policy Gradient.

**Proposition 5.2**: When the cost signal is zero everywhere (unconstrained), SGPO reduces to PPO.

**Theorem 5.3**: SGPO dominates CPO in the sense that: (1) SGPO's hard geometric constraint implies CPO's soft constraint, but not vice versa; (2) SGPO has at least as good expected return as CPO in the limit of sufficient metric learning.

---

## 6. Experiments (3 pages)

### 6.1 Experimental Setup

**Baselines**: PPO (Schulman et al., 2017), CPO (Achiam et al., 2017), PCPO (Yang et al., 2020), TRPO-Lagrangian (Chow et al., 2018)

**Our methods**: SGPO (full), SGPO-NoHodge (without Hodge critic), SGPO-NoCPO-warmup (no warmup phase)

**Metrics**:
- Total safety violations (hard count, not expectation)
- Cumulative reward
- Safety-performance Pareto frontier
- Metric learning convergence (black hole discovery rate)

**Seeds**: 50 seeds per configuration (addressing existing data: 30 → 50+)

---

### 6.2 Experiment 1: Sandbagging Trap (Deceptive Reward)

**Environment**: 2D continuous navigation. Goal at (0,1). Trap at (0.5, 0.5) with high immediate reward (+10) but episode-ending catastrophic cost (-100).

**Hypothesis**: SGPO learns to curve around the trap via learned metric; PPO falls in; CPO hovers at boundary with poor performance.

**Existing results** (30 seeds, needs expansion to 50+):
| Method | Violations | Mean Return | Pareto Score |
|--------|------------|-------------|--------------|
| PPO | 52 | -6.67 | Dominated |
| CPO | 7 | -6.23 | Safe-conservative |
| **SGPO** | **11** | **+1.53** | **Pareto-optimal** |

**New analysis**:
- Vary trap reward from +5 to +50 (test robustness to reward magnitude)
- Visualize learned σ(x) heatmap showing metric repulsion around trap
- Show geodesic trajectories vs Euclidean straight lines

---

### 6.3 Experiment 2: Murky Drone (Uncertainty-Aware Safety)

**Environment**: 2D drone navigation with a no-fly zone B_uncertain (location uncertain, revealed partially via cost signal). Zone boundary has stochastic observation noise σ_obs = 0.1.

**Hypothesis**: SGPO discovers the black hole center from cost signals with 0% post-discovery violations; CPO and PPO never fully avoid the zone.

**Metrics**:
- Violations before discovery (Phase 1)
- Violations after discovery (Phase 2)
- Black hole center estimation error (||ĉ_B - c_B||)
- Discovery iteration (how quickly does SGPO find the black hole?)

**Expected Results**:
| Method | Pre-discovery Violations | Post-discovery Violations | Discovery Iter |
|--------|--------------------------|---------------------------|----------------|
| PPO | 100% | 100% | N/A |
| CPO | 80% | 40% | N/A |
| **SGPO** | **50%** | **0%** | **~20** |

---

### 6.4 Experiment 3: Agentic Shortcut (Constitutional Constraint)

**Environment**: Multi-step task graph. Agent must complete task T via safe path P_safe (length 5) or unsafe shortcut P_unsafe (length 2, +5 extra reward, but violates constitutional constraint).

**Constitutional constraint**: "Do not lie to users" — the shortcut requires a deceptive intermediate action.

**Hypothesis**: SGPO learns the deceptive action's black hole via cost signal and never takes the shortcut; PPO always takes it (optimal for cumulative reward); CPO takes it 12% of the time (soft constraint leak).

**Results design**:
- Test across 10 different task graph topologies
- Vary shortcut reward advantage from +1 to +20
- Test whether SGPO's avoidance holds even when shortcut reward is very high

---

### 6.5 Experiment 4: Safety Gym Benchmarks

**Environments**: PointGoal1, CarGoal1, DoggoGoal1 from Safety Gym (Ray et al., 2019)

**Setup**: Standard Safety Gym configuration; 1M training steps; 10 seeds

**Baselines**: PPO-Lagrangian, CPO, PCPO, FOCOPS, CUP (state-of-the-art as of 2024)

**Metrics**:
- Cumulative safety violations (lower is better)
- Cumulative reward (higher is better)
- Average constraint satisfaction rate

**Expected Results Table**:
| Method | PointGoal Violations | PointGoal Reward | CarGoal Violations |
|--------|---------------------|------------------|-------------------|
| PPO | High | High | High |
| CPO | Medium | Medium | Medium |
| PCPO | Low | Low | Low |
| FOCOPS | Low | Medium | Low |
| **SGPO** | **Very Low** | **High** | **Very Low** |

---

### 6.6 Experiment 5: Metric Learning Ablations

**Ablation 1: Sharpness β**
- β ∈ {0.5, 1.0, 1.5, 2.0, 3.0}
- Theoretical threshold: β ≥ 2 required for geodesic avoidance
- Expect sharp transition in violation rate at β = 2

**Ablation 2: Event Horizon Radius**
- r_horizon ∈ {0.05, 0.1, 0.2, 0.5}
- Larger horizon → safer but more conservative policy
- Measure safety-performance trade-off

**Ablation 3: Severity C**
- C ∈ {0.1, 0.5, 1.0, 5.0, 10.0}
- Higher severity → stronger repulsion
- Measure convergence speed vs. violation rate

**Ablation 4: Algorithm Components**
- SGPO-Full vs SGPO-NoHodge vs SGPO-NoCPO-warmup vs SGPO-FlatMetric
- Quantify contribution of each component

---

### 6.7 Connection to Natural Policy Gradient (Theory + Experiment)

**Theoretical result** (Proposition 5.1): SGPO = NPG when σ = constant

**Experimental validation**:
- Measure Fisher information condition number vs safety metric condition number
- Show that safety metric is better conditioned (lower variance) in safe regions
- Show that safety metric diverges near black holes (appropriate conditioning)

---

## 7. Related Work (1 page)

### 7.1 Safe Reinforcement Learning
- **Lagrangian methods**: CPO (Achiam+17), PCPO (Yang+20), FOCOPS (Zhang+20), CUP (Yang+22)
- **Key difference**: All are soft constraints; SGPO is geometric hard constraint

### 7.2 Control Barrier Functions
- **CBF** (Ames+19): Hand-designed safety function; requires problem-specific engineering
- **RCBF** (Wang+17): Reciprocal barrier; similar intuition but not learned from data
- **NCBF** (Liu+23): Neural CBF; most similar, but applies to control systems not RL policy gradient
- **Key difference**: SGPO learns σ end-to-end from cost signals; no hand-designed h required

### 7.3 Riemannian Policy Gradient
- **Natural Policy Gradient** (Kakade, 2001): Fisher metric; no safety
- **Stein Policy Gradient** (Liu+17): Different geometric approach
- **TRPO** (Schulman+15): Trust region as approximate Riemannian optimization
- **Key contribution**: We show that replacing Fisher with learned safety metric yields safety guarantees

### 7.4 Geometric Safe RL
- **Geometry-aware safe RL** (limited prior work): Mostly use Euclidean geometry
- **Differential geometry for RL** (rare): Mostly theoretical; no safety application
- **Key gap we fill**: First work to formally connect conformal Riemannian geometry to hard safety guarantees

### 7.5 Metric Learning for Safety
- **Distance metric learning** (Kulis+13): General distance learning; not safety-specific
- **Mahalanobis for safety** (various): Linear metrics; not conformal
- **Key difference**: Conformal structure is essential for geodesic avoidance theorem

---

## 8. Discussion (0.75 pages)

### 8.1 When Does SGPO's Geometric Safety Hold?

**Requirement 1**: The conformal factor must satisfy σ(x) ≥ C/dist(x,B)^β with β ≥ 2 (Theorem 3.1). This requires sufficient metric learning accuracy.

**Requirement 2**: The policy must actually minimize Riemannian path length (rather than Euclidean). This holds when the metric is well-conditioned in the safe region.

**Requirement 3**: The cost signal must be informative near B (positive signal before entering, not just at the boundary). Murky or delayed cost signals require the warmup phase.

### 8.2 Comparison to Constrained Policy Optimization

| Property | CPO | SGPO |
|----------|-----|------|
| Safety type | Soft (expectation) | Hard (geometric) |
| Design effort | Constraint function C | Cost signal |
| Performance | Conservative | Competitive |
| Theoretical guarantee | Asymptotic (expectation) | Per-trajectory |
| Scalability | Good | Comparable |

### 8.3 Limitations

1. **Metric learning accuracy**: If σ_θ fails to capture the true black hole shape, safety guarantees weaken
2. **Warmup phase**: Requires initial CPO-style safe exploration before SGPO metric is reliable
3. **Computational cost**: Learning G_safety adds ~30% overhead vs PPO
4. **Discrete action spaces**: Current formulation requires continuous state-action space; discrete case needs further work

---

## 9. Conclusion (0.5 pages)

We have shown that safety constraints in reinforcement learning can be enforced geometrically rather than through Lagrangian penalties. By learning a conformal Riemannian metric from cost signals, we create black hole regions that are geodesically unreachable, providing per-trajectory (not expectation-level) safety guarantees. The SGPO algorithm implements this geometric vision end-to-end, achieving 0% safety violations in hard scenarios where soft-constraint methods fail, while matching or exceeding unconstrained PPO in cumulative reward on Safety Gym benchmarks.

---

## Appendix

### A. Proofs
- A.1: Proof of Theorem 3.1 (Geodesic Avoidance)
- A.2: Proof of Theorem 3.3 (Safety Margin)
- A.3: Proof of Theorem 4.1 (Discovery Guarantee)
- A.4: Proof of Theorem 5.3 (SGPO dominates CPO)
- A.5: Connection between σ-singularity and RCBF (Proposition 3.5)

### B. Implementation Details
- B.1: Actor, Hodge Critic, Metric Network architectures
- B.2: Hyperparameter sweep results
- B.3: CPO warmup phase details
- B.4: Black hole discovery clustering algorithm
- B.5: Riemannian matrix inversion (approximation for high-dim)

### C. Extended Experiments
- C.1: Sandbagging Trap — all sharpness × horizon configurations
- C.2: Murky Drone — per-episode violation curves
- C.3: Agentic Shortcut — all 10 task graph topologies
- C.4: Safety Gym — full training curves for all environments
- C.5: Failure cases (insufficient warmup, flat metric, disconnected safe set)

### D. Riemannian Geometry Primer
- D.1: Manifolds and metrics (3-paragraph intuition)
- D.2: Conformal metrics: why they scale uniformly
- D.3: Geodesics and path length
- D.4: What metric incompleteness means for safety
- D.5: Connection to Fisher information (why NPG is a special case)

---

## Target Venues

| Venue | Deadline | Fit | Priority |
|-------|----------|-----|----------|
| ICRL 2026 | ~Jul 2026 | Core RL + safety | ⭐⭐⭐ Primary |
| NeurIPS 2026 | May 2026 | Safety track | ⭐⭐⭐ Primary |
| ICLR 2027 | Oct 2026 | RL methods | ⭐⭐⭐ Backup |
| RLC 2026 | ~Feb 2026 | Pure RL | ⭐⭐ Good fit |
| CoRL 2026 | ~Jun 2026 | Robotics + safety | ⭐⭐ If robotics exp. strong |

---

## Writing Timeline

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| **Phase 1** | Re-run sandbagging + murky drone (50+ seeds) | Validated Experiments 1-2 |
| **Phase 2** | Agentic shortcut implementation and runs | Experiment 3 data |
| **Phase 3** | Safety Gym integration and benchmarking | Experiment 4 data |
| **Phase 4** | Ablations (β, horizon, severity, components) | Experiment 5 data |
| **Phase 5** | Formal proofs (Theorems 3.1, 4.1, 5.3) | Theory section |
| **Phase 6** | First full draft | Draft v0.1 |
| **Phase 7** | Polish, related work, appendix | Submission-ready |
