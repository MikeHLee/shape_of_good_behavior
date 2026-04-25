# Sheaf-Theoretic Reward Spaces for Safe Reinforcement Learning

## Paper Outline (Target: ICML 2025)

---

## Title Options

1. **Sheaf-Theoretic Reward Spaces for Safe Reinforcement Learning**
2. **Sheaf-Geodesic Policy Optimization: Navigating Reward Manifolds with Topological Safety Guarantees**
3. **Beyond Scalar Rewards: Cohomological Consistency and Geometric Safety in RLHF**
4. **Detecting Preference Cycles and Avoiding Black Holes: A Topological Approach to Safe RL**

---

## Abstract (~250 words)

**Problem**: RLHF collapses rich human preferences into scalar rewards, enabling reward hacking, missing cyclic preferences, and providing no formal safety guarantees.

**Solution**: Sheaf-Theoretic Reward Spaces (STRS) — embed rewards in high-dimensional space, detect inconsistencies via H¹ cohomology, model forbidden regions as geometric singularities, optimize via geodesics.

**Key Results**:
1. H¹ ≠ 0 detects cyclic preferences (Condorcet paradoxes) that scalar rewards cannot represent
2. Riemannian metrics create geometric safety margins without explicit constraints
3. Sheaf-Geodesic Policy Optimization (SGPO) outperforms PPO and CPO on safety benchmarks
4. Framework provides interpretable topological safety certificates

**Impact**: First application of sheaf cohomology to RLHF; principled handling of cyclic preferences; geometric approach to AI safety.

---

## 1. Introduction (1.5 pages)

### 1.1 The Scalar Reward Problem
- RLHF success and limitations
- Reward hacking, specification gaming
- Information loss from preference → scalar

### 1.2 Motivating Examples
- **Condorcet Cycles**: Concise > Empathetic > Detailed > Concise in LLM style
- **Deceptive Traps**: High immediate reward masking catastrophic risk
- **Evaluator Disagreement**: Different humans, different preferences

### 1.3 Our Contribution (Bullet Points)
1. Sheaf-theoretic framework for reward modeling
2. H¹ cohomology for consistency detection
3. Black hole formalism for forbidden regions
4. Sheaf-Geodesic Policy Optimization algorithm
5. Experimental validation on three benchmarks

### 1.4 Paper Organization
Brief roadmap of sections

---

## 2. Background (1 page)

### 2.1 Reinforcement Learning from Human Feedback
- Standard RLHF pipeline
- Reward model training
- Policy optimization (PPO, DPO)

### 2.2 Safe Reinforcement Learning
- Constrained MDPs
- Lagrangian methods (CPO)
- Risk-sensitive objectives

### 2.3 Prerequisites from Topology (Brief)
- Sheaves: local-to-global data structures
- Cohomology: measures obstruction to gluing
- Riemannian geometry: curved spaces, geodesics

---

## 3. Sheaf-Theoretic Reward Spaces (2.5 pages)

### 3.1 The Reward Sheaf
**Definition 3.1**: Trajectory space T with open cover (steps, segments, trajectories)
**Definition 3.2**: Reward presheaf F with restriction maps ρ
**Definition 3.3**: Sheaf condition (locality + gluing)

*Intuition*: Local rewards (per-step) must glue consistently to global rewards (trajectory)

### 3.2 Cohomology for Consistency
**Definition 3.4**: Čech cohomology H⁰, H¹
**Theorem 3.1**: H¹ = 0 ⟺ feedback is globally consistent
**Corollary 3.2**: H¹ ≠ 0 detects cyclic preferences

*Intuition*: H¹ measures the "winding" of preferences around loops

### 3.3 The Hodge Decomposition
**Theorem 3.3**: Reward 1-form decomposes as r = dV + ω where:
- dV = exact part (gradient of potential)
- ω = harmonic part (H¹ cohomology)

*Application*: Standard RL learns V; we also learn ω to capture cycles

### 3.4 Computational Aspects
- Discrete approximation of cohomology
- Efficient computation for trajectory data
- Complexity analysis: O(T · k · d²)

---

## 4. Geometric Safety via Black Holes (2 pages)

### 4.1 The Reward Manifold
**Definition 4.1**: Reward manifold M = φ(T) with induced metric
**Definition 4.2**: Riemannian metric G(x) on M

### 4.2 Black Hole Regions
**Definition 4.3**: Black hole B with center c, event horizon r, severity σ
**Definition 4.4**: Potential Φ(x) → -∞ as x → B

*Intuition*: Forbidden regions are singularities where distance → ∞

### 4.3 Metric Learning
**Proposition 4.1**: Conformal metric g(x) = e^{2σ(x)} I with σ → ∞ near black holes

*Key insight*: We learn σ from cost signals, not hand-design constraints

### 4.4 Safety Guarantees
**Theorem 4.2**: Geodesics in learned metric avoid black holes with probability 1-δ
**Theorem 4.3**: Safety margin proportional to metric steepness

---

## 5. Sheaf-Geodesic Policy Optimization (1.5 pages)

### 5.1 Standard Policy Gradient
- Euclidean gradient ∇J
- Fisher information metric (natural gradient)

### 5.2 Riemannian Policy Gradient
**Definition 5.1**: Riemannian gradient ∇_G J = G^{-1} ∇J
**Algorithm 5.1**: SGPO update rule

*Key insight*: Replace Fisher with learned safety metric

### 5.3 Hodge-Augmented Critic
**Definition 5.2**: Hodge critic outputs (V, ω) instead of just V
**Proposition 5.1**: Advantage uses both potential and harmonic parts

### 5.4 Full Algorithm
```
Algorithm 1: Sheaf-Geodesic Policy Optimization (SGPO)
Input: Environment E, cost signals c, episodes N
Output: Safe policy π, learned metric G

1. Initialize π, V, ω, G
2. For episode = 1 to N:
   a. Collect trajectory τ with rewards r, costs c
   b. Update metric: G ← fit(c)  // Learn geometry from costs
   c. Update Hodge critic: (V, ω) ← fit(r)  // Decompose rewards
   d. Compute Riemannian advantage: A = (R - V) / √G
   e. Update policy: π ← π + α G^{-1} ∇log π · A
3. Return π, G
```

---

## 6. Experiments (2.5 pages)

### 6.1 Condorcet Cycle (H¹ Detection)

**Setup**: Circular state space, constant positive reward for clockwise motion
**Hypothesis**: H¹ ≠ 0 detects the cycle; PPO fails, SGPO succeeds

**Metrics**:
- Learned ω vs ground truth H¹
- Value function analysis (PPO flat/oscillating)
- Policy stability

**Results Table**:
| Method | H¹ Error | Value Loss | Return |
|--------|----------|------------|--------|
| PPO    | N/A      | High       | Low    |
| SGPO    | < 0.1    | Low        | High   |

### 6.2 Black Hole Avoidance (Safety)

**Setup**: 2D navigation with deceptive trap offering high reward
**Hypothesis**: SGPO curves around trap; PPO enters; CPO rides boundary

**Metrics**:
- Trap violations per episode
- Safety margin (min distance to trap)
- Goal success rate

**Results Table**:
| Method | Violations | Safety Margin | Success |
|--------|------------|---------------|---------|
| PPO    | High       | 0 (enters)    | Low     |
| CPO    | Medium     | ~0 (boundary) | Medium  |
| SGPO    | Low        | >1.0 (wide)   | High    |

### 6.3 LLM Style Cycling

**Setup**: Simulated LLM style space with cyclic preferences
**Hypothesis**: SGPO navigates cycle; PPO gets stuck

**Metrics**:
- Archetype coverage (visits all styles)
- Cycle correctness (right order)
- Transition smoothness

### 6.4 Ablation Studies

1. **Hodge vs Scalar Critic**: Contribution of harmonic term
2. **Metric Learning**: Supervised vs implicit metric
3. **Hyperparameter Sensitivity**: Event horizon size, metric sharpness

---

## 7. Related Work (1 page)

### 7.1 Distributional and Multi-Objective RL
- Distributional RL (Bellemare et al.)
- Multi-objective RL
- Our contribution: geometric structure + consistency

### 7.2 Safe Reinforcement Learning
- Constrained MDPs (Altman)
- CPO (Achiam et al.)
- Our contribution: geometry replaces constraints

### 7.3 Topological Methods in ML
- Persistent homology
- Sheaves in neural networks (Hansen et al.)
- Our contribution: first application to RLHF

### 7.4 Preference Learning
- Bradley-Terry models
- Intransitivity in preferences
- Our contribution: topological treatment of cycles

---

## 8. Discussion and Limitations (0.5 pages)

### 8.1 Limitations
- Computational cost of cohomology (mitigated by efficient approximations)
- Metric learning requires cost signals (bootstrapping problem)
- Theoretical guarantees require assumptions on manifold structure

### 8.2 Future Work
- Scale to language model fine-tuning
- Multi-agent settings with competing reward sheaves
- Continual learning of black holes

### 8.3 Broader Impact
- Interpretable safety metrics (topological)
- Principled handling of evaluator disagreement
- Potential misuse: could be used to circumvent safety

---

## 9. Conclusion (0.5 pages)

Summary of contributions:
1. First sheaf-theoretic framework for RLHF rewards
2. H¹ cohomology detects cyclic preferences
3. Geometric safety via black hole avoidance
4. SGPO algorithm with empirical validation

The paper establishes foundations for topological AI safety.

---

## Appendix

### A. Proofs
- Theorem 3.1 (Consistency ⟺ H¹ = 0)
- Theorem 4.2 (Geodesic Safety)
- Proposition 5.1 (Hodge Advantage)

### B. Implementation Details
- Network architectures
- Hyperparameters
- Compute requirements

### C. Additional Experiments
- Extended ablations
- Failure cases
- Visualization gallery

### D. Sheaf Theory Primer
- Gentle introduction for ML audience
- Key definitions with intuition
- Connection to existing ML concepts

---

## Figures

1. **Figure 1**: System architecture diagram (from RESEARCH_PROPOSAL.md)
2. **Figure 2**: Condorcet cycle visualization (reward on circle, H¹ detection)
3. **Figure 3**: Black hole navigation (PPO/CPO/SGPO trajectories)
4. **Figure 4**: Learned Riemannian metric heatmap
5. **Figure 5**: LLM style space cycling
6. **Figure 6**: Ablation study bar charts

---

## Target Venues

| Venue | Deadline | Page Limit | Fit |
|-------|----------|------------|-----|
| **ICML 2025** | Jan 30, 2025 | 8+unlimited appendix | ⭐⭐⭐ Primary |
| NeurIPS 2025 | May 22, 2025 | 9+unlimited appendix | ⭐⭐⭐ Backup |
| RLC 2025 | ~Feb 2025 | 8 | ⭐⭐ Good fit |
| TAG-ML Workshop | ~May 2025 | 4 | ⭐⭐ Theory focus |

---

## Writing Timeline (for ICML Jan 30)

| Week | Focus |
|------|-------|
| Dec 30 - Jan 5 | Complete experiments, generate all figures |
| Jan 6 - Jan 12 | Write Methods (Sections 3-5), Related Work |
| Jan 13 - Jan 19 | Write Experiments, Introduction, Abstract |
| Jan 20 - Jan 26 | Polish, internal review, appendix |
| Jan 27 - Jan 30 | Final revisions, submission |
