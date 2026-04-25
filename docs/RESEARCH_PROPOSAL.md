# Modular Safe RLHF: Discrete HodgeRank + Conformal Safety

> **⚠️ REVISION NOTICE (February 2026)**: This document is being revised to correct
> categorical conflations between discrete topology and continuous Riemannian geometry.
> See `handoffs/14_MATHEMATICAL_RESTRUCTURING.md` for the corrected 3-module framework.
> Key corrections:
> - **Curl ≠ Curvature**: Discrete curl is a coboundary operator, not a curvature metric
> - **Harmonic = Bad**: Harmonic flow represents Condorcet paradoxes to DISCARD, not preserve
> - **Module Separation**: Discrete HodgeRank (graphs) vs Conformal Safety (manifolds)

## Abstract

We propose a novel framework for reinforcement learning from human feedback (RLHF) that embeds human evaluations into high-dimensional reward spaces and applies sheaf-theoretic methods to detect inconsistencies, quantify risk, and guide safe policy optimization. Traditional scalar reward functions collapse the rich structure of human preferences, creating vulnerabilities to reward hacking and specification gaming. Our approach models human feedback as sections of a sheaf over the trajectory space, where **cohomological invariants** measure the consistency of local evaluations and their compatibility with global objectives. We introduce the concept of **"black hole" regions** — forbidden outcome manifolds modeled as singularities in the reward space — and formulate policy optimization as geodesic navigation that avoids these dangerous regions. This framework provides (1) a principled method for filtering inconsistent training data, (2) real-time risk quantification during deployment, (3) interpretable safety guarantees via topological invariants, and (4) a unified treatment of multi-evaluator disagreement. We outline theoretical foundations, propose a practical architecture, and discuss connections to existing work in distributional RL, multi-objective optimization, and topological data analysis.

---

## 1. Introduction

### 1.1 The Problem with Scalar Rewards

Reinforcement learning from human feedback (RLHF) has become the dominant paradigm for aligning large language models and autonomous agents with human preferences. The standard approach trains a **reward model** R(s,a) → ℝ from human preference data, then optimizes a policy to maximize expected cumulative reward.

This approach has fundamental limitations:

1. **Information Loss**: Human preferences are multi-dimensional (helpfulness, harmlessness, honesty, creativity, etc.), but scalar rewards collapse these into a single number.

2. **Reward Hacking**: Agents exploit inconsistencies in the reward function to achieve high scores without satisfying the intended objective.

3. **Evaluator Disagreement**: Different humans have different preferences, but scalar aggregation obscures where and why they disagree.

4. **Temporal Inconsistency**: Per-step rewards may not compose into coherent trajectory-level evaluations.

5. **No Safety Guarantees**: There is no formal mechanism to ensure policies avoid catastrophically bad outcomes.

### 1.2 Our Contribution

We propose **Sheaf-Theoretic Reward Spaces (STRS)**, a framework that addresses these limitations by:

1. **Embedding rewards in high-dimensional space**: Human evaluations are mapped to ℝᵈ, preserving geometric structure of preferences.

2. **Modeling feedback as a sheaf**: Local evaluations (per-step, per-segment) must satisfy gluing conditions to form consistent global rewards.

3. **Using cohomology for consistency detection**: H¹ ≠ 0 signals that local evaluations cannot be reconciled — a red flag for training data quality or reward specification.

4. **Introducing "black hole" regions**: Forbidden outcomes are modeled as singularities in the reward manifold, with event horizons that policies must avoid.

5. **Sheaf-Geodesic Policy Optimization**: Optimal policies follow geodesics in reward space that navigate around dangerous regions.

### 1.3 Why Sheaf Theory?

Sheaf theory is the natural mathematical language for this problem because it formalizes:

- **Local-to-global consistency**: When do local observations determine a unique global object?
- **Obstruction theory**: What prevents local data from gluing together?
- **Multi-scale structure**: How do fine-grained and coarse-grained descriptions relate?

These are precisely the questions we face in RLHF: When do per-step rewards compose into coherent trajectory rewards? When do different evaluators' feedback reconcile? What prevents a reward function from being "well-defined"?

---

## 2. Mathematical Framework

### 2.1 Preliminaries

**Definition 2.1 (Trajectory Space).** Let S be a state space and A an action space. A *trajectory* of length T is a sequence τ = ((s₀,a₀), (s₁,a₁), ..., (sₜ,aₜ)). The *trajectory space* is T = ⋃ₜ (S × A)ᵀ.

**Definition 2.2 (Evaluation Space).** An *evaluation* is an element of a space E that encodes human feedback. This may include:
- Preference pairs: (τ₁ ≻ τ₂)
- Scalar ratings: r ∈ [0,1]
- Natural language: l ∈ Σ*
- Categorical labels: c ∈ {good, bad, harmful, ...}

**Definition 2.3 (Reward Embedding).** A *reward embedding* is a function φ: T × E → ℝᵈ that maps trajectory-evaluation pairs to a d-dimensional reward space.

### 2.2 The Reward Sheaf

**Definition 2.4 (Open Cover of Trajectory Space).** We define a natural open cover of T based on temporal granularity:
- Uₛₜₑₚ = {single (s,a) pairs}
- Uₛₑₘ = {contiguous segments of k steps}
- Uₜᵣₐⱼ = {full trajectories}

These form a **poset** under inclusion: Uₛₜₑₚ ⊂ Uₛₑₘ ⊂ Uₜᵣₐⱼ

**Definition 2.5 (Reward Presheaf).** A *reward presheaf* F assigns to each open set U a vector space F(U) ⊆ ℝᵈ of possible reward embeddings, together with *restriction maps* ρᵥᵤ: F(V) → F(U) for U ⊆ V that describe how coarse-grained rewards decompose into fine-grained rewards.

**Definition 2.6 (Reward Sheaf).** A reward presheaf F is a *sheaf* if it satisfies:

1. **Locality**: If two sections s, t ∈ F(U) agree on every element of an open cover {Uᵢ} of U, then s = t.

2. **Gluing**: If {sᵢ ∈ F(Uᵢ)} is a collection of local sections that agree on overlaps (sᵢ|ᵤᵢ∩ᵤⱼ = sⱼ|ᵤᵢ∩ᵤⱼ), then there exists a unique global section s ∈ F(U) with s|ᵤᵢ = sᵢ.

**Interpretation**: The sheaf condition says that if local reward assignments are mutually consistent, they determine a unique global reward. Violations indicate fundamental inconsistencies in the feedback.

### 2.3 Cohomology and Consistency

**Definition 2.7 (Čech Cohomology).** Given a sheaf F and open cover U = {Uᵢ}, the *Čech cohomology groups* are:

- **H⁰(U, F)** = Global sections (rewards consistent across all scales)
- **H¹(U, F)** = Obstructions to gluing (inconsistencies between local evaluations)

**Theorem 2.1 (Consistency Criterion).** *Human feedback is globally consistent if and only if H¹ = 0.*

**Proof Sketch**: H¹ measures the failure of local sections to glue. If H¹ ≠ 0, there exist local reward assignments that agree on overlaps but cannot come from any global reward function. □

**Corollary 2.2 (Inconsistency Localization).** *The support of H¹ identifies the specific trajectory regions where evaluations are inconsistent.*

### 2.4 The Reward Manifold and Black Holes

**Definition 2.8 (Reward Manifold).** The *reward manifold* M is the image of the trajectory space under the reward embedding: M = φ(T × E) ⊆ ℝᵈ, equipped with the induced Riemannian metric from ℝᵈ.

**Definition 2.9 (Black Hole Region).** A *black hole* B ⊂ M is a forbidden region characterized by:
- **Center**: c ∈ ℝᵈ (prototype of forbidden behavior)
- **Event horizon**: r > 0 (boundary of no return)
- **Severity**: σ ∈ ℝ₊ ∪ {∞} (depth of singularity)

The *potential function* near a black hole is:

$$\Phi(x) = \begin{cases} -\infty & \text{if } \|x - c\| < r \\ -\frac{\sigma}{(\|x - c\| - r)^2} & \text{if } \|x - c\| \geq r \end{cases}$$

**Definition 2.10 (Safe Reward Function).** The *safe reward function* is:

$$R_{\text{safe}}(τ) = R_{\text{base}}(φ(τ)) + \sum_{B \in \mathcal{B}} \Phi_B(φ(τ))$$

where R_base is the learned reward model and B is the set of known black holes.

### 2.5 Sheaf-Sheaf-Geodesic Policy Optimization

**Definition 2.11 (Geodesic).** A *geodesic* on the reward manifold M is a curve γ: [0,1] → M that locally minimizes arc length. In the presence of black holes, geodesics curve around forbidden regions.

**Theorem 2.3 (Safe Policy Existence).** *If the reward manifold M \ ⋃B (manifold minus black holes) is path-connected, then there exists a policy that reaches any target reward region while avoiding all black holes.*

**Definition 2.12 (Sheaf-Sheaf-Geodesic Policy Optimization).** The optimal safe policy solves:

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi\left[\int_0^T R_{\text{safe}}(τ_t) dt\right]$$

subject to: $\min_t d(φ(τ_t), \mathcal{B}) > 0$ (stay outside all event horizons)

### 2.6 Semantic Markov Decision Process (S-MDP)

We formally extend the classical MDP tuple $(S, A, P, R)$ to a **Semantic SAPR Tuple** grounded in causal reasoning and natural language.

1.  **Semantic State Space ($S$)**: States are **Kolmogorov Minimal Descriptions**—the shortest natural language strings that preserve causal sufficiency. Unlike raw tensors, these states reside on a "meaning manifold" accessible to LLMs.
    $$ s_t = \arg\min_{d} \{ L(d) : P(\text{outcome} | d, a) \approx P(\text{outcome} | \text{history}, a) \} $$

2.  **Structured Action Space ($A$)**: Actions are defined by the **Model Context Protocol (MCP)**, forming a structured topology of tool calls rather than a flat set of tokens.
    $$ a_t \in \{ \text{tool}(\text{args}) \mid \text{tool} \in \mathcal{T} \} $$

3.  **Stochastic Transition ($P$)**: The transition kernel $P(s'|s,a)$ represents the causal logic of the environment (simulated by an Oracle/World Model).
    $$ s_{t+1} \sim \text{WorldModel}(s_t, a_t) $$

4.  **Topological Reward Space ($R$)**: Pairwise preferences form edge flows on a **discrete** preference graph, decomposed via **discrete Hodge theory** (Module 1) into:
    - **Gradient (∇φ)**: Transitive consensus (Borda count) — **USE FOR TRAINING**
    - **Curl (δψ)**: Local cyclic inconsistencies in 3-cliques — **DISCARD**
    - **Harmonic (h)**: Global Condorcet paradoxes — **DISCARD**
    
    > ⚠️ **CORRECTION**: This is DISCRETE combinatorial Hodge theory on graphs, NOT continuous vector calculus on manifolds. The curl here is a coboundary operator, not a rotation.

### 2.7 Manifold Optimization: Beyond PPO and CPO

Standard Reinforcement Learning relies on **Markov Chains**—sequences of states where the future depends only on the present. Modern optimization techniques like **Proximal Policy Optimization (PPO)** work by enforcing a **Trust Region** on the policy update:
$$ \max_\theta \mathbb{E} \left[ \frac{\pi_\theta}{\pi_{\text{old}}} A \right] \quad \text{s.t.} \quad KL(\pi_{\text{old}} || \pi_\theta) < \delta $$
Geometrically, PPO performs descent on the statistical manifold of policies, using the Fisher Information Metric to prevent catastrophic updates.

**Constrained Policy Optimization (CPO)** extends this to safety by adding cost constraints:
$$ \max J(\pi) \quad \text{s.t.} \quad J_{\text{cost}}(\pi) \le C $$
However, CPO suffers from the **"Means Justify the Ends"** problem: it satisfies constraints *in expectation*. A policy can be 99% safe and 1% catastrophic, still satisfying the constraint. It lacks a topological barrier.

**Our Approach: Modular Safe RLHF (CORRECTED)**

> ⚠️ **CRITICAL**: We now separate this into TWO mathematically distinct modules:

**Module 1: Discrete HodgeRank (Reward Model Training)**
- Domain: Discrete simplicial complex (preference graph)
- Apply discrete Helmholtz-Hodge decomposition to extract ONLY the gradient (transitive) component
- Train reward model exclusively on gradient flow, discarding curl and harmonic
- This eliminates cyclic preferences that enable reward hacking

**Module 2: Conformal Safety Metric (Policy Optimization)**
- Domain: Continuous latent embedding space
- Define conformal metric: $g_{ij}(x) = e^{2\sigma(x)} \delta_{ij}$ where $\sigma(x) \to \infty$ at danger boundary
- Geodesic distance to danger diverges to infinity → **geometric unreachability**
- Natural policy gradient preconditioned by $G^{-1}$ automatically suppresses movement toward danger

**Key Insight**: These are SEPARATE mathematical domains. Do not conflate discrete topology (Module 1) with continuous Riemannian geometry (Module 2).

---

## 3. Proposed Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FEEDBACK COLLECTION                              │
│  Human evaluations at multiple scales (step, segment, trajectory)   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    REWARD EMBEDDING                                 │
│  φ: (trajectory, evaluation) → ℝᵈ                                   │
│  • Contrastive learning on preference pairs                         │
│  • Language encoder for textual feedback                            │
│  • Temporal attention for trajectory structure                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SHEAF CONSTRUCTION                               │
│  • Learn restriction maps ρ: F(coarse) → F(fine)                    │
│  • Compute local sections at each scale                             │
│  • Build cochain complex for cohomology                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    COHOMOLOGY ENGINE                                │
│  • Compute H⁰ (consistent reward signal)                            │
│  • Compute H¹ (inconsistency measure)                               │
│  • Localize inconsistencies to trajectory regions                   │
│  • Flag data for review or exclusion                                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                        ┌───────┴───────┐
                        ▼               ▼
┌───────────────────────────┐ ┌─────────────────────────────────────┐
│   BLACK HOLE DETECTION    │ │   TRAINING DATA CURATION            │
│  • Cluster harm reports   │ │  • Filter by H¹ < threshold         │
│  • Learn event horizons   │ │  • Weight by consistency score      │
│  • Update forbidden set   │ │  • Exclude black hole trajectories  │
└───────────────────────────┘ └─────────────────────────────────────┘
                        │               │
                        └───────┬───────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    POLICY OPTIMIZATION                              │
│  • Geodesic optimization in reward space                            │
│  • Black hole avoidance constraints                                 │
│  • Multi-objective Pareto optimization                              │
│  • Uncertainty-aware exploration                                    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RUNTIME MONITORING                               │
│  • Real-time trajectory embedding                                   │
│  • Distance to nearest black hole                                   │
│  • Cohomology-based anomaly detection                               │
│  • Intervention triggers                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Components

#### 3.2.1 Reward Embedding Network

```python
class RewardEmbedding(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim=256):
        self.trajectory_encoder = TransformerEncoder(...)  # Temporal structure
        self.evaluation_encoder = T5Encoder(...)           # Language feedback
        self.preference_encoder = SiameseNetwork(...)      # Pairwise preferences
        self.projection = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, trajectory, evaluation):
        τ_embed = self.trajectory_encoder(trajectory)
        e_embed = self.evaluation_encoder(evaluation)
        joint = torch.cat([τ_embed, e_embed], dim=-1)
        return self.projection(joint)
```

#### 3.2.2 Sheaf Consistency Checker

```python
class SheafConsistency:
    def __init__(self, restriction_networks: Dict[str, nn.Module]):
        self.ρ = restriction_networks  # Learned restriction maps
    
    def compute_cohomology(self, sections: Dict[str, Tensor]) -> CohomologyResult:
        # C⁰: trajectory-level sections
        C0 = sections['trajectory']
        
        # C¹: (segment - restricted_trajectory, step - restricted_segment)
        C1_seg = sections['segment'] - self.ρ['traj→seg'](C0)
        C1_step = sections['step'] - self.ρ['seg→step'](sections['segment'])
        C1 = torch.cat([C1_seg, C1_step], dim=-1)
        
        # H¹ = ||C1||² (simplified; full version uses kernel/image)
        h1_magnitude = torch.norm(C1, dim=-1)
        
        return CohomologyResult(h0=C0, h1_magnitude=h1_magnitude)
```

#### 3.2.3 Black Hole Manager

```python
class BlackHoleManager:
    def __init__(self, embed_dim: int):
        self.black_holes: List[BlackHole] = []
        self.clustering = HDBSCAN(min_cluster_size=5)
    
    def update_from_harm_reports(self, reports: List[HarmReport], embedder: RewardEmbedding):
        embeddings = [embedder(r.trajectory, r.evaluation) for r in reports]
        severities = [r.severity for r in reports]
        
        # Cluster into black hole regions
        clusters = self.clustering.fit_predict(embeddings)
        
        for cluster_id in set(clusters):
            if cluster_id == -1:
                continue  # Noise
            mask = clusters == cluster_id
            center = np.mean(embeddings[mask], axis=0)
            radius = np.max(np.linalg.norm(embeddings[mask] - center, axis=1))
            severity = np.max(severities[mask])
            
            self.black_holes.append(BlackHole(center, radius * 1.5, severity))
    
    def safe_reward(self, base_reward: float, embedding: np.ndarray) -> float:
        for bh in self.black_holes:
            dist = np.linalg.norm(embedding - bh.center)
            if dist < bh.event_horizon:
                return -np.inf
            base_reward -= bh.severity / (dist - bh.event_horizon + 1e-6)**2
        return base_reward
```

---

## 4. Theoretical Analysis

### 4.1 Consistency Guarantees

**Theorem 4.1 (Data Quality Bound).** *Let D be a dataset of trajectory-evaluation pairs with cohomology H¹(D). The generalization error of a reward model trained on D is bounded by:*

$$\epsilon_{\text{gen}} \leq \epsilon_{\text{train}} + C \cdot \|H^1(D)\| + O(1/\sqrt{n})$$

*where C is a constant depending on the embedding dimension.*

**Interpretation**: Inconsistent training data (high H¹) leads to worse generalization. Filtering by cohomology improves model quality.

### 4.2 Safety Guarantees

**Theorem 4.2 (Black Hole Avoidance).** *Let π be a policy trained with black hole constraints. If the reward manifold M has bounded curvature κ and all black holes have event horizon r > 2/κ, then with probability 1 - δ:*

$$\Pr_\pi[\text{trajectory enters black hole}] \leq \delta$$

**Interpretation**: Sufficiently large event horizons provide probabilistic safety guarantees.

### 4.3 Computational Complexity

**Proposition 4.3.** *Computing H¹ for a trajectory of length T with k evaluators requires O(T · k · d²) operations, where d is the embedding dimension.*

This is tractable for typical RLHF settings (T ~ 1000, k ~ 10, d ~ 256).

---

## 5. Connections to Existing Work

### 5.1 Distributional RL

Our framework extends distributional RL (Bellemare et al., 2017) from scalar to vector-valued returns. The key difference is that we model the *geometric structure* of the reward space, not just the distribution of scalar returns.

**Related Work**:
- MD3QN (Zhang et al., 2021): Multi-dimensional distributional DQN
- Off-policy RL with high-dimensional reward (2024): Banach space rewards

### 5.2 Multi-Objective RL

Multi-objective RL (MORL) optimizes over Pareto frontiers of multiple objectives. Our approach subsumes MORL by embedding objectives as dimensions of the reward space, but adds:
- Consistency checking via cohomology
- Forbidden region constraints
- Geometric structure preservation

### 5.3 Topological Data Analysis

We leverage tools from TDA, particularly:
- **Persistent homology**: Detecting topological features of the reward landscape
- **Sheaf cohomology**: Measuring local-global consistency
- **Morse theory**: Analyzing critical points (optima, saddles) of the reward function

### 5.4 Safe RL

Existing safe RL methods use:
- **Constrained MDPs**: Hard constraints on expected cost
- **Risk-sensitive objectives**: CVaR, worst-case optimization
- **Shielding**: Runtime intervention

Our black hole formalism provides a geometric interpretation of safety constraints and enables continuous risk quantification.

### 5.5 Natural Language State Machines & Sequence Models

Recent work has begun to bridge the gap between language modeling and reinforcement learning state representations:

1.  **Sequence Modeling in RL**: The **Decision Transformer** (Chen et al., 2021) and **Decision Mamba** (Ota et al., 2024) cast RL as a conditional sequence modeling problem. While they leverage the architecture of LLMs/SSMs, they typically operate on tokenized sequences of numerical states or discrete codes. Our framework extends this by treating the *state itself* as a semantic object (natural language description), not just the model architecture.

2.  **LLM-Empowered State Representation (LESR)**: Wang et al. (2024) propose using LLMs to enrich sparse numerical states with world knowledge. Our **Semantic State Machine** takes this further: instead of enriching a numerical state, we replace it entirely with a **Kolmogorov minimal narrative**, asserting that the semantic description is the primary causal object.

3.  **State-Driven Workflows**: **StateFlow** (Wu et al., 2024) models LLM agents as explicit finite state machines (FSMs) to improve reliability. Our approach generalizes this to a **Semantic Turing Machine**, where the "state" is not a fixed node in a graph but a dynamic point on a continuous belief manifold, allowing for open-ended evolution while preserving topological structure via Hodge theory.

---

## 6. Experimental Plan

We validate the Sheaf-Theoretic Reward Spaces (STRS) framework using the **Storytelling Machine** paradigm—a controlled yet semantically rich environment that serves as a proxy for general causal reasoning.

### 6.1 Environment: The Semantic Turing Machine
We utilize a text-adventure style environment where the state is a **Kolmogorov minimal natural language description** and actions are **MCP tool calls**.
- **Oracle**: GPT-4o simulates world dynamics $P(s'|s,a)$.
- **Agent**: Small LLMs (Llama-3-8B, Mistral-7B) act as the policy $\pi_\theta$.
- **Task**: Goal-directed navigation in a partially observable semantic world (e.g., "Diagnose and fix the server outage" or "Escape the dungeon").

### 6.2 Experiment 1: Manifold Reconstruction from Feedback
**Hypothesis**: High-dimensional embeddings of belief states + Hodge decomposition can recover the latent reward structure better than scalar reward modeling.
- **Protocol**: Collect (state, action, critique) triples. Embed using sentence-transformers. Compute Hodge decomposition.
- **Metric**: **Topological Fidelity**—does the recovered gradient field $\nabla \phi$ match the ground-truth causal distance to the goal?
- **Baseline**: Standard scalar Reward Model (RM) trained on the same preferences.

### 6.3 Experiment 2: Safety via Black Holes
**Hypothesis**: Geodesic optimization with black hole constraints significantly reduces catastrophic failures compared to soft cost penalties (CPO/PPO-Lagrangian).
- **Setup**: Define semantic "trap" states (e.g., "System data deleted", "Dead end").
- **Method**: Identify these regions as topological black holes. Train policy using Geodesic DPO.
- **Metric**: **Survival Rate** (percentage of trajectories avoiding traps) vs. Task Success.

### 6.4 Experiment 3: Consistency Checking
**Hypothesis**: The cohomology group $H^1$ correlates with "reward hacking" opportunities.
- **Setup**: Introduce cyclical preferences in the Oracle feedback (A > B > C > A).
- **Analysis**: Measure $\|H^1\|$ magnitude. Verify if filtering high-$H^1$ regions improves policy robustness.

### 6.5 Ablation: Structure of the State
Compare **Semantic States** (our approach) vs. **Token Sequences** (Decision Transformer style) to demonstrate the data efficiency of using causally condensed narrative states.

---

## 7. Broader Impact

### 7.1 Positive Impacts

- **Interpretable Safety**: Topological invariants provide auditable safety metrics
- **Evaluator Fairness**: Cohomology identifies outlier evaluators without silencing minority views
- **Reduced Reward Hacking**: Geometric structure makes exploitation harder

### 7.2 Risks and Mitigations

- **Computational Cost**: Cohomology computation adds overhead; mitigate with efficient approximations
- **Black Hole Misspecification**: Incorrectly placed black holes could prevent beneficial behaviors; require human review of detected regions
- **Complexity**: Framework requires mathematical sophistication; provide accessible tooling

---

## 8. Conclusion

We have proposed Sheaf-Theoretic Reward Spaces (STRS), a framework that brings the power of algebraic topology to reinforcement learning from human feedback. By modeling rewards as sections of a sheaf and forbidden outcomes as singularities in a reward manifold, we obtain principled methods for consistency checking, risk quantification, and safe policy optimization. This work opens new directions at the intersection of topology, machine learning, and AI safety.

---

## References

See [BIBLIOGRAPHY.md](../references/BIBLIOGRAPHY.md) for full reference list.

---

## Appendix A: Sheaf Theory Primer

See [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) for accessible introduction to prerequisites.

## Appendix B: Proofs

*To be completed.*

## Appendix C: Implementation Details

*To be completed.*

---

## Appendix D: Process-Level Anomaly Detection

### D.1 Motivation: Why Outcome Feedback Is Insufficient

Standard RLHF collects feedback on **outcomes** (full trajectories). This creates fundamental blind spots for topological anomalies that manifest in the **process**:

| Anomaly | Why Outcome Feedback Fails |
|---------|---------------------------|
| **Wormhole** | Shortcut and legitimate path have identical outcomes |
| **Cliff** | Outcome is catastrophic, but *which step* caused it? |
| **Plateau** | Outcome may be acceptable despite wasted effort |

**Thesis**: Detecting topological anomalies requires **process supervision**—evaluation of the temporal structure of behavior, not just its endpoint.

### D.2 Formal Framework: Value Compositionality

**Definition D.1 (Step Value Function).** For a trajectory τ = (s₀,a₀), ..., (sₜ,aₜ), the *step value function* v: S × A → ℝᵈ assigns a reward embedding to each step.

**Definition D.2 (Compositional Assumption).** A reward function R is *compositional* if there exists an aggregation operator ⊕ such that:

$$R(τ) = v(s_0, a_0) ⊕ v(s_1, a_1) ⊕ \cdots ⊕ v(s_T, a_T)$$

For standard RL, ⊕ is discounted summation: R(τ) = Σᵢ γⁱ v(sᵢ, aᵢ).

**Definition D.3 (Compositionality Residual).** The *compositionality residual* measures deviation from compositional value:

$$\Delta(τ) = R_{\text{outcome}}(τ) - \bigoplus_{t=0}^{T} v(s_t, a_t)$$

**Theorem D.1 (Anomaly Signatures).** *Topological anomalies manifest as characteristic patterns in Δ(τ) and its temporal structure:*

| Anomaly | Signature |
|---------|-----------|
| **Wormhole** | Δ(τ) >> 0 (outcome exceeds process) |
| **Cliff** | Δ(τ) << 0 with localized step having v(sₜ, aₜ) << 0 |
| **Plateau** | Δ(τ) ≈ 0 but dv/dt ≈ 0 for extended interval |
| **Black Hole** | R_outcome(τ) = -∞ (absorbing state) |

### D.3 Process Supervision Feedback Design

To detect anomalies, we propose a **multi-scale feedback form**:

#### D.3.1 Trajectory-Level (Standard)
- Overall quality rating (1-5)
- Preference comparison to alternatives
- Harm/safety checkboxes

#### D.3.2 Step-Level (Process Supervision)
- **Progress indicator**: "At step t, how much closer was the agent to the goal?" (+1/0/-1)
- **Discontinuity flag**: "Did something suddenly go wrong/right here?" (boolean)
- **Effort-outcome ratio**: "Was this step productive or wasteful?"

#### D.3.3 Counterfactual Probes (Anomaly-Specific)

**Wormhole Detection**:
> "The agent achieved [outcome] via [shortcut path]. If it had instead taken [canonical path], would the outcome be:
> (a) Equivalent  (b) Worse  (c) Better but slower  (d) This shortcut is problematic"

**Cliff Detection**:
> "At step [t], the agent did [action]. If it had instead done [alternative], the outcome would be:
> (a) Similar  (b) Much better  (c) Much worse"
> 
> (Large (b)/(c) asymmetry indicates cliff proximity)

**Plateau Detection**:
> "From step [t₁] to [t₂], the agent appeared to make little progress. Was this:
> (a) Necessary exploration  (b) Stuck/confused  (c) Gaming the system  (d) Other"

### D.4 Mathematical Formalization

#### D.4.1 Wormhole as Sheaf Non-Triviality

A wormhole exists when two trajectories τ₁, τ₂ with the same endpoints (s₀ → sₜ) have:
- Different step-level sections: F(τ₁) ≠ F(τ₂) 
- Same outcome-level section: ρ(F(τ₁)) = ρ(F(τ₂))

This is a **non-trivial loop** in trajectory space that becomes trivial under the restriction map—precisely the failure of the sheaf to be a "local homeomorphism."

**Detection**: Compute the fiber F⁻¹(outcome) over each outcome. If |F⁻¹(outcome)| > 1 with qualitatively different processes, flag as wormhole candidate.

#### D.4.2 Cliff as Discontinuity in Restriction Map

A cliff at step t exists when:

$$\left\| \frac{\partial}{\partial a_t} \rho_{\text{step→outcome}}(v(s_t, a_t)) \right\| \to \infty$$

The restriction map has unbounded derivative—small changes in step-level value cause large changes in outcome.

**Detection**: Collect step-level and outcome-level ratings for trajectory pairs that differ only at step t. If outcome variance >> step variance, flag as cliff.

#### D.4.3 Plateau as Kernel of Progress Operator

Define the *progress operator* P: F(step) → ℝ that measures instantaneous goal approach:

$$P(v_t) = \langle v_t, \nabla_\tau V^* \rangle$$

where V* is the optimal value function. A plateau is a region where P ≈ 0 for extended periods.

**Detection**: Track cumulative progress ∫P dt. If this flatlines while steps accumulate, flag as plateau.

### D.5 Implementation: Anomaly-Aware Training Pipeline

```python
class AnomalyAwareRewardLearning:
    def __init__(self, embed_dim: int):
        self.outcome_model = RewardEmbedding(embed_dim)  # τ → ℝᵈ
        self.step_model = StepValueNetwork(embed_dim)     # (s,a) → ℝᵈ
        self.aggregator = LearnedAggregator(embed_dim)    # [v₁,...,vₜ] → ℝᵈ
        
    def compute_residual(self, trajectory, outcome_rating, step_ratings):
        """Compute compositionality residual Δ(τ)."""
        # Outcome embedding
        r_outcome = self.outcome_model(trajectory, outcome_rating)
        
        # Aggregated step embeddings
        step_values = [self.step_model(s, a, r) 
                       for (s,a), r in zip(trajectory, step_ratings)]
        r_process = self.aggregator(step_values)
        
        return r_outcome - r_process
    
    def detect_anomalies(self, trajectories, outcome_ratings, step_ratings):
        anomalies = {'wormholes': [], 'cliffs': [], 'plateaus': []}
        
        for τ, r_out, r_steps in zip(trajectories, outcome_ratings, step_ratings):
            Δ = self.compute_residual(τ, r_out, r_steps)
            
            # Wormhole: outcome >> process
            if torch.norm(Δ) > self.wormhole_threshold and Δ.mean() > 0:
                anomalies['wormholes'].append(WormholeCandidate(τ, Δ))
            
            # Cliff: find step with max negative impact
            step_impacts = self.compute_step_impacts(τ, r_steps)
            cliff_idx = torch.argmin(step_impacts)
            if step_impacts[cliff_idx] < self.cliff_threshold:
                anomalies['cliffs'].append(CliffCandidate(τ, cliff_idx, step_impacts[cliff_idx]))
            
            # Plateau: extended period of no progress
            progress = self.compute_progress_curve(τ, r_steps)
            plateau_regions = find_flat_regions(progress, min_length=5)
            for start, end in plateau_regions:
                anomalies['plateaus'].append(PlateauCandidate(τ, start, end))
        
        return anomalies
    
    def compute_step_impacts(self, trajectory, step_ratings):
        """Compute marginal impact of each step on outcome."""
        impacts = []
        for t in range(len(trajectory)):
            # Counterfactual: what if step t were average?
            modified_ratings = step_ratings.copy()
            modified_ratings[t] = self.average_step_rating
            
            original_outcome = self.aggregator(
                [self.step_model(s, a, r) for (s,a), r in zip(trajectory, step_ratings)])
            modified_outcome = self.aggregator(
                [self.step_model(s, a, r) for (s,a), r in zip(trajectory, modified_ratings)])
            
            impacts.append(torch.norm(original_outcome - modified_outcome))
        
        return torch.stack(impacts)
```

### D.6 Feedback Form Implementation

```python
@dataclass
class ProcessSupervisionForm:
    """Multi-scale feedback for anomaly detection."""
    
    # === Trajectory Level ===
    trajectory_id: str
    overall_quality: int  # 1-5
    safety_flags: List[str]  # ["harmful", "deceptive", ...]
    
    # === Step Level ===
    step_progress: List[int]  # +1/0/-1 per step
    discontinuity_steps: List[int]  # indices where quality suddenly changed
    
    # === Anomaly Probes ===
    # Wormhole
    shortcut_detected: bool
    shortcut_assessment: Optional[Literal[
        "legitimate_efficiency",
        "gaming_the_system",
        "violates_spirit",
        "unsure"
    ]]
    
    # Cliff
    critical_step: Optional[int]  # Step that "broke everything"
    critical_step_alternatives: Optional[str]  # What should it have done?
    
    # Plateau
    spinning_wheels_start: Optional[int]
    spinning_wheels_end: Optional[int]
    spinning_assessment: Optional[Literal[
        "necessary_exploration",
        "stuck_confused",
        "stalling",
        "other"
    ]]


def compute_anomaly_labels(forms: List[ProcessSupervisionForm]) -> AnomalyLabels:
    """Aggregate human feedback into anomaly labels."""
    labels = AnomalyLabels()
    
    for form in forms:
        τ = form.trajectory_id
        
        # Wormhole consensus
        if form.shortcut_detected:
            labels.wormhole_votes[τ].append(form.shortcut_assessment)
        
        # Cliff localization
        if form.critical_step is not None:
            labels.cliff_votes[τ].append(form.critical_step)
        
        # Plateau detection
        if form.spinning_wheels_start is not None:
            labels.plateau_votes[τ].append(
                (form.spinning_wheels_start, form.spinning_wheels_end))
    
    # Aggregate with majority vote / clustering
    labels.finalize()
    return labels
```

### D.7 Theoretical Guarantees

**Theorem D.2 (Wormhole Identifiability).** *Given counterfactual feedback comparing shortcut and canonical paths, wormholes are identifiable with sample complexity O(|Π| · log(1/δ)) where |Π| is the number of distinct path classes.*

**Theorem D.3 (Cliff Localization).** *Given step-level discontinuity flags from k evaluators, a cliff at step t can be localized with probability 1 - δ if at least (k+1)/2 evaluators flag step t, requiring O(k/ε²) samples for ε-precision.*

**Theorem D.4 (Plateau Detection).** *Plateaus of duration ≥ L steps are detectable with O(T/L) progress measurements, where T is trajectory length.*

### D.8 Connection to Sheaf Cohomology

The anomaly detection framework integrates with the main STRS framework:

1. **Wormholes → Non-trivial H¹**: Multiple processes mapping to same outcome = non-unique section lifting = H¹ ≠ 0 over the outcome fiber.

2. **Cliffs → Singular restriction maps**: The sheaf condition requires restriction maps to be continuous. Cliffs are points where this fails—the presheaf is not a sheaf at cliff locations.

3. **Plateaus → Degenerate local sections**: Plateau regions have dim(F(U)) < expected, indicating reward space has collapsed locally.

**Corollary D.5.** *A complete anomaly catalog (black holes, wormholes, cliffs, plateaus) corresponds to a complete characterization of sheaf pathologies: singularities, non-trivial topology, discontinuous restrictions, and dimensional degeneracy.*
