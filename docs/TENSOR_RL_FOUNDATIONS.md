# Tensor Reinforcement Learning Foundations: From Bellman to Natural Language State Spaces

## Abstract

This document provides a foundational bridge between classical reinforcement learning theory (Bellman equations) and our novel framework for natural language state space evolution models. We translate the standard mathematical machinery into plain language probability statements, then develop proposals for training models that: (A) learn world state evolution as token sequences, (B) predict evolution pathways given actions, (C) optimize policies under constraints, and (D) assign consistent, perspective-invariant rewards with Condorcet cycle detection. Finally, we critically evaluate our current simulation suite against these principles.

---

## 1. The Tensor Flow of Reinforcement Learning

### 1.1 User's Core Insight (Annotated)

> "Current state tensor is arrived at through the previous state-tensor, which provided previous conditions to the state-transition-probability-given-action tensor, which provided predictions to the policy tensor, which gave the previous turn's commands to the action tensor, which then actuated the world environment to be perceived as the current state tensor."

This statement captures the **causal chain** of RL as a tensor computation graph:

```
s_{t-1} → P(s_t | s_{t-1}, a_{t-1}) → π(a_{t-1} | s_{t-1}) → a_{t-1} → Environment → s_t
   ↑                                                                                    │
   └────────────────────────────────────────────────────────────────────────────────────┘
```

> "The trajectory space (all of the states, actions, and predictions in a task session or game) is scored using discounted sum over all periods to populate the reward tensor, which is then numerically differentiated to train the policy tensor, which is what creates the trajectory space."

This captures the **feedback loop**: trajectories generate rewards, rewards shape policies via gradients, policies generate new trajectories.

---

## 2. Bellman Equations: Mathematical and Plain Language

### 2.1 The Bellman Expectation Equation

**Mathematical Form:**
$$V^\pi(s) = \mathbb{E}_\pi\left[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s\right]$$

**Plain Language Translation:**

> "The value of being in state $s$ and following policy $\pi$ equals the expected immediate reward plus the discounted value of wherever you end up next, averaged over all possible actions you might take and all possible states you might land in."

**Probability Statement:**
$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

> "For each action you might choose (weighted by your policy's probability of choosing it), and for each state you might transition to (weighted by the world's probability of sending you there), add up the immediate reward plus the future value."

### 2.2 The Bellman Optimality Equation

**Mathematical Form:**
$$V^*(s) = \max_a \mathbb{E}\left[R_{t+1} + \gamma V^*(S_{t+1}) \mid S_t = s, A_t = a\right]$$

**Plain Language Translation:**

> "The optimal value of a state is achieved by picking the action that maximizes expected immediate reward plus discounted future optimal value."

**Probability Statement:**
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

> "Consider every action. For each action, compute what you'd expect to get (immediate reward + future value) across all possible outcomes, weighted by their likelihood. The optimal value is the best of these."

### 2.3 The Q-Function (Action-Value)

**Mathematical Form:**
$$Q^\pi(s,a) = \mathbb{E}_\pi\left[R_{t+1} + \gamma Q^\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a\right]$$

**Plain Language Translation:**

> "The value of taking action $a$ in state $s$ and then following policy $\pi$ equals the expected immediate reward plus the discounted value of the next state-action pair."

**Probability Statement:**
$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]$$

> "Given you're in state $s$ and take action $a$: average over all possible next states (weighted by transition probability), and for each next state, average over all actions you might take there (weighted by your policy)."

### 2.4 The Policy Gradient Theorem

**Mathematical Form:**
$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)\right]$$

**Plain Language Translation:**

> "To improve the policy, nudge the parameters in the direction that makes high-value actions more likely and low-value actions less likely."

**Tensor Interpretation (User's Language):**

> "The reward tensor is numerically differentiated (via the policy gradient) to train the policy tensor."

This "numerical differentiation" is precisely the gradient $\nabla_\theta \log \pi_\theta(a|s)$ — the direction in parameter space that increases the probability of action $a$ — weighted by how good that action turned out to be ($Q^\pi(s,a)$).

---

## 3. Connection to High-Dimensional Sequence Embeddings

### 3.1 The Fundamental Shift

Traditional RL treats states as vectors $s \in \mathbb{R}^n$ or discrete symbols $s \in \mathcal{S}$. Our framework treats states as **natural language descriptions** embedded into high-dimensional token sequences via transformer/SSM architectures.

| Classical RL | Natural Language RL |
|--------------|---------------------|
| $s \in \mathbb{R}^n$ | $s \in \text{Embed}(\Sigma^*)$ — embedded text |
| $P(s'|s,a)$ as matrix | $P(s'|s,a)$ as LLM/SSM world model |
| $\pi(a|s)$ as softmax | $\pi(a|s)$ as autoregressive generation |
| $R: S \times A \to \mathbb{R}$ | $R: S \times A \to \mathbb{R}^d$ (vector field on manifold) |

### 3.2 Why This Matters

1. **Semantic Compositionality**: Natural language states compose semantically (word → sentence → paragraph), enabling transfer and generalization.

2. **Causal Sufficiency**: A well-formed state description contains exactly the information needed to predict outcomes — the **Kolmogorov minimal description**.

3. **Human-Interpretable**: States are readable, debuggable, and can receive natural language feedback.

4. **Topological Structure**: Embedding spaces have geometric structure we can exploit via Hodge theory and sheaf cohomology.

---

## 4. Proposals for Natural Language State Space Evolution Models

### 4.1 Proposal (A): World State Evolution → Token Sequences

**Goal**: Learn a model that represents the evolution of an environment/world and embeds it into a sequential state of tokens.

**Architecture: Semantic State Encoder (SSE)**

```
Input: Raw observation (text, image, sensor data)
Output: Minimal semantic state token sequence

Pipeline:
1. Perception Layer: Map raw input to feature representation
   - Text: Transformer encoder
   - Image: Vision encoder (ViT, CNN)
   - Multimodal: Cross-attention fusion

2. Compression Layer: Distill to Kolmogorov minimal description
   - Objective: Minimize description length while preserving predictive power
   - L_compress = L(s) + λ · L_pred(s' | s, a)
   - Where L(s) = token count, L_pred = prediction error

3. State Tokenization: Convert to discrete token sequence
   - Option A: VQ-VAE style discrete bottleneck
   - Option B: Direct text generation via small LLM
   - Option C: Hybrid (continuous embedding + text description)

4. Sequence Evolution: Autoregressively model state transitions
   s_t = f_θ(s_{t-1}, a_{t-1}, s_{t-2}, a_{t-2}, ...)
```

**Key Innovation**: The state is not a fixed-dimension vector but a **variable-length semantic description** that captures causal sufficiency.

**Training Objective**:
$$\mathcal{L}_{\text{SSE}} = \underbrace{\mathbb{E}[\|s'_{\text{pred}} - s'_{\text{true}}\|^2]}_{\text{prediction accuracy}} + \underbrace{\lambda_1 \cdot \text{len}(s)}_{\text{compression}} + \underbrace{\lambda_2 \cdot H^1(s)}_{\text{consistency}}$$

**Dataset Requirements**:
- Trajectories with rich state descriptions (not just numerical vectors)
- Multi-scale annotations: step, segment, trajectory
- Counterfactual branches where available

---

### 4.2 Proposal (B): Evolution Pathway Prediction

**Goal**: Learn a function that predicts evolution pathways of the world state sequence given past values and potential action tokens.

**Architecture: World Model Transformer (WMT)**

```
Input: 
  - History: [(s_0, a_0), (s_1, a_1), ..., (s_t, ?)]
  - Query action: a_t (or set of candidate actions)

Output:
  - Predicted next state distribution: P(s_{t+1} | history, a_t)
  - Uncertainty estimate: σ(s_{t+1})
  - Trajectory rollout: [(s_{t+1}, ?), (s_{t+2}, ?), ...] for planning
```

**Mathematical Framework**:

The world model learns the **transition kernel** in semantic space:
$$P(s_{t+1} | s_{\leq t}, a_{\leq t}) = \text{WMT}_\theta(s_{\leq t}, a_{\leq t})$$

For multiple candidate actions $\{a^{(1)}, a^{(2)}, ..., a^{(k)}\}$, we can compute:
$$\{P(s_{t+1}^{(i)} | s_{\leq t}, a^{(i)})\}_{i=1}^k$$

This enables **tree search** in semantic space:
- MCTS with LLM world model (similar to MuZero but with semantic states)
- Beam search over action-state trajectories
- Counterfactual reasoning: "What if I had taken action $a'$ instead?"

**Training Objective**:
$$\mathcal{L}_{\text{WMT}} = -\mathbb{E}_{(s,a,s') \sim \mathcal{D}}[\log P_\theta(s' | s, a)] + \lambda \cdot \text{KL}[P_\theta \| P_{\text{prior}}]$$

**Connection to Bellman**:

The world model directly provides the $P(s'|s,a)$ term in the Bellman equation:
$$Q(s,a) = \sum_{s'} \underbrace{P_{\text{WMT}}(s'|s,a)}_{\text{learned}} \left[ R(s,a,s') + \gamma V(s') \right]$$

**Dataset Requirements**:
- State-action-next_state tuples with semantic descriptions
- Diverse action spaces (not just single optimal trajectories)
- Failed trajectories (important for learning what doesn't work)

---

### 4.3 Proposal (C): Constrained Policy Optimization

**Goal**: Learn a policy that returns actions to maximize expected reward while satisfying given constraints.

**Architecture: Geodesic Policy Network (GPN)**

```
Input: Current semantic state s_t (embedded)
Output: 
  - Action distribution: π(a | s_t)
  - Value estimate: V(s_t)
  - Constraint satisfaction score: C(s_t)

Constraints encoded via:
  1. Hard geometric barriers (black holes in reward manifold)
  2. Soft Lagrangian penalties (expected cost constraints)
  3. Topological invariants (H¹ = 0 for consistency)
```

**Mathematical Framework**:

**Standard Constrained Optimization (CPO)**:
$$\max_\pi J(\pi) = \mathbb{E}_\pi\left[\sum_t \gamma^t R(s_t, a_t)\right]$$
$$\text{s.t. } J_C(\pi) = \mathbb{E}_\pi\left[\sum_t \gamma^t C(s_t, a_t)\right] \leq d$$

**Limitation**: This enforces constraints *in expectation*. A policy can be 99% safe and 1% catastrophic while still satisfying the constraint.

**Our Approach (SGPO)**:
$$\max_\pi J_{\text{Hodge}}(\pi) = \mathbb{E}_\pi\left[\sum_t \gamma^t \langle \nabla\phi(s_t), \Delta e_t \rangle\right]$$
$$\text{s.t. } d_g(s_t, \mathcal{B}) > 0 \quad \forall t \quad (\text{hard barrier})$$

Where:
- $\nabla\phi$ is the Hodge gradient (consistent reward direction)
- $\Delta e_t$ is the embedding change from action
- $d_g$ is geodesic distance on the Riemannian reward manifold
- $\mathcal{B}$ is the set of black hole (forbidden) regions

**Safety Guarantee**:

By constructing the metric $g(x) = \frac{1}{(d(x, \mathcal{B}))^\alpha}$ near black holes, geodesics cannot enter forbidden regions because the path length becomes infinite.

**Training Procedure**:
1. Collect trajectories with reward and constraint annotations
2. Embed states and compute Hodge decomposition
3. Identify black holes from negative feedback
4. Optimize policy via PPO with:
   - Reward = Hodge gradient alignment
   - Adaptive trust region based on $H^1$ and curvature
   - Hard rejection of trajectories entering black holes

---

### 4.4 Proposal (D): Consistent, Perspective-Invariant Reward Assignment

**Goal**: Learn to assign rewards that are consistent when feedback agrees, and appropriately variant when perspectives differ, with ability to detect Condorcet cycles.

**Architecture: Sheaf Reward Network (SRN)**

```
Input: 
  - Trajectory τ with multi-scale feedback (step, segment, trajectory)
  - Multi-evaluator annotations (different humans/perspectives)

Output:
  - Global reward: R(τ) ∈ ℝ^d (vector, not scalar)
  - Consistency score: H¹(τ) (cohomology magnitude)
  - Perspective decomposition: {R_i(τ)} for each evaluator
  - Cycle detection: List of Condorcet cycles if H¹ > 0
```

**Mathematical Framework**:

**Sheaf Construction**:
- Base space $X$: Trajectory graph (states as nodes, transitions as edges)
- Sheaf $\mathcal{F}$: Assigns reward vectors to each open set
- Restriction maps $\rho$: Define how trajectory rewards decompose to segments to steps

**Consistency Condition** (Sheaf Axiom):
$$\rho_{U \to U_i}(R(U)) = R(U_i) \quad \text{for all } U_i \subset U$$

> "The reward for a trajectory, when restricted to a sub-segment, equals the reward for that sub-segment."

**Cohomology as Inconsistency Measure**:
- $H^0(\mathcal{F})$: Global sections = consistent reward assignments
- $H^1(\mathcal{F})$: Obstructions to gluing = inconsistencies

**Condorcet Cycle Detection**:

A Condorcet cycle occurs when: $A \succ B \succ C \succ A$

In our framework:
1. Preferences define edge weights in the trajectory graph
2. A cycle with non-zero "circulation" indicates Condorcet paradox
3. This circulation is precisely what $H^1$ measures

**Detection Algorithm**:
```python
def detect_condorcet_cycles(preferences: List[Tuple[str, str, float]]):
    """
    preferences: List of (winner, loser, margin) tuples
    Returns: List of cycles with non-zero circulation
    """
    # Build preference graph
    G = nx.DiGraph()
    for winner, loser, margin in preferences:
        G.add_edge(loser, winner, weight=margin)
    
    # Find all simple cycles
    cycles = list(nx.simple_cycles(G))
    
    # Compute circulation for each cycle
    condorcet_cycles = []
    for cycle in cycles:
        circulation = 0
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            circulation += G[u][v]['weight']
        
        if abs(circulation) > epsilon:
            condorcet_cycles.append((cycle, circulation))
    
    return condorcet_cycles
```

**Perspective-Invariant Core**:

When multiple evaluators provide feedback:
1. Compute individual evaluator cochains $c_i \in C^0(\mathcal{F})$
2. Compute consensus via weighted harmonic mean (respects metric)
3. $H^1$ captures inter-evaluator disagreement
4. **Invariant core**: $H^0 = \cap_i \ker(\delta_i)$ — what all evaluators agree on

**Perspective-Variant Components**:

When evaluators disagree:
1. Decompose disagreement into orthogonal "opinion directions"
2. Each direction is a **legitimate perspective** (not noise)
3. Policy can be conditioned on perspective: $\pi(a|s, \text{perspective}_i)$
4. Or aggregate with explicit uncertainty: $\pi(a|s) = \sum_i w_i \pi_i(a|s)$

**Training Objective**:
$$\mathcal{L}_{\text{SRN}} = \underbrace{\mathcal{L}_{\text{ranking}}}_{\text{pairwise preferences}} + \lambda_1 \underbrace{\|H^1\|^2}_{\text{consistency regularizer}} + \lambda_2 \underbrace{\mathcal{L}_{\text{restriction}}}_{\text{scale coherence}}$$

Where:
- $\mathcal{L}_{\text{ranking}}$: Standard preference learning (BT model)
- $\|H^1\|^2$: Penalize inconsistencies (or use as diagnostic, not penalty)
- $\mathcal{L}_{\text{restriction}}$: Enforce restriction map coherence

---

## 5. Dataset Construction Guidelines

### 5.1 Multi-Scale Annotation Protocol

For each trajectory:
1. **Trajectory-level**: Overall quality rating (1-5), preference vs alternatives, outcome assessment
2. **Segment-level**: Mid-level goals achieved? Intermediate progress?
3. **Step-level**: Was this specific action helpful? Harmful? Neutral?

### 5.2 Multi-Evaluator Protocol

For detecting perspective variance:
1. Minimum 3 evaluators per trajectory (ideally 5+)
2. Diverse evaluator pool (different backgrounds, values)
3. Record evaluator ID with each annotation
4. Include "confidence" rating for each judgment

### 5.3 Counterfactual Annotation

For learning world models:
1. At key decision points, annotate alternative actions
2. If possible, collect "what would have happened" judgments
3. Flag high-stakes decisions (cliffs, black holes nearby)

### 5.4 Feedback Question Types

**Progress Questions**:
- "How much closer is the agent to the goal after this action?" (+2/+1/0/-1/-2)
- "Is this action productive or wasteful?"

**Safety Questions**:
- "Could this action lead to an unrecoverable state?"
- "Is this action risky but potentially high-reward?"

**Consistency Questions**:
- "Does this action align with the agent's stated goal?"
- "Is there a contradiction between this action and a previous action?"

**Preference Questions**:
- "Which of these two trajectories do you prefer?"
- "Rank these 4 alternative actions from best to worst."

---

## 6. Critical Evaluation of Current Simulation Suite

### 6.1 What We're Doing Well

| Component | Alignment with Core Concepts | Score |
|-----------|------------------------------|-------|
| **Hodge Decomposition** | Correctly separates consistent (gradient) from inconsistent (curl) reward components | ✓✓✓ |
| **Black Hole Formalism** | Provides hard geometric constraints, not just soft penalties | ✓✓✓ |
| **Multi-Evaluator Support** | `SheafResolver` handles multiple perspectives | ✓✓ |
| **Semantic Embeddings** | States are embedded via sentence transformers | ✓✓ |
| **Condorcet Detection** | H¹ magnitude correlates with cyclic preferences | ✓✓ |

### 6.2 Gaps and Misalignments

#### Gap 1: World Model is Missing or Implicit

**Problem**: Current implementation lacks an explicit **learned world model** $P(s'|s,a)$.

- `semantic_sapr_demo.py` shows SAPR concepts but the "P" (transition probability) is not a learned neural network
- `MultimodalSSMAgent` has a `world_head` but it predicts next *tokens*, not next *semantic states*
- No mechanism for multi-step rollouts or planning

**Fix**: Implement a proper World Model Transformer (Section 4.2) that:
- Takes (state_embedding, action_embedding) → next_state_distribution
- Supports uncertainty quantification
- Enables tree search / planning

#### Gap 2: Restriction Maps Are Assumed, Not Learned

**Problem**: The sheaf restriction maps $\rho: F(\text{trajectory}) \to F(\text{segment}) \to F(\text{step})$ are currently identity or hand-designed.

- `HodgeCritic` computes Hodge decomposition but restriction maps are implicit
- No mechanism to learn how trajectory-level feedback decomposes to step-level

**Fix**: Add a `RestrictionNetwork` that:
- Learns $\rho_{\text{traj→seg}}$ and $\rho_{\text{seg→step}}$ from data
- Is trained to minimize reconstruction error + sheaf axiom violation
- Can extrapolate to new trajectories

#### Gap 3: State Representation is Pre-computed, Not Causal

**Problem**: States are embedded via a frozen sentence transformer, not a learned Kolmogorov-minimal encoder.

- No compression objective (minimize description length)
- No causal sufficiency verification (does state predict outcomes?)
- Embeddings don't evolve with training

**Fix**: Implement Semantic State Encoder (Section 4.1) that:
- Jointly learns state representation and dynamics
- Has compression penalty for minimal descriptions
- Is end-to-end trainable with the policy

#### Gap 4: Condorcet Detection is Post-Hoc, Not Integrated

**Problem**: We can detect Condorcet cycles via H¹, but this doesn't feed back into training.

- `condorcet_experiment.py` demonstrates detection but doesn't resolve cycles
- No mechanism to ask for clarifying feedback when cycles detected
- Policy doesn't know how to behave when preferences are inconsistent

**Fix**: Integrate cycle detection into the training loop:
- When H¹ > threshold, flag for human review
- Implement cycle-breaking heuristics (e.g., remove weakest preference)
- Condition policy on "perspective" when fundamental disagreement exists

#### Gap 5: Reward is Still Scalar in Practice

**Problem**: Despite the theoretical framework of vector rewards, the actual policy optimization uses scalar rewards.

- `semantic_mdp_rl.py`: `RolloutBuffer` stores `rewards: List[float]`
- SGPO adds Hodge alignment as a scalar bonus, not a vector objective
- Multi-objective Pareto optimization is mentioned but not implemented

**Fix**: Implement true vector reward optimization:
- Store reward vectors in buffer
- Use Pareto-based selection for policy updates
- Visualize reward vector trajectories in manifold explorer

#### Gap 6: Transition P(s'|s,a) Conflates Oracle and Agent

**Problem**: In `semantic_sapr_demo.py`, the "World Model" predicts next tokens, but this is internal to the agent, not modeling the external environment.

- The agent's world model should predict what the *environment* will do
- Currently, the architecture conflates agent's internal prediction with environment dynamics

**Fix**: Clearly separate:
1. **Environment/Oracle**: The actual dynamics (e.g., TextWorld game engine, GPT-4 as world simulator)
2. **Agent's World Model**: The agent's *learned approximation* of the environment
3. **Policy**: The agent's action selection given its beliefs

---

## 7. Recommended Edits to Simulation Suite

### 7.1 Immediate Fixes (Align with Core Concepts)

1. **Add explicit world model to `MultimodalSSMAgent`** that predicts environment response, separate from internal token prediction

2. **Store vector rewards in `RolloutBuffer`** even if collapsed to scalar for current training

3. **Add cycle detection to `HodgeCritic.compute_hodge_decomposition()`** that returns a list of detected Condorcet cycles

4. **Add causal sufficiency metric** to state encoder: can the state alone predict the next state given action?

### 7.2 Structural Changes (For Full Framework)

1. **Implement `WorldModelTransformer`** class in new file `src/world_model.py`

2. **Implement `RestrictionNetwork`** in `src/sheaf_resolver.py` with learnable restriction maps

3. **Implement `SemanticStateEncoder`** with compression objective in `src/state_encoder.py`

4. **Refactor `semantic_mdp_rl.py`** to use these components in a unified pipeline

---

## 8. Summary: The Tensor Flow, Restated

With all components in place, the data flow becomes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PERCEPTION (Proposal A)                              │
│  Raw observation → Semantic State Encoder → Kolmogorov minimal s_t           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREDICTION (Proposal B)                              │
│  (s_t, candidate actions) → World Model Transformer → P(s_{t+1} | s_t, a)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ACTION (Proposal C)                                │
│  (s_t, value estimates) → Geodesic Policy Network → π(a | s_t)              │
│  Subject to: d_g(s, BlackHoles) > 0, H¹ < threshold                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENVIRONMENT TRANSITION                                │
│  (s_t, a_t) → Oracle/Environment → s_{t+1}                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FEEDBACK (Proposal D)                               │
│  Trajectory τ → Multi-scale, multi-evaluator feedback → Sheaf Reward Network│
│  → Hodge decomposition → ∇φ (consistent gradient) + H¹ (inconsistency)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          POLICY UPDATE                                       │
│  ∇_θ J = E[∇_θ log π_θ(a|s) · ⟨∇φ, Δe⟩]                                     │
│  (Bellman via Hodge gradient, not raw scalar reward)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

This is the **Tensor RL** framework: state tensors flow through learned transformations, generating trajectory tensors, which are evaluated by the reward sheaf tensor, which is differentiated to update the policy tensor.

---

## References

- Sutton & Barto (2018). Reinforcement Learning: An Introduction
- Bellman (1957). Dynamic Programming
- Schulman et al. (2017). Proximal Policy Optimization
- Achiam et al. (2017). Constrained Policy Optimization
- Robinson (2014). Topological Signal Processing
- Curry (2014). Sheaves, Cosheaves and Applications
- Chen et al. (2021). Decision Transformer
- Ota et al. (2024). Decision Mamba

---

*Document created: January 2025*
*Part of the Sheaf-Theoretic Reward Spaces research project*
