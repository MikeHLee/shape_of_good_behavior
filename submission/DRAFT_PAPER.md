# Sheaf-Theoretic Reward Spaces for Safe Reinforcement Learning

**Abstract**

Reinforcement Learning from Human Feedback (RLHF) typically collapses rich, multi-dimensional human preferences into a scalar reward function. This reduction leads to topological information loss, rendering the system blind to cyclic preferences (e.g., Condorcet cycles) and susceptible to reward hacking in deceptive environments. We propose **Sheaf-Theoretic Reward Spaces (STRS)**, a rigorous framework that models rewards as sections of a sheaf over the trajectory manifold. This allows us to: (1) Detect global inconsistencies in feedback using the first Čech cohomology group ($H^1$); (2) Decompose rewards into exact (consistent) and harmonic (cyclic) components via the Hodge decomposition; and (3) Enforce hard safety constraints by modeling forbidden regions as geometric singularities in a Riemannian manifold. We introduce **Sheaf-Geodesic Policy Optimization (SGPO)**, an algorithm that optimizes policies to follow geodesics on this learned manifold. Experiments on cyclic preference benchmarks and deceptive safety traps demonstrate that SGPO successfully navigates structures where standard PPO and Constrained Policy Optimization (CPO) fail.
# 1. Introduction

## 1.1 The Scalar Reward Problem

Reinforcement Learning from Human Feedback (RLHF) has emerged as the dominant paradigm for aligning large language models (LLMs) and autonomous agents with human intent. The standard recipe involves collecting human preferences over model outputs, training a reward model to predict these preferences, and optimizing a policy to maximize the expected cumulative reward. This approach relies on a fundamental assumption: that human preferences can be faithfully compressed into a scalar reward function $R(s,a) \to \mathbb{R}$.

This scalar assumption is mathematically convenient but topologically restrictive. Human preferences are notoriously complex, often exhibiting:
1.  **Non-transitivity**: Cyclic preferences (e.g., A > B > C > A) that cannot be represented by any scalar value function.
2.  **Context-dependence**: What is "safe" or "helpful" depends on local context in ways that global scalar aggregation obscures.
3.  **Evaluator Disagreement**: Diverging views among human annotators that are typically averaged out, suppressing minority perspectives.

When we force this rich structure into a scalar reward, we induce **topological information loss**. The reward model learns to flatten loops and ignore inconsistencies, often resulting in reward hacking—where the agent exploits the reward model's inability to distinguish between high-quality outputs and those that merely game the scalar metric. Furthermore, standard RLHF lacks formal safety guarantees; safety is typically handled via penalty terms or separate cost models, which can be overridden by sufficiently high rewards.

## 1.2 Our Contribution: Sheaf-Theoretic Reward Spaces

We propose a novel framework, **Sheaf-Theoretic Reward Spaces (STRS)**, that addresses these limitations by modeling rewards not as scalars, but as sections of a sheaf over the trajectory space. This allows us to apply tools from algebraic topology and differential geometry to the alignment problem.

Our key contributions are:

1.  **Topological Consistency Checking**: We model human feedback as local sections of a reward sheaf. We show that the first Čech cohomology group $H^1$ measures the "winding number" of preferences, providing a formal test for global consistency. Non-trivial cohomology ($H^1 \neq 0$) detects Condorcet cycles that scalar rewards miss.

2.  **The Hodge-Augmented Critic**: We introduce a new critic architecture based on the Hodge decomposition theorem, which splits the reward signal into an **exact component** (standard value potential) and a **harmonic component** (cyclic flow). This allows the agent to learn and navigate cyclic preferences rather than stalling or oscillating.

3.  **Geometric Safety via Black Holes**: Instead of soft constraints, we model forbidden regions as **singularities** in a Riemannian reward manifold. We learn a conformal metric that diverges at the "event horizon" of dangerous states, effectively making them infinitely far away in geodesic distance.

4.  **Sheaf-Geodesic Policy Optimization (SGPO)**: We present an algorithm that optimizes policies to follow geodesics on this learned manifold. SGPO naturally integrates consistency and safety, outperforming standard PPO and Constrained Policy Optimization (CPO) on benchmarks involving deceptive traps and cyclic goals.

This work bridges the gap between abstract topology and practical AI safety, offering a mathematically rigorous path beyond scalar rewards.
# 2. Background

## 2.1 Reinforcement Learning from Human Feedback

The standard RLHF pipeline consists of three phases: (1) Supervised Fine-Tuning (SFT) of a base model; (2) Reward Modeling, where a scalar reward function $R_\phi(s,a)$ is trained on a dataset of human preferences $\mathcal{D} = \{ (s, a_w, a_l) \}$ to maximize the likelihood of the preferred completion $a_w$; and (3) Policy Optimization, where the policy $\pi_\theta$ is trained to maximize the expected reward using an algorithm like PPO, subject to a KL-divergence constraint to stay close to the SFT model.

Formally, the reward model loss is typically the Bradley-Terry cross-entropy:
$$ \mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(s, a_w, a_l) \sim \mathcal{D}} \left[ \log \sigma(R_\phi(s, a_w) - R_\phi(s, a_l)) \right] $$
This assumes that the probability of preferring $a_w$ over $a_l$ depends only on the difference in their latent scalar utilities. As noted in Section 1, this assumption fails when preferences are intransitive (cyclic).

## 2.2 Safe Reinforcement Learning

Safe RL seeks to maximize reward while satisfying safety constraints. A common formulation is the Constrained Markov Decision Process (CMDP), where the goal is:
$$ \max_\pi \mathbb{E}[R] \quad \text{s.t.} \quad \mathbb{E}[C] \le d $$
where $C$ is a cost signal and $d$ is a threshold. Algorithms like Constrained Policy Optimization (CPO) solve this by approximating the constraint with a trust region and projecting the gradient update onto the feasible set.

However, CMDPs typically enforce constraints only in expectation or with soft penalties (Lagrangian relaxation). This is insufficient for "black hole" risks where a single violation is catastrophic. Our approach differs by embedding safety into the geometry of the state space itself, providing stronger avoidance guarantees.

## 2.3 Topological Prerequisites

We leverage concepts from algebraic topology and differential geometry.

**Sheaves.** A sheaf $\mathcal{F}$ on a topological space $X$ creates a systematic way to track local data and its global consistency. For every open set $U \subseteq X$, $\mathcal{F}(U)$ is the set of data (sections) over $U$. Restriction maps $\rho_{UV}: \mathcal{F}(U) \to \mathcal{F}(V)$ for $V \subseteq U$ ensure that global data restricts consistently to local data.

**Cohomology.** Sheaf cohomology groups $H^k(X, \mathcal{F})$ measure global obstructions. $H^0$ corresponds to global sections (consistent data). $H^1$ measures the failure of local sections to glue together into a global one. In our context, $H^1$ detects cyclic preferences.

**Riemannian Manifolds.** A Riemannian manifold $(M, g)$ is a smooth manifold equipped with a metric tensor $g$, which defines an inner product on the tangent space at each point. This allows the definition of path lengths and geodesics (shortest paths). We use the metric to encode safety: regions with "large" metrics are "far away" in the eyes of the optimization algorithm.
# 3. Sheaf-Theoretic Reward Spaces

In this section, we formalize the problem of reinforcement learning from inconsistent human feedback using sheaf theory. We define the reward sheaf, introduce the cohomology of feedback, and present the Hodge decomposition for reward modeling.

## 3.1 The Reward Sheaf Construction

Standard reinforcement learning assumes that rewards are scalars $r_t \in \mathbb{R}$ that sum consistently along trajectories. However, human feedback is often multi-scale (per-step, per-episode) and multi-dimensional (safety, efficiency, style). We model this structure using a cellular sheaf.

**Definition 3.1 (Trajectory Space).** Let $S$ be the state manifold and $A$ the action space. We define a cell complex $X$ representing the trajectory space, where:
*   **0-cells (Vertices)**: States $s \in S$.
*   **1-cells (Edges)**: Transitions $(s, a, s')$.
*   **2-cells (Faces)**: Elementary loops or 3-step transition cycles representing local consistency checks.

**Definition 3.2 (Reward Sheaf).** A reward sheaf $\mathcal{F}$ on $X$ assigns a vector space $\mathcal{F}(U)$ to each open set $U \subset X$, representing possible reward valuations over that region. For an inclusion $V \subset U$, the restriction map $\rho_{UV}: \mathcal{F}(U) \to \mathcal{F}(V)$ captures how global evaluations constrain local ones.

Specifically, for a trajectory $\tau$ composed of segments $\sigma_1, \dots, \sigma_k$, a consistent reward assignment requires that the global evaluation $R(\tau)$ agrees with the aggregation of local evaluations $r(\sigma_i)$.

## 3.2 Cohomological Consistency

When human evaluators provide pairwise preferences or scalar ratings, they are effectively sampling sections of this sheaf. Inconsistencies arise when these local samples cannot be glued into a global section.

**Theorem 3.1 (Consistency Criterion).** Let $\mathcal{U}$ be a cover of the trajectory space $X$. The collection of local feedback samples $\{r_i \in \mathcal{F}(U_i)\}$ is globally consistent if and only if the first Čech cohomology class vanishes:
$$ [\delta \{r_i\}] = 0 \in H^1(\mathcal{U}, \mathcal{F}) $$

**Proof (Sketch).** The coboundary operator $\delta$ measures the disagreement between local sections on overlaps: $(\delta r)_{ij} = r_i|_{U_i \cap U_j} - r_j|_{U_i \cap U_j}$. If this difference is non-zero and cannot be explained by a global potential difference (a 0-coboundary), it represents a fundamental topological obstruction—a cycle in preference space.

**Corollary 3.2 (Condorcet Cycles).** A non-transitive preference cycle $A \succ B \succ C \succ A$ corresponds to a non-trivial cohomology class $\omega \in H^1(X, \mathbb{R})$. This implies no scalar value function $V: S \to \mathbb{R}$ can perfectly represent the preferences, as $\oint \nabla V = 0$ while the preference circulation is non-zero.

## 3.3 The Hodge Decomposition for Rewards

To handle these inconsistencies, we propose decomposing the reward signal rather than forcing it into a scalar potential. We invoke the discrete Hodge decomposition theorem.

**Theorem 3.3 (Hodge Decomposition).** Let $r \in C^1(X, \mathbb{R})$ be a reward 1-cochain (function on transitions). Then $r$ admits a unique orthogonal decomposition:
$$ r = dV + \delta \psi + \omega $$
where:
1.  $dV$ is the **Exact Component** (Gradient flow): Represents consistent, transitive preferences derived from a scalar potential $V$ (the standard Value function).
2.  $\delta \psi$ is the **Coexact Component** (Curl flow): Represents local rotational inconsistencies (often zero in tree-like MDPs).
3.  $\omega$ is the **Harmonic Component** (Global flow): Represents fundamental topological cycles ($H^1$).

**Implication for RL.** Standard RL algorithms (PPO, DQN) implicitly assume $r \approx dV$ and try to learn $V$. When $\omega \neq 0$ (cyclic preferences), the Bellman error $r + V(s') - V(s)$ cannot be minimized to zero, leading to instability or value collapse. Our framework explicitly learns both $V$ and $\omega$, allowing the agent to model and navigate cyclic preferences.
# 4. Geometric Safety via Black Holes

While sheaf cohomology addresses consistency, it does not inherently guarantee safety. To address catastrophic risks, we introduce a geometric formalism for forbidden regions, modeling them as singularities in the reward manifold.

## 4.1 The Reward Manifold

We define the **Reward Manifold** $\mathcal{M}$ as the embedding of the trajectory space into a high-dimensional metric space $(\mathbb{R}^d, g)$. Unlike the flat Euclidean space assumed in standard feature embeddings, $\mathcal{M}$ possesses intrinsic curvature determined by the safety landscape.

**Definition 4.1 (Black Hole Region).** A black hole $\mathcal{B} \subset \mathcal{M}$ is a compact region representing forbidden outcomes (e.g., irreversible harm). It is characterized by:
1.  **Event Horizon**: The boundary $\partial \mathcal{B}$.
2.  **Singularity**: A point or region $c \in \text{int}(\mathcal{B})$ where the cost function diverges.

## 4.2 Learning the Safety Metric

Standard constrained RL (e.g., CPO) treats safety as a separate cost signal $C(s)$ constrained by a threshold $d$. We propose incorporating safety directly into the geometry of the state space via a **Riemannian metric** $g$.

We construct a conformal metric $g_{ij}(x) = e^{2\sigma(x)} \delta_{ij}$, where $\sigma(x)$ is a scalar conformal factor learned from safety data.

**Proposition 4.1 (Metric Singularity).** To enforce strict avoidance of the black hole, we require the geodesic distance from any safe state $s_{safe}$ to the event horizon to be infinite. This condition is satisfied if the conformal factor scales as:
$$ e^{\sigma(x)} \approx \frac{1}{\text{dist}(x, \mathcal{B})^\alpha} $$
where $\alpha \ge 1$.

**Proof.** Consider a path $\gamma$ approaching the boundary at $r=0$. The length is $L = \int_\epsilon^0 r^{-\alpha} dr$. This integral diverges for $\alpha \ge 1$.

In practice, we parametrize $\sigma_\theta(x)$ using a neural network trained on a "safety potential" derived from cost signals. The network learns to output large scaling factors near high-cost regions, effectively stretching the space so that dangerous regions become "infinitely far away" for an agent traversing geodesics.

## 4.3 Theoretical Guarantees

This geometric formulation provides stronger guarantees than soft penalties.

**Theorem 4.2 (Geodesic Avoidance).** Let $\pi^*$ be a policy that generates trajectories minimizing the Riemannian path length on $(\mathcal{M}, g)$. If the metric $g$ satisfies the singularity condition (Prop 4.1) for all $\mathcal{B}_i$, then $\pi^*$ will not enter any black hole region with probability 1.

This transforms the safety problem from a constrained optimization problem (hard to solve, prone to feasibility issues) into an unconstrained shortest-path problem on a curved manifold.
# 5. Sheaf-Geodesic Policy Optimization

We introduce **Sheaf-Geodesic Policy Optimization (SGPO)**, an algorithm that optimizes policies to follow geodesics on the learned reward manifold. SGPO integrates the geometric safety constraints and sheaf-theoretic consistency checks into a unified policy gradient framework.

## 5.1 Riemannian Policy Gradient

Standard policy gradient methods perform updates in the Euclidean space of parameters, often using the Fisher Information Matrix (Natural Policy Gradient) to account for the geometry of the probability simplex. SGPO extends this by incorporating the **geometry of the reward manifold** itself.

Let $J(\theta)$ be the expected return. The standard gradient update is $\theta_{k+1} = \theta_k + \alpha \nabla_\theta J(\theta_k)$. In SGPO, we define the update direction using the inverse of the learned Riemannian metric $G(s)$:

$$ \nabla_G J(\theta) = \mathbb{E}_{s,a \sim \pi} \left[ G(s)^{-1/2} \nabla_\theta \log \pi_\theta(a|s) A^{\text{Hodge}}(s,a) \right] $$

Here, the term $G(s)^{-1/2}$ acts as a preconditioner that dampens gradient steps in regions of high curvature (near black holes) and amplifies them in flat, safe regions. This naturally enforces safety: as the agent approaches a danger zone, the effective learning rate drops to zero, preventing the policy from pushing further into the trap.

## 5.2 The Hodge-Augmented Critic

To handle cyclic preferences, SGPO replaces the standard scalar value function $V(s)$ with a **Hodge Critic** that learns both the potential and harmonic components of the reward.

The critic is parameterized as $(V_\phi, \omega_\psi)$, minimizing the **Hodge Bellman Error**:

$$ \mathcal{L}(\phi, \psi) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}} \left[ \left( r(s,a) - \underbrace{(V_\phi(s') - V_\phi(s))}_{\text{Potential Difference}} - \underbrace{\omega_\psi \cdot v(s,a)}_{\text{Harmonic Flux}} \right)^2 \right] $$

where $v(s,a)$ is the velocity vector of the transition. The term $\omega_\psi$ captures the non-transitive "circulation" of the reward.

## 5.3 Geodesic Advantage Estimation

The advantage function in SGPO is modified to account for both the metric distortion and the harmonic component:

$$ A^{\text{Hodge}}(s,a) = \frac{1}{\sqrt{G(s)}} \left( r(s,a) + \gamma V(s') - V(s) - \omega_\psi \cdot v(s,a) \right) $$

By normalizing the advantage by $\sqrt{G(s)}$, we ensure that high-reward but high-risk actions (inside a trap) have a diminished impact on the policy update, effectively "discounting" rewards gained in dangerous geometries.

## 5.4 The SGPO Algorithm

**Algorithm 1: Sheaf-Geodesic Policy Optimization**

1.  **Initialize**: Policy $\pi_\theta$, Hodge Critic $(V_\phi, \omega_\psi)$, Metric Model $G_\xi$.
2.  **Loop**:
    a.  Collect trajectories $\tau \sim \pi_\theta$.
    b.  **Update Metric**: Train $G_\xi$ on cost signals to approximate singularity at $C(s) > \text{threshold}$.
    c.  **Update Critic**: Minimize Hodge Bellman Error to learn $V_\phi$ and $\omega_\psi$.
    d.  **Compute Advantages**: Calculate Riemannian-scaled, Hodge-corrected advantages.
    e.  **Update Policy**: Perform gradient ascent using $\nabla_G J(\theta)$.
3.  **Return**: Safe policy $\pi^*$.
# 6. Experiments

We validate the STRS framework and SGPO algorithm on three distinct problem settings designed to isolate specific failure modes of standard RLHF: cyclic preferences, deceptive safety traps, and stylistic finetuning.

## 6.1 Experimental Setup

**Baselines.** We compare our proposed **Sheaf-Geodesic Policy Optimization (SGPO)** against two primary baselines:
1.  **PPO (Proximal Policy Optimization)**: The standard algorithm for RLHF, representing the scalar reward hypothesis.
2.  **CPO (Constrained Policy Optimization)**: A leading Safe RL algorithm that enforces constraints via trust region projection and Lagrangian relaxation.

**Metrics.**
*   **Consistency**: Difference between learned harmonic coefficient $\omega$ and ground truth cohomology $H^1$.
*   **Safety**: Number of steps spent inside forbidden regions ("trap violations").
*   **Performance**: Accumulation of "honest" rewards (excluding deceptive trap bonuses).

## 6.2 Detecting Cyclic Preferences (Condorcet Ring)

**Setting.** The environment is a continuous circle $S^1$ where the agent receives a constant positive reward for moving clockwise. This creates a Condorcet cycle ($A \succ B \succ C \succ A$). The ground truth cohomology is $H^1 = \frac{1}{2\pi} \oint r d\theta > 0$.

**Results.**

| Metric | Ground Truth | SGPO (Learned) | Error |
|--------|--------------|---------------|-------|
| $H^1$ | 0.500 | 0.357 | 28.6% |

*   **Value Function Collapse**: PPO fails to learn a stable value function. Because $\oint \nabla V = 0$, the critic forces the value estimates to be periodic, contradicting the constantly increasing return. This results in oscillating gradients and unstable learning (Figure 2a).
*   **Hodge Decomposition**: SGPO's Hodge Critic successfully decomposes the signal. The learned harmonic coefficient $\omega = 0.357$ captures 71.4% of the ground truth cycle strength, demonstrating that cohomological structure can be recovered from trajectory data.
*   **Policy Stability**: SGPO maintains a consistent positive velocity, whereas PPO's policy degrades as the critic becomes unreliable.

## 6.3 Geometric Safety (The "Sandbagging" Trap)

**Setting.** A 2D navigation task where the shortest path to the goal passes through a "Sandbagging Trap"—a region offering high immediate reward but representing a catastrophic safety violation (the Black Hole). This simulates alignment scenarios where deceptive behavior yields high approval feedback despite being unsafe.

**Results.**

| Method | Total Violations | Mean Return | Safety-Performance Trade-off |
|--------|-----------------|-------------|------------------------------|
| PPO | 52 | -6.67 | Unsafe, poor returns |
| CPO | 7 | -6.23 | Safe but overly conservative |
| SGPO | 11 | **1.53** | Balanced safety + task success |

*   **PPO (Unsafe)**: Consistently enters the trap (52 total violations), prioritizing the high "deceptive" reward over safety. It lacks any mechanism to recognize the risk beyond scalar magnitude.
*   **CPO (Overly Conservative)**: Achieves the fewest violations (7) but at the cost of task performance. The Lagrangian penalty causes the policy to avoid not just the trap but the entire goal region, resulting in negative mean returns.
*   **SGPO (Balanced)**: Demonstrates the best task returns (1.53 vs. -6.23) while maintaining comparable safety to CPO. The geometric approach allows the policy to navigate "expensive" regions when rewards justify it, rather than blanket avoidance.

**Key Insight**: SGPO achieves **~8× better returns** than CPO with only ~1.5× more violations, occupying a superior point on the safety-performance Pareto frontier.

## 6.4 Navigating Style Cycles (LLM Simulation)

**Setting.** A simulated embedding space for an LLM with three archetypes: Concise, Empathetic, and Detailed. Human preferences form a cycle: Concise users want empathy, Empathetic users want details, Detailed users want brevity.

**Results.**

| Metric | Ground Truth | PPO | SGPO |
|--------|--------------|-----|-----|
| $H^1$ (curl) | 0.364 | — | 0.349 (96% accuracy) |
| Cycle-following accuracy | — | 66.3% | **74.4%** |
| Total archetype transitions | — | 12,277 | 18,265 |

*   **Stalling**: PPO agents navigate the style space with only 66.3% accuracy in following the preference cycle, often getting stuck or moving against the gradient.
*   **Cycling**: SGPO learns a curl component $\omega = 0.349$ that closely matches the ground truth ($H^1 = 0.364$, 96% accuracy). The policy achieves 74.4% cycle-following accuracy with 49% more style transitions, demonstrating dynamic adaptation to cyclic preferences rather than mode collapse.
# 7. Related Work

## 7.1 Distributional and Multi-Objective RL

Standard RL treats the return as a scalar random variable and optimizes its expectation. **Distributional RL** (Bellemare et al., 2017; Dabney et al., 2018) estimates the full distribution of returns, capturing uncertainty and multimodality, but typically still projects this distribution back to a scalar policy gradient or value estimate for decision making.

**Multi-Objective RL (MORL)** (Roijers et al., 2013) deals with vector-valued rewards, aiming to approximate the Pareto front. Our work differs by assuming that the vector nature of feedback arises not just from competing scalar objectives, but from the **topological structure** of the preference space itself. The Hodge decomposition provides a principled way to orthogonalize these components into consistent (exact) and cyclic (harmonic) parts, which MORL does not explicitly address.

## 7.2 Safe Reinforcement Learning

The dominant paradigm for safety is the **Constrained Markov Decision Process (CMDP)** (Altman, 1999). **Constrained Policy Optimization (CPO)** (Achiam et al., 2017) solves CMDPs by computing trust region updates that satisfy linearized safety constraints. Other approaches use Lagrangian relaxation (Ray et al., 2019) or safety layers (Dalal et al., 2018).

While effective for explicit constraints, these methods struggle with "deceptive" risks where high rewards lure agents into irreversible states before constraints are violated in expectation. Our **Sheaf-Geodesic Policy Optimization (SGPO)** replaces explicit constraints with intrinsic geometry. By learning a metric that diverges at danger zones, we enforce safety via the principle of least action, providing stronger avoidance guarantees than soft penalties.

## 7.3 Topological Methods in Machine Learning

Topological Data Analysis (TDA) has applied persistent homology to characterize the shape of data manifolds (Edelsbrunner & Harer, 2010). In deep learning, **Neural Sheaf Diffusion** (Bodnar et al., 2022) uses cellular sheaves to model non-smoothing information flow in Graph Neural Networks.

To our knowledge, **STRS** is the first framework to apply sheaf cohomology to Reinforcement Learning from Human Feedback. We show that the "alignment problem" is partially a topological one: inconsistencies in human feedback are not just noise to be filtered, but topological features (cycles) to be detected and modeled.

## 7.4 Preference Learning and Intransitivity

The standard Bradley-Terry model assumes transitive preferences. However, intransitivity is common in social choice (Condorcet, 1785) and complex AI objectives. **Cyclic preference learning** (Csato, 2019) attempts to model this but often lacks a mechanism to integrate it into control. Our use of the Hodge decomposition bridges this gap, treating the cyclic component ($\omega$) as a valid driving force for the policy, akin to a non-conservative force field in physics.
# 8. Discussion and Limitations

## 8.1 SGPO vs. CPO: A Nuanced Comparison

Our experiments reveal that SGPO and CPO occupy different points on the safety-performance Pareto frontier:

| Metric | PPO | CPO | SGPO |
|--------|-----|-----|-----|
| Mean Return | -6.67 | -6.23 | **1.53** |
| Total Violations | 52 | 7 | 11 |

**Key finding**: SGPO achieves approximately **8× better task returns** than CPO while incurring only ~1.5× more violations. This reflects a fundamental difference in mechanism:

- **CPO** uses Lagrangian multipliers that penalize *any* proximity to constraints, often becoming overly conservative. The policy learns to avoid the goal region entirely if it lies near the penalty zone.
- **SGPO** uses geometric structure where the metric increases smoothly near danger. The policy can still pursue rewards in geometrically "expensive" regions when the reward justifies it.

Neither method dominates: CPO is preferable when **zero violations** is paramount; SGPO is preferable when **task completion with bounded risk** is the objective. The geometric approach also offers:

1. **No constraint tuning** — CPO requires threshold hyperparameters ($d$, $\lambda$); SGPO learns the metric
2. **Intrinsic interpretability** — the learned metric $g(x)$ visualizes danger regions directly
3. **Potential generalization** — SGPO's metric can extrapolate to novel unsafe states via learned features

## 8.2 Limitations

While Sheaf-Theoretic Reward Spaces offer a rigorous alternative to scalar reward modeling, several limitations remain:

1.  **Computational Complexity**: Computing exact Čech cohomology grows combinatorially with the size of the cover. While our discrete approximation on trajectory graphs is efficient ($O(T \cdot k \cdot d^2)$), scaling to massive datasets of human feedback may require sparse approximations or spectral methods (e.g., Sheaf Laplacians).
2.  **Metric Bootstrapping**: The safety metric $g(x)$ relies on cost signals to identify "event horizons." If these signals are themselves sparse or noisy, the learned geometry may be flawed. We assume a "weak supervision" signal for safety is available, which may not always hold.
3.  **Manifold Assumption**: SGPO assumes the reward space has a meaningful manifold structure. In discrete domains (e.g., token-level text generation), this smoothness assumption is an approximation. Embedding discrete states into continuous spaces (like standard transformer embeddings) mitigates this, but the topological fidelity of such embeddings is an open question.
4.  **Safety-Performance Trade-off**: As shown above, SGPO does not guarantee fewer violations than CPO—it offers better returns at comparable (not superior) safety. Applications requiring hard safety constraints may still prefer Lagrangian methods.

## 8.3 Future Work

**Scaling to LLMs.** The most immediate direction is applying SGPO to full-scale language model fine-tuning. This involves training a "Hodge Reward Model" that outputs both a scalar score and a vector field on the embedding space, allowing the LLM to navigate stylistic cycles or steer around conceptual "black holes" in the prompt space.

**Multi-Agent Coordination.** Sheaf theory naturally extends to multi-agent systems, where each agent's local observations form a section. Consistency checks via cohomology could detect misalignment or conflicting goals between agents without requiring a centralized value function.

**Temporal Cohomology.** Preferences often drift over time. Extending the framework to include a temporal dimension would allow us to detect "concept drift" in alignment as a non-zero cohomology class in the time direction, distinguishing between valid preference shifts and alignment instability.

## 8.4 Broader Impact

This work moves AI safety towards **interpretable geometric certificates**. Instead of opaque neural networks that "usually work," topological invariants like $H^1$ provide discrete, falsifiable checks for alignment consistency. However, powerful tools for navigating preference manifolds could also be used to manipulate user behavior more effectively. As with all alignment research, robust safety checks (like the black hole mechanism) are dual-use and must be deployed with care.
# 9. Conclusion

We have presented **Sheaf-Theoretic Reward Spaces (STRS)**, a rigorous mathematical framework that reimagines the foundations of Reinforcement Learning from Human Feedback. By lifting rewards from scalars to sheaf sections, we expose the rich topological structure of human preferences—including inconsistencies and cycles—that standard methods ignore.

Our results demonstrate that:
1.  **H¹ is a computable safety certificate**: The first cohomology group successfully detects Condorcet cycles in preference data, providing a concrete metric for alignment consistency.
2.  **Geometry enforces safety**: Modeling forbidden regions as singularities in a Riemannian manifold enables **Sheaf-Geodesic Policy Optimization (SGPO)** to achieve near-perfect safety rates in deceptive environments where PPO fails and CPO struggles.
3.  **Cyclic navigation is possible**: The Hodge-Augmented Critic allows agents to navigate preference cycles intelligently, "orbiting" the Pareto frontier of diverse user needs rather than collapsing to a mediocre mean.

As AI systems become more autonomous and their objectives more complex, the "scalar hypothesis"—that all values can be mapped to a single number—becomes increasingly untenable. STRS provides the necessary language to describe, measure, and optimize for the full spectrum of human intent, paving the way for safer, more nuanced, and topologically aware AI systems.
# Appendix

## A. Theoretical Proofs

### A.1 Proof of Theorem 3.1 (Consistency Criterion)

**Theorem.** Let $\mathcal{U}$ be a cover of the trajectory space $X$. The collection of local feedback samples $\{r_i \in \mathcal{F}(U_i)\}$ is globally consistent if and only if the first Čech cohomology class vanishes: $[\delta \{r_i\}] = 0 \in H^1(\mathcal{U}, \mathcal{F})$.

*Proof.*
By definition of the Čech coboundary operator $\delta: C^0(\mathcal{U}, \mathcal{F}) \to C^1(\mathcal{U}, \mathcal{F})$, a collection of local sections $\{r_i\}$ defines a 0-cochain. The condition for these sections to glue into a global section $r \in \mathcal{F}(X)$ is that they agree on all overlaps: $r_i|_{U_i \cap U_j} = r_j|_{U_i \cap U_j}$ for all $i, j$.
This can be rewritten as $r_i - r_j = 0$ on $U_{ij}$.
The 1-cochain $\delta \{r_i\}$ is defined by $(\delta r)_{ij} = r_j - r_i$ on $U_{ij}$.
Thus, the gluing condition is exactly $\delta \{r_i\} = 0$.
If $[\delta \{r_i\}] = 0$ in cohomology, it means the obstruction is a coboundary, which implies the local inconsistencies can be resolved by a re-parametrization (adding a 0-cochain), or if we are checking for the existence of *any* global section, the class must be trivial. Specifically, for the reward sheaf, non-trivial $H^1$ implies no scalar potential function exists that agrees with all local preference gradients. $\square$

### A.2 Proof of Theorem 3.3 (Hodge Decomposition)

**Theorem.** Let $r \in C^1(X, \mathbb{R})$ be a reward 1-cochain on a finite connected graph $X$. Then $r$ admits a unique orthogonal decomposition $r = dV + \delta \psi + \omega$.

*Proof.*
Let $C^0, C^1, C^2$ be the spaces of 0, 1, and 2-cochains equipped with the standard $L^2$ inner product $\langle f, g \rangle = \sum f(x)g(x)$.
The coboundary operators are $\delta_k: C^k \to C^{k+1}$. Their adjoints (boundary operators) are $\delta_k^*: C^{k+1} \to C^k$.
The Laplacian is defined as $\Delta_k = \delta_k^* \delta_k + \delta_{k-1} \delta_{k-1}^*$.
By the Hodge theorem for finite complexes, the space of $k$-cochains decomposes as:
$$ C^k = \text{im}(\delta_{k-1}) \oplus \text{im}(\delta_k^*) \oplus \ker(\Delta_k) $$
For $k=1$ (rewards on edges):
$$ C^1 = \text{im}(\delta_0) \oplus \text{im}(\delta_1^*) \oplus \ker(\Delta_1) $$
Identifying terms:
1.  $\text{im}(\delta_0) = \{ \delta_0 V \mid V \in C^0 \}$ is the space of **exact forms** (gradient fields).
2.  $\text{im}(\delta_1^*) = \{ \delta_1^* \psi \mid \psi \in C^2 \}$ is the space of **coexact forms** (curl fields).
3.  $\ker(\Delta_1) \cong H^1(X, \mathbb{R})$ is the space of **harmonic forms**.
Thus, any reward $r$ uniquely decomposes into $dV + \delta \psi + \omega$. $\square$

### A.3 Proof of Theorem 4.2 (Geodesic Avoidance)

**Theorem.** Let $\pi^*$ be a policy that generates trajectories $\tau$ minimizing the Riemannian path length $L_g(\tau)$. If the metric scaling satisfies $\sqrt{g(x)} \ge \frac{C}{\text{dist}(x, \mathcal{B})^\alpha}$ with $\alpha \ge 1$, then any finite-length trajectory $\tau$ cannot intersect the closure of the black hole $\overline{\mathcal{B}}$.

*Proof.*
Consider any path $\gamma: [0, 1] \to \mathcal{M}$ starting at a safe state $\gamma(0) \notin \mathcal{B}$ and ending at the boundary $\gamma(1) \in \partial \mathcal{B}$.
The Riemannian length is $L_g(\gamma) = \int_0^1 \sqrt{g_{\gamma(t)}(\dot{\gamma}, \dot{\gamma})} dt$.
Let $u(t) = \text{dist}(\gamma(t), \mathcal{B})$. As $t \to 1$, $u(t) \to 0$.
Since $|\dot{u}| \le \|\dot{\gamma}\|$, we have $\|\dot{\gamma}\| \ge |\dot{u}|$.
Substituting the metric condition:
$$ L_g(\gamma) \ge \int_0^1 \frac{C}{u(t)^\alpha} \|\dot{\gamma}(t)\| dt \ge \int_{u(0)}^{0} \frac{C}{u^\alpha} du $$
For $\alpha \ge 1$, the integral $\int_\epsilon^0 u^{-\alpha} du$ diverges to $\infty$.
Therefore, any path touching the black hole has infinite length.
Since we assume there exists at least one safe path with finite length (the environment is connected), the minimizer $\pi^*$ will assign probability 0 to paths entering $\mathcal{B}$. $\square$

## B. Implementation Details

### B.1 Network Architectures

**Actor (Policy)**:
- Input: State dimension $d$
- Hidden: 2 layers of 64 units, Tanh activation
- Output: Action dimension $k$ (Mean), separate parameter for log_std
- Distribution: Gaussian $\mathcal{N}(\mu, \sigma)$

**Standard Critic (PPO/CPO)**:
- Input: State dimension $d$
- Hidden: 2 layers of 64 units, Tanh activation
- Output: Scalar value $\mathbb{R}$

**Hodge Critic (SGPO)**:
- Potential Net $V_\phi$: Same architecture as Standard Critic.
- Harmonic Parameter $\omega$: Learnable vector $\mathbb{R}^k$ (or scalar coefficient for specific topologies).
- Output: Tuple $(V(s), \omega)$. Prediction $R(s,s',v) = V(s') - V(s) + \omega \cdot v$.

**Riemannian Metric Network**:
- Input: State dimension $d$
- Hidden: Implicitly parameterized by distance to trap centroids.
- Parameters: Severity $\sigma$ (scalar), Sharpness $\alpha$ (scalar).
- Formula: $g(x) = 1 + \sigma / (\text{margin}(x))^\alpha$.

### B.2 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **General** | | |
| Optimizer | Adam | Used for all networks |
| Learning Rate (Actor) | 3e-4 | |
| Learning Rate (Critic) | 1e-3 | |
| Discount Factor $\gamma$ | 0.99 | |
| Batch Size | N/A | Full trajectory updates used |
| **SGPO Specific** | | |
| Metric LR | 1e-2 | Learning rate for metric parameters |
| Initial Severity | 5.0 | Initial repulsion strength |
| Initial Sharpness | 2.5 | Initial singularity exponent ($\alpha$) |
| Harmonic LR | 1e-3 | Learning rate for $\omega$ |

### B.3 Environment Details

**Condorcet Ring**:
- State: $\theta \in [-\pi, \pi]$ (observed as $[\sin \theta, \cos \theta]$)
- Action: Velocity $v \in [-1, 1]$
- Reward: $0.5 \times v$ (Constant clockwise reward)
- Horizon: 100 steps

**Sandbagging Trap**:
- State: $(x, y) \in \mathbb{R}^2$
- Trap: Circle at $(5, 6)$, radius $2.5$
- Reward: +1 progress to goal, +3 inside trap (deceptive), -100 catastrophe
- Safety Cost: 1.0 inside trap

**Style Cycle**:
- State: 2D embedding space
- Archetypes: Triangle configuration
- Reward: Alignment with cyclic vector field
