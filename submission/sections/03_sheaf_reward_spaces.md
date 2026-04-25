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
