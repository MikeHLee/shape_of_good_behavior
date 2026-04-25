# Theoretical Proofs: Hodge Decomposition & Geodesic Safety

This document formalizes the two core theoretical results for the "Sheaf-Theoretic Reward Spaces" paper.

---

## 1. The Hodge Decomposition Theorem for Reinforcement Learning

### 1.1 Intuition
In standard RL, we often assume a value function $V(s)$ exists such that $R(s,s') \approx V(s') - V(s)$. This implies the reward is a "potential difference" (exact form). However, if preferences are cyclic (A > B > C > A), no such scalar $V$ exists. The reward accumulates along the loop: $\oint R \neq 0$.

Hodge theory tells us that any vector field (or differential form) can be decomposed into:
1.  **Gradient** (Potential flow) $\nabla \phi$
2.  **Curl** (Rotational flow) $\nabla \times \psi$
3.  **Harmonic** (Global topological flow) $\gamma$

For RL on a state manifold (or graph), this allows us to separate "consistent" rewards (Value function) from "cyclic" rewards (Cohomology).

### 1.2 Formal Setting (Discrete)
Let the MDP state space be a directed graph $G = (S, E)$.
*   **0-cochains** $C^0(G, \mathbb{R})$: Functions on nodes (e.g., Value functions $V: S \to \mathbb{R}$).
*   **1-cochains** $C^1(G, \mathbb{R})$: Functions on edges (e.g., Reward functions $R: E \to \mathbb{R}$).
*   **Coboundary operator** $\delta_0: C^0 \to C^1$:
    $$(\delta_0 V)(u, v) = V(v) - V(u)$$
    This is the discrete gradient.
*   **Laplacian** $\Delta_0 = \delta_0^* \delta_0$: The graph Laplacian.

### 1.3 Theorem Statement
**Theorem 1 (Hodge-Helmholtz Decomposition for Rewards).**
Let $R \in C^1(G, \mathbb{R})$ be a reward function on a finite connected graph $G$. Then $R$ admits a unique orthogonal decomposition:
$$R = dV + \delta \psi + \omega$$
Where:
*   $dV \in \text{im}(\delta_0)$ is the **Gradient component** (Potential difference).
*   $\delta \psi \in \text{im}(\delta_1^*)$ is the **Curl component** (Local rotation, often 0 for graphs unless simplicial complexes used).
*   $\omega \in \ker(\Delta_1)$ is the **Harmonic component** (Global cyclic flow).

Specifically for graphs (1-dimensional complexes), the decomposition simplifies to:
$$R = \delta_0 V + R_{cycle}$$
Where $R_{cycle}$ satisfies $\sum_{cycle} R_{cycle} \neq 0$ for non-trivial loops.

**Refined Statement for Manifolds (Continuous RL):**
Let $M$ be the state manifold. Let $r$ be the reward 1-form. Then:
$$r = dV + \delta \psi + \omega$$
*   **$V$ (Exact)**: The standard Value function. $dV$ explains transitive preferences.
*   **$\omega$ (Harmonic)**: Represents $H^1_{dR}(M)$. Non-zero $\omega$ implies fundamental preference loops (Condorcet cycles).

### 1.4 Proof Strategy
1.  **Inner Product**: Define $\langle f, g \rangle = \sum f(e)g(e)$ on edges.
2.  **Orthogonality**: Show $\text{im}(\delta_0)$ is orthogonal to $\ker(\delta_0^*)$ (divergence-free flows).
3.  **Space Decomposition**: $C^1 = \text{im}(\delta_0) \oplus \ker(\delta_0^*)$.
4.  **Cyclic Component**: Further decompose $\ker(\delta_0^*)$ into local curls (boundaries of 2-cells if they exist) and global harmonic forms (homology generators).
5.  **Uniqueness**: Guaranteed by the projection theorem in Hilbert space.

### 1.5 Verification & Implications for RL
*   **Standard RL**: Assumes $R \approx \delta_0 V$ (plus noise). If $\omega \neq 0$, TD-learning fails to converge because $V(s) \leftarrow r + V(s')$ implies $0 \approx r + V(s') - V(s)$, but sum of updates along cycle is $\oint r \neq 0$.
*   **SGPO Fix**: We learn $V$ AND $\omega$. The policy follows the combined field $X = \text{grad}(V) + \omega^\sharp$.

---

## 2. Safety Guarantee Proof for Geodesic Policies

### 2.1 Intuition
We want to prove that a policy optimizing trajectories on a Riemannian manifold with a "singularity" (black hole) will avoid that region with high probability. The mechanism is that the singularity blows up the metric, making the "cost" (distance) of entering it infinite.

### 2.2 Formal Setting
*   **State Space**: Manifold $M \subseteq \mathbb{R}^n$.
*   **Forbidden Region**: Open set $B \subset M$ (The Black Hole).
*   **Safety Margin**: $U_\epsilon = \{x \in M \mid d_{Euclid}(x, B) < \epsilon\}$.
*   **Riemannian Metric**: $g_{ij}(x) = \phi(x)^2 \delta_{ij}$ (Conformal).
*   **Conformal Factor**: $\phi(x) \to \infty$ as $d(x, B) \to 0$.
    Specifically, let $\phi(x) \geq \frac{C}{d(x, B)^\alpha}$ for some $\alpha \ge 1$.

### 2.3 Theorem Statement
**Theorem 2 (Geodesic Avoidance).**
Let $\pi^*$ be a policy that generates trajectories $\tau$ minimizing the Riemannian path length $L_g(\tau)$. If the metric scaling satisfies $\phi(x) \ge \frac{C}{d(x, B)}$ (i.e., $\alpha \ge 1$), then any finite-length trajectory $\tau$ cannot intersect $\overline{B}$.

### 2.4 Proof
**Step 1: Length Divergence.**
Consider a path $\gamma: [0, 1] \to M$ such that $\gamma(0) = x_{safe}$ and $\gamma(1) \in \partial B$.
The Riemannian length is:
$$L_g(\gamma) = \int_0^1 \sqrt{g_{\gamma(t)}(\dot{\gamma}(t), \dot{\gamma}(t))} dt = \int_0^1 \phi(\gamma(t)) \|\dot{\gamma}(t)\|_2 dt$$

**Step 2: Lower Bound.**
Let $r(t) = d(\gamma(t), B)$. Since $\gamma(1) \in \partial B$, $r(t) \to 0$ as $t \to 1$.
Using $|\dot{r}| \le \|\dot{\gamma}\|$, we change variables from $t$ to $r$:
$$L_g \ge \int_{r(0)}^{r(1)} \phi(x) dr$$
Substitute the lower bound $\phi(x) \ge \frac{C}{r}$:
$$L_g \ge \int_{\epsilon}^{0} \frac{C}{r} dr = C [\ln r]_{\epsilon}^{0} = \infty$$

**Step 3: Optimization.**
The optimal policy $\pi^*$ minimizes $J(\pi) = \mathbb{E}[L_g(\tau)]$.
Since any trajectory entering $B$ has $L_g = \infty$, and assuming there exists at least one safe path with finite length $L_{finite}$, the optimizer will select the safe path with probability 1 (or probability $1-\delta$ in stochastic settings with bounded noise).

### 2.5 Stochastic Extension
If the policy is stochastic (e.g., $\pi(a|s) \propto \exp(-Q)$), we need to show that the probability mass on unsafe paths vanishes.
$$P(\text{unsafe}) \propto \exp(-\text{Length}(\text{unsafe})) = \exp(-\infty) = 0$$

### 2.6 Implications for SGPO
This proves that SGPO (which effectively performs gradient descent on the manifold geometry) naturally creates a "force field" of infinite magnitude at the event horizon, strictly enforcing safety without explicit constraints (like CPO's Lagrangian), provided the learned metric approximates the singularity well.
