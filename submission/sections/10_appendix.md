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
