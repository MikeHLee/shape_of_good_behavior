# Section 3: Sheaf-Theoretic Reward Spaces

This section formalizes our framework for representing reward functions as sections of a sheaf over the trajectory space. We introduce the core components: the Sheaf Reward Model (SRM), the Hodge Decomposition for detecting inconsistencies, and the Sheaf-Geodesic Policy Optimization (SGPO) algorithm for safe navigation.

## 3.1 Mathematical Framework

### 3.1.1 The Trajectory Sheaf
Let $\mathcal{T}$ be the space of all possible trajectories in an environment. We construct a topology on $\mathcal{T}$ where open sets $U \subseteq \mathcal{T}$ are collections of trajectories (e.g., all trajectories passing through a specific state).

**Definition 3.1 (Reward Sheaf).** The reward sheaf $\mathcal{F}$ assigns to each open set $U$ a vector space $\mathcal{F}(U)$ of local reward valuations. For $V \subset U$, the restriction map $\rho_{U \to V}: \mathcal{F}(U) \to \mathcal{F}(V)$ describes how broad judgments constrain specific instances.

A **global section** $s \in \mathcal{F}(\mathcal{T})$ represents a fully consistent reward function defined over the entire domain. The **sheaf cohomology group** $H^1(\mathcal{T}, \mathcal{F})$ measures the obstruction to gluing local reward observations into a global reward function.

### 3.1.2 Hodge Decomposition of Feedback
Human feedback often contains contradictions (e.g., cyclic preferences $A \succ B \succ C \succ A$). Standard scalar reward models fail to capture this, averaging conflicting signals into noise. We model the reward signal $R$ as a 1-form on the trajectory graph and apply the discrete Hodge decomposition:

$$R = dV + \delta\psi + \omega$$

Where:
1.  **Gradient ($dV$)**: The consistent, integrable part of the reward. $V$ corresponds to the scalar value function $V(s)$. $\oint dV = 0$ along any loop.
2.  **Curl ($\delta\psi$)**: Local rotational components representing inconsistent preferences within a small neighborhood (e.g., local non-transitivity).
3.  **Harmonic ($\omega$)**: Global topological invariants representing fundamental cycles in the preference structure (Condorcet cycles). $\omega \in \text{ker}(\Delta_1) \cong H^1$.

**Theorem 3.1 (Consistency).** A reward signal $R$ is consistent (i.e., derived from a single scalar value function) if and only if $\|\delta\psi\| = 0$ and $\|\omega\| = 0$.

We implement this via the `HodgeCritic`, which computes the decomposition on the feedback graph. The "clean" reward signal used for policy updates is $R_{\text{clean}} = dV + \omega$ (preserving the global topological structure while denoising local curl), or strictly $dV$ if enforcing scalar consistency is desired.

## 3.2 Geometric Safety via Black Holes

Traditional constrained RL (e.g., CPO) treats safety as a cost limit $\mathbb{E}[C] \le d$. This allows for catastrophic failures with low probability. We propose a geometric approach where forbidden regions are modeled as singularities in the Riemannian manifold of the state space.

**Definition 3.2 (Safety Metric).** Let $\mathcal{B} \subset \mathcal{S}$ be the set of "Black Hole" states (catastrophic failures). We define a Riemannian metric $g_{ij}(s)$ on the state manifold:

$$g_{ij}(s) = \left( 1 + \sum_{b \in \mathcal{B}} \frac{\alpha}{d(s, b)^k} \right) \delta_{ij}$$

Where $d(s, b)$ is the Euclidean distance to the black hole, $\alpha$ is a severity parameter, and $k \ge 2$ controls the sharpness.

**Theorem 3.2 (Geodesic Avoidance).** For $k \ge 2$, the geodesic distance from any safe state $s_{safe}$ to any state $b \in \mathcal{B}$ is infinite. Consequently, any policy that minimizes geodesic path length will avoid $\mathcal{B}$ with probability 1.

This creates a natural "force field" that repels the agent from unsafe regions without requiring explicit boundary constraints in the optimization problem.

## 3.3 Sheaf-Geodesic Policy Optimization (SGPO)

We introduce Sheaf-Geodesic Policy Optimization (SGPO), an algorithm that optimizes policies to follow the gradient of the consistent value function $V$ while minimizing path length on the safety manifold $(M, g)$.

**Algorithm 1: Sheaf-Geodesic Policy Optimization**
1.  **Collect Trajectories**: $\tau \sim \pi_{\theta_{old}}$
2.  **Construct Reward Sheaf**:
    *   Embed states $s_t \to e(s_t)$ via semantic encoder.
    *   Compute Hodge decomposition of feedback $R = dV + \delta\psi + \omega$.
    *   Extract consistent gradient $\nabla \phi = dV$.
3.  **Update Safety Metric**:
    *   Identify black holes $\mathcal{B}$ from negative feedback.
    *   Update metric $g(s)$ parameters to effectively "puncture" the manifold at $\mathcal{B}$.
4.  **Policy Update**:
    Maximize objective:
    $$J(\theta) = \mathbb{E}_{\tau} \left[ \sum_t \langle \nabla \phi(s_t), \Delta e_t \rangle_g - \lambda L_g(\tau) \right]$$
    Where $\langle \cdot, \cdot \rangle_g$ is the inner product under metric $g$, and $L_g(\tau)$ is the Riemannian path length.

In practice, we implement the metric constraint by scaling the advantages:
$$A_{SGPO}(s, a) = \frac{A_{PPO}(s, a)}{\sqrt{g(s)}}$$
This naturally dampens updates in high-curvature (dangerous) regions, preventing the policy from stepping into black holes.

## 3.4 Multi-Perspective Conflict Resolution

When multiple evaluators provide conflicting feedback, we model them as sections of a sheaf over a base space of "perspectives" $P$. The `SheafResolver` computes the cohomology $H^0(P, \mathcal{F})$ to find the consensus reward. If $H^1(P, \mathcal{F})$ is large, it indicates irreducible conflict. In such cases, SGPO decomposes the policy into perspective-conditioned components rather than forcing a potentially disastrous compromise.
