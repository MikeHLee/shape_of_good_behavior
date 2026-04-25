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
