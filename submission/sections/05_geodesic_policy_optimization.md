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
