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
