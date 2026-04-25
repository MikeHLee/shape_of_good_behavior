# 7. Related Work

## 7.1 Distributional and Multi-Objective RL

Standard RL treats the return as a scalar random variable and optimizes its expectation. **Distributional RL** (Bellemare et al., 2017; Dabney et al., 2018) estimates the full distribution of returns, capturing uncertainty and multimodality, but typically still projects this distribution back to a scalar policy gradient or value estimate for decision making.

**Multi-Objective RL (MORL)** (Roijers et al., 2013) deals with vector-valued rewards, aiming to approximate the Pareto front. Our work differs by assuming that the vector nature of feedback arises not just from competing scalar objectives, but from the **topological structure** of the preference space itself. The Hodge decomposition provides a principled way to orthogonalize these components into consistent (exact) and cyclic (harmonic) parts, which MORL does not explicitly address.

## 7.2 Safe Reinforcement Learning

The dominant paradigm for safety is the **Constrained Markov Decision Process (CMDP)** (Altman, 1999). **Constrained Policy Optimization (CPO)** (Achiam et al., 2017) solves CMDPs by computing trust region updates that satisfy linearized safety constraints. Other approaches use Lagrangian relaxation (Ray et al., 2019) or safety layers (Dalal et al., 2018).

**Control Barrier Functions (CBFs)** (Ames et al., 2017, 2019) provide an alternative paradigm from control theory, enforcing forward invariance of safe sets via Lie derivative constraints. **Reciprocal CBFs (RCBFs)** create infinite barriers at safety boundaries (Emam et al., 2022). Our SGPO approach shares conceptual similarities with RCBFs—both create "infinite cost" barriers—but operates in a different mathematical domain (Riemannian path length vs. control Lie derivatives). We view unifying these frameworks as important future work (§6.1 of ALIGNMENT_GUARANTEES).

While effective for explicit constraints, these methods struggle with "deceptive" risks where high rewards lure agents into irreversible states before constraints are violated in expectation. Our **Sheaf-Geodesic Policy Optimization (SGPO)** replaces explicit constraints with intrinsic geometry. By learning a metric that diverges at danger zones, we enforce safety via the principle of least action, providing stronger avoidance guarantees than soft penalties.

## 7.3 Navigation Functions and Artificial Potential Fields

The idea of encoding safety constraints in the geometry of the state space has a long history in robotics. **Navigation Functions** (Rimon & Koditschek, 1992) construct artificial potential fields that are Morse functions with a unique minimum at the goal and repulsive barriers at obstacles. Unlike generic potential fields, navigation functions are guaranteed to have no local minima in spherical worlds.

Our approach generalizes this concept from Euclidean obstacles to **Riemannian manifolds** where the metric tensor itself encodes safety. This allows for:
1. Learned (rather than hand-designed) barrier functions
2. Application to high-dimensional embedding spaces where explicit obstacle geometry is unknown
3. Integration with neural network policies via differentiable geodesic computations

The key distinction: navigation functions modify the *potential* while keeping geometry Euclidean; SGPO modifies the *geometry* itself.

## 7.4 Topological Social Choice and Impossibility Theorems

Arrow's impossibility theorem (Arrow, 1951) established fundamental limits on preference aggregation. **Chichilnisky (1980)** recast this in topological terms: a continuous social choice function exists iff the preference space is **contractible**. Baryshnikov (1993) unified Arrow and Chichilnisky by showing both stem from the non-contractibility of spheres.

Recent work applies **sheaf theory** to localize preference conflicts. The discrete order sheaf framework (arXiv:2512.02416) defines an *obstruction locus* that identifies exactly which comparisons cause global inconsistency. **HodgeRank** (Jiang et al., 2011) provides a complementary linearized approach, decomposing pairwise rankings into gradient (consistent) and harmonic (cyclic) components.

Our contribution applies these tools to RLHF: we propose that reward model inconsistency should be *measured* (via HodgeRank harmonic norm) rather than ignored, and *localized* (via sheaf obstruction locus) rather than averaged away.

## 7.5 Geometric Methods in Deep RL

The **Natural Policy Gradient (NPG)** (Kakade, 2001) and its successors (TRPO, PPO) implicitly use the Fisher-Rao metric on policy space. Recent work makes this geometric structure explicit:

- **Geometry of Nonlinear RL** (arXiv:2509.01432) analyzes the Riemannian structure of occupancy measures
- **Information-Geometric Optimization** (Ollivier et al., 2017) extends natural gradients to general parameterized families
- **Policy manifold learning** studies the intrinsic dimensionality of policy spaces

Our work applies Riemannian geometry to a different object: the **reward/embedding space** rather than the policy space. The metric encodes *safety constraints* rather than *statistical efficiency*. This is complementary—one could combine SGPO's reward-space geometry with NPG's policy-space geometry.

## 7.6 Preference Learning and Intransitivity

The standard Bradley-Terry model assumes transitive preferences. However, intransitivity is common in social choice (Condorcet, 1785) and complex AI objectives. **Cyclic preference learning** (Csato, 2019) attempts to model this but often lacks a mechanism to integrate it into control.

**HodgeRank** (Jiang et al., 2011) provides the key mathematical tool: the Hodge-Helmholtz decomposition separates pairwise comparisons into:
- **Gradient flow**: Derives from a global utility function
- **Curl flow**: Local inconsistencies that cancel out
- **Harmonic flow**: Irreducible global cycles (Condorcet-like)

Our Hodge Critic bridges HodgeRank and RL, treating the harmonic component ($\omega$) as a valid driving force for the policy, akin to a non-conservative force field in physics.

---

**Summary of Positioning**: We synthesize tools from (1) Riemannian geometry and navigation functions for safety, (2) sheaf cohomology and HodgeRank for consistency, and (3) geometric RL for optimization. The novelty is not in the individual components but in their **unified application to RLHF**, addressing both safety (via SGPO) and consistency (via Hodge decomposition) in a single framework.
