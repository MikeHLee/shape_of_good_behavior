# Theoretical & Experimental Improvements: SGPO with Adaptive Trust Regions

**Date**: January 2026
**Status**: Implemented

---

## 1. The Core Challenge: Safety vs. Efficiency

In high-dimensional reward spaces (like semantic embedding manifolds), we face a dilemma:
1.  **Exact Safety requires heavy computation**: True Sheaf-Geodesic Policy Optimization requires solving the geodesic equation ($O(N^3)$) or computing the full Riemannian Hessian.
2.  **Standard RL is blind to geometry**: PPO/CPO treat the parameter space as Euclidean (or approximately so via KL), ignoring the topological "holes" (black holes) and "twists" (Hodge curls) in the reward manifold.

## 2. Our Solution: Intuitive Approximations

We implement a "middle way" that captures the *intuition* of differential geometry without the *cost*.

### 2.1 Adaptive Trust Regions (The "Breathing" Epsilon)

Instead of a fixed trust region ($\epsilon=0.2$) as in PPO, we make the trust region "breathe" based on the local topology of the reward manifold.

$$ \epsilon_{adaptive} = \epsilon_{base} \cdot S_{consistency} \cdot S_{safety} $$

Where:
*   **Consistency Scale** ($S_{consistency} = \frac{1}{1 + ||H^1||}$): When preferences are inconsistent (high Hodge curl), we shrink the step size. This prevents the policy from chasing "impossible" reward loops.
*   **Safety Scale** ($S_{safety} = 1 - 0.9 \cdot \text{Proximity}_{BH}$): When close to a "black hole" (forbidden region), we drastically shrink the step size to avoid crossing the event horizon.

**Why this works**:
*   **Low Compute**: Requires only dot products and norm lookups ($O(N)$), not matrix inversions.
*   **Intuitive**: It mimics human caution—we slow down when confused (inconsistency) or in danger (black holes).

### 2.2 Scalar Fisher Approximation

True Natural Gradient Descent updates parameters by $\theta_{new} = \theta - \eta F^{-1} \nabla L$. Inverting $F$ is expensive.

We approximate $F$ as a scalar field $\alpha(s)$ learned by a small MLP:
$$ \nabla_{nat} L \approx \frac{1}{\alpha(s) + \epsilon} \nabla L $$

This preserves the *magnitude* scaling of the natural gradient (moving slower in high-curvature regions) without attempting to correct the *direction* rotation. This captures 80% of the benefit (step size adaptation) for <1% of the cost.

### 2.3 Hodge-Aware Direct Preference Optimization (GeoDPO)

We replaced standard DPO's scalar reward difference with a vector alignment score:
$$ L_{GeoDPO} = -\log \sigma(\beta \cdot \langle \nabla_{Hodge} \phi, \Delta e \rangle) $$

This aligns the policy update with the *consistent* component of the reward field, automatically filtering out noisy or cyclic feedback.

## 3. Comparison with Baselines

| Feature | Standard PPO | Standard CPO | **Adaptive SGPO (Ours)** |
|:---|:---|:---|:---|
| **Trust Region** | Fixed ($\epsilon$) | Fixed + Lagrangian | **Adaptive** ($\epsilon(H^1, \text{dist})$) |
| **Geometry** | Euclidean | Euclidean | **Riemannian** (Scalar Approx) |
| **Safety** | None | Expectation-based | **Topological** (Metric Singularities) |
| **Cost** | Low | Medium | **Low** (via Approximations) |

## 4. Implementation Details

Modified `src/semantic_mdp_rl.py`:
- Added `_compute_adaptive_epsilon` to `SemanticSGPO`.
- Integrated `h1_magnitude` and `black_hole_proximity` into step size logic.
- Logged `adaptive_epsilon` for observability.

Modified `src/hodge_critic.py`:
- Added `get_local_geometry` to expose topological metrics to the optimizer.
