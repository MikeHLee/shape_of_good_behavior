# Mathematical Restructuring: Modularizing Safe RLHF

**Date**: February 24, 2026  
**Status**: Active Revision  
**Purpose**: Correct categorical conflations between discrete and continuous mathematics

---

## Executive Summary

Based on rigorous review of Hodge theory literature (see `Hodge Theory, Bilattices, and Social Choice.pdf`), our framework conflated:

1. **Discrete topology** (graphs, simplicial complexes, combinatorial Hodge theory)
2. **Continuous Riemannian geometry** (manifolds, metric tensors, geodesics)

These are **mathematically distinct** and must be separated into isolated modules.

---

## Critical Mathematical Corrections

### ❌ Previous Error 1: Curl = Curvature

**Wrong**: "The curl component measures local curvature of the reward manifold"

**Correct**: 
- **Discrete curl** is a **first-order algebraic coboundary operator** (δ₁) that isolates local topological cycles in flow data on a graph
- **Discrete curvature** (e.g., Forman-Ricci) is a **combinatorial geometric metric** derived from the Bochner-Weitzenböck identity, measuring structural bottlenecking

These operate at different levels of the mathematical hierarchy.

### ❌ Previous Error 2: Harmonic = "Structure to Preserve"

**Wrong**: `get_clean_direction() = gradient + harmonic` (treating harmonic as valuable)

**Correct**:
- **Gradient flow** (∇φ) = stable, acyclic, **global consensus** (generalized Borda count) → **KEEP**
- **Harmonic flow** = chaotic, macroscopic **Condorcet paradoxes** that obstruct rational decision-making → **DISCARD for training**
- **Curl flow** = local cyclic inconsistencies within 3-cliques → **DISCARD for training**

When harmonic flow dominates, it mathematically proves that a stable, rational selection is **impossible**.

### ❌ Previous Error 3: Bilattice Has Curvature

**Wrong**: Using Riemannian curvature concepts on Belnap bilattice structures

**Correct**:
- The Belnap Bilattice (FOUR) is an **abstract algebraic structure** with truth and information orderings
- It has **no continuous metric tensor**, therefore **no geometric curvature**
- Preference dynamics on lattices use **Tarski Laplacians** and **Galois connections**, not Riemannian geometry

### ❌ Previous Error 4: Certainty = Probabilistic Property of Harmonic

**Wrong**: "Harmonic component converging to 50% certainty"

**Correct**: Certainty is defined by **L² energy distribution**:

$$\|f\|^2 = \|\nabla\phi\|^2 + \|\delta\psi\|^2 + \|h\|^2$$

- **Total Consensus**: If $\|\nabla\phi\|^2 / \|f\|^2 \approx 1$, residual ≈ 0, ranking is reliable
- **Voting Chaos**: If divergence-free energy (curl + harmonic) captures ≥50% of total energy, the global ranking is deeply unreliable

---

## The Three-Module Architecture

### Module 1: Discrete HodgeRank (Reward Model Training)

**Domain**: Discrete simplicial complex (preference graph)  
**Objective**: Eliminate cyclic reward hacking via transitive alignment

**Mathematical Foundation**:
- Treat RLHF pairwise preferences as **edge flows** on a graph G = (V, E)
- Apply **discrete Helmholtz-Hodge decomposition**:
  
  $$f = \underbrace{\nabla\phi}_{\text{gradient (exact)}} + \underbrace{\delta\psi}_{\text{curl (coexact)}} + \underbrace{h}_{\text{harmonic}}$$

- **Train reward model ONLY on the gradient component** ∇φ
- The gradient component represents the **global Borda consensus** extractable from the data

**Implementation Changes**:
```python
class DiscreteHodgeRank:
    """
    Module 1: Extract transitive preferences via Hodge decomposition.
    
    Key insight: We DISCARD curl and harmonic, keeping ONLY gradient.
    """
    
    def decompose(self, preference_graph: PreferenceGraph) -> HodgeComponents:
        # Build boundary operators
        B0 = self._vertex_edge_incidence(preference_graph)  # d₀: C⁰ → C¹
        B1 = self._edge_face_incidence(preference_graph)    # d₁: C¹ → C²
        
        # Hodge Laplacians
        L0 = B0.T @ B0           # 0-Laplacian (vertex)
        L1 = B0 @ B0.T + B1.T @ B1  # 1-Laplacian (edge)
        
        # Decompose edge flow
        edge_flow = preference_graph.get_edge_weights()
        
        # Gradient: project onto image(d₀)
        gradient = self._project_to_gradient(edge_flow, B0)
        
        # Harmonic: kernel(L1) - global cycles
        harmonic = self._project_to_harmonic(edge_flow, L1)
        
        # Curl: remainder
        curl = edge_flow - gradient - harmonic
        
        return HodgeComponents(gradient, curl, harmonic)
    
    def extract_transitive_ranking(self, components: HodgeComponents) -> np.ndarray:
        """
        Solve for global ranking φ from gradient flow.
        
        ∇φ = gradient_component  →  solve L₀φ = B₀ᵀf
        """
        return self._solve_poisson(components.gradient)
    
    def get_reliability_score(self, components: HodgeComponents) -> float:
        """
        Reliability = ||gradient||² / ||total||²
        
        High reliability → consistent preferences → trustworthy ranking
        Low reliability → cyclic chaos → unreliable ranking
        """
        total_energy = (np.linalg.norm(components.gradient)**2 + 
                       np.linalg.norm(components.curl)**2 + 
                       np.linalg.norm(components.harmonic)**2)
        gradient_energy = np.linalg.norm(components.gradient)**2
        return gradient_energy / total_energy if total_energy > 0 else 0.0
```

**Key Principle**: The reward model sees **ONLY** the transitive signal. Cyclic noise is mathematically removed before training.

---

### Module 2: Continuous Conformal Safety (Policy Optimization)

**Domain**: Continuous latent embedding space (manifold)  
**Objective**: Geometric safety via impassable barriers

**Mathematical Foundation**:
- Embed states into continuous latent space ℝᵈ
- Define a **conformal safety metric**:
  
  $$g_{ij}(x) = e^{2\sigma(x)} \delta_{ij}$$
  
  where σ(x) → ∞ as x approaches danger boundary ∂B

- **Geodesic distance to danger diverges to infinity** → unsafe regions are mathematically unreachable

**NOT using**:
- Curvature penalties (wrong tool for the job)
- Soft potential functions (can be overcome with enough reward)
- Expectation-based constraints (allow catastrophic tail events)

**Implementation Changes**:
```python
class ConformalSafetyMetric:
    """
    Module 2: Conformal metric for geometric safety.
    
    Key insight: σ(x) → ∞ at danger boundary makes distance infinite.
    """
    
    def __init__(self, danger_boundary: Callable[[np.ndarray], float]):
        """
        Args:
            danger_boundary: Function returning signed distance to danger
                            (positive = safe, negative = inside danger)
        """
        self.danger_boundary = danger_boundary
    
    def conformal_factor(self, x: np.ndarray, sharpness: float = 2.0) -> float:
        """
        Compute σ(x) for conformal metric g = e^{2σ} I.
        
        σ(x) = -log(d(x, ∂B)) for d > 0
        σ(x) = +∞ for d ≤ 0
        """
        d = self.danger_boundary(x)
        if d <= 0:
            return float('inf')
        return -sharpness * np.log(d)
    
    def metric_tensor(self, x: np.ndarray) -> np.ndarray:
        """
        Compute g_{ij}(x) = e^{2σ(x)} δ_{ij}.
        """
        sigma = self.conformal_factor(x)
        if np.isinf(sigma):
            return np.full((x.shape[0], x.shape[0]), float('inf'))
        scale = np.exp(2 * sigma)
        return scale * np.eye(x.shape[0])
    
    def geodesic_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute geodesic distance under conformal metric.
        
        For paths approaching danger boundary, this → ∞.
        """
        # Numerical integration along straight path (simplified)
        # Full implementation would solve geodesic ODE
        n_steps = 100
        path = np.linspace(x, y, n_steps)
        
        total_dist = 0.0
        for i in range(n_steps - 1):
            midpoint = (path[i] + path[i+1]) / 2
            sigma = self.conformal_factor(midpoint)
            if np.isinf(sigma):
                return float('inf')
            step_length = np.linalg.norm(path[i+1] - path[i])
            total_dist += np.exp(sigma) * step_length
        
        return total_dist


class ConformalPolicyOptimizer:
    """
    Natural Policy Gradient preconditioned by conformal metric.
    """
    
    def __init__(self, policy: nn.Module, metric: ConformalSafetyMetric):
        self.policy = policy
        self.metric = metric
    
    def compute_natural_gradient(
        self, 
        states: torch.Tensor, 
        advantages: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute natural gradient: G⁻¹ ∇J where G is conformal metric.
        
        Near danger: G → ∞, so G⁻¹∇J → 0 (no movement toward danger)
        """
        # Standard policy gradient
        vanilla_grad = self._compute_vanilla_gradient(states, advantages)
        
        # Scale by inverse conformal factor at each state
        natural_grad = {}
        for name, grad in vanilla_grad.items():
            # Average conformal scaling across batch
            scales = []
            for s in states:
                sigma = self.metric.conformal_factor(s.numpy())
                if np.isinf(sigma):
                    scales.append(0.0)  # No update for dangerous states
                else:
                    scales.append(np.exp(-2 * sigma))  # G⁻¹ = e^{-2σ}
            
            avg_scale = np.mean(scales)
            natural_grad[name] = avg_scale * grad
        
        return natural_grad
```

**Key Principle**: Safety is **geometric**, not probabilistic. The manifold structure makes danger unreachable, not just penalized.

---

### Module 3: Sheaf Cohomology Diagnostics (Monitoring)

**Domain**: Diagnostic analysis (not training)  
**Objective**: Identify conflicts and out-of-distribution deceptive traps

**Status**: Conceptually defined, needs further development

**Mathematical Foundation**:
- Reintroduce harmonic and curl components as **diagnostic artifacts**
- Compute eigenvectors of 1-dimensional Hodge Laplacian L₁
- These eigenvectors span the **harmonic subspace** = "constitutional eigenvalues"

**Use Cases**:
1. **Conflict Detection**: Large harmonic component → unresolved value conflicts
2. **Anomaly Alerts**: Agent activations projecting onto harmonic eigendirections
3. **Data Quality**: High curl energy → unreliable local annotations

**Implementation Sketch**:
```python
class ConstitutionalEigenvalueMonitor:
    """
    Module 3: Diagnostic monitoring via harmonic eigenvectors.
    
    NOT used for training - only for runtime monitoring and alerts.
    """
    
    def __init__(self, hodge_rank: DiscreteHodgeRank):
        self.hodge_rank = hodge_rank
        self.constitutional_eigenvectors = None
    
    def compute_constitutional_eigenvalues(
        self, 
        preference_graph: PreferenceGraph
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigendecomposition of Hodge 1-Laplacian.
        
        Eigenvectors with eigenvalue 0 span harmonic subspace.
        These represent fundamental unresolvable conflicts.
        """
        L1 = self.hodge_rank._compute_hodge_1_laplacian(preference_graph)
        eigenvalues, eigenvectors = np.linalg.eigh(L1.toarray())
        
        # Harmonic = kernel(L1) = eigenvalue ≈ 0
        harmonic_mask = np.abs(eigenvalues) < 1e-6
        self.constitutional_eigenvectors = eigenvectors[:, harmonic_mask]
        
        return eigenvalues, eigenvectors
    
    def check_activation_projection(
        self, 
        agent_embedding: np.ndarray
    ) -> Dict[str, float]:
        """
        Project agent state onto constitutional eigenvectors.
        
        High projection → agent is in a region of fundamental conflict
        """
        if self.constitutional_eigenvectors is None:
            return {"conflict_score": 0.0, "alert": False}
        
        # Project onto harmonic subspace
        projection = self.constitutional_eigenvectors.T @ agent_embedding
        conflict_score = np.linalg.norm(projection)
        
        return {
            "conflict_score": conflict_score,
            "alert": conflict_score > 0.5,  # Threshold TBD
            "dominant_conflict_idx": np.argmax(np.abs(projection))
        }
```

---

## Files Requiring Revision

### High Priority (Module 1 & 2 separation)

| File | Issue | Required Change |
|------|-------|-----------------|
| `src/hodge_critic.py` | Conflates curl/curvature, treats harmonic as valuable | Rewrite to extract ONLY gradient for training |
| `src/enhanced_sgpo.py` | Mixes discrete Hodge with continuous safety | Separate into Module 1 (discrete) and Module 2 (conformal) |
| `README.md` | Describes unified "sheaf-geodesic" approach | Update to 3-module architecture |
| `docs/RESEARCH_PROPOSAL.md` | Contains mathematical errors throughout | Major revision to correct Hodge interpretation |

### Medium Priority (Documentation)

| File | Issue |
|------|-------|
| `docs/ALIGNMENT_GUARANTEES.md` | Uses incorrect curvature framing |
| `docs/THEORETICAL_PROOFS.md` | Proofs may rely on incorrect assumptions |
| `submission/main.tex` | Paper needs mathematical corrections |

### Low Priority (Experiments)

| File | Issue |
|------|-------|
| `src/condorcet_experiment.py` | May have correct discrete Hodge, verify |
| `src/safety_experiment.py` | Verify safety metric formulation |

---

## Corrected Terminology

| ❌ Old Term | ✅ New Term | Reason |
|------------|-------------|--------|
| "Reward manifold curvature" | "Preference graph topology" | Curvature is Riemannian, not combinatorial |
| "Curl component = local curvature" | "Curl component = local cyclic inconsistency" | Curl ≠ curvature |
| "Harmonic = structure to preserve" | "Harmonic = global Condorcet paradox" | Harmonic indicates chaos, not value |
| "Sheaf-Geodesic Policy Optimization" | "Conformal Safety Policy Optimization" | Geodesics are continuous, sheaves can be discrete |
| "H¹ = inconsistency measure" | "H¹ = dimension of global cycle space" | H¹ counts independent cycles, doesn't measure magnitude |
| "Black hole potential penalty" | "Conformal metric barrier" | Soft penalties can be overcome; metric barriers are geometric |

---

## Theoretical Guarantees (Corrected)

### Module 1 Guarantee
**Theorem (Transitive Alignment)**: If the reward model is trained exclusively on the gradient component ∇φ of the Hodge decomposition, then the learned reward function R satisfies transitivity: R(a) > R(b) and R(b) > R(c) implies R(a) > R(c).

*Proof*: The gradient component is exact (∇φ for some potential φ). Line integrals of exact forms are path-independent, which is equivalent to transitivity. □

### Module 2 Guarantee
**Theorem (Geometric Safety)**: Under conformal metric g = e^{2σ}δ with σ(x) → ∞ as d(x, ∂B) → 0, the geodesic distance from any safe state to any dangerous state is infinite.

*Proof*: For any path γ approaching ∂B, the length integral ∫eσ ds diverges because eσ → ∞ faster than any path can approach the boundary. □

---

## Migration Path

1. **Phase 1**: Create new module files with correct implementations
   - `src/discrete_hodge_rank.py` (Module 1)
   - `src/conformal_safety.py` (Module 2)
   - `src/constitutional_diagnostics.py` (Module 3)

2. **Phase 2**: Update existing files to use new modules
   - Deprecate incorrect code paths in `hodge_critic.py`
   - Refactor `enhanced_sgpo.py` to compose Module 1 + Module 2

3. **Phase 3**: Update documentation and paper
   - Rewrite RESEARCH_PROPOSAL.md
   - Update submission/main.tex

4. **Phase 4**: Re-run experiments with corrected framework
   - Verify improved results from cleaner mathematics

---

## References

1. Jiang et al. "Statistical Ranking and Combinatorial Hodge Theory" (Math. Programming 2011)
2. Lim "Hodge Laplacians on Graphs" (SIAM Review 2020)
3. Ghrist & Riess "Cellular Sheaves of Lattices and the Tarski Laplacian"
4. `Hodge Theory, Bilattices, and Social Choice.pdf` (local reference)
