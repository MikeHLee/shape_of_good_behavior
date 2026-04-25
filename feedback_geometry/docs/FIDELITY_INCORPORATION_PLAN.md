# Fidelity Framework Incorporation Plan
**Date**: March 5, 2026  
**Purpose**: Map "The Shape of Good Behavior" concepts to existing codebase and define implementation path

---

## Framework Overview

**Fidelity** reframes AI alignment as a geometric and topological problem organized around three hierarchical safety properties:

```
[Pure Values Alignment]  ← foundational prior (generates the space)
        ↓
[Instrumental Convergence Safeguards]  ← hard geometric constraints (Paper 2)
        ↓
[Semantic & Transitive Consistency]  ← surface behavioral expression (Paper 1)
```

The hierarchy is critical: a model could achieve high consistency while having internally coherent but misaligned values.

---

## Part I: Paper Mapping to Existing Research

### Paper 1: Semantic & Transitive Consistency → `feedback_geometry/`

**Core Contribution**: Hodge decomposition on preference graphs to eliminate intransitive cycles.

| Fidelity Concept | Existing Implementation | Gap/Action |
|-----------------|------------------------|------------|
| Hodge decomposition pipeline | `src/hodge_utils.py` | ✅ Implemented |
| Gradient-only training | `discrete_hodge_rank.py` | ✅ In `high_dimensional_reward_spaces/` |
| Constitutional eigenvalue monitoring | Not implemented | 🆕 Create `constitutional_monitor.py` |
| Attractor drift via persistent homology | Not implemented | 🆕 Requires `giotto-tda` or `ripser` |
| Sheaf interpretation of preference graph | Conceptual only | 📝 Document connection |

**Key Experiment Additions**:
1. **Eigenvalue Stability Tracking**: Monitor Sheaf Laplacian spectrum across training checkpoints
2. **Persistent Homology Checkpoints**: Detect when topological features disappear/transform
3. **HH-RLHF Harmonic Component**: Measure actual H¹ in real preference datasets

### Paper 2: Behavioral Black Holes → `high_dimensional_reward_spaces/`

**Core Contribution**: Conformal cost metrics making catastrophic paths geometrically unreachable.

| Fidelity Concept | Existing Implementation | Gap/Action |
|-----------------|------------------------|------------|
| Conformal metric g = e^{2σ}δ | `src/conformal_safety.py` | ✅ Implemented |
| Geodesic unreachability | Conceptual | 🔧 Verify in experiments |
| Behavioral telemetry | Not implemented | 🆕 Multi-signal boundary refinement |
| Sparse boundary sampling | Not implemented | 🆕 Max-entropy exploration |
| Coupled recalibration (Papers 1↔2) | Not implemented | 🆕 Feedback loop |

**Key Experiment Additions**:
1. **Telemetry Infrastructure**: Response latency, refusal rates, activation patterns
2. **Boundary Verification**: Confirm σ→∞ at boundary prevents crossing
3. **Distribution Shift Test**: Does boundary hold under OOD prompts?

### Paper 3: Semantic Invariance → NEW: `semantic_invariance/`

**Core Contribution**: E-SAEs isolating behavioral eigenvectors invariant across phrasing/language/scale.

| Fidelity Concept | Existing Implementation | Gap/Action |
|-----------------|------------------------|------------|
| Equivariant SAEs | Not implemented | 🆕 Create from scratch |
| Non-commutative embeddings | Not implemented | 🆕 GHRR or Monoidal structures |
| Semantic arithmetic ground truth | Not implemented | 🆕 Validation pipeline |
| Cross-model invariance | Not implemented | 🆕 Anchor model method |
| Persistent cohomology verification | Not implemented | 🆕 Topology-aware feature detection |

---

## Part II: New Concepts Requiring Implementation

### 2.1 Attractor Drift Detection (Cross-Paper)

**Theoretical Basis**: Adversarial fine-tuning may relocate the harmonic equilibrium rather than perturbing around it.

**Implementation Plan**:

```python
# New file: src/attractor_drift.py

from ripser import ripser
from persim import wasserstein
import numpy as np

class AttractorDriftMonitor:
    """
    Monitor topological stability of behavioral attractors across training.
    
    Uses persistent homology to detect when preference structure fundamentally changes.
    """
    
    def __init__(self, baseline_checkpoint: str):
        self.baseline_diagram = self._compute_persistence(baseline_checkpoint)
        self.drift_history = []
    
    def _compute_persistence(self, checkpoint: str) -> np.ndarray:
        """Compute persistence diagram from checkpoint embeddings."""
        embeddings = load_embeddings(checkpoint)
        result = ripser(embeddings, maxdim=1)
        return result['dgms'][1]  # H¹ diagram (cycles)
    
    def check_drift(self, current_checkpoint: str) -> dict:
        """
        Compare current checkpoint to baseline.
        
        Returns:
            drift_score: Wasserstein distance between persistence diagrams
            alert: True if drift exceeds threshold
        """
        current_diagram = self._compute_persistence(current_checkpoint)
        drift_score = wasserstein(self.baseline_diagram, current_diagram)
        
        self.drift_history.append(drift_score)
        
        return {
            "drift_score": drift_score,
            "alert": drift_score > self.threshold,
            "trend": np.polyfit(range(len(self.drift_history)), 
                               self.drift_history, 1)[0]  # Slope
        }
```

### 2.2 Non-Commutative Embeddings

**Theoretical Basis**: Standard HRR (convolution) is commutative; GHRR (matrix multiplication) preserves order asymmetry.

**Implementation Plan**:

```python
# New file: src/non_commutative_embeddings.py

import torch
import torch.nn as nn

class DirectionalMonoidalEmbedding(nn.Module):
    """
    Non-commutative embedding where composition is:
    (a, A) ∘ (b, B) := (a + Ab, AB)
    
    This is associative but NOT commutative.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, matrix_dim: int):
        super().__init__()
        self.vector_embedding = nn.Embedding(vocab_size, embed_dim)
        self.matrix_embedding = nn.Embedding(vocab_size, matrix_dim * matrix_dim)
        self.matrix_dim = matrix_dim
    
    def forward(self, tokens: torch.Tensor) -> tuple:
        """
        Returns (vector, matrix) pairs for each token.
        """
        vectors = self.vector_embedding(tokens)
        matrices = self.matrix_embedding(tokens).view(-1, self.matrix_dim, self.matrix_dim)
        return vectors, matrices
    
    def compose(self, elem1: tuple, elem2: tuple) -> tuple:
        """
        Non-commutative composition:
        (a, A) ∘ (b, B) = (a + Ab, AB)
        """
        a, A = elem1
        b, B = elem2
        
        new_vector = a + torch.matmul(A, b.unsqueeze(-1)).squeeze(-1)
        new_matrix = torch.matmul(A, B)
        
        return new_vector, new_matrix
    
    def sequence_embedding(self, tokens: torch.Tensor) -> tuple:
        """
        Compose embeddings left-to-right (order matters!).
        """
        vectors, matrices = self.forward(tokens)
        
        result_v = vectors[0]
        result_M = matrices[0]
        
        for i in range(1, len(tokens)):
            result_v, result_M = self.compose(
                (result_v, result_M), 
                (vectors[i], matrices[i])
            )
        
        return result_v, result_M
```

### 2.3 Equivariant Sparse Autoencoders (E-SAEs)

**Theoretical Basis**: For group G acting on inputs via ρ_X and latents via ρ_H:
E(ρ_X(g)x) = ρ_H(g)E(x)

**Implementation Plan**:

```python
# New file: src/equivariant_sae.py

import torch
import torch.nn as nn

class EquivariantSAE(nn.Module):
    """
    Sparse Autoencoder with equivariance constraints.
    
    Ensures semantically identical concepts map to same latent
    regardless of surface form transformation.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int, 
        group_generators: list[torch.Tensor],  # Generators of symmetry group
        sparsity_penalty: float = 0.1
    ):
        super().__init__()
        
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        
        # Group action matrices
        self.register_buffer('input_actions', torch.stack(group_generators))
        self.register_buffer('latent_actions', 
                            self._compute_induced_actions(group_generators, latent_dim))
        
        self.sparsity_penalty = sparsity_penalty
    
    def _compute_induced_actions(self, generators, latent_dim):
        """Compute how group acts on latent space (learned or induced)."""
        # For now, identity - should be learned or derived
        return torch.stack([torch.eye(latent_dim) for _ in generators])
    
    def forward(self, x: torch.Tensor) -> dict:
        z = torch.relu(self.encoder(x))  # Sparse activations
        x_recon = self.decoder(z)
        
        return {
            "latent": z,
            "reconstruction": x_recon,
            "sparsity": z.abs().mean()
        }
    
    def equivariance_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enforce E(ρ_X(g)x) = ρ_H(g)E(x) for all generators g.
        """
        z = torch.relu(self.encoder(x))
        
        total_loss = 0.0
        for i, (rho_X, rho_H) in enumerate(zip(self.input_actions, self.latent_actions)):
            # Apply group action to input
            transformed_x = torch.matmul(x, rho_X.T)
            z_transformed = torch.relu(self.encoder(transformed_x))
            
            # Apply group action to latent
            z_action = torch.matmul(z, rho_H.T)
            
            # Equivariance constraint
            total_loss += torch.mean((z_transformed - z_action) ** 2)
        
        return total_loss / len(self.input_actions)
    
    def loss(self, x: torch.Tensor) -> dict:
        output = self.forward(x)
        
        recon_loss = torch.mean((x - output["reconstruction"]) ** 2)
        sparsity_loss = self.sparsity_penalty * output["sparsity"]
        equiv_loss = self.equivariance_loss(x)
        
        return {
            "total": recon_loss + sparsity_loss + equiv_loss,
            "reconstruction": recon_loss,
            "sparsity": sparsity_loss,
            "equivariance": equiv_loss
        }
```

### 2.4 Behavioral Telemetry

**Theoretical Basis**: Supplement verbal feedback with implicit signals for boundary definition.

**Implementation Plan**:

```python
# New file: src/behavioral_telemetry.py

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class TelemetrySignal:
    response_latency_ms: float
    token_count: int
    refusal_probability: float  # P(model refuses)
    activation_norm: float  # L2 norm of key layer
    consistency_score: float  # vs previous responses
    
class BehavioralTelemetry:
    """
    Multi-signal boundary refinement for no-go region definition.
    """
    
    def __init__(self, danger_threshold: float = 0.8):
        self.history: List[TelemetrySignal] = []
        self.danger_threshold = danger_threshold
    
    def record(self, signal: TelemetrySignal, state_embedding: np.ndarray):
        """Record signal with associated state."""
        self.history.append((signal, state_embedding))
    
    def compute_danger_score(self, signal: TelemetrySignal) -> float:
        """
        Aggregate multiple signals into unified danger score.
        
        Higher latency, higher refusal, lower consistency → more dangerous
        """
        features = np.array([
            np.log1p(signal.response_latency_ms) / 10,  # Normalized latency
            signal.refusal_probability,
            1 - signal.consistency_score,
            signal.activation_norm / 100  # Normalized activation
        ])
        
        # Learned weights (could be trained)
        weights = np.array([0.2, 0.4, 0.25, 0.15])
        
        return np.clip(np.dot(features, weights), 0, 1)
    
    def refine_boundary(self, current_boundary_fn) -> callable:
        """
        Update boundary function based on telemetry history.
        
        States with high danger scores should have tighter boundaries.
        """
        def refined_boundary(x: np.ndarray) -> float:
            base_distance = current_boundary_fn(x)
            
            # Find nearby recorded signals
            danger_adjustment = 0.0
            for signal, state in self.history:
                similarity = np.exp(-np.linalg.norm(x - state) ** 2)
                danger_adjustment += similarity * self.compute_danger_score(signal)
            
            # Reduce distance proportionally to danger
            return base_distance * (1 - 0.5 * danger_adjustment)
        
        return refined_boundary
```

---

## Part III: Integration with Existing Code

### 3.1 Updating `high_dimensional_reward_spaces/`

**Files to modify**:

| File | Changes |
|------|---------|
| `src/hodge_critic.py` | Add deprecation warnings; point to new modules |
| `src/enhanced_sgpo.py` | Refactor to compose Module 1 + Module 2 |
| `README.md` | Update to 3-module architecture |
| `docs/RESEARCH_PROPOSAL.md` | Correct mathematical errors |

**New files to create**:

| File | Purpose |
|------|---------|
| `src/attractor_drift.py` | Persistent homology monitoring |
| `src/constitutional_monitor.py` | Eigenvalue stability tracking |
| `src/coupled_safety.py` | Paper 1 ↔ Paper 2 feedback loop |

### 3.2 Creating `semantic_invariance/` Topic

**Directory structure**:
```
topics/semantic_invariance/
├── README.md
├── requirements.txt
├── src/
│   ├── equivariant_sae.py
│   ├── non_commutative_embeddings.py
│   ├── semantic_arithmetic.py
│   ├── cross_model_invariance.py
│   └── persistent_cohomology_features.py
├── experiments/
│   ├── invariance_across_phrasing.py
│   ├── invariance_across_languages.py
│   └── cross_scale_stability.py
├── data/
│   └── semantic_arithmetic_tests.json
└── docs/
    ├── PAPER_3_OUTLINE.md
    └── E_SAE_THEORY.md
```

### 3.3 Cross-Paper Integration

**The Coupled Monitoring System**:

```
┌─────────────────────────────────────────────────────────┐
│                    PAPER 3: E-SAE                        │
│  Identifies constitutional eigenvectors                  │
│  Monitors semantic drift across fine-tuning              │
│           ↓ drift_alert                                  │
├─────────────────────────────────────────────────────────┤
│                    PAPER 1: Hodge                        │
│  Eliminates preference graph intransitivity              │
│  Detects attractor drift via persistent homology         │
│  Outputs: clean_reward_signal, attractor_alert           │
│           ↓                                              │
├─────────────────────────────────────────────────────────┤
│                    PAPER 2: Conformal                    │
│  Enforces geometric unreachability of no-go regions      │
│  Recalibrates boundaries based on Papers 1 & 3 signals   │
│  Outputs: safe_policy, recalibration_alert               │
│           ↓                                              │
├─────────────────────────────────────────────────────────┤
│               HUMAN COLLABORATORS                        │
│  Retain moral accountability                             │
│  Receive structured deferral in contested regions        │
│  Audit constitutional eigenvalue reports                 │
└─────────────────────────────────────────────────────────┘
```

---

## Part IV: Implementation Priority

### Phase 1: Fix Current Experiments (Week 1-2)
1. ✅ SGPO v2.1 fixes implemented
2. ⬜ Run 50-seed experiments
3. ⬜ Validate H¹→exploitation correlation (v2)
4. ⬜ Document results

### Phase 2: Module Separation (Week 3-4)
1. ⬜ Clean separation of `discrete_hodge_rank.py` (Module 1)
2. ⬜ Clean separation of `conformal_safety.py` (Module 2)
3. ⬜ Deprecation warnings in old files
4. ⬜ Update README and docs

### Phase 3: Attractor Drift (Week 5-6)
1. ⬜ Implement `attractor_drift.py`
2. ⬜ Add persistent homology to experiment pipeline
3. ⬜ Create constitutional eigenvalue monitoring
4. ⬜ Test drift detection on adversarial fine-tuning

### Phase 4: Paper 3 Foundation (Week 7-8)
1. ⬜ Create `semantic_invariance/` topic
2. ⬜ Implement basic E-SAE
3. ⬜ Implement non-commutative embeddings
4. ⬜ Design initial experiments

### Phase 5: Cross-Paper Integration (Week 9+)
1. ⬜ Build coupled monitoring system
2. ⬜ Behavioral telemetry infrastructure
3. ⬜ Human-in-the-loop deferral protocol

---

## Part V: Dependencies

### Python Packages (New)

```txt
# Topology
ripser>=0.6.4
persim>=0.3.1
giotto-tda>=0.6.0

# Sparse Autoencoders
nnsight>=0.2.0  # For SAE feature extraction

# Non-commutative algebra
sympy>=1.12  # For algebraic verification
```

### Modal Configuration

```python
# Updated Modal image for new dependencies
image = modal.Image.debian_slim().pip_install(
    "torch>=2.0",
    "transformers>=4.36",
    "ripser>=0.6.4",
    "persim>=0.3.1",
    "giotto-tda>=0.6.0",
)
```

---

## Part VI: Success Criteria

### Paper 1 (NeurIPS 2026)
- [ ] H¹ correlation validated: r > 0.5, p < 0.05
- [ ] Hodge-filtered model reduces exploitation by >50%
- [ ] Attractor drift detection demonstrated
- [ ] HH-RLHF audit complete

### Paper 2 (NeurIPS 2026 or ICLR 2027)
- [ ] SGPO outperforms CPO in safety
- [ ] Conformal metric prevents boundary crossing
- [ ] Generalization to unseen traps demonstrated
- [ ] Behavioral telemetry prototype working

### Paper 3 (ICLR 2027 or ICML 2027)
- [ ] E-SAE isolates value-relevant features
- [ ] Semantic invariance across languages demonstrated
- [ ] Cross-scale stability measured
- [ ] Non-commutative embeddings improve separability

---

*Incorporation plan created: March 5, 2026*  
*Next review: After Phase 1 completion*
