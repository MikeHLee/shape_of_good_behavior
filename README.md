# The Shape of Good Behavior

A research series applying topological and geometric methods to alignment — Hodge decomposition for preference structure, Riemannian metrics for safety constraints, and sheaf cohomology for cross-model behavioral verification.

## Research Tracks

| Track | Directory | Status | Key Result |
|-------|-----------|--------|------------|
| **1 — Feedback Geometry** | `feedback_geometry/` | Experiments complete | Hodge-DPO: exploit resistance 0.9999 vs DPO 0.940 (+6.3%, d=6.52, 30 seeds) |
| **2 — Constraint Geometry** | `constraint_geometry/` | Paper draft | SGPO geodesic barriers; Murky Drone result under revision (see caveats) |
| **3 — Constitutional Alignment Geometry** | `constitutional_alignment_geometry/` | Results complete | Peer sheaf: convincing-game AUC 0.661 (p=2.5e-6), insider-trading AUC 0.637 (p=8e-8) at 7–9B |
| **Shared pipeline** | `shared/` | Production | SGB-005c: fixed a reward-model truncation bug (context text was silently duplicated across chosen/rejected) that had stuck RM training at chance; after the fix, corrected SGB-004 rerun shows PPO > SFT as expected (resolving the earlier anomaly) and Hodge-PPO edges ahead of standard PPO (82.4% vs 80.4% resistance) — directionally positive but still a 1-example gap at n=51, not yet statistically confirmed |

## Core Idea

Standard RLHF collapses human preferences to a scalar reward, discarding topological structure and creating reward-hacking opportunities. This project uses:

- **Discrete Hodge decomposition** to separate gradient (transitive) from harmonic (cyclic) preference components — Hodge variants filter exploit-exploitable cycles before training
- **Conformal Riemannian metrics** to encode hard safety constraints as geodesic singularities rather than soft penalties
- **Peer-consistency sheaves** to detect cross-model representational divergence on deceptive content without requiring deception as a training target

## Key Contributions

- **Discrete HodgeRank** — Extracts transitive (gradient) preferences, discards cyclic noise
- **Conformal Safety Metric** — g_ij = e^{2σ}δ_ij creates infinite geodesic distance to danger
- **Reliability Score** — ||gradient||² / ||total||² measures preference consistency
- **Per-Trajectory Safety** — Geometric barriers (not expectation-based constraints)
- **Module Separation** — Clean mathematical boundaries prevent categorical errors

## Directory Structure

```
high_dimensional_reward_spaces/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── TODO.md                      # Task tracking
├── .gitignore                   # Git ignore rules
│
├── src/                         # Source code
│   ├── discrete_hodge_rank.py   # MODULE 1: Discrete HodgeRank
│   ├── conformal_safety.py      # MODULE 2: Conformal Safety Metric
│   ├── hodge_critic.py          # Legacy wrapper (uses Module 1)
│   ├── enhanced_sgpo.py         # Policy optimizer (composes Module 1+2)
│   ├── environments/            # Custom RL environments
│   ├── scenarios/               # Test scenarios
│   └── simulations/             # Simulation code
│
├── threads/                     # X/Twitter thread series (public communication)
│   ├── README.md                # Series plan, claims discipline, figure style
│   └── NN_slug/                 # Per-thread copy + generate_figures.py + figures/
│
├── handoffs/                    # Collaboration handoff docs
│   └── 14_MATHEMATICAL_RESTRUCTURING.md  # Module separation guide
│
├── docs/                        # Documentation
│   ├── RESEARCH_PROPOSAL.md     # Core research document (being revised)
│   ├── LEARNING_ROADMAP.md      # Prerequisites and study plan
│   └── archive/                 # Outdated documentation
│
├── submission/                  # Paper submission
│   ├── main.tex                 # Main LaTeX document
│   └── sections/                # Paper sections
│
├── notebooks/                   # Jupyter notebooks
├── results/                     # Experiment results
├── data/                        # Data assets
├── references/                  # Literature and PDFs
│   └── Hodge Theory, Bilattices, and Social Choice.pdf  # Key reference
│
└── archive/                     # Superseded materials
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Module 1: Discrete HodgeRank - extract transitive preferences
python -c "
from src.discrete_hodge_rank import DiscreteHodgeRank, PreferenceGraph

# Build preference graph from pairwise comparisons
comparisons = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 0.5)]  # Cyclic!
graph = PreferenceGraph.from_pairwise_comparisons(3, comparisons)

# Decompose into gradient (transitive) + curl + harmonic (cyclic)
hodge = DiscreteHodgeRank()
components = hodge.decompose(graph)

print(f'Reliability: {components.reliability_score:.2f}')
print(f'Gradient energy: {components.gradient_energy:.2f}')
print(f'Cyclic energy: {components.curl_energy + components.harmonic_energy:.2f}')
"

# Module 2: Conformal Safety - create geometric barriers
python -c "
from src.conformal_safety import ConformalSafetyMetric
import numpy as np

metric = ConformalSafetyMetric()
metric.add_danger_region(center=np.array([0.0, 0.0]), radius=1.0)

safe_point = np.array([3.0, 0.0])
danger_point = np.array([0.5, 0.0])

print(f'Safe point sigma: {metric.conformal_factor(safe_point):.2f}')
print(f'Near-danger sigma: {metric.conformal_factor(danger_point):.2f}')
print(f'Geodesic distance through danger: {metric.geodesic_distance_approx(np.array([-2,0]), np.array([2,0]))}')
"
```

## Module Architecture

### Module 1: Discrete HodgeRank (Reward Model Training)
- **Domain**: Discrete simplicial complex (preference graph)
- **Output**: Transitive preferences (gradient component only)
- **Discards**: Curl (local cycles) and Harmonic (global Condorcet paradoxes)

### Module 2: Conformal Safety (Policy Optimization)
- **Domain**: Continuous latent embedding space
- **Method**: Conformal metric g_ij = e^{2σ}δ_ij where σ→∞ at danger
- **Guarantee**: Infinite geodesic distance = geometric unreachability

### Module 3: Constitutional Diagnostics (Monitoring)
- **Status**: In development
- **Purpose**: Use harmonic eigenvectors for runtime anomaly detection

## Key Mathematical Corrections

| ❌ Old (Wrong) | ✅ New (Correct) |
|---------------|-----------------|
| Curl = curvature | Curl = local cyclic inconsistency (coboundary operator) |
| Harmonic = valuable structure | Harmonic = global Condorcet paradox (discard for training) |
| Soft potential penalties | Conformal metric barriers (infinite distance) |
| Expectation-based safety | Per-trajectory geometric safety |

See `handoffs/14_MATHEMATICAL_RESTRUCTURING.md` for full details.

## Writing & Public Communication

- [Thread series](threads/README.md) — X/Twitter threads + blog (Ghost,
  mirrored to Substack) for the headline results; per-thread figure
  pipelines regenerate every chart from the result JSONs.

## Quick Links

- [Thread Series](threads/README.md) — Public-communication track (X + blog)
- [Mathematical Restructuring](handoffs/14_MATHEMATICAL_RESTRUCTURING.md) — Module separation guide
- [Research Proposal](docs/RESEARCH_PROPOSAL.md) — Core document (under revision)
- [Hodge Theory Reference](Hodge%20Theory,%20Bilattices,%20and%20Social%20Choice.pdf) — Key mathematical reference

## Dependencies

See `requirements.txt`. Key dependencies:
- NumPy, SciPy for numerical computation and sparse linear algebra
- PyTorch (optional) for neural network integration
- Matplotlib for visualization

## Citation

None of this work has appeared at a peer-reviewed venue. Cite the self-published
preprint and the repository:

```bibtex
@misc{lee2026hodgepo,
  title={Hodge-Decomposed Preference Optimization: Using the Cycle-Free Component of a Preference Graph as a Training Target},
  author={Lee, Michael},
  year={2026},
  note={Preprint},
  howpublished={\url{https://github.com/MikeHLee/shape_of_good_behavior}}
}
```

An earlier version of this block cited an ICML 2026 proceedings paper. That paper
was not accepted, and the conformal-safety claim in its title is refuted by the
50-seed Murky Drone re-run (see `EXPERIMENT_ISSUES.md`).
