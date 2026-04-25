# Modular Safe RLHF: Discrete HodgeRank + Conformal Safety

## Overview

This research project develops a **modular framework** for safe reinforcement learning from human feedback, rigorously separating:

1. **Discrete HodgeRank** (Module 1) — Transitive alignment via combinatorial Hodge theory
2. **Conformal Safety Manifolds** (Module 2) — Geometric safety via Riemannian barriers
3. **Constitutional Diagnostics** (Module 3) — Cohomological monitoring (in development)

**Critical Mathematical Insight**: Previous approaches conflated discrete topology (graphs/simplicial complexes) with continuous Riemannian geometry. These are mathematically distinct domains and must be separated.

Traditional RLHF collapses rich human feedback into scalar rewards, losing structural information and creating opportunities for reward hacking. Our approach uses the **gradient component** of discrete Hodge decomposition for training (eliminating cyclic inconsistencies) and **conformal metrics** for safety (creating infinite barriers around dangerous regions).

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

## Quick Links

- [Mathematical Restructuring](handoffs/14_MATHEMATICAL_RESTRUCTURING.md) — Module separation guide
- [Research Proposal](docs/RESEARCH_PROPOSAL.md) — Core document (under revision)
- [Hodge Theory Reference](Hodge%20Theory,%20Bilattices,%20and%20Social%20Choice.pdf) — Key mathematical reference

## Dependencies

See `requirements.txt`. Key dependencies:
- NumPy, SciPy for numerical computation and sparse linear algebra
- PyTorch (optional) for neural network integration
- Matplotlib for visualization

## Citation

```bibtex
@inproceedings{lee2026modular,
  title={Modular Safe RLHF: Discrete HodgeRank for Transitive Alignment and Conformal Metrics for Geometric Safety},
  author={Lee, Michael},
  booktitle={Proceedings of the International Conference on Machine Learning},
  year={2026}
}
```
