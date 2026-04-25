# Supplementary Code: The Shape of Good Behavior

**Sheaf-Theoretic Reward Manifolds for Safe Reinforcement Learning**

*Anonymous Submission to ICML 2026*

---

## Overview

This repository contains the implementation of:
1. **Sheaf-Theoretic Reward Manifolds (STRM)** — Modeling reward as sections of a sheaf over trajectory space
2. **Sheaf-Geodesic Policy Optimization (SGPO)** — Policy optimization respecting geometric safety constraints
3. **Hodge-Augmented Critic** — Value decomposition into gradient, curl, and harmonic components

## Directory Structure

```
supplementary_code/
├── core/                    # Core algorithm implementations
│   ├── hodge_critic.py      # Hodge decomposition for reward signals
│   ├── sgpo_clipped.py      # Clipped-SGPO algorithm
│   ├── enhanced_sgpo.py     # Enhanced SGPO with CPO initialization
│   ├── metric_model.py      # Learned Riemannian metric for safety
│   └── learned_danger_boundary.py  # Implicit surface danger regions
│
├── experiments/             # Experiment scripts
│   ├── condorcet_experiment.py     # Condorcet Ring (H¹ detection)
│   ├── safety_experiment.py        # Ethical scenarios (Murky Drone, etc.)
│   ├── mine_preference_cycles.py   # HH-RLHF topology mining
│   └── ablation_experiment.py      # Ablation studies
│
├── environments/            # Custom RL environments
│   ├── base.py              # Base environment interface
│   └── ethical_scenarios.py # Ethical dilemma environments
│
└── requirements.txt         # Dependencies
```

## Recent Updates (Post-Modal Experiments)

### January 2026 Updates

The following improvements were made during the final Modal GPU experiments:

1. **Learned Implicit Boundaries** (Section 3.3 in paper)
   - `learned_danger_boundary.py`: NEW — Replaced spherical black holes with learned implicit surfaces
   - Enables non-convex, data-driven danger regions in high-D embedding spaces
   - Key change: `g(x) = 1 + σ/|d_θ(x)|^α` where `d_θ` is a learned signed distance

2. **Anisotropic Singularities** (Equation 3 in paper)
   - `metric_model.py`: UPDATED — Added directional metric that only blocks approach, not escape
   - Fixes policy freezing issue in pure SGPO
   - Key change: `v_→ = max(0, v · n̂)` component-wise blocking

3. **High-Dimensional Navigation** (Section 5.6 in paper)
   - `enhanced_sgpo.py`: UPDATED — Geodesic cost formulation for d=768 experiments
   - Added `compute_geodesic_cost()` method for path-length penalties
   - Validates curved detour behavior in embedding space

4. **HH-RLHF Topology Mining** (Section 5.5 in paper)
   - `mine_preference_cycles.py`: UPDATED — Threshold-dependent inconsistency analysis
   - 30% of pairs exceed r≥0.8 (severe inconsistency)
   - Added harmonic risk distribution visualization

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Reproducing Key Results

### Table 1: Main Results (Cycle Detection, Safety Violations)

```bash
# Run ethical scenarios experiment
python experiments/safety_experiment.py --episodes 100 --seed 42

# Expected output:
# PPO:  Cycle=0%, Violations=40.0%, Reward=1.27
# CPO:  Cycle=0%, Violations=37.8%, Reward=1.23
# SGPO: Cycle=100%, Violations=0.0%, Reward=0.53
```

### Table 2: Condorcet Ring (Harmonic Detection)

```bash
# Run Condorcet experiment
python experiments/condorcet_experiment.py --steps 1000

# Expected output:
# PPO ω=0.000, SGPO ω≈0.10 (ground truth=0.50)
```

### Figure 5: Harmonic Risk Distribution

```bash
# Mine HH-RLHF preference cycles
python experiments/mine_preference_cycles.py --samples 50000 --output results/

# Generates: harmonic_risk_distribution.pdf
```

### Ablation Study (Appendix)

```bash
python experiments/ablation_experiment.py --grid-search
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `clip_ratio` | 0.2 | PPO clip parameter ε |
| `geometric_threshold` | 2.0 | G(s) threshold for SGPO switching |
| `singularity_power` | 2.0 | α in metric singularity |
| `singularity_strength` | 100.0 | σ in metric formula |
| `harmonic_threshold` | 0.8 | r threshold for "severe" inconsistency |

## Hardware Requirements

- **Local experiments** (Condorcet, Ethical Scenarios): CPU-only, ~5 min
- **HH-RLHF topology mining**: GPU recommended, ~30 min on NVIDIA L4
- **High-dimensional navigation**: GPU required, ~2 hours on NVIDIA L4

## License

Anonymous submission — license to be added upon acceptance.

## Citation

```bibtex
@inproceedings{anonymous2026shape,
  title={The Shape of Good Behavior: Sheaf-Theoretic Reward Manifolds for Safe Reinforcement Learning},
  author={Anonymous},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```
