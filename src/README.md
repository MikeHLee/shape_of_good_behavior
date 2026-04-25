# Source Code Documentation

This directory contains the implementation of the Sheaf-Theoretic Reward Spaces framework.

## Core Modules

### Hodge Decomposition & Topology
- **`hodge_critic.py`** — Core Hodge decomposition implementation for reward signals
  - `HodgeCritic` class for separating gradient, harmonic, and curl components
  - Topological gradient computation for policy optimization
  
- **`sheaf_resolver.py`** — Sheaf cohomology computation
  - Consistency checking between local and global evaluations
  - H¹ cohomology computation for detecting reward inconsistencies

- **`embedding_topology_analyzer.py`** — Interpretability tools for embedding spaces
  - Persistent homology analysis
  - Topological feature extraction

### Reinforcement Learning
- **`semantic_mdp_rl.py`** — Semantic MDP with natural language states
  - Integration with language models for state descriptions
  - Semantic action spaces

- **`gpo_clipped.py`** — Clipped Sheaf-Geodesic Policy Optimization
  - Safe policy optimization respecting metric barriers
  - Black hole avoidance via Riemannian geometry

- **`enhanced_gpo.py`** — Enhanced SGPO with additional features
  - Adaptive learning rates
  - Trust region constraints

- **`cpo_to_blackhole.py`** — CPO-initialized black hole regions
  - Automatic unsafe region detection
  - Metric barrier construction

### Models & Architectures
- **`agent_architectures.py`** — Neural network architectures for agents
- **`metric_model.py`** — Learned Riemannian metric models
- **`world_model.py`** — World model implementations
- **`mlx_mamba_agent.py`** — Mamba-based agent using MLX

### Experiments
- **`safety_experiment.py`** — Safety benchmark experiments
- **`safety_experiment_hard.py`** — Challenging safety scenarios
- **`condorcet_experiment.py`** — Methodological verification (Unit Test) for H¹ detection
- **`style_experiment.py`** — High-dimensional style preference experiments
- **`ablation_experiment.py`** — Ablation studies

### Visualization
- **`visualize_embedding_topology.py`** — Embedding space visualization
- **`visualize_hodge_matrix.py`** — Hodge decomposition visualization
- **`visualize_sheaf_zoom.py`** — Sheaf structure visualization
- **`reward_manifold_3d.py`** — 3D reward manifold rendering
- **`projection_flow_demo.py`** — Projection flow visualization
- **`integrated_topology_demo.py`** — Integrated topology dashboard

### Utilities
- **`generate_paper_diagrams.py`** — Generate publication figures
- **`create_architecture_diagram.py`** — Architecture diagram generation
- **`aggregate_metrics.py`** — Metrics aggregation utilities

## Subdirectories

### `environments/`
Custom RL environments for experiments:
- TextWorld-based environments
- Safety-constrained grid worlds
- Semantic action environments

### `scenarios/`
Test scenarios for evaluation:
- Ethical dilemma scenarios
- Preference conflict scenarios
- Safety boundary test cases

### `simulations/`
Simulation code for experiments:
- Monte Carlo simulations
- Policy rollouts
- Trajectory generation

### `gpo_process_supervision/`
Process supervision implementation for SGPO:
- Step-level reward models
- Process reward verification

## Usage

### Running Experiments (Cloud/Modal)
For high-dimensional experiments requiring GPU acceleration (e.g., Style Space, Safety Gym), use the Modal runner:

```bash
# High-Dimensional Style Verification
modal run notebooks/modal_runner/geodpo_experiments.py::high_dim_style_verification

# Full GeoDPO Pipeline (Topology -> Train -> Analyze)
modal run notebooks/modal_runner/geodpo_experiments.py::run_full_pipeline
```

### Local Experiments (Unit Tests)
```python
# Basic Hodge decomposition verification
from condorcet_experiment import CondorcetRingEnv, train_gpo

critic = HodgeCritic(embedding_dim=64)
gradient, harmonic, curl = critic.decompose(reward_signal)

# SGPO optimization
from gpo_clipped import ClippedSGPO

gpo = ClippedSGPO(policy, critic, black_holes)
policy = gpo.optimize(trajectories)
```

## Dependencies

Core dependencies are listed in `../requirements.txt`. Key imports:
- `torch` — Neural network operations
- `numpy`, `scipy` — Numerical computation
- `networkx` — Graph operations for sheaf computation
