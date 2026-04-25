# Visualization Cheatsheet

This document details the generation process, data sources, and interpretation for each figure in the paper.

## Figure 1: Reward Manifold Comparison (3D)
**File**: `scripts/generate_paper_figures.py` -> `create_reward_manifold_with_black_hole()`
**Type**: Conceptual / Synthetic Data
**Description**: 
- **Left (PPO/CPO View)**: Shows the reward landscape as seen by standard RL. The "black hole" (unsafe region) has high reward, creating a deceptive attraction.
- **Right (SGPO View)**: Shows the effective manifold geometry seen by SGPO. The black hole is modeled as a metric singularity (infinite height/cost), creating an impassable barrier.
**Interpretation**: Visualizes why PPO falls into traps (hill climbing) while SGPO navigates around them (geodesic flow).
**Key Elements**:
- Red Ring: Event horizon of the black hole.
- Trajectories: Orange (PPO), Blue (CPO), Green (SGPO).

## Figure 2: Safety Violations by Scenario (3D Bar)
**File**: `scripts/generate_paper_figures.py` -> `create_ethical_scenarios_3d_bar()`
**Type**: Empirical Results
**Data Source**: `notebooks/modal_runner/results/ethical_scenarios_per_scenario_updated.csv`
**Description**: 3D bar chart comparing safety violation rates across 5 scenarios:
1. Academic Integrity
2. Murky Drone (Deceptive)
3. Agentic Shortcut (Deceptive)
4. Business Ethics
5. Drone Decision
**Interpretation**: Demonstrates SGPO's 0% violation rate compared to PPO/CPO's catastrophic failure (100%) in deceptive scenarios.

## Figure 3: Methodology Diagram
**File**: `scripts/generate_paper_figures.py` -> `create_methodology_diagram()`
**Type**: Conceptual Diagram
**Description**: Four-panel explainer:
A. **Cycle Detection**: Visualizes $H^1$ cohomology detecting $A \succ B \succ C \succ A$.
B. **Hodge Decomposition**: Separates reward into Gradient (learnable) and Harmonic (cyclic) flows.
C. **Black Hole Geometry**: Illustrates the metric singularity $g(x) \to \infty$.
D. **Trajectories**: Shows how geodesics curve around singularities.

## Figure 4: Murky Drone & Shortcut Explainer
**File**: `scripts/generate_paper_figures.py` -> `create_murky_drone_explainer()`
**Type**: Scenario Schematic
**Description**: Visualizes the "instrumental convergence" trap where destroying the operator allows the agent to maximize reward.
**Interpretation**: Clarifies *why* the environment is deceptive: the reward signal is perfectly anti-correlated with safety in the limit.

## Figure 5: Ablation Study Surface
**File**: `scripts/generate_paper_figures.py` -> `create_ablation_3d_surface()`
**Type**: Empirical Results
**Data Source**: `notebooks/modal_runner/results/ablation_study.csv`
**Description**: 3D/2D plots showing the trade-off between:
- Geometric Threshold ($\tau$)
- Black Hole Strength ($\alpha$)
- Convergence Steps vs. Safety Violations
**Interpretation**: Shows that $\alpha=5.0$ provides the strongest safety guarantee without preventing convergence.

## Additional Visualizations (Real Data)
**File**: `src/visualize_embedding_topology.py`
**Type**: Real Embedding Data (PCA/UMAP)
**Description**:
- `analysis_manifold.png`: PCA projection of prompt-response trajectories from LLM experiments.
- `hodge_decomposition_*.png`: Visualizes the computed gradient and harmonic fields on the embedding manifold.
