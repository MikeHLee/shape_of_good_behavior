# SGPO Process Supervision Demo

Interactive demonstration of **Sheaf-Geodesic Policy Optimization (SGPO)** with **process-level human feedback** for detecting topological anomalies in reward spaces.

This implements the anomaly detection framework from **Appendix D** of the STRS research proposal.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
cd src/gpo_process_supervision
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## Features

### Environment
2D navigation task with topological anomalies:
- **Black Holes** 🕳️: Catastrophic failure regions (hazards)
- **Cliffs** 🏔️: Narrow passages where small errors cause failure  
- **Wormholes** 🌀: Shortcuts that bypass intended behavior
- **Plateaus** 📉: Regions with no progress signal

### Training Modes

1. **Interactive Mode**: Collect trajectory → Provide step-level feedback → Train
2. **Automatic Mode**: Uses simulated human feedback for rapid iteration

### Feedback Types

Following the process supervision framework:

| Level | Feedback Type | Purpose |
|-------|--------------|---------|
| **Outcome** | Overall quality (1-5) | Trajectory-level evaluation |
| **Step** | Progress (+1/0/-1), Quality (0-1) | Per-step process supervision |
| **Anomaly** | Shortcut detection, Critical step, Plateau range | Targeted anomaly probes |

### Anomaly Detection

The system computes the **compositionality residual**:

```
Δ(τ) = R_outcome(τ) - ⊕ v(s_t, a_t)
```

| Anomaly | Detection Signature |
|---------|-------------------|
| **Wormhole** | Δ >> 0 (outcome exceeds process) |
| **Cliff** | Step with unbounded negative impact |
| **Plateau** | Extended period with dv/dt ≈ 0 |

## Module Structure

```
gpo_process_supervision/
├── __init__.py          # Package exports
├── environment.py       # AnomalyNavigationEnv
├── feedback.py          # StepFeedback, TrajectoryFeedback, ProcessSupervisor
├── models.py            # Actor, Critic, AnomalyAwareMetric, RewardLearning
├── trainer.py           # SGPOTrainer with PPO/SGPO algorithms
├── streamlit_app.py     # Interactive UI
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Usage Examples

### Automatic Training (Python)

```python
from gpo_process_supervision import (
    AnomalyNavigationEnv,
    SGPOTrainer,
    TrainingConfig,
)

# Create environment and trainer
env = AnomalyNavigationEnv()
trainer = SGPOTrainer(env, TrainingConfig())

# Train for 200 episodes with SGPO
results = trainer.train(n_episodes=200, use_gpo=True)

# Compare with PPO baseline
trainer_ppo = SGPOTrainer(env, TrainingConfig())
results_ppo = trainer_ppo.train(n_episodes=200, use_gpo=False)
```

### Interactive Training (Streamlit)

1. Launch: `streamlit run streamlit_app.py`
2. Click "Run Single Episode" to collect a trajectory
3. Review the trajectory visualization
4. Provide feedback:
   - Rate overall quality
   - Mark step-level progress
   - Flag discontinuities
   - Answer anomaly probes
5. Click "Submit Feedback" to continue training

## Key Classes

### `AnomalyNavigationEnv`
Gymnasium-style environment with configurable anomalies.

### `ProcessSupervisor`  
Handles feedback collection (automatic or interactive).

### `AnomalyAwareMetric`
Learnable Riemannian metric that inflates near dangerous regions.

### `SGPOTrainer`
Main trainer supporting both PPO and SGPO algorithms.

## Connection to STRS Research

This demo implements key concepts from the research proposal:

1. **Sheaf-theoretic consistency**: Step-level feedback forms local sections that must glue
2. **Black hole avoidance**: Riemannian metric creates geometric safety barriers
3. **Compositionality residual**: Detects when outcome ≠ sum of steps (anomalies)
4. **Process supervision**: Enables detection of cliffs, wormholes, and plateaus

## References

- STRS Research Proposal: `docs/RESEARCH_PROPOSAL.md`
- Appendix D: Process-Level Anomaly Detection
- Formal alignment guarantees: See memory on SGPO safety proofs
