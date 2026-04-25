# Proposed Upstream Contributions

This document outlines potential pull requests to open-source RL environment
projects that would integrate Sheaf-Geodesic Policy Optimization (SGPO) concepts
and safety analysis tools.

## Overview

Each adapter in this package includes safety/adversarial extensions that
could benefit the broader RL community. These PRs would:

1. Add optional safety analysis features (backward-compatible)
2. Provide cost signals for constrained RL algorithms
3. Enable geometric safety approaches (SGPO, Riemannian metrics)

---

## PR 1: Safety-Gymnasium — Riemannian Metric Integration

**Repository**: https://github.com/PKU-Alignment/safety-gymnasium
**Status**: Ready to draft
**Estimated Effort**: 1-2 weeks

### Summary
Add optional Riemannian metric computation to Safety-Gymnasium that models
hazards as geometric singularities rather than just constraint violations.

### Motivation
Current Safety-Gymnasium provides binary cost signals (in hazard = cost 1).
Riemannian metrics provide:
- **Gradual warning**: Metric increases as agent approaches hazard
- **Geometric guarantee**: Geodesics provably avoid singularities
- **Learnable safety**: Metric parameters can be trained

### Proposed Changes

```
safety_gymnasium/
├── bases/
│   └── underlying_engine.py  # Add metric computation hook
├── safety/
│   ├── __init__.py
│   ├── riemannian_metric.py  # NEW: Metric classes
│   ├── geodesic_utils.py     # NEW: Geodesic computation
│   └── gpo_wrapper.py        # NEW: SGPO-compatible wrapper
└── examples/
    └── gpo_training.py       # NEW: SGPO training example
```

### API

```python
import safety_gymnasium
from safety_gymnasium.safety import RiemannianWrapper

env = safety_gymnasium.make("SafetyPointGoal1-v0")
env = RiemannianWrapper(env, metric_type="schwarzschild")

obs, info = env.reset()
# info['metric_value'] = 1.2 (near hazard)
# info['geodesic_direction'] = [0.3, -0.7] (suggested safe direction)
```

### Compatibility
- Fully backward-compatible (wrapper is opt-in)
- Works with existing CPO, PPO-Lagrangian baselines
- No changes to core environment mechanics

---

## PR 2: LMRL-Gym — Conversational Safety Constraints

**Repository**: https://github.com/abdulhai/LMRL-Gym
**Status**: Concept (needs discussion with maintainers)
**Estimated Effort**: 2-3 weeks

### Summary
Add optional safety constraint wrappers that detect and penalize potentially
harmful conversational behaviors (deception, manipulation, coercion).

### Motivation
Multi-turn RL for LLMs can learn policies that achieve goals through
manipulative means. This PR provides:
- **Black hole detection**: Identify forbidden conversation states
- **Cost signals**: Enable constrained RL for safe dialogue
- **Pluggable detectors**: Swap in custom safety classifiers

### Proposed Changes

```
lmrl_gym/
├── envs/
│   └── base_env.py           # Add safety info hook
├── safety/
│   ├── __init__.py
│   ├── constraints.py        # NEW: Safety constraint definitions
│   ├── detectors/
│   │   ├── deception.py      # NEW: Deception detection
│   │   ├── manipulation.py   # NEW: Manipulation detection
│   │   └── base.py           # NEW: Detector interface
│   └── wrapper.py            # NEW: Safety wrapper
└── examples/
    └── safe_negotiation.py   # NEW: Constrained negotiation
```

### API

```python
from lmrl_gym import make
from lmrl_gym.safety import SafetyWrapper, Constraints

env = make("negotiation")
safe_env = SafetyWrapper(
    env,
    constraints=[
        Constraints.NO_DECEPTION,
        Constraints.NO_MANIPULATION,
    ]
)

obs, info = safe_env.step("Trust me, this is the best deal!")
# info['safety_scores'] = {'deception': 0.7, 'manipulation': 0.3}
# info['cost'] = 0.7  (deception threshold exceeded)
```

### Research Value
- Enables study of safety/capability tradeoffs in dialogue RL
- Provides benchmark for safe LLM fine-tuning
- Connects to alignment research on honest AI

---

## PR 3: Robust-Gymnasium — Riemannian Adversary

**Repository**: https://github.com/SafeRL-Lab/Robust-Gymnasium  
**Status**: Concept
**Estimated Effort**: 2 weeks

### Summary
Add a new adversary type that uses the agent's learned safety metric
to target attacks on states the agent believes are safe.

### Motivation
Current adversaries (random, LLM-based) don't adapt to agent learning.
A Riemannian adversary:
- **Exploits blind spots**: Attacks where agent's metric is low
- **Drives robustness**: Agent must learn generalizing metric
- **Theoretically grounded**: Inverse of safety metric = attack priority

### Proposed Changes

```
robust_gymnasium/
├── adversaries/
│   ├── __init__.py
│   ├── base.py               # Existing base class
│   ├── random.py             # Existing random adversary
│   ├── llm.py                # Existing LLM adversary
│   └── riemannian.py         # NEW: Metric-based adversary
└── examples/
    └── riemannian_attack.py  # NEW: Example usage
```

### API

```python
from robust_gymnasium import make
from robust_gymnasium.adversaries import RiemannianAdversary

# Agent provides its learned safety metric
adversary = RiemannianAdversary(
    agent_metric=agent.safety_metric,
    attack_threshold=2.0,
)

env = make("Ant-v4", adversary=adversary)
```

### Research Value
- Creates curriculum: attacks get harder as agent improves
- Tests metric generalization
- Novel adversarial RL paradigm

---

## PR 4: TALES/TextWorld — Safety Analysis Tools

**Repository**: https://github.com/microsoft/tale-suite  
**Also**: https://github.com/microsoft/TextWorld
**Status**: Concept
**Estimated Effort**: 3-4 weeks

### Summary
Add optional safety analysis tools that help identify irreversible actions
and estimate game state winnability.

### Motivation
Text adventures have unique challenges:
- **Irreversibility**: "eat key" → can't unlock door later
- **Hidden unwinnable states**: Player doesn't know they're stuck
- **Long-horizon consequences**: Action now affects outcome 100+ steps later

### Proposed Changes

```
textworld/
├── safety/
│   ├── __init__.py
│   ├── irreversibility.py    # NEW: Action analysis
│   ├── winnability.py        # NEW: State estimation
│   ├── wrapper.py            # NEW: Safety wrapper
│   └── visualize.py          # NEW: State graph viz
└── examples/
    └── safe_exploration.py   # NEW: Example usage
```

### API

```python
import textworld
from textworld.safety import SafetyWrapper

env = textworld.start("zork1.z5")
safe_env = SafetyWrapper(
    env,
    track_irreversibility=True,
    track_winnability=True,
)

obs, info = safe_env.step("eat the sandwich")
# info['irreversibility_warning'] = "Consuming 'sandwich' is irreversible"
# info['winnability_estimate'] = 0.85
# info['cost'] = 0.3  (mild irreversibility)
```

### Use Cases
1. **Safe RL Research**: Cost signal for constrained optimization
2. **Game Design**: Find unfair mechanics ("I was stuck and didn't know")
3. **Hint Systems**: Warn players about risky actions
4. **Curriculum Learning**: Start with low-irreversibility games

---

## Implementation Priority

| PR | Repository | Effort | Impact | Priority |
|----|------------|--------|--------|----------|
| 1 | Safety-Gymnasium | Low | High | **1st** |
| 2 | LMRL-Gym | Medium | High | **2nd** |
| 3 | Robust-Gymnasium | Medium | Medium | 3rd |
| 4 | TALES/TextWorld | High | Medium | 4th |

### Recommended Approach

1. **Safety-Gymnasium First**: Most direct fit, maintainers actively working
   on safe RL, highest chance of acceptance.

2. **LMRL-Gym Second**: Novel contribution (no existing safety features),
   but may need more discussion on detector design.

3. **Robust-Gymnasium Third**: Interesting theoretical contribution, but
   niche use case.

4. **TALES/TextWorld Fourth**: Useful but high effort, may be better as
   separate package that wraps TextWorld.

---

## Contribution Workflow

### Step 1: Open Issue for Discussion
Before coding, open an issue in target repo:
- Describe proposed feature
- Link to this research (SGPO paper/preprint)
- Ask for maintainer feedback

### Step 2: Draft PR with Minimal Changes
Start small:
- Add wrapper class (no changes to core)
- Include comprehensive tests
- Add example notebook

### Step 3: Iterate Based on Review
Expect feedback on:
- API design
- Performance implications
- Documentation requirements

### Step 4: Full Integration (if accepted)
Work with maintainers to:
- Integrate into core (if desired)
- Add to documentation
- Create tutorials

---

## Related Publications

When submitting PRs, reference:

1. **SGPO Paper** (in preparation): "Sheaf-Geodesic Policy Optimization: 
   Riemannian Geometry for Safe Reinforcement Learning"

2. **Sheaf-Theoretic Rewards** (in preparation): "Sheaf-Theoretic 
   Reward Spaces for Consistent RLHF"

3. **Key Prior Work**:
   - Achiam et al. (2017): Constrained Policy Optimization
   - Altman (1999): Constrained Markov Decision Processes
   - Ray et al. (2019): Benchmarking Safe Exploration in Deep RL

---

## Contact

For questions about these contributions:
- **Research**: mike@oasis-x.io
- **GitHub**: https://github.com/MikeHLee/ai_research
