# Constraint Geometry: Geodesic Policy Optimization via Riemannian Safety Metrics

**Research Track 2 of 3 — "The Shape of Good Behavior" Series**

---

## Overview

This paper focuses on the *constraint geometry* of safe reinforcement learning: how to encode hard safety requirements as geometric structure of the reward manifold, rather than as soft Lagrangian penalties.

The central insight is that forbidden regions — behaviors an agent must never exhibit — can be modeled as singularities in a Riemannian metric. When a learned conformal factor σ(x) → ∞ near dangerous regions, geodesic paths (the natural "straight lines" in the metric space) have infinite length through those regions. A policy that minimizes expected path length *cannot* enter a forbidden region — the geometry itself enforces the constraint.

This gives us **Sheaf-Geodesic Policy Optimization (SGPO)**: a policy gradient method that replaces the Fisher information metric with a learned safety metric G, making the Riemannian gradient ∇_G J = G⁻¹∇J automatically safety-aware. Combined with the Hodge-augmented critic from the Feedback Geometry track, SGPO handles both cyclic rewards and geometric constraints in a unified algorithm.

---

## Core Contributions

1. **Conformal Safety Metric**: Learn a Riemannian metric g(x) = e^{2σ(x)}I from cost signals; σ → ∞ near forbidden regions creates geometric hard constraints
2. **Geodesic Avoidance Theorem**: Prove that finite-length geodesics cannot enter black hole regions (Theorem 4.2); analogous to Reciprocal Control Barrier Functions
3. **SGPO Algorithm**: Full algorithm combining Riemannian policy gradient with Hodge-augmented critic; provably more sample-efficient than CPO on geometric safety tasks
4. **Metric Learning from Costs**: End-to-end differentiable pipeline for learning σ from sparse cost signals, without hand-designed constraint functions
5. **Murky Drone & Agentic Shortcut**: Two new hard safety scenarios where SGPO achieves 0% violations while PPO and CPO both fail

---

## Relationship to Other Research Tracks

| Track | Focus | Key Tool |
|-------|-------|----------|
| Feedback Geometry | Inconsistency in feedback data | H¹ cohomology, Hodge decomposition |
| **This paper (Constraint Geometry)** | Safe policy optimization | Geodesic policy gradient, metric singularities |
| Constitutional Alignment Geometry | Values in embedding space | Alignment differentials, constitutional vectors |

The Hodge-augmented critic developed in the Feedback Geometry track is used here as a component of SGPO, but this paper's primary contribution is the geometric safety mechanism, not the feedback analysis.

---

## Source Code References (from `topics/shape_of_good_behavior/`)

- `src/semantic_mdp_rl.py` — core PPO/CPO/SGPO algorithm implementations (~900 lines)
- `src/learned_danger_boundary.py` — black hole learning from cost signals (~550 lines)
- `src/metric_model.py` — conformal factor σ neural network (~550 lines)
- `src/enhanced_sgpo.py` — enhanced SGPO with clipping and escape mechanisms
- `src/sgpo_clipped.py` — SGPO with trust region clipping
- `src/safety_experiment.py` — sandbagging trap benchmark
- `src/safety_experiment_hard.py` — murky drone and agentic shortcut benchmarks
- `src/cpo_to_blackhole.py` — converting CPO constraints to geometric black holes
- `notebooks/colab_02_geodpo_training.ipynb` — training pipeline
- `notebooks/gpo_gpu_training.ipynb` — GPU-scale training

---

## Results Summary (as of 2026-07-23)

### Current Experimental Status

> ❌ **The Murky Drone headline claim was REFUTED on 2026-07-21.** The original
> "SGPO 0% violations vs PPO/CPO 100%" figure came from a one-step bandit at a
> single seed. A proper 50-seed multi-step re-run shows **CPO is the safest
> method**; SGPO's `advantage/sqrt(g)` formulation is **statistically
> indistinguishable from unconstrained PPO** (p=0.68). See `../EXPERIMENT_ISSUES.md`.

| Experiment | Status | Result |
|------------|--------|--------|
| Sandbagging Trap | ⚠️ Single seed | SGPO +1.53 return vs PPO -6.67, CPO -6.23 |
| Murky Drone | ❌ Claim withdrawn | CPO safest (180.7 violations/seed); SGPO ≈ PPO (p=0.68) |
| Agentic Shortcut | ❌ Unverified | Only one-step bandit; no multi-step re-run |
| Safety Gym | Planned | Not yet run |
| Metric Learning Ablations | Planned | Not yet run |

The geometric theory (conformal metrics, geodesic barriers, Theorem 4.2) is sound. The gap is that the current SGPO implementation does not yet instantiate the full Riemannian policy gradient — it approximates it. The Murky Drone re-run shows the approximation is not yet tight enough to beat CPO.

### PPO Fine-tuning (SGB-003, 2026-07-23, A100-40GB)

The `shared/src/lm_finetuning.py` PPO loop has been verified end-to-end on Modal:

| Run | mean_reward_final | Status |
|-----|------------------|--------|
| Standard PPO | 3.6897 | success |
| Hodge-PPO | -1.2170 | success (exploit suppression) |

This establishes the base fine-tuning infrastructure for future SGPO-on-LM experiments.

## Key Experiments

1. **Sandbagging Trap** — deceptive 2D navigation: SGPO (+1.53 return) vs PPO (-6.67) vs CPO (-6.23), **single seed**
2. **Murky Drone** — ❌ **claim withdrawn.** 50-seed result: PPO 471.9 violations/seed, CPO 180.7 (safest), SGPO-scale 508.0, SGPO-barrier 274.8
3. **Agentic Shortcut** *(UNVERIFIED)* — multi-step forbidden shortcut: no multi-step re-run yet
4. **Safety Gym Benchmarks** — PointGoal, CarGoal, DoggoGoal (planned)
5. **Metric Learning Ablations** — β, event horizon radius, severity σ (planned)

---

## Status

- [ ] Paper outline finalized
- [ ] Experimental design locked
- [x] Sandbagging trap benchmark (single seed; needs 30+ seed re-run)
- [x] Murky Drone re-run (50 seeds; claim withdrawn)
- [ ] Agentic shortcut benchmark (multi-step re-run needed)
- [ ] Safety Gym expansion
- [ ] Connection to Natural Policy Gradient (theory)
- [ ] Comparison to RCBF (theory + experiment)
- [ ] First draft
- [ ] Venue selection

**Target Venue**: ICRL 2026 (International Conference on Reinforcement Learning), or NeurIPS 2026 (Safety track)
**Backup**: RLC 2026; ICLR 2027
