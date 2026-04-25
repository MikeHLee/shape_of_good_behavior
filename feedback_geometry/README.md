# Feedback Geometry: Topological Inconsistency Detection in Human Preference Data

**Research Track 1 of 3 — "The Shape of Good Behavior" Series**

---

## Overview

This paper focuses on the *feedback geometry* of RLHF: the mathematical structure of human preference data and how that structure reveals inconsistency, cycling, and contradiction that scalar reward models cannot represent.

The central insight is that human feedback is not just noisy — it is *topologically structured*. Preferences form a sheaf over the space of comparisons, and the first cohomology group H¹ of that sheaf measures the degree to which those preferences fail to be globally consistent. When H¹ ≠ 0, no scalar potential can rationalize the feedback — there are genuine Condorcet cycles baked into the data.

This work adapts and extends **Jiang et al.'s HodgeRank** (Statistical Ranking and Combinatorial Hodge Theory, 2011) into the RLHF setting, adds a neural implementation of the Hodge critic, and demonstrates that decomposing feedback into its exact (gradient), coexact (curl), and harmonic (cycle) components is practically useful for training more robust reward models.

---

## Core Contributions

1. **Preference Cohomology**: Formalize RLHF feedback as sections of a sheaf over the comparison graph; H¹ measures cyclic inconsistency
2. **Hodge-Rank for RLHF**: Adapt combinatorial Hodge decomposition to pairwise LLM preference data with vector embeddings
3. **Cycle-Aware Reward Modeling**: Reward models trained on H¹-filtered or H¹-reweighted data are more calibrated and transferable
4. **Condorcet Audit Tool**: Practical pipeline for auditing arbitrary preference datasets for topological inconsistency
5. **Multi-Evaluator Sheaf Construction**: When multiple raters score the same examples, restriction maps encode their systematic disagreements; H¹ measures rater incoherence beyond simple kappa

---

## Relationship to Other Research Tracks

| Track | Focus | Key Tool |
|-------|-------|----------|
| **This paper (Feedback Geometry)** | Inconsistency in feedback data | H¹ cohomology, Hodge decomposition |
| Constraint Geometry | Safe policy optimization | Geodesic policy gradient, metric singularities |
| Constitutional Alignment Geometry | Values in embedding space | Alignment differentials, constitutional vectors |

The SGPO algorithm (Constraint Geometry paper) uses the Hodge critic developed here; this paper stands alone as a contribution to preference learning and reward modeling.

---

## Source Code References (from `../high_dimensional_reward_spaces/`)

- `src/hodge_critic.py` — core Hodge decomposition and Condorcet cycle detection
- `src/mine_preference_cycles.py` — preference graph construction and cycle mining
- `src/sheaf_resolver.py` — multi-evaluator sheaf with learnable restriction maps
- `src/condorcet_experiment.py` — Condorcet ring benchmark
- `notebooks/colab_01_topology_mining.ipynb` — topology mining pipeline
- `data/condorcet_benchmark.json` — Condorcet experiment data
- `data/ethical_scenarios_summary.csv` — multi-scenario preference data

---

## Status

- [ ] Paper outline finalized
- [ ] Experimental design locked
- [ ] Condorcet ring benchmark (re-runs needed: 30 → 200+ seeds)
- [ ] HH-RLHF topological audit (new experiment)
- [ ] Multi-evaluator sheaf analysis (new experiment)
- [ ] Hodge calibration comparison (new experiment)
- [ ] First draft
- [ ] Venue selection

**Target Venue**: NeurIPS 2026 (Theory/ML track), or ICML 2027
**Backup**: JMLR; or TAG-ML workshop for preliminary work
