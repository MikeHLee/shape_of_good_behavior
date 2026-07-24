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

## Source Code References (from `topics/shape_of_good_behavior/`)

- `src/hodge_critic.py` — core Hodge decomposition and Condorcet cycle detection
- `src/mine_preference_cycles.py` — preference graph construction and cycle mining
- `src/sheaf_resolver.py` — multi-evaluator sheaf with learnable restriction maps
- `src/condorcet_experiment.py` — Condorcet ring benchmark
- `notebooks/colab_01_topology_mining.ipynb` — topology mining pipeline
- `data/condorcet_benchmark.json` — Condorcet experiment data
- `data/ethical_scenarios_summary.csv` — multi-scenario preference data

---

## Results Summary (as of 2026-07-23)

### Optimizer Comparison — Hodge variants vs baselines (30 seeds, 500 pairs + 1482 edges, rm_epochs=50)

Results saved: `shared/results/optimizer_comparison_hodge_v3_30seed.json`

| Method | Exploit Resistance | vs Baseline |
|--------|--------------------|-------------|
| **Hodge-DPO** | 0.9999 ± 0.001 | **+6.3% vs DPO** (d=6.52, p<0.0001) |
| **Hodge-KTO** | 0.9964 ± 0.003 | **+24.5% vs KTO** (d=16.47, p<0.0001) |
| **Hodge-GRPO** | 1.000 ± 0.000 | = GRPO (both at ceiling) |
| GRPO | 1.000 ± 0.000 | — |
| DPO | 0.940 ± 0.013 | — |
| KTO | 0.800 ± 0.017 | — |
| ORPO | 0.622 ± 0.080 | — |

Key implementation notes:
- Cross-pair k-NN edges must be preserved during subsampling (H1=0 without them)
- Potential-alignment regularizer (λ=0.05) replaces batch harmonic penalty
- Sign convention: Hodge potential has `potential[exploit] > potential[ideal]`
- Node-level cycle participation used for per-sample weights (not per-edge harmonic fraction)

### PPO Fine-tuning (SGB-003, 2026-07-23, A100-40GB)

| Run | mean_reward_final | Steps | Status |
|-----|------------------|-------|--------|
| Standard PPO | 3.6897 | 64 | success |
| Hodge-PPO | -1.2170 | 64 | success |

Hodge-PPO negative reward is expected: it penalizes exploit-trajectory reward, so lower mean on exploit queries is the training signal. Manifests in `_runs/SGB-SGB-003_2026-07-23_*.json`.

Note: these checkpoints were retrained on a corrected train/holdout split for SGB-004 (below); the numbers above are from the original (in-sample) SGB-003 run and are kept for the crash-safe manifest verification record, not as a valid exploit-resistance measurement.

### Exploit Resistance Eval (SGB-004, 2026-07-24, true 217/51 train/holdout split)

| Model | Exploit Resistance (n=51 holdout) |
|-------|-----------------------------------|
| base | 11.76% |
| **SFT** | **35.29%** |
| PPO | 23.53% |
| Hodge-PPO | 25.49% |

No LM-level Hodge-PPO advantage over standard PPO was found (25.49% vs 23.53% is noise at n=51) — this does not replicate the embedding-level Hodge-DPO/Hodge-KTO advantage from the optimizer comparison above. Unexpectedly, plain SFT beats both PPO variants; not yet investigated. Full writeup: `shared/README.md#exploit-resistance-eval--sgb-004-2026-07-24-true-trainholdout-split`, data: `shared/results/finetune/sgb004_exploit_resistance_holdout.json`.

## Status

- [x] Optimizer comparison benchmark (30 seeds)
- [x] PPO fine-tuning verified on A100 (SGB-003)
- [x] Exploit resistance eval on true holdout split (SGB-004) — hypothesis not supported, SFT unexpectedly best
- [ ] Condorcet ring benchmark (extend to 200+ seeds)
- [ ] HH-RLHF topological audit
- [ ] Multi-evaluator sheaf analysis
- [ ] First draft
- [ ] Venue selection

**Target Venue**: NeurIPS 2026 (Theory/ML track), or ICML 2027
**Backup**: JMLR; or TAG-ML workshop for preliminary work
