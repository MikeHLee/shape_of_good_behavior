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

No LM-level Hodge-PPO advantage over standard PPO was found (25.49% vs 23.53% is noise at n=51) — this does not replicate the embedding-level Hodge-DPO/Hodge-KTO advantage from the optimizer comparison above. Unexpectedly, plain SFT beats both PPO variants. Full writeup: `shared/README.md#exploit-resistance-eval--sgb-004-2026-07-24-true-trainholdout-split`, data: `shared/results/finetune/sgb004_exploit_resistance_holdout.json`.

**Follow-up (SGB-005b):** tested whether the Hodge decomposition is more useful as an auxiliary *feature* than as a training-time loss reweight. A first pass looked like a strong win (98% holdout accuracy separating ideal/exploit text) but that turned out to be near-tautological — the graph is fed a near-certain direct edge asserting the exact ranking being tested. An ablation removing that edge (keeping only unsupervised cross-pair kNN structure) found a much weaker, honest signal: 55–65% accuracy, above chance but not reliable at n=268. The more important finding: both reward models' training loss converged to ≈log(2) — they barely learned to discriminate at all, which likely explains why SFT beat both PPO variants above. Full writeup: `shared/README.md#hodge-as-featurizer-test--sgb-005b-2026-07-24`.

**Root cause + corrected rerun (SGB-005c, 2026-07-27):** the RM's stuck-at-chance loss traced to a real bug, not just undertraining — TRACE `context_text` is long enough (~1010 tokens average) that the default right-truncation at `rm_max_length=512` cut the sequence off before the assistant response even started, making chosen/rejected byte-identical for ~98% of pairs. Fixed with left-truncation + a longer context window; both RMs then converged properly, verified at **100% train and 100% holdout ranking accuracy** on reference pairs (up from 49%/45% chance). Retrained both PPO variants against the fixed RM and reran eval:

| Model | Mean Reward | Exploit Resistance (n=51) |
|-------|-------------|-----------------------------|
| base | 1.1432 | 68.63% |
| SFT | 1.9272 | 80.39% |
| PPO | 2.2174 | 80.39% |
| **Hodge-PPO** | **2.5873** | **82.35%** |

This resolves the SGB-004 anomaly: PPO now clearly beats SFT (2.22 vs 1.93 mean reward), confirming the earlier result was an artifact of the broken RM, not a real PPO/Hodge property. Hodge-PPO edges ahead of standard PPO on both metrics — the first time this investigation's direction has matched the embedding-level hypothesis — but the resistance gap is still one example at n=51 (42/51 vs 41/51), so it's directionally positive, not statistically confirmed. Full writeup: `shared/README.md#rm-root-cause-fix--corrected-rerun--sgb-005c-2026-07-27`, data: `shared/results/finetune/sgb004_exploit_resistance_holdout_v2.json`.

**Follow-up writeup (SGB-005, 2026-07-27):** folded the embedding-level 30-seed benchmark and the corrected 1.5B LM-level result above into one note with a method×scale figure, shipped now (per the steady-publishing-cadence preference) rather than waiting for the 7B run. Headline: the Hodge advantage is large and clean at the embedding level (+6.3%/+24.5% over DPO/KTO) but collapses to a one-example, unconfirmed margin at the LM level (82.35% vs 80.39%, n=51) — directionally consistent, not yet a replication. Full writeup: `WRITEUP_OPEN_MODEL_EXPLOIT_RESISTANCE.md`, figure: `figures/sgb005_fig1_method_by_scale.png`.

## Status

- [x] Optimizer comparison benchmark (30 seeds)
- [x] PPO fine-tuning verified on A100 (SGB-003)
- [x] Exploit resistance eval on true holdout split (SGB-004) — superseded by SGB-005c below
- [x] Hodge-as-featurizer ablation (SGB-005b) — weak (55–65%) unsupervised signal once tautological confound removed; root cause of SGB-004 traced to undertrained RM
- [x] Fix reward-model training convergence (SGB-005c) — truncation bug found and fixed; corrected rerun shows PPO > SFT (anomaly resolved) and Hodge-PPO directionally ahead of standard PPO (not yet significant at n=51)
- [x] First draft (SGB-005) — `WRITEUP_OPEN_MODEL_EXPLOIT_RESISTANCE.md`, ships the 1.5B result now with caveats per the steady-publishing-cadence preference
- [ ] More holdout data or repeated seeds to firm up the Hodge-PPO vs standard-PPO gap
- [ ] Scale Hodge-PPO to 7B/8B (SGB-006) and fold into a follow-up revision of the writeup
- [ ] Condorcet ring benchmark (extend to 200+ seeds)
- [ ] HH-RLHF topological audit
- [ ] Multi-evaluator sheaf analysis
- [ ] Venue selection

**Target Venue**: NeurIPS 2026 (Theory/ML track), or ICML 2027
**Backup**: JMLR; or TAG-ML workshop for preliminary work
