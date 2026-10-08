# SGB-046 results: Hodge structure as a reward-model diagnostic (simulation)

This file was generated from `shared/results/sgb046_reward_diagnostic_v1.json` by `scripts/sgb046_reward_diagnostic_sim.py` on 2026-10-07 17:14:37. All numbers come from that JSON file.

- Pre-registration: `feedback_geometry/SGB046_PREREGISTRATION.md`, SHA-256 `5e19fdf1572ab3ce9a6c68638ca625a96f1ab7d318f243789c13362976e8a550` (the script checked this hash before the run: True).
- Git HEAD at the run: `64ee6db41c374b84b9d8eb040d3a56c4713ae03f`. Runtime: 249 s on 12 CPU workers.
- Command: `./venv/bin/python3 scripts/sgb046_reward_diagnostic_sim.py --workers 12`
- Seeds: 0–39 (40 seeds). Blocks per seed: 200 train + 100 held-out. k = 5. P in [4, 6]. V in [5, 25]. Bootstrap B = 200.
- Everything is simulated. A synthetic result is not evidence about human or real-grader preferences (pre-registration § 5).

## 1. D1 calibration gate (read this first)

**D1 gate: PASS.** The rule: false-positive rate (FPR) ≤ 0.05 and the upper 95% Clopper–Pearson bound ≤ 0.08, in each transitive scenario, at each (P, V) cell.

**Positive control (`condorcet`, block-level recall at V = 25): PASS** P4_V25: 0.695, P6_V25: 0.633 (rule: ≥ 0.5).

The null uses deflation variant `reoriented` with c = 0.0 (see § 6, item 1, and § 8).

FPR = fraction of blocks with no true cycle that the test flagged. Recall = fraction of blocks with a true cycle that the test flagged.

| scenario | cell | blocks without a true cycle | FPR | 95% CP | gate | naive curl-test FPR | blocks with a true cycle | recall | flag precision |
|---|---|---|---|---|---|---|---|---|---|
| `transitive` | P4_V5 | 12000 | 0.000 | [0.000, 0.001] | pass | 0.318 | 0 | n/a | 0.000 |
| `transitive` | P4_V25 | 12000 | 0.007 | [0.006, 0.009] | pass | 0.929 | 0 | n/a | 0.000 |
| `transitive` | P6_V5 | 12000 | 0.004 | [0.003, 0.005] | pass | 0.314 | 0 | n/a | 0.000 |
| `transitive` | P6_V25 | 12000 | 0.025 | [0.022, 0.028] | pass | 0.923 | 0 | n/a | 0.000 |
| `transitive_binarized` | P4_V5 | 12000 | 0.000 | [0.000, 0.001] | pass | 0.743 | 0 | n/a | 0.000 |
| `transitive_binarized` | P4_V25 | 12000 | 0.008 | [0.007, 0.010] | pass | 0.979 | 0 | n/a | 0.000 |
| `transitive_binarized` | P6_V5 | 12000 | 0.003 | [0.002, 0.004] | pass | 0.813 | 0 | n/a | 0.000 |
| `transitive_binarized` | P6_V25 | 12000 | 0.025 | [0.022, 0.028] | pass | 0.989 | 0 | n/a | 0.000 |
| `transitive_neartie` | P4_V5 | 12000 | 0.005 | [0.004, 0.007] | pass | 0.017 | 0 | n/a | 0.000 |
| `transitive_neartie` | P4_V25 | 12000 | 0.001 | [0.000, 0.001] | pass | 0.024 | 0 | n/a | 0.000 |
| `transitive_neartie` | P6_V5 | 12000 | 0.005 | [0.004, 0.006] | pass | 0.014 | 0 | n/a | 0.000 |
| `transitive_neartie` | P6_V25 | 12000 | 0.001 | [0.000, 0.002] | pass | 0.024 | 0 | n/a | 0.000 |
| `condorcet` | P4_V5 | 7232 | 0.002 | [0.001, 0.003] | — | n/a | 4768 | 0.385 | 0.992 |
| `condorcet` | P4_V25 | 7232 | 0.002 | [0.001, 0.003] | — | n/a | 4768 | 0.695 | 0.996 |
| `condorcet` | P6_V5 | 6574 | 0.003 | [0.002, 0.005] | — | n/a | 5426 | 0.304 | 0.987 |
| `condorcet` | P6_V25 | 6574 | 0.002 | [0.001, 0.003] | — | n/a | 5426 | 0.633 | 0.996 |
| `harmless_disagreement` | P4_V5 | 12000 | 0.000 | [0.000, 0.000] | — | n/a | 0 | n/a | n/a |
| `harmless_disagreement` | P4_V25 | 12000 | 0.000 | [0.000, 0.000] | — | n/a | 0 | n/a | n/a |
| `harmless_disagreement` | P6_V5 | 12000 | 0.000 | [0.000, 0.000] | — | n/a | 0 | n/a | n/a |
| `harmless_disagreement` | P6_V25 | 12000 | 0.000 | [0.000, 0.000] | — | n/a | 0 | n/a | n/a |
| `mixed` | P4_V5 | 5712 | 0.003 | [0.001, 0.004] | — | n/a | 6288 | 0.374 | 0.994 |
| `mixed` | P4_V25 | 5712 | 0.002 | [0.001, 0.003] | — | n/a | 6288 | 0.682 | 0.998 |
| `mixed` | P6_V5 | 5844 | 0.002 | [0.001, 0.004] | — | n/a | 6156 | 0.370 | 0.994 |
| `mixed` | P6_V25 | 5844 | 0.002 | [0.001, 0.004] | — | n/a | 6156 | 0.679 | 0.997 |
| `position_bias` | P4_V5 | 12000 | 0.000 | [0.000, 0.001] | — | 0.022 | 0 | n/a | 0.000 |
| `position_bias` | P4_V25 | 12000 | 0.004 | [0.003, 0.006] | — | 0.741 | 0 | n/a | 0.000 |
| `position_bias` | P6_V5 | 12000 | 0.001 | [0.001, 0.002] | — | 0.038 | 0 | n/a | 0.000 |
| `position_bias` | P6_V25 | 12000 | 0.016 | [0.014, 0.018] | — | 0.749 | 0 | n/a | 0.000 |

Triangle level (certified cycles): 

| scenario | cell | true cycles | certified cycles | triangle recall | triangle precision | blocks with an observed cycle |
|---|---|---|---|---|---|---|
| `condorcet` | P4_V5 | 9608 | 3853 | 0.381 | 0.949 | 0.415 |
| `condorcet` | P4_V25 | 9608 | 6762 | 0.688 | 0.977 | 0.406 |
| `condorcet` | P6_V5 | 10330 | 3193 | 0.289 | 0.935 | 0.461 |
| `condorcet` | P6_V25 | 10330 | 6573 | 0.617 | 0.970 | 0.455 |
| `mixed` | P4_V5 | 13094 | 5007 | 0.362 | 0.946 | 0.529 |
| `mixed` | P4_V25 | 13094 | 9034 | 0.676 | 0.980 | 0.525 |
| `mixed` | P6_V5 | 12708 | 4854 | 0.363 | 0.949 | 0.511 |
| `mixed` | P6_V25 | 12708 | 8837 | 0.680 | 0.978 | 0.514 |
| `harmless_disagreement` | P4_V5 | 0 | 0 | n/a | n/a | 0.011 |
| `harmless_disagreement` | P4_V25 | 0 | 0 | n/a | n/a | 0.002 |
| `harmless_disagreement` | P6_V5 | 0 | 0 | n/a | n/a | 0.011 |
| `harmless_disagreement` | P6_V25 | 0 | 0 | n/a | n/a | 0.002 |

Naive curl-mass test (expected to fail on `transitive_binarized`; not a gate). Mean curl fraction of the observed aggregate flow:

| scenario | cell | mean curl fraction | naive curl-test FPR |
|---|---|---|---|
| `transitive` | P4_V5 | 0.157 | 0.318 |
| `transitive` | P4_V25 | 0.131 | 0.929 |
| `transitive` | P6_V5 | 0.142 | 0.314 |
| `transitive` | P6_V25 | 0.117 | 0.923 |
| `transitive_binarized` | P4_V5 | 0.233 | 0.743 |
| `transitive_binarized` | P4_V25 | 0.232 | 0.979 |
| `transitive_binarized` | P6_V5 | 0.245 | 0.813 |
| `transitive_binarized` | P6_V25 | 0.244 | 0.989 |
| `transitive_neartie` | P4_V5 | 0.379 | 0.017 |
| `transitive_neartie` | P4_V25 | 0.172 | 0.024 |
| `transitive_neartie` | P6_V5 | 0.352 | 0.014 |
| `transitive_neartie` | P6_V25 | 0.151 | 0.024 |
| `position_bias` | P4_V5 | 0.167 | 0.022 |
| `position_bias` | P4_V25 | 0.136 | 0.741 |
| `position_bias` | P6_V5 | 0.156 | 0.038 |
| `position_bias` | P6_V25 | 0.128 | 0.749 |

Antisymmetrization check (`position_bias`, one presentation order, bias 1.0 log-odds). The same blocks were also scored with the second order added:

| cell | curl fraction, one order | curl fraction, both orders | flag rate, one order | flag rate, both orders |
|---|---|---|---|---|
| P4_V5 | 0.167 | 0.147 | 0.000 | 0.001 |
| P4_V25 | 0.136 | 0.122 | 0.004 | 0.007 |
| P6_V5 | 0.156 | 0.138 | 0.001 | 0.004 |
| P6_V25 | 0.128 | 0.115 | 0.016 | 0.025 |

## 2. Primary family: D2 and D3 (Holm over {D2, D3})

Rule: D2 p = max of the two scenario bootstrap p-values (both scenarios are predicted); D3 p = paired Wilcoxon on per-seed Spearman difference in mixed; Holm over {D2, D3}.

| hypothesis | p | Holm-adjusted p | significant at 0.05 | verdict |
|---|---|---|---|---|
| D2 | 1.0e-04 | 1.0e-04 | yes | **CONFIRMED** |
| D3 | 1.8e-12 | 3.6e-12 | yes | **CONFIRMED in mixed** |

**Plain reading.** The verdicts above are against the pre-registered baselines. Two stronger comparisons change the meaning:

- D2, `condorcet`: when the baseline also knows whether the RM sign disagrees with the observed panel label, the baseline AUROC is 0.971 and the Hodge features add Δ = 0.0001 (95% CI [-0.0002, 0.0004]). The Hodge features then add nothing measurable.
- D2, `mixed`: when the baseline also knows whether the RM sign disagrees with the observed panel label, the baseline AUROC is 0.970 and the Hodge features add Δ = 0.0001 (95% CI [-0.0002, 0.0004]). The Hodge features then add nothing measurable.
- D3, `condorcet`: against 'pivotality' (edges whose aggregate sign changes under leave-one-out, no cycles), the Hodge score wins by mean Spearman 0.083 (Hodge 0.881, pivotality 0.798; Wilcoxon p 1.4e-06, wins/ties/losses 33/0/7).
- D3, `mixed`: against 'pivotality' (edges whose aggregate sign changes under leave-one-out, no cycles), the Hodge score wins by mean Spearman 0.043 (Hodge 0.900, pivotality 0.857; Wilcoxon p 0.0019, wins/ties/losses 30/0/10).
- The D3 truth is defined as the estimand of the Hodge score (§ 4). The large win over raw disagreement is therefore expected by construction.

### 2.1 D2: failure localization

Target: the scalar BT reward model is wrong on a held-out pair, relative to the true aggregate relation. Score: held-out AUROC of a logistic regression that was fitted on training blocks only. Baseline features: vote entropy across principles, |observed aggregate margin|, |RM margin|. Hodge features: certified cycles through the edge, |curl component|, block flag. Δ = AUROC(baseline + Hodge) − AUROC(baseline). Per seed, Δ is the mean over the 4 (P, V) cells. The CI is a paired bootstrap over 40 seeds (10 000 resamples).

| scenario | AUROC baseline | AUROC baseline + Hodge | mean Δ | 95% CI | bootstrap p | Δ ≥ 0.02 and CI > 0 |
|---|---|---|---|---|---|---|
| `transitive` | 0.950 | 0.949 | -0.0009 | [-0.0016, -0.0004] | 1.0e-04 | no |
| `transitive_binarized` | 0.950 | 0.949 | -0.0016 | [-0.0024, -0.0010] | 1.0e-04 | no |
| `transitive_neartie` | 0.973 | 0.971 | -0.0020 | [-0.0026, -0.0014] | 1.0e-04 | no |
| `condorcet` (confirmatory) | 0.759 | 0.795 | 0.0366 | [0.0343, 0.0390] | 1.0e-04 | yes |
| `harmless_disagreement` | 0.969 | 0.959 | -0.0095 | [-0.0135, -0.0063] | 1.0e-04 | no |
| `mixed` (confirmatory) | 0.750 | 0.808 | 0.0577 | [0.0553, 0.0602] | 1.0e-04 | yes |
| `position_bias` | 0.953 | 0.952 | -0.0012 | [-0.0018, -0.0006] | 1.0e-04 | no |

Per (P, V) cell, confirmatory scenarios:

| scenario | cell | RM error rate (held-out) | AUROC base | AUROC base + Hodge | Δ mean | Δ 95% CI |
|---|---|---|---|---|---|---|
| `condorcet` | P4_V5 | 0.187 | 0.756 | 0.785 | 0.0286 | [0.0254, 0.0320] |
| `condorcet` | P4_V25 | 0.184 | 0.751 | 0.800 | 0.0494 | [0.0453, 0.0538] |
| `condorcet` | P6_V5 | 0.207 | 0.766 | 0.792 | 0.0267 | [0.0236, 0.0298] |
| `condorcet` | P6_V25 | 0.205 | 0.762 | 0.804 | 0.0417 | [0.0376, 0.0462] |
| `mixed` | P4_V5 | 0.242 | 0.730 | 0.782 | 0.0523 | [0.0484, 0.0563] |
| `mixed` | P4_V25 | 0.241 | 0.725 | 0.792 | 0.0665 | [0.0629, 0.0703] |
| `mixed` | P6_V5 | 0.240 | 0.774 | 0.825 | 0.0513 | [0.0485, 0.0541] |
| `mixed` | P6_V25 | 0.241 | 0.771 | 0.832 | 0.0608 | [0.0572, 0.0642] |

### 2.2 D3: attribution to principle pairs

Hodge score of pair (p, q) = certified cycles that leave-one-out of p or of q removes (re-tested with the D1 test), averaged over the 300 blocks. Baseline = raw disagreement rate of graders p and q. Truth = true cycles that leave-one-out of p or q removes, on the true utilities. Per seed, the metric is the mean over the 4 (P, V) cells. Test: paired Wilcoxon over seeds.

| scenario | metric | Hodge mean | baseline mean | mean diff | Wilcoxon p | wins/ties/losses |
|---|---|---|---|---|---|---|
| `condorcet` | Spearman | 0.881 | 0.071 | 0.810 | 1.8e-12 | 40/0/0 |
| `condorcet` | precision@k | n/a | n/a | n/a | n/a | 0/0/0 |
| `harmless_disagreement` | Spearman | n/a | n/a | n/a | n/a | 0/0/0 |
| `harmless_disagreement` | precision@k | n/a | n/a | n/a | n/a | 0/0/0 |
| `mixed` | Spearman | 0.900 | 0.385 | 0.515 | 1.8e-12 | 40/0/0 |
| `mixed` | precision@k | 1.000 | 0.763 | 0.237 | 6.0e-09 | 40/0/0 |

`harmless_disagreement` has no true cycles by construction, so the true tension is 0 for every pair and Spearman and precision@k are undefined there (see § 7). The substitute metric is the share of each method's total score that falls on the harmless pair ((0, 1) in `harmless_disagreement`; (0, 3) at P = 4 and (3, 4) at P = 6 in `mixed`). Lower is better.

| scenario | Hodge share | disagreement share | pivotality share | Wilcoxon p (disagreement − Hodge) | harmless pair ranked first: Hodge / disagreement |
|---|---|---|---|---|---|
| `harmless_disagreement` | 0.000 | 0.193 | 0.004 | 1.8e-12 | 0.000 / 1.000 |
| `mixed` | 0.090 | 0.169 | 0.097 | 1.8e-12 | 0.000 / 1.000 |

Mean scores per principle pair (`mixed`):

- P4_V5: pairs [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
  - truth: [1.0912, 1.0912, 1.0796, 1.0912, 0.8892, 0.8969]
  - Hodge: [0.4172, 0.4172, 0.4164, 0.4172, 0.3797, 0.3833]
  - disagreement: [0.6074, 0.6143, 0.7514, 0.6135, 0.3728, 0.3715]
  - pivotality: [4.4379, 4.4466, 4.3489, 4.5417, 2.5486, 2.4545]
  - mean Spearman: Hodge 0.889, disagreement 0.561, pivotality 0.905
- P4_V25: pairs [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
  - truth: [1.0912, 1.0912, 1.0796, 1.0912, 0.8892, 0.8969]
  - Hodge: [0.7528, 0.7528, 0.7482, 0.7528, 0.641, 0.6492]
  - disagreement: [0.5931, 0.6009, 0.7418, 0.5992, 0.361, 0.3605]
  - pivotality: [4.459, 4.4593, 4.2743, 4.4782, 2.3957, 2.3425]
  - mean Spearman: Hodge 0.955, disagreement 0.558, pivotality 0.919
- P6_V5: pairs [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]
  - truth: [1.0474, 1.0474, 0.8823, 0.8823, 0.8823, 1.0474, 0.876, 0.876, 0.876, 0.8752, 0.8752, 0.8752, 0.0, 0.0, 0.0]
  - Hodge: [0.4036, 0.4036, 0.3722, 0.3718, 0.3732, 0.4037, 0.372, 0.3709, 0.3732, 0.3688, 0.3696, 0.3713, 0.0563, 0.0543, 0.0534]
  - disagreement: [0.6099, 0.6101, 0.4939, 0.4968, 0.4794, 0.6085, 0.4934, 0.4929, 0.4806, 0.4959, 0.4931, 0.4794, 0.9025, 0.497, 0.4957]
  - pivotality: [3.7911, 3.7834, 2.5492, 2.5498, 2.6269, 3.7249, 2.5298, 2.5287, 2.5785, 2.4911, 2.4916, 2.5158, 0.1766, 0.1744, 0.1736]
  - mean Spearman: Hodge 0.838, disagreement 0.227, pivotality 0.797
- P6_V25: pairs [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]
  - truth: [1.0474, 1.0474, 0.8823, 0.8823, 0.8823, 1.0474, 0.876, 0.876, 0.876, 0.8752, 0.8752, 0.8752, 0.0, 0.0, 0.0]
  - Hodge: [0.7321, 0.732, 0.6398, 0.6392, 0.64, 0.732, 0.6403, 0.6409, 0.6412, 0.6375, 0.6377, 0.6382, 0.0161, 0.016, 0.0158]
  - disagreement: [0.5942, 0.5952, 0.4834, 0.485, 0.4643, 0.5947, 0.4831, 0.4811, 0.4661, 0.4836, 0.4821, 0.4648, 0.8967, 0.4872, 0.4852]
  - pivotality: [3.7388, 3.7372, 2.4973, 2.4979, 2.5147, 3.7249, 2.4953, 2.4955, 2.5063, 2.481, 2.4815, 2.4871, 0.036, 0.0358, 0.0358]
  - mean Spearman: Hodge 0.917, disagreement 0.192, pivotality 0.808

## 3. D4: decision relevance (exploratory, no correction)

Skew-symmetric pairwise model (linear part + skew bilinear form on item features) vs the scalar BT model, both fitted on the same training blocks. Gain = held-out accuracy (skew) − accuracy (BT) against the true aggregate relation. Interaction = gain in D1-flagged blocks − gain in unflagged blocks (per seed).

| scenario | acc BT | acc skew | gain, all | gain, flagged blocks | gain, unflagged blocks | interaction (95% CI) | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| `transitive` | 0.956 | 0.948 | -0.0076 | -0.0160 | -0.0075 | -0.0085 [-0.0206, 0.0036] | 0.49 |
| `transitive_binarized` | 0.956 | 0.949 | -0.0073 | -0.0132 | -0.0071 | -0.0061 [-0.0206, 0.0091] | 0.23 |
| `transitive_neartie` | 0.979 | 0.958 | -0.0210 | -0.0370 | -0.0210 | -0.0166 [-0.0448, 0.0114] | 0.22 |
| `condorcet` | 0.804 | 0.803 | -0.0017 | 0.0011 | -0.0023 | 0.0034 [0.0012, 0.0056] | 0.0028 |
| `harmless_disagreement` | 0.995 | 0.989 | -0.0057 | n/a | -0.0057 | n/a [n/a, n/a] | n/a |
| `mixed` | 0.759 | 0.756 | -0.0030 | -0.0014 | -0.0038 | 0.0023 [-0.0011, 0.0058] | 0.25 |
| `position_bias` | 0.958 | 0.950 | -0.0085 | -0.0070 | -0.0085 | 0.0014 [-0.0139, 0.0152] | 0.24 |

## 4. Ways this simulation could favour the diagnostic

- **D2, `condorcet`: oracle check.** A score that counts the TRUE cycles through an edge has held-out AUROC P4_V5 0.673, P4_V25 0.674, P6_V5 0.672, P6_V25 0.672. The certified-cycle count alone has AUROC P4_V5 0.561, P4_V25 0.617, P6_V5 0.550, P6_V25 0.604. The RM error rate on true-cycle edges is P4_V5 0.442, P4_V25 0.437, P6_V5 0.462, P6_V25 0.457 and on other edges is P4_V5 0.124, P4_V25 0.121, P6_V5 0.137, P6_V25 0.136. So part of the Hodge signal is structural (a scalar must be wrong on one edge of every true cycle), as the pre-registration disclosed.
- **D2, `condorcet`: is disagreement low on cycle edges?** Mean vote entropy on true-cycle edges vs other edges: P4_V5 0.634 vs 0.588, P4_V25 0.632 vs 0.586, P6_V5 0.653 vs 0.611, P6_V25 0.652 vs 0.610.
- **D2, `condorcet`: stronger baseline.** The pre-registered baseline does not contain the most direct error signal that is available on a held-out block: whether the RM sign disagrees with the observed panel label. With that feature and the signed product (RM margin × observed margin) added, the baseline AUROC is 0.971 and the Hodge features add Δ = 0.0001 (95% CI [-0.0002, 0.0004], bootstrap p 0.52). The single feature 'RM disagrees with the observed label' has AUROC P4_V5 0.935, P4_V25 0.971, P6_V5 0.928, P6_V25 0.967.
- **D2, `mixed`: oracle check.** A score that counts the TRUE cycles through an edge has held-out AUROC P4_V5 0.669, P4_V25 0.669, P6_V5 0.670, P6_V25 0.671. The certified-cycle count alone has AUROC P4_V5 0.562, P4_V25 0.615, P6_V5 0.564, P6_V25 0.617. The RM error rate on true-cycle edges is P4_V5 0.463, P4_V25 0.463, P6_V5 0.470, P6_V25 0.473 and on other edges is P4_V5 0.161, P4_V25 0.160, P6_V5 0.159, P6_V25 0.159. So part of the Hodge signal is structural (a scalar must be wrong on one edge of every true cycle), as the pre-registration disclosed.
- **D2, `mixed`: is disagreement low on cycle edges?** Mean vote entropy on true-cycle edges vs other edges: P4_V5 0.633 vs 0.566, P4_V25 0.633 vs 0.564, P6_V5 0.659 vs 0.617, P6_V25 0.658 vs 0.615.
- **D2, `mixed`: stronger baseline.** The pre-registered baseline does not contain the most direct error signal that is available on a held-out block: whether the RM sign disagrees with the observed panel label. With that feature and the signed product (RM margin × observed margin) added, the baseline AUROC is 0.970 and the Hodge features add Δ = 0.0001 (95% CI [-0.0002, 0.0004], bootstrap p 0.49). The single feature 'RM disagrees with the observed label' has AUROC P4_V5 0.936, P4_V25 0.973, P6_V5 0.934, P6_V25 0.970.
- **D3: the truth is the estimand of the Hodge score.** The true tension is 'true cycles removed by leave-one-out of p or q'. The Hodge score is the same quantity computed on noisy labels. The disagreement baseline estimates a different quantity. So D3 favours the Hodge score by definition, and a win shows only that the plug-in estimate survives the label noise.
- **D3: a fairer non-Hodge baseline.** 'Pivotality' counts the edges whose observed aggregate sign changes under leave-one-out of p or q. It uses no cycles and no bootstrap. In `mixed`, mean Spearman: Hodge 0.900, pivotality 0.857, disagreement 0.385. Hodge − pivotality: mean 0.043, Wilcoxon p 0.0019, wins/ties/losses 30/0/10.
- **D3: the harmless pairs were built to disagree maximally** (utilities exactly or strongly anti-correlated). That is the case the pre-registration names, but it is the worst case for the disagreement baseline. Mean disagreement rate per pair is in § 2.2.
- **D1: the null design (order rule, deflation variant, c) was chosen on this simulator** (pilot seeds 1000–1004). The main-seed FPR is an out-of-sample check on new seeds of the same simulator. It is not a check on a different noise model. Real graders with other noise (for example correlated votes, or label noise that is not binomial) can make the test anti-conservative.
- **The signal-to-noise scale was raised on pilot seeds after block recall at V = 25 was about 0.5.** At the earlier scale, about half of the true-cycle blocks hinged on a principle that was close to a tie on a pivotal edge, and no calibrated test can certify such a cycle. The D2–D4 results are therefore conditional on a regime where most planted cycles are certifiable at V = 25. At V = 5 recall is much lower; see the D1 table.
- **Grader noise is independent binomial votes with a known logit link.** The null resamples votes with the same link. This matches the simulator exactly, which favours calibration.
- **Principle utilities are linear in the item features and each principle reads its own feature dimension.** The skew-bilinear model in D4 and the BT model in D2 use the same features, so model misspecification comes only from the aggregation rule.

## 5. What this experiment cannot show

- Everything is simulated. The graders are a model of constitutional principle graders. They are not real LLM judges and not humans. The result shows only that the diagnostic **can** work when the structure is present.
- The real-grader test is SGB-045 Stage 2 (log-prob judges with a constitution ablation).

## 6. Implementation choices that the pre-registration did not fix

1. **D1 null (transitive-consistent parametric bootstrap).** (a) Order: the total order of the 5 responses that reorients the least plug-in evidence, found by exhaustive search over all 120 orders (cost = sum of |z| of the observed aggregate sign over the edges that disagree with the order; ties broken by the least-squares BT order). This is the maximum-likelihood transitive order under a normal approximation. (b) On edges that disagree with the order, the null flips all principle labels, so every edge points along the order with its observed strength; the expected null relation is therefore transitive. (c) Optional deflation toward a tie by c standard units (winner's-curse correction), on the reoriented edges only or on all edges. (d) Votes are resampled from the null log-odds in both orders with the observed position effect, B = 200. The variant and c were chosen on pilot seeds 1000–1004 by the fixed rule in § 8. A pure-tie null (every disagreeing edge set to a tie) was rejected at design time: it can never certify a single cycle, because the tied edge reproduces the cycle in about half of the replicates.
2. **Certified cycle.** An observed directed 3-cycle on triangle t in a D1-flagged block, with per-triangle bootstrap p ≤ 0.05 (the null reproduces the same directed cycle on t in ≤ 5% of replicates).
3. **Aggregators.** Constitution vote = weighted principle vote sum_p v_p sign(f_p) with tie-break weights of at most 0.001. Weighted-sum control = sum_p f_p (unit weights) on graded labels, so the truth is sign(sum_p eta_p) with eta the true log-odds.
4. **Scenario parameters.** Utilities u_p = g·q + s·(principle component), g = 1.667, s = 5.0, grader temperatures T_p ~ U(0.5, 1.0), first-position bias 0.3 log-odds. `condorcet`: principle components are centred across the P principles (balanced trade-off). `harmless_disagreement`: u_0 = g·q + 2s·x_0, u_1 = g·q − 2s·x_0, and principle 2 has vote weight P − 0.5, so it decides every vote. `mixed`: principles 0–2 form a centred trade-off trio (Condorcet cycles); P = 4: u_3 = g·q − 2s·x_0 with weight 0.5; P = 6: u_3, u_4 = g·q ± 2s·x_3 and u_5 = g·q + s·x_4, weights 0.3. These principles are never pivotal in the true vote. `transitive_neartie`: utilities × 0.05. `position_bias`: one order (the lower-index response is shown first), bias 1.0 log-odds.
5. **Reward models.** Logistic regression without intercept on antisymmetric pair features, trained on the sign of the observed aggregate label (ties dropped). C is chosen by 5-fold grouped CV over training blocks from [0.01, 0.1, 1.0, 10.0, 100.0]. The D2 error predictor is trained on out-of-fold (cross-fitted) RM margins of the training blocks, then scored on held-out blocks with the RM fitted on all training blocks.
6. **Combining P and V.** The pre-registration does not say how the 4 (P, V) cells combine. Each seed's metric is the mean over its 4 cells; tests run over the 40 seed-level values. Per-cell values are in the JSON.
7. **One p-value per hypothesis.** D2 predicts a gain in BOTH `condorcet` and `mixed`, so p_D2 = the larger of the two bootstrap p-values (intersection-union). D3's kill criterion names `mixed`, so p_D3 = the Wilcoxon p for the Spearman difference in `mixed`. Holm runs over {D2, D3}. precision@k is secondary.
8. **D3 pair score uses the union** of the single-principle leave-one-out sets ('p or q'), as written. The intersection variant is reported in the JSON as exploratory.
9. **precision@k** uses expected precision under random tie-breaking. When every pair is truly positive or none is, precision@k is undefined and is not counted. Spearman of a constant score is set to 0.

## 7. Deviations from pre-registration

- **D3 in `harmless_disagreement`:** the scenario has no true cycles by construction (as its table row says), so the true tension is 0 for every pair and the pre-registered Spearman and precision@k are undefined. The results file reports a substitute: the share of each score on the harmless pair. This substitute is not part of the Holm family. The D3 kill criterion is defined on `mixed` only, so it is not affected.
- **D1 gate scope:** the gate is applied to `transitive`, `transitive_binarized` and `transitive_neartie`. `position_bias` also has a transitive truth, but the pre-registration lists it as the antisymmetrization check (one order only), so its FPR is reported and is not a gate.
- **D1 null details (order, deflation variant, c):** the pre-registration does not fix them. They were selected on pilot seeds that are disjoint from seeds 0–39, before the main run (§ 6 item 1, § 8).
- **Design history before the main run (disclosed in full).** (1) One debugging pass ran one task per scenario on MAIN seed 0 with an earlier scenario design (no shared quality factor and degenerate tie-break weights, so the true vote tied on 3–3 splits and the scalar RM was near chance everywhere). Those numbers were discarded and that design was replaced. Seed 0 was later re-run with the final design like every other seed. (2) All later design work used pilot seeds 1000–1004 only: (a) shared quality factor and binary tie-break weights; (b) the D1 order changed from the least-squares BT order to the minimum-evidence order, because the least-squares order depended on binarization of the aggregate label and gave a higher FPR in `transitive_binarized`; (c) two deflation variants were compared; (d) the signal-to-noise scale (g, s) was raised by 5/3 because condorcet block recall at V = 25 was about 0.5 at the earlier scale, which is at the pre-registered floor of § 4. The true relation (and so every true cycle) does not depend on this scale; only vote noise relative to signal changes. Item (d) favours the diagnostic (see § 4).

## 8. Calibration pilot for the D1 null (seeds 1000–1004, not used in any result above)

Rule: for each deflation variant ['reoriented', 'all']: c* = smallest c in [0.0, 0.5, 1.0, 1.5, 2.0] whose worst pilot FPR over the 3 gate scenarios x 4 (P, V) cells is <= 0.04; then the variant with the higher mean condorcet block recall at V = 25. Pilot seeds [1000, 1001, 1002, 1003, 1004] (disjoint from the main seeds).

Chosen: deflation variant `reoriented`, c = 0.0.

| variant | c | worst gate FPR | condorcet recall P4_V5 | condorcet recall P4_V25 | condorcet recall P6_V5 | condorcet recall P6_V25 |
|---|---|---|---|---|---|---|
| `reoriented` | 0.0 | 0.031 | 0.407 | 0.706 | 0.338 | 0.655 |
| `reoriented` | 0.5 | 0.019 | 0.343 | 0.657 | 0.260 | 0.580 |
| `reoriented` | 1.0 | 0.010 | 0.193 | 0.603 | 0.157 | 0.516 |
| `reoriented` | 1.5 | 0.005 | 0.071 | 0.561 | 0.050 | 0.464 |
| `reoriented` | 2.0 | 0.005 | 0.022 | 0.497 | 0.020 | 0.418 |
| `all` | 0.0 | 0.031 | 0.407 | 0.706 | 0.338 | 0.655 |
| `all` | 0.5 | 0.017 | 0.275 | 0.635 | 0.199 | 0.564 |
| `all` | 1.0 | 0.007 | 0.078 | 0.556 | 0.056 | 0.466 |
| `all` | 1.5 | 0.005 | 0.007 | 0.475 | 0.007 | 0.397 |
| `all` | 2.0 | 0.005 | 0.000 | 0.407 | 0.001 | 0.347 |

