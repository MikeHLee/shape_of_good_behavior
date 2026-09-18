# `shared/results/` — provenance and supersession

**Last updated**: 2026-09-18

This directory holds raw result JSON from the Track 2 (`shared/`) pipeline. Several
files measure the same thing at different points in the method's development and
**disagree with each other**. This file records which supersedes which and why, so
the disagreement reads as a method changing over time rather than as a contested
claim.

---

## Optimizer comparison — read this before citing any number

Four files report the same benchmark (`shared/src/optimizer_comparison.py`,
exploit resistance across DPO / GRPO / ORPO / KTO and their Hodge variants). They
were produced in this order:

| # | File | Date | Seeds | Hodge mechanism | Status |
|---|------|------|-------|-----------------|--------|
| 1 | `optimizer_comparison.json` | 2026-03-20 | 30 | batch harmonic penalty | **superseded** |
| 2 | `optimizer_comparison_modal_5seed.json` | 2026-03-21 | 5 | batch harmonic penalty | **superseded** (pilot) |
| 3 | `optimizer_comparison_modal_30seed.json` | 2026-03-23 | 30 | batch harmonic penalty | **superseded** |
| 4 | `optimizer_comparison_hodge_v3_30seed.json` | 2026-03-24 | 30 | potential-alignment regulariser (v3) | **in-sample only; not reproducible** |
| 5 | `optimizer_comparison_heldout_v1.json` | 2026-09-18 | 5 splits × 30 | v3 regulariser + margin control, held-out | **current** |

> **Bottom line (2026-09-18): the Hodge gain does not survive held-out evaluation.**
> File 5 trains on 400 pairs and scores 100 held-out pairs (5 splits × 30 seeds). On
> held-out pairs every method, including a plain linear probe, scores 0.52–0.59 (chance
> is 0.50): DPO 0.550, Hodge-DPO 0.537, KTO 0.542, Hodge-KTO 0.543. The split-level
> paired tests are not significant (Hodge-DPO − DPO −0.013, p = 0.49; Hodge-KTO − KTO
> +0.001, p = 0.97). On the training pairs, a **margin control** — the same loss term
> with every pair's target set to the mean Hodge target, so no pair-specific Hodge
> information — reproduces the full in-sample gain (DPO 1.000, KTO 0.997). File 4's
> 0.9999 / 0.9964 figures are therefore in-sample fitting from an extra positive-margin
> term, not an effect of cycle-aware targets. Do not cite file 4 as evidence for HodgePO.

**Files 1–3 report a null effect. This is expected and is not a contradiction of
file 4.** In all three, every Hodge variant is *numerically identical* to its base
method — not merely statistically indistinguishable, but equal to four decimal
places in mean, std, se and ci95:

```
optimizer_comparison_modal_30seed.json
  DPO        mean 0.9107  std 0.0125      Hodge-DPO   mean 0.9107  std 0.0125
  KTO        mean 0.8218  std 0.0149      Hodge-KTO   mean 0.8218  std 0.0149
  GRPO       mean 1.0     std 0.0         Hodge-GRPO  mean 1.0     std 0.0
```

The reported `t_stat: 0.0, p_value: 1.0, cohens_d: 0.0` in file 1 follows directly.
The cause is documented in the source at
[`shared/src/hodge_preference_optimizers.py:8-13`](../src/hodge_preference_optimizers.py#L8-L13):
the original batch harmonic penalty computes a Hodge decomposition on in-batch
rewards, and **that quantity is identically zero for a scalar reward model**,
because scalar predictions are always gradient-consistent. The penalty term was
mathematically incapable of changing the gradient. Files 1–3 are therefore a
correct measurement of a method that was a no-op — the right result for the code
that produced it.

v3 (file 4) replaced that term with a *potential-alignment* regulariser against the
precomputed Hodge potential of the global preference graph, which is not
identically zero. That is the change that produces the headline table:

```
optimizer_comparison_hodge_v3_30seed.json
  Hodge-DPO   0.9999 ± 0.0005  vs DPO   0.9403 ± 0.0129   (d = 6.52,  p < 0.0001)
  Hodge-KTO   0.9964 ± 0.0026  vs KTO   0.8004 ± 0.0166   (d = 16.47, p < 0.0001)
  Hodge-GRPO  1.000  ± 0.0     =  GRPO  1.000  ± 0.0      (both at ceiling)
```

### Caveats that must travel with file 4

1. **The v3 effect size itself is valid; the cross-version story is what's
   confounded.** *(This item was overstated when first written on 2026-07-21 and
   is corrected here.)* Within file 4, DPO and Hodge-DPO run on **the same 30
   seeds and the same preference graph**, and all 30 paired differences are
   positive (mean +0.0596, sd 0.0130). That is a legitimate paired A/B of the loss
   function, and `d = 6.52` stands as a within-run measurement.

   What is *not* established is the **explanation** for why files 1–3 showed
   nothing and file 4 shows a large effect. Two things changed together: the
   regulariser, and the graph construction (cross-pair k-NN edges are now
   preserved under subsampling; without them H¹ is zero). The recorded
   `diagnosis_exploit_fraction` — an output of `HodgeDiagnosticCritic`, written
   into `config` at [`optimizer_comparison.py:202`](../src/optimizer_comparison.py#L202)
   — moved from **0.99–1.00** to **0.094**, and the baselines shifted too
   (DPO 0.9107 → 0.9403, KTO 0.8218 → 0.8004). So "the potential-alignment
   regulariser is what made Hodge variants work" is a plausible but unproven
   attribution. The clean ablation — v3 regulariser on the *old* graph, old
   penalty on the *new* graph — has not been run.

2. **GRPO is at the metric ceiling in all four files** (mean exactly 1.0, std
   exactly 0.0). Hodge-DPO at 0.9999 is effectively there too. A metric that
   saturates cannot rank the methods above it, and the headroom for the reported
   DPO improvement is the 6% between 0.94 and the ceiling.

3. **"Exploit resistance" here is ranking accuracy over a frozen list of preference
   pairs.** The `shared/` track has no environment, no generation step and no
   verifier — its "policies" are MLPs over frozen MiniLM embeddings. Nothing in
   this directory measures reward hacking under optimisation pressure, which is
   what the name suggests. See `.swarm/handoff_verifier_gap_unblock.md` for the
   thread that addresses this.

4. **The published training set was not the committed file (found 2026-09-18).**
   File 4 was produced by `shared/modal_runner.py::run_optimizer_comparison` from
   `pipeline/mapping.pkl` on the Modal volume `reward-hacking-results` (the volume's
   `pipeline/optimizer_comparison.json` is byte-identical to file 4). That mapping holds
   **2268 pairs: 2000 HH-RLHF harmless-base + 268 TRACE LLM counterfactuals**, and
   21,767 edges. The runner took a random 500-pair subsample with the **unseeded**
   global `np.random.choice` (`modal_runner.py:99`), so the exact subsample cannot be
   recovered and file 4 cannot be reproduced exactly. A replay of 200 seeded draws
   (`scripts/replay_published_v3_subsample.py`) gives 1470 ± 197 kept edges (1482 is the
   52nd percentile) and about 59 TRACE pairs per draw; the published baselines fall
   inside the across-draw range (DPO 0.907–0.963, KTO 0.735–0.827). The committed
   `counterfactual_pairs.json` (500 HH-RLHF pairs) is a different set. Fallback-row
   census on that committed file: 0 of 500 rows use the literal fallback `ideal_text`;
   a census of the 268 TRACE pairs inside the volume mapping has not been done.

5. **In-sample, and the targets belong to other pairs (found 2026-09-18).**
   `optimizer_comparison.py:167` calls `trainer.train(samples)`, and `train()` scores
   `evaluate_exploit_resistance(samples)` on the same list (`preference_optimizers.py:161`),
   so every number in file 4 is accuracy on the training pairs. In addition, after the
   subsample the kept edges stay in the original pair order while the samples follow the
   random draw order, and `hodge_diagnostic.py:230-251` reads sample *i*'s target and
   cycle weight from edge *i*. In the replay, on average 1.02 of 500 samples (maximum 6)
   received their own edge. **In the published run, each sample's Hodge target and cycle
   weight came from a different pair.** Every target was positive, so the regularizer
   acted as a generic positive-margin term. File 5's Misaligned arms reproduce this and
   also reach 1.000 in-sample.

6. **The cycles are constructed (found 2026-09-18).** Each direct edge joins two
   responses that appear in no other pair, so the direct edges contain no cycles.
   All cyclic structure comes from `preference_mapper.py::_compute_similarity_preferences`,
   which sets edge probabilities by fixed formulas of embedding cosine
   similarity, from every node toward each of its 5 nearest neighbours (so
   mutual neighbours get edges in both directions, each with p > 0.5). This
   benchmark does **not** measure how cyclic human preference data is. The
   "28% harmonic energy in HH-RLHF" figure that appeared in the July paper
   draft and READMEs has no supporting result file and is withdrawn.

### If you cite one number

Cite **file 5**: on held-out pairs, Hodge-DPO and Hodge-KTO do not beat their base
methods or a margin control, and all methods are near chance. If you mention file 4,
call it in-sample ranking accuracy from a non-reproducible run with misaligned targets
(caveats 4–6), never "exploit resistance" and never evidence for HodgePO. Do not delete
files 1–4: they are the
evidence that the null result was diagnosed and explained rather than discarded,
and a reviewer who finds them without this README will reasonably assume the
latter.

---

## Other files in this directory

- `peer_sheaf_e*.json` — SGB-012 peer-consistency sheaf experiments (E0–E6).
  `peer_sheaf_modal_summary.json` aggregates the Modal runs. Findings summarised in
  the project memory note on peer-sheaf results; `‖δ¹c‖` is a *selective* detector
  (strong on convincing-game and insider-trading, sign-flipped on alignment-faking
  reasoning, blind to overt/instructed lies). Do not describe it as a general
  deception detector.
- `counterfactual_pairs.json` — 500 HH-RLHF harmless-base pairs
  (`source = hh_rlhf_direct`); the data set for file 5. It is **not** the training set
  of file 4 (see caveat 4).
- `hh_rlhf_quick_results.json` — early smoke test, superseded by the optimizer
  comparison files. Not cited anywhere.
- `figures/`, `pipeline/` — generated artefacts.
