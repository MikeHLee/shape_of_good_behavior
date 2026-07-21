# `shared/results/` — provenance and supersession

**Last updated**: 2026-07-21

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
| 4 | `optimizer_comparison_hodge_v3_30seed.json` | 2026-03-24 | 30 | potential-alignment regulariser (v3) | **current** |

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

1. **v3 is not a clean A/B of the loss function.** The recorded
   `diagnosis_exploit_fraction` — an output of `HodgeDiagnosticCritic`, written into
   `config` at [`optimizer_comparison.py:202`](../src/optimizer_comparison.py#L202)
   — is **0.094** in v3 versus **0.99–1.00** in files 1–3. The graph construction
   changed at the same time as the loss (cross-pair k-NN edges are now preserved
   under subsampling; without them H¹ is zero). The baselines moved too
   (DPO 0.9107 → 0.9403, KTO 0.8218 → 0.8004), which they would not have done had
   only the Hodge term changed. The v3 effect size is therefore confounded between
   "new regulariser" and "different preference graph". A clean ablation — v3
   regulariser on the *old* graph, and the old penalty on the *new* graph — has
   not been run.

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

4. **Counterfactual fallback rows.** `shared/src/counterfactual_gen.py:245,258`
   silently substitutes the literal string `"I cannot assist with that request."`
   as `ideal_text` when an API call fails. Those rows are inside
   `counterfactual_pairs.json` and inside the exploit-resistance metric. A census
   of how many exist has **not** been done.

### If you cite one number

Cite file 4, with caveats 1–3 attached. Do not delete files 1–3: they are the
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
- `counterfactual_pairs.json` — TRACE counterfactual cache output. See caveat 4.
- `hh_rlhf_quick_results.json` — early smoke test, superseded by the optimizer
  comparison files. Not cited anywhere.
- `figures/`, `pipeline/` — generated artefacts.
