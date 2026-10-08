# SGB-046 pre-registration — Hodge structure as a reward-model diagnostic for heterogeneous grader panels

**Written:** 2026-10-07, before any SGB-046 code or data exists.
**Status:** pre-registered. Do not edit the hypotheses, metrics, or kill criteria after a run.
Add any later change in § 9 with a date and the reason.

## 0. Why this experiment

- SGB-045 Stage 1 (`SGB045_STAGE01_RESULTS.md`) showed that Hodge-targeted **acquisition** gives
  a worse global ranking than random or Fiedler acquisition. The decomposition itself works.
- The design note `docs/DESIGN_NOTE_RELATIVE_REWARDS.md` (Kolmogorov, 2026-09-28) showed:
  - curl mass is not intransitivity: binary labels and single-direction judging create curl on
    transitive data;
  - on a complete block (K_n) the harmonic part is zero, so value conflict lives in **curl**;
  - the directed **cycle count** with a null computed from the observed margins is the primary
    screen; curl is a secondary magnitude.
- SGB-045 Stage 1 used binary labels, one annotator per pair, and a sparse global graph. Part of
  the curl it targeted was therefore a thresholding artefact (see § 9 of that file's note).

So this experiment does **not** ask whether Hodge improves training or data. It asks a narrower
question: **in a panel of heterogeneous graders (for example constitutional principles), does
the cycle and curl structure tell us (a) where a scalar reward model will fail and (b) which
principles cause the failure, better than simple disagreement statistics do?**

## 1. Data-generating process (simulation, CPU only)

- **Prompt blocks.** Each block has k = 5 responses and all 10 pairs (b₁ of the graph = 6; the
  clique complex has 10 triangles). This is the collection design from the design note § 5.
- **Responses** have latent scores on P principles (P = 4 or 6).
- **Principle graders.** Grader p prefers a over b with a Bradley–Terry probability on the
  utility gap u_p(a) − u_p(b), with a per-grader temperature. Each grader is transitive by itself.
- **Each comparison is judged in both presentation orders**, with an optional first-position bias,
  and then antisymmetrized: f ← ½(f(a,b) − f(b,a)). Labels are **graded** (log-odds from V votes
  per grader per order), not binary, except in the binarization control.
- **Aggregate label** = majority over principles (constitution vote), or a weighted sum (the
  transitive control).
- **Ground truth** per block, from noise-free utilities: the true aggregate relation, the true
  directed 3-cycles, and for every principle pair the true **cycle-causing tension**: the set of
  true cycles that disappear when principle p or q is removed (leave-one-out on the true utilities).

### Scenarios (fixed now)

| Scenario | Purpose |
|---|---|
| `transitive` — weighted-sum aggregation | negative control: no true cycles |
| `transitive_binarized` — same, labels thresholded to ±1 | artefact control (curl expected, no cycles) |
| `transitive_neartie` — same, items nearly tied | noise-flip control |
| `condorcet` — majority over P balanced principles | true cycles from aggregation |
| `harmless_disagreement` — two principles disagree strongly, a dominant third decides the majority | disagreement without cycles |
| `mixed` — some principle pairs cause cycles, some only disagree | the main test for D3 |
| `position_bias` — one presentation order only, first-position bias | antisymmetrization check |

## 2. Hypotheses and metrics

### D1 — calibration gate (must pass before D2–D4 are read)

Per-block test: "this block contains a directed 3-cycle", using the cycle count with a
parametric-bootstrap null from the block's observed margins (B = 200).

- **Pass:** false-positive rate ≤ 0.05 (upper 95% Clopper–Pearson bound ≤ 0.08) on each transitive
  scenario, including `transitive_binarized` and `transitive_neartie`.
- **Expected and reported, not a gate:** a curl-mass test with a naive null has an inflated
  false-positive rate on `transitive_binarized` (this replicates the design note).
- **If D1 fails, stop.** Report the failure. D2–D4 are not interpreted.

### D2 — failure localization (primary)

A scalar Bradley–Terry reward model is fitted on the aggregate labels of training blocks
(shared item features, so it must generalize across blocks). On held-out blocks, predict the
event "the scalar RM is wrong on this pair relative to the true aggregate relation".

- **Baseline features (per pair):** panel disagreement (vote entropy across principles), |observed
  aggregate margin|, and the RM's own |predicted margin|.
- **Hodge features (per pair):** the number of certified cycles through the edge, the edge's curl
  component |c_e|, and the block-level cycle flag.
- **Metric:** held-out AUROC of a logistic regression with baseline features vs baseline + Hodge
  features, fitted on training blocks and scored on held-out blocks.
- **Prediction:** Hodge features add ≥ 0.02 AUROC in `condorcet` and `mixed`, with a paired
  bootstrap 95% CI over seeds that excludes 0.
- **Kill:** ΔAUROC CI includes 0 in both scenarios → the diagnostic adds nothing over disagreement.
- **Construction disclosure:** a scalar fit must err on at least one edge of every true 3-cycle.
  So some predictive power is structural. The test is whether the certificate, computed from
  finite noisy panel labels, localizes the errors better than disagreement does. Disagreement can
  be low on every edge of a Condorcet cycle; that is the case where the two must differ.

### D3 — attribution to principle pairs (primary)

For each principle pair (p, q), estimate a tension score from data:
**Hodge score** = the number of certified cycles that are removed by leave-one-out of p or q
(recompute the aggregate without that principle, re-test cycles), averaged over blocks.

- **Baseline score:** the raw pairwise disagreement rate between graders p and q over all pairs.
- **Truth:** the cycle-causing tension from § 1.
- **Metric:** Spearman correlation with the truth over principle pairs, and precision@k
  (k = number of truly cycle-causing pairs), per seed; paired Wilcoxon over seeds, Hodge vs baseline.
- **Prediction:** Hodge > baseline in `mixed` and `harmless_disagreement` (the baseline flags the
  harmless pair), p < 0.05 after Holm correction over {D2, D3}.
- **Kill:** no significant win in `mixed` → leave-one-out cycle attribution is not better than
  counting disagreements.

### D4 — decision relevance (secondary, exploratory)

Fit a skew-symmetric pairwise model f(a, b) (a general preference model, not a difference of
scores) and the scalar BT model on the same training blocks. Report the held-out accuracy gain
on the true aggregate relation, **stratified by the D1 block flag**.

- **Prediction:** the gain is concentrated in flagged blocks (interaction > 0).
- This is the design note's falsifier 2. No correction; reported as exploratory.

## 3. Design constants (fixed now)

- Seeds: 40 (0–39). Blocks per seed: 200 train, 100 held-out.
- k = 5. P ∈ {4, 6}. Votes per grader per order V ∈ {5, 25}. Bootstrap B = 200.
- Primary family: D2 and D3, Holm correction over the two. D1 is a gate. D4 is exploratory.

## 4. Assertions the code must make

- Item, edge, triangle, and label indices are aligned (assert, do not assume).
- No held-out block contributes to any fitted model or threshold.
- Both presentation orders are collected in every scenario except `position_bias`.
- The positive control (`condorcet`) must recover planted cycles. If recall is < 0.5 at V = 25,
  the pipeline is broken; stop and report.

## 5. What this experiment cannot show

- Everything is simulated. The graders are a model of constitutional principle graders, not real
  LLM judges and not humans. Per the design note § 4, a synthetic result must never be cited as
  evidence about human or real-grader preferences. It shows only that the diagnostic **can** work
  when the structure is present.
- The real-grader test is SGB-045 Stage 2 (Kolmogorov, commit 64ee6db): log-prob judges with a
  constitution ablation.

## 9. Changes after registration

(none)
