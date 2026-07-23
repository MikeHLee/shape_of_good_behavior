# Inquiry: Peer-Consistency Sheaves for Behavioral Shape Verification

**Date**: 2026-05-12
**Status**: Early-stage research question (extends INITIAL_INQUIRY.md)
**Track**: Part 3 of "The Shape of Good Behavior" series
**Companion docs**: [`README.md`](./README.md), [`INITIAL_INQUIRY.md`](./INITIAL_INQUIRY.md)

---

## The Central Question

Tracks 1 and 2 give us tools for measuring the shape of *one* model's feedback geometry and constraint geometry. Track 3 (as originally framed) asks what alignment looks like inside a *single* LLM's embedding space.

This inquiry adds a new axis: **does alignment have a shape that is invariant across independently-trained models, and can we use that invariance to detect models whose behavior is locally hiding misalignment?**

Concretely:

> Treat a set of independently-trained models as the open cover of a sheaf over the space of inputs. Each model is a stalk; cross-model feature-correspondence maps are restriction maps. Then "good behavior" is a *global section* of that sheaf, and a model that is locally faking it shows up as a sheaf-consistency failure (non-zero H¹) on its restriction maps to its peers.

The intuition: even if a single model learns to *mimic* the embedding-space signature of an aligned response, it is much harder to mimic that signature *and* keep it consistent with the way an unrelated peer model represents the same input. Hiding becomes a coordination problem across the peer panel.

---

## Why Now: Two Inputs That Reframe the Question

### 1. Goodfire's "The World Inside Neural Networks" (2026)
Goodfire's universality work shows that independently-trained models converge on a surprisingly large shared dictionary of features — that there *is* a "world inside" that is, to first order, model-independent. This gives us a credible *restriction map*: a way to translate a feature in model A into the corresponding feature in model B, with a measurable translation error.

If universal features are a real phenomenon and not an artifact of shared training data, then the shape of any concept — including alignment-relevant concepts — can be measured *across* models, not just within one.

### 2. Our Prior Thought Space Work
Thought Space is a coverage / clustering structure over ideas in embedding space — a portfolio that partitions embeddings into cells of related ideas. It is a *separate* artifact from the ontology work (which lives in `oasis-cloud/src/data`) and from the *Structure of Clear Thinking* track (a distinct research line). Useful here in one specific way: a shared clustering over ideas gives a categorical, model-agnostic feature ("which Thought Space cell does this response land in?") that can serve as a discrete, low-dimensional restriction-map target when continuous linear maps are too brittle. Not core to the sheaf construction — kept as an optional discrete-feature arm of E1+.

---

## The Sheaf Object, Precisely

Let:
- **Base space** `X`: the space of prompts (or prompt-context pairs) on which we want to verify behavior. In practice, a finite sample `X_n ⊂ X` plus a similarity / nearness relation that makes `X_n` into a simplicial complex or graph.
- **Peer panel** `M = {M_1, ..., M_k}`: a set of independently-trained models. Independence is the load-bearing assumption — see the red-herring discussion below.
- **Stalk over `x ∈ X` from `M_i`**: a feature vector (or small set of features) `F_i(x)` extracted from model `M_i` on input `x`. Candidates:
  - hidden-state embedding at a chosen layer
  - SAE / dictionary-learning feature activations (Goodfire-style)
  - principle-vector projections (Experiment B in `README.md`)
  - Thought Space projection onto a shared ontology
- **Restriction maps `ρ_{i→j}: F_i(x) → F_j(x)`**: learned or measured maps that translate `M_i`'s representation of `x` into the coordinates `M_j` would use. Candidates:
  - linear maps fit via Procrustes / CCA over a calibration set
  - permutation + scaling on matched SAE feature indices (universality-based)
  - projection through a shared Thought Space basis (model-agnostic embedding)
- **Sheaf condition (local consistency)**: on overlaps — i.e., on the same input `x` viewed by both `M_i` and `M_j` — we require `ρ_{i→j}(F_i(x)) ≈ F_j(x)`. The residual is the *peer-disagreement signal*.

A **global section** is an assignment `s: X → ⊔_i F_i` that is consistent under every `ρ_{i→j}`. The *shape of good behavior* on `X` is the conjectured global section that all sufficiently-aligned models agree on.

Note this is the same sheaf-theoretic skeleton as Track 1 (which sheafifies *preferences* over a comparison graph). The difference is what the stalks contain (features, not pairwise preferences) and what the cover is (models, not annotators).

---

## The Lossiness Metric

For a fixed input `x` and feature extractor choice, define the **peer-disagreement residual**:

    r_{ij}(x) = || ρ_{i→j}(F_i(x)) − F_j(x) ||

Aggregate to a per-input scalar **lossiness**:

    L(x) = (1 / |pairs|) · Σ_{i<j} r_{ij}(x)

We also want the *cohomological* aggregate — the bit that cannot be removed by any choice of section. Following Track 1's Hodge construction on cellular sheaves:

1. Build the cellular sheaf with 0-cells = (model, input) pairs and 1-cells = (input `x`, model pair `(i,j)`) with coboundary `δ_{(i,j)}(s)(x) = ρ_{i→j}(s_i(x)) − s_j(x)`.
2. Compute the Hodge decomposition of the 1-cochain space.
3. **Harmonic mass** of the residual = the part of disagreement that no choice of stalk values can explain away. This is the cross-model analogue of Track 1's H¹ obstruction.

Three derived quantities, each a candidate alignment signal:
- **Cross-model H¹ density** over a region of input space — high values flag prompts where peers structurally disagree, not just numerically.
- **Per-model leave-one-out residual** — how much consistency improves when `M_i` is dropped from the panel; a model that disagrees more than its peers is the suspect.
- **Restriction-map rank deficiency** — if `ρ_{i→j}` has to drop rank on a region of `X` to fit the data, that region is one where `M_i` and `M_j` represent different things; useful as an interpretability handle.

---

## Tokenizer Mismatch as Part of the Sheaf, Not Noise

Different models tokenize the same raw text differently, so the "input" the model sees is already model-specific before any hidden state is computed. Two ways to handle it inside the sheaf:

- **Lump it** (default for E1): treat tokenization as internal to the feature extractor `F_i`, with `F_i: \text{Text} → ℝ^{d_i}`. All sources of disagreement — tokenizer + representation — get absorbed into `r_{ij}(x)`. Simplest; what E0 already does.
- **Factor it** (E2+): add a tokenizer-level cell layer to the complex so that tokenizer residual and post-tokenization residual live at different cell-layers, each with its own coboundary.

**Methodological caveat on attribution.** Hodge decomposition separates *resolvable* (exact, `δs` for some 0-cochain) from *irreducible* (harmonic) disagreement. It does **not**, on its own, attribute the harmonic part to *causes* like "tokenization" vs "representation" — it is blind to cause. To attribute harmonic mass to a cause requires either a richer cell complex (with distinct coboundaries for the candidate causes) or intervention experiments (e.g. force-align tokenizations, recompute, take the delta). Track 1's validation of Hodge on preference cochains shows the *resolvable vs irreducible* split works robustly; *cause attribution* is not something Track 1 has tested and would need its own validation here.

The deeper point: tokenizer-induced disagreement is *part of* the peer-consistency signal, not noise to scrub. A panel of models with identical tokenization would be cheating (shared inductive bias — exactly what S3 stratifies against). We want the diversity; we just want to know how to read it.

---

## Why This Survives Some Hiding Strategies

A single model trying to hide misaligned behavior has to make its *outputs* look fine. But a peer-consistency sheaf measures the shape of its *internal representation* relative to peers. To fool the sheaf, the model would have to:
1. Produce an output that looks aligned, and
2. Produce an internal representation of that output whose translation under every `ρ_{i→j}` matches what each peer would have produced for an *actually-aligned* response.

Step 2 requires a coordinated, multi-coordinate-system spoof. It is strictly harder than output-only spoofing, and gets harder as the panel diversifies. This is the core of the conjecture: peer-consistency is hard to fake even when output-consistency is not.

---

## The Red-Herring Failure Mode (Sentinel Design)

The above only works if peers are *independent enough* that they would not all reward-hack toward the same wrong shape. The failure case is shared inductive bias: every model on the panel was trained on overlapping corpora and similar objectives, so they all learn the same proxy, and the sheaf is maximally consistent on a behavior that is uniformly wrong. Consistency then becomes evidence of *shared confusion*, not of alignment.

So the sheaf gives high-confidence evidence in *one direction only*: high disagreement is evidence something is off. High agreement is not, on its own, evidence things are fine.

We need explicit sentinels to calibrate the "too-clean" case. Three to start:

### Sentinel S1: Known-Hard Inputs
Curate inputs where the *correct* aligned behavior is empirically known to be non-trivial — adversarial red-team prompts, alignment-faking probes, subtle deception scenarios. Peers that *all* score high consistency on these inputs are either (a) all genuinely competent, or (b) all sharing a proxy. Use a held-out, mechanistically-verified subset to distinguish.

### Sentinel S2: Injected Proxy-Hack Pair
Construct paired inputs `(x, x')` that differ only in a feature the proxy would key on but that the true alignment property does not depend on. A panel that locks consistency to `x` vs. `x'` in proportion to the proxy feature is reward-hacking the proxy in unison. Quantify with a *proxy-sensitivity index*: drop in cross-model H¹ density between `x` and `x'` after a known-irrelevant edit.

### Sentinel S3: Diversity Stratification
Stratify the peer panel by training-corpus overlap and objective family. Re-compute the sheaf within each stratum *and* across strata. If consistency only survives within a stratum (e.g., among models trained on the same web crawl) but collapses across strata, the "global section" is corpus-shaped, not alignment-shaped. Report the *stratum gap* alongside the headline lossiness.

A panel is *certified* for this method only if S1, S2, S3 all pass on a calibration set with known ground truth.

---

## Connections Back to Tracks 1 and 2

| | Track 1 (Feedback Geom.) | Track 2 (Constraint Geom.) | Track 3 (Peer-Consistency Sheaf) |
|---|---|---|---|
| **What is the sheaf over?** | annotators / comparison graph | safety-state manifold | inputs, with peer models as cover |
| **What does H¹ ≠ 0 mean?** | irreducible preference cycles | (n/a — uses metric, not sheaf) | irreducible cross-model representational disagreement |
| **What is the global section?** | a consistent scalar reward | a safe geodesic policy | the shape of good behavior |

The same Hodge machinery already implemented for Track 1 (`shared/src/hodge_diagnostic.py`, `feedback_geometry/`) should port to this setting with a different coboundary operator. That is the cheapest path to a first experiment.

---

## First Experiments (Order of Effort)

### E1: Procrustes baseline on a small panel (~1–2 weeks)
- Panel: 3 open-weights models with diverse training (e.g., Llama-3-8B, Mistral-7B, Qwen-7B; later add a Pythia for an objective-family contrast).
- Feature extractor: mid-layer hidden state, mean-pooled over response tokens.
- Restriction maps: Procrustes / CCA fit on a 5k-prompt calibration set.
- Inputs: HH-RLHF aligned/non-aligned pairs.
- Measure: `r_{ij}(x)` and `L(x)` distributions; do they separate aligned from non-aligned? Does the separation generalize to a held-out adversarial set?

### E2: Hodge harmonic mass (~2–3 weeks)
- Build the cellular sheaf and compute H¹ via the Track 1 Hodge code, adapted.
- Compare: per-input H¹ density vs. peer-disagreement residual `L(x)`. The harmonic component is the part that matters; everything else is noise/recoverable.

### E3: Red-herring sentinels (~3–4 weeks)
- Build S1, S2, S3 calibration sets.
- Run on E1's panel — does the panel pass certification?
- If not, what does the failure pattern look like? Report it; that itself is a finding.

### E4: SAE / universal-feature restriction maps (~exploratory)
- Replace linear restriction maps with feature-index correspondences via SAEs (Goodfire-style universality). Expected: tighter residuals on inputs where universal features apply, larger residuals on model-specific behaviors — itself a diagnostic.

### E5: Discrete-cell restriction maps (~exploratory)
- Use a shared clustering over response embeddings (e.g. Thought Space-style coverage cells) as a categorical feature: each model assigns each response to a cluster; the restriction map becomes "does model i's cluster assignment match model j's?" Trades continuous fidelity for model-agnosticism and is robust to tokenizer / dim mismatch; useful as a sanity arm when the linear restriction maps are unstable.

---

## Falsification Criteria

The whole research direction is dead if any of:
- `L(x)` does not separate aligned from non-aligned responses better than a random projection baseline (E1).
- The harmonic component of cross-model disagreement is empirically negligible compared to noise (E2).
- No panel of plausibly-independent models passes the S1–S3 sentinels (E3) — meaning we can never get a clean signal.

The direction is *interesting but not load-bearing* if `L(x)` separates aligned from non-aligned but only at granularity comparable to a single-model linear probe — in that case it is a redundant signal, not a complementary one.

---

## Status and Next Steps

**Immediate (this month)**:
1. Stand up E1 on a 3-model panel with existing `embedding_topology_analyzer` infrastructure.
2. Write a small `peer_sheaf/` module under `shared/src/` exposing `compute_residuals`, `compute_harmonic_mass`, and `run_sentinels`.
3. Decide: does this become its own paper, or a section of the Track 3 paper? (Initial lean: section, until E1/E2 results suggest otherwise — see `feedback_publishing_cadence` memory.)

**Open questions to resolve before E1**:
- Which layer's hidden state to use as the stalk? (Probably sweep; prior literature suggests mid-to-late.)
- Calibration-set size for stable Procrustes fits across model dimensions of different sizes?
- How to handle tokenizer mismatch — response-level pooling is the obvious dodge but loses positional structure.

---

## Results Log

> **Writeup**: consolidated narrative in [`WRITEUP_PEER_DECEPTION_DIVERGENCE.md`](./WRITEUP_PEER_DECEPTION_DIVERGENCE.md).

### E6 / SGB-017→021 (2026-05-21→24) — intentional-deception RESPONSES; length + scale controls; the decisive result

- **Setup**: Swapped HH-RLHF for **LIARS'-BENCH** (`Cadenza-Labs/liars-bench`, gated; access granted via click-through) — pre-generated model responses with a boolean `deceptive` label, so the response itself carries a hiding posture. Same sheaf/`‖δ¹c‖` metric. Statistic switched to **AUC** over 5 split-seeds (the norm is heavy-tailed; Cohen's d is unstable, AUC reproducible). Added per-edge / per-triangle / per-model **structure** output, a `--length-match` control (per-quantile-bin equalization of dec/hon response length), and `cocycle_blockwise` (per-triangle δ¹c without the unused 23.5k×11.8k SVD).
- **Code**: [`scripts/peer_sheaf_e6_liars.py`](../../scripts/peer_sheaf_e6_liars.py), [`scripts/peer_sheaf_e6_modal_analysis.py`](../../scripts/peer_sheaf_e6_modal_analysis.py), [`shared/modal_peer_sheaf.py`](../../shared/modal_peer_sheaf.py) (`embed_liars_panel`). Results: `shared/results/peer_sheaf_e6_*.json`, `..._modal_*.json`.
- **Config screening**: 4 of 7 LIARS'-BENCH configs are response-degeneracy artifacts (single-token / single-char / templated-homogeneous classes) — soft-trigger, harm-pressure-choice, gender-secret, harm-pressure-knowledge-report — discarded. Clean substantive configs: convincing-game (persuasion), insider-trading (strategic/agentic), instructed-deception (overt lie).
- **Length confound (small panel)**: raw AUC 0.61–0.70 → length-matched: insider-trading 0.625→0.555 (n.s.), instructed-deception 0.610→0.536, convincing-game 0.701→**0.678** (held; deceptive responses are *shorter* → confound runs against it).
- **Capacity confound / decisive 7–9B test** (length-matched, panel {Yi-1.5-9B, Zephyr-7B, Qwen2.5-7B}):
  - convincing-game: **AUC 0.662 ± 0.033, p=2.5e−6** — HOLDS at scale.
  - insider-trading: **AUC 0.637 ± 0.006, p=8e−8** — RECOVERS at scale (sub-2B collapse was a capacity limit, not length). 
  - Per-model divergence ~uniform under length control; the small-panel "odd-one-out" asymmetry was an artifact.
- **Headline**: cross-model representational divergence on deceptive *responses* is a **real, length-independent, scale-robust** signal for ≥2 deception types — but it is a *relative divergence pattern*, not an absolute detector, and it **emerges with model capacity**. The earlier HH-RLHF "competence-not-deception" reading (E3) was confounded by n=5 deceptive + deceptive/harmful pooling; the in-distribution deceptive cell, once grown to n=21, separates at AUC 0.71 (d=+0.72, p=0.006).
- **Cost**: ~$7 Sonnet labeling + ~$0.50 A100. All else local.

### E5 / SGB-016 (2026-05-21) — bare true/false statements are FLAT (weak proxy)

- **Setup**: `notrichardren/azaria-mitchell` true/false statements (ungated), small panel, `‖δ¹c‖`(false) vs (true). [`scripts/peer_sheaf_e5_truefalse.py`](../../scripts/peer_sheaf_e5_truefalse.py).
- **Result**: AUC 0.504, d=+0.026 (flat). **Caveat**: ultra-short single declaratives the small panel likely doesn't *know* → nothing to disagree about. A weak test. E6 (same azaria-mitchell-cities content reframed as deceptive vs honest *responses*) is the controlled upgrade and is NOT flat — isolating "deceptive posture" from "bare truth-value".

### E2 (2026-05-13) — Vector-valued Hodge on cached E1 embeddings: harmonic is vacuously zero; cocycle-violation is the real signal

- **Setup**: Cellular sheaf on the E1 panel. 0-cells = models, 1-cells = directed model pairs, 2-cells = ordered triples. Stalks = R^{d_i} (full hidden-dim per model). Restriction maps = the same affine ridge-fit maps as E1. Coboundary δ⁰: V → E with (δ⁰s)_{(i,j)} = W_{ij} s_i − s_j. Triangle coboundary δ¹: E → F with (δ¹c)_{(i,j,k)} = ρ_{j→k}(c_{(i,j)}) + c_{(j,k)} − c_{(i,k)}. D_V = 3904, D_E = 7808, D_F = 7808, rank δ⁰ = 3904.
- **Code**: [`shared/src/peer_hodge.py`](../../shared/src/peer_hodge.py), [`scripts/peer_sheaf_e2.py`](../../scripts/peer_sheaf_e2.py). Results: [`shared/results/peer_sheaf_e2.json`](../../shared/results/peer_sheaf_e2.json).
- **Methodological finding (headline)**: under this sheaf, **the harmonic component (im δ⁰)^⊥ is numerically zero** for every measured cochain. Identity: c_{(i,j)}(x) = W_{ij} F_i(x) + b_{ij} − F_j(x) = W_{ij}(F_i(x) − μ_i) − (F_j(x) − μ_j) = (δ⁰(F(x) − μ))_{(i,j)}, so the data itself supplies a valid stalk section. The originally hoped-for "irreducible vs resolvable" split — the H¹ analogue from Track 1 — is *vacuous in this formulation*: with arbitrary linear restriction maps and full-rank stalks, every cochain is exactly a coboundary. The "no choice of stalk values explains it away" framing fails because s = F(x) − μ is always such a choice.
- **What actually carries sheaf structure**: the cocycle violation δ¹c. With linear maps fit independently per pair, the panel's pairwise translations do *not* compose around triangles (δ¹δ⁰ ≠ 0), so δ¹c is a non-trivial channel: it measures the part of c that would require simultaneously satisfying three pairwise constraints. This is the genuine sheaf-cohomological obstruction to a global section, and it cannot be absorbed by any stalk reassignment.
- **Numbers** (eval n=1000):
  - total norm ||c||:                 d = +0.062, AUC = 0.531, p = 0.17
  - resolvable ||c_res|| ≡ ||c||:     d = +0.062, AUC = 0.531  (harmonic ≈ 1e-13)
  - **cocycle violation ||δ¹c||**:    d = +0.052, AUC = 0.526, p = 0.25
  - All 6 ordered triangles show rejected > chosen (sign-consistency p = 0.5⁶ = 0.016). Best individual triangle SmolLM2→Qwen2.5→TinyLlama has p = 0.043.
- **Interpretation**: at this scale the cocycle-violation signal exists and is sign-consistent across all triangles (the 6/6 finding from E1 reproduces at the cocycle level), but it is *not stronger than the raw cochain norm*. Restriction-map non-composability is a present but not dominant channel of the chosen/rejected separation in small instruct LMs.
- **Implications for the inquiry**:
  1. The E2 falsification criterion ("harmonic component is empirically negligible compared to noise") needs to be re-stated. Under the affine-restriction-map sheaf the harmonic component is zero *by construction*, not because of noise. Either we replace the sheaf (low-rank stalk subspace; see SGB-014) or we adopt cocycle-violation as the irreducible-channel metric.
  2. The E1 finding "L separates with d=0.07" is correctly read as a *total-norm* finding; nothing in E2 splits it into a meaningful resolvable/harmonic dichotomy without further machinery.
  3. Per-input ||δ¹c|| is the right metric for SGB-013 stratification (Anna Karenina vs coordination). If specifically deceptive-rejected responses show high cocycle violation while incoherent-rejected responses do not, that argues for the coordination hypothesis.
- **Action**:
  - (a) **SGB-013** with ||δ¹c|| as the per-input score on category-stratified rejected pile.
  - (b) **SGB-014** (new): replace stalks with shared low-rank subspaces (e.g. top-k PCA per model on the calibration set) so im(δ⁰) becomes a proper subspace of E and the harmonic split becomes nontrivial. Test whether the same 6/6 sign-consistency holds.
  - (c) SGB-012 (7B–8B Modal panel) — scaling — should re-use the same peer_hodge code path.

---

### E1 (2026-05-13) — Small instruct-LM panel on HH-RLHF: weak positive in predicted direction

- **Setup**: Panel = SmolLM2-360M-Instruct (960d), Qwen2.5-0.5B-Instruct (896d), TinyLlama-1.1B-Chat (2048d). Three different orgs, three different bases. Feature = last-real-token hidden state of the final layer (no chat template — raw response text as continuation context). 2000 HH-RLHF pairs, 50/50 calibration/eval split, calibration pooled alignment-agnostically across chosen+rejected, ridge linear restriction maps (λ=1e-3), cosine residual. Runtime: 13 min on this Mac (MPS, bf16).
- **Code**: [`scripts/peer_sheaf_e1.py`](../../scripts/peer_sheaf_e1.py). Results: [`shared/results/peer_sheaf_e1.json`](../../shared/results/peer_sheaf_e1.json). Cached per-model embeddings: `shared/data/cache/peer_sheaf_e1/`.
- **Headline**: L(chosen) = 0.303 ± 0.234 vs L(rejected) = 0.319 ± 0.236. Welch t = 1.58 (p = 0.11), Cohen's d = +0.071, AUC = 0.530.
- **Direction-consistency finding**: All 6 directed pairs show `delta = L_rejected − L_chosen > 0` (range 0.010–0.024). Under a null with 50/50 sign direction, all-six-positive has p = 0.5⁶ = 0.016 — *the sign consistency is significant even though the t-test on the pooled metric is not*. Per-pair AUCs all fall in [0.520, 0.535]; per-model LOO AUCs all in [0.528, 0.530].
- **Comparison to E0**: AUC moved from 0.482 (E0, encoder-style features) to 0.530 (E1, autoregressive features), and the sign of Cohen's d flipped from −0.065 to +0.071 (predicted direction). This is the central scoping result: peer-consistency lossiness *does* track aligned-vs-non-aligned in autoregressive features and *does not* in retrieval-trained encoders. The "what kind of feature carries the signal" question now has a first answer.
- **Why the t-test is weaker than the sign-consistency test**: the t-test sees the residuals as i.i.d. scalars and washes out structure. The sign-consistency test exploits that we have six *independent* directed-pair channels all measuring the same underlying object. The latter is the more informative null for sheaf-based signals.
- **What this is NOT**: a publishable headline. The effect size is small (d ≈ 0.07), the panel is small (sub-2B params), and the calibration set is small enough that linear maps are heavily ridge-regularized. This is *evidence that the inquiry is alive*, not a result.
- **Action**: Two parallel next steps. (a) E2 on cached embeddings: compute the Hodge harmonic mass per input and re-test — the resolvable component may be diluting the signal. (b) Modal arm with a 3× 7B-8B panel for the scaling test — if d goes from 0.07 to ≥0.3, the inquiry survives; if d stays at 0.07, the signal is real but too weak to be the alignment-faking detector we wanted.

---

### E0 (2026-05-12) — Sentence-transformer panel on HH-RLHF: null result, as expected

- **Setup**: Panel = `all-MiniLM-L6-v2` (384d), `all-mpnet-base-v2` (768d), `BAAI/bge-small-en-v1.5` (384d). 2000 HH-RLHF pairs, response-only embedding, 50/50 calibration/eval split with calibration pooled across chosen+rejected (alignment-agnostic). Ridge linear restriction maps (λ=1e-3), cosine residual.
- **Code**: [`scripts/peer_sheaf_e0.py`](../../scripts/peer_sheaf_e0.py), [`shared/src/peer_sheaf.py`](../../shared/src/peer_sheaf.py). Results: [`shared/results/peer_sheaf_e0.json`](../../shared/results/peer_sheaf_e0.json).
- **Headline**: L(chosen) = 0.289 ± 0.079 vs L(rejected) = 0.284 ± 0.077. Welch t = −1.45 (p = 0.15), Cohen's d = −0.065, AUC = 0.482. All 6 directed-pair AUCs and all 3 leave-one-out AUCs fall in [0.474, 0.492].
- **Interpretation**: The mechanics work — restriction maps converge, residuals are well-defined and stable. But peer-consistency on these encoders shows **no separation between aligned and non-aligned responses**, and what tiny direction exists is *reversed* from the hypothesis (rejected is marginally more peer-consistent).
- **Why this is not a falsification of the inquiry**: Sentence-similarity encoders are trained *to suppress* style/intent differences between paraphrases — exactly the features the inquiry wants to detect. E0 tells us this class of encoder cannot carry the signal, which is informative scoping for E1 (LLM hidden states are explicitly different in this respect).
- **Action**: Proceed to E1 with instruction-tuned LLM hidden states. Treat E0 as a "negative control encoder" — if E1 also shows AUC ≈ 0.5, that *is* a falsification.
