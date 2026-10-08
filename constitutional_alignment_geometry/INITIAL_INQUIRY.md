# Initial Inquiry: Constitutional Alignment Geometry

**Date**: February 2026
**Status**: Early-stage hypothesis generation
**Track**: Part 3 of "The Shape of Good Behavior" series

---

## The Central Observation

In Tracks 1 and 2, we developed tools for:
1. **Measuring inconsistency in feedback** (Hodge decomposition of preference graphs)
2. **Building geometric hard safety constraints** (conformal metric singularities)

Both tracks operate on *external* feedback — what humans say about model outputs. But LLMs have their own internal representations — **embedding spaces** that are the natural computational substrate of the model. The models themselves "think" in these spaces.

Track 3 opens a different question: **what does alignment look like from the inside?**

---

## The Alignment Differential

Consider two responses from an LLM to the same prompt:
- **Aligned response**: Honest, helpful, not deceptive
- **Non-aligned response**: Deceptive, harmful, sycophantic

These responses have *different embeddings* in the model's hidden state space. The **alignment differential** is the vector in embedding space pointing from non-aligned to aligned:

    Δ_align = E[embed(aligned)] - E[embed(non-aligned)]

This is the simplest possible "constitutional gradient" — a linear direction in embedding space that points toward better behavior.

**Key questions**:
1. Is Δ_align consistent across prompts, models, and domains?
2. Is it related to the principal component of the alignment-relevant information?
3. Does the Hodge decomposition of Δ_align over the space of constitutional principles reveal value tensions?

---

## Three Geometric Hypotheses

### H1: Alignment is Directional
**Hypothesis**: The alignment differential Δ_align is a consistent direction in embedding space — not just a noise vector. Formally, the cosine similarity between Δ_align computed on different prompt/response pairs is significantly > 0.

**Why this matters**: If true, we can extract a single "alignment axis" from data, enabling geometric interpretability of alignment without supervised labels on individual tokens.

**Falsification criteria**: If cosine similarity ≈ 0 across different prompts/models, the hypothesis fails. It would indicate alignment is prompt-specific rather than geometric.

---

### H2: Non-Aligned Responses Have Higher Topological Complexity

**Hypothesis**: Non-aligned responses occupy regions of embedding space with higher H¹ cohomology (more "holes", more topological inconsistency) than aligned responses.

**Intuition**: Deceptive or harmful responses require internally inconsistent reasoning — saying one thing while "meaning" another. This internal inconsistency should manifest geometrically as higher H¹ in the neighborhood of the embedding.

**Formal connection**: If the embedding space is modeled as a sheaf over the prompt-response comparison graph, then non-aligned responses contribute disproportionately to the harmonic component ω (Track 1).

**Falsification criteria**: If H¹ is equivalent for aligned and non-aligned responses, the hypothesis fails.

---

### H3: Constitutional Principles Form a Non-Transitive Preference Order

> **⚠️ CORRECTED 2026-09-28 — read this before using H3.** As written below, H3 is
> **false as stated** on a dense comparison graph, and its empirical version is
> **unmeasurable** in our current data. Two separate errors:
>
> **(1) The invariant is wrong: conflict is curl, not H¹.** On a *complete*
> comparison graph the clique complex of Kₙ is a simplex — contractible — so
> **H¹ = 0 identically**, and the harmonic component is exactly zero no matter how
> violently the principles disagree. All of the non-scalar residual lives in
> **curl** (local 3-cycles), not in cohomology. Demonstrated on identical utilities
> in `cai_geometry/mixture_curl.py`: on K₆ the residual is pure curl with H¹ = 0,
> while the same utilities on a 4-cycle (no triangles) give a genuine harmonic
> circulation of ±0.03433 on every edge. **A nonzero harmonic class therefore
> measures annotation sparsity, not value conflict** — collect the triangles and the
> hole closes. Any claim of "irreducible value tension" read off harmonic mass is
> partly reading off our own sampling design. Restate H3 in terms of curl, or argue
> that the comparison graph is *intrinsically* sparse and say why.
>
> **(2) The data cannot exhibit the phenomenon at all.** The label graph of
> `counterfactual_pairs.json` has **b₁ = E − V + C = 500 − 998 + 498 = 0** (a
> forest). On a forest, curl *and* harmonic energy are exactly zero for every
> possible edge flow — so no labelling of our data could have shown intransitivity.
> And `principles_violated` is **empty on all 500 pairs**, so no
> principle-conditioned labels exist to form a mixture from in the first place.
> See `shared/results/README.md` § 6a.
>
> **What survives.** The underlying obstruction is real and theorem-shaped: a
> mixture of principle-conditioned Bradley–Terry models is not representable by any
> single scalar reward, because a scalar's logit flow is grad s and hence exactly
> curl-free. That is a statement about *label processes*, and it is why H3 is worth
> rescuing. But note the sharp limit found the same day: the obstruction does **not**
> imply a performance benefit. A scalar reward fitted to curl-bearing mixture labels
> never escaped the convex hull of the Pareto frontier across a full β×w sweep
> (`cai_geometry/nonconvex_frontier.py`), whereas a one-line concave aggregator (Nash
> log, or Chebyshev max-min) reaches the concave interior immediately with no geometry
> at all. **Curl > 0 constrains what the labels are, not where the fit lands.**
>
> **To make H3 testable**, the data must be collected differently: ≥ 3 responses per
> prompt with all pairs compared (K₃ gives b₁ = 1, K₅ gives b₁ = 6), each comparison
> labelled under a *named* principle, and ≥ 2 principles that actually disagree on
> some triangle. Cost is O(n²) comparisons per prompt, not O(1).

**Hypothesis**: When constitutional principles are modeled as vectors {v₁, ..., vₖ} in embedding space, they exhibit non-trivial harmonic structure — obeying one principle fully may make another harder to satisfy.

**Example tension**: "Be maximally helpful" vs. "Be maximally safe" are not collinear in embedding space. A policy that optimally follows one may fail the other — this is a genuine value conflict, not a noise artifact. The harmonic component of the principle system (via Hodge decomposition) quantifies the irresolvable tension.

**Implication for Constitutional AI**: Constitutional AI assumes principles can be ranked and combined — but if their embedding vectors have non-trivial H¹, no scalar alignment score can capture all of them. A Hodge-augmented alignment model is necessary.

---

## The Value of This for the "Shape of Good Behavior" Blog

The blog post ("The Shape of Good Behavior: A Geometric Approach to AI Alignment") needs this track because:

1. **Tracks 1 and 2 are about training** — how to detect bad feedback and enforce safety during training.
2. **Track 3 is about the model itself** — what the model "knows" about alignment in its own representations.

The blog narrative arc:
- Part 1: "The problem with scalar feedback" (Track 1 motivation)
- Part 2: "How to detect when your feedback data is self-contradictory" (Track 1 results)
- Part 3: "How to build geometric safety constraints that cannot be violated" (Track 2 results)
- Part 4: "What good behavior looks like from the inside" (Track 3 preliminary results)

Track 3 is the most philosophically interesting: it's not about training algorithms but about the *geometry of values* — the claim that "being helpful", "being honest", and "avoiding harm" are not just verbal commitments but geometric structures in the AI's representation space.

---

## Connecting Formalism to Intuition

### The embedding space as a "mind map"

Think of the LLM's embedding space as a high-dimensional map of meanings. Each position in this space corresponds to a concept, sentiment, intent, or reasoning pattern. When the model generates a response, it traces a path through this space.

Constitutionally aligned responses trace paths through certain regions — the "good behavior region." Non-aligned responses end up in different regions — the "bad behavior zones" or "alignment black holes."

This track is about mapping those regions precisely.

### The alignment gradient as a compass

If H1 is true (alignment is directional), then we can construct a "compass" that always points toward more aligned responses. Fine-tuning a model in the direction of this compass is equivalent to pushing the model's representations into the good behavior region.

This is related to RLHF but geometric: rather than training on human labels (which have H¹ ≠ 0 issues identified in Track 1), we train on geometric labels derived from the structure of the embedding space itself.

### Constitutional tensions as topological holes

If H3 is true (constitutional principles have harmonic structure), then there are "holes" in the alignment space — regions where you cannot simultaneously satisfy all principles. These holes are precisely the H¹ generators of the principle vector system.

This formalizes and extends Constitutional AI: instead of assuming a fixed constitution, we *discover* the constitutional geometry from data, including the unavoidable tensions between principles.

---

## Immediate Literature to Review

1. **Bai et al. (2022)**: Constitutional AI — foundation for this work
2. **Zou et al. (2023)**: "Representation Engineering" — finds linear directions in LLM embeddings corresponding to emotional states; most directly related to H1
3. **Elhage et al. (2022)**: "Toy Models of Superposition" — how features are geometrically organized in LLM embeddings
4. **Park et al. (2023)**: "The Linear Representation Hypothesis" — supports H1 (alignment is a linear direction)
5. **Anthropic's Model Card methodology** — implicit constitutional principles
6. **Marks & Tegmark (2023)**: "The Geometry of Truth" — finds linear truth directions in LLM embeddings; parallel to our alignment directions
7. **Burns et al. (2023)**: "Discovering Latent Knowledge in LLMs" — unsupervised alignment signal from model internals

---

## What Success Looks Like

This research track produces a **paper** (12-24 months from now) that:

1. Establishes H1-H3 empirically on a large-scale LLM (LLaMA-3 or equivalent)
2. Provides a formal mathematical framework connecting constitutional principles to geometric structures in embedding space
3. Shows that Hodge decomposition of the principle vector system reveals genuine value tensions
4. Demonstrates that SGPO-style fine-tuning using geometric alignment signals produces more robust alignment than standard RLHF

**Interim deliverable** (3-6 months): A blog post section for "The Shape of Good Behavior" describing the geometric view of alignment in accessible terms, with preliminary plots showing embedding geometry of aligned vs. non-aligned responses.

---

## Open Questions (Research Design)

1. **Which layer?** Constitutional alignment signals may be strongest in early, middle, or late transformer layers. Need empirical investigation.

2. **Which embedding?** Should we use hidden state embeddings, token embeddings, or attention pattern features?

3. **How to define "aligned"?** Binary (aligned/not) or scalar (alignment score)? Constitutional AI uses a scalar critique score — can we use this as supervision?

4. **Cross-model comparison**: How do we align embedding spaces of different models for cross-model analysis? Requires cross-model representational alignment (CKA or similar).

5. **Causality**: Does the alignment direction *cause* aligned behavior, or just correlate? This is the key question for whether geometric fine-tuning would work.

6. **The CAG loop**: Constitutional Alignment Geometry could create a feedback loop: discover alignment geometry → fine-tune toward alignment → re-discover geometry → verify. Is this loop stable or does it create blind spots?
