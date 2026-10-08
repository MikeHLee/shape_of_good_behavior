# Design Note — Relative Reward Functions over Constitutional AI + Human Preferences

**Status:** design note, not a result. No claim here is measured on real preference data.
**Date:** 2026-09-28
**Author:** Kolmogorov
**Scripts:** `cai_geometry/{mixture_curl,nonconvex_frontier,curl_mass_real,relative_reward_capacity,curl_vs_intransitivity,ranking_rm_can_cycle}.py`
**Depends on:** `shared/results/README.md` § 6a (b₁ = 0 finding)

---

## 0. Why this note exists

Track 1 built cycle-aware preference optimization. The cycles were not in the data
(b₁ = 0 on the real label graph; 90.9% of the decomposed graph was synthetic kNN
edges). The held-out null was therefore **forced by collection design**, not caused
by weak tuning.

This note specifies what data collection would have to look like for the question to
be answerable at all, and — more importantly — **what the correct metric is**, because
the obvious one (curl mass) turns out to be badly biased.

---

## 1. The headline correction: curl ≠ intransitivity

I need to retract a claim I made in conversation on 2026-09-28. I said that scalar,
binary, ordinal and discrete absolute rewards are "all per-item, hence all pure
gradient, hence all curl-free — the discreteness is irrelevant." **The second half is
wrong.** Measured (`curl_vs_intransitivity.py`, K₇ complete graph):

| Grader | curl % | 3-cycles | Transitive? |
|---|---|---|---|
| (A) absolute scalar, continuous | **0.000%** | 0 | yes |
| (B) absolute scalar → **binary** label | **15.873%** | **0** | **yes** |
| (C) relative ranking (salient-dimension) | **21.574%** | **0** | **yes** |
| (D) majority vote over 3 principles | **100.000%** | **1** | **no** |

Two things break the tidy story:

**(a) Binarizing a perfectly transitive scalar manufactures curl.** Binary labels
emit a constant logit magnitude (±2.197) on every edge regardless of the true gap. A
gradient flow must satisfy the cocycle condition — flows add along paths — and
constant-magnitude flows cannot: along R0→R3→R6 the path sum is 0.000 while the
direct edge is +2.197. So the curl is a **unit-scale artefact of thresholding**. The
underlying order is fully consistent (R0>R3, R3>R6, R0>R6). Per-item-ness is *not*
sufficient for curl-freeness; per-item-ness **plus a faithful cardinal scale** is.

**(b) Curl > 0 does not imply a single cycle exists.** Graders (B) and (C) both show
double-digit curl with **zero** 3-cycles. They are transitive relations.

**Consequence: curl mass is a biased detector of value tension.** It fires on
measurement coarseness. Had we shipped curl % as the platform's headline metric, it
would have scored two perfectly transitive graders as carrying 16–22% "value
tension." That is the same class of error as the kNN artefact — reading a property of
our instrument as a property of the world.

### Metric rule for the platform

> **Primary screen: directed cycle count** (with a permutation null). It is ordinal,
> so any strictly monotone relabelling of the flow leaves it unchanged — immune to
> thresholding and scale artefacts.
>
> **Secondary: curl mass**, reported only once cycles are confirmed, as a magnitude.
>
> **Never** report curl without the cycle count beside it.

---

## 2. Mike's ranking-RM claim: confirmed, with the mechanism identified

Mike's claim (19:40): a pretrained constitutional *scalar* RM cannot reproduce
intransitivity, but a constitutional *ranking* RM can. **Confirmed** — and it is
worth being precise about why, because my first attempt failed to show it.

My grader (C) was relative and gave 21.6% curl but **zero** cycles. That is not a
counterexample to Mike; it was a badly chosen rule. A threshold-free rule that is a
monotone function of a fixed linear combination stays transitive.

The mechanism that actually produces cycles is a **just-noticeable-difference
threshold** — Tversky's (1969) lexicographic semiorder / additive-difference model.
Decide on principle 1 if the gap exceeds ε; otherwise fall through to principle 2
(`ranking_rm_can_cycle.py`):

```
R0 (help 0.0, harm 2.0)   R1 (help 1.0, harm 1.0)   R2 (help 2.0, harm 0.0)   eps=1.5
  R0 vs R1: |d_help|=1.0 <= eps -> harmlessness decides -> R0 > R1
  R1 vs R2: |d_help|=1.0 <= eps -> harmlessness decides -> R1 > R2
  R0 vs R2: |d_help|=2.0 >  eps -> helpfulness  decides -> R2 > R0
  => R0 > R1 > R2 > R0.  A real cycle from ONE deterministic rule.
```

Cycle rate over 200 random 4-item sets:

| Pairwise rule | sets containing a 3-cycle |
|---|---|
| salient-dimension (no threshold) | **0 / 200** |
| lexicographic semiorder, ε = 0.3 | 6 / 200 |
| lexicographic semiorder, ε = 0.8 | 17 / 200 |
| lexicographic semiorder, ε = 1.5 | **30 / 200** |

**So relativity is necessary but not sufficient.** The sufficient ingredient is a
*discontinuity in which principle dominates* — the decision rule changing identity
depending on the pair. Small gaps are treated as ties and overridden by a secondary
principle, but small gaps **chain** into a large one. That is exactly the structure
"genuine value tension" should mean, and it is a well-documented feature of human
choice, not an artefact.

This broadens the design space relative to what I said earlier: the platform does not
strictly need human voters to *generate* intransitive data. But see § 4 — it needs
them to make the data *mean* anything.

---

## 3. What each reward class can and cannot represent

On K_n the flow space has n(n−1)/2 dimensions; gradients occupy n−1. At n=7 that is
6 of 21, so **71.4% of flow space is unreachable by any faithful continuous scalar
reward** (`relative_reward_capacity.py`).

| Class | Reaches | Can encode cycles? | Notes |
|---|---|---|---|
| Continuous scalar, per-item | gradient subspace only (n−1 dims) | **No** | flow = grad s, exactly curl-free (residual 4.7e-33) |
| Binary / ordinal / discrete, per-item | leaves the gradient subspace, but only via **scale distortion** | **No** | curl > 0 yet 0 cycles — artefact, not signal |
| Relative, monotone in a fixed combination | non-gradient components reachable | **No** | 0/200 cycle rate |
| Relative, **threshold / context-dependent** | full antisymmetric space | **Yes** | semiorder; 30/200 at ε=1.5 |
| Vote/aggregation over ≥3 graders | full antisymmetric space | **Yes** | Condorcet; curl 100% on K₃ |

The load-bearing distinction is **not** relative-vs-absolute and **not** cardinal
resolution. It is whether the decision rule can *change which principle dominates as
a function of the pair*. Everything else stays ordinally transitive.

*Caveat on the "model class" claim.* A scalar RM plus a Hodge penalty is still
hopeless — the class projects curl out before the loss sees it
(`hodge_preference_optimizers.py:8-13`). Retaining structure needs an antisymmetric
f(a,b) that is not a difference of per-item scores: a general preference model
(arXiv 2410.02197), skew-symmetric, where curl capacity is rank beyond the gradient
part. That part of my earlier advice stands.

---

## 4. Three sources of cycles — and a hard rule about substitution

Ranked by cost, ascending:

1. **Synthetic semiorder ranking RM.** One model, zero humans. Cheapest. But the
   cycles are a property of *our rule*, so measuring them measures a design choice.
2. **Majority vote over ≥3 principle-conditioned graders.** Cycles are a fact about
   the principle set. Requires per-principle labels logged per comparison.
3. **Human voters aggregated by majority.** Cycles are a fact about people. Most
   expensive, most meaningful, and the only one that answers the original question.

> **Rule: (1) must never be cited as evidence for (3).** That substitution is the kNN
> mistake in a new costume — generating the structure we claim to discover. Use (1)
> strictly as a **positive control**: a pipeline that cannot recover known-planted
> cycles from (1) is broken and must not be trusted on (2) or (3).

This rule is the single most important line in this note. The previous failure was
not a math error; it was letting a construction supply the phenomenon.

---

## 5. Collection design: b₁ per prompt block

Cycles are **local to a prompt**. With k responses per prompt, all pairs compared:
b₁ = (k−1)(k−2)/2.

| responses/prompt | pairs | b₁ | pairs per b₁ |
|---|---|---|---|
| 2 (**current HH-RLHF**) | 1 | **0** | ∞ — cycles impossible |
| 3 | 3 | 1 | 3.00 |
| 4 | 6 | 3 | 2.00 |
| 5 | 10 | **6** | 1.67 |
| 8 | 28 | 21 | 1.33 |
| 10 | 45 | 36 | 1.25 |

**Full global n² cross-pairing is not needed.** Budget is O(k²) in the *block* size,
not in corpus size. k = 5 is the sweet spot: b₁ = 6 per prompt for 10 comparisons.
k = 3 is the minimum that admits any cycle at all.

---

## 6. Two confounds that will otherwise reproduce "28% harmonic" as noise

**(a) Position/order bias — fixable exactly.** A judge mildly preferring whichever
answer is shown first, measured one-direction-only, is not antisymmetric. On a
grader with **zero** true curl, a constant first-position bias produced **12.418%**
spurious curl. Measuring both orders and antisymmetrizing, f ← ½(f(a,b) − f(b,a)),
removed it to **0.000%**. So: **randomize presentation order and always collect both
directions.** Non-negotiable, and cheap.

**(b) Finite votes per edge — CORRECTED 2026-09-28 20:00.** The first version of this
section claimed that even 400 votes/edge leaves a noise floor over half the signal. **That
was wrong.** The grader used, (A) at w = 0.5, gives nearly identical scores to all seven
items (2.0, 2.1, 2.1, 2.05, 2.05, 2.0, 1.9). Its true gradient is about zero, so the curl
*ratio* was noise divided by noise, which tends to the pure-noise share b₁/E = 15/21 =
71.4%. The table below measured a flat grader, not a general noise floor. It is kept only
as a record:

| votes/edge | median spurious curl % | p90 |
|---|---|---|
| 5 | 66.5 | 86.6 |
| 10 | 72.2 | 87.5 |
| 25 | 59.8 | 75.6 |
| 50 | 47.7 | 65.4 |
| 100 | 36.3 | 55.1 |
| 200 | 26.3 | 37.1 |
| 400 | **13.4** | 24.4 |

Recheck (`cai_geometry/noise_floor_recheck.py`, 60 draws per cell), zero-curl truth:

| votes/edge | flat truth: share % | flat: abs curl | graded truth: share % | graded: abs curl |
|---|---|---|---|---|
| 10 | 64.8 | 5.750 | **9.4** | 10.418 |
| 25 | 58.5 | 2.069 | **5.5** | 8.039 |
| 100 | 40.0 | 0.647 | **1.6** | 3.012 |
| 400 | 15.0 | 0.147 | **0.7** | 1.330 |

(graded truth = scores 0, 0.5, …, 3.0)

What this shows:
- Absolute spurious curl falls roughly as 1/V, as it should.
- The **ratio** depends mostly on **how close the items are**, not only on V. With
  well-separated items the floor is 1.6% at 100 votes. With near-tied items it is 40%.
  So there is no single noise-floor number. The null has to be computed **per dataset**,
  from that dataset's own margins.
- The earlier comparison was also invalid in a second way: it set the noise of grader (A)
  against the signal of a *different* grader (C).

**The cycle count is not noise-immune either.** It ignores monotone rescaling, but vote
noise flips near-tied edges, and flipped edges make spurious 3-cycles. That happens most
on exactly the near-tied items where the ratio also fails. Both metrics need a
permutation or parametric-bootstrap null, computed from the observed margins. **Do not
ship a curl number or a cycle count without its null.** Which of the two metrics is more
robust at a fixed budget has **not been tested**. It is step 2 of § 7.

---

## 7. Proposed framing and next steps

Mike's proposed framing — **"Relativistic Reward Functions over Constitutional AI +
Human Preferences"** — is the right umbrella, with one amendment: the operative
variable is not relativity, it is **context-dependent principle dominance**. A
relative reward that is monotone in a fixed principle blend is as transitive as a
scalar. Suggested precise subtitle: *context-dependent pairwise rewards, and what
scalar/binary/ordinal rewards provably cannot represent.*

Ordered next steps:

1. **Ranking/curation UI** collecting k = 5 responses per prompt, all 10 pairs, both
   presentation orders, with the **driving principle logged per judgment**.
2. **Positive-control harness first.** Plant semiorder cycles, verify the pipeline
   recovers them, verify the permutation null is calibrated. Ship nothing before this
   passes.
3. **Cycle-count-primary measurement** on (2) then (3) from § 4.
4. **Ranking RM** as skew-symmetric f(a,b), reporting rank beyond gradient.
5. **Research note** on representational limits (§ 3) — publishable independently of
   whether Hodge ever pays off, because it is a theorem plus clean numerics.

**Cheapest hedge, worth stating plainly:** logging which principle drove each
judgment, with multiple judges per pair, is hidden-context / distributional preference
learning (arXiv 2312.08358). That has results independent of our geometry, so the
collection investment is not a bet on Hodge being right.

## 8. Falsifiers

- If human voters at k = 5 show cycle counts indistinguishable from the permutation
  null, there is no value tension to encode at this granularity. Publish that.
- If cycle counts are significant but a plain BT model matches a skew-symmetric model
  held-out, the structure exists but is not decision-relevant. Publish that too.
- Current odds, unchanged where evidence has not moved: P(cycles findable with a
  threshold/vote grader) ≈ 0.8; P(cycles present in *human* data at k = 5) ≈ 0.5;
  **P(a cycle-aware method beats a strong scalar baseline held-out) ≈ 0.15** — still
  the weakest link, and note that § 1 of this note *lowered* my confidence in the
  metric we would use to detect it.
