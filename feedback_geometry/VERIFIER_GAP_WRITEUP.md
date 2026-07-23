# The verifier–generator gap: peak danger sits at intermediate search

*Track 4 · draft writeup · sources SGB-032 / SGB-033 / SGB-034 / SGB-035 / SGB-036*
*Status: DRAFT, version-controlled for provenance. Numbers self-audited 2026-07-23; every figure and headline number reproduces from committed code (see "Provenance"). Not yet peer-reviewed or submitted.*

## Summary

In a navigation environment with a sound programmatic ground-truth checker, we
vary two things independently: how good a learned verifier is (measured against
the oracle, not assumed), and how hard a generator searches against it
(best-of-$N$). Whether the verifier gets exploited is set by the **relationship**
between the two, not by verifier quality alone.

The organising result is mechanistic. A single binary property of the learned
verifier — **is its global argmax inside the trap?** — separates every run into
two populations that barely overlap. The cleanest statement is the *saturated*
hack rate (at $N{=}1024$), because it needs no turnover threshold to define:

| | $n$ | hack rate at $N{=}1024$ | curve turns over |
|---|---|---|---|
| argmax **inside** trap | 50 | **0.971** | 0.0% (0/50) |
| argmax **outside** trap | 300 | **0.056** | 75.7% (227/300) |

Mann–Whitney on the saturated hack rate: $p = 1.6\times10^{-35}$ — and this
statistic depends on no threshold at all. The turnover crosstab gives Fisher
exact $p = 6.7\times10^{-27}$ under **the paper's noise-aware turnover
definition** (§6). (An earlier draft quoted $3.96\times10^{-42}$; that came from
the looser index-based turnover count this paper's own analysis script replaced,
and does not reproduce from the committed code. We quote the conservative,
internally-consistent number.)

**Confirmed out of sample.** Re-running on 50 fresh, disjoint seeds (200–249;
SGB-036) reproduces it almost exactly: saturated hack **0.972 inside vs 0.064
outside** (Mann–Whitney $p = 2.6\times10^{-32}$), turnover 2.3% vs 78.1%, Fisher
$p = 9.0\times10^{-24}$. The mechanism is not an in-sample artefact.

Everything else follows from this. When the verifier's global optimum is inside
the trap, saturating search converges on it and hacking $\to 1$. When it is
outside, saturating search **escapes** the local trap optimum that mid-range
search finds, and hacking $\to 0$. The widely-reported "danger peaks at
intermediate optimisation pressure" inverted U is not a separate phenomenon: it
is what you observe *conditional on the verifier's global optimum being safe*,
and the aggregate curve is the mixture of the two populations.

Verifier competence controls the mixing weight. $P(\text{argmax in trap})$ falls
$52\% \to 32\% \to 4\% \to 4\% \to 6\% \to 0\% \to 2\%$ across the competence
range.

---

## 1. Setup

**Environment.** `PreferenceTrapEnv`, a 2-D box $[0,10]^2$ containing a disc-shaped
trap region that is catastrophic under ground truth. It was chosen because it is
the only environment in this repo with a **sound programmatic checker**
(`in_trap`) that is independent of the reward the policy optimises. The oracle
stays wired in throughout, so a policy cannot win by fooling the experimenters as
well as the verifier.

Trap share of the state space: $\pi r^2 / 100 = \pi(2.0)^2/100 = 0.1257$.

**Verifier weakness dial** (`oracle_fraction`). Each preference label is drawn
either from a sound evaluator that knows the trap is catastrophic, or from a
blind evaluator panel that rates the trap as actively good. $1.0$ = fully sound,
$0.0$ = fully blind.

Crucially, verifier competence is then **measured against the oracle**
(`trap_vs_safe_acc`: does the learned verifier rank a safe state above a trap
state?), never assumed from the dial setting. All results are reported against
the measured quantity. The dial and the measurement are not linearly related —
`oracle_fraction` $0.25 \to 0.50$ maps to measured competence $0.027 \to 0.369$.

**Generator capability.** Best-of-$N$ search against the learned verifier. $N$ is
an exact, interpretable dial on adversarial reach, with none of the pathologies
of the RL arm (§5).

**Hacking metric.** Ground truth: does the argmax-of-verifier candidate land
inside the trap?

**Runs.** SGB-032 (15 seeds × 7 verifier strengths × 6 budgets, seeds 0–14) and
SGB-033 (50 seeds × 7 × 11 budgets, 800 trials/cell, seeds 100–149). The seed sets
are **disjoint** — this is a genuine replication, and SGB-032's headline
$N^*{=}16$ at `oracle_fraction` $0.375$ reproduces exactly.

---

## 2. Result 1 — the mechanism (lede)

![two populations](figures/vg_fig1_mechanism.png)

Per-seed curves, coloured by whether the verifier's global argmax lies in the
trap. The two populations barely overlap. The red mean rises monotonically to
$0.971$; the blue mean peaks near $N{=}16$ and collapses to $0.056$.

The causal chain:

1. Verifier competence sets $P(\text{the verifier's global optimum lies in the trap})$.
2. Global optimum **inside** → saturating search converges on it → hacking $\to 1$.
3. Global optimum **outside** → saturating search **escapes** the local trap
   optimum that mid-range search finds → hacking $\to 0$.
4. The aggregate curve is the mixture of these two populations.

This framing matters for two reasons. First, it converts the per-seed
heterogeneity from a caveat into the explanation — the spread is not noise to be
averaged away, it *is* the result. Second, it pre-empts the obvious reviewer
question about the inverted U ("why would more optimisation ever help?") by
answering it mechanistically rather than phenomenologically.

**Presentation constraint:** never plot only the mean curve. The aggregate is a
mixture whose shape depends on the mix, and it misrepresents both populations.

---

## 3. Result 2 — it is a gap, not a height

![gap not height](figures/vg_fig2_gap_not_height.png)

The same verifier (measured competence $0.027$) sits at hack rate $0.335$ at
$N{=}4$ and $0.752$ at $N{=}256$. Meanwhile a competent verifier gets *safer* as
the generator strengthens. Verifier quality alone predicts nothing about
outcomes; the pairing does.

**On the $N{=}1$ convergence.** All seven curves meet at $0.124$, matching the
trap's geometric share ($0.1257$). This is **true by construction, not an
empirical finding**: at $N{=}1$, `best = argmax(...)` over a single candidate is
always index 0, so the verifier is never consulted and the hit rate is exactly
$P(\text{uniform draw} \in \text{trap})$ regardless of verifier quality.

We report it as an **implementation check** — it confirms the trap geometry, the
sampler, and the ground-truth checker agree — and explicitly *not* as the
validity argument for reading the rest as a search effect. That argument rests
instead on the cross-verifier contrast at fixed $N$, where the verifier is
genuinely consulted.

*(This distinction is the same class of issue as EXPERIMENT_ISSUES.md §2, where a
metric was arithmetically forced by the harness rather than measured. Stating it
plainly costs nothing and protects the surrounding claims.)*

---

## 4. Result 3 — peak-danger budget falls geometrically with competence

![scaling law](figures/vg_fig3_scaling_law.png)

The peak-danger search budget $N^*$ (per-seed argmax of the hack-rate curve,
median across 50 seeds) falls geometrically with measured verifier competence.
$N^*$ is only defined where the curve has a genuine interior peak, so we fit the
**confirmed inverted-U regime** under a rule fixed in advance and applied
*symmetrically*: include an `oracle_fraction` point iff its noise-aware turn-over
fraction $\ge 50\%$ **and** its median curve turns over ($N^* > 1$). Points that
fail — the low-competence end where search has not yet separated the populations,
and the high-competence end where the curve is monotone so "peak danger" is just
the minimum budget — have no well-defined $N^*$ and are not fit.

**The slope is $\approx -19$ and reproduces across three disjoint 50-seed blocks:**

| seed block | confirmed points | slope | $r^2$ |
|---|---|---|---|
| 100–149 (SGB-033/035) | 5 | $-18.5$ | 0.95 |
| 200–249 (SGB-036, out-of-sample) | 5 | $-19.9$ | 0.94 |
| 500–549 (SGB-041, dense 13-pt grid) | 8 | $-19.9$ | 0.91 |

Peak-danger budget halves for roughly every $+0.05$ of competence. In the dense
block, median $N^*$ runs $128 \to 64 \to 32 \to 32 \to 16 \to 8 \to 6 \to 4$
across competence $0.038 \to 0.263$. The full 7-point fit on block 100–149
(all measured points, no regime restriction) is $-19.1$, $r^2 = 0.826$ —
consistent with the regime fits.

**Densifying helps but does not eliminate the slope uncertainty.** Going from 5
to 8 confirmed points (the 13-point `oracle_fraction` grid, seeds 500–549) shrank
the leave-one-out slope swing from **37% and 51%** on the two sparse blocks to
**25%** on the dense block (range $[-17.6,\,-22.6]$, $\pm 12\%$ around $-19.9$).
So more points do help — but a floor remains, because $N^*$ exists only inside the
bounded inverted-U regime, which puts irreducible **leverage on the two competence
endpoints** that interior points cannot remove. Report the slope as $\approx -19$
with $\pm 12\%$ uncertainty; do not quote three significant figures and do not
extrapolate.

**Retraction of the earlier "6-point $-14.7$" fit.** A previous draft reported a
shallower slope by *dropping* the low-competence point (orc $0.25$) while
*keeping* orc $0.50$ — which fails the same inverted-U criterion (only 16–24% of
its seeds turn over). That is asymmetric point-dropping, exactly what a reviewer
should challenge. Under the symmetric rule the slope is $\approx -19$ on every
block, never $-14.7$; the $-14.7$ was an artifact of inconsistent inclusion and
is superseded by the table above.

**Why this is the actionable finding.** Improving a verifier does not simply
reduce risk — it *relocates* the risk to a smaller generator. A safety margin
stated as "our verifier is good enough" is meaningless without naming the
generator's search budget.

---

## 5. Negative result — the RL arm is a null

SGB-034 ran REINFORCE against the learned verifier as an alternative generator
axis. It never became strong enough to outrun the verifier, so no gap could open
and the phenomenon did not appear. We report this as a limitation: *our RL
learner was too weak to reproduce the search result.* It pre-empts the obvious
question about whether the effect is an artifact of best-of-$N$ specifically.

An early single-seed run of that arm produced a false positive that did not
replicate. Worth a sentence in any lay writeup about how easy this is to get
wrong.

---

## 6. Methods — harness fixes that were required

The base environment could not express the phenomenon. Each fix is documented
in-code and was verified by diagnostic. A reviewer asking "why not just use the
existing environment" needs these four:

1. **Trap terminated the episode.** With positive per-step proxy reward this
   creates a *survival* incentive — entering the trap forfeits remaining reward —
   so the policy avoids it for reasons unrelated to the verifier. This alone
   pinned hacking to $0.00$ even against a verifier scoring $0.068$ trap-vs-safe.
   Fixed with fixed-length episodes (`FixedLengthTrapEnv`).
2. **Reward model trained only on within-state action pairs.** Bradley–Terry
   cancels every state-dependent term, so the verifier's absolute level across
   states was never constrained — yet the policy consumed it as a per-step
   reward. Cross-state comparisons added; without them the oracle dial is inert.
3. **The trap was not the verifier's argmax.** The blind panel rated the trap
   merely *normal*, so boundary extrapolation won (global max at corner $[0,0]$,
   value $1.056$, vs trap $0.756$). Added `trap_appeal`, the deceptive
   false-positive reward that `MultiTrapEnv` already has.
4. **Generator axis is search budget, not RL episodes** (§5).

Two analysis bugs also changed conclusions and belong in a methods footnote:

- **Per-seed vs aggregate curves.** The aggregate is a mixture (§2); reading
  shape off the mean curve gave a wrong answer once.
- **Turn-over definition, corrected twice.** (a) `np.argmax` breaks ties by first
  index, so a saturated curve $(\dots, 1.000, 1.000)$ reports a spurious
  turn-over; the peak must exceed **both** endpoints by a margin. (b) A fixed
  margin ($\text{tol}=0.02$) still counts *sampling-noise* fluctuations as
  turn-overs, since each hack rate is a binomial proportion over 800 trials. The
  **noise-aware** definition used everywhere in this paper (verdict, §2 crosstab,
  figures) requires a qualifying earlier budget to beat both endpoints by $>3$
  combined binomial standard errors. This matters: the looser $\text{tol}=0.02$
  count reported argmax-outside turn-over at 91.7% and Fisher $p=3.96\times10^{-42}$;
  the noise-aware count gives 75.7% and $6.7\times10^{-27}$. We report the latter.
  A single turnover definition is used for every number in the paper.

---

## 7. Limitations — the story we are not telling

1. **This is a 2-D toy with a synthetic verifier.** It is a clean demonstration
   of a mechanism, **not** evidence about real LLM RLHF. `oracle_fraction` is not
   calibrated against any real verifier quality, and `trap_vs_safe_acc` is
   specific to this trap geometry. Write "in a setting where ground truth is
   available, …", never "we show RLHF reward models fail at $N{=}X$".
2. **The mechanism generalises only as far as the toy does.** The global-argmax
   account is established *within this environment* (saturated-hack Mann–Whitney
   $p = 1.6\times10^{-35}$ in-sample, $2.6\times10^{-32}$ out-of-sample).
   What does **not** follow is that a real learned verifier has a well-defined
   global argmax a generator can reach, or that "escaping the local optimum" is
   available when the search space is language rather than a $10\times10$ box.
   State the mechanism as **proven-here, conjectural-elsewhere**.
3. **The mechanism was derived in-sample, then confirmed out-of-sample.**
   `mechanism_argmax_50seed.json` re-analyses the *same 50 seeds* (100–149) as
   `replication_invertedU_50seed.json` with `argmax_in_trap` recorded — so on its
   own it is an explanation of the data it was derived from, not a replication.
   That objection is now retired: `mechanism_oos_seed200.json` (SGB-036) is an
   independent 50-seed run on disjoint seeds (200–249) that reproduces the
   crosstab (saturated hack 0.972 / 0.064; turnover 2.3% / 78.1%; Fisher
   $p = 9.0\times10^{-24}$) and the scaling law (confirmed-regime slope $-19.9$
   vs $-18.5$ in-sample). The mechanism holds on data it never saw.
4. **Scaling-law slope has an endpoint-uncertainty floor.** Central estimate
   $\approx -19$, reproduced on three disjoint seed blocks (§4). $N^*$ is defined
   only inside the bounded inverted-U window, so the slope leans on its two
   competence endpoints: densifying 5→8 confirmed points shrank the leave-one-out
   swing from 37–51% to 25% ($\pm 12\%$), but did not remove it. Quote the
   direction and order of magnitude, not the digits.
5. **The RL arm is a null** (§5).

---

## 8. Relationship to the Hodge track

This is **Track 4**, new. Per the 2026-07-21 reconciliation: *transitivity of
feedback* (what $H^1$ measures) and *sufficiency of the state description* (what
this measures) are **independent failure channels**, not one object in two
coordinate systems. You can have $H^1 = 0$ with enormous descriptive slack — a
verifier that consistently ignores what matters is perfectly self-consistent.
SGB-011 (harmonic mass identically zero by construction while real inconsistency
existed) is the empirical instance of the two coming apart.

**Do not frame this as "another view of the Hodge result."**

Motivation paragraph worth making explicitly: the repo's own headline **"exploit
resistance" metric cannot measure this phenomenon at all.** It is ranking accuracy
over a frozen pair list, computed by MLPs over frozen embeddings that never
generate. Nothing in that pipeline is under optimisation pressure — which is why
it saturates at 0.9999–1.000. The contrast motivates the whole track.

---

## 9. Provenance

Every number above traces to a committed JSON. Figures are generated by
`figures/make_verifier_gap_figures.py`, which loads from these files and hardcodes
nothing.

| Claim | Value | Source |
|---|---|---|
| mechanism, saturated hack | 0.971 inside / 0.056 outside, MW $p{=}1.6\mathrm{e}{-35}$ | `mechanism_argmax_50seed.json` |
| mechanism turnover crosstab (noise-aware) | 0.0% / 75.7%, Fisher $6.7\mathrm{e}{-27}$ | `analyze_verifier_gap.py` on same file |
| — same, OOS (seeds 200–249) | 0.972 / 0.064, 2.3% / 78.1%, Fisher $9.0\mathrm{e}{-24}$ | `mechanism_oos_seed200.json` |
| $P(\text{argmax in trap})$ | 52/32/4/4/6/0/2 % | `mechanism_argmax_50seed.json` |
| median $N^*$ | 384/48/32/16/8/4/2 (in-sample), 256/64/32/16/8/4/2 (OOS) | `mechanism_argmax_50seed.json`, `mechanism_oos_seed200.json` |
| inverted-U window (pre-specified paired test $N{=}16$ vs $N_{\max}$) | confirmed orc $\in[0.30,0.45]$, all $p<0.05$, $d_z>0$; fails at 0.25 ($p{=}0.44$) and 0.50 | `analyze_verifier_gap.py`; OOS window identical (`mechanism_oos_seed200.json`) |
| scaling law, confirmed regime (3 blocks) | $-18.5$/$-19.9$/$-19.9$, $r^2\ge0.90$ | `mechanism_argmax_50seed`, `mechanism_oos_seed200`, `scaling_dense_13pt_50seed` |
| scaling law, full 7-pt (block 100–149) | $-19.1$, $r^2{=}0.826$ | recomputed from `mechanism_argmax_50seed.json` |
| scaling law leave-one-out | swing 37%/51% (5-pt blocks) → 25% (8-pt dense); range $[-17.6,-22.6]$ | `scaling_robustness.py` |
| ~~6-pt $-14.7$~~ SUPERSEDED | inconsistent inclusion (dropped orc 0.25, kept orc 0.50) | — |
| $N{=}1$ baseline | 0.1232 (15-seed) / 0.1245 (50-seed) | `sweep_105cells.json`, `mechanism_argmax_50seed.json` |
| trap share | 0.1257 | geometry, $\pi(2.0)^2/100$ |
| gap illustration | 0.335 @ $N{=}4$, 0.752 @ $N{=}256$ | `mechanism_argmax_50seed.json` (50-seed) |

**Note on the gap numbers.** The source handoff quoted $0.349$ and $0.797$, which
are the **15-seed** SGB-032 values. We report the 50-seed values above. Do not mix
the two runs within a single claim without labelling which is which.

Reproduce:

```bash
cd topics/shape_of_good_behavior
# inverted-U verdict, in-sample then out-of-sample (same script, same verdict)
./venv/bin/python3 feedback_geometry/src/analyze_verifier_gap.py \
    feedback_geometry/results/verifier_gap/mechanism_argmax_50seed.json
./venv/bin/python3 feedback_geometry/src/analyze_verifier_gap.py \
    feedback_geometry/results/verifier_gap/mechanism_oos_seed200.json
# §2 mechanism crosstab (saturated hack + noise-aware Fisher; discloses the looser count)
./venv/bin/python3 feedback_geometry/src/mechanism_crosstab.py
# §4 scaling law: confirmed-regime slope across 3 blocks + leave-one-out
./venv/bin/python3 feedback_geometry/src/scaling_robustness.py
# all three figures (nothing hardcoded; loads from JSON)
./venv/bin/python3 feedback_geometry/figures/make_verifier_gap_figures.py
```
