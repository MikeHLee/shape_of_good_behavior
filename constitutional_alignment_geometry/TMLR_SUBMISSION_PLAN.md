# TMLR Submission Plan — Peer-Consistency Deception Divergence (Track 3)

*Status: PLAN (not yet drafted). Owner decision pending. Target: TMLR, rolling.*

> Tackle this as a focused session of its own — drafting the full paper here
> would blow context. Recommended: compress/clear context first, then run a
> dedicated "TMLR draft" pass using this plan + the vendored repo as inputs.

## Why TMLR

- **Rolling**, no deadline → fits the steady-publishing cadence; no rush vs NeSy.
- **Claims-focused, no novelty bar** → rewards exactly what this work has:
  careful, controlled empirical claims with stated scope, not a SOTA chase.
- Citable as a journal; complements (doesn't compete with) a NeurIPS-workshop
  short version of the same result.

## Central claim (scoped)

Cross-model embedding divergence δ¹c on a 3-model 7–9B panel is a **selective
detector of deception-as-strategy** (persuasion, agentic scheming) — and is
appropriately *blind* to overt/instructed lies and templated refusals. The
selectivity (the NS controls) is the contribution, not a single AUC.

## Result table (already in hand)

| Construct | AUC | p | Source JSON |
|---|---|---|---|
| Convincing-game (persuasion) | 0.661 ± 0.011 | 2.5e−6 | `peer_sheaf_e6_modal_subsample_variance.json` |
| Insider-trading (agentic) | 0.625 ± 0.012 | 8e−8 | `peer_sheaf_e6_modal_subsample_variance.json` |
| Insider-trading ACTION | 0.640 ± 0.021 | 3.8e−6 | `peer_sheaf_e6_modal_insider-trading-action.json` |
| Alignment-faking reasoning (sign-flipped) | 0.384 / 0.391 (redacted) | <1e−4 | `..._alignment-faking-reasoning{,-redacted}.json` |
| Instructed-deception (control, NS) | 0.528 | 0.32 | `..._instructed-deception.json` |
| Gender-secret (control, NS) | 0.442 | 0.91 | `..._gender-secret.json` |

## What's already strong enough (minimum-viable)

- Length-matched (confound controlled) ✔
- Subsample CIs (stability) ✔
- Action-vs-reasoning split (signal localization) ✔
- Cue-redaction robustness (lexical-cue rebuttal) ✔
- Two NS controls (selectivity) ✔

## Gaps to close before submission (strengthen generality)

- **SGB-024** (directional pattern per-edge/per-model at scale) — cheap, uses
  existing data; pre-empts "is the signal an artifact of one model?"
- **SGB-026** (second intentional-deception dataset, e.g. DeceptionBench) —
  tests generalization beyond LIARS'-BENCH. The main reviewer ask to expect.

## Paper skeleton

1. Intro — scalar honesty probes miss deception-as-strategy; geometry of
   cross-model agreement as a complementary signal.
2. Method — peer-consistency sheaf, restriction maps, δ¹c (cite `peer_sheaf.py`,
   `peer_hodge.py`).
3. Panel & data — 7–9B cross-org panel; LIARS'-BENCH + insider-trading +
   alignment-faking; length-matching protocol.
4. Results — the table; selectivity via NS controls; the alignment-faking
   sign-flip (deceptive reasoning is *more* peer-consistent → RLHF-manifold
   interpretation).
5. Analysis — action-vs-reasoning localization; cue-redaction robustness;
   (SGB-024) directional pattern.
6. Threats to validity — panel size (n=3), AUC magnitudes are modest-but-real,
   construct validity of "deception", embedding-layer choice.
7. Related — honesty probes, representation-reading, sheaf/Hodge in ML.
8. Repro — point at the vendored public repo.

## Reproducibility appendix

Vendored at `alignment_research/peer_consistency_geometry/` (now public):
`src/{peer_sheaf,peer_hodge}.py`, `experiments/`, `modal/embed_panel.py`,
`results/` (19 JSONs), `figures/make_figures.py`. Now installs via the new
`requirements.txt`.

## Threats to pre-empt in the writing

- "n=3 panel" → frame as a *minimal* cross-org panel; note the small-LM E1 and
  7–9B E6 agree directionally; flag panel-scaling as future work.
- "AUC ~0.62–0.66 is modest" → the claim is *selective detection*, not a
  deployable classifier; the NS controls + p-values carry the argument.

## Timeline (suggested)

1. (now) NeSy upload + repo/blog go-live — done/in-flight separately.
2. Decide: submit minimum-viable now, or wait on SGB-024/026. *Recommendation:
   run SGB-024 (cheap) → submit; treat SGB-026 as the revision response.*
3. Dedicated drafting session (fresh context) → arXiv + TMLR via OpenReview.

## Checklist

- [ ] Decide minimum-viable-now vs wait-for-SGB-026
- [ ] (optional) Run SGB-024 directional analysis
- [ ] Draft §1–8 (fresh-context session)
- [ ] Regenerate figures via `make_figures.py` against committed result JSONs
- [ ] arXiv preprint
- [ ] TMLR OpenReview submission
