# X/Twitter Thread Series

Public-communication track for the Shape of Good Behavior results.
Conferences are deprioritized as a venue; distribution is **X threads +
blog posts (Ghost, mirrored to Substack)**, with arXiv for citable anchors.
Structure and standards mirror
[`topics/structure_of_clear_thinking/threads/`](https://github.com/MikeHLee/structure_of_clear_thinking).

Each thread directory contains:

- `thread.md` — the tweet-by-tweet copy, with alt text for every image and
  posting notes
- `generate_figures.py` — reproducible figure generation (numbers sourced
  from experiment result files, never hand-entered twice)
- `figures/` — the rendered PNGs, designed to be self-interpretable
  (takeaway in the title, every mark directly labeled)

## Planned series

| # | Thread | Source result | Status |
|---|--------|---------------|--------|
| 01 | Your preferences have loops — and reward models trip on them | `shared/results/optimizer_comparison_hodge_v3_30seed.json` (+ caveats in `shared/results/README.md`) | **✖ WITHDRAWN (2026-09-18)** — held-out re-test finds no Hodge benefit. **Replacement has a spine (2026-09-28): "we built cycle-aware preference optimization and the cycles weren't there."** b₁ = 0 on the real label graph (`shared/results/README.md` § 6a) — the loops in the title were kNN artefacts. Retitle; the premise is the retraction |
| 02 | The shape of a lie (peer-consistency sheaf) | `shared/results/peer_sheaf_e6_modal_*.json` (SGB-021 7–9B panel); pairs with the LIVE Part 4 blog post | **Draft** — pending blog URL + final read |
| 03 | The bug that made our RL look broken (SGB-005c war story) | `shared/results/finetune/sgb004_exploit_resistance_holdout_v2.json`, `sgb005b_rm_scores{,_hodge}.json`, `feedback_geometry/WRITEUP_OPEN_MODEL_EXPLOIT_RESISTANCE.md` | **✖ WITHDRAWN (2026-09-19)** — retrain shows the metric is invalid; PPO drifts to RM-over-rewarded incoherent text |
| 04 | When the judge can't keep up with the contestant (verifier–generator gap) | `feedback_geometry/results/verifier_gap/*.json`, `feedback_geometry/VERIFIER_GAP_WRITEUP.md` | **Draft ready** — numbers recomputed from raw rows 2026-07-31; only writeup/arXiv links pending |
| 05 | We built cycle-aware preference optimization. The cycles weren't there. | `shared/results/README.md` §6a (b₁=0), `optimizer_comparison_heldout_v1.json`, `docs/DESIGN_NOTE_RELATIVE_REWARDS.md` | **Draft (2026-09-28)** — **replaces thread 01**, withdrawing its premise not just its numbers. Needs Mike's review + Fig 1 (label graph) before posting; do not post without Fig 1 |

## Claims discipline

Refuted/retracted claims that must never appear in any thread (documented in
project memory and public errata):

- The "SGPO 0% vs PPO 100% hacking on Murky Drone" claim — refuted by the
  50-seed re-run (CPO is safest; SGPO ≈ plain PPO, p=0.68).
- Paper 1 Table 1's original provenance (single-seed SandbaggingEnv
  mislabelled as 5-seed Murky Drone) — corrected publicly; cite the erratum.
- Verifier-gap: Fisher p=3.96e-42 (looser turnover definition — use the
  noise-aware p=6.7e-27); the 6-pt slope −14.7 (retracted — use ≈ −19 from
  the 3-block analysis); the N=1 control (tautological).
- The peer sheaf described as a general lie detector — it is *selective*
  (misses overt/instructed lies); the blindness is a finding, not a footnote.
- Hodge decomposition attributing irreducible disagreement to causes — it
  separates resolvable from irreducible; attribution needs extra structure.
- **Any h₁ / harmonic-energy / curl-mass number from a kNN-augmented graph,
  presented as a property of human feedback (added 2026-09-28).** The label
  graph of `counterfactual_pairs.json` has **b₁ = E − V + C = 500 − 998 + 498
  = 0** — a forest. On a forest, curl and harmonic energy are exactly zero for
  *every possible* edge flow, so no labelling of that data could have shown
  intransitivity. The decomposed graph was 90.9% synthetic kNN edges, so every
  cycle in it passes through a similarity artefact, and `marginal_h1 ≈ 2.22`
  measures the kNN wiring, not preferences. **Rule: whenever an h₁ or curl
  figure is cited, report the b₁ of the labelled subgraph alongside it.** Never
  let graph augmentation supply the topology being claimed as a finding. Also
  never claim curl from a single scalar grader at any density — one scalar gives
  flow grad s, which is exactly curl-free
  (`hodge_preference_optimizers.py:8-13`). See `shared/results/README.md` § 6a.
- Any LM-level exploit-resistance percentage from SGB-004/005c/006 (base
  68.6%/SFT 80.4%/PPO 80.4%/Hodge-PPO 82.4% at 1.5B; the 7B numbers) — a
  prompt right-truncation bug (SGB-044, found 2026-07-31) affects 100% of
  eval prompts and 98.5% of PPO training queries at both scales. The 2026-09-19
  retrain WITHDRAWS them: "RM score > 0" does not measure exploit resistance, and
  both PPO policies drift toward incoherent text the RMs over-reward. Never claim an LM-level Hodge-PPO benefit.
- Any claim that HodgePO (Hodge-DPO / Hodge-KTO) improves preference
  optimization. The 0.9999 / 0.9964 figures were in-sample, came from a run
  with misaligned targets, and are reproduced by a margin control; held-out,
  there is no benefit (`shared/results/README.md`, found 2026-09-18). Never
  cite "28% of HH-RLHF is cyclic": no result file supports it.

Framing-wide caveat carried by every thread: safe/good behavior is defined
**relative to the preference/constraint structure used** — the geometric
machinery inherits the quality of its inputs.

Citations and X tags for threads 01–02 come from the verified literature
search in `ai_research/.swarm/related_work_SOGB.md` (2026-07-30); the
Track-4 thread's equivalent is `ai_research/.swarm/related_work_and_collaborators.md`.
Tag only handles those files list as verified — never guess a handle.

## Blog mirroring

- Canonical posts publish to the oasis Ghost CMS. **Note**: dev and prod
  share one Ghost instance — publishing surfaces the post publicly
  immediately; there is no staging.
- Part 4 "Shape of a Lie" is already LIVE (since 2026-06-04) — thread 02
  links to it rather than re-publishing. Get the URL with the read-only
  `scripts/_list-posts.js` in the oasis blog repo.
- **Substack mirror**: Substack has no publishing API — mirror by pasting the
  markdown (or importing the Ghost RSS feed once at setup). Keep titles
  identical; canonical URL points at the Ghost post.
- Each thread ends with repo + (when available) arXiv links.

## Figure style

Figures follow the dataviz conventions shared with the SCT series: light
surface `#fcfcfb`, categorical palette blue `#2a78d6` / orange `#eb6834` /
aqua `#1baf7a` (CVD-validated in this order — run the dataviz skill's
validator if you deviate), direct value labels on every mark, takeaway
stated in the title, sample sizes/seeds + source file + repo URL in the
footer. 12×6.75 in @160 dpi. Regenerate with each thread's
`generate_figures.py` (uses `./venv/bin/python3`; numbers load from the
result JSONs, never hand-entered).
