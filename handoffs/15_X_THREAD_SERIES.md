# Handoff 15 — Build the X/Twitter Thread Series for Shape of Good Behavior

**Date**: 2026-07-28
**From**: SCT thread-series session (Claude + Mike)
**Mission**: Build a `threads/` post series for this repo, identical in
structure and standards to the one in
`topics/structure_of_clear_thinking/threads/` (also on GitHub:
`MikeHLee/structure_of_clear_thinking`). Distribution strategy is X threads +
blog (Ghost, mirrored to Substack) with arXiv anchors — conferences are
deprioritized (organizers unreceptive). Prefer steady publishing of solid
results with caveats over waiting for splashier ones.

## Reference implementation — copy this structure exactly

Study `topics/structure_of_clear_thinking/threads/` first:

```
threads/
├── README.md                       # series plan table, mirroring notes, figure style
└── NN_short_slug/
    ├── thread.md                   # tweet-by-tweet copy + alt text + posting notes
    ├── generate_figures.py         # renders figures FROM RESULT FILES (never hand-enter numbers)
    └── figures/*.png               # committed, self-interpretable
```

Standards (all demonstrated in SCT threads 01 and 03):

1. **Accessibility of understanding is an explicit requirement.** Every
   concept gets a plain-language handle ("ontology" → "rulebook") and, where
   possible, a physical analogy. A reader with zero ML background should
   follow every post.
2. **Posts ≤280 chars** unless marked LONG. ~10 posts per thread: hook with
   the strongest figure → setup → result → stress test → why it matters →
   honest caveats → links.
3. **Figures are self-interpretable**: takeaway sentence as the title,
   explanatory subtitle, direct value labels on every mark, sample
   size/seeds + source script + repo URL in the footer. Every image gets
   written **alt text** in `thread.md`.
4. **Figure style**: invoke the `dataviz` skill before writing chart code.
   Light surface `#fcfcfb`; categorical palette in fixed order blue
   `#2a78d6` → orange `#eb6834` → aqua `#1baf7a` (this order is
   CVD-validated; run the skill's validator if you deviate). Direct labels
   satisfy the aqua contrast warning. 12×6.75in @160dpi.
5. **Numbers come from result files** (`shared/results/*.json`, result .md
   reports), loaded by `generate_figures.py` — never typed into the script.
   Visually inspect every rendered PNG (Read the file) for label collisions
   before committing.
6. **Honesty discipline**: state n, seeds, p-values, effect sizes; say "not
   statistically significant" wherever true; every thread ends with the
   caveats post before the links post.

## Proposed series (adjust as evidence warrants)

| # | Thread | Headline numbers (verify against source before use) | Source |
|---|--------|------------------------------------------------------|--------|
| 01 | **Your preferences have loops — and reward models trip on them** (Hodge-DPO) | Hodge-DPO exploit resistance 0.9999±0.001 vs DPO 0.940±0.013 (+6.3%, d=6.52, p<0.0001); Hodge-KTO 0.9964 vs KTO 0.800 (+24.5%, d=16.47); GRPO already at ceiling 1.0 (30 seeds) | `shared/results/optimizer_comparison_hodge_v3_30seed.json` |
| 02 | **The shape of a lie** (peer-consistency sheaf) | ‖δ¹c‖ detects deception-as-strategy at 7–9B: convincing-game AUC 0.661 (p=2.5e-6), insider-trading ~0.625–0.637; alignment-faking reasoning sign-flips (0.391, cue-robust); **blind to overt/instructed lies — selectivity IS the story** | `constitutional_alignment_geometry/`, Part 4 blog (LIVE on Ghost since 2026-06-04) |
| 03 | **The bug that made our RL look broken** (SGB-005c war story) | RM truncation bug (context silently duplicated across chosen/rejected) stuck RM at chance; fix restored PPO > SFT; Hodge-PPO 82.4% vs 80.4% is a 1-example gap at n=51 — directional, NOT significant, say so | SGB-005/005c writeups, memory `project_sgb004_ppo_holdout_eval` |
| 04 | **When the judge can't keep up with the contestant** (verifier–generator gap) | Lead with saturated hack rate 0.971 vs 0.056 (MW p=1.6e-35); scaling slope ≈ −19 across 3 disjoint blocks; mechanism confirmed OOS (SGB-035→036+3rd block) | Track 4 files, memory `project_verifier_gap_track4` |

Thread 01 should be **fully drafted** (copy + rendered figures); others may
be scaffolded with `{{PLACEHOLDER}}` tokens per the SCT thread-03 pattern if
any number needs re-verification.

## HARD constraints — refuted/retracted claims that must NOT appear

These are documented in project memory and public errata. Threading any of
them would repeat a published correction in the wrong direction:

- **NEVER** use the "SGPO 0% vs PPO 100% hacking on Murky Drone" claim — the
  50-seed re-run **refuted** it: CPO is safest and SGPO's `advantage/sqrt(g)`
  is indistinguishable from plain PPO (p=0.68). Struck from all surfaces.
- Paper 1 Table 1 had a provenance erratum (single-seed SandbaggingEnv
  mislabelled as 5-seed Murky Drone) — corrected publicly. Don't cite the
  old table.
- Verifier-gap: the headline Fisher p=3.96e-42 used a looser turnover
  definition — the corrected noise-aware value is p=6.7e-27. The 6-pt slope
  −14.7 is RETRACTED (use ≈ −19 from the 3-block analysis). The N=1 control
  is tautological — don't present it as a control.
- The peer sheaf is a **selective** detector, not a lie detector — it misses
  overt/instructed lies. Present the blindness as a finding, not a footnote.
- Hodge decomposition separates resolvable from irreducible disagreement; it
  does **not** attribute irreducible mass to causes without extra structure.
- Framing-wide caveat (applies to every thread): safe/good behavior is
  defined **relative to the preference/constraint structure used** — the
  geometric machinery inherits the quality of its inputs. Mirror of SCT's
  "the guarantee is only as good as the rulebook."

Project memory (auto-loaded in ai_research sessions) has the details:
`project_paper_claim_provenance`, `project_verifier_gap_track4`,
`project_peer_sheaf_findings`, `feedback_hodge_attribution_scope`,
`project_sgb004_ppo_holdout_eval`, `feedback_publishing_cadence`.

## Publishing mechanics

- Blog: Ghost CMS — **dev and prod share one instance; publishing is
  immediately public**. Part 4 "Shape of a Lie" is already LIVE (verify with
  `scripts/_list-posts.js`, read-only) — thread 02 links to it rather than
  re-publishing.
- Substack mirror: no publishing API — markdown paste per post, or a
  one-time Ghost RSS import. Titles identical; canonical URL = Ghost.
- End each thread with repo + (when available) arXiv links.

## Workflow expectations

- Work in `/Users/Michaellee/Documents/Runes/ai_research/topics/shape_of_good_behavior`.
- There may be unrelated uncommitted WIP in the repo (e.g.
  `shared/modal_finetune.py`) — commit **only** your `threads/` files and
  README link, never `git add -A`.
- Add a `threads/` line to the repo README's directory structure and a
  Writing/series pointer, as done in the SCT README.
- Commit and push to `origin main` (`MikeHLee/shape_of_good_behavior`) —
  Mike has authorized pushes for the thread-series workstream. If anything
  feels destructive or out of scope, stop and ask.
- Send the rendered figures + thread.md to Mike for review when done; drafts
  only, Mike posts them himself.
