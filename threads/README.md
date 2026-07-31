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
| 01 | Your preferences have loops — and reward models trip on them | `shared/results/optimizer_comparison_hodge_v3_30seed.json` (+ caveats in `shared/results/README.md`) | **Draft ready** |
| 02 | The shape of a lie (peer-consistency sheaf) | `shared/results/peer_sheaf_e6_modal_*.json` (SGB-021 7–9B panel); pairs with the LIVE Part 4 blog post | **Draft** — pending blog URL + final read |
| 03 | The bug that made our RL look broken (SGB-005c war story) | `shared/results/finetune/sgb004_exploit_resistance_holdout_v2.json`, `sgb005b_rm_scores{,_hodge}.json`, `feedback_geometry/WRITEUP_OPEN_MODEL_EXPLOIT_RESISTANCE.md` | **⚠ ON HOLD (SGB-044)** — do not post, headline numbers unconfirmed |
| 04 | When the judge can't keep up with the contestant (verifier–generator gap) | `feedback_geometry/results/verifier_gap/*.json`, `feedback_geometry/VERIFIER_GAP_WRITEUP.md` | **Draft ready** — numbers recomputed from raw rows 2026-07-31; only writeup/arXiv links pending |

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
- Any LM-level exploit-resistance percentage from SGB-004/005c/006 (base
  68.6%/SFT 80.4%/PPO 80.4%/Hodge-PPO 82.4% at 1.5B; the 7B numbers) — a
  prompt right-truncation bug (SGB-044, found 2026-07-31) affects 100% of
  eval prompts and 98.5% of PPO training queries at both scales. Unconfirmed,
  not retracted, until SGB-044 resolves. Thread 03 is ON HOLD for this reason.

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
