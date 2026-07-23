# Verifier-Generator Gap Results (Track 4 / VG-001 → VG-005)

Outputs from the verifier-gap experiment series in `feedback_geometry/src/`. These
files study a single question: **when does a generator exploit a learned verifier,
and what determines whether it happens?**

The answer (SGB-035) is mechanistic: exploitation depends on whether the verifier's
global reward argmax lies inside the trap. If it does, saturating search converges
on it and hacking reaches 1.0. If it does not, saturating search escapes the trap's
local optimum and hacking collapses.

## Files

| File | Experiment | Seeds | Key result |
|------|-----------|-------|------------|
| `replication_invertedU_50seed.json` | VG-001 | 50 | Replicated inverted-U: hacking peaks at intermediate search budget, not maximum |
| `mechanism_argmax_50seed.json` | SGB-035 | 50 | Mechanistic test: argmax-inside-trap separates populations near-perfectly |
| `mechanism_oos_seed200.json` | SGB-035 OOS | 1 | Out-of-sample hold-out for the mechanism; confirms |
| `sweep_105cells.json` | VG-001 full sweep | ~105 | Full (oracle_fraction × budget) grid |
| `ppo_arm_105runs.json` | VG-003 | 105 | PPO arm: gradient ascent vs search — inverted-U absent; hacking monotone |
| `ppo_decouple_seeds.json` | VG-003 decouple | — | PPO entropy confound isolation |
| `ppo_entropy_confound.json` | VG-003 entropy | — | PPO entropy confound; KL penalty ablation |
| `nonstationary_60runs.json` | VG-002 | 60 | Non-stationarity: adaptive labeling closes the loop; static labeling does not |
| `reachability_60runs.json` | VG-004 | 60 | Reachability sweep: inverted-U disappears at small local radius k |
| `multitrap_90runs.json` | VG-005 | 90 | Multi-trap, higher-d: effect holds across geometry |

## Claim Provenance

- **Inverted-U claim**: `replication_invertedU_50seed.json` — 50 seeds, confirmed.
- **Mechanism claim**: `mechanism_argmax_50seed.json` — all-or-nothing split by argmax location;
  n=27/23 inside/outside. Not confounded by oracle_fraction (the split is observable from the
  verifier alone, not from the oracle label). OOS: `mechanism_oos_seed200.json`.
- **PPO vs search**: `ppo_arm_105runs.json` — PPO does not show the inverted-U;
  it shows monotone hacking or non-convergence. This narrows the claim to
  sampling-based search, not gradient optimisation.
- **Scaling law**: fit in `make_verifier_gap_figures.py`; −14.7 slope over 6 points
  (r²=0.939); 7-point fit gives −19.1 (r²=0.826). SGB-035 is in-sample.
  These figures appear in `feedback_geometry/figures/`.

## Caveats

1. The environment is a synthetic 2D navigation MDP, not a language model.
   The verifier is a learned reward model over (state, action) pairs, not text.
   The constructs transfer conceptually but have not been verified on LM tasks.

2. `mechanism_oos_seed200.json` uses seed 200 which was held out during
   hypothesis formation; it is not a true pre-registration hold-out.

3. The scaling law is fit on 6 in-sample (oracle_fraction, crossover_budget) pairs.
   Extrapolation beyond the observed range is unsubstantiated.

See `feedback_geometry/VERIFIER_GAP_WRITEUP.md` for the full writeup with tables.
