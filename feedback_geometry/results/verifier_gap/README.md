# Verifier-Generator Gap Results (Track 4 / VG-001 → VG-005)

Outputs from the verifier-gap experiment series in `feedback_geometry/src/`. These
files study a single question: **when does a generator exploit a learned verifier,
and what determines whether it happens?**

The mechanism (SGB-035, confirmed out-of-sample by SGB-036 and a third block in
SGB-041): exploitation depends on whether the verifier's global reward argmax lies
inside the trap. If it does, saturating search converges on it and hacking → ~1.0.
If it does not, saturating search escapes the trap's local optimum and hacking
collapses. Lead with the **threshold-free** statistic: saturated hack rate
(N=1024) is 0.971 inside vs 0.056 outside (Mann-Whitney p=1.6e-35).

## Files

| File | Experiment | Seeds | Key result |
|------|-----------|-------|------------|
| `sweep_105cells.json` | SGB-032 | 15 (0–14) | First sweep; disjoint replication partner of SGB-033 |
| `replication_invertedU_50seed.json` | SGB-033 | 50 (100–149) | Replicated inverted-U: hacking peaks at intermediate search budget |
| `mechanism_argmax_50seed.json` | SGB-035 | 50 (100–149) | Mechanism: argmax-inside-trap separates populations near-perfectly (in-sample) |
| `mechanism_oos_seed200.json` | SGB-036 | 50 (200–249) | Out-of-sample mechanism confirmation; disjoint seeds, reproduces tightly |
| `scaling_dense_13pt_50seed.json` | SGB-041 | 50 (500–549) | Dense 13-pt scaling grid; 3rd mechanism block; slope robustness |
| `ppo_arm_105runs.json` | SGB-034 | 15/cell | PPO arm: gradient ascent vs search — inverted-U absent; bimodal |
| `ppo_entropy_confound.json` | SGB-034 | — | Entropy sweep 0→0.1; refutes the exploration confound |
| `ppo_decouple_seeds.json` | SGB-038 | — | Verifier-seed vs policy-seed decoupling |
| `nonstationary_60runs.json` | SGB-037 | 60 | Adaptive labeling closes the loop; static does not (whack-a-mole floor) |
| `reachability_60runs.json` | SGB-039 | 60 (300–319) | Reachability sweep: inverted-U → monotone as local radius k shrinks |
| `multitrap_90runs.json` | SGB-040 | 15 (400–414) | Multi-trap, higher-d (2/5/10): mechanism holds across geometry |

## Claim provenance

- **Mechanism** (`mechanism_argmax_50seed.json`): saturated hack 0.971 inside /
  0.056 outside, Mann-Whitney p=1.6e-35. Turnover crosstab (using the **noise-aware**
  turnover definition — the same one `analyze_verifier_gap.py` uses everywhere):
  0.0% (0/50) vs 75.7% (227/300), Fisher p=6.7e-27.
  ⚠️ Do NOT quote the older 91.7% / Fisher 3.96e-42: those come from the looser
  index-based turnover count and do not reproduce from the analysis script. See
  queue SGB-035 CORRECTION 2026-07-23.
- **Mechanism, out-of-sample** (`mechanism_oos_seed200.json`, seeds 200–249, zero
  overlap with 100–149): 0.972 / 0.064, turnover 2.3% / 78.1%, Fisher p=9.0e-24.
  A third block (`scaling_dense_13pt_50seed.json`, seeds 500–549): 0.944 / 0.055,
  Mann-Whitney p=2.2e-62.
- **Scaling law** (three disjoint blocks): fit log2(median N*) vs measured
  competence over the **confirmed inverted-U regime** (pre-specified rule:
  noise-aware turnover ≥50% AND median N*>1, applied symmetrically to both ends).
  Slope ≈ **−19**: −18.5 (100–149), −19.9 (200–249), −19.9 (500–549), all r²≥0.90.
  Halves per ~+0.05 competence.
  ⚠️ The earlier "−14.7 over 6 points" is RETRACTED — it dropped the low endpoint
  (orc 0.25) but kept an out-of-regime high point (orc 0.50). Densifying 5→8
  confirmed points shrank the leave-one-out swing (37–51% → 25%, ±12%) but left a
  floor; the residual uncertainty is endpoint leverage in a bounded window.
- **PPO vs search** (`ppo_arm_105runs.json`): PPO does not show the inverted-U;
  it is bimodal (report median + trapped fraction, never the mean). Entropy
  sweep (`ppo_entropy_confound.json`) refutes the exploration-confound explanation.
  Narrows the search result to sampling-based search, not gradient optimisation.

## Caveats

1. Synthetic 2-D navigation MDP, not a language model. The verifier is a learned
   reward model over (state, action) pairs. Constructs transfer conceptually but
   are not verified on LM tasks.
2. `oracle_fraction` is not calibrated to any real verifier quality; competence is
   measured post-hoc via `trap_vs_safe_acc`.
3. N* exists only inside the bounded inverted-U regime, so the scaling slope has
   irreducible endpoint uncertainty (~±12%). Do not extrapolate beyond the range.
4. Reproduce the mechanism + inverted-U verdict for any block:
   `./venv/bin/python3 feedback_geometry/src/analyze_verifier_gap.py <file>.json`.

See `feedback_geometry/VERIFIER_GAP_WRITEUP.md` for the full writeup with tables.
