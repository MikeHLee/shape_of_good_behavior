# Venue Tracker — Constraint Geometry Paper

## Paper Identity
**Working Title**: Geodesic Policy Optimization: Geometric Hard Safety via Conformal Metric Learning
**Track**: Constraint Geometry (Track 2 of "The Shape of Good Behavior" series)
**Status**: Pre-draft (core experiments exist, need scale-up + Safety Gym)
**Current Date**: February 2026

---

## ⚠️ UNVERIFIED CLAIMS — do not submit until resolved

Audit of 2026-07-21 (see `../EXPERIMENT_ISSUES.md`, "Independent Audit"). The
numbers below are **real experiment output, not fabricated**, but they cannot be
re-derived as evidence for the claims they are attached to. Each must be either
re-run properly or removed before this paper goes to any venue.

| Claim | Appears in | Why UNVERIFIED |
|-------|-----------|----------------|
| ❌ **REFUTED** "Murky Drone — SGPO 0% violations vs 100% PPO/CPO" | `README.md:57`, `PAPER_OUTLINE.md:26`, `submission/main.tex:46`, `submission/sections/introduction.tex:20`, `submission/sections/experiments.tex:36-61` | Re-run properly at **50 seeds** on a real multi-step env (`src/murky_drone_experiment.py`): violations/seed PPO 471.9, **CPO 180.7**, SGPO-scale 508.0, SGPO-barrier 274.8. **CPO is safest**; SGPO's headline formulation is indistinguishable from unconstrained PPO (p=0.68). The old figure came from a one-step bandit that handed SGPO the flag it was scored on. **This claim must be struck from every location listed, not merely caveated.** (§8, §12) |
| "Agentic Shortcut — SGPO 0% vs PPO 100% / CPO 89%" | `submission/sections/experiments.tex:48-56` | Same harness, same circularity, same single seed. (§2) |
| "SGPO detects 94% of cyclic preferences vs 0% for PPO/CPO" | `geodpo_experiments.py:3855` | The 0% baseline is a hardcoded branch — PPO/CPO have no harmonic head and cannot register detection. The function can only ever emit 0% or 100%, never 94%. (§1) |
| "Clipped-SGPO matches SGPO safety with 2.1× faster convergence" | `geodpo_experiments.py:4634` | `ablation_study` sweeps Clipped-SGPO hyperparameters only — there is no SGPO/PPO/CPO arm, so no ratio is computable from it. (§3) |
| Ethical-scenarios table (all cells) | `submission/sections/experiments.tex:48-56` | Faithfully transcribed from real output, but every environment sets `done=True` after one step. (§2, §7) |

**Also**: `submission/` is a skeleton — 5 of 7 section files do not exist and
`icml2026.sty` is missing, so it cannot build end-to-end. Its LaTeX sources were
separately found corrupted (escape sequences expanded to control characters) and
were repaired 2026-07-21; see §9.

**Blocking action**: implement a genuine multi-step Murky Drone against the
existing `src/safety_experiment_hard.py` trainers and re-run at 50+ seeds
(decision of 2026-07-21). Until then, treat every row above as unciteable.

---

## Primary Targets

### NeurIPS 2026
- **Deadline**: ~May 22, 2026
- **Page limit**: 9 pages + unlimited appendix
- **Fit**: Safety/alignment track; strong RL methods presence
- **Risk**: Needs Safety Gym results to compete with safety RL literature
- **Action needed**: Implement Safety Gym experiment by April

### ICRL 2026 (International Conference on Reinforcement Learning)
- **Deadline**: ~July 2026 (check exact)
- **Page limit**: 8 pages + appendix
- **Fit**: Core RL audience; safe RL is a primary track
- **Advantage**: More time for experiments; smaller but focused venue

---

## Backup Venues

### ICLR 2027
- **Deadline**: ~October 2026
- **Page limit**: 8 pages
- **Fit**: Strong RL + safety community; top-tier visibility
- **Timeline**: Most time to develop complete experimental suite

### RLC 2026 (Reinforcement Learning Conference)
- **Deadline**: ~February 2026 (near-term!)
- **Page limit**: 8 pages
- **Fit**: Pure RL, strong fit
- **Risk**: Very tight timeline for Safety Gym experiments

### CoRL 2026 (Conference on Robot Learning)
- **Deadline**: ~June 2026
- **Page limit**: 8 pages
- **Fit**: If robotics simulation experiment is strong (MuJoCo/Safety Gym)
- **Note**: Should only target if robotics framing is developed

---

## Workshop Venues

### Safe and Trustworthy RL Workshop (NeurIPS/ICML)
- Use for early feedback on safety theorem
- Can publish alongside main venue submission

### Alignment Forum (Blog format)
- "Shape of Good Behavior" component for safety community
- Explain geometric safety without math for broader audience

---

## Conference Timeline (2026)

| Month | Event | Action |
|-------|-------|--------|
| Feb 2026 | Current | Design paper structure, expand existing experiments |
| Mar 2026 | — | Run Safety Gym experiments |
| Apr 2026 | — | Complete ablation suite, formal proofs |
| May 2026 | NeurIPS 2026 deadline | Submit if Safety Gym is strong |
| Jul 2026 | ICRL 2026 deadline | Submit if NeurIPS rejected |
| Oct 2026 | ICLR 2027 deadline | Backup |

---

## Submission History

| Venue | Date | Decision | Notes |
|-------|------|----------|-------|
| ICML 2026 (combined paper) | Jan 2026 | Pending | Submitted as combined paper; now bifurcating for expanded experiments |

---

## Notes on Differentiation from ICML Submission

The ICML 2026 submission combined feedback geometry and constraint geometry in 8 pages, limiting experimental depth for each. The standalone constraint geometry paper will:

1. **Deeper safety experiments**: Safety Gym benchmarks (PointGoal, CarGoal, DoggoGoal) currently absent
2. **Formal proofs completed**: Theorems 3.1, 4.1, 5.3 with rigorous β ≥ 2 analysis
3. **Agentic shortcut scenario**: New constitutional-constraint scenario absent from ICML draft
4. **RCBF comparison**: Formal proof of equivalence between SGPO and RCBF
5. **Extended ablations**: Full β × horizon × severity grid, 50 seeds per configuration

## Key Advantage Over Existing Safe RL Papers

CPO (Achiam+17) is the canonical safe RL paper. SGPO's key differentiator:
- CPO: E[C(τ)] ≤ d (soft, expectation over episodes)
- SGPO: P(enter B) = 0 for any trajectory (hard, geometric, per-trajectory)

This is a fundamentally different guarantee, not just a quantitative improvement.
