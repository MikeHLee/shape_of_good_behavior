# Venue Tracker — Constraint Geometry Paper

## Paper Identity
**Working Title**: Geodesic Policy Optimization: Geometric Hard Safety via Conformal Metric Learning
**Track**: Constraint Geometry (Track 2 of "The Shape of Good Behavior" series)
**Status**: Pre-draft (core experiments exist, need scale-up + Safety Gym)
**Current Date**: February 2026

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
