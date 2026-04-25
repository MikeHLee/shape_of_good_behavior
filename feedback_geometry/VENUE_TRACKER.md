# Venue Tracker — Feedback Geometry Paper

## Paper Identity
**Working Title**: Preference Cohomology: Detecting and Decomposing Cyclic Inconsistencies in RLHF Feedback
**Track**: Feedback Geometry (Track 1 of "The Shape of Good Behavior" series)
**Status**: Pre-draft (experiments in progress)
**Current Date**: February 2026

---

## Primary Target

### NeurIPS 2026
- **Deadline**: ~May 22, 2026
- **Page limit**: 9 pages + unlimited appendix
- **Fit**: Theory + empirical methods track; strong precedent for topology in ML
- **Risk**: May need stronger real-data experiments (HH-RLHF audit is key)
- **Action needed**: Confirm deadline, identify relevant area chairs

---

## Backup Venues

### ICML 2027
- **Deadline**: ~January 2027
- **Page limit**: 8 pages + unlimited appendix
- **Fit**: Core ML methods; well-aligned with preference learning track
- **Advantage**: More time to develop stronger experiments

### JMLR (Rolling)
- **Deadline**: Rolling submission
- **Page limit**: None
- **Fit**: Long-form theory paper with full proofs; best if theoretical contribution is primary
- **Timeline**: 6-12 months review cycle

### ICLR 2027
- **Deadline**: ~October 2026
- **Page limit**: 8 pages
- **Fit**: RLHF + alignment audience; strong community overlap

---

## Workshop Venues (Preliminary / Parallel Submission)

### TAG-ML (Topology, Algebra, Geometry in ML)
- **Deadline**: Varies (typically NeurIPS/ICML workshop)
- **Page limit**: 4 pages
- **Fit**: Core audience (topology + ML); good for establishing presence
- **Use**: Early feedback on theoretical framework

### RLHF Workshop (NeurIPS/ICML)
- **Deadline**: Varies
- **Page limit**: 4 pages
- **Fit**: RLHF-focused audience
- **Use**: Empirical results on HH-RLHF audit

### Alignment Forum (Blog format)
- **Deadline**: None
- **Use**: "Shape of Good Behavior" blog post component; reach safety community

---

## Conference Timeline (2026)

| Month | Event | Action |
|-------|-------|--------|
| Feb 2026 | Current | Set up paper folder, design experiments |
| Mar 2026 | — | Run HH-RLHF audit experiments |
| Apr 2026 | — | Complete Hodge-calibrated DPO experiments |
| May 2026 | NeurIPS 2026 deadline | Submit if ready |
| Jun 2026 | — | Revisions if needed |
| Oct 2026 | ICLR 2027 deadline | Backup submission |
| Jan 2027 | ICML 2027 deadline | Final backup |

---

## Submission History

| Venue | Date | Decision | Notes |
|-------|------|----------|-------|
| ICML 2026 (combined paper) | Jan 2026 | Pending | Submitted as combined paper with constraint geometry; now bifurcating |

---

## Notes on Differentiation

The combined ICML submission blended feedback geometry and constraint geometry into one paper. The key differentiation for this standalone paper:

1. **Deeper empirical audit**: Real preference data (HH-RLHF), not just synthetic Condorcet rings
2. **Calibration contribution**: Hodge-reweighted DPO as a practical training improvement
3. **Multi-evaluator theory**: Restriction maps as a new theoretical contribution absent from ICML draft
4. **No policy optimization**: This paper stands alone as a contribution to reward modeling, not RL
