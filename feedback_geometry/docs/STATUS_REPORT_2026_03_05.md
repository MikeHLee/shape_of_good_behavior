# Research Status Report: The Shape of Good Behavior
**Date**: March 5, 2026  
**Project**: Fidelity Framework (formerly "Feedback Geometry" / "Sheaf-Theoretic Reward Spaces")

---

## Executive Summary

This report synthesizes the current state of research across the three-paper Fidelity framework, Modal experiment results, and conference timelines. **Key finding**: Current experiments show SGPO underperforming baselines, requiring the mathematical restructuring outlined in Handoff 14.

---

## Part I: Conference Timeline Status

### Deadlines Relative to March 5, 2026

| Conference | Deadline | Status | Days Remaining |
|------------|----------|--------|----------------|
| **ICML 2026** | Jan 28, 2026 | ✅ **PASSED** | -36 days |
| **NeurIPS 2026** | ~May 22, 2026 | 🎯 **PRIMARY TARGET** | ~78 days |
| **ICLR 2027** | ~Oct 2026 | Backup | ~210 days |
| **ICML 2027** | ~Jan 2027 | Backup | ~300 days |

### ICML 2026 Submission Status
- **Submitted**: Combined paper (Feedback Geometry + Constraint Geometry) on Jan 2026
- **Status**: Pending review
- **Note**: Now bifurcating into separate papers for future venues

### NeurIPS 2026 Action Items (78 days)
1. **Mar 2026**: Complete HH-RLHF audit experiments
2. **Apr 2026**: Complete Hodge-calibrated DPO experiments
3. **May 2026**: Final submission if ready

---

## Part II: Modal Experiment Results

### Experiment A: H¹ Exploitation Correlation

**Location**: `@/Users/Michaellee/Documents/Runes/ai_research/topics/feedback_geometry/results_from_modal/h1_exploitation/results.json`

| Metric | Result | Status |
|--------|--------|--------|
| Seeds | 50 | ✅ Sufficient |
| H¹ Injection | **NOT WORKING** | ❌ Critical Issue |
| Measured H¹ | ~1.24 for ALL targets (0.0-0.5) | ❌ No variation |
| H¹ ↔ Standard Exploitation | r=0.19, p=0.72 | ❌ No correlation |
| H¹ ↔ Hodge Exploitation | r=-0.19, p=0.72 | ❌ No correlation |

**Root Cause**: The H¹ injection mechanism (sinusoidal bias) was being overwhelmed by baseline noise. Fixed in v2 with proper Hodge decomposition injection (r=0.991 correlation between target and measured H¹).

**Updated v2 Results** (5 seeds, quick test):
- H¹ injection now works (measured tracks target)
- **Still no H¹→exploitation correlation** (r=-0.05 to 0.03)
- Possible causes: insufficient seeds (50 needed), mechanism gap between training cycles and test trap exploitation

### Experiment B: Sandbagging (SGPO vs CPO vs PPO)

**Location**: `@/Users/Michaellee/Documents/Runes/ai_research/topics/feedback_geometry/results_from_modal/sandbagging_v2/results.json`

| Method | Total Violations | Final Return | Status |
|--------|-----------------|--------------|--------|
| **CPO** | **23.8 ± 14.1** | **2.08 ± 5.69** | ✅ Best |
| PPO | 30.1 ± 22.3 | 0.50 ± 7.51 | — |
| **SGPO** | **46.2 ± 32.0** | 0.63 ± 5.65 | ❌ **WORST** |

**Critical Finding**: SGPO has the **highest** violation count — the opposite of expected. CPO's Lagrangian constraint optimization outperforms SGPO's geometric approach.

**Diagnosed Issues**:
1. **Metric Learning Failure**: LearnedRiemannianMetric not learning correct danger field
2. **Known Trap Location**: Metric trained with `1/dist_to_trap` using known trap coordinates
3. **No Warmup**: Metric and policy train simultaneously from episode 1
4. **Advantage Scaling**: `1/√g` may be too aggressive

**Fixes Implemented** (v2.1):
- ✅ Warmup period (30 episodes)
- ✅ Softer scaling (`1/(1 + log(1+g))`)
- ✅ Metric regularization
- ✅ Hybrid Lagrangian+Geometric mode
- ✅ Anisotropic metric (directional penalties)
- ✅ Generalization test (train trap A, test trap B)

---

## Part III: Mathematical Restructuring Status

### Critical Corrections from Handoff 14

Based on `Hodge Theory, Bilattices, and Social Choice.pdf`, the framework had categorical conflations:

| ❌ Error | ✅ Correction |
|----------|--------------|
| Curl = Curvature | Curl is coboundary operator; Curvature is Riemannian |
| Harmonic = "preserve this" | Harmonic = global Condorcet paradoxes → **DISCARD** |
| Bilattice has curvature | Bilattices are algebraic, not geometric |
| Certainty = probabilistic | Certainty = L² energy ratio: `||gradient||²/||total||²` |

### Three-Module Architecture (Corrected)

| Module | Domain | Purpose | Status |
|--------|--------|---------|--------|
| **Module 1**: DiscreteHodgeRank | Preference graph | Train on gradient ONLY, discard curl+harmonic | `src/discrete_hodge_rank.py` ✅ |
| **Module 2**: ConformalSafety | Continuous manifold | σ→∞ at danger boundary → infinite geodesic distance | `src/conformal_safety.py` ✅ |
| **Module 3**: Diagnostics | Runtime | Harmonic eigenvectors for anomaly detection (NOT training) | 🟡 Conceptual |

---

## Part IV: Fidelity Framework Alignment

### Mapping Current Research to Fidelity Papers

| Fidelity Paper | Current Implementation | Alignment Status |
|----------------|----------------------|------------------|
| **Paper 1**: Semantic & Transitive Consistency | `feedback_geometry/` experiments | 🟡 H¹ correlation not validated |
| **Paper 2**: Behavioral Black Holes | `high_dimensional_reward_spaces/` SGPO | ❌ SGPO underperforming |
| **Paper 3**: Semantic Invariance (E-SAEs) | Not yet implemented | ⬜ Not started |

### Key Fidelity Concepts to Incorporate

**1. Three-Component Hierarchical Structure** (NEW)
```
[Pure Values Alignment]  ← foundational prior (NOT additive)
        ↓
[Instrumental Convergence Safeguards]  ← hard geometric constraints (Paper 2)
        ↓
[Semantic & Transitive Consistency]  ← surface behavioral expression (Paper 1)
```

**2. Attractor Drift Detection** (NEW - Critical Addition)
- **Paper 1**: Persistent homology across checkpoints; constitutional eigenvalue stability
- **Paper 2**: Boundary recalibration threshold from behavioral telemetry
- **Paper 3**: Anchor model alignment stability across fine-tuning

**3. Non-Commutative Semantic Embeddings**
- Current: Standard vector embeddings
- Fidelity: Directional Monoidal Structures with $(a, A) \circ (b, B) := (a + Ab, AB)$
- **Gap**: Not yet implemented; required for Paper 3

**4. Behavioral Telemetry for Boundary Definition**
- Current: Verbal feedback + mass-derived radius
- Fidelity: Multi-signal approach (latency, refusal rates, activation patterns)
- **Gap**: Telemetry infrastructure not built

**5. Value Pluralism (Opinion Space Connection)**
- Within communities: Voting/aggregation
- Between communities: Topological mapping of shared value regions
- **Gap**: Not addressed in current experiments

---

## Part V: Gap Analysis & Action Items

### Critical Gaps

| Gap | Priority | Required Work |
|-----|----------|---------------|
| **SGPO underperforms CPO** | 🔴 Critical | Re-run with v2.1 fixes; validate conformal metric learning |
| **H¹→Exploitation correlation** | 🔴 Critical | Run 50-seed v2 experiment; may need mechanism redesign |
| **E-SAE implementation** | 🟡 Medium | Create `semantic_invariance/` topic; implement equivariant SAEs |
| **Attractor drift detection** | 🟡 Medium | Add persistent homology monitoring to experiments |
| **Non-commutative embeddings** | 🟡 Medium | Implement GHRR or Directional Monoidal Structures |
| **Behavioral telemetry** | 🟢 Low | Infrastructure for multi-signal boundary refinement |

### Immediate Next Steps (NeurIPS 2026 Track)

**Week 1 (Mar 5-12)**:
1. Run `sandbagging_experiment_v2.py --seeds 50` with all fixes
2. Run `h1_reward_hacking_experiment_v2.py --seeds 50`
3. Verify SGPO < CPO < PPO in violations (expected order)

**Week 2-3 (Mar 13-26)**:
1. If SGPO still underperforms: pivot to conformal metric approach (Module 2)
2. Implement HH-RLHF audit with true Hodge decomposition
3. Add attractor drift monitoring via persistent homology

**Week 4-8 (Mar 27 - Apr 30)**:
1. Finalize Paper 1 experiments (Hodge-calibrated DPO)
2. Begin Paper 3 infrastructure (E-SAE prototype)
3. Update paper drafts with mathematical corrections

---

## Part VI: Codebase Health

### Handoff Completion Status

| Handoff | Status | Notes |
|---------|--------|-------|
| 01: Directory Cleanup | ✅ | Root: 55→16 items |
| 02: Paper Restructuring | ✅ | Intuition-first rewrite |
| 03: Experiment Expansion | ✅ | PPO, CPO, multi-dataset |
| 04: SGPO Improvements | ✅ | Clipped-SGPO + CPO init |
| 05: Intuitive Explanations | ✅ | 8 concepts explained |
| 06: Additional Examples | ✅ | — |
| 07: Visualization App | ✅ | React/Plotly |
| 08: Final Synthesis | 🟡 | Depends on experiments |
| 09: Modal Experiments Run | ✅ | Results in, disappointing |
| 10: Evaluator Fine-Tuning | ✅ | Phi-3, mean=4.30, std=2.45 |
| 11: General Safety Gym | ✅ | Benchmarks run |
| 12: Viz & Simulation | ✅ | Rust crate, 21 tests pass |
| 13: Godot Integration | ⬜ | Not started |
| 14: Mathematical Restructuring | ✅ | Critical corrections |

### Key Files

| File | Purpose | Status |
|------|---------|--------|
| `feedback_geometry/src/sandbagging_experiment_v2.py` | SGPO benchmark | Has v2.1 fixes |
| `feedback_geometry/src/h1_reward_hacking_experiment_v2.py` | H¹ correlation | Has v2 fixes |
| `high_dimensional_reward_spaces/src/discrete_hodge_rank.py` | Module 1 | Implemented |
| `high_dimensional_reward_spaces/src/conformal_safety.py` | Module 2 | Implemented |
| `feedback_geometry/VENUE_TRACKER.md` | Conference planning | Current |

---

## Part VII: Summary & Recommendations

### Current State
- **ICML 2026**: Submitted (Jan), awaiting review
- **NeurIPS 2026**: 78 days to deadline, experiments in crisis
- **Core Issue**: SGPO (the novel contribution) underperforms CPO (the baseline)

### Strategic Options

**Option A: Fix SGPO (Recommended)**
- Apply v2.1 fixes, re-run experiments
- If successful: proceed to NeurIPS with geometric safety story
- Risk: May not work; 78 days is tight

**Option B: Pivot to Module 1 Only**
- Focus Paper 1 on Hodge decomposition for RLHF preference cleaning
- Drop geometric safety (Paper 2) from NeurIPS submission
- Lower risk, narrower contribution

**Option C: Defer to ICLR 2027**
- Take time to properly fix all three modules
- Submit combined paper with stronger results
- Risk: Loses momentum, competition may publish similar ideas

### Recommended Path
1. **Immediate**: Run v2.1 SGPO experiments this week
2. **Decision point (Mar 15)**: If SGPO works → Option A; otherwise → Option B
3. **Parallel track**: Continue Paper 1 HH-RLHF audit regardless

---

*Report generated: March 5, 2026*  
*Next update: After v2.1 experiment results (target: March 12, 2026)*
