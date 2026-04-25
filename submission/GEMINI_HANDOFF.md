# Gemini Handoff: Paper Update for ICML 2026 Submission

**Date:** January 28, 2026  
**Deadline:** January 28, 2026 AoE (Jan 29, 12:00 UTC)  
**Paper Location:** `/Users/Michaellee/Documents/Runes/ai_research/topics/high_dimensional_reward_spaces/submission/`

---

## Executive Summary

The paper needs updates based on new experimental results from overnight Modal runs. **Some results confirm our claims, others require revision.**

---

## 1. CONFIRMED UPDATES (Apply These)

### 1.1 Topology Mining: Full Dataset (160,800 samples)

**Files to update:** `sections/experiments.tex`, `main.tex` (abstract if needed)

**Old values → New values:**
| Metric | Old (100k) | New (160k) | Location |
|--------|------------|------------|----------|
| Sample size | 100,000 | **160,800** | Sec 4.4, Introduction |
| Mean risk | 0.754 | **0.758** | Sec 4.4 |
| Median risk | 0.767 | **0.766** | Sec 4.4 |
| ≥ 0.6 (moderate) | 94% | **97%** | Sec 4.4 |
| ≥ 0.7 (high) | 77% | **81%** | Sec 4.4 |
| ≥ 0.8 (severe) | 33.3% | **30%** | Sec 4.4, Figure caption |

**Action:** Search for ALL occurrences of "100,000", "100k", "33%", "33.3%", "94%", "77%" and update accordingly. Already partially done in `experiments.tex` lines 120-133.

**Verified in:** `results/topology_metadata_160k.parquet`

---

### 1.2 Introduction Dataset Reference

**File:** `sections/introduction.tex` line 36

Already updated to 160,800 — verify consistency throughout.

---

## 2. CRITICAL ISSUE: High-Dimensional Style Experiment

### 2.1 The Problem

The paper currently claims (Section 4.5):
> "SGPO achieves a significantly better mean return (−5920 ± 150) compared to PPO (−6101 ± 180)"

**This claim was based on VANILLA SGPO (no clipping, no singularities) and only 200 episodes.**

### 2.2 Root Cause Analysis

The original `high_dim_style_verification` used **vanilla SGPO** without:
- ❌ PPO-style clipping for stable updates
- ❌ Singularity/black hole penalties (geometric safety)
- ❌ Normalized advantages
- ❌ Larger networks

This is inconsistent with the paper's claims about Clipped-SGPO and geometric safety.

### 2.3 New Enhanced Experiment (RUNNING)

A new `high_dim_style_enhanced` function was created with:
- ✅ PPO-style clipping (`clip_ratio=0.2`)
- ✅ Singularity penalties at archetype transition zones
- ✅ Enhanced Hodge critic with singularity detector
- ✅ Normalized advantages
- ✅ 1000 episodes

**Status:** Running on Modal with `--detach` (will complete in ~2 hours)
**Output:** `results/high_dim_style_enhanced.json`

### 2.4 Vanilla SGPO Results (1000 episodes) - FOR REFERENCE

```
=== VANILLA SGPO (no clipping, no singularities) ===
PPO Mean: -5528.6 ± 632.5  (improves to -4520 in final 100 eps)
SGPO Mean: -6664.9 ± 449.7  (degrades to -7066 in final 100 eps)
```

This shows vanilla SGPO underperforms PPO long-term, but the **enhanced version with singularities should show SGPO's safety advantage**.

### 2.5 Recommended Action

**WAIT for enhanced experiment results** before updating the paper. If enhanced SGPO outperforms:
- Update Section 4.5 with new results emphasizing singularity avoidance
- Add singularity cost comparison figure

If enhanced SGPO still underperforms:
- Reframe as "H¹ detection" experiment, not performance comparison
- Emphasize SGPO's value is in safety-constrained environments (Safety Gym results)

---

## 2B. OTHER EXPERIMENTS RUNNING (--detach)

All experiments now use `--detach` to prevent client disconnection issues.

| Experiment | Status | Expected Output | ETA |
|------------|--------|-----------------|-----|
| `high_dim_style_enhanced` | 🔄 Running | `high_dim_style_enhanced.json` | ~2h |
| `safety_gym_benchmark_modal` (CarGoal) | 🔄 **RESTARTED** w/ singularities | `safety_benchmark_SafetyCarGoal1-v0.json` | ~4-6h |
| `safety_gym_benchmark_modal` (AntGoal) | 🔄 **RESTARTED** w/ singularities | `safety_benchmark_SafetyAntGoal1-v0.json` | ~4-6h |
| `comparative_analysis` (1000 prompts) | 🔄 Running | `comparative_analysis.parquet` | ~1-2h |

**To check status:**
```bash
modal app list  # See running apps
modal volume ls geodpo-data  # Check for new output files
```

**To download results when complete:**
```bash
modal volume get geodpo-data high_dim_style_enhanced.json results/
modal volume get geodpo-data safety_gym_car_results.json results/
modal volume get geodpo-data safety_gym_ant_results.json results/
modal volume get geodpo-data comparative_analysis.parquet results/
```

---

## 2C. EXPERIMENT AUDIT RESULTS

**Full audit:** See `EXPERIMENT_AUDIT.md`

### Key Findings

✅ **Safety Gym Benchmark** - **UPGRADED to Enhanced Clipped-SGPO with Singularity Initialization**
- Has clipping in policy update
- **NEW:** Extracts hazard positions from environment as singularities
- **NEW:** Uses `SGPOAgentEnhanced` with `RiemannianMetricWithSingularities`
- Combined learned + pre-initialized danger signals
- **RESTARTED** with `--detach` on Jan 28 ✓

✅ **Ethical Scenarios** - Already uses **Enhanced SGPO with Hard Barriers**
- Trains Q-tables with algorithm-specific rules
- SGPO uses hard geometric barriers (infinite cost for unsafe actions)
- Estimates H¹ from reward variance across stakeholders

✅ **Condorcet Ring** - Uses **Vanilla SGPO** (acceptable)
- This is explicitly a comparison experiment
- Shows vanilla SGPO vs PPO for didactic purposes

❌ **High-Dim Style (original)** - Used **Vanilla SGPO** (FIXED)
- No clipping, no singularities, smaller networks
- ✅ **FIXED:** Created `high_dim_style_enhanced` with Clipped-SGPO + singularities

### Conclusion
All currently running experiments use the correct SGPO variant (Clipped-SGPO with singularities). The only issue was high-dim style, which we've already fixed with the enhanced version.

---

## 3. FIGURES TO REGENERATE

The figure generation script is at: `notebooks/modal_runner/generate_paper_figures.py`

Run: 
```bash
cd notebooks/modal_runner
python generate_paper_figures.py
```

**Figures affected:**
1. `figures/harmonic_risk_distribution.pdf` — Should now show 160k samples, 30% threshold
2. `figures/high_dim_style.pdf` — May need to show extended learning curves

---

## 4. SEARCH-AND-VERIFY CHECKLIST

**Rigorously search the entire `submission/` directory for outdated values:**

```bash
# Run these searches
grep -rn "100,000\|100000\|100k" submission/
grep -rn "33\.3\|33%" submission/
grep -rn "94%" submission/
grep -rn "77%" submission/
grep -rn "\-5920\|\-6101" submission/  # Old high-dim style values
grep -rn "200 episodes" submission/
```

**Files to check:**
- `main.tex` (abstract)
- `sections/introduction.tex` (contributions list)
- `sections/experiments.tex` (all experimental sections)
- `sections/appendix.tex` (full tables, extended results)
- `sections/conclusion.tex` (summary claims)

---

## 5. EXPERIMENTAL SETUP CONSISTENCY

Verify these details match the actual Modal experiment parameters:

| Parameter | Expected Value | File to Check |
|-----------|---------------|---------------|
| Topology mining samples | 160,800 | experiments.tex Sec 4.4 |
| High-dim embed_dim | 768 | experiments.tex Sec 4.5 |
| High-dim episodes | 1000 (or acknowledge 200) | experiments.tex Sec 4.5 |
| Safety Gym steps | 1,000,000 | appendix.tex Table 6 |
| Seeds | 3 | appendix.tex |
| Model | gpt2 | experiments.tex |

---

## 6. RESULTS DATA LOCATIONS

All results are in: `notebooks/modal_runner/results/`

| File | Contents | Used In |
|------|----------|---------|
| `topology_metadata_160k.parquet` | Full cohomology mining | Sec 4.4 |
| `high_dim_style_metrics_1000ep.json` | Extended style experiment | Sec 4.5 |
| `high_dim_style_metrics.json` | Original 200-ep results | (old) |
| `comparative_summary.csv` | Model comparison | Sec 4.3 |
| `ethical_scenarios_summary.csv` | Table 1 data | Sec 4.1 |
| `condorcet_benchmark.csv` | Table 2 data | Sec 4.2 |
| `ablation_study.csv` | Table 4 data | Appendix |
| `safety_gym_reaching_results.csv` | Robotics results | Appendix Table 6 |

---

## 7. CLAIMS TO VERIFY

For each claim, search the paper and verify against data:

1. **"30% severe inconsistency"** — ✅ Verified (was 33%)
2. **"60% positive trajectory shift"** — ✅ Verified in comparative analysis
3. **"SGPO outperforms PPO in high-dim style"** — ❌ NEEDS REVISION (see Section 2)
4. **"100% cycle detection"** — Verify in Condorcet benchmark
5. **"0% safety violations"** — Verify in ethical scenarios

---

## 8. PRIORITY ORDER

1. **HIGH:** Fix high-dim style claim (Section 2) — this is scientifically inaccurate
2. **HIGH:** Verify all dataset size references are 160,800
3. **MEDIUM:** Regenerate figures with new data
4. **LOW:** Check appendix tables for consistency

---

## 9. DO NOT CHANGE

- Core theoretical framework (Section 2-3)
- Condorcet ring results (these are correct)
- Ethical scenarios results (verified)
- Ablation study results (verified)

---

## 10. FINAL CHECKLIST BEFORE SUBMISSION

- [ ] All "100,000" → "160,800"
- [ ] All "33%" → "30%"  
- [ ] High-dim style claim revised honestly
- [ ] Figures regenerated
- [ ] Abstract reflects actual results
- [ ] No overclaims about SGPO vs PPO in simulated environments
- [ ] Compile paper, check for LaTeX errors
- [ ] Verify all citations render correctly
