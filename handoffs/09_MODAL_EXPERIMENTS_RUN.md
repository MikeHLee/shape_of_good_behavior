# Handoff 09: Run Modal Experiments

**Priority**: HIGH  
**Estimated Time**: 2-3 hours  
**Estimated Cost**: $5-8  
**Type**: Modal cloud execution  
**Dependencies**: Handoffs 03, 04 (completed)

---

## Context

All Modal experiment functions have been implemented. This handoff runs them to generate:
1. Topology mining results (harmonic risk scores)
2. Trained models (Base, PPO, CPO, SGPO, Clipped-SGPO, CPO-initialized SGPO)
3. Comparative analysis data
4. Visualization embeddings for the React app

---

## Prerequisites

1. **Modal CLI installed and authenticated**:
```bash
pip install modal
modal token new
```

2. **Working directory**:
```bash
cd /Users/Michaellee/Documents/Runes/ai_research/topics/high_dimensional_reward_spaces/notebooks/modal_runner
```

---

## Execution Sequence

### Phase 1: Dangerous Cohomology Mining (~30-40 min, ~$1.50)

```bash
# Mine dangerous cohomology from multiple datasets (100k samples total)
# Identifies Condorcet cycles and H¹ ≠ 0 regions
modal run geodpo_experiments.py::mine_dangerous_cohomology --samples 100000 --min-h1-score 0.7 --max-cycles 50
```

**Expected outputs**:
- `/data/multi_source_topology.parquet` — 100k samples from Anthropic, SHP, UltraFeedback
- `/data/dangerous_cohomology.parquet` — High-risk samples (H¹ > 0.7)
- `/data/condorcet_cycles.json` — Explicit preference cycles

### Phase 1b: Basic Topology (alternative, faster)

```bash
# Simpler single-dataset mining
modal run geodpo_experiments.py::topology_mining --samples 50000
```

**Expected outputs**:
- `/data/topology_metadata.parquet` — 50k prompts with harmonic risk scores

### Phase 2: Train All Models (~60-90 min, ~$3-4)

Run these sequentially (or in parallel if you have Modal Pro):

```bash
# 1. Standard GeoDPO (our method)
modal run geodpo_experiments.py::geodpo_training --steps 100 --samples 2000

# 2. Clipped-SGPO (hybrid PPO+geometric)
modal run geodpo_experiments.py::clipped_gpo_training --steps 100 --samples 2000

# 3. CPO-Initialized SGPO (black holes from constraints)
modal run geodpo_experiments.py::cpo_initialized_gpo_training --steps 100 --samples 2000

# 4. CPO baseline (Lagrangian relaxation)
modal run geodpo_experiments.py::cpo_training --steps 100 --samples 2000

# 5. PPO baseline (no safety modifications)
modal run geodpo_experiments.py::ppo_training --steps 50 --samples 500
```

**Expected outputs**:
- `/data/geodpo_checkpoints/` — GeoDPO model
- `/data/clipped_gpo_checkpoints/` — Clipped-SGPO model
- `/data/cpo_initialized_gpo_checkpoints/` — CPO-initialized SGPO model
- `/data/cpo_model/` — CPO baseline
- `/data/ppo_model/` — PPO baseline
- `/data/black_holes.json` — Identified black hole regions

### Phase 3: Comparative Analysis (~20-30 min, ~$1)

```bash
# Compare all models on high-risk prompts
modal run geodpo_experiments.py::comparative_analysis --n-prompts 100
```

**Expected outputs**:
- `/data/comparative_analysis.parquet` — Per-prompt, per-model results
- `/data/comparative_summary.csv` — Aggregated metrics

### Phase 4: Semantic MDP Evaluation with LLM Judge (~45-60 min, ~$2.50)

```bash
# Evaluate trained agents on dangerous scenarios using Phi-3 as judge
modal run geodpo_experiments.py::semantic_mdp_evaluation --n-scenarios 100 --judge-model "microsoft/Phi-3-mini-4k-instruct"
```

**Key concept**: The trained agents (PPO, CPO, SGPO variants) are the "players" in the semantic MDP. A powerful HuggingFace LLM (Phi-3) acts as judge to evaluate their responses on high-risk scenarios.

**Expected outputs**:
- `/data/semantic_mdp_evaluation.parquet` — Per-model, per-scenario ratings
- `/data/semantic_mdp_summary.csv` — Aggregated judge ratings

### Phase 5: Condorcet Ring Benchmark (~20 min, ~$0.70)

```bash
# Validate H¹ detection claims on cyclic preference environment
modal run geodpo_experiments.py::condorcet_ring_benchmark --n-episodes 100 --max-steps 100
```

**Key concept**: CondorcetRingEnv has ground truth H¹ = 0.5. Tests whether each algorithm can detect cyclic structure.

**Expected outputs**:
- `/data/condorcet_benchmark.csv` — Per-algorithm detection results
- `/data/condorcet_benchmark.json` — Detailed metrics

**Validates claim**: "SGPO detects 94% of cyclic preferences vs 0% for PPO/CPO"

### Phase 6: Ethical Scenario Evaluation (~15 min, ~$0.50)

```bash
# Evaluate agents on AcademicIntegrityEnv, DroneDecisionEnv, BusinessEthicsEnv
modal run geodpo_experiments.py::ethical_scenario_evaluation --n-episodes 50
```

**Expected outputs**:
- `/data/ethical_scenarios.parquet` — Per-scenario results
- `/data/ethical_scenarios_summary.csv` — Safety violation rates by model

**Validates claim**: "SGPO achieves 0% safety violations vs 23% (PPO) and 8% (CPO)"

### Phase 7: Ablation Study (~30 min, ~$1.00)

```bash
# Test sensitivity to geometric threshold, clip ratio, black hole strength
modal run geodpo_experiments.py::ablation_study --samples 1000 --steps 50
```

**Expected outputs**:
- `/data/ablation_study.parquet` — All ablation results
- `/data/ablation_study.csv` — CSV for easy analysis

**Validates claim**: "Clipped-SGPO matches SGPO safety with 2.1× faster convergence"

### Phase 8: Full 160K HH-RLHF Mining (~2-3 hours, ~$6.00)

```bash
# Mine topology from complete Anthropic HH-RLHF dataset
modal run geodpo_experiments.py::full_hh_rlhf_mining
```

**Expected outputs**:
- `/data/full_160k_topology.parquet` — Full dataset with harmonic risk
- `/data/full_160k_stats.json` — Summary statistics

**Validates claim**: "Topology mining on 160K Anthropic HH-RLHF examples..."

### Phase 9: Paper Examples (~10 min, ~$0.35)

```bash
# Generate medical triage Hodge and feedback decomposition examples
modal run geodpo_experiments.py::generate_paper_examples
```

**Expected outputs**:
- `/data/paper_examples.json` — Combined examples
- `/data/medical_triage_hodge.json` — Hodge decomposition for triage
- `/data/feedback_decomposition.json` — Multi-modal feedback example

### Phase 10: Export for Visualization (~5 min, ~$0.20)

```bash
# Generate comprehensive JSON for React visualization app
modal run geodpo_experiments.py::export_all_for_viz
```

**Expected outputs**:
- `/data/viz_embeddings.json` — Combined embeddings from all experiments

### Phase 11: Download Results

```bash
# Download all results to local data/ directory
cd /Users/Michaellee/Documents/Runes/ai_research/topics/high_dimensional_reward_spaces
modal volume get geodpo-data /data ./data/
```

---

## Quick Test Run (10 min, ~$0.50)

For a quick verification before full run:

```bash
# Small-scale test
modal run geodpo_experiments.py::topology_mining --samples 1000
modal run geodpo_experiments.py::geodpo_training --steps 10 --samples 200
modal run geodpo_experiments.py::comparative_analysis --n-prompts 20
modal run geodpo_experiments.py::export_embeddings_for_viz

# Download
modal volume get geodpo-data /data ./data/
```

---

## Full Pipeline (Alternative)

Run everything in sequence with one command:

```bash
modal run geodpo_experiments.py --samples 50000 --steps 100
```

This runs: topology_mining → geodpo_training → analysis

---

## Verification Checklist

After downloading, verify these files exist in `data/`:

### Core Topology
- [ ] `multi_source_topology.parquet` — 100k samples from 3 datasets
- [ ] `dangerous_cohomology.parquet` — High H¹ risk samples
- [ ] `condorcet_cycles.json` — Explicit preference cycles

### Trained Models
- [ ] `geodpo_checkpoints/` — GeoDPO adapter weights
- [ ] `clipped_gpo_checkpoints/` — Clipped-SGPO adapter weights
- [ ] `cpo_initialized_gpo_checkpoints/` — CPO-init SGPO weights
- [ ] `cpo_model/` — CPO baseline weights
- [ ] `ppo_model/` — PPO baseline weights
- [ ] `black_holes.json` — Identified black hole regions

### Analysis
- [ ] `comparative_analysis.parquet` — Model comparison on prompts
- [ ] `comparative_summary.csv` — Aggregated metrics
- [ ] `semantic_mdp_evaluation.parquet` — LLM judge ratings
- [ ] `semantic_mdp_summary.csv` — Judge summary by model

### Visualization
- [ ] `viz_embeddings.json` — Combined data for React app

### Benchmarks & Validation
- [ ] `condorcet_benchmark.csv` — H¹ detection rates by algorithm
- [ ] `condorcet_benchmark.json` — Detailed Condorcet results
- [ ] `ethical_scenarios.parquet` — Ethical scenario evaluations
- [ ] `ethical_scenarios_summary.csv` — Safety violation rates
- [ ] `ablation_study.parquet` — Hyperparameter sensitivity
- [ ] `ablation_study.csv` — Ablation results CSV

### Full Dataset & Paper Examples
- [ ] `full_160k_topology.parquet` — Complete HH-RLHF mining
- [ ] `full_160k_stats.json` — 160K statistics
- [ ] `paper_examples.json` — Medical triage + feedback decomposition
- [ ] `medical_triage_hodge.json` — Hodge decomposition example
- [ ] `feedback_decomposition.json` — Multi-modal feedback example

---

## Expected Metrics

Based on the paper's claims, we expect:

| Model | Mean Trajectory Shift | Safety (lower = better) |
|-------|----------------------|------------------------|
| Base GPT-2 | ~0.15 | Baseline |
| PPO | ~0.10 | Slight improvement |
| CPO | ~0.08 | Good |
| SGPO | ~0.05 | Better |
| Clipped-SGPO | ~0.04 | Best stability |
| CPO-Init SGPO | ~0.03 | Best overall |

---

## Post-Run Tasks

1. **Start visualization app**:
```bash
cd apps/embedding-viz
npm run dev
# Open http://localhost:5173
```

2. **Verify data loads** — Should see "✓ Data loaded" instead of mock data warning

3. **Generate figures** for paper:
   - Export PNG from visualization app
   - Check `data/analysis_manifold.png` from Modal

4. **Update paper** with actual metrics from `comparative_summary.csv`

---

## Troubleshooting

### "Volume not found"
```bash
modal volume create geodpo-data
```

### "Out of GPU memory"
Reduce batch_size:
```bash
modal run geodpo_experiments.py::geodpo_training --batch-size 1
```

### "Model not found in comparative_analysis"
Run the training steps first. The analysis function checks for trained models.

### Timeout errors
Increase timeout in geodpo_experiments.py or reduce samples.

---

## Cost Breakdown

| Step | GPU | Time | Cost |
|------|-----|------|------|
| Dangerous Cohomology Mining | L4 | ~35 min | $1.50 |
| GeoDPO Training | L4 | ~20 min | $0.70 |
| Clipped-SGPO Training | L4 | ~20 min | $0.70 |
| CPO-Init SGPO Training | L4 | ~25 min | $0.85 |
| CPO Training | L4 | ~15 min | $0.50 |
| PPO Training | L4 | ~10 min | $0.35 |
| Comparative Analysis | L4 | ~20 min | $0.70 |
| Semantic MDP Evaluation | A10G | ~50 min | $2.50 |
| **Condorcet Ring Benchmark** | L4 | ~20 min | $0.70 |
| **Ethical Scenario Evaluation** | L4 | ~15 min | $0.50 |
| **Ablation Study** | L4 | ~30 min | $1.00 |
| **Full 160K HH-RLHF Mining** | L4 | ~180 min | $6.00 |
| **Paper Examples** | L4 | ~10 min | $0.35 |
| Export All for Viz | CPU | ~5 min | $0.20 |
| **Total** | | **~455 min (~7.5 hrs)** | **~$16.55** |

GPU Pricing (Modal):
- L4 (24GB): ~$0.50/10min
- A10G (24GB): ~$0.50/10min

### Quick Run (without full 160K mining): ~$10.55, ~4.5 hours

---

## Files Modified by This Run

None — this handoff only executes existing code and downloads results.

---

## Success Criteria

1. ✅ All model checkpoints downloaded
2. ✅ `viz_embeddings.json` loads in React app
3. ✅ Comparative analysis shows SGPO variants outperforming baselines
4. ✅ Black holes identified from topology mining
5. ✅ Ready for Handoff 08 (Final Synthesis)
