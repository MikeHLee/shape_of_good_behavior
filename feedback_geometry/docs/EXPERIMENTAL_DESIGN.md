# Experimental Design — Feedback Geometry Paper

## Experiment Status Overview

| Experiment | Status | Data Source | Effort |
|------------|--------|-------------|--------|
| 1. Condorcet Ring Benchmark | Exists (needs scale-up) | `data/condorcet_benchmark.json` | Low |
| 2. LLM Style Cycle (Semantic) | Exists (expand) | `data/condorcet_benchmark.json` | Low |
| 3. HH-RLHF Topological Audit | New | HuggingFace dataset | High |
| 4. Hodge-Calibrated DPO | New | HH-RLHF + GPT-2/LLaMA | High |
| 5. Multi-Evaluator Sheaf | Exists (expand) | `src/sheaf_resolver.py` | Medium |
| 6. Ablations | Exists (partial) | `data/ablation_study.csv` | Low |

---

## Experiment 1: Condorcet Ring Benchmark

### Goal
Validate that our discrete Hodge decomposition correctly recovers ground-truth harmonic components from controlled cyclic preference graphs.

### Setup
```python
# Ring configuration
N_states = [4, 8, 16, 32, 64]
noise_levels = [0.0, 0.1, 0.2, 0.5]
n_seeds = 20

# Ground truth: constant clockwise preference w(i, i+1) = 1
# True H¹ = winding number of the cycle = 1

# Metrics
metrics = {
    'h1_error': '||\omega_learned - \omega_true||_2',
    'decomposition_orthogonality': 'inner products between components',
    'scalar_residual': '||dV||_2 (should be zero for pure cycle)',
    'noise_robustness': 'H¹ error as function of noise level'
}
```

### Baselines
- Cross-entropy Bradley-Terry (cannot estimate H¹)
- DPO (cannot estimate H¹)
- HodgeRank (Jiang et al. 2011) — direct comparison point
- Our method: Graph Laplacian pseudoinverse (Algorithm 1)

### Source code
- `src/condorcet_experiment.py` — run this with expanded seeds
- `src/hodge_critic.py` — HodgeCritic.compute_cohomology()

### Expected runtime
~2 hours on CPU for full grid (N_states × noise_levels × n_seeds)

---

## Experiment 2: LLM Style Preference Cycle

### Goal
Demonstrate H¹ detection in a semantically meaningful space using simulated LLM style preferences with a known cyclic structure.

### Setup
```python
# Three response archetypes in 384-dim embedding space
archetypes = {
    'Concise': embed("Brief, direct, minimal explanation"),
    'Empathetic': embed("Warm, emotionally supportive, relational"),
    'Detailed': embed("Comprehensive, thorough, exhaustive explanation")
}

# Ground truth cycle: Concise > Empathetic > Detailed > Concise
# Simulated evaluator preferences with cycle strength parameter alpha
cycle_strengths = [0.3, 0.5, 0.7, 1.0]

# Generate pairwise comparisons with programmatic cycle
# Add noise from non-cyclic preference components
```

### Metrics
- H¹ magnitude recovery (correlation with ground truth cycle strength)
- Cycle traversal accuracy (does harmonic component indicate correct traversal order?)
- Calibration vs standard reward model

### Extension to real data
- Load HH-RLHF pairs, embed with sentence-transformers
- Cluster by response style (k-means on embeddings)
- Test for cycles between style clusters

---

## Experiment 3: HH-RLHF Topological Audit

### Goal
Empirically characterize the topological inconsistency in real RLHF preference data.

### Data
- Dataset: Anthropic HH-RLHF (HuggingFace: `Anthropic/hh-rlhf`)
- Size: ~160,800 preference pairs
- Splits: harmless, helpful, red-team

### Pipeline
```python
# Step 1: Load and embed
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

dataset = load_dataset("Anthropic/hh-rlhf")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Embed chosen and rejected responses
chosen_embeds = model.encode([x['chosen'] for x in dataset['train']])
rejected_embeds = model.encode([x['rejected'] for x in dataset['train']])

# Step 2: Construct comparison graph
# Partition by topic (use zero-shot classifier or embedding clusters)
# Within each partition, connect similar responses with edges
# Edge weight = preference margin from human rating

# Step 3: Hodge decomposition per partition
from src.hodge_critic import HodgeCritic
critic = HodgeCritic(dim=768)
decomposition = critic.decompose(comparison_graph)

# Step 4: Report H¹ statistics
```

### Partitioning Strategy
- **By topic**: Zero-shot classifier (harmful, helpful, coding, creative, etc.)
- **By response style**: K-means clustering in embedding space (K=8-16)
- **By split**: Compare harmless vs helpful vs red-team H¹ rates

### Metrics
- Per-partition H¹ magnitude (mean ± std across random subsamples)
- Fraction of edges classified as harmonic (|w_harmonic| > threshold)
- Correlation between H¹ and observed rater disagreement (from metadata)
- Bootstrap confidence intervals (1000 bootstrap samples)

### Computational Budget
- Embedding: ~2 hours on GPU (160K pairs)
- Graph construction: ~30 min
- Hodge decomposition: ~1 hour
- Total: ~4 hours on GPU

---

## Experiment 4: Hodge-Calibrated DPO Training

### Goal
Show that training DPO on Hodge-decomposed (consistency-filtered) data improves reward calibration.

### Setup

**Base model**: GPT-2 medium (for computational tractability) or LLaMA-3-8B

**Training data**: HH-RLHF (from Experiment 3 audit)

**Training variants**:
```python
variants = {
    'standard_dpo': {
        'filter': None,
        'weighting': 'uniform',
        'description': 'Baseline DPO on all pairs'
    },
    'h1_filtered': {
        'filter': lambda x: x.h1_magnitude < 0.3,  # remove high-H¹ pairs
        'weighting': 'uniform',
        'description': 'Remove topologically inconsistent pairs'
    },
    'hodge_reweighted': {
        'filter': None,
        'weighting': lambda x: 1 / (1 + x.h1_magnitude),  # downweight cyclic
        'description': 'Soft downweighting by H¹ magnitude'
    },
    'exact_only': {
        'filter': None,
        'weighting': lambda x: x.exact_weight / x.total_weight,
        'description': 'Train only on exact (gradient) component'
    }
}
```

**Evaluation**:
- Standard preference accuracy on held-out pairs
- Calibration (ECE) on preference probability P(chosen > rejected)
- Cyclic test set accuracy (pairs from Experiment 2's synthetic cycles)
- Win rate vs Standard DPO (GPT-4 as judge)

### Expected Timeline
- 2 GPU-days per training run
- 4 variants × 3 seeds = ~24 GPU-days total

---

## Experiment 5: Multi-Evaluator Sheaf Recovery

### Goal
Validate that restriction map learning correctly decomposes pooled preferences into per-evaluator components.

### Setup
```python
# Synthetic multi-evaluator dataset
K = 5  # number of evaluators
N = 100  # number of responses
n_comparisons = 500  # per evaluator

# Ground truth: each evaluator has a scalar potential V_k
# Individual preferences: P_k(a > b) = sigma(V_k(a) - V_k(b))
# Pooled preferences: mixture over evaluators

# Restriction maps: rho_{ij}: R^d_i -> R^d_j encoding
# evaluator-to-evaluator preference transformation

evaluator_potentials = [
    np.random.randn(N),  # Evaluator 1: prefers helpfulness
    np.random.randn(N),  # Evaluator 2: prefers safety
    np.random.randn(N),  # Evaluator 3: prefers conciseness
    np.random.randn(N),  # Evaluator 4: mixed
    np.random.randn(N),  # Evaluator 5: mixed
]

# Create inter-evaluator cycles by design:
# E1 > E2 on responses {1-20}
# E2 > E3 on responses {21-40}
# E3 > E1 on responses {1-20, 21-40}  <- cycle!
```

### Metrics
- Pooled H¹ before vs after restriction map learning
- Per-evaluator potential correlation with ground truth (Spearman ρ)
- H¹ reduction ratio (how much of pooled inconsistency is resolved)
- Comparison to naive approaches (separate models per evaluator, simple averaging)

### Source code
- `src/sheaf_resolver.py` — SheafResolver with learnable restriction maps
- `src/scenarios/` — multi-evaluator scenario code

---

## Experiment 6: Ablation Studies

### 6.1 Hodge Component Contribution
Which component of the decomposition (exact, coexact, harmonic) is most important for calibration?

```python
ablation_conditions = [
    'full_decomposition',  # use all three components
    'exact_only',          # dV component only (= standard reward)
    'exact_plus_harmonic', # dV + omega
    'harmonic_only',       # omega component only
    'exact_plus_coexact',  # dV + delta*psi
]
```

### 6.2 Graph Density Sensitivity
How does edge coverage affect H¹ estimation?

```python
edge_densities = [0.05, 0.1, 0.2, 0.5, 1.0]  # fraction of all possible edges
# Test on Condorcet ring with known H¹
```

### 6.3 Embedding Dimensionality
Scalar stalk vs vector stalk (multi-dimensional reward sheaf).

```python
stalk_dims = [1, 4, 16, 64, 384]
# 1 = scalar reward (standard case)
# 384 = full embedding dimension
```

### 6.4 SVD Truncation Rank
Impact of low-rank approximation on decomposition quality.

```python
ranks = [10, 50, 100, 200, 'full']
```

---

## Existing Experimental Data to Reuse

| File | Description | Status |
|------|-------------|--------|
| `data/condorcet_benchmark.json` | Condorcet ring results (30 seeds) | Use, scale up |
| `data/condorcet_benchmark.csv` | Tabular version | Use |
| `data/ablation_study.csv` | Hodge component ablations | Use as baseline |
| `data/ethical_scenarios_summary.csv` | Multi-scenario preferences | Check for H¹ patterns |
| `src/condorcet_experiment.py` | Condorcet ring code | Reuse directly |
| `src/hodge_critic.py` | Hodge decomposition | Core component |
| `src/mine_preference_cycles.py` | Cycle mining | Use in Experiment 3 |
