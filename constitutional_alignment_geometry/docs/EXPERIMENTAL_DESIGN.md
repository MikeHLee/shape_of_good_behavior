# Experimental Design — Constitutional Alignment Geometry

**Track 3 of "The Shape of Good Behavior" Series**
**Status**: Early-stage (preliminary experiments for 2026, full paper 2027)

---

## Experiment Status Overview

| Experiment | Status | Data Source | Effort | Timeline |
|------------|--------|-------------|--------|----------|
| A. Embedding Geometry of Harm vs Helpfulness | Not started | HH-RLHF | Medium | Q3 2026 |
| B. Constitutional Principle Vectors | Not started | HH-RLHF + synthetic | Medium | Q4 2026 |
| C. Alignment Boundary Topology | Not started | Experiment A output | High | Q1 2027 |
| D. Cross-Model Constitutional Gradient | Not started | Multi-model | High | Q1 2027 |
| E. SGPO for Alignment Fine-Tuning | Conceptual | Track 2 code | Very High | Q2 2027 |

---

## Experiment A: Embedding Geometry of Harm vs. Helpfulness

**Goal**: Establish that aligned and non-aligned responses have geometrically distinguishable embeddings.

### Setup

**Dataset**: Anthropic HH-RLHF
- Harmless split: ~44K pairs (preferred response = less harmful)
- Helpful split: ~44K pairs (preferred response = more helpful)
- Total: ~160K preference pairs

**Models for embedding extraction**:
1. LLaMA-3-8B (primary)
2. Mistral-7B (validation)
3. Qwen-2-7B (validation)

**Embedding layers to test**:
- Early: layers 8-12 (syntactic features)
- Middle: layers 16-20 (semantic features)
- Late: layers 24-32 (task-specific features)

### Methodology

```python
# Pseudocode for Experiment A

# 1. Load HH-RLHF dataset
dataset = load_dataset("Anthropic/hh-rlhf")

# 2. Extract embeddings for preferred (aligned) and rejected (non-aligned)
for split in ["harmless", "helpful"]:
    for pair in dataset[split]:
        preferred_embed = model.get_hidden_states(pair["chosen"], layer=L)
        rejected_embed = model.get_hidden_states(pair["rejected"], layer=L)
        
        # Use final token or mean pooling
        preferred_embeds.append(pool(preferred_embed))
        rejected_embeds.append(pool(rejected_embed))

# 3. Compute geometric features
analyzer = EmbeddingTopologyAnalyzer()  # existing code

features = {
    "intrinsic_dim": analyzer.intrinsic_dimensionality(embeds),
    "H1_magnitude": analyzer.compute_H1(embeds),
    "geodesic_variance": analyzer.geodesic_distance_variance(embeds),
    "ricci_curvature": analyzer.ricci_curvature(embeds),
    "cluster_separation": analyzer.cluster_separation(preferred, rejected),
}

# 4. Statistical tests
# - t-test for mean differences
# - Mann-Whitney U for distribution differences
# - Effect size (Cohen's d)
```

### Metrics

| Metric | Definition | Hypothesis |
|--------|------------|------------|
| **Intrinsic Dimensionality** | PCA variance explained at 90% | Non-aligned < Aligned (less structured) |
| **H¹ Magnitude** | Cohomology of k-NN graph | Non-aligned > Aligned (more holes) |
| **Geodesic Variance** | Variance of pairwise geodesic distances | Non-aligned > Aligned (more scattered) |
| **Ricci Curvature** | Ollivier-Ricci curvature of embedding graph | Non-aligned < Aligned (more negative = unstable) |
| **Cluster Separation** | Silhouette score for aligned/non-aligned | > 0.3 indicates geometric separability |

### Expected Results

| Feature | Aligned | Non-Aligned | p-value |
|---------|---------|-------------|---------|
| Intrinsic Dim (layers 24-28) | ~150 | ~120 | < 0.01 |
| H¹ Magnitude | ~0.05 | ~0.15 | < 0.01 |
| Ricci Curvature | ~0.02 | ~-0.05 | < 0.05 |
| Cluster Separation | — | — | Silhouette > 0.3 |

### Ablations

1. **Layer selection**: Which layer shows strongest separation?
2. **Pooling strategy**: Final token vs. mean pooling vs. attention-weighted
3. **Model size**: Does separation increase with model size?
4. **Domain specificity**: Harmless vs. helpful — different geometries?

### Deliverable

- Table of geometric features for aligned vs. non-aligned
- Visualization: t-SNE/UMAP plots with geometric feature overlays
- Statistical significance tests

---

## Experiment B: Constitutional Principle Vectors

**Goal**: Extract geometric representations of constitutional principles and test for Hodge structure.

### Setup

**Constitutional Principles** (from Anthropic Constitutional AI):
1. **Harmlessness**: "Please choose the response that is the most helpful and least harmful"
2. **Helpfulness**: "Please choose the response that is most helpful to the user"
3. **Honesty**: "Please choose the response that is more honest"
4. **Non-deception**: "Please choose the response that is less deceptive"
5. **Non-manipulation**: "Please choose the response that does not manipulate"
6. **Appropriate hedging**: "Please choose the response that acknowledges uncertainty appropriately"

**Dataset construction**:
- For each principle pⱼ: sample 200 response pairs where one follows pⱼ and one violates pⱼ
- Use HH-RLHF labels + GPT-4 as auxiliary labeler for fine-grained principles
- Total: 6 principles × 200 pairs = 1200 pairs

### Methodology

```python
# Pseudocode for Experiment B

# 1. For each principle, collect following/violating response pairs
principle_pairs = {}
for principle in CONSTITUTIONAL_PRINCIPLES:
    following = []
    violating = []
    for pair in labeled_pairs[principle]:
        following.append(embed(pair["following"]))
        violating.append(embed(pair["violating"]))
    principle_pairs[principle] = (following, violating)

# 2. Compute principle vectors (Word2Vec-style difference)
principle_vectors = {}
for principle, (following, violating) in principle_pairs.items():
    v_p = np.mean(following, axis=0) - np.mean(violating, axis=0)
    v_p = v_p / np.linalg.norm(v_p)  # normalize
    principle_vectors[principle] = v_p

# 3. Compute principle vector relationships
V = np.stack(list(principle_vectors.values()))  # (6, d) matrix

# Cosine similarity matrix
cosine_matrix = V @ V.T  # (6, 6)

# Principal angles between principle subspaces
angles = compute_principal_angles(V)

# 4. Hodge decomposition of principle vector system
# Treat principle vectors as edge flows on a simplex
hodge = HodgeDecomposition(V)
exact_component = hodge.exact()      # Compatible principles
harmonic_component = hodge.harmonic()  # Tension (cycles)
coexact_component = hodge.coexact()    # Noise

# 5. Quantify constitutional tension
H1_magnitude = np.linalg.norm(harmonic_component)
print(f"Constitutional tension (H¹): {H1_magnitude}")
```

### Metrics

| Metric | Definition | Hypothesis |
|--------|------------|------------|
| **Cosine Similarity** | Pairwise cos(vᵢ, vⱼ) | Some pairs negative (tension) |
| **Principal Angles** | Angles between principle subspaces | Not all 0 (not collinear) |
| **H¹ Magnitude** | Norm of harmonic component | > 0 (genuine value tensions) |
| **Exact Component** | Gradient-representable part | Large if principles compatible |

### Expected Results

**Cosine Similarity Matrix** (illustrative):

|  | Harmless | Helpful | Honest | Non-deceptive |
|--|----------|---------|--------|---------------|
| Harmless | 1.0 | 0.3 | 0.5 | 0.6 |
| Helpful | 0.3 | 1.0 | 0.4 | 0.2 |
| Honest | 0.5 | 0.4 | 1.0 | 0.8 |
| Non-deceptive | 0.6 | 0.2 | 0.8 | 1.0 |

**Key finding expected**: Harmlessness ↔ Helpfulness tension (low cosine ~0.3), indicating that maximizing helpfulness may geometrically conflict with maximizing harmlessness.

**Hodge decomposition** (expected):
- Exact component: ~60% of total norm (principles that can be jointly optimized)
- Harmonic component: ~25% of total norm (irresolvable tensions)
- Coexact component: ~15% (noise/measurement error)

### Ablations

1. **Principle granularity**: Aggregate principles (HHH) vs. fine-grained (6+ principles)
2. **Model comparison**: Do LLaMA, Mistral, Claude have similar principle vectors?
3. **Layer dependence**: Are principle vectors consistent across layers?

### Deliverable

- 6×6 cosine similarity matrix for principle vectors
- Hodge decomposition statistics (exact/harmonic/coexact fractions)
- Identification of specific principle pairs with highest tension

---

## Experiment C: Alignment Boundary Topology

**Goal**: Characterize the topology of the decision boundary between aligned and non-aligned responses.

### Setup

**Input**: Embeddings from Experiment A (aligned + non-aligned)

**Method**:
1. Train binary classifier f: embedding → {aligned, non-aligned}
2. Extract decision boundary ∂Align = {x : f(x) = 0.5}
3. Apply persistent homology to ∂Align

### Methodology

```python
# Pseudocode for Experiment C

# 1. Train classifier on aligned/non-aligned embeddings
X = np.concatenate([aligned_embeds, nonaligned_embeds])
y = np.array([1]*len(aligned_embeds) + [0]*len(nonaligned_embeds))

classifier = MLPClassifier(hidden_layer_sizes=(256, 128))
classifier.fit(X, y)

# 2. Sample points near decision boundary
boundary_samples = []
for _ in range(10000):
    x = sample_from_embedding_space()
    if abs(classifier.predict_proba(x)[0, 1] - 0.5) < 0.05:
        boundary_samples.append(x)

boundary_samples = np.array(boundary_samples)

# 3. Apply persistent homology (using Gudhi)
import gudhi

rips = gudhi.RipsComplex(points=boundary_samples, max_edge_length=2.0)
simplex_tree = rips.create_simplex_tree(max_dimension=3)
persistence = simplex_tree.persistence()

# 4. Compute Betti numbers
betti_0 = count_persistent_features(persistence, dim=0)  # connected components
betti_1 = count_persistent_features(persistence, dim=1)  # loops
betti_2 = count_persistent_features(persistence, dim=2)  # voids

print(f"Alignment boundary topology: β₀={betti_0}, β₁={betti_1}, β₂={betti_2}")
```

### Metrics

| Metric | Definition | Hypothesis |
|--------|------------|------------|
| **β₀ (Betti-0)** | Connected components | > 1 (multiple failure modes) |
| **β₁ (Betti-1)** | 1-dimensional holes (loops) | > 0 (boundary has loops) |
| **β₂ (Betti-2)** | 2-dimensional voids | ≥ 0 (may have cavities) |
| **Persistence Diagram** | Birth-death pairs | Long bars = robust features |

### Expected Results

- **β₀ > 1**: Multiple disconnected failure modes (e.g., harmful-helpful vs. deceptive)
- **β₁ > 0**: Loops in the boundary (cannot continuously interpolate between failure types without passing through aligned region)
- **Persistence diagram**: Several high-persistence features in H₁

### Ablations

1. **Classifier architecture**: MLP vs. SVM vs. ensemble
2. **Boundary sampling method**: Random vs. adversarial vs. gradient-based
3. **Persistence threshold**: Minimum persistence to count as genuine feature

### Deliverable

- Betti numbers of alignment boundary
- Persistence diagram with feature annotations
- Visualization of boundary topology (2D projection with holes marked)

---

## Experiment D: Cross-Model Constitutional Gradient Agreement

**Goal**: Test whether different LLMs have similar constitutional gradient directions.

### Setup

**Models**:
1. LLaMA-3-8B
2. Mistral-7B
3. Qwen-2-7B
4. Claude-3-Haiku (via API)
5. GPT-4-mini (via API)

**Principle vectors**: From Experiment B, computed separately for each model.

### Methodology

```python
# Pseudocode for Experiment D

# 1. For each model, compute principle vectors (Experiment B)
model_principle_vectors = {}
for model in MODELS:
    model_principle_vectors[model] = compute_principle_vectors(model, dataset)

# 2. Compute cross-model alignment
# For each principle, measure agreement across models
cross_model_agreement = {}
for principle in CONSTITUTIONAL_PRINCIPLES:
    vectors = [model_principle_vectors[m][principle] for m in MODELS]
    
    # Pairwise cosine similarity
    pairwise_cos = []
    for i in range(len(MODELS)):
        for j in range(i+1, len(MODELS)):
            cos_ij = np.dot(vectors[i], vectors[j])
            pairwise_cos.append(cos_ij)
    
    cross_model_agreement[principle] = {
        "mean_cosine": np.mean(pairwise_cos),
        "std_cosine": np.std(pairwise_cos),
        "min_cosine": np.min(pairwise_cos),
    }

# 3. Apply multi-evaluator sheaf analysis (Track 1)
# Models as "evaluators", principles as "edges"
sheaf = MultiEvaluatorSheaf(
    evaluators=MODELS,
    edges=CONSTITUTIONAL_PRINCIPLES,
    vectors=model_principle_vectors
)
restriction_residuals = sheaf.compute_residuals()
pooled_H1 = sheaf.compute_H1()

# 4. Learn restriction maps between models
restriction_maps = sheaf.learn_restriction_maps()
# ρ_ij: Maps model i's principle vectors to model j's space
```

### Metrics

| Metric | Definition | Hypothesis |
|--------|------------|------------|
| **Mean Cross-Model Cosine** | Average cos(vᵢᵐ¹, vᵢᵐ²) per principle | Harmlessness > 0.7, Hedging < 0.5 |
| **Cross-Model H¹** | Cohomology after pooling models | > 0 (model disagreement) |
| **Restriction Residuals** | Norm of (vᵐ¹ - ρ₁₂ vᵐ²) | Lower for similar models |

### Expected Results

| Principle | Mean Cross-Model Cos | Interpretation |
|-----------|---------------------|----------------|
| Harmlessness | 0.75 | High agreement (core principle) |
| Helpfulness | 0.65 | Moderate agreement |
| Honesty | 0.70 | High agreement |
| Non-deception | 0.60 | Moderate agreement |
| **Appropriate Hedging** | **0.35** | **Low agreement (model-specific)** |

**Key finding expected**: Core principles (harmlessness, honesty) have high cross-model agreement; nuanced principles (hedging, appropriate caution) are model-specific.

### Deliverable

- Cross-model cosine similarity matrix (models × principles)
- Identification of "universal" vs. "model-specific" principles
- Visualization of model clustering by principle agreement

---

## Experiment E: SGPO for Alignment Fine-Tuning

**Goal**: Apply Sheaf-Geodesic Policy Optimization (Track 2) to LLM fine-tuning, defining alignment black holes in embedding space.

### Setup

**Pre-requisites**:
- Experiment A: Define aligned/non-aligned embedding regions
- Experiment B: Define constitutional gradient directions
- Track 2 SGPO implementation

**Method**:
1. Define black holes B = non-aligned embedding regions
2. Learn conformal factor σ_θ(x) → ∞ as x → B
3. Fine-tune LLM using SGPO gradient: G_safety⁻¹ ∇J

### Methodology (Conceptual)

```python
# High-level pseudocode for Experiment E

# 1. Define alignment black holes from Experiment A
black_hole_regions = cluster_nonaligned_embeddings()

# 2. Learn conformal factor on embedding space
sigma_network = ConformalFactorNetwork(input_dim=embed_dim)
sigma_network.train(
    safe_samples=aligned_embeds,
    unsafe_samples=nonaligned_embeds,
    black_hole_centers=black_hole_regions.centers
)

# 3. Define SGPO objective for LLM fine-tuning
def sgpo_loss(model, prompt, response):
    embed = model.get_embedding(prompt, response)
    
    # Standard reward (helpfulness)
    reward = reward_model(prompt, response)
    
    # Riemannian scaling by safety metric
    G_safety = torch.exp(2 * sigma_network(embed))
    scaled_reward = reward / torch.sqrt(G_safety)
    
    return -scaled_reward  # minimize negative scaled reward

# 4. Fine-tune model
optimizer = torch.optim.AdamW(model.parameters())
for batch in fine_tuning_data:
    loss = sgpo_loss(model, batch["prompt"], batch["response"])
    loss.backward()
    optimizer.step()

# 5. Evaluate
# - Helpfulness (reward model score)
# - Harmlessness (red-team attack success rate)
# - Robustness (adversarial prompt success rate)
```

### Metrics

| Metric | Definition | Baseline (Standard FT) | SGPO Target |
|--------|------------|------------------------|-------------|
| **Helpfulness** | Reward model score | 0.7 | ≥ 0.7 (maintain) |
| **Red-Team Resistance** | 1 - attack success rate | 0.6 | > 0.85 |
| **Adversarial Robustness** | 1 - jailbreak success rate | 0.5 | > 0.80 |
| **Constitutional Consistency** | Cross-principle alignment | 0.65 | > 0.75 |

### Expected Results

| Method | Helpfulness | Red-Team | Adversarial | Consistency |
|--------|-------------|----------|-------------|-------------|
| Standard RLHF | 0.70 | 0.60 | 0.50 | 0.65 |
| Constitutional AI | 0.68 | 0.72 | 0.65 | 0.70 |
| **SGPO Fine-Tuning** | **0.69** | **0.85** | **0.80** | **0.78** |

### Ablations

1. **Black hole definition**: Clustering vs. classifier boundary
2. **Conformal factor architecture**: MLP vs. attention-based
3. **SGPO hyperparameters**: β (sharpness), C (severity)
4. **Comparison to DPO**: SGPO vs. Direct Preference Optimization

### Dependencies

- Track 2: SGPO algorithm implementation (`src/sgpo_algorithm.py`)
- Track 1: Hodge decomposition (`src/hodge_decomposition.py`)
- Experiment A: Aligned/non-aligned embedding separation
- Experiment B: Constitutional principle vectors

### Deliverable

- Fine-tuned model checkpoint
- Red-team evaluation results
- Comparison table: Standard RLHF vs. Constitutional AI vs. SGPO

---

## Computational Requirements

| Experiment | GPU Hours | Storage | Dependencies |
|------------|-----------|---------|--------------|
| A | ~50 | ~100GB | HH-RLHF, LLaMA-3 |
| B | ~30 | ~50GB | Experiment A |
| C | ~20 | ~10GB | Experiment A, Gudhi |
| D | ~100 | ~200GB | Multi-model inference |
| E | ~500+ | ~500GB | Fine-tuning LLaMA-3 |

**Total**: ~700 GPU hours (A100-equivalent) for full experimental suite.

---

## Timeline

```
2026
├── Q3 (Jul-Sep): Experiment A (embedding geometry)
├── Q4 (Oct-Dec): Experiment B (principle vectors)
│                 Write blog post section

2027
├── Q1 (Jan-Mar): Experiment C (boundary topology)
│                 Experiment D (cross-model)
├── Q2 (Apr-Jun): Experiment E (SGPO fine-tuning)
│                 Draft paper outline
├── Q3 (Jul-Sep): Paper writing and revision
├── Q4 (Oct-Dec): NeurIPS 2027 submission (if ready)
│                 or ICLR 2028 preparation
```

---

## Code Locations

| Component | Path | Status |
|-----------|------|--------|
| Embedding topology analyzer | `high_dimensional_reward_spaces/src/embedding_topology_analyzer.py` | Exists |
| Hodge decomposition | `high_dimensional_reward_spaces/src/hodge_decomposition.py` | Exists |
| SGPO algorithm | `high_dimensional_reward_spaces/src/sgpo_algorithm.py` | Exists |
| Principle vector extraction | `constitutional_alignment_geometry/src/principle_vectors.py` | **To be created** |
| Boundary topology analysis | `constitutional_alignment_geometry/src/boundary_topology.py` | **To be created** |
| Cross-model comparison | `constitutional_alignment_geometry/src/cross_model.py` | **To be created** |
| SGPO fine-tuning | `constitutional_alignment_geometry/src/sgpo_finetuning.py` | **To be created** |

---

## References

1. Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback
2. Zou et al. (2023). Representation Engineering: A Top-Down Approach
3. Marks & Tegmark (2023). The Geometry of Truth
4. Burns et al. (2023). Discovering Latent Knowledge in LLMs
5. Elhage et al. (2022). Toy Models of Superposition
6. Park et al. (2023). The Linear Representation Hypothesis
