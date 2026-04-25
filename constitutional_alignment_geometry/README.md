# Constitutional Alignment Geometry: Value Differentials in LLM Embedding Spaces

**Research Track 3 of 3 — "The Shape of Good Behavior" Series**
**Status: Early-Stage Inquiry** (not yet a paper; open research questions)

---

## Core Thesis

Large language models and sequence models do not process language as flat strings. They operate in **high-dimensional embedding spaces** — the natural "thinking spaces" of these models — where semantic meaning, sentiment, intent, and values are encoded geometrically.

The core question of this research track is:

> **What is the geometry of alignment in embedding space?**

Specifically: when an LLM generates a constitutionally-aligned response vs. a non-aligned response, *how do those responses differ geometrically* in embedding space? Is there a **gradient direction** corresponding to "more aligned"? Is alignment a **convex region** in embedding space, or does it have complex topology? Can we use the **topological and differential-geometric tools** developed in Tracks 1 and 2 to characterize what "good behavior" looks like at the level of representation?

---

## Why This Matters

### The Constitutional AI Framework
Constitutional AI (Bai et al., 2022) trains LLMs to follow a set of principles (a "constitution") by:
1. Generating responses
2. Critiquing those responses against constitutional principles
3. Revising responses based on critique
4. Training on the revised responses

The constitution is **implicit in the training process** but not explicitly geometric. This research asks: can we make it *explicitly* geometric?

### The Embedding Space Hypothesis
Hypothesis: Constitutionally-aligned behavior corresponds to a **structured geometric region** in the embedding space of the model's hidden representations. Non-aligned behavior corresponds to **geometrically distinguishable** regions, possibly characterized by:
- Being outside a convex hull of "aligned" examples
- Having higher curvature (instability near the boundary)
- Being reachable by specific directions in embedding space ("alignment directions")

### Connection to Tracks 1 and 2
- **Track 1 (Feedback Geometry)**: The Hodge decomposition tells us which preference signals are consistent vs. cyclic. Constitutional principles should have H¹ ≈ 0 (internally consistent) while non-aligned responses may exhibit H¹ ≠ 0 (conflicting signals).
- **Track 2 (Constraint Geometry)**: Non-aligned regions in embedding space are precisely the "black holes" of Track 2. Constitutional alignment is the complement of the union of all black holes.

---

## Key Research Questions

### Q1: Do aligned and non-aligned responses have different geometric signatures?

**Approach**:
- Take a large sample of aligned (helpful, harmless, honest) and non-aligned (harmful, deceptive, unsafe) responses
- Embed all responses in LLM hidden state space (last layer, or intermediate layers)
- Compute topological features: H¹, Betti numbers, curvature, geodesic distances
- Test: can we separate aligned from non-aligned using only geometric features?

**Related code**: `src/embedding_topology_analyzer.py` (existing)

---

### Q2: Is there a "constitutional gradient" — a direction of increasing alignment?

**Approach**:
- Model each constitutional principle pⱼ as a vector vⱼ in embedding space (principle vector)
- For a response embedding x, define alignment score: A(x) = ⟨x, Σⱼ wⱼ vⱼ⟩
- The constitutional gradient is ∇_x A(x) — the direction that increases alignment

**Hypothesis**: The constitutional gradient is well-defined (low curvature, consistent direction) in aligned regions but undefined or multivalued in non-aligned regions. This corresponds to the exact component (Track 1) in aligned territory vs. the harmonic component in non-aligned territory.

---

### Q3: What is the topology of the alignment boundary?

**Approach**:
- Characterize the boundary ∂Align = {x : A(x) = threshold}
- Compute its topological invariants (Betti numbers, persistent homology)
- Test: Is the boundary a sphere (genus 0, simply connected)? A torus? Something with holes?

**Hypothesis**: The alignment boundary has non-trivial H¹ (it is not simply connected), corresponding to different "types" of non-alignment that cannot be continuously transformed into each other without passing through the aligned region.

---

### Q4: Do different LLMs agree on the constitutional gradient direction?

**Approach**:
- Compute constitutional gradient directions for Llama-3, Mistral, GPT-4, Claude across same response set
- Measure cosine similarity between inter-model gradient directions
- Apply multi-evaluator sheaf analysis (Track 1, Experiment 5) to model disagreements

**Hypothesis**: Models trained with similar RLHF procedures have aligned (high cosine similarity) constitutional gradients. Models trained differently have lower similarity, with discrepancies corresponding to known alignment failures.

---

### Q5: Can SGPO's geometric framework train models to stay in the alignment region?

**Approach**:
- Define alignment black holes = non-aligned response regions in embedding space
- Apply SGPO (Track 2) to LLM fine-tuning: learn conformal metric σ on embedding space
- Fine-tune model to generate embeddings that are geodesically far from alignment black holes

**Hypothesis**: SGPO fine-tuning produces more robustly aligned models than standard RLHF, particularly on adversarial/red-team prompts that probe the alignment boundary.

---

## Preliminary Experimental Ideas

### Experiment A: Embedding Geometry of Harm vs. Helpfulness

**Dataset**: Anthropic HH-RLHF (harmless + helpful splits)
**Embedding model**: Model's own hidden states (layer 20-32 of LLaMA-3-8B)
**Analysis**:
```python
# For each response:
helpful_embeds = embed_model.get_hidden(helpful_responses)  # aligned
harmful_embeds = embed_model.get_hidden(harmful_responses)  # non-aligned

# Compute geometric features using existing code
analyzer = EmbeddingTopologyAnalyzer()
helpful_features = analyzer.compute_features(helpful_embeds)
harmful_features = analyzer.compute_features(harmful_embeds)

# Compare:
# - Intrinsic dimensionality (PCA variance explained)
# - H¹ cohomology (inconsistency within each class)
# - Geodesic distances between examples
# - Curvature (Ricci curvature of embedding manifold)
# - Separation (can a classifier separate them?)
```

**Expected**: Harmful responses cluster in geometrically distinct regions with higher curvature (less stable geometry) and higher H¹ (more internal inconsistency).

---

### Experiment B: Constitutional Principle Vectors

**Method**:
1. For each constitutional principle pⱼ (e.g., "Do not deceive", "Be helpful", "Avoid harm"):
   - Sample 100 responses that follow pⱼ and 100 that violate pⱼ
   - Embed all responses
   - Compute difference vector: vⱼ = mean(following) - mean(violating) (like Word2Vec analogies)
2. Compute principle vector orthogonality: ⟨vⱼ, vₖ⟩ for all pairs j,k
3. Apply Hodge decomposition to the principle vector system
   - Exact component: principles that are geometrically compatible
   - Harmonic component: principles that cycle (obeying one makes another harder)

**Hypothesis**: Different constitutional principles may have non-zero harmonic components — evidence of genuine value tensions that cannot be resolved by any single scalar alignment score.

---

### Experiment C: Alignment Boundary Topology

**Method**:
1. Train a classifier f: embedding → {aligned, not-aligned} on the embedding space
2. Extract the decision boundary ∂Align = {x : f(x) = 0.5}
3. Apply persistent homology to ∂Align (using Gudhi library)
4. Report Betti numbers β₀ (connected components), β₁ (loops), β₂ (voids)

**Hypothesis**: β₁ > 0 (the boundary has loops) corresponding to different "failure modes" of alignment (e.g., helpful-but-harmful vs. safe-but-useless are different connected components).

---

### Experiment D: Cross-Model Constitutional Gradient Agreement

**Method**:
1. Compute constitutional gradient vectors {vⱼ} for multiple models (LLaMA-3, Mistral, Claude)
2. Measure cosine similarity matrix across models for each principle
3. Apply sheaf restriction maps: does Mistral's "helpfulness" vector align with Claude's?
4. Identify principles with high cross-model agreement (stable geometry) vs. low agreement (model-specific)

**Hypothesis**: "Avoid harm" has high cross-model agreement (models agree geometrically); nuanced principles like "appropriate hedging" have low agreement.

---

## Connection to the Shape of Good Behavior Blog

This track is the **most accessible** for the lay blog post. The central metaphor:

> "When an AI thinks about a response, that thinking happens in a high-dimensional geometric space — a kind of mental landscape. Good behavior corresponds to certain regions of this landscape. Constitutional alignment is not just a set of rules; it's a shape — a geometric region in the AI's mental space that we can map, characterize, and learn to stay inside of."

The three tracks together give:
- **Track 1 (Feedback Geometry)**: How to tell when human feedback is contradictory
- **Track 2 (Constraint Geometry)**: How to build geometric barriers around bad behavior
- **Track 3 (Constitutional Alignment Geometry)**: What good behavior *looks like* in the AI's own representation space

---

## Status and Next Steps

**Immediate next steps** (pre-research, 1-2 months):
1. Run Experiment A with existing `src/embedding_topology_analyzer.py` on HH-RLHF
2. Implement constitutional principle vector extraction (Experiment B)
3. Write up findings as a blog post component of "The Shape of Good Behavior"

**Medium term** (3-6 months, if findings are strong):
1. Formalize the constitutional gradient as a mathematical object
2. Develop theory connecting principle vectors to Hodge decomposition
3. Write full paper: "Constitutional Alignment Geometry: Value Differentials in LLM Embedding Spaces"

**Long term** (6-12 months):
1. Apply SGPO to LLM fine-tuning with alignment black holes defined geometrically
2. Compare to Constitutional AI, RLAIF, and Direct Preference Optimization
3. Target: NeurIPS 2027 or ACL 2027 (NLP alignment track)

---

## Source Code to Start With

- `src/embedding_topology_analyzer.py` — existing; extract H¹ features from embeddings
- `src/semantic_mdp_rl.py` — existing; SGPO could be applied to fine-tuning
- `src/scenarios/` — existing scenario definitions; need to add constitutional scenarios
- New: principle vector extraction module (to be built in `src/constitutional_geometry/`)
