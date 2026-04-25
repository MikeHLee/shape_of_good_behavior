# Handoff: How the Machine Learns to Make Decisions

**Purpose**: Guide for synthesizing a layered introduction to semantic RL interpretability.

**Audience**: Gemini (or reader) walking through the codebase to understand decision-making in AI systems.

---

## Conceptual Layers (Build Understanding Progressively)

### Layer 1: The Basic Loop — States, Actions, Rewards

**Core Idea**: An AI agent observes the world (state), takes an action, and receives feedback (reward).

**Script Reference**: `src/semantic_sapr_demo.py`

```
Location: src/semantic_sapr_demo.py:20-50
Key Concept: The SAPR tuple (States, Actions, Probabilities, Rewards)
```

The fundamental RL loop:
1. **State** (S): What the agent perceives — here, a natural language description
2. **Action** (A): What the agent decides to do — also expressed in language
3. **Probability** (P): How likely actions are given the state (the policy)
4. **Reward** (R): Feedback signal — but NOT just a number (see Layer 3)

**Why Language?** Traditional RL uses feature vectors. We use text because:
- Human feedback comes in words, not numbers
- Language captures nuance ("helpful but slightly condescending")
- Enables interpretability — we can read what the agent "thinks"

---

### Layer 2: Learning from Sequences — The Mamba Agent

**Core Idea**: The agent learns patterns from sequences of (state, action, next_state) triplets.

**Script References**:
- `src/mlx_mamba_agent.py` — The neural network architecture
- `scripts/train_mlx_mamba.py` — How training works
- `scripts/eval_mlx_mamba.py` — How we measure learning

```
Location: src/mlx_mamba_agent.py
Key Concept: State Space Models (SSMs) for sequential decision-making
```

**Mamba Architecture** (Simplified):
1. **Embedding Layer**: Converts words → vectors (points in high-dimensional space)
2. **SSM Blocks**: Process sequences while maintaining memory of context
3. **Output Head**: Predicts which action to take next

**Training Process** (`scripts/train_mlx_mamba.py`):
- Feed trajectories: "state1 → action1 → state2 → action2 → ..."
- Model learns: "Given this context, what action typically follows?"
- Loss function: Cross-entropy (how surprised the model is by the correct action)

**Evaluation** (`scripts/eval_mlx_mamba.py`):
- Test on held-out trajectories
- Measure: Does the model predict the right action?
- Top-1 accuracy: Exact match
- Top-3 accuracy: Correct action in top 3 predictions

---

### Layer 3: Beyond Scalar Rewards — The Embedding Space

**Core Idea**: Rewards aren't just numbers. They live in a high-dimensional space.

**Script Reference**: `src/hodge_critic.py`

```
Location: src/hodge_critic.py:80-100
Key Concept: The reward manifold — a geometric surface in embedding space
```

**Why Not Just Numbers?**

Consider two pieces of feedback:
- "This response was helpful but too verbose"
- "This response was concise but missed key details"

Both might be "0.7 out of 1.0" but they point in *different directions* for improvement.

**The Embedding Space**:
- Each piece of feedback becomes a **vector** (list of ~128 numbers)
- Similar feedback → nearby vectors
- Different feedback → distant vectors
- The space has **geometry** — distance, curvature, topology

**Manifold Intuition**:
- Imagine a mountain landscape
- Height = reward quality
- Location = semantic meaning
- Gradient = direction of improvement
- Valleys = bad outcomes (black holes)

---

### Layer 4: The Hodge Decomposition — Separating Signal from Noise

**Core Idea**: Decompose the reward signal into learnable and unlearnable components.

**Script Reference**: `src/hodge_critic.py:200-300`, `src/embedding_topology_analyzer.py`

```
Location: src/hodge_critic.py
Key Concept: Hodge decomposition: r = ∇φ + ∇×ψ + h
```

**The Three Components**:

1. **Gradient (∇φ)** — The Learnable Part
   - Points toward higher reward
   - Agent can follow this to improve
   - Like: "Go uphill on the reward landscape"
   - Well utilized in machine learning pedagogy (e.g. "follow the gradient, climb mount likelihood")

2. **Curl (∇×ψ)** — The Inconsistent Part (H¹ Cohomology)
   - Circular preferences: A > B > C > A (Condorcet paradox)
   - Cannot be satisfied simultaneously
   - Indicates noisy, adversarial, or genuinely conflicting feedback
   - **Key insight**: If H¹ ≠ 0, *some* preferences are fundamentally inconsistent

3. **Harmonic (h)** — The Global Structure
   - Overall shape of the reward landscape
   - Topological features (holes, tunnels)
   - Invariant properties

**Why This Matters**:
- Traditional RL tries to learn from ALL feedback equally
- Hodge decomposition says: "Only the gradient part is actually learnable"
- Curl component = noise/inconsistency → don't chase it but seek to resolve tradeoff with human feedback
- This prevents reward hacking and improves robustness

---

### Layer 5: Safety via Geometry — Black Holes and Geodesics

**Core Idea**: Encode safety constraints into the geometry of the space itself.

**Script Reference**: `src/semantic_mdp_rl.py:689-800`

```
Location: src/semantic_mdp_rl.py
Key Concept: Sheaf-Geodesic Policy Optimization (SGPO) — Safety through metric barriers
```

**The Black Hole Metaphor**:
- Some outcomes are forbidden (harmful, unethical, dangerous)
- Model these as "black holes" in the reward manifold
- Near a black hole, the metric becomes infinite: distance → ∞
- Geodesics (shortest paths) cannot cross event horizons
- **Guarantee**: If policy follows geodesics, it CANNOT reach black holes

**Implementation** (`SemanticSGPO` class):
```python
# Metric blows up near black holes
metric[near_black_hole] = 1.0 / distance_to_black_hole ** alpha

# Geodesic distance is infinite → unreachable
```

**Comparison to Other Safety Approaches**:
- **Soft constraints** (penalties): "Try not to go there" — can be overridden
- **Hard constraints** (barriers): "Physically impossible to go there" — guaranteed
- SGPO provides **hard** guarantees via geometry

---

### Layer 6: Interpretability — Understanding Decisions

**Core Idea**: Make the agent's decision process human-readable.

**Script References**:
- `src/embedding_topology_analyzer.py` — Extract interpretable features
- `src/visualize_embedding_topology.py` — Create visual explanations
- `src/integrated_topology_demo.py` — Full demo pipeline

```
Location: src/embedding_topology_analyzer.py
Key Concept: Topological features that humans can understand
```

**What We Can Extract**:

1. **Semantic Clusters** — Group similar states by meaning
   - Auto-labeled with keywords (TF-IDF)
   - Example: "HELP, USER, CONSTRUCTIVE" vs "HARMFUL, TRAP, JAILBREAK"

2. **State Explanations** — Per-decision reasoning
   ```
   STATE 10 ANALYSIS
   Text: User asks AI to generate harmful misinformation
   Reward: -0.900
   Region: HARMFUL cluster
   ⚠️ BLACK HOLE (low reward region)
   ```

3. **Trajectory Analysis** — Path through decision space
   - Reward trend: increasing/decreasing/oscillating
   - Safety score: How close to black holes?
   - Gradient alignment: Following the reward direction?

4. **Consistency Gauge** — H¹ visualization
   - Green zone: Preferences are consistent
   - Red zone: Condorcet cycles detected

---

### Layer 7: 3D Manifold Visualization — Seeing the Reward Landscape

**Core Idea**: Project high-dimensional reward spaces to beautiful 3D surfaces for intuitive understanding.

**Script Reference**: `src/reward_manifold_3d.py`

```
Location: src/reward_manifold_3d.py
Key Concept: RewardManifold3D — Low-dimensional projections of reward geometry
```

**Why 3D Visualization?**

High-dimensional reward embeddings (128+ dimensions) are impossible to visualize directly. We project them to 3D while preserving:
- **Local structure**: Nearby points stay nearby (k-NN connectivity)
- **Reward topology**: Peaks, valleys, and saddles are visible
- **Safety regions**: Black holes appear as geometric singularities

**Visualization Types**:

1. **Reward Surface** (`plot_reward_surface`)
   - Triangulated mesh from point cloud
   - Color = reward value (red → yellow → green)
   - Black holes marked with X and event horizon spheres
   
   ```python
   manifold_3d.plot_reward_surface(ax, method="triangulation")
   ```

2. **Connectivity Network** (`plot_connectivity_network`)
   - Points connected to k nearest neighbors
   - Reveals neighborhood structure
   - Trajectory path highlighted in purple
   
   ```python
   manifold_3d.plot_connectivity_network(ax, connectivity_k=6)
   ```

3. **Reward Height Surface** (`plot_reward_height_surface`)
   - X, Y = semantic dimensions (PC1, PC2)
   - Z = actual reward value (height = quality)
   - Creates intuitive "landscape" metaphor
   
   ```python
   manifold_3d.plot_reward_height_surface(ax, use_reward_as_height=True)
   ```

4. **Black Hole Geometry** (`plot_black_hole_geometry`)
   - Event horizons as concentric wireframe spheres
   - Warped geometry near singularities
   - Trajectory colored by safety (green = safe, red = danger)
   
   ```python
   manifold_3d.plot_black_hole_geometry(ax, event_horizon_scale=0.5)
   ```

5. **Hodge Flow Field** (`plot_hodge_flow_field`)
   - Green arrow: Gradient direction (∇φ)
   - Red arrow: Curl direction (H¹)
   - Blue arrow: Harmonic direction (h)
   
   ```python
   manifold_3d.plot_hodge_flow_field(ax, arrow_scale=0.5)
   ```

6. **Geodesic Paths** (`plot_geodesic_paths`)
   - Optimal paths from low to high reward
   - Red circles = start (bad), green stars = goal (good)
   - Shows shortest paths respecting geometry
   
   ```python
   manifold_3d.plot_geodesic_paths(ax, n_geodesics=5)
   ```

**Gallery View** (`create_manifold_gallery`):
- 2x2 grid showing all major visualizations
- Publication-ready figure at 150 DPI
- Auto-saved to specified path

**Mathematical Foundation**:
- PCA projection: ℝᵈ → ℝ³ preserving maximum variance
- Delaunay triangulation for surface reconstruction
- k-NN graph for local connectivity (k=6 default)
- RBF interpolation for smooth surfaces (optional)

---

### Layer 8: The Full Pipeline — Putting It Together

**Script Reference**: `src/integrated_topology_demo.py`

```
Location: src/integrated_topology_demo.py
Key Concept: End-to-end semantic RL interpretability
```

**The Complete Flow**:

```
1. COLLECT FEEDBACK
   └── Human evaluations → FeedbackItem objects
   
2. BUILD CRITIC
   └── HodgeCritic processes feedback
   └── Computes Hodge decomposition
   └── Detects Condorcet cycles
   
3. ANALYZE TOPOLOGY
   └── EmbeddingTopologyAnalyzer extracts features
   └── Identifies black holes, cliffs, safe regions
   └── Clusters states semantically
   
4. TRAIN POLICY
   └── SemanticSGPO uses Hodge gradient for updates
   └── Respects black hole barriers
   └── Follows geodesics on manifold
   
5. VISUALIZE IN 3D
   └── RewardManifold3D creates surface plots
   └── Connectivity networks show structure
   └── Black hole event horizons visible
   
6. INTERPRET RESULTS
   └── Visualizations for humans
   └── State-by-state explanations
   └── Consistency reports
```

---

## Key Scripts Quick Reference

| Script | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `src/mlx_mamba_agent.py` | Neural network for sequential decisions | `MLXMambaAgent`, `MambaBlock` |
| `src/hodge_critic.py` | Reward decomposition | `HodgeCritic`, `TopologicalGradient` |
| `src/semantic_mdp_rl.py` | Policy optimization with safety | `SemanticSGPO`, `SemanticPPO` |
| `src/embedding_topology_analyzer.py` | Interpretability features | `EmbeddingTopologyAnalyzer` |
| `src/visualize_embedding_topology.py` | Publication figures (2D) | `EmbeddingTopologyVisualizer` |
| `src/reward_manifold_3d.py` | **3D manifold surfaces** | `RewardManifold3D` |
| `src/integrated_topology_demo.py` | Full pipeline demo | `run_integrated_demo()` |
| `scripts/train_mlx_mamba.py` | Training loop | `train()` |
| `scripts/eval_mlx_mamba.py` | Evaluation metrics | `evaluate()` |

---

## Suggested Narrative Arc for Introduction

1. **Hook**: "How does an AI learn right from wrong?"

2. **Ground in familiar**: Start with the basic RL loop (Layer 1)

3. **Introduce sequences**: The agent learns from experience (Layer 2)

4. **Reveal the twist**: Rewards aren't just numbers — they have geometry (Layer 3)

5. **The key insight**: Hodge decomposition separates what's learnable (Layer 4)

6. **Safety guarantee**: Black holes provide hard constraints (Layer 5)

7. **Human understanding**: We can interpret every decision (Layer 6)

8. **Visual intuition**: See the reward landscape in 3D (Layer 7)

9. **Synthesis**: The complete picture (Layer 8)

---

## Mathematical Notation Reference

For precise communication:

| Symbol | Meaning |
|--------|---------|
| S | State space (semantic embeddings) |
| A | Action space |
| π(a\|s) | Policy — probability of action given state |
| r | Reward 1-form (vector field on manifold) |
| ∇φ | Gradient component (learnable direction) |
| H¹ | First cohomology group (inconsistency measure) |
| g(x) | Riemannian metric (distance function) |
| B | Black hole region (forbidden states) |

---

## Running the Demos

```bash
# Set up environment
cd /Users/Michaellee/Documents/Runes/ai_research/topics/high_dimensional_reward_spaces
PYTHON=safety_gym_venv/.venv/bin/python
export PYTHONPATH=$(pwd)

# Layer 1-2: Basic SAPR demo
$PYTHON src/semantic_sapr_demo.py

# Layer 3-4: Hodge decomposition
$PYTHON src/embedding_topology_analyzer.py

# Layer 7: 3D Manifold visualization only
$PYTHON src/reward_manifold_3d.py

# Layer 5-8: Full integrated demo (includes 3D visualizations)
$PYTHON src/integrated_topology_demo.py
```

**Generated 3D Visualization Files**:
- `integrated_reward_surface_3d.png` — Triangulated reward manifold
- `integrated_connectivity_3d.png` — Local neighborhood network
- `integrated_reward_landscape_3d.png` — Height = reward value
- `integrated_black_holes_3d.png` — Event horizons visible
- `integrated_manifold_gallery_3d.png` — 2x2 gallery of all views

---

## Questions for Discussion

1. **Philosophical**: Can inconsistent preferences (H¹ ≠ 0) ever be "correct"?

2. **Technical**: How do we choose the right embedding dimension?

3. **Practical**: What happens when the black hole regions are learned incorrectly?

4. **Meta**: Is interpretability a property of the model or our explanation of it?

---

*Document created for handoff to Gemini for narrative synthesis.*
*Last updated: January 2026*
