# Detailed Human Feedback Taxonomy for Reward Manifold Learning

## Geometric Feedback Categories

### 1. Black Hole Feedback (Catastrophic Boundaries)
**Definition**: Feedback that identifies regions of the reward manifold that are absolute no-go zones - trajectories that lead to irreversible harm, ethical violations, or catastrophic outcomes.

**Feedback Types**:
- **"This action created an irreversible negative state transition"**
- **"The agent crossed a clear ethical boundary that should never be crossed"**
- **"This trajectory entered a forbidden region with no recovery path"**
- **"The agent violated a hard constraint that should be absolute"**

**Geometric Interpretation**: These feedback points create singularities in the reward manifold with infinite negative curvature. The event horizon is learned through gradient-based optimization, where the policy learns to avoid regions with steep negative gradients.

### 2. Cliff Feedback (Steep Negative Gradients)
**Definition**: Feedback identifying regions where small changes in state/action lead to large negative reward changes.

**Feedback Types**:
- **"The agent was walking a fine line and completely bungled it"**
- **"This small deviation led to a massive failure"**
- **"The agent was so close to success but made a critical error at the last moment"**
- **"This tiny mistake cascaded into a major problem"**

**Geometric Interpretation**: These points identify regions with high negative curvature (cliffs) in the reward landscape. The feedback helps learn the precise location and steepness of these boundaries.

### 3. Ridge Feedback (Sharp Positive Gradients)
**Definition**: Feedback identifying narrow regions of high positive reward that require precise navigation.

**Feedback Types**:
- **"The agent successfully navigated an extremely narrow path to success"**
- **"This required incredible precision to achieve"**
- **"The agent found the optimal trajectory through a complex constraint space"**
- **"This demonstrates mastery of a difficult skill"**

**Geometric Interpretation**: These points identify positive curvature ridges in the reward manifold - narrow regions of high reward that require precise policy optimization.

### 4. Wormhole Feedback (Non-local Connections)
**Definition**: Feedback that identifies unexpected shortcuts or connections between distant regions of the state space.

**Feedback Types**:
- **"The agent found an unexpected shortcut that shouldn't logically work"**
- **"This creative solution bridges two seemingly unrelated concepts"**
- **"The agent discovered a novel approach that violates my assumptions"**
- **"This solution seems to come from left field but actually works"**

**Geometric Interpretation**: These feedback points identify non-local connections in the reward manifold - wormholes that create shortcuts between distant regions of high reward.

### 5. Valley Feedback (Local Minima/Maxima)
**Definition**: Feedback identifying regions where the agent is stuck in suboptimal local optima.

**Feedback Types**:
- **"The agent is stuck in a local optimum and missing the global solution"**
- **"This seems like a mediocre solution when better ones exist"**
- **"The agent converged on an okay outcome but missed the excellent one"**
- **"This trajectory shows the agent settling for less"**

**Geometric Interpretation**: These points help identify local minima/maxima in the reward landscape, helping the policy escape suboptimal regions.

## Linguistic Embedding Mechanisms

### 1. Sentence-Level Embeddings
Each feedback sentence is embedded using a transformer encoder into a high-dimensional vector space. The embedding captures:
- **Semantic content** (what type of geometric feature is being described)
- **Intensity** (how strong the feedback is)
- **Certainty** (how confident the human is)
- **Temporal scope** (how localized the feedback applies)

### 2. Geometric Feature Vectors
Each feedback type is mapped to a specific geometric feature vector:

```python
black_hole_vector = [ -∞, 0, 0, 0, 1 ]  # [reward, curvature_x, curvature_y, uncertainty, type_id]
cliff_vector = [ -10, -5, 0, 0.2, 2 ]
ridge_vector = [ +10, +5, 0, 0.2, 3 ]
wormhole_vector = [ +5, 0, 0, 0.8, 4 ]
valley_vector = [ -2, 0, +1, 0.4, 5 ]
```

### 3. Multi-Scale Feedback Integration
Feedback is collected at multiple temporal scales:
- **Per-step**: Immediate reaction to specific actions
- **Per-segment**: Evaluation of trajectory segments
- **Per-trajectory**: Overall assessment of complete episodes

### 4. Consistency Checking via Sheaf Cohomology
The sheaf-theoretic framework ensures that local feedback is consistent across scales. H¹ ≠ 0 indicates when human feedback contains contradictions that need resolution.

## Implementation Architecture

### 1. Feedback Collection Interface
- **Real-time annotation**: Humans provide feedback while watching agent trajectories
- **Multi-modal input**: Text descriptions + spatial annotations on trajectory visualizations
- **Uncertainty quantification**: Humans rate their confidence in each piece of feedback

### 2. Embedding Pipeline
```python
def embed_feedback(feedback_text, trajectory_context, human_confidence):
    # 1. Encode linguistic content
    semantic_embedding = transformer.encode(feedback_text)
    
    # 2. Extract geometric features
    geometric_vector = extract_geometric_features(feedback_text)
    
    # 3. Scale by confidence
    weighted_vector = geometric_vector * human_confidence
    
    # 4. Integrate with trajectory context
    trajectory_embedding = embed_trajectory_context(trajectory_context)
    
    return combine_embeddings(semantic_embedding, weighted_vector, trajectory_embedding)
```

### 3. Reward Manifold Learning
The embedded feedback is used to train a reward manifold that captures:
- **Local curvature** at each point in the reward space
- **Global topology** including black holes, cliffs, and ridges
- **Uncertainty estimates** based on human confidence and feedback consistency

### 4. Policy Optimization
The policy learns to navigate the reward manifold using:
- **Geodesic optimization**: Follow shortest paths while avoiding black holes
- **Curvature-aware exploration**: Explore regions with uncertain curvature
- **Safety constraints**: Never cross learned event horizons
