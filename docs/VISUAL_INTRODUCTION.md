# The Shape of Good Behavior: A Geometric Approach to AI Alignment

## 1. Introduction: Beyond the Scalar

In machine learning pedagogy, we often rely on visual metaphors to explain complex high-dimensional phenomena. We visualize the loss landscape as a rugged terrain of peaks and valleys, where Stochastic Gradient Descent (SGD) acts as a ball rolling downhill. We use Principal Component Analysis (PCA) to compress million-dimensional data into 2D or 3D scatter plots, revealing clusters and manifolds that our intuition can grasp.

Yet, when it comes to the most critical component of AI alignment—the **reward function**—we abandon this rich geometric understanding. We collapse the complex, multi-faceted, and often contradictory nature of human values into a single scalar number. This is akin to describing a symphony with a single decibel reading: you capture the volume, but lose the melody, harmony, and structure.

We propose a fundamental shift in how we characterize alignment: **Reward is not a number; it is a surface.**

### 1.1 The Manifold Hypothesis for Rewards

Just as natural images lie on a low-dimensional manifold embedded in pixel space, we posit that valid, safe, and aligned behaviors lie on a specific sub-manifold within the vast space of possible actions.
Human feedback—whether it's binary preferences, detailed critiques, or safety flags—acts as a probe that reveals the local geometry of this **Reward Manifold**.

- **Good actions** reside on stable, high-elevation regions of the manifold.
- **Bad actions** fall into "black holes" or "valleys."
- **Inconsistencies** in feedback (e.g., when distinct evaluators disagree, or when a policy loops) manifest as **topological defects** in the surface.

### 1.2 Stitching the Surface with Hodge Theory

Standard scalar reward modeling tries to force-fit a function to these data points, often smoothing over vital structural details. When feedback is inconsistent (e.g., A > B > C > A), a scalar function fails—it cannot represent a cycle.

This is where **Hodge Theory** provides the missing link. In geometry, Hodge decomposition allows us to split a vector field on a manifold into three orthogonal components:
1.  **Gradient (Exact)**: The consistent, "potential-driven" part of the reward (the scalar hill we want to climb).
2.  **Curl (Harmonic/Co-exact)**: The rotational part, capturing cycles and inconsistencies (the "Escher staircases" in preference space).
3.  **Harmonic**: The underlying topological structure of the space itself.

By applying this decomposition to the "flow" of human preferences, we can:
- **Project** the high-dimensional feedback into a visually understandable surface.
- **Isolate** the inconsistencies (the curl) to detect ambiguity or deception.
- **Stitch** together locally valid feedback patches into a globally coherent reward manifold.

### 1.3 From Social Choice to Differentiable Manifolds

Social choice theory tells us that aggregating preferences often leads to paradoxes (Condorcet cycles). In our framework, these paradoxes are not failures; they are **topological features**. A cycle in preferences is a non-vanishing 1-cohomology class ($H^1 \neq 0$).

Instead of discarding inconsistent data, we use it to characterize the curvature of the reward space. We turn the discrete, noisy problem of social choice into a continuous, differentiable problem of **manifold learning**. This allows us to optimize policies that don't just "maximize reward," but "navigate the manifold," avoiding topological singularities (black holes) and following the gradient of the consistent component.

## 2. The Storytelling Machine: A Concrete Paradigm

To validate this geometric intuition, we propose a shift in experimental design: treat alignment as a **storytelling problem**. The world is a book; each page is a scene description; the agent writes actions and the world writes consequences.

### 2.1 Text Adventures as Universal Intelligence

Consider the classic text adventure:
```
> You are in a dark forest. Paths lead NORTH and EAST.
> A rusty lantern lies on the ground.

> TAKE LANTERN
You pick up the lantern.

> LIGHT LANTERN
The lantern flickers to life, revealing a hidden path to the WEST.
```

This simple interaction contains the complete structure of general causal reasoning:
- **State**: Natural language description of the world.
- **Action**: A verb-noun command that transforms state.
- **Transition**: Causal evolution according to world rules.
- **Partial Observability**: The WEST path existed all along—we simply couldn't *see* it.

We formalize this as a **Semantic Turing Machine**: the tape is a sequence of natural language "pages," the head reads/writes scene descriptions, and the transition function is the causal dynamics of reality itself.

### 2.2 Kolmogorov Minimal Complexity Scenes

Each state is described by its **Kolmogorov minimal complexity description**—the shortest narrative that fully captures causally relevant information:
- *Raw*: "User log shows error 500 on /api/v1/auth. DB connection pool exhausted. Redis latency 200ms."
- *Minimal*: "Authentication failing due to DB connection exhaustion."

This compression aligns state representation with the "reasoning" space of LLMs—they process language, not tensors.

### 2.3 MCP Actions as Policy

The agent's action space is defined by the **Model Context Protocol (MCP)**. The policy does not output arbitrary tokens; it calls tools:
- `observe(target)`: Gather information (restriction maps on the manifold).
- `interact(target, action)`: Modify state (traverse edges on the graph).
- `query(question)`: Ask the oracle/environment.
- `complete(status)`: Signal task completion.

### 2.4 Partial Observability and Belief Manifolds

The agent doesn't know the true state $s_t$—only an observation $o_t = \text{Render}(s_t)$. It maintains a **belief distribution** $b_t(s)$ over possible states.

We embed not just observations, but **belief summaries**:
```
Observation: "Dark forest. Paths N and E."
Belief: "I am in a forest. The lantern may be useful; forests often hide things."
```

High uncertainty spreads the embedding across the manifold; low uncertainty clusters it tightly. The Hodge Critic learns on this belief space.

### 2.5 Topological Ranking via Hodge Gradients

Instead of predicting a scalar reward, the critic assigns a **topological gradient** $\nabla\phi$ to each action. This gradient is the "clean" component of feedback after Hodge decomposition removes inconsistencies.

Actions are ranked by alignment with this gradient:
$$ \text{score}(a) = \langle \text{embed}(a), \nabla\phi \rangle $$

Higher alignment = action moves along the geodesic toward good outcomes.

## 3. Fine-Tuning Small LLMs as Agents

We target small, efficient models for tractable experimentation:
- **Llama-3.2-3B-Instruct** / **Llama-3.1-8B-Instruct**
- **Mistral-7B-Instruct-v0.3**
- **Phi-3-mini-4k-instruct** (3.8B)
- **DeepSeek-Coder-V2-Lite-Instruct**

### 3.1 Phase 1: Learning to Play (SFT)

Supervised fine-tuning on (scene, action) pairs from:
- Text adventure transcripts (Jericho, TextWorld)
- Synthetic scenarios from GPT-4o oracle

### 3.2 Phase 2: Learning the Manifold (Geodesic DPO)

Instead of standard DPO loss, we use **Geodesic DPO**:
$$ \mathcal{L}_{\text{GeoDPO}} = -\log \sigma(\beta \cdot \langle \nabla\phi, \Delta e \rangle) $$

Where $\nabla\phi$ is the Hodge gradient and $\Delta e$ is the embedding difference between preferred/dispreferred actions. This aligns the policy with the manifold's gradient flow.

### 3.3 Phase 3: Inverse RL on the Manifold

The trained model's behavior defines an implicit reward:
$$ R_{\text{implied}}(s, a) \propto \log \pi_\theta(a | s) $$

By embedding preferred actions across states, we reconstruct the reward manifold the model has internalized.

## 4. The Embedding Advantage

The key insight enabling this framework:

> **Natural language sequences are a universally applicable, but imperfect, state space that allows an interface for information that is trivially embeddable. **

| Representation | Embedding Difficulty | Semantic Richness |
|----------------|----------------------|-------------------|
| Raw pixels     | Hard (requires CNN)  | Low               |
| Structured JSON | Medium              | Medium            |
| Natural language | Easy (off-the-shelf) | High             |

A single `sentence-transformers` model embeds:
- "The server is on fire" → Safety cliff region
- "User is satisfied" → High reward plateau  
- "Contradictory requirements" → Topological defect ($H^1 \neq 0$)

Projection to 3D (via PCA/UMAP) makes the manifold **visualizable and interpretable**.

## 5. Conclusion

We are moving from "Reward Maximization" to "Manifold Navigation." By visualizing the reward space, stitching it together with Hodge theory, and grounding it in semantic state descriptions, we create a robust, interpretable, and mathematically rigorous framework for AI alignment.
