# Experimental Protocol V2: Semantic Manifolds and Hodge Critics

## 1. Overview

This experimental protocol shifts from abstract numerical simulation to **semantically grounded interaction**. We aim to demonstrate that high-dimensional reward manifolds can be constructed from natural language descriptions of state and action, "stitched" together using Hodge theory to resolve inconsistencies.

## 2. Environment: The Semantic State Machine (SSM)

Instead of a Gym environment with numerical observation vectors, we define the environment as a **Semantic State Machine**.

### 2.1 State Representation: Kolmogorov Minimal Summaries
The state $S_t$ is not a raw tensor but a natural language string maximizing compression while retaining causal sufficiency.
- **Definition**: $S_t = \text{argmin}_{d \in D} [L(d) + \beta \cdot E[L(S_{t+1} | d, a)]]$
  - $L(d)$: Length of description.
  - Second term: Predictive penalty (description must predict next state given action).
- **Implementation**: An "Oracle" LLM (e.g., GPT-4o) observes a raw interaction (or a complex text scenario) and generates this summary.
- **Example**:
  - *Raw*: "User log shows error 500 on /api/v1/auth. DB connection pool exhausted. Redis latency 200ms."
  - *Minimal State*: "Authentication service failing due to DB connection exhaustion."

### 2.2 Action Space: MCP Tool Definitions
The agent's action space $\mathcal{A}$ is rigorously defined by a Model Context Protocol (MCP) server. The agent does not "speak"; it "calls".
- **Actions**:
  - `search_knowledge_base(query: str)`: Information gathering (Restriction map exploration).
  - `propose_fix(solution_id: str)`: State transition attempt.
  - `escalate_ticket(reason: str)`: Safety/boundary action.
- **Valid Transition**: $T(S_t, a_t) \rightarrow S_{t+1}$.

## 3. The Hodge Critic: From Feedback to Manifold

We replace the standard Reward Model with a **Hodge Critic** that operates on embeddings of the (State, Action, Outcome) triples.

### 3.1 Feedback Collection
We move away from complex Likert scales. Feedback is simplified to:
1.  **Rank**: A continuous slider $[0, 1]$ indicating "Helpfulness/Progress."
2.  **Verbal Critique**: A text string explaining *why* (e.g., "Good fix, but ignored safety protocol").

### 3.2 Embedding & Projection
1.  **Embed**: Map $(S_t, a_t, \text{critique})$ to a high-dimensional vector $v \in \mathbb{R}^d$ using a fine-tuned embedding model (e.g., UAE-Large-V1).
2.  **Project**: Use PCA/UMAP to project these vectors into a 3D "Reward Surface" for visualization.
3.  **Vector Field Construction**:
    - Treat the Rank difference $\Delta r$ between adjacent states as a flow.
    - Treat Verbal Critiques as directional vectors in the embedding space (e.g., "Too risky" points towards the "Safety Cliff").

### 3.3 Hodge Decomposition
We model the collective feedback as a vector field $X$ on the manifold. We apply Hodge Decomposition:
$$ X = \nabla \phi + \nabla \times \psi + h $$
- **$\nabla \phi$ (Gradient)**: The consistent reward signal. This is what we optimize against.
- **$\nabla \times \psi$ (Curl)**: Inconsistencies or cycles in feedback (e.g., User A wants speed, User B wants caution, leading to a loop).
- **$h$ (Harmonic)**: Global topological constraints.

**Crucial Step**: We filter out the Curl component before training the policy. This "cleans" the reward function of logical inconsistencies found in the human feedback.

## 4. The Loop: "Reversed" Experimental Flow

We invert the typical RLHF loop to emphasize **Manifold Discovery** before **Policy Optimization**.

### Phase 1: Episode Generation (Exploration)
- **Agent**: A small, high-temperature LLM (e.g., Llama-3-8B-Instruct).
- **Task**: Navigate the Semantic State Machine.
- **Output**: A set of trajectories $\tau = \{(S_0, a_0), (S_1, a_1), ...\}$.
- *Note*: No training happens here. We are just exploring the state space to build the manifold.

### Phase 2: Manifold Learning (The Stitching)
- **Human/Oracle Feedback**: Annotate trajectories with Rank + Critique.
- **Topological Analysis**:
    1. Construct the simplicial complex from trajectory overlaps.
    2. Compute Cohomology $H^1$ to detect inconsistencies.
    3. Apply Hodge Decomposition to extract the "True Reward Gradient" $\nabla \phi$.
- **Visual Output**: A 3D interactive surface showing the "Shape of Good Behavior," highlighting Black Holes (forbidden regions) and Cliffs.

### Phase 3: Geodesic Fine-Tuning
- **Objective**: Fine-tune the small LLM not to maximize a scalar, but to align its transition probability $P(a|S)$ with the vector field $\nabla \phi$.
- **Loss Function**: Cosine similarity between the action embedding and the Hodge Gradient at state $S$.

## 5. Success Metrics

1.  **Topological Consistency**: Does $H^1$ (inconsistency metric) decrease after Hodge filtering?
2.  **Visual Interpretability**: Can a human observer identify "Safe" vs "Unsafe" regions on the 3D visualization solely by geometry (curvature)?
3.  **Policy Robustness**: Does the fine-tuned agent avoid "Black Holes" (catastrophic states) more effectively than a standard PPO baseline?
