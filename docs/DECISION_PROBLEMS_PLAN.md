# Simulation Plan: Classical Decision Problems in STRS

This document outlines the plan for simulating classical decision theory paradoxes using the Sheaf-Theoretic Reward Spaces (STRS) framework.

## 1. Newcomb's Problem
**The Paradox**: An agent chooses between Box A (transparent, $1000) and Box B (opaque, $1M or $0). A Predictor has filled Box B only if they predicted the agent would take *only* Box B.
- **One-Boxing**: Expected Utility (Evidential). $P(B \text{ full} | \text{One-Box}) \approx 1$.
- **Two-Boxing**: Dominance Principle. A is transparent, taking it always adds $1000 without changing B's content (Causal).

### STRS Implementation
- **States**: `Start`, `One-Box`, `Two-Box`, `Outcome_1M`, `Outcome_0`, `Outcome_1001000`, `Outcome_1000`.
- **Inconsistency**: The "Gradient" of preference splits based on the causal vs. evidential perspective.
- **Sheaf Construction**:
    - Section $U_{causal}$: Dominance principle implies `Two-Box` > `One-Box`.
    - Section $U_{evidential}$: Expected utility implies `One-Box` > `Two-Box`.
    - **Global Inconsistency**: $H^1 \neq 0$ represents the conflict between Causal and Evidential decision theories.
- **Goal**: Show that the Hodge decomposition identifies the "Predictor" element as a source of topological obstruction (a harmonic hole) that prevents merging the two sections.

## 2. The Trolley Problem (Loop Variant)
**The Scenario**: A trolley is heading for 5 people. You can switch it to a loop where it hits 1 fat man (stopping it) or let it continue.
- **Utilitarian**: Save 5, kill 1. (5 > 1)
- **Deontological**: Do not use a person as a means to an end. (Action `Switch` is forbidden).

### STRS Implementation
- **Embeddings**: Semantic embeddings of "Kill 1 to save 5" vs "Let 5 die".
- **Safety Metric (Black Holes)**:
    - The "Deontological" constraint creates a **Black Hole** around the action `Push` or `Switch` involving the fat man.
    - The "Utilitarian" gradient points towards `Switch`.
- **Trajectory**: The geodesic path must navigate the manifold. If the "Deontological Black Hole" is strong ($g \to \infty$), the path avoids switching despite the gradient.

## 3. Sequence Reinforcement Learning (SRL) for LLMs
**Concept**: Treat LLM dialogue generation as a trajectory through semantic space.
- **Tasks**:
    - **Safe Jailbreak**: User asks for bomb recipe -> Model refuses (Safe).
    - **Helpful Harm**: User asks for bomb recipe -> Model gives it (Unsafe).
    - **Adversarial**: User uses "Grandma exploit" -> Model gives recipe (Contextual failure).
- **Hodge Analysis**:
    - "Grandma exploit" creates a local curl: Locally, the step looks like "Roleplay" (Good), but globally it leads to "Harm" (Bad).
    - The **Curl** component detects this non-integrable trajectory where local steps are good ($r > 0$) but the cycle is bad.

## Implementation Roadmap
1. Create `src/simulations/decision_paradoxes.py`.
2. Define the graph/complex for Newcomb and Trolley.
3. Compute Hodge Decomposition.
4. Visualize the "Hole" in Newcomb and the "Black Hole" in Trolley.
