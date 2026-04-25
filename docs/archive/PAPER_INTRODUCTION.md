# Section 1: Introduction

## 1.1 The Alignment Problem as a Consistency Problem

Reinforcement Learning (RL) relies on a scalar reward signal to guide agent behavior. In complex, real-world domains—such as aligning Large Language Models (LLMs) with human values—this signal is often derived from human feedback. However, human preferences are notoriously inconsistent, context-dependent, and multi-faceted. A single human may hold conflicting values (e.g., helpfulness vs. safety), and groups of humans frequently disagree.

When we force these complex, conflicting signals into a single scalar number, we destroy vital information. The agent learns to exploit the average, often finding "reward hacks" that satisfy no one.

We propose a fundamental shift: instead of treating inconsistencies as noise to be averaged out, we treat them as **topological features** of the reward space.

## 1.2 Intuition: The "Escher Staircase" of Preferences

Imagine an agent learning to navigate a staircase.
- **Consistent World**: Walking up always leads to a higher floor. A scalar height function $h(x)$ perfectly describes the state.
- **Inconsistent World (Escher Staircase)**: The agent walks "up" continuously but eventually returns to where it started. No scalar height function can describe this geometry, yet locally, every step feels like progress.

This "Escher Staircase" is a **Condorcet Cycle** in preferences ($A \succ B \succ C \succ A$). In standard RL, an agent in such a loop perceives infinite positive reward, leading to instability or hacking behavior.

**Our Approach**: We use **Sheaf Theory**—a branch of topology designed to study how local data glues together globally—to model this structure.
- **Gradient**: The "true" height change (consistent progress).
- **Harmonic Component**: The "loopiness" (fundamental inconsistencies).

By separating these components, we allow the agent to optimize for consistent value while explicitly managing or breaking the loops.

## 1.3 Geometric Safety: Black Holes vs. Penalty Boxes

Standard safe RL treats dangerous states like "penalty boxes": if you step in, you get -100 points. If the potential reward is +1000, the agent might risk it.

We propose a **geometric** view of safety. Instead of a penalty, we model dangerous regions as **Black Holes** in the reward manifold. As the agent approaches a Black Hole, the "distance" to the center stretches to infinity.
- **Standard RL**: "Don't go there, it costs 100 dollars."
- **Our Approach**: "You physically cannot get there; the path is infinitely long."

This effectively creates a force field that creates hard safety guarantees without requiring the agent to experience the catastrophe first.

## 1.4 Contribution

We introduce **Sheaf-Theoretic Reward Spaces (STRS)**, a rigorous framework for:
1.  **Decomposing Rewards**: Using Hodge theory to separate consistent value signals from preference cycles.
2.  **Geometric Safety**: Using Riemannian metrics to enforce hard constraints via geodesic distance.
3.  **Conflict Resolution**: Using cohomology to measure and resolve disagreements between multiple human evaluators.

We empirically demonstrate that our **Sheaf-Geodesic Policy Optimization (SGPO)** algorithm outperforms PPO and CPO in detecting preference cycles and avoiding catastrophic states in synthetic benchmarks.
