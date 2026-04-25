# 1. Introduction

## 1.1 The Scalar Reward Problem

Reinforcement Learning from Human Feedback (RLHF) has emerged as the dominant paradigm for aligning large language models (LLMs) and autonomous agents with human intent. The standard recipe involves collecting human preferences over model outputs, training a reward model to predict these preferences, and optimizing a policy to maximize the expected cumulative reward. This approach relies on a fundamental assumption: that human preferences can be faithfully compressed into a scalar reward function $R(s,a) \to \mathbb{R}$.

This scalar assumption is mathematically convenient but topologically restrictive. Human preferences are notoriously complex, often exhibiting:
1.  **Non-transitivity**: Cyclic preferences (e.g., A > B > C > A) that cannot be represented by any scalar value function.
2.  **Context-dependence**: What is "safe" or "helpful" depends on local context in ways that global scalar aggregation obscures.
3.  **Evaluator Disagreement**: Diverging views among human annotators that are typically averaged out, suppressing minority perspectives.

When we force this rich structure into a scalar reward, we induce **topological information loss**. The reward model learns to flatten loops and ignore inconsistencies, often resulting in reward hacking—where the agent exploits the reward model's inability to distinguish between high-quality outputs and those that merely game the scalar metric. Furthermore, standard RLHF lacks formal safety guarantees; safety is typically handled via penalty terms or separate cost models, which can be overridden by sufficiently high rewards.

## 1.2 Our Contribution: Sheaf-Theoretic Reward Spaces

We propose a novel framework, **Sheaf-Theoretic Reward Spaces (STRS)**, that addresses these limitations by modeling rewards not as scalars, but as sections of a sheaf over the trajectory space. This allows us to apply tools from algebraic topology and differential geometry to the alignment problem.

Our key contributions are:

1.  **Topological Consistency Checking**: We model human feedback as local sections of a reward sheaf. We show that the first Čech cohomology group $H^1$ measures the "winding number" of preferences, providing a formal test for global consistency. Non-trivial cohomology ($H^1 \neq 0$) detects Condorcet cycles that scalar rewards miss.

2.  **The Hodge-Augmented Critic**: We introduce a new critic architecture based on the Hodge decomposition theorem, which splits the reward signal into an **exact component** (standard value potential) and a **harmonic component** (cyclic flow). This allows the agent to learn and navigate cyclic preferences rather than stalling or oscillating.

3.  **Geometric Safety via Black Holes**: Instead of soft constraints, we model forbidden regions as **singularities** in a Riemannian reward manifold. We learn a conformal metric that diverges at the "event horizon" of dangerous states, effectively making them infinitely far away in geodesic distance.

4.  **Sheaf-Geodesic Policy Optimization (SGPO)**: We present an algorithm that optimizes policies to follow geodesics on this learned manifold. SGPO naturally integrates consistency and safety, outperforming standard PPO and Constrained Policy Optimization (CPO) on benchmarks involving deceptive traps and cyclic goals.

This work bridges the gap between abstract topology and practical AI safety, offering a mathematically rigorous path beyond scalar rewards.
