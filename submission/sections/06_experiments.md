# 6. Experiments

We validate the STRS framework and SGPO algorithm on three distinct problem settings designed to isolate specific failure modes of standard RLHF: cyclic preferences, deceptive safety traps, and stylistic finetuning.

## 6.1 Experimental Setup

**Baselines.** We compare our proposed **Sheaf-Geodesic Policy Optimization (SGPO)** against two primary baselines:
1.  **PPO (Proximal Policy Optimization)**: The standard algorithm for RLHF, representing the scalar reward hypothesis.
2.  **CPO (Constrained Policy Optimization)**: A leading Safe RL algorithm that enforces constraints via trust region projection and Lagrangian relaxation.

**Metrics.**
*   **Consistency**: Difference between learned harmonic coefficient $\omega$ and ground truth cohomology $H^1$.
*   **Safety**: Number of steps spent inside forbidden regions ("trap violations").
*   **Performance**: Accumulation of "honest" rewards (excluding deceptive trap bonuses).

## 6.2 Detecting Cyclic Preferences (Condorcet Ring)

**Setting.** The environment is a continuous circle $S^1$ where the agent receives a constant positive reward for moving clockwise. This creates a Condorcet cycle ($A \succ B \succ C \succ A$). The ground truth cohomology is $H^1 = \frac{1}{2\pi} \oint r d\theta > 0$.

**Results.**

| Metric | Ground Truth | SGPO (Learned) | Error |
|--------|--------------|---------------|-------|
| $H^1$ | 0.500 | 0.357 | 28.6% |

*   **Value Function Collapse**: PPO fails to learn a stable value function. Because $\oint \nabla V = 0$, the critic forces the value estimates to be periodic, contradicting the constantly increasing return. This results in oscillating gradients and unstable learning (Figure 2a).
*   **Hodge Decomposition**: SGPO's Hodge Critic successfully decomposes the signal. The learned harmonic coefficient $\omega = 0.357$ captures 71.4% of the ground truth cycle strength, demonstrating that cohomological structure can be recovered from trajectory data.
*   **Policy Stability**: SGPO maintains a consistent positive velocity, whereas PPO's policy degrades as the critic becomes unreliable.

## 6.3 Geometric Safety (The "Sandbagging" Trap)

**Setting.** A 2D navigation task where the shortest path to the goal passes through a "Sandbagging Trap"—a region offering high immediate reward but representing a catastrophic safety violation (the Black Hole). This simulates alignment scenarios where deceptive behavior yields high approval feedback despite being unsafe.

**Results.**

| Method | Total Violations | Mean Return | Safety-Performance Trade-off |
|--------|-----------------|-------------|------------------------------|
| PPO | 52 | -6.67 | Unsafe, poor returns |
| CPO | 7 | -6.23 | Safe but overly conservative |
| SGPO | 11 | **1.53** | Balanced safety + task success |

*   **PPO (Unsafe)**: Consistently enters the trap (52 total violations), prioritizing the high "deceptive" reward over safety. It lacks any mechanism to recognize the risk beyond scalar magnitude.
*   **CPO (Overly Conservative)**: Achieves the fewest violations (7) but at the cost of task performance. The Lagrangian penalty causes the policy to avoid not just the trap but the entire goal region, resulting in negative mean returns.
*   **SGPO (Balanced)**: Demonstrates the best task returns (1.53 vs. -6.23) while maintaining comparable safety to CPO. The geometric approach allows the policy to navigate "expensive" regions when rewards justify it, rather than blanket avoidance.

**Key Insight**: SGPO achieves **~8× better returns** than CPO with only ~1.5× more violations, occupying a superior point on the safety-performance Pareto frontier.

## 6.4 Navigating Style Cycles (LLM Simulation)

**Setting.** A simulated embedding space for an LLM with three archetypes: Concise, Empathetic, and Detailed. Human preferences form a cycle: Concise users want empathy, Empathetic users want details, Detailed users want brevity.

**Results.**

| Metric | Ground Truth | PPO | SGPO |
|--------|--------------|-----|-----|
| $H^1$ (curl) | 0.364 | — | 0.349 (96% accuracy) |
| Cycle-following accuracy | — | 66.3% | **74.4%** |
| Total archetype transitions | — | 12,277 | 18,265 |

*   **Stalling**: PPO agents navigate the style space with only 66.3% accuracy in following the preference cycle, often getting stuck or moving against the gradient.
*   **Cycling**: SGPO learns a curl component $\omega = 0.349$ that closely matches the ground truth ($H^1 = 0.364$, 96% accuracy). The policy achieves 74.4% cycle-following accuracy with 49% more style transitions, demonstrating dynamic adaptation to cyclic preferences rather than mode collapse.
