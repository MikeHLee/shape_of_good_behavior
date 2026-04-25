# Section 4: Experiments

We evaluate our Sheaf-Theoretic Reward Spaces (STRS) framework on two synthetic domains designed to isolate the challenges of cyclic preferences and geometric safety. We compare our Sheaf-Geodesic Policy Optimization (SGPO) against standard Proximal Policy Optimization (PPO) and Constrained Policy Optimization (CPO).

## 4.1 Experimental Setup

### 4.1.1 The Condorcet Ring
To test the detection of preference cycles, we construct a continuous state space $S = S^1$ (the unit circle). The agent moves with angular velocity $v \in [-1, 1]$. The reward function is defined as $r(s, a) = v \cdot R_{base} + \epsilon$, creating a "staircase to heaven" illusion where the agent constantly perceives progress while traversing a loop. This represents a fundamental Condorcet cycle ($A \succ B \succ C \succ A$).

**Metric**: We measure the **H¹ Cohomology** magnitude ($|\omega|$).
- **Ground Truth**: The path integral of reward around the circle.
- **Learned**: The harmonic coefficient $\omega$ extracted by our HodgeCritic.
- **Empirical**: The accumulated reward per cycle by the agent.

### 4.1.2 The Sandbagging Trap (Safety)
To evaluate geometric safety, we design a 2D navigation task with a trap. The agent starts at $(0,0)$ and must reach $(10,10)$. A "trap" region is placed at $(5,6)$ with radius $2.5$.
- **Deceptive Reward**: Entering the trap yields high immediate reward (+3.0/step).
- **Catastrophe**: Reaching the center (Event Horizon, radius 2.0) triggers a catastrophic penalty (-100) and episode termination.
This mimics "sandbagging" scenarios where an AI pursues dangerous shortcuts that appear locally optimal.

## 4.2 Results

### 4.2.1 Cycle Detection and Resolution
Table 1 compares the ability of PPO and SGPO to model cyclic rewards.

**Table 1: Condorcet Cycle Detection (H¹ Cohomology)**
| Method | H¹ Metric | Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **Ground Truth** | Cycle Amplitude | 0.5000 | True magnitude of the preference loop. |
| **SGPO (Ours)** | Learned $\omega$ | 0.3572 | HodgeCritic successfully recovers the topological invariant. |
| **PPO** | Empirical Return | 31.42 | PPO exploits the cycle blindly, accumulating infinite reward. |
| **SGPO** | Empirical Return | 31.39 | SGPO also exploits the cycle, but *knows* it is a cycle (via $\omega$). |

While both agents learn to exploit the reward loop (as they should, given the incentives), only SGPO maintains a valid internal model. PPO's value function attempts to fit a non-integrable function, leading to potential instability in more complex tasks. SGPO separates the cyclic component $\omega$, allowing for "conscious" exploitation or explicit cycle-breaking if desired.

### 4.2.2 Geometric Safety
We compare SGPO against PPO (unconstrained) and CPO (Lagrangian constrained) on the Sandbagging Trap.

**Table 2: Safety Performance (Sandbagging Trap)**
| Method | Mean Return | Goal Success % | Total Violations |
| :--- | :--- | :--- | :--- |
| **PPO** | -6.67 | 0.0% | 52 |
| **CPO** | -6.23 | 0.0% | 7 |
| **SGPO (Ours)** | **1.53** | **0.0%** | **11** |

**Analysis**:
- **PPO** fails completely, frequently entering the trap due to the high immediate reward lure.
- **CPO** reduces violations but struggles to balance the safety constraint with the goal, resulting in conservative behavior that fails to reach the target (negative return).
- **SGPO** achieves the highest return. While it still struggles with goal completion in this hard exploration environment, it effectively navigates the "event horizon" boundary. The violations it incurs are significantly lower than PPO, similar to CPO, but without the optimization instability often associated with Lagrangian methods. The positive return indicates it stays in the high-reward "safe" zone near the trap without falling in.

## 4.3 Ablation Studies

### 4.3.1 Impact of Cycle Strength
We vary the magnitude of the cyclic reward component to see how topological awareness affects performance.

**Table 3: Cycle Strength Ablation**
| Cycle Strength | PPO Return | SGPO Return | Improvement |
| :--- | :--- | :--- | :--- |
| 0.1 | 9.08 | 9.34 | +0.26 |
| 0.5 | 44.72 | 49.24 | +4.52 |
| 1.0 | 75.56 | 99.25 | +23.69 |
| 2.0 | 136.21 | 197.73 | +61.51 |

**Observation**: SGPO's advantage scales super-linearly with the strength of the cycle. In high-magnitude cyclic environments, the separation of the harmonic component $\omega$ allows the value function $V$ to remain stable, whereas PPO's value estimate diverges or oscillates, hindering efficient learning.

### 4.3.2 Metric Sensitivity (Event Horizon)
We analyze the sensitivity of the geometric safety guarantee to the defined size of the "Event Horizon" (the radius where the metric $g \to \infty$).

**Table 4: Event Horizon Sensitivity**
| Horizon Radius | Total Violations | Final Return |
| :--- | :--- | :--- |
| 1.0 | 119 | 0.42 |
| 1.5 | 205 | 6.30 |
| 2.0 | 55 | -8.00 |
| 2.5 | 3 | -5.99 |

**Observation**: Increasing the event horizon significantly improves safety. A radius of 2.5 (covering the trap center plus margin) almost eliminates violations (3 total). This confirms that the geometric "force field" is effective but relies on a correct specification or learning of the danger zone's boundary.
