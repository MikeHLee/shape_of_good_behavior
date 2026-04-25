# 8. Discussion and Limitations

## 8.1 SGPO vs. CPO: A Nuanced Comparison

Our experiments reveal that SGPO and CPO occupy different points on the safety-performance Pareto frontier:

| Metric | PPO | CPO | SGPO |
|--------|-----|-----|-----|
| Mean Return | -6.67 | -6.23 | **1.53** |
| Total Violations | 52 | 7 | 11 |

**Key finding**: SGPO achieves approximately **8× better task returns** than CPO while incurring only ~1.5× more violations. This reflects a fundamental difference in mechanism:

- **CPO** uses Lagrangian multipliers that penalize *any* proximity to constraints, often becoming overly conservative. The policy learns to avoid the goal region entirely if it lies near the penalty zone.
- **SGPO** uses geometric structure where the metric increases smoothly near danger. The policy can still pursue rewards in geometrically "expensive" regions when the reward justifies it.

Neither method dominates: CPO is preferable when **zero violations** is paramount; SGPO is preferable when **task completion with bounded risk** is the objective. The geometric approach also offers:

1. **No constraint tuning** — CPO requires threshold hyperparameters ($d$, $\lambda$); SGPO learns the metric
2. **Intrinsic interpretability** — the learned metric $g(x)$ visualizes danger regions directly
3. **Potential generalization** — SGPO's metric can extrapolate to novel unsafe states via learned features

## 8.2 Limitations

While Sheaf-Theoretic Reward Spaces offer a rigorous alternative to scalar reward modeling, several limitations remain:

1.  **Computational Complexity**: Computing exact Čech cohomology grows combinatorially with the size of the cover. While our discrete approximation on trajectory graphs is efficient ($O(T \cdot k \cdot d^2)$), scaling to massive datasets of human feedback may require sparse approximations or spectral methods (e.g., Sheaf Laplacians).
2.  **Metric Bootstrapping**: The safety metric $g(x)$ relies on cost signals to identify "event horizons." If these signals are themselves sparse or noisy, the learned geometry may be flawed. We assume a "weak supervision" signal for safety is available, which may not always hold. 
3.  **Manifold Assumption**: SGPO assumes the reward space has a meaningful manifold structure. In discrete domains (e.g., token-level text generation), this smoothness assumption is an approximation. Embedding discrete states into continuous spaces (like standard transformer embeddings) mitigates this, but the topological fidelity of such embeddings is an open question.

## 8.3 Future Work

**Scaling to LLMs.** The most immediate direction is applying SGPO to full-scale language model fine-tuning. This involves training a "Hodge Reward Model" that outputs both a scalar score and a vector field on the embedding space, allowing the LLM to navigate stylistic cycles or steer around conceptual "black holes" in the prompt space.

**Multi-Agent Coordination.** Sheaf theory naturally extends to multi-agent systems, where each agent's local observations form a section. Consistency checks via cohomology could detect misalignment or conflicting goals between agents without requiring a centralized value function.

**Temporal Cohomology.** Preferences often drift over time. Extending the framework to include a temporal dimension would allow us to detect "concept drift" in alignment as a non-zero cohomology class in the time direction, distinguishing between valid preference shifts and alignment instability.

## 8.4 Broader Impact

This work moves AI safety towards **interpretable geometric certificates**. Instead of opaque neural networks that "usually work," topological invariants like $H^1$ provide discrete, falsifiable checks for alignment consistency. However, powerful tools for navigating preference manifolds could also be used to manipulate user behavior more effectively. As with all alignment research, robust safety checks (like the black hole mechanism) are dual-use and must be deployed with care.
