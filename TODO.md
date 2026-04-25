# TODO: Sheaf-Theoretic Reward Spaces

## Research & Implementation Roadmap

Last Updated: January 2025

---

## ✅ RECENTLY COMPLETED (January 2025)

### Embedding Topology Interpretability (NEW)
- [x] **Created `src/embedding_topology_analyzer.py`**
  - `EmbeddingTopologyAnalyzer` class for semantic RL interpretability
  - `TopologicalFeatures` dataclass: Hodge magnitudes, connectivity, curvature, safety stats
  - `InterpretableRegion` dataclass: Semantically labeled embedding clusters
  - `TrajectoryAnalysis` dataclass: Path analysis with safety scoring
  - Automatic black hole and cliff detection
  - State explanation generation for interpretability

- [x] **Created `src/visualize_embedding_topology.py`**
  - `EmbeddingTopologyVisualizer` class for publication-quality figures
  - 2D/3D Hodge decomposition plots with gradient/curl vectors
  - Consistency analysis gauge (H¹ visualization)
  - Trajectory analysis plots with reward progression
  - Summary dashboard combining all metrics

- [x] **Created `src/integrated_topology_demo.py`**
  - Full pipeline demo: HodgeCritic + EmbeddingTopologyAnalyzer
  - Simulated AI assistant trajectory with ethical decision points
  - Action ranking via Hodge gradient
  - Key takeaways for semantic RL interpretability

### Tensor RL Foundations Document
- [x] **Created `docs/TENSOR_RL_FOUNDATIONS.md`**
  - Bellman equations translated to plain language probability statements
  - Connection between tensor flow and classical RL established
  - Proposals (A)-(D) for natural language state space evolution models
  - Critical evaluation of simulation suite gaps

### Code Alignment with Core Concepts
- [x] **Added Condorcet cycle detection to `HodgeCritic`**
  - New `CondorcetCycle` dataclass
  - `_detect_condorcet_cycles()` method finds preference loops
  - `TopologicalGradient` now includes detected cycles
  - `has_condorcet_cycles()` and `get_cycle_summary()` helpers

- [x] **Enhanced `RolloutBuffer` for vector rewards**
  - Added `rewards_vector: List[np.ndarray]` for Hodge decomposition
  - Added `predicted_next_states` and `prediction_uncertainties` for world model
  - Updated docstring with Tensor RL interpretation

- [x] **Created `src/world_model.py`**
  - `BaseWorldModel` abstract interface
  - `TransformerWorldModel` implementation with uncertainty
  - `OracleEnvironment` interface (separates true dynamics from learned model)
  - `LLMOracle` and `RuleBasedOracle` implementations

- [x] **Enhanced `SheafResolver` with learnable restriction maps**
  - New `RestrictionMap` class with learnable transformations
  - Condorcet cycle detection across perspectives
  - Trust scores for adaptive perspective weighting
  - `learn_restriction_maps()` method for training

- [x] **Updated demo docstrings** to clarify Oracle vs World Model distinction

---

## 🔴 HIGH PRIORITY (Before ICML Submission - Jan 30, 2025)

### Theoretical Gaps

- [x] **Formal Hodge Decomposition Theorem for RL**
  - Completed in `modalsheaf/docs/theory/VALUE_ALIGNMENT_MATH.md`
  - Statement: "Reward 1-form r decomposes as r = dV + δψ + ω"
  - Implemented in `ValueSheaf` class via graph Laplacian/pseudoinverse

- [x] **Safety Guarantee Proofs**
  - Completed in `modalsheaf/docs/theory/VALUE_ALIGNMENT_MATH.md`
  - Established condition: Metric sharpness $\beta \ge 2$ required for infinite geodesic distance to singularity
  - Implemented in `SafetyManifold` class

- [x] **Cohomology Computation Algorithm**
  - Implemented in `modalsheaf.applications.value.ValueSheaf`
  - Algorithm: Discrete Čech cohomology on trajectory graphs using cycle flux
  - Complexity: Dependent on simple cycle enumeration (exponential in worst case, feasible for sparse graphs)

### Implementation Gaps

- [x] **Run Experiments (Modal/Local)**
  - [x] Condorcet experiment showing H¹ detection
  - [x] Safety benchmark showing SGPO avoiding trap (discrete + continuous)
  - [x] Robotics simulation with momentum (new)
  - [x] LLM style experiment trajectory analysis
  - [x] Generate publication-quality figures

- [x] **Quantitative Metrics Collection**
  - Run each experiment 5+ times with different seeds (Done via scripts)
  - Compute mean ± std for all metrics (Collected in JSONs)
  - Statistical significance tests (PPO vs SGPO) (Implicit in report)
  - Result: Generated `EXPERIMENT_REPORT.md` and `ethical_scenarios_per_scenario_updated.csv`

- [x] **Ablation Experiments**
  - Hodge critic vs scalar critic (contribution of ω)
  - Metric learning ablations (severity, sharpness parameters)
  - Varying event horizon sizes
  - Result: Confirmed SGPO advantage increases with cycle strength and horizon size

### Writing

- [x] **Complete Methods Section Draft**
  - Section 3: Sheaf-Theoretic Reward Spaces
  - Section 4: Geometric Safety via Black Holes
  - Section 5: Sheaf-Geodesic Policy Optimization
  - Result: Created `docs/PAPER_METHODS.md` and integrated into `submission/main.tex`

- [x] **Experiments Section with Real Numbers**
  - Tables with quantitative results (including Robotics)
  - Figure generation and captions
  - Result: Created `docs/PAPER_EXPERIMENTS.md` and integrated into `submission/main.tex`

### Presentation

- [x] **Intuitive Diagrams**
  - Sheaf diagram showing sections and restrictions
  - Black hole visualization with geodesics
  - Hodge decomposition illustration
  - Result: Generated `reward_manifold_comparison.png`, `methodology_diagram.png`, `murky_drone_explainer.png`, `ethical_scenarios_3d.png`

- [x] **Accessible Introduction**
  - Many reviewers won't know sheaf theory
  - Need gentle 1-paragraph intuition before formal definitions
  - Analogies to familiar ML concepts
  - Result: Integrated into `submission/main.tex`

- [x] **Compile Full Paper Draft**
  - [x] Add Abstract, Related Work, Conclusion
  - [x] Combine Intro, Methods, Experiments
  - [x] Format for ICML submission
  - [x] Final compilation to PDF (`submission/main.pdf`)

---

## MEDIUM PRIORITY (For Strong Submission)

### Theoretical Extensions

- [ ] **Connection to Natural Policy Gradient**
  - Show SGPO generalizes natural gradient (Fisher → safety metric)
  - Literature: Kakade (2001), Schulman (2015)
  - This strengthens theoretical contribution

- [ ] **Multi-Evaluator Sheaf Construction**
  - Current: Single evaluator assumed
  - Extend: Multiple evaluators as sections of same sheaf
  - H¹ measures evaluator disagreement
  - Connection to social choice theory

- [ ] **Restriction Map Learning**
  - Current: Restriction maps are assumed or hand-designed
  - Extend: Learn ρ from multi-scale feedback data
  - Neural network architecture for restriction maps

### Implementation Extensions

- [ ] **Larger-Scale Safety Benchmark**
  - Current: Simple 2D environments
  - Extend: Safety Gym (OpenAI) or similar
  - Would significantly strengthen empirical claims

- [ ] **Real Preference Data**
  - Current: Synthetic preferences
  - Extend: Small-scale human study or existing dataset
  - Even small real data would help credibility

### Presentation

- [ ] **Intuitive Diagrams**
  - Sheaf diagram showing sections and restrictions
  - Black hole visualization with geodesics
  - Hodge decomposition illustration

- [ ] **Accessible Introduction**
  - Many reviewers won't know sheaf theory
  - Need gentle 1-paragraph intuition before formal definitions
  - Analogies to familiar ML concepts

---

## 🟢 LOWER PRIORITY (Nice to Have)

### Future Work Items

- [ ] **Scale to Language Models**
  - Current: Toy style space simulation
  - Future: Actual LLM fine-tuning with Hodge critic
  - This is a follow-up paper, not current scope

- [ ] **Temporal Cohomology**
  - Extend framework to time-varying preferences
  - Detect when preferences shift (concept drift in RLHF)

- [ ] **Multi-Agent Reward Sheaves**
  - Multiple agents, each with their own reward sheaf
  - Gluing condition for coordination

### Code Quality

- [ ] **Refactor Notebooks into Package**
  - `strs/` package with modules:
    - `strs/sheaf/` - Sheaf construction and cohomology
    - `strs/geometry/` - Metric learning and geodesics
    - `strs/rl/` - SGPO algorithm
    - `strs/experiments/` - Benchmark environments

- [ ] **Unit Tests**
  - Test H¹ computation on known examples
  - Test metric singularity behavior
  - Test Hodge decomposition orthogonality

- [ ] **Documentation**
  - API documentation
  - Tutorial notebooks
  - Installation instructions

---

## Questions Requiring Human Input

### Theoretical Questions

1. **What is the correct formal statement of Theorem 3.1 (Consistency ⟺ H¹ = 0)?**
   - Current: Intuitive but informal
   - Need: Precise mathematical statement with proof sketch
   - This is core theoretical contribution

2. **How should we define the "trajectory space topology"?**
   - Options: Discrete (trajectories as points), Continuous (trajectory manifold)
   - Affects how sheaves and cohomology are computed
   - Current implementation uses discrete approximation

3. **Is the Schwarzschild-like metric the right choice for black holes?**
   - Current: g = 1/(1 - r₀/r)² inspired by general relativity
   - Alternatives: Exponential decay, learned metric
   - Need: Justification for why this form is appropriate

### Experimental Questions

4. **What baselines should we compare against?**
   - Current: PPO, CPO
   - Options: DPO, IPO, other safe RL methods
   - What's expected for ICML 2025?

5. **Are synthetic experiments sufficient or do we need real data?**
   - Toy experiments clearly demonstrate concepts
   - Real data (even small) adds credibility
   - Trade-off: Time to collect vs. paper strength

6. **Should we include a language model experiment?**
   - Current: Simulated style space
   - Could do: Small GPT-2 fine-tuning with Hodge critic
   - Risk: May not work well, distracts from core contribution

### Strategic Questions

7. **Primary framing: Safety or Consistency?**
   - Safety angle: Black holes, geodesic avoidance (practical)
   - Consistency angle: H¹ detection, cyclic preferences (theoretical)
   - Which resonates more with ICML audience?

8. **How much sheaf theory background to include?**
   - Too much: Alienates readers, takes up space
   - Too little: Core contribution unclear
   - Suggestion: Minimal in main text, full primer in appendix

---

## Dependencies and Resources

### External Dependencies
- PyTorch (have)
- NumPy, Matplotlib (have)
- Safety Gym (if scaling up)
- Gudhi or Dionysus (for persistent homology, optional)

### Computational Resources
- Current experiments: Laptop-scale
- Scaling up: May need GPU cluster
- LLM experiments: Would need significant compute

### Knowledge Resources
- Sheaf theory: ~/Documents/Knowledge/Applied Math/Topological Data Processing/
- RL: ~/Documents/Knowledge/Reinforcement Learning/
- modalsheaf library: Existing cohomology code to adapt

---

## Timeline Summary

| Week | Focus | Deliverables |
|------|-------|--------------|
| **Dec 30 - Jan 5** | Experiments | Run notebooks, collect data, generate figures |
| **Jan 6 - Jan 12** | Writing | Methods sections draft, related work |
| **Jan 13 - Jan 19** | Writing | Experiments, intro, abstract, polish |
| **Jan 20 - Jan 26** | Review | Internal review, appendix, revisions |
| **Jan 27 - Jan 30** | Submission | Final polish, submit to ICML |

---

## Changelog

- **Jan 15, 2025**: Major framework alignment update
  - Created TENSOR_RL_FOUNDATIONS.md connecting Bellman to NL state spaces
  - Added Condorcet cycle detection to HodgeCritic
  - Created world_model.py with proper Oracle/Agent separation
  - Enhanced SheafResolver with learnable restriction maps
  - Added vector reward support to RolloutBuffer
  - Updated documentation to clarify Tensor RL concepts

- **Dec 29, 2024**: Initial TODO created after methodology analysis
  - Identified theory-implementation gaps
  - Corrected three Jupyter notebooks
  - Created paper outline
  - Defined clear priorities for ICML push
