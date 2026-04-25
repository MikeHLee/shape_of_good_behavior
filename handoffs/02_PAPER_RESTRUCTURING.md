# Handoff 02: Paper Restructuring - Intuition-First, Formalisms to Appendix

**Priority**: HIGH  
**Estimated Effort**: 4-6 hours  
**Type**: Writing, LaTeX editing  
**Dependencies**: Handoff 01 (directory cleanup) should be complete

---

## Context

The ICML 2026 submission currently has:
- **Main body**: ~7 pages (limit: 8 pages)
- **Abstract**: ~200 words (guideline: 4-6 sentences, ~100-150 words)
- **Appendix**: 1 page (unlimited)

### ICML Requirements (from `example_paper.tex`)
- Main body: **8 pages maximum** (excluding references, appendix, impact statement)
- Abstract: **4-6 sentences**, single paragraph
- References: Unlimited pages
- Appendix: Unlimited pages
- Impact Statement: Required, does not count toward limit

### Current Issues
1. Abstract is too long (~200 words vs. recommended ~100-150)
2. Main body is dense with formalisms that could move to appendix
3. Needs more intuitive explanations before formal definitions
4. Missing motivating questions as exposition tool

---

## Progress Tracking

**IMPORTANT**: Before starting this handoff, read `handoffs/00_PROGRESS_STATUS.md` to understand the current project state.

When you begin work:
1. Update the "Handoff 02" section in `00_PROGRESS_STATUS.md` with status 🟡 In Progress
2. Add start timestamp
3. Update "Current Session" section with your active task

When you complete tasks:
1. Check off completed items in the "Handoff 02" section
2. Add artifacts to "Artifacts Created"
3. Note any issues in "Issues/Notes"

When you finish or need to hand off:
1. Update status to ✅ Completed (or ⚠️ Blocked if issues)
2. Add a "Session Handoff" entry with what was done and next steps
3. Update the overall status table

---

## Key Files

```
submission/
├── main.tex                    # Main document
├── sections/
│   ├── introduction.tex        # ~1 page
│   ├── background.tex          # ~0.75 page  
│   ├── method.tex              # ~2.5 pages (NEEDS WORK)
│   ├── experiments.tex         # ~2 pages
│   ├── related_work.tex        # ~0.75 page
│   ├── conclusion.tex          # ~0.5 page
│   └── appendix.tex            # ~1 page (EXPAND)
├── references.bib
└── figures/
```

---

## Task 1: Shorten and Focus the Abstract

### Current Abstract (~200 words)
Too long, too dense. Contains implementation details that belong in main text.

### Target Abstract (~120 words, 4-6 sentences)
Structure:
1. **Problem**: RLHF collapses preferences to scalar, loses structure
2. **Key insight**: Sheaf theory captures local-global consistency
3. **Method summary**: Hodge decomposition + geometric safety
4. **Main result**: 8× better safety-performance on benchmarks, validated on 50k HH-RLHF

### Suggested Rewrite
```latex
\begin{abstract}
Reinforcement Learning from Human Feedback (RLHF) collapses rich human preferences into scalar rewards, losing critical structure: cyclic preferences become invisible, and safety constraints reduce to soft penalties. We propose Sheaf-Theoretic Reward Spaces (STRS), modeling rewards as sections of a sheaf where the first cohomology group $H^1$ detects global inconsistencies like Condorcet cycles. The Hodge decomposition separates learnable gradients from irreducible cycles, while geometric ``black holes'' enforce hard safety constraints through metric singularities. Our Sheaf-Geodesic Policy Optimization (SGPO) algorithm achieves 8$\times$ better returns than Constrained Policy Optimization while maintaining safety on cyclic preference benchmarks. Topology mining on 50,000 Anthropic HH-RLHF examples reveals that 33\% of prompts lie in high-risk regions of local preference inconsistency.
\end{abstract}
```

---

## Task 2: Move Formalisms to Appendix

### Items to Move from `method.tex` → `appendix.tex`

#### 2.1 Definition 1 (Trajectory Space) - Lines 10-18
**Keep in main**: Intuitive 1-sentence description
**Move to appendix**: Full cell complex definition

```latex
% IN MAIN TEXT (method.tex)
We model the trajectory space as a cell complex $X$ where states are vertices, transitions are edges, and local consistency checks form faces. (See Appendix~\ref{app:trajectory_space} for the formal construction.)

% IN APPENDIX (appendix.tex)
\subsection{Trajectory Space Construction}
\label{app:trajectory_space}
[Full Definition 1 content here]
```

#### 2.2 Definition 2 (Reward Sheaf) - Lines 20-23
**Keep in main**: Intuitive explanation (already at line 25)
**Move to appendix**: Formal definition with restriction maps

#### 2.3 Theorem 1 (Consistency Criterion) - Lines 31-37
**Keep in main**: Statement in plain language
**Move to appendix**: Formal statement with Čech cohomology notation, proof

```latex
% IN MAIN TEXT
\textbf{Key Result:} Local feedback samples are globally consistent if and only if they can be ``glued'' together---formally, the first cohomology class $H^1$ vanishes. Non-zero $H^1$ signals fundamental preference cycles that no scalar reward can represent. (Theorem~\ref{thm:consistency_full} in Appendix.)
```

#### 2.4 Corollary 1 (Condorcet Cycles) - Lines 41-44
**Keep in main**: The intuitive statement with A > B > C > A example
**Move to appendix**: The line integral argument

#### 2.5 Theorem 2 (Hodge Decomposition) - Lines 50-62
**Keep in main**: Intuitive decomposition (gradient + curl + harmonic)
**Move to appendix**: Full orthogonality proof, discrete Hodge theory

#### 2.6 Proposition 1 (Metric Singularity) - Lines 91-98
**Keep in main**: Intuitive "infinite distance" explanation
**Move to appendix**: The conformal factor derivation

#### 2.7 Theorem 3 (Geodesic Avoidance) - Lines 106-109
**Keep in main**: Plain language safety guarantee
**Move to appendix**: Proof sketch

#### 2.8 Equations 1-6 
Keep key equations but move derivations to appendix.

---

## Task 3: Add Motivating Questions

Insert questions throughout the paper to guide reader intuition.

### In Introduction
After problem statement, add:
> **Question**: *If human preferences can be cyclic (A preferred to B, B to C, C to A), how can we learn from them without forcing a false consistency?*

### In Method (Sheaf Section)
Before Definition 1:
> **Question**: *What mathematical structure captures the idea that "local evaluations must agree when combined into global assessments"?*

### In Method (Safety Section)
Before Definition 2 (Black Hole):
> **Question**: *How can we guarantee an agent will **never** enter a forbidden state, not just "probably" avoid it?*

### In Method (SGPO Section)
Before Algorithm 1:
> **Question**: *If the reward landscape has both consistent gradients and irreducible cycles, how should an agent navigate?*

---

## Task 4: Intuitive Explanations for Key Concepts

Add brief intuitive explanations BEFORE each formal concept.

### 4.1 Probability Simplex (background.tex)
```latex
% ADD before any formal use
The \textbf{probability simplex} is simply the space of all valid probability distributions---a triangle in 3D (where corners are "100\% option A", etc.), or higher-dimensional analogs.
```

### 4.2 Bradley-Terry Cross Entropy (experiments.tex or background.tex)
```latex
% ADD before use
\textbf{Bradley-Terry} models pairwise preferences as: "the probability A beats B depends on their relative strengths." The cross-entropy loss trains a model to predict these win probabilities.
```

### 4.3 H¹ Cohomology (method.tex)
```latex
% ADD before Theorem 1
Intuitively, \textbf{$H^1$ cohomology} measures ``holes'' in preference space. If preferences form a consistent ranking, $H^1 = 0$ (no holes). If preferences cycle (A > B > C > A), there's a ``hole'' and $H^1 \neq 0$.
```

### 4.4 Hodge Decomposition (method.tex)
```latex
% ADD before Theorem 2
Just as any vector field can be split into curl-free (flows from source to sink) and divergence-free (rotational) parts, \textbf{Hodge decomposition} splits preferences into:
\begin{itemize}
    \item \textbf{Gradient}: Consistent preferences ($A > B > C$, learnable as a value function)
    \item \textbf{Harmonic}: Fundamental cycles ($A > B > C > A$, cannot be ``unrolled'')
\end{itemize}
```

### 4.5 Cochain Complexes (appendix.tex)
```latex
% In appendix only
A \textbf{cochain complex} is a sequence of vector spaces connected by ``boundary'' maps. Think: 0-cochains assign values to points, 1-cochains to edges, 2-cochains to faces. The boundary maps check consistency across dimensions.
```

### 4.6 Hodge-Bellman Error (method.tex)
```latex
% ADD before Equation 5
The \textbf{Hodge-Bellman error} extends the standard TD error to account for preference cycles. Instead of forcing $r = V(s') - V(s)$ (which fails for cycles), we learn both a potential $V$ and a ``circulation'' $\omega$ that absorbs the cyclic component.
```

### 4.7 Scalar Hypothesis (introduction.tex)
```latex
% ADD early in intro
We call the standard RLHF assumption the \textbf{``scalar hypothesis''}: that human preferences can always be captured by a single number per trajectory. This paper shows why this hypothesis fails and how to move beyond it.
```

---

## Task 5: Expand Appendix

Current appendix is ~1 page. Expand to 3-5 pages with:

### A. Full Proofs (2 pages)
- Theorem 1 (Consistency) - Complete proof
- Theorem 2 (Hodge) - Orthogonality proof
- Theorem 3 (Avoidance) - Geodesic argument
- Proposition 1 (Singularity) - Metric derivation

### B. Mathematical Background (1 page)
- Sheaf theory primer (for readers unfamiliar)
- Čech cohomology definition
- Discrete Hodge theory on graphs

### C. Implementation Details (1 page)
- Hodge Critic architecture
- Metric learning network
- Hyperparameters for all experiments

### D. Additional Experiments (if space)
- Ablation studies (already have data)
- Sensitivity analysis

---

## Task 6: Add Visualizations

Replace or supplement formal definitions with diagrams.

### Figure: Sheaf Intuition
Show local sections on overlapping regions, gluing condition.
**Location**: After Definition 1

### Figure: Hodge Decomposition Example
3-node preference graph with gradient and harmonic components separated.
**Location**: After Theorem 2

### Figure: Black Hole Metric
Show how geodesics bend around singularity.
**Location**: After Definition 2 (Black Hole)

---

## Verification Checklist

- [ ] Abstract is 4-6 sentences, ~120 words
- [ ] Main body ≤ 8 pages after changes
- [ ] All formal definitions have intuitive lead-ins
- [ ] Proofs moved to appendix with main-text references
- [ ] Motivating questions added (at least 3)
- [ ] Key terms have lay explanations (see Task 4 list)
- [ ] Appendix expanded to 3+ pages
- [ ] Paper still compiles: `pdflatex main.tex && bibtex main && pdflatex main.tex`
- [ ] **Progress tracking**: Updated `00_PROGRESS_STATUS.md` with completion status

---

## OpenReview Submission Format

For OpenReview submission:
1. **PDF**: Upload compiled `main.pdf`
2. **Abstract**: Copy text from `\begin{abstract}...\end{abstract}` 
3. **Keywords**: "Reinforcement Learning, Human Feedback, Sheaf Theory, Algebraic Topology, Safe RL, Policy Optimization"
4. **Supplementary**: Can include code as ZIP if desired

Note: OpenReview uses plain text for the abstract field. Remove LaTeX commands like `\textbf{}` and convert math to Unicode or plain text where possible.

---

## Dependencies

**Requires**: Handoff 01 complete (figure paths may change)  
**Affects**: Handoff 05 (intuitive expansions build on this structure)

---

## Notes

- Keep a backup of current `method.tex` before major edits
- Use `\input{sections/appendix_proofs}` to organize appendix content
- Consider adding a "Notation" table in appendix for symbol reference
