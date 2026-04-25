# Handoff 01: Directory Cleanup and Reorganization

**Priority**: HIGHEST (Upstream - must complete first)  
**Estimated Effort**: 2-3 hours  
**Type**: Filesystem refactor, documentation

---

## Context

The `/topics/high_dimensional_reward_spaces/` directory has grown organically during research and now contains 50+ items at the root level. This cluttered structure makes it difficult for new contributors to understand the project organization and creates confusion about which files are current vs. outdated.

**Current Root-Level Item Count**: ~55 files and directories  
**Target**: Logical organization with <15 root items

---

## Progress Tracking

**IMPORTANT**: Before starting this handoff, read `handoffs/00_PROGRESS_STATUS.md` to understand the current project state.

When you begin work:
1. Update the "Handoff 01" section in `00_PROGRESS_STATUS.md` with status 🟡 In Progress
2. Add start timestamp
3. Update "Current Session" section with your active task

When you complete tasks:
1. Check off completed items in the "Handoff 01" section
2. Add artifacts to "Artifacts Created"
3. Note any issues in "Issues/Notes"

When you finish or need to hand off:
1. Update status to ✅ Completed (or ⚠️ Blocked if issues)
2. Add a "Session Handoff" entry with what was done and next steps
3. Update the overall status table

---

## Current Structure Analysis

### Items to KEEP at Root
- `README.md` - Main documentation (needs update)
- `requirements.txt` - Dependencies
- `TODO.md` - Task tracking (consider archiving old items)

### Items to REORGANIZE

#### 1. Source Code → `src/`
Already exists and is well-organized with 51 items including:
- `hodge_critic.py` - Core Hodge decomposition
- `semantic_mdp_rl.py` - Semantic RL implementation
- `environments/` - Custom environments
- `scenarios/` - Test scenarios
- `simulations/` - Simulation code

**Action**: Keep as-is, add `src/README.md` documenting module structure.

#### 2. Notebooks → `notebooks/`
Already exists with:
- `modal_runner/` - Scale experiments (current)
- `gcp_runner/` - GCP experiments (older)
- `colab_*.ipynb` - Colab notebooks

**Action**: Keep as-is.

#### 3. Paper Submission → `submission/`
Already exists with LaTeX files:
- `main.tex`, `sections/`, `figures/`
- ICML style files
- Draft markdown versions

**Action**: Keep as-is.

#### 4. Documentation → `docs/`
Already exists with 27 items - needs internal cleanup:
- `RESEARCH_PROPOSAL.md` - Keep (core document)
- `LEARNING_ROADMAP.md` - Keep
- `PAPER_*.md` - May be outdated, compare with `submission/`
- Multiple overlapping docs

**Action**: Archive outdated docs, keep only current ones.

#### 5. Generated Figures → `figures/` (NEW)
Currently scattered at root level:
```
fig1_sheaf_structure.png
fig2_geometric_safety.png
fig3_hodge_decomp.png
fig4_hodge_matrix_decomposition.png
ablation_plots.png
condorcet_results.png
consistency_analysis.png
gpo_high_dim_demo.png
hodge_decomposition_2d.png
hodge_decomposition_3d.png
integrated_*.png (8 files)
projection_flow_viz.png
safety_benchmark_results.png
safety_hard_results.png
semantic_sapr_mamba.png
sheaf_zoom_viz.png
style_cycle_results.png
topology_dashboard.png
trajectory_analysis.png
```

**Action**: Move ALL `.png` files to `figures/` with subdirectories:
- `figures/paper/` - Publication-ready figures (fig1-4)
- `figures/experiments/` - Experiment result plots
- `figures/diagrams/` - Conceptual diagrams
- `figures/archive/` - Old/unused figures

#### 6. Experiment Results → `results/` (NEW)
Currently scattered:
```
ablation_results.json
condorcet_metrics.json
safety_benchmark_metrics.json
safety_hard_metrics.json
style_cycle_metrics.json
mlx_mamba_metadata.json
mlx_mamba_textworld.npz
```

**Action**: Create `results/` directory with subdirectories:
- `results/condorcet/` - Condorcet cycle experiments
- `results/safety/` - Safety benchmark results
- `results/style/` - Style experiment results
- `results/scale/` - HH-RLHF scale experiments (link from modal_runner)

#### 7. References → `references/`
Already exists with 34 items - literature, PDFs.

**Action**: Keep as-is.

#### 8. Data → `data/`
Already exists with 3 items.

**Action**: Keep as-is.

#### 9. TextWorld Games → `tw_games/`, `tw_games_generated/`
Contains 36+ JSON game files for experiments.

**Action**: Move to `data/textworld/` to consolidate data assets.

#### 10. Old Paper Drafts → `archive/` (NEW)
```
The Shape of Good Behavior - A Geometric Approach to AI Alignment/
Evaluating RL Paper Impact.pdf
```

**Action**: Create `archive/` for superseded materials.

#### 11. Jupyter Notebooks at Root → Move to `notebooks/`
```
AI_Safety_Benchmark.ipynb
Condorcet_Cycle_Experiment.ipynb
LLM_Style_Tuning_Experiment.ipynb
```

**Action**: Move to `notebooks/legacy/` (superseded by `src/` scripts).

#### 12. Virtual Environments
```
safety_gym_venv/
apps/
```

**Action**: Add to `.gitignore`, document in README.

---

## Target Structure

```
high_dimensional_reward_spaces/
├── README.md                    # Updated with project overview
├── requirements.txt             # Dependencies
├── TODO.md                      # Task tracking
├── .gitignore                   # Updated
│
├── src/                         # Source code (unchanged)
│   ├── README.md               # NEW: Module documentation
│   ├── hodge_critic.py
│   ├── semantic_mdp_rl.py
│   ├── environments/
│   ├── scenarios/
│   └── ...
│
├── notebooks/                   # Jupyter notebooks
│   ├── modal_runner/           # Current scale experiments
│   ├── gcp_runner/             # GCP experiments
│   ├── colab_*.ipynb           # Colab notebooks
│   └── legacy/                 # OLD: Root-level notebooks moved here
│
├── submission/                  # ICML paper (unchanged)
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── ...
│
├── docs/                        # Documentation
│   ├── RESEARCH_PROPOSAL.md
│   ├── LEARNING_ROADMAP.md
│   ├── ALIGNMENT_GUARANTEES.md
│   └── archive/                # OLD: Outdated docs
│
├── figures/                     # NEW: All generated figures
│   ├── paper/                  # Publication figures
│   ├── experiments/            # Experiment plots
│   ├── diagrams/               # Conceptual diagrams
│   └── archive/                # Unused/old figures
│
├── results/                     # NEW: Experiment results
│   ├── condorcet/
│   ├── safety/
│   ├── style/
│   └── scale/                  # Symlink to modal_runner/results
│
├── data/                        # Data assets
│   ├── textworld/              # TW games (moved from tw_games/)
│   └── ...
│
├── references/                  # Literature (unchanged)
│
├── handoffs/                    # NEW: These handoff documents
│
└── archive/                     # NEW: Old/superseded materials
    ├── old_paper_drafts/
    └── old_experiments/
```

---

## Detailed Tasks

### Task 1: Create New Directories
```bash
cd /Users/Michaellee/Documents/Runes/ai_research/topics/high_dimensional_reward_spaces

mkdir -p figures/{paper,experiments,diagrams,archive}
mkdir -p results/{condorcet,safety,style,scale}
mkdir -p notebooks/legacy
mkdir -p docs/archive
mkdir -p archive/{old_paper_drafts,old_experiments}
mkdir -p data/textworld
mkdir -p handoffs
```

### Task 2: Move Figures
```bash
# Paper figures
mv fig1_sheaf_structure.png figures/paper/
mv fig2_geometric_safety.png figures/paper/
mv fig3_hodge_decomp.png figures/paper/
mv fig4_hodge_matrix_decomposition.png figures/paper/

# Experiment plots
mv ablation_plots.png figures/experiments/
mv condorcet_results.png figures/experiments/
mv safety_benchmark_results.png figures/experiments/
mv safety_hard_results.png figures/experiments/
mv style_cycle_results.png figures/experiments/
mv trajectory_analysis.png figures/experiments/
mv consistency_analysis.png figures/experiments/

# Diagrams
mv gpo_high_dim_demo.png figures/diagrams/
mv hodge_decomposition_2d.png figures/diagrams/
mv hodge_decomposition_3d.png figures/diagrams/
mv integrated_*.png figures/diagrams/
mv projection_flow_viz.png figures/diagrams/
mv semantic_sapr_mamba.png figures/diagrams/
mv sheaf_zoom_viz.png figures/diagrams/
mv topology_dashboard.png figures/diagrams/
```

### Task 3: Move Results
```bash
mv condorcet_metrics.json results/condorcet/
mv safety_benchmark_metrics.json results/safety/
mv safety_hard_metrics.json results/safety/
mv ablation_results.json results/safety/
mv style_cycle_metrics.json results/style/

# Create symlink to modal_runner results
ln -s ../notebooks/modal_runner/results results/scale
```

### Task 4: Move TextWorld Data
```bash
mv tw_games/* data/textworld/
mv tw_games_generated/* data/textworld/generated/
rmdir tw_games tw_games_generated
```

### Task 5: Move Legacy Notebooks
```bash
mv AI_Safety_Benchmark.ipynb notebooks/legacy/
mv Condorcet_Cycle_Experiment.ipynb notebooks/legacy/
mv LLM_Style_Tuning_Experiment.ipynb notebooks/legacy/
```

### Task 6: Archive Old Materials
```bash
mv "The Shape of Good Behavior - A Geometric Approach to AI Alignment" archive/old_paper_drafts/
mv "Evaluating RL Paper Impact.pdf" archive/old_paper_drafts/
mv mlx_mamba_*.* archive/old_experiments/

# Archive outdated docs
mv docs/PAPER_*.md docs/archive/  # After verifying not needed
mv docs/COLAB_HANDOFF_PLAN.md docs/archive/
mv docs/STREAMLIT_APP_V2.md docs/archive/
```

### Task 7: Update .gitignore
Add:
```
# Virtual environments
safety_gym_venv/
.venv/
apps/

# Large data files
*.npz
*.parquet

# OS files
.DS_Store
```

### Task 8: Create/Update README.md

The new README should include:
1. **Project Title**: Sheaf-Theoretic Reward Spaces for Safe RL
2. **Overview**: 2-3 paragraph description
3. **Key Contributions**: Bullet list
4. **Directory Structure**: Tree diagram
5. **Quick Start**: How to run experiments
6. **Paper**: Link to submission/main.tex
7. **Dependencies**: Reference requirements.txt
8. **Citation**: BibTeX placeholder

### Task 9: Create src/README.md
Document the module structure:
- `hodge_critic.py` - Core Hodge decomposition and topological gradients
- `semantic_mdp_rl.py` - Semantic MDP with natural language states
- `embedding_topology_analyzer.py` - Interpretability tools
- `environments/` - Custom RL environments
- etc.

### Task 10: Update Paths in Code
After moving files, grep for hardcoded paths and update:
```bash
grep -r "fig1_sheaf" src/ notebooks/ submission/
grep -r "condorcet_metrics" src/ notebooks/
```

---

## Verification Checklist

- [ ] Root directory has ≤15 items
- [ ] All `.png` files in `figures/`
- [ ] All `.json` result files in `results/`
- [ ] No orphaned/outdated files at root
- [ ] `README.md` updated with new structure
- [ ] `.gitignore` updated
- [ ] All symlinks working
- [ ] Paper compilation still works (`pdflatex main.tex`)
- [ ] Notebooks can still find their data
- [ ] **Progress tracking**: Updated `00_PROGRESS_STATUS.md` with completion status

---

## Dependencies

**Downstream handoffs affected**:
- Handoff 02 (Paper Restructuring) - paths to figures
- Handoff 03 (Experiments) - paths to results
- Handoff 07 (Visualization App) - paths to data

**Must complete before**: All other handoffs

---

## Notes

- Back up the directory before making changes: `cp -r high_dimensional_reward_spaces high_dimensional_reward_spaces_backup`
- Use `git mv` instead of `mv` to preserve history
- Test paper compilation after moving figures
- Update any absolute paths in notebooks
