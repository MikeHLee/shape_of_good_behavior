# Experiment A: Pre-filtered Reward Models Analysis

## Overview
Experiment A tests whether filtering out invalid (within-context) preference cycles improves reward model robustness by reducing exploitation potential.

## Methodology

### Data
- **Source**: Anthropic HH-RLHF dataset (160k examples, sampled 5000 per seed)
- **Embedding**: Sentence-Transformers `all-MiniLM-L6-v2` (384-dim)
- **Context**: Prompt hash (exact match) groups preferences by prompt

### H¹ Computation
1. **Marginal H¹**: Cyclic preference component across all items
2. **Conditional H¹**: Weighted average of per-context H¹ values
3. **Valid Contextual H¹**: Variation explained by context (marginal - conditional)

### Filtering Strategy
- Items grouped by context (prompt)
- Contexts with H¹ > threshold (0.8) marked as "invalid"
- **Raw set**: Random sample of all items (size = n_filtered)
- **Filtered set**: Items from low-H¹ contexts (size = n_filtered)
- Both trained on equal sample counts for fair comparison

### Reward Model Training
- **Architecture**: 2-layer MLP (384 → 128 → 1)
- **Loss**: Bradley-Terry preference loss: `-log P(chosen > rejected)`
- **Training**: 50 epochs, batch size 32, Adam optimizer (lr=1e-4)
- **Test set**: 200 held-out examples from raw set

### Evaluation Metrics

**Accuracy**:
- Fraction of test pairs where model predicts chosen > rejected
- Measures basic preference learning capability
- Range: [0, 1]

**Exploitation Rate**:
- Fraction of test pairs where model strongly prefers rejected (P < 0.3)
- Indicates reward hacking potential
- **Key metric**: Lower exploitation = more robust model
- Range: [0, 1]

## Expected Results

### Hypothesis
H¹-filtered training should reduce exploitation by removing inconsistent preference signals.

### Predicted Outcomes
| Metric | Raw Model | Filtered Model | Expected Difference |
|--------|-----------|----------------|-------------------|
| Accuracy | ~70-80% | ~70-80% | Minimal (both learn preferences) |
| Exploitation | Higher | Lower | **Filtered should be 10-20% lower** |

### Interpretation
- If filtered exploitation << raw exploitation: H¹ filtering works
- If similar: Invalid cycles don't contribute to hacking
- If filtered > raw: Context filtering removes important signal

## Results from Modal Run

### Summary Statistics (50 seeds)
```
Raw Model:      Accuracy=?, Exploitation=?
Filtered Model: Accuracy=?, Exploitation=?
Exploitation Reduction: ?
```

### Per-Seed Breakdown
Each seed records:
- `marginal_h1`: Total cyclic preference component
- `conditional_h1`: Within-context cycles
- `n_train`: Training set size (normalized)
- `raw_accuracy`: Test accuracy on raw model
- `raw_exploitation`: Exploitation rate on raw model
- `filtered_accuracy`: Test accuracy on filtered model
- `filtered_exploitation`: Exploitation rate on filtered model

## Key Insights

### What This Measures
1. **Reward hacking vulnerability**: Does H¹ in preferences lead to exploitable models?
2. **Context-conditional validity**: Are invalid cycles truly context-specific?
3. **Filtering effectiveness**: Can we improve robustness by removing invalid cycles?

### Critical Interpretation Points
- **Zero exploitation**: Suggests reward models are naturally robust (or test set too small)
- **Large reduction**: Strong evidence that H¹ drives hacking
- **Accuracy preservation**: Filtering doesn't hurt learning (good sign)
- **H¹ correlation**: Seeds with high H¹ should show larger exploitation differences

## Next Steps
1. Retrieve results.json from Modal volume
2. Plot exploitation vs. H¹ magnitude (correlation analysis)
3. Compute 95% CI for exploitation reduction
4. Statistical test (Welch's t-test) for significance
5. Analyze per-context H¹ distribution

## Connection to Experiment C
Experiment A validates that H¹ filtering works in isolation (reward models).
Experiment C applies this to policy optimization (SGPO_ANIS_CCHC advantage computation).
