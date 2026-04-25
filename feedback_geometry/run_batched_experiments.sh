#!/bin/bash
# Run experiments in 5 batches of 10 seeds each on Modal GPU

MODAL="/Users/Michaellee/Documents/Runes/ai_research/topics/feedback_geometry/venv/bin/modal"

echo "============================================================"
echo "Running Batched Experiments on Modal GPU"
echo "============================================================"
echo ""
echo "Optimizations applied:"
echo "  - Timeout: 8 hours (was 2 hours)"
echo "  - Batch size: 10 seeds per batch (5 batches total)"
echo "  - Experiment C episodes: 150 (was 300)"
echo "  - Experiment C max_steps: 200 (was 100)"
echo "  - Experiment C trap_reward: 5.0 (HIGHLY TEMPTING - learn to avoid after catastrophe)"
echo "  - Conformal sharpness: 4.0 (was 2.0)"
echo ""

# Run 5 batches sequentially
for batch in {0..4}; do
    echo "============================================================"
    echo "BATCH $((batch + 1))/5: Seeds $((batch * 10 + 1))-$((batch * 10 + 10))"
    echo "============================================================"
    
    $MODAL run modal_runner_v2.py \
        --experiment both \
        --mode full \
        --batch-size 10 \
        --batch-num $batch
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "ERROR: Batch $((batch + 1)) failed with exit code $EXIT_CODE"
        echo "Continuing to next batch..."
    else
        echo ""
        echo "✓ Batch $((batch + 1)) completed successfully"
    fi
    
    echo ""
done

echo "============================================================"
echo "ALL BATCHES COMPLETE"
echo "============================================================"
echo ""
echo "Combining results..."

# Combine CSV files
python3 << 'EOF'
import pandas as pd
from pathlib import Path

results_dir = Path("results/modal")

# Combine Experiment A results
exp_a_files = sorted(results_dir.glob("experiment_a_batch*.csv"))
if exp_a_files:
    dfs_a = [pd.read_csv(f) for f in exp_a_files]
    combined_a = pd.concat(dfs_a, ignore_index=True)
    combined_a.to_csv(results_dir / "experiment_a_combined.csv", index=False)
    print(f"Combined {len(exp_a_files)} Experiment A batches -> experiment_a_combined.csv")
    print(f"  Total seeds: {combined_a['seed'].nunique()}")

# Combine Experiment C results
exp_c_files = sorted(results_dir.glob("experiment_c_batch*.csv"))
if exp_c_files:
    dfs_c = [pd.read_csv(f) for f in exp_c_files]
    combined_c = pd.concat(dfs_c, ignore_index=True)
    combined_c.to_csv(results_dir / "experiment_c_combined.csv", index=False)
    print(f"Combined {len(exp_c_files)} Experiment C batches -> experiment_c_combined.csv")
    print(f"  Total seeds: {combined_c['seed'].nunique()}")
EOF

echo ""
echo "Results saved to results/modal/"
echo "  - experiment_a_combined.csv"
echo "  - experiment_c_combined.csv"
