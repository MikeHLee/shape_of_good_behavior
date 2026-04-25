#!/bin/bash
# Download GeoDPO experiment results from Modal volume
# Usage: ./download_results.sh [output_dir]

set -e

OUTPUT_DIR="${1:-./results}"
VOLUME_NAME="geodpo-data"
# In the container, we mount to /data, so files are at the root of the volume.
# When accessing via CLI, we shouldn't use /data prefix.
VOLUME_PATH=""

echo "============================================================"
echo "Downloading GeoDPO Results from Modal"
echo "============================================================"
echo "Volume: $VOLUME_NAME"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

MODAL_CMD="/Users/Michaellee/Documents/Runes/ai_research/topics/high_dimensional_reward_spaces/safety_gym_venv/.venv/bin/modal"

echo "Listing available files..."
$MODAL_CMD volume ls "$VOLUME_NAME" "$VOLUME_PATH" || {
    echo "❌ Volume not found or error accessing it. Run the experiments first:"
    echo "   $MODAL_CMD run geodpo_experiments.py --samples 50000 --steps 50"
    exit 1
}

echo ""
echo "Downloading files..."

# Download each file type separately to handle missing files gracefully
for file in topology_metadata.parquet analysis_report.csv analysis_manifold.png \
            high_dim_style_metrics.json high_dim_style_results.png \
            safety_benchmark_SafetyPointGoal1-v0.json safety_benchmark_SafetyCarGoal1-v0.json \
            safety_gym_reaching_results.csv safety_gym_navigation_results.csv safety_gym_calibration.csv \
            ethical_scenarios_trained_policies.json ethical_scenarios_table3.csv ethical_scenarios_summary.csv \
            ethical_scenarios.parquet ablation_study.parquet ablation_study.csv \
            condorcet_benchmark.json condorcet_benchmark.csv \
            viz_embeddings.json semantic_mdp_summary.csv semantic_mdp_evaluation.parquet \
            comparative_summary.csv comparative_analysis.parquet \
            dangerous_cohomology.parquet condorcet_cycles.json multi_source_topology.parquet \
            black_holes.json enhanced_gpo_black_holes.json \
            finetuned_critic_summary.csv finetuned_critic_analysis.csv \
            evaluator_results_finetuned.csv evaluator_training_data.json; do
    echo "  → $file"
    if [ -z "$VOLUME_PATH" ]; then
        REMOTE_PATH="$file"
    else
        REMOTE_PATH="$VOLUME_PATH/$file"
    fi
    $MODAL_CMD volume get --force "$VOLUME_NAME" "$REMOTE_PATH" "$OUTPUT_DIR/" || echo "    (not found, skipping)"
done

# Download checkpoint directories if exists
for dir in geodpo_checkpoints clipped_gpo_checkpoints cpo_initialized_gpo_checkpoints \
           enhanced_gpo_checkpoints ppo_model cpo_model evaluator_model; do
    echo "  → $dir/"
    if [ -z "$VOLUME_PATH" ]; then
        REMOTE_PATH="$dir"
    else
        REMOTE_PATH="$VOLUME_PATH/$dir"
    fi
    $MODAL_CMD volume get --force "$VOLUME_NAME" "$REMOTE_PATH" "$OUTPUT_DIR/" || echo "    (not found, skipping)"
done

echo ""
echo "============================================================"
echo "✅ Download complete!"
echo "============================================================"
echo ""
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR" 2>/dev/null || true
