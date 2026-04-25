#!/bin/bash
# Run final tuned experiments in parallel

VENV="/Users/Michaellee/Documents/Runes/ai_research/topics/feedback_geometry/venv/bin/python"

echo "Starting final experiments with tuned hyperparameters..."
echo "============================================================"
echo ""
echo "Tuning applied:"
echo "  - Experiment C: beta=4.0 (was 2.0), warmup=10 (was 30), trap_reward=3.0 (was 5.0)"
echo "  - Experiment A: Using real HH-RLHF data"
echo ""

# Create output directory
mkdir -p results/final

# Run Experiment A with HH-RLHF in background
echo "[1/2] Starting Experiment A (HH-RLHF, 50 seeds)..."
$VENV src/run_experiments_modal.py \
    --mode full \
    --experiment A \
    --local \
    > results/final/experiment_a_hhrlhf.log 2>&1 &
PID_A=$!
echo "  Started (PID: $PID_A)"

# Run Experiment C with tuned hyperparameters in background
echo "[2/2] Starting Experiment C (Tuned, 50 seeds)..."
$VENV src/run_experiments_modal.py \
    --mode full \
    --experiment C \
    --local \
    --use-synthetic \
    > results/final/experiment_c_tuned.log 2>&1 &
PID_C=$!
echo "  Started (PID: $PID_C)"

echo ""
echo "Both experiments running in parallel..."
echo "  Experiment A (HH-RLHF): PID $PID_A"
echo "  Experiment C (Tuned):   PID $PID_C"
echo ""
echo "Monitor progress:"
echo "  tail -f results/final/experiment_a_hhrlhf.log"
echo "  tail -f results/final/experiment_c_tuned.log"
echo ""
echo "Waiting for completion..."

# Wait for both
wait $PID_A
STATUS_A=$?
echo "Experiment A completed (exit code: $STATUS_A)"

wait $PID_C
STATUS_C=$?
echo "Experiment C completed (exit code: $STATUS_C)"

echo ""
echo "============================================================"
echo "FINAL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to:"
ls -lh results/corrected_v2/*.csv | tail -2

exit 0
