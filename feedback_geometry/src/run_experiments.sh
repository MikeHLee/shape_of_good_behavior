#!/bin/bash
# Run script for Feedback Geometry experiments
# Usage: ./run_experiments.sh [quick|full|h1|sandbagging|modal|modal-full]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use the project's virtual environment
VENV_PYTHON="/Users/Michaellee/Documents/Runes/ai_research/topics/high_dimensional_reward_spaces/safety_gym_venv/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Warning: Virtual environment not found at $VENV_PYTHON"
    echo "Using system Python"
    VENV_PYTHON="python3"
fi

# Create results directories
mkdir -p ../results/h1_exploitation
mkdir -p ../results/sandbagging_v2

case "${1:-quick}" in
    quick)
        echo "Running QUICK experiments locally (5 seeds each)..."
        echo ""
        echo "=== H¹ Exploitation Experiment ==="
        $VENV_PYTHON h1_reward_hacking_experiment.py --quick
        echo ""
        echo "=== Sandbagging Trap v2 ==="
        $VENV_PYTHON sandbagging_experiment_v2.py --quick
        ;;
    
    full)
        echo "Running FULL experiments locally (50 seeds each)..."
        echo "This will take several hours."
        echo ""
        echo "=== H¹ Exploitation Experiment ==="
        $VENV_PYTHON h1_reward_hacking_experiment.py --seeds 50
        echo ""
        echo "=== Sandbagging Trap v2 ==="
        $VENV_PYTHON sandbagging_experiment_v2.py --seeds 50
        ;;
    
    h1)
        echo "Running H¹ Exploitation Experiment locally..."
        SEEDS="${2:-50}"
        $VENV_PYTHON h1_reward_hacking_experiment.py --seeds $SEEDS
        ;;
    
    sandbagging)
        echo "Running Sandbagging Trap v2 locally..."
        SEEDS="${2:-50}"
        $VENV_PYTHON sandbagging_experiment_v2.py --seeds $SEEDS
        ;;
    
    modal)
        echo "Running experiments on Modal GPU (L4, 5 seeds quick test)..."
        modal run modal_runner.py
        echo ""
        echo "To download results: ./download_modal_results.sh"
        ;;
    
    modal-h1)
        echo "Running H¹ experiment on Modal GPU..."
        SEEDS="${2:-50}"
        modal run modal_runner.py::run_h1_experiment --num-seeds $SEEDS
        ;;
    
    modal-sandbagging)
        echo "Running Sandbagging experiment on Modal GPU..."
        SEEDS="${2:-50}"
        modal run modal_runner.py::run_sandbagging_experiment --num-seeds $SEEDS
        ;;
    
    modal-full)
        echo "Running FULL experiments on Modal GPU (L4, 50 seeds)..."
        echo "This will take ~2-3 hours on GPU."
        modal run modal_runner.py::run_all_experiments --num-seeds 50
        echo ""
        echo "To download results: ./download_modal_results.sh"
        ;;
    
    download)
        echo "Downloading results from Modal..."
        ./download_modal_results.sh
        ;;
    
    *)
        echo "Usage: $0 [command] [seeds]"
        echo ""
        echo "Local execution:"
        echo "  quick           Run both experiments locally with 5 seeds (fast test)"
        echo "  full            Run both experiments locally with 50 seeds"
        echo "  h1 [N]          Run H¹ experiment locally with N seeds (default: 50)"
        echo "  sandbagging [N] Run sandbagging experiment locally with N seeds"
        echo ""
        echo "Modal GPU execution:"
        echo "  modal           Quick test on Modal GPU (5 seeds)"
        echo "  modal-h1 [N]    Run H¹ experiment on Modal with N seeds"
        echo "  modal-sandbagging [N]  Run sandbagging on Modal with N seeds"
        echo "  modal-full      Run both experiments on Modal GPU (50 seeds)"
        echo "  download        Download results from Modal volume"
        exit 1
        ;;
esac

echo ""
echo "Done! Results saved to:"
echo "  - ../results/h1_exploitation/"
echo "  - ../results/sandbagging_v2/"
