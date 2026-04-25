#!/bin/bash
# Master Runner for GeoDPO Scale Experiments
# Usage: ./run_experiments.sh [local|gcp] [--samples N] [--steps N]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-local}"
SAMPLES="${SAMPLES:-50000}"
STEPS="${STEPS:-50}"

# Parse additional args
shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples) SAMPLES="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "GeoDPO Scale Experiments"
echo "========================================"
echo "Mode: $MODE"
echo "Samples: $SAMPLES"
echo "Training Steps: $STEPS"
echo "========================================"

run_local() {
    echo ""
    echo "=== Step 1: Topology Mining ==="
    python "$SCRIPT_DIR/01_topology_mining.py" \
        --samples "$SAMPLES" \
        --output topology_metadata.parquet
    
    echo ""
    echo "=== Step 2: GeoDPO Training ==="
    python "$SCRIPT_DIR/02_geodpo_training.py" \
        --topology topology_metadata.parquet \
        --samples "$SAMPLES" \
        --steps "$STEPS" \
        --output ./geodpo_checkpoints
    
    echo ""
    echo "=== Step 3: Analysis ==="
    python "$SCRIPT_DIR/03_analysis.py" \
        --topology topology_metadata.parquet \
        --adapter ./geodpo_checkpoints \
        --output-prefix results
    
    echo ""
    echo "=== Complete ==="
    echo "Outputs:"
    ls -la topology_metadata.parquet results_*.csv results_*.png 2>/dev/null || echo "  (check current directory)"
}

run_gcp() {
    echo ""
    echo "=== Creating GCP VM ==="
    "$SCRIPT_DIR/setup_gcp_vm.sh" create
    
    echo ""
    echo "=== Uploading Files ==="
    "$SCRIPT_DIR/setup_gcp_vm.sh" scp-up
    
    echo ""
    echo "=== Running Experiments on VM ==="
    # Run with environment variables
    gcloud compute ssh geodpo-experiments --zone=us-central1-a --command="
        cd ~/experiments
        export SAMPLES=$SAMPLES
        export STEPS=$STEPS
        
        echo '=== Step 1: Topology Mining ==='
        python 01_topology_mining.py --samples $SAMPLES --output topology_metadata.parquet 2>&1 | tee run_01.log
        
        echo '=== Step 2: GeoDPO Training ==='
        python 02_geodpo_training.py --topology topology_metadata.parquet --samples $SAMPLES --steps $STEPS 2>&1 | tee run_02.log
        
        echo '=== Step 3: Analysis ==='
        python 03_analysis.py --topology topology_metadata.parquet --output-prefix results 2>&1 | tee run_03.log
        
        echo '=== Done ==='
        ls -la *.parquet *.csv *.png 2>/dev/null
    "
    
    echo ""
    echo "=== Downloading Results ==="
    "$SCRIPT_DIR/setup_gcp_vm.sh" scp-down
    
    echo ""
    echo "=== Cleaning Up VM (to save costs) ==="
    read -p "Delete VM? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        "$SCRIPT_DIR/setup_gcp_vm.sh" delete
    else
        echo "VM left running. Remember to delete with: ./setup_gcp_vm.sh delete"
    fi
}

case "$MODE" in
    local)
        run_local
        ;;
    gcp)
        run_gcp
        ;;
    dry-run)
        echo "[Dry run - showing commands only]"
        echo "python 01_topology_mining.py --samples $SAMPLES"
        echo "python 02_geodpo_training.py --steps $STEPS"
        echo "python 03_analysis.py"
        ;;
    *)
        echo "Usage: $0 [local|gcp|dry-run] [--samples N] [--steps N]"
        echo ""
        echo "Modes:"
        echo "  local   - Run on current machine (requires GPU)"
        echo "  gcp     - Provision GCP VM, run, download results"
        echo "  dry-run - Show commands without running"
        exit 1
        ;;
esac
