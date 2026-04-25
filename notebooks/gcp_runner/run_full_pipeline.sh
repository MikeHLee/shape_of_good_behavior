#!/bin/bash
# ============================================================
# GeoDPO Full Pipeline: Setup → Run → Download → Teardown
# ============================================================
# Usage: ./run_full_pipeline.sh [OPTIONS]
#
# Options:
#   --samples N     Number of topology mining samples (default: 50000)
#   --steps N       GeoDPO training steps (default: 50)
#   --test-only     Just test VM setup/teardown (no experiments)
#   --dry-run       Show what would run without executing
#   --skip-confirm  Skip confirmation prompt (for automation)
#
# Features:
#   - Full logging to timestamped log file
#   - Guaranteed VM teardown on exit (success or failure)
#   - Step-by-step progress monitoring
#   - Estimated cost: ~$0.50-1.00 total (1-2 hours)
# ============================================================

# Exit on undefined variables, pipefail to catch errors in pipes
set -u -o pipefail

# ============================================================
# Configuration
# ============================================================
GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
PROJECT_ID="oasis-training-suite"
INSTANCE_NAME="geodpo-runner"
MACHINE_TYPE="n1-standard-8"
# GPU types to try (L4 is newer with better availability, T4 as fallback)
GPU_TYPES=("nvidia-l4" "nvidia-tesla-t4")
GPU_TYPE=""  # Will be set when VM is created
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="200GB"

# Zones to try (in order of preference)
ZONES=("us-central1-a" "us-central1-b" "us-central1-c" "us-west1-a" "us-west1-b" "us-east1-b" "us-east1-c" "us-east4-a" "us-east4-c")
ZONE=""  # Will be set when VM is created

# Experiment parameters (can override via CLI)
SAMPLES=50000
STEPS=50
TEST_ONLY=0
DRY_RUN=0
SKIP_CONFIRM=0

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples) SAMPLES="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --test-only) TEST_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --skip-confirm) SKIP_CONFIRM=1; shift ;;
        -h|--help)
            head -25 "$0" | tail -23
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results_$TIMESTAMP"
LOG_FILE="$SCRIPT_DIR/pipeline_$TIMESTAMP.log"

# Track if VM was created (for cleanup)
VM_CREATED=0
PIPELINE_SUCCESS=0

# ============================================================
# Logging functions
# ============================================================
log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log_section() {
    log ""
    log "============================================================"
    log "$*"
    log "============================================================"
}

log_error() {
    log "❌ ERROR: $*"
}

log_success() {
    log "✅ $*"
}

log_warning() {
    log "⚠️  $*"
}

# ============================================================
# Cleanup function (ALWAYS runs on exit)
# ============================================================
cleanup() {
    local exit_code=$?
    
    log_section "CLEANUP"
    
    if [[ $VM_CREATED -eq 1 ]] && [[ -n "$ZONE" ]]; then
        log "Deleting VM '$INSTANCE_NAME' in zone '$ZONE'..."
        if $GCLOUD compute instances delete "$INSTANCE_NAME" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --quiet 2>> "$LOG_FILE"; then
            log_success "VM deleted successfully"
        else
            log_warning "VM deletion failed or VM not found"
        fi
    else
        log "No VM to clean up"
    fi
    
    log ""
    if [[ $PIPELINE_SUCCESS -eq 1 ]]; then
        log_success "Pipeline completed successfully!"
        log "Results: $RESULTS_DIR"
    else
        log_error "Pipeline failed with exit code $exit_code"
        log "Check log file: $LOG_FILE"
    fi
    
    log "Log saved to: $LOG_FILE"
    
    # Wait for user acknowledgment before exiting
    echo ""
    read -p "Press Enter to exit..." -r
    exit $exit_code
}

# Register cleanup to ALWAYS run on script exit
trap cleanup EXIT

# ============================================================
# Helper: Run command with logging
# ============================================================
run_logged() {
    log "Running: $*"
    if "$@" 2>&1 | tee -a "$LOG_FILE"; then
        return 0
    else
        return 1
    fi
}

# ============================================================
# Helper: SSH with logging
# ============================================================
ssh_vm() {
    $GCLOUD compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="$1" 2>&1 | tee -a "$LOG_FILE"
}

# ============================================================
# Print configuration
# ============================================================
log_section "GeoDPO Full Pipeline"
log "Timestamp:  $TIMESTAMP"
log "Project:    $PROJECT_ID"
log "Zone:       $ZONE"
log "Instance:   $INSTANCE_NAME"
log "GPU:        $GPU_TYPE"
log "Samples:    $SAMPLES"
log "Steps:      $STEPS"
log "Results:    $RESULTS_DIR"
log "Log:        $LOG_FILE"
log ""
if [[ $TEST_ONLY -eq 1 ]]; then
    log "Mode:       TEST ONLY (setup + teardown, no experiments)"
else
    log "Estimated cost: ~\$0.50-1.00 (VM auto-deleted after)"
fi

if [[ $DRY_RUN -eq 1 ]]; then
    log ""
    log "[DRY RUN] Would execute the pipeline. Exiting."
    PIPELINE_SUCCESS=1
    exit 0
fi

# Confirmation
if [[ $SKIP_CONFIRM -eq 0 ]]; then
    echo ""
    read -p "Proceed? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Aborted by user."
        PIPELINE_SUCCESS=1
        exit 0
    fi
fi

# ============================================================
# Step 1: Create VM (try multiple zones)
# ============================================================
log_section "[1/5] Creating GPU VM"
log "Trying zones: ${ZONES[*]}"
log ""

# Try each GPU type, then each zone
for gpu in "${GPU_TYPES[@]}"; do
    log "Trying GPU type: $gpu"
    
    for zone in "${ZONES[@]}"; do
        log "  Attempting zone: $zone with $gpu"
        
        # Capture output and exit code separately (tee masks exit codes)
        set +e  # Temporarily allow errors
        VM_OUTPUT=$($GCLOUD compute instances create "$INSTANCE_NAME" \
            --project="$PROJECT_ID" \
            --zone="$zone" \
            --machine-type="$MACHINE_TYPE" \
            --accelerator="type=$gpu,count=1" \
            --image-family="$IMAGE_FAMILY" \
            --image-project="$IMAGE_PROJECT" \
            --boot-disk-size="$BOOT_DISK_SIZE" \
            --boot-disk-type="pd-ssd" \
            --maintenance-policy=TERMINATE \
            --metadata="install-nvidia-driver=True" \
            --scopes="https://www.googleapis.com/auth/cloud-platform" 2>&1)
        VM_EXIT_CODE=$?
        set -e  # Re-enable exit on error
        
        # Log output (but keep it brief for failed attempts)
        if [[ $VM_EXIT_CODE -ne 0 ]]; then
            echo "$VM_OUTPUT" >> "$LOG_FILE"  # Log only, don't print full error
            log_warning "  Zone $zone unavailable, trying next..."
        else
            echo "$VM_OUTPUT" | tee -a "$LOG_FILE"
            # Verify VM actually exists
            if $GCLOUD compute instances describe "$INSTANCE_NAME" --zone="$zone" --project="$PROJECT_ID" &>/dev/null; then
                VM_CREATED=1
                ZONE="$zone"
                GPU_TYPE="$gpu"
                log_success "VM created and verified: $gpu in $ZONE"
                break 2  # Break out of both loops
            else
                log_warning "  VM creation reported success but VM not found"
            fi
        fi
    done
done

if [[ $VM_CREATED -eq 0 ]]; then
    log_error "Failed to create VM in any zone. All zones exhausted."
    log "You may need to request GPU quota or try again later."
    exit 1
fi

log "Waiting 90s for VM startup and driver installation..."
log "(You can monitor with: tail -f $LOG_FILE)"
sleep 90

# Verify VM is reachable
log "Verifying VM connectivity..."
# Retry SSH connection (VM may take time to be ready)
SSH_RETRIES=3
SSH_SUCCESS=0
for i in $(seq 1 $SSH_RETRIES); do
    log "SSH attempt $i of $SSH_RETRIES..."
    set +e
    SSH_OUTPUT=$(ssh_vm "echo 'VM is reachable' && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU driver loading...'")
    SSH_EXIT=$?
    set -e
    
    if [[ $SSH_EXIT -eq 0 ]] && [[ "$SSH_OUTPUT" == *"reachable"* ]]; then
        log_success "VM is reachable"
        SSH_SUCCESS=1
        break
    else
        log_warning "SSH attempt $i failed, waiting 30s..."
        sleep 30
    fi
done

if [[ $SSH_SUCCESS -eq 0 ]]; then
    log_error "Cannot connect to VM after $SSH_RETRIES attempts"
    exit 1
fi

# ============================================================
# Test-only mode: skip experiments
# ============================================================
if [[ $TEST_ONLY -eq 1 ]]; then
    log_section "TEST COMPLETE"
    log "VM was created and is reachable."
    log "Teardown will happen automatically..."
    PIPELINE_SUCCESS=1
    exit 0
fi

# ============================================================
# Step 2: Install dependencies
# ============================================================
log_section "[2/5] Installing Python dependencies"
log "This will take 2-3 minutes..."

if ssh_vm "
    pip install -q sentence-transformers datasets faiss-cpu scipy networkx pyarrow \
        torch transformers 'trl>=0.12.0' peft bitsandbytes pandas accelerate \
        scikit-learn matplotlib seaborn 2>&1
    mkdir -p ~/experiments
    echo 'Dependencies installed'
"; then
    log_success "Dependencies installed"
else
    log_error "Failed to install dependencies"
    exit 1
fi

# ============================================================
# Step 3: Upload experiment scripts
# ============================================================
log_section "[3/5] Uploading experiment scripts"

if $GCLOUD compute scp \
    "$SCRIPT_DIR/01_topology_mining.py" \
    "$SCRIPT_DIR/02_geodpo_training.py" \
    "$SCRIPT_DIR/03_analysis.py" \
    "$INSTANCE_NAME":~/experiments/ \
    --zone="$ZONE" 2>&1 | tee -a "$LOG_FILE"; then
    log_success "Scripts uploaded"
else
    log_error "Failed to upload scripts"
    exit 1
fi

# Verify files
ssh_vm "ls -la ~/experiments/*.py"

# ============================================================
# Step 4: Run experiments
# ============================================================
log_section "[4/5] Running experiments"
log "Samples: $SAMPLES, Steps: $STEPS"
log "This may take 1-2 hours. Progress will be shown below."
log ""

# Run each step separately for better error tracking
log "--- Step 4a: Topology Mining ---"
if ssh_vm "
    cd ~/experiments
    python 01_topology_mining.py --samples $SAMPLES --k-neighbors 15 --output topology_metadata.parquet
"; then
    log_success "Topology mining complete"
else
    log_error "Topology mining failed"
    exit 1
fi

log ""
log "--- Step 4b: GeoDPO Training ---"
if ssh_vm "
    cd ~/experiments
    python 02_geodpo_training.py \
        --model gpt2 \
        --topology topology_metadata.parquet \
        --samples 1000 \
        --steps $STEPS \
        --lambda-geo 0.5 \
        --output ./geodpo_checkpoints
"; then
    log_success "GeoDPO training complete"
else
    log_error "GeoDPO training failed"
    exit 1
fi

log ""
log "--- Step 4c: Analysis ---"
if ssh_vm "
    cd ~/experiments
    python 03_analysis.py \
        --topology topology_metadata.parquet \
        --adapter ./geodpo_checkpoints \
        --samples 50 \
        --output-prefix results
"; then
    log_success "Analysis complete"
else
    log_warning "Analysis failed (non-critical, continuing)"
fi

# List outputs
log ""
log "--- Output files on VM ---"
ssh_vm "ls -la ~/experiments/*.parquet ~/experiments/*.png ~/experiments/*.csv 2>/dev/null || echo 'Checking for outputs...'"

# ============================================================
# Step 5: Download results
# ============================================================
log_section "[5/5] Downloading results"
mkdir -p "$RESULTS_DIR"

# Download each file type separately for robustness
log "Downloading parquet files..."
$GCLOUD compute scp "$INSTANCE_NAME":~/experiments/*.parquet "$RESULTS_DIR/" \
    --zone="$ZONE" 2>&1 | tee -a "$LOG_FILE" || log_warning "No parquet files found"

log "Downloading PNG files..."
$GCLOUD compute scp "$INSTANCE_NAME":~/experiments/*.png "$RESULTS_DIR/" \
    --zone="$ZONE" 2>&1 | tee -a "$LOG_FILE" || log_warning "No PNG files found"

log "Downloading CSV files..."
$GCLOUD compute scp "$INSTANCE_NAME":~/experiments/*.csv "$RESULTS_DIR/" \
    --zone="$ZONE" 2>&1 | tee -a "$LOG_FILE" || log_warning "No CSV files found"

log "Downloading model checkpoints..."
$GCLOUD compute scp --recurse "$INSTANCE_NAME":~/experiments/geodpo_checkpoints "$RESULTS_DIR/" \
    --zone="$ZONE" 2>&1 | tee -a "$LOG_FILE" || log_warning "No checkpoints found"

log ""
log "--- Downloaded files ---"
ls -la "$RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Success!
# ============================================================
PIPELINE_SUCCESS=1
log_section "PIPELINE COMPLETE"
log_success "All experiments finished successfully!"
log "Results saved to: $RESULTS_DIR"
log ""
log "VM will be deleted automatically in cleanup..."
