#!/bin/bash
# GCP GPU VM Setup for Scale Experiments
# Usage: ./setup_gcp_vm.sh [create|delete|ssh|status]

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-oasis-training-suite}"
ZONES=("us-central1-b" "us-central1-c" "us-central1-f" "us-west1-a" "us-west1-b" "us-east1-c")
INSTANCE_NAME="geodpo-experiments"
MACHINE_TYPE="n1-standard-8"  # 8 vCPU, 30GB RAM
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="200GB"
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"
ZONE=""  # Will be set on successful creation

# Load saved zone if exists (for non-create commands)
if [[ -f /tmp/geodpo_zone.txt ]]; then
    ZONE=$(cat /tmp/geodpo_zone.txt)
fi

echo "Project: $PROJECT_ID"
echo "Zones: ${ZONES[*]}"
echo "Instance: $INSTANCE_NAME"

case "${1:-create}" in
    create)
        # Cost estimate
        echo "========================================"
        echo "COST ESTIMATE (us-central1-a)"
        echo "Machine: $MACHINE_TYPE (8 vCPU, 30GB)"
        echo "GPU: $GPU_TYPE x$GPU_COUNT"
        echo "Disk: $BOOT_DISK_SIZE SSD"
        echo "Type: Preemptible (Spot)"
        echo "----------------------------------------"
        echo "Est. Cost: ~\$0.13 / hour"
        echo "========================================"
        
        if [[ -z "$FORCE_YES" ]]; then
            read -p "Create VM? This will incur costs. [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Aborted."
                exit 1
            fi
        fi

        echo "Creating GPU VM (trying multiple zones)..."
        for zone in "${ZONES[@]}"; do
            echo "Trying zone: $zone"
            if gcloud compute instances create "$INSTANCE_NAME" \
                --project="$PROJECT_ID" \
                --zone="$zone" \
                --machine-type="$MACHINE_TYPE" \
                --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
                --image-family="$IMAGE_FAMILY" \
                --image-project="$IMAGE_PROJECT" \
                --boot-disk-size="$BOOT_DISK_SIZE" \
                --boot-disk-type="pd-ssd" \
                --maintenance-policy=TERMINATE \
                --metadata="install-nvidia-driver=True" \
                --scopes="https://www.googleapis.com/auth/cloud-platform" 2>&1; then
                ZONE="$zone"
                echo "VM created in zone: $ZONE"
                break
            else
                echo "Zone $zone unavailable, trying next..."
            fi
        done
        
        if [[ -z "$ZONE" ]]; then
            echo "ERROR: Failed to create VM in any zone."
            exit 1
        fi
        
        # Save zone for other commands
        echo "$ZONE" > /tmp/geodpo_zone.txt
        
        echo ""
        echo "VM created. Waiting 60s for startup..."
        sleep 60
        
        echo "Installing dependencies..."
        gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
            pip install -q sentence-transformers datasets faiss-gpu scipy networkx pyarrow \
                torch transformers trl>=0.12.0 peft bitsandbytes pandas accelerate \
                scikit-learn matplotlib seaborn papermill
        "
        
        echo ""
        echo "VM ready! Connect with: $0 ssh"
        ;;
    
    delete)
        echo "Deleting VM..."
        gcloud compute instances delete "$INSTANCE_NAME" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --quiet
        echo "VM deleted."
        ;;
    
    ssh)
        echo "Connecting to VM..."
        gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE"
        ;;
    
    scp-up)
        # Upload experiment files
        echo "Uploading experiment files..."
        gcloud compute scp --recurse \
            "$(dirname "$0")/../"*.ipynb \
            "$(dirname "$0")"/*.py \
            "$INSTANCE_NAME":~/experiments/ \
            --zone="$ZONE"
        echo "Files uploaded to ~/experiments/"
        ;;
    
    scp-down)
        # Download results
        echo "Downloading results..."
        mkdir -p ./results
        gcloud compute scp --recurse \
            "$INSTANCE_NAME":~/experiments/*.parquet \
            "$INSTANCE_NAME":~/experiments/*.png \
            "$INSTANCE_NAME":~/experiments/*.csv \
            ./results/ \
            --zone="$ZONE" 2>/dev/null || echo "Some files may not exist yet"
        echo "Results downloaded to ./results/"
        ;;
    
    status)
        gcloud compute instances describe "$INSTANCE_NAME" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --format="table(name,status,networkInterfaces[0].accessConfigs[0].natIP)"
        ;;
    
    run)
        # Run all experiments via papermill
        echo "Running experiments on VM..."
        gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
            cd ~/experiments
            echo '=== Running Topology Mining ==='
            papermill colab_01_topology_mining.ipynb output_01.ipynb -k python3 2>&1 | tee run_01.log
            
            echo '=== Running GeoDPO Training ==='
            papermill colab_02_geodpo_training.ipynb output_02.ipynb -k python3 2>&1 | tee run_02.log
            
            echo '=== Running Analysis ==='
            papermill colab_03_analysis.ipynb output_03.ipynb -k python3 2>&1 | tee run_03.log
            
            echo '=== Complete ==='
            ls -la *.parquet *.png *.csv 2>/dev/null || echo 'Checking outputs...'
        "
        ;;
    
    *)
        echo "Usage: $0 [create|delete|ssh|scp-up|scp-down|status|run]"
        echo ""
        echo "Commands:"
        echo "  create   - Create GPU VM with dependencies"
        echo "  delete   - Delete VM (saves cost!)"
        echo "  ssh      - SSH into VM"
        echo "  scp-up   - Upload notebook/script files"
        echo "  scp-down - Download results"
        echo "  status   - Check VM status"
        echo "  run      - Execute all experiments via papermill"
        exit 1
        ;;
esac
