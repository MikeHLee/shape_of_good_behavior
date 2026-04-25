# GCP Runner - CLI Execution for Scale Experiments

Run the GeoDPO scale experiments from your terminal using Google Cloud CLI, bypassing the Colab web interface.

## Prerequisites

```bash
# 1. Install Google Cloud CLI
brew install google-cloud-sdk  # macOS

# 2. Authenticate
gcloud auth login

# 3. Enable Compute Engine API (will prompt on first run)
gcloud services enable compute.googleapis.com
```

## Quick Start (Recommended)

### One-Command Full Pipeline

```bash
cd notebooks/gcp_runner

# Run everything: Create VM → Run experiments → Download results → Delete VM
./run_full_pipeline.sh --samples 50000 --steps 50
```

**Features**:
- ✅ Automatic zone fallback (tries multiple zones if GPU unavailable)
- ✅ Guaranteed VM teardown (even on errors or Ctrl+C)
- ✅ Full logging to timestamped file
- ✅ Waits for user keypress before exiting
- ✅ Estimated cost: ~$0.50-1.00 total

**Options**:
```bash
./run_full_pipeline.sh --help           # Show options
./run_full_pipeline.sh --test-only      # Test setup/teardown (no experiments)
./run_full_pipeline.sh --dry-run        # Show what would run
./run_full_pipeline.sh --skip-confirm   # Skip confirmation prompt
```

### Run Locally (if you have a GPU)

```bash
cd notebooks/gcp_runner

# Install dependencies
pip install sentence-transformers datasets faiss-gpu scipy networkx pyarrow \
    torch transformers "trl>=0.12.0" peft bitsandbytes pandas accelerate \
    scikit-learn matplotlib seaborn

# Run experiments
python 01_topology_mining.py --samples 50000
python 02_geodpo_training.py --steps 50
python 03_analysis.py
```

### Manual VM Control (Advanced)

```bash
cd notebooks/gcp_runner

# Step-by-step control:
./setup_gcp_vm.sh create      # Create T4 GPU VM (~$0.50/hr)
./setup_gcp_vm.sh scp-up      # Upload scripts
./setup_gcp_vm.sh ssh         # SSH into VM and run manually
./setup_gcp_vm.sh scp-down    # Download results
./setup_gcp_vm.sh delete      # Delete VM (important for cost!)
```

## Individual Scripts

### 1. Topology Mining
```bash
python 01_topology_mining.py --samples 50000 --k-neighbors 15 --output topology_metadata.parquet
```

**Output**: `topology_metadata.parquet` containing harmonic risk scores for each prompt.

### 2. GeoDPO Training
```bash
python 02_geodpo_training.py \
    --model gpt2 \
    --topology topology_metadata.parquet \
    --samples 1000 \
    --steps 50 \
    --lambda-geo 0.5 \
    --output ./geodpo_checkpoints
```

**Output**: LoRA adapter weights in `./geodpo_checkpoints/`

### 3. Analysis
```bash
python 03_analysis.py \
    --topology topology_metadata.parquet \
    --adapter ./geodpo_checkpoints \
    --samples 50 \
    --output-prefix results
```

**Output**: 
- `results_report.csv` - Detailed metrics per prompt
- `results_manifold.png` - PCA visualization of trajectory bending

## GCP VM Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Machine | n1-standard-8 | 8 vCPU, 30GB RAM |
| GPU | nvidia-tesla-t4 | 16GB VRAM, ~$0.35/hr |
| Zone | us-central1-a | Good T4 availability |
| Image | pytorch-latest-gpu | Pre-installed PyTorch + CUDA |
| Disk | 100GB SSD | For datasets + checkpoints |

### Cost Estimate
- T4 GPU: ~$0.35/hr
- Full pipeline (~2hr): ~$0.70
- **Remember to delete VM when done!**

## Expected Runtime

| Step | T4 GPU | A100 |
|------|--------|------|
| Topology Mining (50k) | ~20 min | ~5 min |
| GeoDPO Training (50 steps) | ~30 min | ~10 min |
| Analysis | ~10 min | ~3 min |
| **Total** | **~1 hr** | **~20 min** |

## Troubleshooting

### GPU Quota Error
```bash
# Request quota increase at:
# https://console.cloud.google.com/iam-admin/quotas
# Filter: "GPUs (all regions)" and request increase
```

### FAISS GPU Error
```bash
# Fallback to CPU FAISS
pip uninstall faiss-gpu
pip install faiss-cpu
```

### TRL Version Mismatch
```bash
pip install "trl>=0.12.0" --upgrade
```

## Files

```
gcp_runner/
├── README.md                 # This file
├── setup_gcp_vm.sh          # VM lifecycle management
├── run_experiments.sh       # Master runner
├── 01_topology_mining.py    # Step 1: Build risk map
├── 02_geodpo_training.py    # Step 2: Train GeoDPO
└── 03_analysis.py           # Step 3: Validate & visualize
```
