# Modal Runner - Serverless GPU for GeoDPO Experiments

Run GeoDPO experiments on Modal's serverless GPUs. No quota requests needed, pay only for compute time.

## Quick Start

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate (one-time setup)

```bash
modal setup
```

This opens a browser for authentication. Free tier includes **$30/month credits**.

### 3. Run Full Pipeline

```bash
cd notebooks/modal_runner

# Run everything (topology → training → analysis)
modal run geodpo_experiments.py --samples 50000 --steps 50
```

**Estimated time**: ~1-2 hours  
**Estimated cost**: ~$2-3 (covered by free tier)

## Individual Steps

Run steps separately if needed:

```bash
# Step 1: Topology Mining (~30 min)
modal run geodpo_experiments.py::topology_mining --samples 50000

# Step 2: GeoDPO Training (~45 min)  
modal run geodpo_experiments.py::geodpo_training --steps 50

# Step 3: Analysis (~15 min)
modal run geodpo_experiments.py::analysis
```

## Download Results

Results are stored in a Modal volume. Download them:

```bash
# Download all results
modal volume get geodpo-data /data ./results

# Or specific files
modal volume get geodpo-data /data/topology_metadata.parquet ./
modal volume get geodpo-data /data/analysis_report.csv ./
modal volume get geodpo-data /data/analysis_manifold.png ./
```

## Pricing

| GPU | $/hour | Our Experiments |
|-----|--------|-----------------|
| L4 (24GB) | ~$0.80 | Default |
| T4 (16GB) | ~$0.59 | Fallback |
| A10 (24GB) | ~$1.10 | Faster |

**Free tier**: $30/month credits (covers ~30+ hours of L4)

## Academic Credits

Graduate students and researchers can apply for **$10,000 free credits**:
https://modal.com/academics

## Troubleshooting

### "Volume not found"
The volume is created on first run. Just run the pipeline again.

### "GPU not available"
Modal auto-schedules GPUs. If L4 is busy, it may queue briefly. You can modify `gpu="L4"` to `gpu="T4"` in the code for faster availability.

### Checking logs
```bash
modal app logs geodpo-experiments
```

### Viewing volume contents
```bash
modal volume ls geodpo-data /data
```

## File Structure

```
modal_runner/
├── geodpo_experiments.py   # Main Modal app with all 3 steps
├── README.md               # This file
└── download_results.sh     # Helper to download results
```

## Comparison with GCP

| Feature | Modal | GCP |
|---------|-------|-----|
| Setup time | 2 min | 30+ min |
| Quota needed | No | Yes |
| Billing | Per-second | Per-minute |
| Free credits | $30/mo | $300 trial |
| Cold start | ~30s | N/A (VM) |
| Auto-cleanup | Yes | Manual |

Modal is ideal for one-off experiments. GCP is better for long-running jobs or if you need persistent VMs.
