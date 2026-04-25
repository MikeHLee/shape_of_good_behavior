# Colab Testing Checklist - Scale Experiments

**Date**: Jan 22, 2026  
**Purpose**: Validate notebooks for ICML submission experiments

---

## Pre-Testing Setup

1. [ ] Upload all 3 notebooks to Google Colab
2. [ ] Ensure GPU runtime is enabled: `Runtime > Change runtime type > T4 GPU`
3. [ ] For full-scale runs, request A100 if available (Colab Pro)

---

## Notebook 1: `colab_01_topology_mining.ipynb`

### Expected Runtime
- **T4 GPU**: ~15-20 min (50k samples)
- **A100**: ~5-8 min

### Cell-by-Cell Checklist

| Cell | Description | Expected Output | ✓ |
|------|-------------|-----------------|---|
| 0 | Markdown intro | N/A | |
| 1 | Setup & deps | `Running on cuda` | |
| 2 | Load & encode data | `Manifold Shape: (50000, 384)` | |
| 3 | Embed preference vectors | `Preference field computed.` | |
| 4 | Construct k-NN graph | `Graph constructed. Neighbors found for 50000 nodes.` | |
| 5 | Hodge analysis | `Mean Risk: ~0.3-0.5`, `Max Risk: ~1.0` | |
| 6 | Export metadata | `topology_metadata.parquet` saved | |

### Key Metrics to Record
```
Mean Harmonic Risk: _______
Max Harmonic Risk: _______
Manifold dimensions: _______
Runtime (cell 5): _______
```

### Potential Issues
- **OOM on cell 2/3**: Reduce `BATCH_SIZE` to 64 or `SAMPLE_LIMIT` to 25000
- **FAISS GPU error**: Check CUDA availability, may need `pip install faiss-cpu` fallback

---

## Notebook 2: `colab_02_geodpo_training.ipynb`

### Expected Runtime
- **T4 GPU**: ~30 min (50 steps, GPT-2)
- **A100 + Llama-2**: ~2-4 hours (full training)

### Pre-requisites
- [ ] `topology_metadata.parquet` from Notebook 1 must be in Colab filesystem
  - Run: `!ls -la topology_metadata.parquet` to verify

### Cell-by-Cell Checklist

| Cell | Description | Expected Output | ✓ |
|------|-------------|-----------------|---|
| 0 | Markdown intro | N/A | |
| 1 | Setup & deps | `Using Model: gpt2` | |
| 2 | Data loading | `Final Dataset Size: ~1000` | |
| 3 | GeoDPO Trainer def | `GeoDPO Trainer defined successfully.` | |
| 4 | Training loop | `GeoDPO Trainer initialized successfully!` | |

### To Run Actual Training
Uncomment last lines in cell 4:
```python
trainer.train()
trainer.save_model(OUTPUT_DIR)
```

### Key Metrics to Record
```
Training Loss (step 10): _______
Training Loss (step 50): _______
Geo Penalty (if logged): _______
Runtime: _______
```

### Potential Issues
- **Missing topology file**: Will use dummy data (random risks)
- **TRL version mismatch**: Ensure `trl>=0.12.0` installed
- **OOM**: Reduce `per_device_train_batch_size` to 1

---

## Notebook 3: `colab_03_analysis.ipynb`

### Expected Runtime
- **T4 GPU**: ~10-15 min
- **CPU fallback**: ~30 min

### Pre-requisites
- [ ] `topology_metadata.parquet` from Notebook 1
- [ ] `./geodpo_model_checkpoints/` from Notebook 2 (or will use dummy comparison)

### Cell-by-Cell Checklist

| Cell | Description | Expected Output | ✓ |
|------|-------------|-----------------|---|
| 0 | Markdown intro | N/A | |
| 1 | Setup & deps | `Running Analysis on cuda` | |
| 2 | Load high-risk prompts | `Selected 50 high-risk prompts...` | |
| 3 | Generate responses | Progress bars for both models | |
| 4 | Distance metrics | Mean similarities printed | |
| 5 | Visualization | `manifold_visualization.png` saved | |

### Key Metrics to Record (for paper)
```
Mean Similarity (Base): _______
Mean Similarity (GeoDPO): _______
Average Trajectory Shift: _______
```

### Expected Visualization
- Red dots: Risk prompts (cluster center)
- Gray dots: Base responses (close to red)
- Blue dots: GeoDPO responses (shifted away from red)

---

## Full Pipeline Test Order

1. Run **Notebook 1** completely → produces `topology_metadata.parquet`
2. Run **Notebook 2** cells 1-4 (verify setup works)
3. If time permits, uncomment training and run full training
4. Run **Notebook 3** for analysis (works with or without trained model)

---

## Results for Paper

After testing, update these results in `submission/DRAFT_PAPER.md`:

### Section 5.2: Scalability
- Dataset size tested: _______ samples
- Mining runtime (T4): _______ min
- Training runtime (T4): _______ min

### Section 5.3: Safety Metrics
- Mean trajectory shift: _______
- Proportion of responses shifted away from risk: _______% 

### Figure 5: Manifold Visualization
- Save `manifold_visualization.png` to `submission/figures/`

---

## Troubleshooting Commands

```python
# Check GPU
!nvidia-smi

# Check installed packages
!pip list | grep -E "trl|transformers|torch|faiss"

# Download file from Colab
from google.colab import files
files.download('topology_metadata.parquet')
files.download('manifold_visualization.png')

# Mount Google Drive (to persist files)
from google.colab import drive
drive.mount('/content/drive')
```

---

*Checklist created: Jan 22, 2026*
