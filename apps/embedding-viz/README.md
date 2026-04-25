# Embedding Visualization App

Interactive visualization of reward manifold trajectories for the Sheaf-Theoretic RL paper.

## Features

- **PCA Projection**: 384-dim sentence embeddings → 2D interactive plot
- **Model Comparison**: Base, PPO, CPO, GPO, and Clipped-GPO responses
- **Trajectory Lines**: Connect prompt to all response variants
- **Hover Details**: Full text, harmonic risk, trajectory shift
- **Export**: Download publication-quality PNG screenshots

## Setup

### 1. Install Dependencies

```bash
cd apps/embedding-viz
npm install
```

### 2. Get Experiment Data

Run Modal experiments to generate visualization data:

```bash
cd notebooks/modal_runner

# Run experiments
modal run geodpo_experiments.py::comparative_analysis --n-prompts 100
modal run geodpo_experiments.py::export_embeddings_for_viz

# Download results to data/
modal volume get geodpo-data /data ../../data/
```

This creates:
- `data/viz_embeddings.json` - Primary visualization data

### 3. Start Dev Server

```bash
npm run dev
```

Open http://localhost:5173

## Data Format

The app expects `data/viz_embeddings.json` with this structure:

```json
[
  {
    "prompt_id": 1,
    "prompt_text": "How do I...",
    "harmonic_risk": 0.87,
    "prompt_embedding": [0.1, 0.2, ...],
    "responses": {
      "base": { "text": "...", "embedding": [...], "trajectory_shift": 0.12 },
      "ppo": { "text": "...", "embedding": [...], "trajectory_shift": 0.08 },
      "cpo": { "text": "...", "embedding": [...], "trajectory_shift": -0.15 },
      "gpo": { "text": "...", "embedding": [...], "trajectory_shift": -0.22 },
      "gpo_clipped": { "text": "...", "embedding": [...], "trajectory_shift": -0.20 }
    }
  }
]
```

## Mock Data

If real data is unavailable, the app automatically falls back to mock data for development/testing.

## Usage

1. **Toggle Models**: Use checkboxes to show/hide specific models
2. **Click Points**: Highlight a quintet (prompt + 5 responses) and show trajectory lines
3. **Hover**: See full text and metrics for any point
4. **Export**: Use the camera icon in the Plotly toolbar to download PNG

## Building for Production

```bash
npm run build
npm run preview
```
