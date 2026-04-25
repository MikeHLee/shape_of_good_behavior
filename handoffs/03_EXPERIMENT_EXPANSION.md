# Handoff 03: Experiment Expansion - PPO, Clipped-SGPO, Larger Datasets

**Priority**: HIGH  
**Estimated Effort**: 8-12 hours  
**Type**: Coding, experiments, ML infrastructure  
**Dependencies**: Handoff 01 (paths), Handoff 04 (SGPO improvements may affect implementation)

---

## Context

Current experiments compare:
- **SGPO** (our method)
- **CPO** (Constrained Policy Optimization)
- *PPO mentioned but not fully implemented in Modal experiments*

### Gaps to Address
1. **Missing baselines**: Need PPO and Clipped-SGPO in Modal scale experiments
2. **Dataset scale**: Currently 50k HH-RLHF; could expand to 160k+ full dataset
3. **Dataset diversity**: Only using Anthropic HH-RLHF; could add other preference datasets

---

## Progress Tracking

**IMPORTANT**: Before starting this handoff, read `handoffs/00_PROGRESS_STATUS.md` to understand the current project state.

When you begin work:
1. Update the "Handoff 03" section in `00_PROGRESS_STATUS.md` with status 🟡 In Progress
2. Add start timestamp
3. Update "Current Session" section with your active task

When you complete tasks:
1. Check off completed items in the "Handoff 03" section
2. Add artifacts to "Artifacts Created"
3. Note any issues in "Issues/Notes"

When you finish or need to hand off:
1. Update status to ✅ Completed (or ⚠️ Blocked if issues)
2. Add a "Session Handoff" entry with what was done and next steps
3. Update the overall status table

---

## Current Experiment Infrastructure

### Modal Runner (`notebooks/modal_runner/`)
```
geodpo_experiments.py      # Main experiment script
├── topology_mining()      # Step 1: Extract harmonic risk from embeddings
├── geodpo_training()      # Step 2: Fine-tune with GeoDPO
├── analysis()             # Step 3: Compare trajectories
└── run_full_pipeline()    # Run all steps

results/
├── topology_metadata.parquet   # 50k prompts with risk scores
├── analysis_report.csv         # Base vs GeoDPO comparison
└── analysis_manifold.png       # PCA visualization
```

### Key Implementation Files
- `src/hodge_critic.py` - Hodge decomposition, topological gradients
- `src/semantic_mdp_rl.py` - Semantic MDP with SGPO
- `src/safety_experiment.py` - Safety benchmark implementation

---

## Task 1: Add PPO Baseline to Modal Experiments

### 1.1 Implement PPO Training Function

Add to `geodpo_experiments.py`:

```python
@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    volumes={VOLUME_PATH: volume},
)
def ppo_training(steps: int = 50, batch_size: int = 4):
    """Train PPO baseline on high-risk prompts for comparison."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import PPOTrainer, PPOConfig
    from peft import LoraConfig, get_peft_model
    import pandas as pd
    
    # Load high-risk prompts from topology mining
    topology_df = pd.read_parquet(f"{VOLUME_PATH}/topology_metadata.parquet")
    high_risk = topology_df.nlargest(50, "harmonic_risk")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add LoRA
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn"])
    model = get_peft_model(model, lora_config)
    
    # PPO config - standard settings
    ppo_config = PPOConfig(
        batch_size=batch_size,
        learning_rate=1e-5,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
    )
    
    # Simple reward: just use scalar preference score (no topology)
    def reward_fn(response_texts):
        # Placeholder: Use a reward model or simple heuristic
        return [0.5] * len(response_texts)  # Replace with actual reward
    
    trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        tokenizer=tokenizer,
    )
    
    # Training loop
    for step in range(steps):
        # Generate responses
        prompts = high_risk.sample(batch_size)["prompt"].tolist()
        # ... PPO update logic ...
    
    # Save model
    model.save_pretrained(f"{VOLUME_PATH}/ppo_model")
    return {"status": "complete", "steps": steps}
```

### 1.2 Modify Analysis to Include PPO

Update `analysis()` function to:
1. Load PPO model alongside GeoDPO model
2. Generate responses from both
3. Compute embeddings and trajectory shifts
4. Add PPO column to comparison table

---

## Task 2: Implement Clipped-SGPO

### Motivation
Standard SGPO may have unbounded advantages in safe regions. Clipped-SGPO adds PPO-style clipping to stabilize training while preserving geometric safety.

### 2.1 Clipped-SGPO Algorithm

```python
# In src/gpo_clipped.py (NEW FILE)

class ClippedSGPOTrainer:
    """SGPO with PPO-style clipping for stability."""
    
    def __init__(
        self,
        model,
        hodge_critic,
        metric_model,
        clip_ratio: float = 0.2,  # PPO clip parameter
        geometric_weight: float = 1.0,
    ):
        self.model = model
        self.hodge_critic = hodge_critic
        self.metric_model = metric_model
        self.clip_ratio = clip_ratio
        self.geometric_weight = geometric_weight
    
    def compute_clipped_advantage(self, states, actions, rewards):
        """Compute Hodge advantage with PPO-style clipping."""
        # Get Hodge-corrected advantage
        hodge_adv = self.hodge_critic.compute_advantage(states, actions, rewards)
        
        # Get metric scaling
        metric_scale = self.metric_model.get_scaling(states)
        
        # Geodesic advantage (from SGPO)
        geo_adv = hodge_adv / np.sqrt(metric_scale + 1e-8)
        
        # PPO-style clipping on the ratio
        # This prevents too-large updates while preserving geometric structure
        clipped_adv = np.clip(
            geo_adv,
            -self.clip_ratio * np.abs(geo_adv.mean()),
            self.clip_ratio * np.abs(geo_adv.mean())
        )
        
        return clipped_adv
    
    def update(self, batch):
        """Single training step with clipped geodesic objective."""
        states, actions, rewards, old_log_probs = batch
        
        # Compute clipped advantages
        advantages = self.compute_clipped_advantage(states, actions, rewards)
        
        # Policy ratio
        new_log_probs = self.model.log_prob(states, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective (PPO-style)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        
        # Take minimum (pessimistic bound)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Add Hodge-Bellman critic loss
        critic_loss = self.hodge_critic.compute_loss(states, actions, rewards)
        
        # Total loss
        loss = policy_loss + 0.5 * critic_loss
        
        return loss
```

### 2.2 Add to Modal Experiments

```python
@app.function(image=image, gpu="L4", timeout=3600, volumes={VOLUME_PATH: volume})
def clipped_gpo_training(steps: int = 50, clip_ratio: float = 0.2):
    """Train Clipped-SGPO (SGPO + PPO clipping) on high-risk prompts."""
    # Similar to geodpo_training but using ClippedSGPOTrainer
    ...
```

---

## Task 3: Expand to Larger Datasets

### 3.1 Full HH-RLHF Dataset

Current: 50,000 samples  
Available: ~160,000 samples in full training split

```python
# In topology_mining(), change:
# FROM:
dataset = dataset.select(range(min(samples, len(dataset))))
# TO:
if samples is None or samples >= len(dataset):
    print(f"Using full dataset: {len(dataset)} samples")
else:
    dataset = dataset.select(range(samples))
```

Add CLI option:
```python
@app.local_entrypoint()
def main(samples: int = None, full_dataset: bool = False):
    if full_dataset:
        samples = None  # Use all
    topology_mining.remote(samples=samples)
```

### 3.2 Additional Preference Datasets

Consider adding:

| Dataset | Size | Description | Source |
|---------|------|-------------|--------|
| **Stanford SHP** | 385k | Human preferences on Reddit | `stanfordnlp/shp` |
| **OpenAssistant** | 161k | Conversational preferences | `OpenAssistant/oasst1` |
| **UltraFeedback** | 64k | Multi-aspect preference scores | `openbmb/UltraFeedback` |
| **PKU-SafeRLHF** | 330k | Safety-focused preferences | `PKU-Alignment/PKU-SafeRLHF` |

#### Implementation for Multiple Datasets

```python
DATASETS = {
    "hh-rlhf": {
        "path": "anthropic/hh-rlhf",
        "extract_fn": extract_hh_pairs,
    },
    "shp": {
        "path": "stanfordnlp/shp",
        "extract_fn": extract_shp_pairs,
    },
    "ultrafeedback": {
        "path": "openbmb/UltraFeedback", 
        "extract_fn": extract_ultrafeedback_pairs,
    },
    "saferlhf": {
        "path": "PKU-Alignment/PKU-SafeRLHF",
        "extract_fn": extract_saferlhf_pairs,
    },
}

def extract_shp_pairs(example):
    """Extract preference pairs from Stanford SHP format."""
    return {
        "prompt": example["history"],
        "chosen_response": example["human_ref_A"] if example["labels"] == 1 else example["human_ref_B"],
        "rejected_response": example["human_ref_B"] if example["labels"] == 1 else example["human_ref_A"],
    }

@app.function(image=image, gpu="L4", timeout=7200, volumes={VOLUME_PATH: volume})
def multi_dataset_topology(datasets: list = ["hh-rlhf"], samples_per_dataset: int = 50000):
    """Mine topology across multiple preference datasets."""
    all_results = []
    
    for dataset_name in datasets:
        config = DATASETS[dataset_name]
        dataset = load_dataset(config["path"], split="train")
        dataset = dataset.select(range(min(samples_per_dataset, len(dataset))))
        processed = dataset.map(config["extract_fn"])
        
        # Run topology mining
        results = mine_topology(processed)
        results["dataset"] = dataset_name
        all_results.append(results)
    
    combined = pd.concat(all_results)
    combined.to_parquet(f"{VOLUME_PATH}/multi_dataset_topology.parquet")
    return {"total_samples": len(combined), "datasets": datasets}
```

---

## Task 4: Comparative Analysis Pipeline

### 4.1 Extended Analysis Function

```python
@app.function(image=image, gpu="L4", timeout=1800, volumes={VOLUME_PATH: volume})
def comparative_analysis():
    """Compare all models: Base GPT-2, PPO, CPO, SGPO, Clipped-SGPO."""
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    
    MODELS = {
        "base": "gpt2",
        "ppo": f"{VOLUME_PATH}/ppo_model",
        "cpo": f"{VOLUME_PATH}/cpo_model",  # Need to implement
        "gpo": f"{VOLUME_PATH}/geodpo_model",
        "gpo_clipped": f"{VOLUME_PATH}/gpo_clipped_model",
    }
    
    # Load high-risk prompts
    topology_df = pd.read_parquet(f"{VOLUME_PATH}/topology_metadata.parquet")
    test_prompts = topology_df.nlargest(100, "harmonic_risk")
    
    # Load sentence encoder
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    results = []
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path) and model_path != "gpt2":
            continue
            
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        for _, row in test_prompts.iterrows():
            # Generate response
            response = generate_response(model, tokenizer, row["prompt"])
            
            # Compute embeddings
            prompt_emb = encoder.encode(row["prompt"])
            response_emb = encoder.encode(response)
            
            results.append({
                "prompt_id": row.name,
                "model": model_name,
                "prompt": row["prompt"][:200],
                "response": response[:500],
                "harmonic_risk": row["harmonic_risk"],
                "prompt_emb": prompt_emb.tolist(),
                "response_emb": response_emb.tolist(),
                "trajectory_shift": compute_shift(prompt_emb, response_emb),
            })
    
    df = pd.DataFrame(results)
    df.to_parquet(f"{VOLUME_PATH}/comparative_analysis.parquet")
    
    # Summary statistics
    summary = df.groupby("model").agg({
        "trajectory_shift": ["mean", "std"],
        "harmonic_risk": "mean",
    })
    summary.to_csv(f"{VOLUME_PATH}/comparative_summary.csv")
    
    return summary.to_dict()
```

### 4.2 Export Embedding Data for Visualization

```python
@app.function(image=image, timeout=600, volumes={VOLUME_PATH: volume})
def export_embeddings_for_viz():
    """Export embedding data in format suitable for React visualization app."""
    import json
    import pandas as pd
    
    df = pd.read_parquet(f"{VOLUME_PATH}/comparative_analysis.parquet")
    
    # Group by prompt to create quintets
    grouped = df.groupby("prompt_id")
    
    viz_data = []
    for prompt_id, group in grouped:
        entry = {
            "prompt_id": int(prompt_id),
            "prompt_text": group.iloc[0]["prompt"],
            "harmonic_risk": float(group.iloc[0]["harmonic_risk"]),
            "responses": {}
        }
        
        for _, row in group.iterrows():
            entry["responses"][row["model"]] = {
                "text": row["response"],
                "embedding": row["response_emb"],
                "trajectory_shift": float(row["trajectory_shift"]),
            }
        
        # Add prompt embedding
        entry["prompt_embedding"] = group.iloc[0]["prompt_emb"]
        
        viz_data.append(entry)
    
    with open(f"{VOLUME_PATH}/viz_embeddings.json", "w") as f:
        json.dump(viz_data, f)
    
    return {"exported_quintets": len(viz_data)}
```

---

## Task 5: Implement CPO Baseline

CPO (Constrained Policy Optimization) is referenced but may not be fully implemented in Modal.

```python
@app.function(image=image, gpu="L4", timeout=3600, volumes={VOLUME_PATH: volume})
def cpo_training(steps: int = 50, cost_limit: float = 0.1):
    """Train CPO baseline with safety constraints."""
    # CPO requires:
    # 1. Reward function
    # 2. Cost function (safety violations)
    # 3. Trust region with constraint projection
    
    # Use TRL or custom implementation
    # Key: CPO uses Lagrangian relaxation for constraints
    ...
```

---

## Expected Outputs

After completing this handoff, you should have:

1. **New Modal functions**:
   - `ppo_training()`
   - `clipped_gpo_training()`
   - `cpo_training()`
   - `multi_dataset_topology()`
   - `comparative_analysis()`
   - `export_embeddings_for_viz()`

2. **New results files**:
   - `ppo_model/` - Trained PPO model
   - `cpo_model/` - Trained CPO model  
   - `gpo_clipped_model/` - Trained Clipped-SGPO model
   - `multi_dataset_topology.parquet` - Combined topology from multiple datasets
   - `comparative_analysis.parquet` - Full comparison data
   - `comparative_summary.csv` - Summary statistics
   - `viz_embeddings.json` - Export for visualization app

3. **Updated experiments.tex** tables with 5-model comparison

---

## Verification Checklist

- [ ] PPO training runs on Modal without errors
- [ ] Clipped-SGPO training runs on Modal without errors
- [ ] CPO training runs on Modal without errors
- [ ] Multi-dataset topology mining works with ≥2 datasets
- [ ] Comparative analysis produces valid embeddings
- [ ] Export function creates JSON compatible with Handoff 07
- [ ] **Progress tracking**: Updated `00_PROGRESS_STATUS.md` with completion status

---

## Estimated Costs

| Experiment | GPU | Time | Cost |
|------------|-----|------|------|
| PPO training (50 steps) | L4 | ~20 min | ~$0.50 |
| Clipped-SGPO training | L4 | ~20 min | ~$0.50 |
| CPO training | L4 | ~20 min | ~$0.50 |
| Full HH-RLHF topology (160k) | L4 | ~45 min | ~$1.00 |
| Multi-dataset (4 datasets) | L4 | ~2 hr | ~$3.00 |
| Comparative analysis | L4 | ~30 min | ~$0.75 |
| **Total** | | ~4 hr | ~$6.25 |

All within Modal free tier limits if run sequentially.

---

## Dependencies

**Requires**: Handoff 01 (directory structure)  
**Coordinates with**: Handoff 04 (SGPO improvements)  
**Feeds into**: Handoff 07 (visualization app needs `viz_embeddings.json`)
