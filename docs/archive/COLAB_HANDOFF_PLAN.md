# Handoff Plan: Scaled Experiments in Google Colab

This document outlines the roadmap for migrating the Sheaf-Theoretic Reward Spaces (STRS) framework from local toy simulations to scaled experiments on Google Colab using real LLMs (Llama-3-8B or GPT-2) and the Anthropic HH-RLHF dataset.

## 1. Environment Setup

**Hardware**: Google Colab Pro+ (A100 GPU recommended) or T4 (for GPT-2/TinyLlama).

**Dependencies**:
```python
!pip install torch transformers datasets sentence-transformers networkx scipy
!pip install accelerate bitsandbytes peft  # For efficient LLM training
```

## 2. Data Pipeline: Mining the Manifold

Before training, we must "map the territory" by computing the Hodge decomposition on the dataset. This creates the topological metadata (Gradient vs. Harmonic) used by the GeoDPO loss.

**Script**: `colab_01_topology_mining.ipynb`

1.  **Load Dataset**: `anthropic/hh-rlhf` (Helpful & Harmless).
2.  **Semantic Embedding**: Encode all prompts/responses using `all-MiniLM-L6-v2` (fast) or `gte-large` (better).
3.  **Construct Graph**:
    *   Nodes: (Prompt, Response) pairs.
    *   Edges: Preference pairs ($y_w \succ y_l$) + Semantic similarity edges.
4.  **Hodge Decomposition**:
    *   Compute $H^1$ (Harmonic component) for the dataset.
    *   **Filter**: Identify "Harmonic Holes" (inconsistent preference cycles).
    *   **Save**: Export the `TopologicalGradient` metadata (consistent reward scores + hole flags) to `topology_metadata.pt`.

## 3. Training: GeoDPO with One-Sided Clipping

We replace the standard RLHF/DPO loss with our Geodesic DPO loss.

**Script**: `colab_02_geodpo_training.ipynb`

### The Loss Function
We implement a custom `HuggingFace Trainer` or `DPOTrainer` subclass.

```python
class GeoDPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. Forward pass (get logits)
        policy_logits = model(inputs)
        
        # 2. Compute Standard DPO/PPO Loss (The "Lure")
        # Clips positive gradients to prevent exploding updates
        policy_loss = ... 
        
        # 3. Compute Geodesic Penalty (The "Force Field")
        # Load pre-computed metric g(s) for the current prompt embedding
        # g(s) is high if prompt is near a "Black Hole" (Harmful cluster)
        current_embeddings = self.embedder(inputs['prompts'])
        g_factors = self.metric_lookup(current_embeddings)
        
        # Penalty: Scale loss if moving towards black hole
        geo_penalty = g_factors * probability_of_harmful_token
        
        # 4. Total Loss
        # Do NOT clip the geo_penalty. Safety must be absolute.
        total_loss = policy_loss + lambda_geo * geo_penalty
        
        return total_loss
```

### Key Experiment: One-Sided Clipping
Test the hypothesis that **clipping positive rewards** (to avoid chasing phantom gradients in loops) while **leaving negative safety penalties unclipped** (to enforce hard boundaries) offers the best stability.

## 4. Evaluation & Visualization

**Script**: `colab_03_analysis.ipynb`

1.  **Metric**: "Distance to Event Horizon". Measure the geodesic distance of the trained model's responses to known harmful clusters.
2.  **Topological Consistency**: Re-run Hodge decomposition on the model's generated preference rankings. Does the model generate fewer cycles ($H^1 \to 0$)?
3.  **Visuals**: Plot the 3D embedding manifold showing the "Safe Trajectories" bending around the "Black Holes."

## 5. Artifact Checklist

*   [ ] `topology_metadata.pt`: The pre-computed manifold map.
*   [ ] `geodpo_llama_adapter.bin`: The LoRA weights of the safe model.
*   [ ] `safety_report.pdf`: Comparison of GeoDPO vs Standard DPO on the "JailbreakBench" dataset.
