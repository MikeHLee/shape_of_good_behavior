#!/usr/bin/env python3
"""
Scalable GeoDPO Training
CLI version of colab_02_geodpo_training.ipynb

Usage: python 02_geodpo_training.py [--model gpt2] [--topology topology_metadata.parquet] [--steps 50]
"""

import argparse
import os
import gc
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
from typing import Dict, List, Any

def parse_args():
    parser = argparse.ArgumentParser(description="GeoDPO Training")
    parser.add_argument("--model", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--topology", type=str, default="topology_metadata.parquet", help="Topology file from step 1")
    parser.add_argument("--output", type=str, default="./geodpo_checkpoints", help="Output directory")
    parser.add_argument("--samples", type=int, default=1000, help="Training samples")
    parser.add_argument("--steps", type=int, default=50, help="Max training steps")
    parser.add_argument("--lambda-geo", type=float, default=0.5, help="Geodesic penalty strength")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--dry-run", action="store_true", help="Setup only, don't train")
    return parser.parse_args()

class GeoDPOTrainer(DPOTrainer):
    """DPO with Geodesic (Topological) Penalty."""
    
    def __init__(self, lambda_geo=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_geo = lambda_geo
    
    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        harmonic_risk = batch.pop("harmonic_risk", None)
        metrics = super().get_batch_loss_metrics(model, batch, train_eval)
        
        if harmonic_risk is not None and "loss" in metrics:
            risk_tensor = harmonic_risk.to(metrics["loss"].device)
            geo_penalty = self.lambda_geo * risk_tensor.mean()
            metrics["loss"] = metrics["loss"] + geo_penalty
            metrics["geo_penalty"] = geo_penalty.item()
        
        return metrics

def load_and_merge_data(topology_path: str, sample_limit: int = None) -> Dataset:
    """Load and merge topology data with preference dataset."""
    
    print(f"Loading topology from {topology_path}...")
    if os.path.exists(topology_path):
        topo_df = pd.read_parquet(topology_path)
    else:
        print("⚠️ Topology file not found. Using random risk scores.")
        topo_df = None
    
    print("Loading Anthropic HH-RLHF...")
    dataset = load_dataset("anthropic/hh-rlhf", split="train")
    if sample_limit:
        dataset = dataset.select(range(min(sample_limit, len(dataset))))
    
    base_df = dataset.to_pandas()
    
    def extract_prompt(text):
        try:
            return text.rpartition("\n\nAssistant:")[0]
        except:
            return text[:100]
    
    def extract_response(text):
        try:
            return text.rpartition("\n\nAssistant:")[2].strip()
        except:
            return ""
    
    base_df["prompt"] = base_df["chosen"].apply(extract_prompt)
    base_df["chosen_response"] = base_df["chosen"].apply(extract_response)
    base_df["rejected_response"] = base_df["rejected"].apply(extract_response)
    
    if topo_df is not None and len(base_df) == len(topo_df):
        base_df["harmonic_risk"] = topo_df["harmonic_risk"].values
    elif topo_df is not None:
        merged = pd.merge(base_df, topo_df[["prompt", "harmonic_risk"]], on="prompt", how="inner")
        base_df = merged
    else:
        base_df["harmonic_risk"] = np.random.rand(len(base_df)).astype(np.float32)
    
    final_df = pd.DataFrame({
        "prompt": base_df["prompt"],
        "chosen": base_df["chosen_response"],
        "rejected": base_df["rejected_response"],
        "harmonic_risk": base_df["harmonic_risk"].astype(np.float32)
    })
    
    final_df = final_df[(final_df["chosen"].str.len() > 0) & (final_df["rejected"].str.len() > 0)]
    print(f"Dataset size: {len(final_df)}")
    
    return Dataset.from_pandas(final_df, preserve_index=False)

def main():
    args = parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {DEVICE}")
    print(f"Config: model={args.model}, lambda={args.lambda_geo}, beta={args.beta}")
    
    # Load data
    print("\n=== Loading Data ===")
    train_dataset = load_and_merge_data(args.topology, args.samples)
    
    print(f"\nSample entry:")
    print(f"  Risk: {train_dataset[0]['harmonic_risk']:.3f}")
    print(f"  Prompt: {train_dataset[0]['prompt'][:60]}...")
    
    # Load model
    print(f"\n=== Loading Model: {args.model} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if DEVICE == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        print("WARNING: No GPU. Loading float32.")
        model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"] if "gpt2" in args.model else ["q_proj", "v_proj"]
    )
    
    # Training config
    training_args = DPOConfig(
        output_dir=args.output,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        max_steps=args.steps,
        fp16=(DEVICE == "cuda"),
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = GeoDPOTrainer(
        lambda_geo=args.lambda_geo,
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    print(f"\n=== GeoDPO Trainer Ready ===")
    print(f"  Lambda (Geodesic): {args.lambda_geo}")
    print(f"  Beta (DPO temp): {args.beta}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Max steps: {args.steps}")
    
    if args.dry_run:
        print("\n[Dry run - skipping training]")
        return
    
    print("\n=== Starting Training ===")
    trainer.train()
    
    print(f"\n=== Saving to {args.output} ===")
    trainer.save_model(args.output)
    print("Training complete!")

if __name__ == "__main__":
    main()
