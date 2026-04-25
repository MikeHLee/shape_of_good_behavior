#!/usr/bin/env python3
"""
Scalable Topology Mining for Reward Spaces
CLI version of colab_01_topology_mining.ipynb

Usage: python 01_topology_mining.py [--samples 50000] [--k-neighbors 15] [--output topology_metadata.parquet]
"""

import argparse
import gc
import torch
import numpy as np
import pandas as pd
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Topology Mining for Reward Spaces")
    parser.add_argument("--samples", type=int, default=50000, help="Number of samples (None for full)")
    parser.add_argument("--k-neighbors", type=int, default=15, help="k for k-NN graph")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--output", type=str, default="topology_metadata.parquet", help="Output file")
    return parser.parse_args()

def extract_pairs(example):
    """Extract prompt and responses from HH-RLHF format."""
    try:
        prompt = example["chosen"].rpartition("\n\nAssistant:")[0]
        chosen_response = example["chosen"].rpartition("\n\nAssistant:")[2]
        rejected_response = example["rejected"].rpartition("\n\nAssistant:")[2]
    except:
        prompt = example["chosen"][:100]
        chosen_response = ""
        rejected_response = ""
    return {
        "prompt": prompt,
        "chosen_response": chosen_response,
        "rejected_response": rejected_response
    }

def main():
    args = parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {DEVICE}")
    print(f"Config: samples={args.samples}, k={args.k_neighbors}, model={args.embedding_model}")
    
    # 1. Load & preprocess data
    print("\n=== Loading Dataset ===")
    dataset = load_dataset("anthropic/hh-rlhf", split="train")
    if args.samples:
        print(f"Subsampling to {args.samples} examples...")
        dataset = dataset.select(range(min(args.samples, len(dataset))))
    
    print("Preprocessing...")
    processed_data = dataset.map(extract_pairs, num_proc=2)
    prompts = processed_data["prompt"]
    
    # 2. Encode prompts
    print("\n=== Embedding Prompts ===")
    model = SentenceTransformer(args.embedding_model, device=DEVICE)
    prompt_embeddings = model.encode(
        prompts, 
        batch_size=args.batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    faiss.normalize_L2(prompt_embeddings)
    print(f"Manifold Shape: {prompt_embeddings.shape}")
    
    # 3. Embed preference vectors
    print("\n=== Computing Preference Vectors ===")
    chosen_embs = model.encode(
        processed_data["chosen_response"], 
        batch_size=args.batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    rejected_embs = model.encode(
        processed_data["rejected_response"], 
        batch_size=args.batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    preference_vectors = chosen_embs - rejected_embs
    norms = np.linalg.norm(preference_vectors, axis=1, keepdims=True)
    preference_directions = preference_vectors / (norms + 1e-8)
    
    del chosen_embs, rejected_embs
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    # 4. Build k-NN graph
    print("\n=== Building k-NN Graph ===")
    d = prompt_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    if DEVICE == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(prompt_embeddings)
    D, I = index.search(prompt_embeddings, args.k_neighbors)
    print(f"Graph constructed. Neighbors found for {len(I)} nodes.")
    
    # 5. Hodge analysis (Harmonic Risk)
    print("\n=== Computing Harmonic Risk ===")
    neighbor_vectors = preference_directions[I]
    local_mean_vectors = np.mean(neighbor_vectors, axis=1)
    mean_norms = np.linalg.norm(local_mean_vectors, axis=1, keepdims=True)
    local_mean_dirs = local_mean_vectors / (mean_norms + 1e-8)
    
    consistencies = np.sum(neighbor_vectors * local_mean_dirs[:, np.newaxis, :], axis=2)
    avg_consistency = np.mean(consistencies, axis=1)
    harmonic_risk_scores = 1.0 - avg_consistency
    
    # Normalize to 0-1
    min_risk = np.min(harmonic_risk_scores)
    max_risk = np.max(harmonic_risk_scores)
    harmonic_risk_scores = (harmonic_risk_scores - min_risk) / (max_risk - min_risk + 1e-8)
    
    print(f"Mean Risk: {np.mean(harmonic_risk_scores):.4f}")
    print(f"Max Risk: {np.max(harmonic_risk_scores):.4f}")
    print(f"Std Risk: {np.std(harmonic_risk_scores):.4f}")
    
    # 6. Export
    print(f"\n=== Exporting to {args.output} ===")
    metadata_df = pd.DataFrame({
        "prompt": prompts,
        "harmonic_risk": harmonic_risk_scores,
        "embedding_id": range(len(prompts)),
    })
    metadata_df.to_parquet(args.output)
    print(f"Saved {len(metadata_df)} entries to {args.output}")
    
    # Show top risks
    print("\nTop 5 High-Risk (Inconsistent) Areas:")
    top_risks = metadata_df.nlargest(5, "harmonic_risk")
    for i, row in top_risks.iterrows():
        print(f"\n[Risk: {row['harmonic_risk']:.3f}] {row['prompt'][:120]}...")
    
    return metadata_df

if __name__ == "__main__":
    main()
