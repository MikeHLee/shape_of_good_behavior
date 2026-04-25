"""
Evaluation Script for MLX Mamba Agent.
Evaluates the trained agent on TextWorld trajectories by checking action prediction accuracy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import json
import argparse
from typing import List, Dict, Any

from src.mlx_mamba_agent import MLXMambaAgent

def load_metadata(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)

def encode_text(text: str, vocab: Dict[str, int], max_len: int = 50) -> mx.array:
    indices = [vocab.get(w, vocab.get("<UNK>", 1)) for w in text.split()]
    if len(indices) > max_len:
        indices = indices[-max_len:]
    else:
        indices = indices + [vocab.get("<PAD>", 0)] * (max_len - len(indices))
    return mx.array([indices]) # Batch size 1

def evaluate(model, vocab, action_map, episodes_path, max_episodes=10):
    print(f"Loading episodes from {episodes_path}...")
    
    total_steps = 0
    correct_top1 = 0
    correct_top3 = 0
    
    inv_action_map = {v: k for k, v in action_map.items()}
    
    with open(episodes_path, 'r') as f:
        lines = f.readlines()
        
    # Evaluate on a subset
    lines = lines[:max_episodes]
    
    for line in lines:
        episode = json.loads(line)
        transitions = episode.get("transitions", [])
        if not transitions:
            continue
            
        current_context = transitions[0].get("state", "Start")
        
        for t in transitions:
            target_action = t.get("action")
            next_obs = t.get("next_state", "")
            
            if target_action not in action_map:
                # Skip actions not in training vocab
                continue
                
            target_idx = action_map[target_action]
            
            # Predict
            x_in = encode_text(current_context, vocab)
            output = model(input_ids=x_in)
            logits = output.action_logits[0] # (n_actions,)
            
            # Top-k
            top_k_indices = np.argsort(logits)[::-1]
            top1 = top_k_indices[0]
            top3 = top_k_indices[:3]
            
            if top1 == target_idx:
                correct_top1 += 1
            if target_idx in top3:
                correct_top3 += 1
                
            total_steps += 1
            
            # Update context
            current_context += f" | > {target_action} | {next_obs}"
            if len(current_context) > 2000:
                current_context = current_context[-2000:]
                
    if total_steps == 0:
        print("No valid steps found for evaluation.")
        return
        
    acc1 = correct_top1 / total_steps
    acc3 = correct_top3 / total_steps
    
    print(f"\nEvaluation Results ({total_steps} steps):")
    print(f"  Top-1 Accuracy: {acc1:.2%}")
    print(f"  Top-3 Accuracy: {acc3:.2%}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="mlx_mamba_textworld.npz")
    parser.add_argument("--metadata_path", default="mlx_mamba_metadata.json")
    parser.add_argument("--data_path", default="tw_games_generated/episodes.jsonl")
    args = parser.parse_args()
    
    # Load metadata
    metadata = load_metadata(args.metadata_path)
    vocab = metadata["vocab"]
    action_map = metadata["action_map"]
    config = metadata["config"]
    
    # Reconstruct model
    print("Initializing model...")
    model = MLXMambaAgent(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        state_dim=config["state_dim"],
        n_layers=config["n_layers"],
        n_actions=config["n_actions"]
    )
    
    # Load weights
    print(f"Loading weights from {args.model_path}...")
    model.load_weights(args.model_path)
    mx.eval(model.parameters())
    
    # Evaluate
    evaluate(model, vocab, action_map, args.data_path)

if __name__ == "__main__":
    main()
