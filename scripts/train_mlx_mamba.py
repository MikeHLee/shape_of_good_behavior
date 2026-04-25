"""
MLX Training Script for Mamba Agent on M4 Max.
Trains the agent using Behavioral Cloning on TextWorld solution trajectories.

Usage:
    python scripts/train_mlx_mamba.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import json
import glob
import time
from typing import List, Tuple, Dict

from src.mlx_mamba_agent import MLXMambaAgent

# Hyperparameters
VOCAB_SIZE = 5000
EMBED_DIM = 256
STATE_DIM = 512
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 10
SEQ_LEN = 20  # Max steps to consider for context

def load_textworld_data(data_dir: str) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Load trajectories from TextWorld data (preferring generated JSONL).
    Returns: (states, actions, vocab_map)
    """
    states = []
    actions = []
    vocab = {"<PAD>": 0, "<UNK>": 1}
    
    # Check for generated episodes.jsonl
    jsonl_path = Path(data_dir) / "episodes.jsonl"
    if jsonl_path.exists():
        print(f"Loading generated trajectories from {jsonl_path}...")
        try:
            with open(jsonl_path, 'r') as f:
                for line in f:
                    episode = json.loads(line)
                    transitions = episode.get("transitions", [])
                    if not transitions:
                        continue
                        
                    # Use initial observation as start context
                    current_context = transitions[0].get("state", "Start")
                    
                    for t in transitions:
                        cmd = t.get("action")
                        next_obs = t.get("next_state", "")
                        
                        states.append(current_context)
                        actions.append(cmd)
                        
                        # Update vocab
                        for word in current_context.split():
                            if word not in vocab:
                                vocab[word] = len(vocab)
                        for word in cmd.split():
                            if word not in vocab:
                                vocab[word] = len(vocab)
                        
                        # Update context with action and result
                        # Truncate context if too long to prevent massive slowdowns
                        current_context += f" | > {cmd} | {next_obs}"
                        if len(current_context) > 2000:
                            current_context = current_context[-2000:]
                            
        except Exception as e:
            print(f"Error loading JSONL: {e}")
            
    else:
        # Fallback to legacy game files
        files = glob.glob(f"{data_dir}/*.json")
        print(f"Loading legacy data from {len(files)} games in {data_dir}...")
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                for quest in data.get("quests", []):
                    commands = quest.get("commands", [])
                    desc = quest.get("desc", "Start")
                    
                    # Context accumulation (Trajectory)
                    current_context = f"Quest: {desc}"
                    
                    for cmd in commands:
                        states.append(current_context)
                        actions.append(cmd)
                        
                        # Update vocab
                        for word in current_context.split():
                            if word not in vocab:
                                vocab[word] = len(vocab)
                        for word in cmd.split():
                            if word not in vocab:
                                vocab[word] = len(vocab)
                                
                        # Update context
                        current_context += f" | > {cmd}"
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
            
    print(f"Loaded {len(states)} training examples.")
    print(f"Vocabulary size: {len(vocab)}")
    return states, actions, vocab

def encode_batch(texts: List[str], vocab: Dict[str, int], max_len: int = 50) -> mx.array:
    """Encode text batch to indices."""
    batch_indices = []
    for text in texts:
        indices = [vocab.get(w, vocab["<UNK>"]) for w in text.split()]
        if len(indices) > max_len:
            indices = indices[-max_len:]
        else:
            indices = indices + [vocab["<PAD>"]] * (max_len - len(indices))
        batch_indices.append(indices)
    return mx.array(batch_indices)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train MLX Mamba Agent")
    parser.add_argument("--data_dir", type=str, default="tw_games", help="Directory containing TextWorld data")
    args = parser.parse_args()

    # 1. Load Data
    states, actions, vocab = load_textworld_data(args.data_dir)
    
    # Map actions to unique IDs
    unique_actions = list(set(actions))
    action_map = {a: i for i, a in enumerate(unique_actions)}
    n_actions = len(unique_actions)
    
    # 2. Initialize Model
    model = MLXMambaAgent(
        vocab_size=len(vocab) + 1000, # Buffer
        embed_dim=EMBED_DIM,
        state_dim=STATE_DIM,
        n_layers=4,
        n_actions=n_actions
    )
    mx.eval(model.parameters())
    
    # 3. Optimization
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    
    # Loss function (Cross Entropy for Action Prediction)
    def loss_fn(model, inputs, targets):
        # inputs: (B, L) indices
        # targets: (B,) action indices
        output = model(input_ids=inputs)
        logits = output.action_logits # (B, n_actions)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")
    
    # Create value_and_grad function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Compile the forward and backward pass
    @mx.compile
    def compute_loss_and_grads(inputs, targets):
        return loss_and_grad_fn(model, inputs, targets)
    
    # 4. Training Loop
    print("\nStarting Training on MLX...")
    
    n_samples = len(states)
    indices = np.arange(n_samples)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        np.random.shuffle(indices)
        
        epoch_loss = 0.0
        steps = 0
        
        for start_idx in range(0, n_samples, BATCH_SIZE):
            batch_idx = indices[start_idx : start_idx + BATCH_SIZE]
            
            batch_states = [states[i] for i in batch_idx]
            batch_actions = [action_map[actions[i]] for i in batch_idx]
            
            # Encode
            x_in = encode_batch(batch_states, vocab)
            y_tgt = mx.array(batch_actions)
            
            # Update
            loss, grads = compute_loss_and_grads(x_in, y_tgt)
            optimizer.update(model, grads)
            
            # Evaluate state to ensure computation happens
            mx.eval(model.parameters(), optimizer.state)
            
            epoch_loss += loss.item()
            steps += 1
            
        dt = time.time() - start_time
        avg_loss = epoch_loss / steps if steps > 0 else 0.0
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {dt:.2f}s | TPS: {n_samples/dt:.0f}")
        
    print("Training Complete.")
    model.save_weights("mlx_mamba_textworld.npz")
    print("Model saved to mlx_mamba_textworld.npz")
    
    # Save metadata for inference
    metadata = {
        "vocab": vocab,
        "action_map": action_map, # action string -> index
        "config": {
            "vocab_size": len(vocab) + 1000,
            "embed_dim": EMBED_DIM,
            "state_dim": STATE_DIM,
            "n_layers": 4,
            "n_actions": n_actions
        }
    }
    
    with open("mlx_mamba_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved to mlx_mamba_metadata.json")

if __name__ == "__main__":
    main()
