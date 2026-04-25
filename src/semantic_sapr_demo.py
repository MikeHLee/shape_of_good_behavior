"""
Semantic SAPR Demo with Mamba Agent

This script demonstrates the "SAPR" (States, Actions, Probabilities, Rewards) tuple
using a Mamba-based Multimodal Sequence Model agent in a semantic environment.

Tensor RL Framework (see TENSOR_RL_FOUNDATIONS.md):
- S (State): Semantic embedding of the current observation (Text + Vision).
          The state tensor s_t is the Kolmogorov minimal description of the world.
- A (Action): Semantic embedding / discrete choice of the agent.
          The action tensor a_t is selected by the policy π(a|s).
- P (Probability): The transition probability P(s_{t+1}|s_t, a_t).
          CRITICAL DISTINCTION:
          - ORACLE/ENVIRONMENT: The *true* dynamics (game engine, physical world, GPT-4 sim)
          - AGENT'S WORLD MODEL: The agent's *learned approximation* of the oracle
          The agent learns a world model to predict what the oracle will do.
- R (Reward): The reward tensor, which can be:
          - Scalar: Traditional R(s,a) → ℝ
          - Vector: High-dimensional R(s,a) → ℝ^d (for Hodge decomposition)
          - Sheaf section: Evaluated at multiple scales (step, segment, trajectory)

The agent uses a minimal Mamba architecture (SSM) for O(L) sequence modeling.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Tuple

from src.agent_architectures import MultimodalSSMAgent

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def demo_semantic_sapr():
    print("=" * 60)
    print("SEMANTIC SAPR DEMO: MAMBA AGENT WORLD MODEL")
    print("=" * 60)

    # 1. Setup Semantic Space
    # -----------------------
    # We define a small "Universe" of concepts for the agent to reason about.
    # Vocabulary: [Safe, Danger, Goal, Wall, Path, Move, Stop, ...]
    vocab = [
        "Empty Space", "Wall", "Goal", "Trap", "Corridor", 
        "Move Forward", "Turn Left", "Turn Right", "Stop",
        "Safe", "Risky", "Collision", "Success"
    ]
    vocab_map = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)
    
    # 2. Initialize Mamba Agent
    # -------------------------
    print("\n[1] Initializing Mamba (SSM) Agent...")
    agent = MultimodalSSMAgent(
        vocab_size=vocab_size,
        img_size=32,
        embed_dim=64,   # Small dim for demo
        state_dim=128,  # Mamba state dim
        n_layers=2,
        n_actions=4     # Forward, Left, Right, Stop
    )
    
    # Mock some weights for the demo to make it "behave" slightly predictably
    # (In a real run, this would be trained)
    # E.g., World model: "Wall" + "Move Forward" -> "Collision"
    
    # 3. SAPR Execution Loop
    # ----------------------
    print("\n[2] Executing SAPR Step...")
    
    # --- S: STATE ---
    # Current observation: "Corridor" (Text) + [Mock Image]
    current_text_idx = torch.tensor([[vocab_map["Corridor"]]])
    current_img = torch.randn(1, 3, 32, 32) # Noise for demo
    
    # Forward pass
    output = agent(input_ids=current_text_idx, pixel_values=current_img)
    
    # Extract State Embedding (Latent state of the agent before heads)
    # We can't easily get the 'final' embedding from the wrapper without a hook,
    # but we can assume the SSM state or the value head input represents it.
    # For visualization, we'll assume the embedding layer output + processing.
    state_embedding = agent.token_embedding(current_text_idx).detach().numpy()[0,0]
    
    print(f"\n[S] State: 'Corridor'")
    print(f"    Vector Norm: {np.linalg.norm(state_embedding):.4f}")
    
    # --- A: ACTION ---
    # Agent picks an action based on policy logits
    action_logits = output.action_logits.detach().numpy()[0]
    action_probs = softmax(action_logits)
    actions = ["Forward", "Left", "Right", "Stop"]
    chosen_idx = np.argmax(action_probs)
    chosen_action = actions[chosen_idx]
    
    print(f"\n[A] Action: '{chosen_action}'")
    print(f"    Policy Dist: {dict(zip(actions, np.round(action_probs, 2)))}")
    
    # --- P: PROBABILITY (Transition) ---
    # The World Head predicts the next token ID P(s' | s, a)
    # Note: In this simple architecture, 'a' is implicitly part of the internal state 
    # if we were autoregressive. Here, the world model predicts 'next' from 'current'.
    # In a full recurrence, we'd feed 'action' back in.
    # We inspect the top predicted next tokens.
    
    wm_logits = output.aux_info["world_logits"].detach().numpy()[0]
    wm_probs = softmax(wm_logits)
    
    top_3_indices = np.argsort(wm_probs)[-3:][::-1]
    top_3_tokens = [vocab[i] for i in top_3_indices]
    top_3_probs = wm_probs[top_3_indices]
    
    print(f"\n[P] Transition Probabilities P(s'|s):")
    for tok, p in zip(top_3_tokens, top_3_probs):
        print(f"    -> '{tok}': {p:.4f}")
        
    # --- R: REWARD ---
    # The Value Head estimates desirability
    estimated_value = output.value.item()
    
    # Calculate a "Cost" for the action (e.g., energy cost)
    action_cost = 0.1 if chosen_action != "Stop" else 0.0
    
    print(f"\n[R] Reward Estimation:")
    print(f"    State Value V(s): {estimated_value:.4f}")
    print(f"    Action Cost C(a): {action_cost:.4f}")
    print(f"    Net Desirability: {estimated_value - action_cost:.4f}")
    
    # 4. Semantic Visualization (Projecting SAPR to 2D)
    # -----------------------------------------------
    print("\n[3] Visualizing Semantic Manifold...")
    
    # Get embeddings for all vocabulary items to show the "Map"
    all_tokens = torch.tensor(list(range(vocab_size)))
    all_embeddings = agent.token_embedding(all_tokens).detach().numpy()
    
    # Reduce dimensionality
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(all_embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.title("Semantic SAPR Manifold (Mamba Agent Internal State)")
    
    # Plot all concepts
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5, label="Concepts")
    for i, word in enumerate(vocab):
        plt.text(reduced_embeddings[i, 0]+0.02, reduced_embeddings[i, 1]+0.02, word, fontsize=9)
        
    # Highlight Current State
    s_idx = vocab_map["Corridor"]
    plt.scatter(reduced_embeddings[s_idx, 0], reduced_embeddings[s_idx, 1], c='blue', s=200, label="Current State (S)")
    
    # Highlight Predicted Next States (P)
    for i in top_3_indices:
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], c='green', s=100 + (wm_probs[i]*500), alpha=0.6, label="Predicted Next (P)")
        # Draw arrow
        plt.arrow(
            reduced_embeddings[s_idx, 0], reduced_embeddings[s_idx, 1],
            reduced_embeddings[i, 0] - reduced_embeddings[s_idx, 0],
            reduced_embeddings[i, 1] - reduced_embeddings[s_idx, 1],
            color='green', alpha=0.3, width=0.01, head_width=0.05
        )

    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('semantic_sapr_mamba.png')
    print("    Plot saved to 'semantic_sapr_mamba.png'")

if __name__ == "__main__":
    demo_semantic_sapr()
