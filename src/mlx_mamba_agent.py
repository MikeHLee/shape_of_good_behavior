"""
MLX Implementation of Mamba-based Multimodal Agent.
Optimized for Apple Silicon (M-series chips).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class AgentOutput:
    action_logits: mx.array
    value: mx.array
    next_state: List[mx.array]
    aux_info: Dict[str, Any]

class SSMLayer(nn.Module):
    """
    Simplified Mamba-like SSM Layer in MLX.
    """
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Projections
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.x_proj = nn.Linear(d_model, d_model + d_state + d_state) # delta, B, C
        self.out_proj = nn.Linear(d_model, d_model)
        
        # SSM Parameters
        self.A_log = mx.random.normal((d_model, d_state))
        self.D = mx.ones((d_model,))
        
    def __call__(self, u: mx.array, state: Optional[mx.array] = None) -> (mx.array, mx.array):
        """
        u: (B, D)
        state: (B, D, N)
        """
        B_dim = u.shape[0]
        
        # 1. Input Projection
        u_proj_gate = self.in_proj(u)
        u_proj, gate = mx.split(u_proj_gate, 2, axis=-1)
        
        # 2. SSM Parameters
        params = self.x_proj(u)
        
        # Split params: delta (D), B (N), C (N)
        # Note: MLX split might differ slightly in API, checking... 
        # mx.split(array, indices_or_sections, axis)
        # We need specific sizes: [d_model, d_state, d_state]
        
        # We can implement split manually with slicing for varied sizes
        delta = params[:, :self.d_model]
        B = params[:, self.d_model:self.d_model+self.d_state]
        C = params[:, self.d_model+self.d_state:]
        
        # Softplus delta
        delta = mx.logaddexp(delta, mx.array(0.0))
        
        # Discretize A
        A = -mx.exp(self.A_log) # (D, N)
        # dA = exp(A * delta) -> (B, D, N)
        # delta is (B, D). A is (D, N). 
        # We need broadcast: delta.unsqueeze(-1) * A.unsqueeze(0)
        dA = mx.exp(mx.expand_dims(delta, -1) * mx.expand_dims(A, 0))
        
        # Discretize B
        # dB = B * delta
        # B is (B, N), delta is (B, D).
        # We want dB to be (B, D, N) effectively?
        # In Mamba S6, B is (B, N).
        # We assume diagonal structure or similar. 
        # Let's map B (B, N) -> (B, D, N) via broadcasting?
        # Typically B is contracted with u.
        # Let's align with the PyTorch impl:
        # dB = B.unsqueeze(1) * delta.unsqueeze(-1)
        dB = mx.expand_dims(B, 1) * mx.expand_dims(delta, -1) # (B, 1, N) * (B, D, 1) -> (B, D, N)
        
        # State Update
        if state is None:
            state = mx.zeros((B_dim, self.d_model, self.d_state))
            
        u_expanded = mx.expand_dims(u_proj, -1) # (B, D, 1)
        new_state = dA * state + dB * u_expanded
        
        # Output
        # y = (C * state) ...
        # C is (B, N). new_state is (B, D, N).
        # We want (B, D).
        # y = sum(new_state * C.unsqueeze(1), dim=-1)
        y = mx.sum(new_state * mx.expand_dims(C, 1), axis=-1)
        
        y = y + self.D * u_proj
        
        # Gating
        y = y * nn.silu(gate)
        
        out = self.out_proj(y)
        
        return out, new_state

class MLXMambaAgent(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        img_size: int = 64,
        embed_dim: int = 256,
        state_dim: int = 512,
        n_layers: int = 4,
        n_actions: int = 10,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Vision Encoder (Simple CNN)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.vis_proj = nn.Linear(32 * (img_size // 4) ** 2, embed_dim)
        
        # Text Encoder
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # SSM Layers
        self.layers = [SSMLayer(embed_dim, state_dim) for _ in range(n_layers)]
        self.norms = [nn.LayerNorm(embed_dim) for _ in range(n_layers)]
        
        # Heads
        self.action_head = nn.Linear(embed_dim, n_actions)
        self.value_head = nn.Linear(embed_dim, 1)
        self.world_head = nn.Linear(embed_dim, vocab_size)
        
        # Final norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        hidden_states: Optional[List[mx.array]] = None
    ) -> AgentOutput:
        
        B = input_ids.shape[0] if input_ids is not None else pixel_values.shape[0]
        
        x = mx.zeros((B, self.embed_dim))
        
        if pixel_values is not None:
            # CNN Forward
            v = nn.relu(self.conv1(pixel_values))
            v = nn.relu(self.conv2(v))
            # Flatten
            v = v.reshape(B, -1)
            v = self.vis_proj(v)
            x = x + v
            
        if input_ids is not None:
            # Bag of words mean for stability
            t = self.token_embedding(input_ids)
            # Mask padding (assuming 0 is padding)
            mask = (input_ids != 0).astype(mx.float32)
            mask = mx.expand_dims(mask, -1)
            
            t = t * mask
            sum_t = mx.sum(t, axis=1)
            count = mx.sum(mask, axis=1)
            
            # Avoid division by zero
            x = x + sum_t / (count + 1e-6)
            
        # Initial processing
        
        new_states = []
        if hidden_states is None:
            hidden_states = [None] * len(self.layers)
            
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # Residual Connection + Pre-Norm
            residual = x
            x_norm = norm(x)
            
            out, h = layer(x_norm, hidden_states[i])
            x = residual + out
            
            new_states.append(h)
            
        x = self.final_norm(x)
            
        action_logits = self.action_head(x)
        value = self.value_head(x)
        world_logits = self.world_head(x)
        
        return AgentOutput(
            action_logits=action_logits,
            value=value,
            next_state=new_states,
            aux_info={"world_logits": world_logits}
        )
