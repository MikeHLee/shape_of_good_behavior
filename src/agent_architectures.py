import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

@dataclass
class AgentOutput:
    action_logits: torch.Tensor
    value: torch.Tensor
    next_state: torch.Tensor  # Hidden state for SSM/RNN
    aux_info: Dict[str, Any]

class MultimodalSSMAgent(nn.Module):
    """
    Smallest Possible Instruction Following Multimodal Sequence Model Agent.
    
    Architecture:
    1. Vision Encoder: Small CNN/ViT (simulated or loaded)
    2. Text Encoder: Token embedding
    3. State Space Model (SSM) Core: Mamba-like recurrence for O(1) inference
    4. Action Head: Policy projection
    5. World Model Head: Next token/observation prediction
    """
    def __init__(
        self,
        vocab_size: int = 1000,
        img_size: int = 64,
        embed_dim: int = 256,
        state_dim: int = 512,  # SSM state dimension
        n_layers: int = 4,
        n_actions: int = 10,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        
        # 1. Modality Encoders
        # Simple conv encoder for "vision"
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (img_size // 4) ** 2, embed_dim)
        )
        
        # Text embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. SSM Core (Simplified Mamba/S4 proxy)
        # We use a Gated Linear Recurrent Unit (GLRU) as a proxy for Mamba behavior
        # allowing parallel training and recurrent inference.
        self.layers = nn.ModuleList([
            SSMLayer(embed_dim, state_dim) for _ in range(n_layers)
        ])
        
        # 3. Heads
        self.action_head = nn.Linear(embed_dim, n_actions)
        self.value_head = nn.Linear(embed_dim, 1)
        self.world_head = nn.Linear(embed_dim, vocab_size)  # Next token prediction
        
        # Normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        hidden_states: Optional[List[torch.Tensor]] = None
    ) -> AgentOutput:
        
        batch_size = input_ids.shape[0] if input_ids is not None else pixel_values.shape[0]
        
        # Fuse modalities
        x = torch.zeros(batch_size, self.embed_dim, device=self.vision_encoder[0].weight.device)
        
        if pixel_values is not None:
            vis_embed = self.vision_encoder(pixel_values)
            x = x + vis_embed
            
        if input_ids is not None:
            # For simplicity, average bag-of-words or take last token if sequential
            # Here we assume single-step RL context for simplicity, or sum embeddings
            txt_embed = self.token_embedding(input_ids).sum(dim=1)
            x = x + txt_embed
            
        x = self.norm(x)
        
        # Pass through SSM layers
        new_hidden_states = []
        if hidden_states is None:
            hidden_states = [None] * len(self.layers)
            
        for i, layer in enumerate(self.layers):
            x, h = layer(x, hidden_states[i])
            new_hidden_states.append(h)
            
        # Heads
        action_logits = self.action_head(x)
        value = self.value_head(x)
        
        # Auxiliary (world model prediction)
        next_token_logits = self.world_head(x)
        
        return AgentOutput(
            action_logits=action_logits,
            value=value,
            next_state=new_hidden_states,
            aux_info={"world_logits": next_token_logits}
        )

class SSMLayer(nn.Module):
    """
    A simplified State Space Model layer.
    Approximates the selective scanning mechanism of Mamba.
    """
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Projections
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.x_proj = nn.Linear(d_model, d_model + d_state + d_state) # delta, B, C
        self.out_proj = nn.Linear(d_model, d_model)
        
        # S4/Mamba parameters (simplified)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))
        
    def forward(self, u, state=None):
        """
        u: (B, D) input
        state: (B, D, N) hidden state
        """
        # 1. Project input
        u_proj, gate = self.in_proj(u).chunk(2, dim=-1)
        
        # 2. SSM Parameters from input (Selection mechanism)
        # x_proj => delta, B, C
        # Simplified: We treat this as a standard RNN step for the agent loop
        # x_t = A x_{t-1} + B u_t
        # y_t = C x_t + D u_t
        
        # In a real Mamba, these are time-varying. 
        # Here we simulate the recurrence:
        
        if state is None:
            state = torch.zeros(u.shape[0], self.d_model, self.d_state, device=u.device)
            
        # Compute dynamic B, C, delta
        params = self.x_proj(u)
        delta, B, C = torch.split(params, [self.d_model, self.d_state, self.d_state], dim=-1)
        
        # Softplus delta to ensure positive time-step
        delta = F.softplus(delta)
        
        # Discretize A (Zero-Order Hold)
        # exp(A * delta)
        # Simplified diagonal A
        A = -torch.exp(self.A_log) # Ensure stable A
        dA = torch.exp(A.unsqueeze(0) * delta.unsqueeze(-1)) # (B, D, N)
        
        # Discretize B
        dB = B.unsqueeze(1) * delta.unsqueeze(-1) # (B, 1, N) * (B, D, 1) -> (B, D, N)
        # Approximate: B is usually (B, N), we broadcast
        
        # Update State
        # x_t = dA * x_{t-1} + dB * u_proj
        u_expanded = u_proj.unsqueeze(-1) # (B, D, 1)
        new_state = dA * state + dB * u_expanded
        
        # Output
        # y = C * x_t + D * u
        # C is (B, D) ? In Mamba C is (B, N).
        # We did C projection to D_model above, let's just project state to output
        y = (new_state * C.unsqueeze(1)).sum(dim=-1) # (B, D)
        
        y = y + self.D * u_proj
        
        # Gating
        y = y * F.silu(gate)
        
        out = self.out_proj(y)
        
        return out, new_state
