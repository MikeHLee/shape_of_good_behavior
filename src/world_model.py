"""
World Model for Natural Language State Spaces

This module implements Proposal (B): a learned function that predicts evolution
pathways of the world state sequence given past values and potential action tokens.

Key Distinction (Critical for Tensor RL):
- ENVIRONMENT/ORACLE: The actual dynamics of the world (e.g., game engine, GPT-4 simulator)
- WORLD MODEL: The agent's *learned approximation* of environment dynamics
- POLICY: The agent's action selection given its beliefs

The World Model learns P(s_{t+1} | s_{\leq t}, a_{\leq t}) from observed transitions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorldModelPrediction:
    """
    Prediction from the World Model.
    
    Contains both the predicted next state and uncertainty estimates,
    enabling tree search and risk-aware planning.
    """
    # Core prediction
    predicted_state: np.ndarray          # Mean predicted next state embedding
    predicted_state_text: Optional[str]  # Optional text decoding
    
    # Uncertainty quantification
    uncertainty: float                   # Scalar uncertainty (e.g., entropy)
    variance: Optional[np.ndarray]       # Full covariance if available
    
    # Distribution over possible outcomes (for multi-modal transitions)
    outcome_distribution: Optional[Dict[str, float]] = None  # {state_text: probability}
    
    # Metadata
    confidence: float = 1.0              # Model's self-assessed confidence
    is_terminal: bool = False            # Does this predict episode termination?


@dataclass
class TransitionData:
    """A single observed transition for training the world model."""
    state: np.ndarray
    state_text: str
    action: Union[int, np.ndarray]
    action_text: str
    next_state: np.ndarray
    next_state_text: str
    reward: float
    done: bool


class BaseWorldModel(ABC):
    """
    Abstract base class for World Models.
    
    A World Model approximates the environment's transition dynamics:
        P(s_{t+1} | s_t, a_t) ≈ WorldModel(s_t, a_t)
    
    This is distinct from:
    - The actual environment (which generates true transitions)
    - The policy (which selects actions)
    - The value function (which estimates cumulative reward)
    """
    
    @abstractmethod
    def predict(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        state_text: Optional[str] = None,
        action_text: Optional[str] = None,
    ) -> WorldModelPrediction:
        """
        Predict the next state given current state and action.
        
        Args:
            state: Current state embedding
            action: Action (index or embedding)
            state_text: Optional text description of state
            action_text: Optional text description of action
            
        Returns:
            WorldModelPrediction with predicted next state and uncertainty
        """
        pass
    
    @abstractmethod
    def predict_trajectory(
        self,
        initial_state: np.ndarray,
        action_sequence: List[Union[int, np.ndarray]],
        initial_state_text: Optional[str] = None,
        action_texts: Optional[List[str]] = None,
    ) -> List[WorldModelPrediction]:
        """
        Predict a full trajectory by rolling out the world model.
        
        This enables planning and tree search in semantic space.
        """
        pass
    
    @abstractmethod
    def train_step(
        self,
        transitions: List[TransitionData],
    ) -> Dict[str, float]:
        """
        Update the world model from observed transitions.
        
        Returns:
            Dict of training metrics (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def compute_prediction_error(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        true_next_state: np.ndarray,
    ) -> float:
        """
        Compute the prediction error for a single transition.
        
        This is used for:
        1. Training the world model
        2. Detecting distribution shift
        3. Identifying novel states
        """
        pass


class TransformerWorldModel(BaseWorldModel, nn.Module):
    """
    Transformer-based World Model for semantic state spaces.
    
    Architecture:
    - Encoder: Processes (state, action) pairs into context
    - Decoder: Predicts next state distribution
    
    Supports both discrete (text token) and continuous (embedding) outputs.
    """
    
    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_actions: Optional[int] = None,  # For discrete action spaces
        dropout: float = 0.1,
        predict_uncertainty: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_actions = num_actions
        self.predict_uncertainty = predict_uncertainty
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Action encoder
        if num_actions is not None:
            # Discrete action space
            self.action_embedding = nn.Embedding(num_actions, hidden_dim)
        else:
            # Continuous action space (embedding input)
            self.action_encoder = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
            )
        
        # Transformer for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        if predict_uncertainty:
            # Predict variance for uncertainty quantification
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),  # Ensure positive variance
            )
        
        # Terminal state prediction
        self.done_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the world model.
        
        Args:
            state: State embeddings [batch, embed_dim]
            action: Action indices [batch] or embeddings [batch, embed_dim]
            
        Returns:
            (predicted_next_state, uncertainty, done_prob)
        """
        # Encode state
        state_enc = self.state_encoder(state)  # [batch, hidden]
        
        # Encode action
        if self.num_actions is not None and action.dim() == 1:
            action_enc = self.action_embedding(action)  # [batch, hidden]
        else:
            action_enc = self.action_encoder(action)  # [batch, hidden]
        
        # Combine as sequence [state, action]
        sequence = torch.stack([state_enc, action_enc], dim=1)  # [batch, 2, hidden]
        
        # Transformer processing
        transformed = self.transformer(sequence)  # [batch, 2, hidden]
        
        # Use the action position's output for prediction
        hidden = transformed[:, 1, :]  # [batch, hidden]
        
        # Predict next state
        predicted_state = self.next_state_head(hidden)  # [batch, embed_dim]
        
        # Predict uncertainty
        if self.predict_uncertainty:
            uncertainty = self.uncertainty_head(hidden).squeeze(-1)  # [batch]
        else:
            uncertainty = torch.zeros(state.shape[0], device=self.device)
        
        # Predict termination
        done_prob = self.done_head(hidden).squeeze(-1)  # [batch]
        
        return predicted_state, uncertainty, done_prob
    
    def predict(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        state_text: Optional[str] = None,
        action_text: Optional[str] = None,
    ) -> WorldModelPrediction:
        """Predict next state from current state and action."""
        self.eval()
        
        with torch.no_grad():
            # Convert to tensors
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            
            if isinstance(action, int):
                action_t = torch.tensor([action], dtype=torch.long, device=self.device)
            else:
                action_t = torch.tensor(action, dtype=torch.float32, device=self.device)
                if action_t.dim() == 1:
                    action_t = action_t.unsqueeze(0)
            
            # Forward pass
            pred_state, uncertainty, done_prob = self.forward(state_t, action_t)
            
            return WorldModelPrediction(
                predicted_state=pred_state[0].cpu().numpy(),
                predicted_state_text=None,  # Would need decoder for this
                uncertainty=float(uncertainty[0].item()),
                variance=None,
                confidence=1.0 - float(uncertainty[0].item()),
                is_terminal=done_prob[0].item() > 0.5,
            )
    
    def predict_trajectory(
        self,
        initial_state: np.ndarray,
        action_sequence: List[Union[int, np.ndarray]],
        initial_state_text: Optional[str] = None,
        action_texts: Optional[List[str]] = None,
    ) -> List[WorldModelPrediction]:
        """Roll out the world model for a sequence of actions."""
        predictions = []
        current_state = initial_state
        
        for i, action in enumerate(action_sequence):
            pred = self.predict(
                current_state,
                action,
                state_text=initial_state_text if i == 0 else None,
                action_text=action_texts[i] if action_texts else None,
            )
            predictions.append(pred)
            
            # Use predicted state for next step
            current_state = pred.predicted_state
            
            if pred.is_terminal:
                break
        
        return predictions
    
    def train_step(
        self,
        transitions: List[TransitionData],
    ) -> Dict[str, float]:
        """Train the world model on a batch of transitions."""
        self.train()
        
        # Prepare batch
        states = torch.tensor(
            np.array([t.state for t in transitions]),
            dtype=torch.float32, device=self.device
        )
        
        if isinstance(transitions[0].action, int):
            actions = torch.tensor(
                [t.action for t in transitions],
                dtype=torch.long, device=self.device
            )
        else:
            actions = torch.tensor(
                np.array([t.action for t in transitions]),
                dtype=torch.float32, device=self.device
            )
        
        next_states = torch.tensor(
            np.array([t.next_state for t in transitions]),
            dtype=torch.float32, device=self.device
        )
        
        dones = torch.tensor(
            [float(t.done) for t in transitions],
            dtype=torch.float32, device=self.device
        )
        
        # Forward pass
        pred_states, uncertainties, done_probs = self.forward(states, actions)
        
        # Compute losses
        state_loss = F.mse_loss(pred_states, next_states)
        done_loss = F.binary_cross_entropy(done_probs, dones)
        
        # Total loss
        total_loss = state_loss + 0.1 * done_loss
        
        return {
            "total_loss": float(total_loss.item()),
            "state_loss": float(state_loss.item()),
            "done_loss": float(done_loss.item()),
            "mean_uncertainty": float(uncertainties.mean().item()),
        }
    
    def compute_prediction_error(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        true_next_state: np.ndarray,
    ) -> float:
        """Compute prediction error for a single transition."""
        pred = self.predict(state, action)
        return float(np.linalg.norm(pred.predicted_state - true_next_state))


class OracleEnvironment(ABC):
    """
    Abstract interface for the true environment/oracle.
    
    This is SEPARATE from the World Model:
    - Oracle: Generates true transitions (may be simulator, LLM, or real world)
    - World Model: Agent's learned approximation of the oracle
    
    The agent interacts with the Oracle to collect transitions,
    which are then used to train the World Model.
    """
    
    @abstractmethod
    def step(
        self,
        state_text: str,
        action_text: str,
    ) -> Tuple[str, float, bool, Dict]:
        """
        Execute action in the environment.
        
        Args:
            state_text: Current state description
            action_text: Action to take
            
        Returns:
            (next_state_text, reward, done, info)
        """
        pass
    
    @abstractmethod
    def reset(self) -> str:
        """Reset environment and return initial state."""
        pass


class LLMOracle(OracleEnvironment):
    """
    LLM-based environment oracle.
    
    Uses a large language model (e.g., GPT-4) to simulate environment dynamics.
    This enables semantic reasoning about state transitions.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        system_prompt: Optional[str] = None,
        api_client: Any = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.api_client = api_client
        self.history: List[Dict] = []
    
    def _default_system_prompt(self) -> str:
        return """You are a world simulator. Given a current state and an action,
        you must predict the next state, reward, and whether the episode ends.
        
        Respond in JSON format:
        {
            "next_state": "description of new state",
            "reward": <number between -1 and 1>,
            "done": <true/false>,
            "explanation": "why this transition occurred"
        }
        
        Be consistent with physics and causality. Actions have logical consequences."""
    
    def step(
        self,
        state_text: str,
        action_text: str,
    ) -> Tuple[str, float, bool, Dict]:
        """Execute action using LLM as world simulator."""
        # This is a placeholder - actual implementation would call the LLM API
        
        prompt = f"""Current state: {state_text}
Action taken: {action_text}

What happens next?"""
        
        # In practice, this would call self.api_client.complete(prompt)
        # For now, return placeholder
        
        return (
            f"{state_text} [after {action_text}]",
            0.0,
            False,
            {"oracle_type": "llm", "model": self.model_name}
        )
    
    def reset(self) -> str:
        """Reset the environment."""
        self.history = []
        return "Initial state"


class RuleBasedOracle(OracleEnvironment):
    """
    Rule-based environment oracle for testing and debugging.
    
    Uses explicit transition rules rather than learned models.
    Useful for verifying World Model learning.
    """
    
    def __init__(
        self,
        transition_rules: Dict[Tuple[str, str], Tuple[str, float, bool]] = None,
    ):
        """
        Args:
            transition_rules: Dict mapping (state_pattern, action) to (next_state, reward, done)
        """
        self.rules = transition_rules or {}
        self.current_state = "start"
    
    def add_rule(
        self,
        state_pattern: str,
        action: str,
        next_state: str,
        reward: float,
        done: bool = False,
    ):
        """Add a transition rule."""
        self.rules[(state_pattern, action)] = (next_state, reward, done)
    
    def step(
        self,
        state_text: str,
        action_text: str,
    ) -> Tuple[str, float, bool, Dict]:
        """Execute action using rule-based transitions."""
        # Try exact match first
        if (state_text, action_text) in self.rules:
            next_state, reward, done = self.rules[(state_text, action_text)]
            self.current_state = next_state
            return next_state, reward, done, {"matched_rule": True}
        
        # Try pattern matching
        for (state_pattern, action), (next_state, reward, done) in self.rules.items():
            if state_pattern in state_text and action == action_text:
                self.current_state = next_state
                return next_state, reward, done, {"matched_rule": True}
        
        # Default: no change
        return state_text, 0.0, False, {"matched_rule": False}
    
    def reset(self) -> str:
        """Reset to initial state."""
        self.current_state = "start"
        return self.current_state
