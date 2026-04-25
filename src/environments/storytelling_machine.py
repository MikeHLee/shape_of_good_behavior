"""
Storytelling Machine Environment

A Semantic Turing Machine environment where:
- States are natural language scene descriptions (Kolmogorov minimal)
- Actions are MCP-style tool calls
- Transitions are causal narrative evolutions
- Observations are partial (agent has beliefs, not omniscience)

This implements the "text adventure as general intelligence" paradigm.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import hashlib

import numpy as np
import torch
import torch.nn as nn

from .base import SGPOWrapperBase, RiemannianMetricBase


class ActionType(Enum):
    """MCP-style action categories."""
    OBSERVE = "observe"      # Gather information (restriction map)
    ACT = "act"              # Modify state (transition)
    QUERY = "query"          # Ask oracle/environment
    COMPLETE = "complete"    # Signal task completion


@dataclass
class MCPAction:
    """An MCP-style tool call action."""
    action_type: ActionType
    tool_name: str
    parameters: Dict[str, Any]
    raw_text: Optional[str] = None  # Original LLM output if applicable
    
    def to_dict(self) -> Dict:
        return {
            "type": self.action_type.value,
            "tool": self.tool_name,
            "params": self.parameters,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "MCPAction":
        return cls(
            action_type=ActionType(d["type"]),
            tool_name=d["tool"],
            parameters=d.get("params", {}),
        )
    
    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self.parameters.items())
        return f"{self.tool_name}({params_str})"


@dataclass
class BeliefState:
    """
    Agent's belief about the world state.
    
    Captures uncertainty and partial observability.
    """
    observation: str                    # What the agent directly sees
    belief_summary: str                 # Agent's interpretation/beliefs
    confidence: float = 0.5             # How certain (0=uncertain, 1=certain)
    possible_states: List[str] = field(default_factory=list)  # Alternative hypotheses
    
    def to_embedding_text(self) -> str:
        """Generate text for embedding that captures belief uncertainty."""
        uncertainty_clause = ""
        if self.confidence < 0.3:
            uncertainty_clause = " [HIGH UNCERTAINTY]"
        elif self.confidence < 0.7:
            uncertainty_clause = " [MODERATE UNCERTAINTY]"
        
        alternatives = ""
        if self.possible_states:
            alternatives = f" Alternatives: {'; '.join(self.possible_states[:3])}"
        
        return f"{self.observation} | Belief: {self.belief_summary}{uncertainty_clause}{alternatives}"


@dataclass
class Page:
    """
    A single 'page' of the storytelling machine's tape.
    
    Represents one state in the narrative.
    """
    scene_description: str              # The Kolmogorov-minimal description
    hidden_state: Optional[Dict] = None # True underlying state (for oracle)
    step_number: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def hash(self) -> str:
        """Content-addressable hash for the page."""
        return hashlib.md5(self.scene_description.encode()).hexdigest()[:12]


@dataclass
class Transition:
    """Records a state transition in the narrative."""
    from_page: Page
    action: MCPAction
    to_page: Page
    reward_signal: Optional[float] = None
    critique: Optional[str] = None
    
    def to_training_example(self) -> Dict:
        """Format for fine-tuning data."""
        return {
            "scene": self.from_page.scene_description,
            "action": str(self.action),
            "next_scene": self.to_page.scene_description,
            "reward": self.reward_signal,
            "critique": self.critique,
        }


class WorldOracle(ABC):
    """
    Abstract oracle that governs world transitions.
    
    Can be:
    - Rule-based (text adventure engine)
    - LLM-based (GPT-4 simulating consequences)
    - Human-in-the-loop
    """
    
    @abstractmethod
    def transition(self, page: Page, action: MCPAction) -> Tuple[Page, Dict]:
        """
        Apply action to current page, return new page.
        
        Returns:
            Tuple of (new_page, transition_info)
        """
        pass
    
    @abstractmethod
    def render_observation(self, page: Page, agent_state: Dict) -> str:
        """
        Render what the agent can observe from the true state.
        
        Implements partial observability.
        """
        pass
    
    @abstractmethod
    def is_terminal(self, page: Page) -> Tuple[bool, Optional[str]]:
        """
        Check if the current page is a terminal state.
        
        Returns:
            Tuple of (is_terminal, reason)
        """
        pass


class LLMOracle(WorldOracle):
    """
    Oracle powered by an LLM (e.g., GPT-4).
    
    Simulates world dynamics through prompted generation.
    """
    
    def __init__(
        self,
        llm_client: Any,  # OpenAI client or similar
        model: str = "gpt-4o",
        system_prompt: Optional[str] = None,
    ):
        self.client = llm_client
        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.history: List[Dict] = []
    
    def _default_system_prompt(self) -> str:
        return """You are a world simulator for an AI training environment.

Given a scene description and an action, output the resulting scene.

Rules:
1. Be concise - use Kolmogorov-minimal descriptions (shortest that captures causally relevant info)
2. Be consistent - respect established world rules
3. Be realistic - actions have plausible consequences
4. Flag terminal states - indicate if the scenario has concluded (success/failure)

Output format:
SCENE: [new scene description]
TERMINAL: [yes/no]
REASON: [if terminal, why]
HIDDEN: [any hidden state changes, JSON]"""

    def transition(self, page: Page, action: MCPAction) -> Tuple[Page, Dict]:
        prompt = f"""Current scene: {page.scene_description}

Action taken: {action}

What happens next?"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        
        output = response.choices[0].message.content
        new_page, info = self._parse_oracle_response(output, page.step_number + 1)
        
        self.history.append({
            "from": page.scene_description,
            "action": str(action),
            "to": new_page.scene_description,
        })
        
        return new_page, info
    
    def _parse_oracle_response(self, response: str, step: int) -> Tuple[Page, Dict]:
        """Parse LLM response into Page and metadata."""
        lines = response.strip().split("\n")
        scene = ""
        terminal = False
        reason = None
        hidden = {}
        
        for line in lines:
            if line.startswith("SCENE:"):
                scene = line[6:].strip()
            elif line.startswith("TERMINAL:"):
                terminal = line[9:].strip().lower() == "yes"
            elif line.startswith("REASON:"):
                reason = line[7:].strip()
            elif line.startswith("HIDDEN:"):
                try:
                    hidden = json.loads(line[7:].strip())
                except json.JSONDecodeError:
                    hidden = {"raw": line[7:].strip()}
        
        if not scene:
            scene = response  # Fallback: use whole response
        
        new_page = Page(
            scene_description=scene,
            hidden_state=hidden,
            step_number=step,
            metadata={"terminal": terminal, "reason": reason},
        )
        
        return new_page, {"terminal": terminal, "reason": reason}
    
    def render_observation(self, page: Page, agent_state: Dict) -> str:
        # For LLM oracle, observation is the scene itself (partial obs handled elsewhere)
        return page.scene_description
    
    def is_terminal(self, page: Page) -> Tuple[bool, Optional[str]]:
        return page.metadata.get("terminal", False), page.metadata.get("reason")


class SemanticEmbeddingMetric(RiemannianMetricBase):
    """
    Riemannian metric defined on semantic embeddings.
    
    Black holes are regions in embedding space corresponding to
    dangerous/forbidden narrative states.
    """
    
    def __init__(
        self,
        embed_dim: int,
        embedding_model: Optional[Any] = None,
        black_hole_embeddings: Optional[List[np.ndarray]] = None,
        black_hole_radii: Optional[List[float]] = None,
    ):
        super().__init__(state_dim=embed_dim)
        self.embed_dim = embed_dim
        self.embedding_model = embedding_model
        
        self._black_hole_centers = black_hole_embeddings or []
        self._black_hole_radii = black_hole_radii or []
        
        # Learnable parameters for metric
        self.base_metric = nn.Parameter(torch.ones(1))
        self.severity_scale = nn.Parameter(torch.tensor(10.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute metric at embedding position x.
        
        g(x) = base + Σ severity / (||x - center|| - radius)²
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        metric = self.base_metric.expand(batch_size, 1)
        
        for center, radius in zip(self._black_hole_centers, self._black_hole_radii):
            center_t = torch.tensor(center, dtype=x.dtype, device=x.device)
            dist = torch.norm(x - center_t, dim=-1, keepdim=True)
            
            # Avoid division by zero; clip distance
            safe_dist = torch.clamp(dist - radius, min=1e-3)
            
            # Add singularity contribution
            metric = metric + self.severity_scale / (safe_dist ** 2)
        
        return metric
    
    def get_black_hole_centers(self) -> List[np.ndarray]:
        return self._black_hole_centers
    
    def get_event_horizons(self) -> List[float]:
        return self._black_hole_radii
    
    def add_black_hole(self, center: np.ndarray, radius: float):
        """Add a new black hole region."""
        self._black_hole_centers.append(center)
        self._black_hole_radii.append(radius)
    
    def add_black_hole_from_text(self, text: str, radius: float = 0.5):
        """Add black hole by embedding a text description."""
        if self.embedding_model is None:
            raise ValueError("No embedding model provided")
        
        embedding = self.embedding_model.encode(text)
        self.add_black_hole(embedding, radius)


class StorytellingMachine:
    """
    The main Semantic Turing Machine environment.
    
    Manages:
    - The tape (sequence of Pages)
    - The agent's belief state
    - Transitions via Oracle
    - Embedding and topological analysis
    """
    
    def __init__(
        self,
        oracle: WorldOracle,
        embedding_model: Optional[Any] = None,
        action_space: Optional[List[Dict]] = None,
        max_steps: int = 100,
    ):
        self.oracle = oracle
        self.embedding_model = embedding_model
        self.action_space_def = action_space or self._default_action_space()
        self.max_steps = max_steps
        
        # State
        self.tape: List[Page] = []
        self.transitions: List[Transition] = []
        self.current_step = 0
        self.belief_state: Optional[BeliefState] = None
        
        # Metric for SGPO
        embed_dim = 384 if embedding_model is None else self._get_embed_dim()
        self.metric = SemanticEmbeddingMetric(embed_dim, embedding_model)
    
    def _default_action_space(self) -> List[Dict]:
        """Default MCP-style action definitions."""
        return [
            {
                "name": "look",
                "type": ActionType.OBSERVE,
                "description": "Observe the current scene more carefully",
                "params": {"target": "string"},
            },
            {
                "name": "interact",
                "type": ActionType.ACT,
                "description": "Interact with an object or entity",
                "params": {"target": "string", "action": "string"},
            },
            {
                "name": "move",
                "type": ActionType.ACT,
                "description": "Move to a different location",
                "params": {"direction": "string"},
            },
            {
                "name": "ask",
                "type": ActionType.QUERY,
                "description": "Ask about something",
                "params": {"question": "string"},
            },
            {
                "name": "complete",
                "type": ActionType.COMPLETE,
                "description": "Signal task completion",
                "params": {"status": "string", "summary": "string"},
            },
        ]
    
    def _get_embed_dim(self) -> int:
        """Detect embedding dimension from model."""
        test_embed = self.embedding_model.encode("test")
        return len(test_embed)
    
    def reset(self, initial_scene: str) -> Tuple[BeliefState, Dict]:
        """Reset the machine with an initial scene."""
        self.tape = [Page(scene_description=initial_scene, step_number=0)]
        self.transitions = []
        self.current_step = 0
        
        observation = self.oracle.render_observation(self.tape[0], {})
        self.belief_state = BeliefState(
            observation=observation,
            belief_summary="Starting scenario. Objective unclear.",
            confidence=0.5,
        )
        
        info = {
            "page_hash": self.tape[0].hash(),
            "step": 0,
        }
        
        return self.belief_state, info
    
    def step(self, action: MCPAction) -> Tuple[BeliefState, float, bool, bool, Dict]:
        """
        Execute an action and transition to the next state.
        
        Returns:
            Tuple of (belief_state, reward, terminated, truncated, info)
        """
        current_page = self.tape[-1]
        
        # Execute transition
        new_page, transition_info = self.oracle.transition(current_page, action)
        self.tape.append(new_page)
        self.current_step += 1
        
        # Record transition (reward filled in later by human/critic)
        transition = Transition(
            from_page=current_page,
            action=action,
            to_page=new_page,
        )
        self.transitions.append(transition)
        
        # Update belief state
        observation = self.oracle.render_observation(new_page, {})
        self.belief_state = BeliefState(
            observation=observation,
            belief_summary=f"After {action.tool_name}: {observation[:100]}...",
            confidence=0.6,  # Slightly more confident after action
        )
        
        # Check termination
        terminated, reason = self.oracle.is_terminal(new_page)
        truncated = self.current_step >= self.max_steps
        
        # Compute metric-based reward signal
        reward = 0.0  # Base reward; topological reward added by critic
        if self.embedding_model is not None:
            embedding = self.embedding_model.encode(observation)
            metric_value = float(self.metric(torch.tensor(embedding)))
            # Penalize being near black holes
            if metric_value > 10.0:
                reward -= np.log(metric_value)
        
        info = {
            "page_hash": new_page.hash(),
            "step": self.current_step,
            "terminal_reason": reason,
            "transition_info": transition_info,
        }
        
        return self.belief_state, reward, terminated, truncated, info
    
    def get_trajectory(self) -> List[Dict]:
        """Export trajectory for training/analysis."""
        return [t.to_training_example() for t in self.transitions]
    
    def get_embeddings(self) -> np.ndarray:
        """Get embeddings of all visited states."""
        if self.embedding_model is None:
            raise ValueError("No embedding model provided")
        
        texts = [p.scene_description for p in self.tape]
        return self.embedding_model.encode(texts)
    
    def annotate_transition(
        self,
        index: int,
        reward: float,
        critique: Optional[str] = None,
    ):
        """Add human feedback to a transition."""
        if 0 <= index < len(self.transitions):
            self.transitions[index].reward_signal = reward
            self.transitions[index].critique = critique


class StorytellingSGPOWrapper(SGPOWrapperBase):
    """
    SGPO wrapper for the Storytelling Machine.
    
    Integrates semantic embeddings with Riemannian metric for
    Sheaf-Geodesic Policy Optimization.
    """
    
    def __init__(
        self,
        machine: StorytellingMachine,
        embedding_model: Any,
    ):
        self.machine = machine
        self.embedding_model = embedding_model
        self._metric = machine.metric
        
        # Observation space is embedding dimension
        self.embed_dim = machine.metric.embed_dim
    
    @property
    def metric(self) -> RiemannianMetricBase:
        return self._metric
    
    def _create_default_metric(self) -> RiemannianMetricBase:
        return self.machine.metric
    
    def _extract_hazards(self) -> Tuple[List[np.ndarray], List[float]]:
        return (
            self.machine.metric.get_black_hole_centers(),
            self.machine.metric.get_event_horizons(),
        )
    
    def reset(self, initial_scene: str, **kwargs) -> Tuple[np.ndarray, Dict]:
        belief, info = self.machine.reset(initial_scene)
        
        # Embed the belief state
        embedding = self.embedding_model.encode(belief.to_embedding_text())
        info["belief"] = belief
        info["metric_value"] = float(self.metric(torch.tensor(embedding)))
        
        return embedding, info
    
    def step(self, action: MCPAction) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        belief, reward, terminated, truncated, info = self.machine.step(action)
        
        # Embed the new belief state
        embedding = self.embedding_model.encode(belief.to_embedding_text())
        
        # Compute metric
        metric_value = float(self.metric(torch.tensor(embedding)))
        info["belief"] = belief
        info["metric_value"] = metric_value
        info["in_black_hole"] = metric_value > 100.0
        
        return embedding, reward, terminated, truncated, info


def create_storytelling_env(
    oracle_type: str = "llm",
    llm_client: Optional[Any] = None,
    embedding_model: Optional[Any] = None,
    **kwargs,
) -> StorytellingSGPOWrapper:
    """
    Factory function to create a Storytelling Machine environment.
    
    Args:
        oracle_type: "llm" or "rule" (rule-based coming soon)
        llm_client: OpenAI client or compatible
        embedding_model: sentence-transformers model
        **kwargs: Additional arguments for StorytellingMachine
    
    Returns:
        StorytellingSGPOWrapper ready for SGPO training
    """
    if oracle_type == "llm":
        if llm_client is None:
            raise ValueError("llm_client required for LLM oracle")
        oracle = LLMOracle(llm_client)
    else:
        raise NotImplementedError(f"Oracle type {oracle_type} not implemented")
    
    machine = StorytellingMachine(
        oracle=oracle,
        embedding_model=embedding_model,
        **kwargs,
    )
    
    return StorytellingSGPOWrapper(machine, embedding_model)
