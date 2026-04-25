"""
TALES/TextWorld SGPO Adapter

Integrates Sheaf-Geodesic Policy Optimization with text adventure environments
from Microsoft's TALES suite (TextWorld, ALFWorld, ScienceWorld, Jericho).

Key Innovation:
- Text adventures have IRREVERSIBLE STATES (use key → can't get it back)
- Model irreversible negative states as black holes with event horizons
- Geodesics guide agent away from sequences leading to unwinnable states

State Space Challenges:
- Discrete, high-dimensional (text)
- Partial observability (can't see whole game state)
- Long-horizon dependencies (action now affects outcome 100 steps later)

PROPOSED UPSTREAM CONTRIBUTION:
This module includes IrreversibilityDetector and WinnabilityTracker that
could be contributed to TALES/TextWorld as optional safety analysis tools.

Installation:
    pip install textworld  # Core TextWorld
    pip install jericho    # Classic text games
    # For TALES: clone https://github.com/microsoft/tale-suite

Usage:
    from environments import create_textworld_gpo_env
    
    env = create_textworld_gpo_env(
        "tw-simple",
        track_irreversibility=True
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
import numpy as np
import torch
import torch.nn as nn
import hashlib

from .base import SGPOWrapperBase, RiemannianMetricBase


class IrreversibilityType(Enum):
    """Types of irreversible actions in text adventures."""
    ITEM_CONSUMPTION = "item_consumption"      # eat food, use key
    ITEM_DESTRUCTION = "item_destruction"      # break object
    STATE_CHANGE = "state_change"              # kill NPC, close door forever
    LOCATION_LOCK = "location_lock"            # enter one-way passage
    QUEST_FAILURE = "quest_failure"            # fail critical objective


@dataclass
class IrreversibleAction:
    """Represents a potentially irreversible action."""
    action_text: str
    irreversibility_type: IrreversibilityType
    severity: float  # How bad is this if wrong?
    keywords: List[str] = field(default_factory=list)
    
    def matches(self, action: str) -> bool:
        action_lower = action.lower()
        return any(kw in action_lower for kw in self.keywords)


COMMON_IRREVERSIBLE_ACTIONS = [
    IrreversibleAction(
        action_text="eat",
        irreversibility_type=IrreversibilityType.ITEM_CONSUMPTION,
        severity=0.5,
        keywords=["eat", "drink", "consume", "swallow"],
    ),
    IrreversibleAction(
        action_text="use key",
        irreversibility_type=IrreversibilityType.ITEM_CONSUMPTION,
        severity=0.7,
        keywords=["use key", "unlock with", "insert key"],
    ),
    IrreversibleAction(
        action_text="destroy",
        irreversibility_type=IrreversibilityType.ITEM_DESTRUCTION,
        severity=0.8,
        keywords=["break", "destroy", "smash", "burn", "tear"],
    ),
    IrreversibleAction(
        action_text="kill",
        irreversibility_type=IrreversibilityType.STATE_CHANGE,
        severity=0.9,
        keywords=["kill", "attack", "murder", "slay"],
    ),
    IrreversibleAction(
        action_text="enter one-way",
        irreversibility_type=IrreversibilityType.LOCATION_LOCK,
        severity=0.6,
        keywords=["jump", "fall", "slide", "enter pit", "go down hole"],
    ),
    IrreversibleAction(
        action_text="give away",
        irreversibility_type=IrreversibilityType.ITEM_CONSUMPTION,
        severity=0.5,
        keywords=["give", "hand over", "offer", "trade"],
    ),
    IrreversibleAction(
        action_text="drop in",
        irreversibility_type=IrreversibilityType.ITEM_DESTRUCTION,
        severity=0.7,
        keywords=["drop in", "throw in", "put in water", "put in fire"],
    ),
]


@dataclass
class GameState:
    """Represents a discrete game state for tracking."""
    observation: str
    inventory: Set[str]
    location: str
    score: int
    step: int
    
    def hash(self) -> str:
        content = f"{self.observation[:100]}|{sorted(self.inventory)}|{self.location}|{self.score}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class IrreversibilityDetector:
    """
    Detects potentially irreversible actions in text adventures.
    
    PROPOSED UPSTREAM CONTRIBUTION:
    Could be added to TextWorld/TALES as a safety analysis tool.
    """
    
    def __init__(
        self,
        irreversible_actions: Optional[List[IrreversibleAction]] = None,
        custom_detector: Optional[Callable[[str, str], float]] = None,
    ):
        self.irreversible_actions = irreversible_actions or COMMON_IRREVERSIBLE_ACTIONS
        self.custom_detector = custom_detector
    
    def analyze_action(
        self,
        action: str,
        observation: str,
    ) -> Tuple[float, Optional[IrreversibleAction]]:
        """
        Analyze an action for irreversibility risk.
        
        Returns:
            Tuple of (risk_score, matched_action)
        """
        for irrev_action in self.irreversible_actions:
            if irrev_action.matches(action):
                return irrev_action.severity, irrev_action
        
        if self.custom_detector:
            custom_score = self.custom_detector(action, observation)
            if custom_score > 0.3:
                return custom_score, None
        
        return 0.0, None


class WinnabilityTracker:
    """
    Tracks whether game states are likely winnable.
    
    Uses heuristics:
    - Have we seen higher scores from similar states?
    - Are we making progress (score increasing)?
    - Have we visited this state before without progress?
    
    PROPOSED UPSTREAM CONTRIBUTION:
    Could be added to TALES as a winnability analysis tool.
    """
    
    def __init__(self, max_history: int = 1000):
        self.state_history: Dict[str, GameState] = {}
        self.best_score_from_state: Dict[str, int] = {}
        self.visit_counts: Dict[str, int] = {}
        self.max_history = max_history
        self.max_score_seen = 0
    
    def record_state(self, state: GameState) -> float:
        """
        Record a state and return estimated winnability.
        
        Returns:
            Float in [0, 1] where 0 = likely unwinnable, 1 = likely winnable
        """
        state_hash = state.hash()
        
        self.visit_counts[state_hash] = self.visit_counts.get(state_hash, 0) + 1
        visits = self.visit_counts[state_hash]
        
        if state.score > self.max_score_seen:
            self.max_score_seen = state.score
        
        self.state_history[state_hash] = state
        if state_hash not in self.best_score_from_state:
            self.best_score_from_state[state_hash] = state.score
        else:
            self.best_score_from_state[state_hash] = max(
                self.best_score_from_state[state_hash],
                state.score
            )
        
        loop_penalty = min(visits / 10.0, 0.5)
        
        if self.max_score_seen > 0:
            progress_score = state.score / self.max_score_seen
        else:
            progress_score = 0.5
        
        best_possible = self.best_score_from_state.get(state_hash, state.score)
        if best_possible > state.score:
            suboptimality_penalty = 0.2
        else:
            suboptimality_penalty = 0.0
        
        winnability = max(0.0, min(1.0, 
            progress_score - loop_penalty - suboptimality_penalty
        ))
        
        return winnability
    
    def is_likely_unwinnable(self, state: GameState, threshold: float = 0.2) -> bool:
        """Check if state is likely unwinnable."""
        winnability = self.record_state(state)
        return winnability < threshold


class TextGameRiemannianMetric(RiemannianMetricBase):
    """
    Riemannian metric for text adventure games.
    
    Black holes form around:
    1. States with low winnability (stuck/looping)
    2. States following irreversible negative actions
    3. States with repeated visits without progress
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        irreversibility_detector: Optional[IrreversibilityDetector] = None,
        winnability_tracker: Optional[WinnabilityTracker] = None,
        base_metric: float = 1.0,
        unwinnability_severity: float = 10.0,
        irreversibility_severity: float = 5.0,
        learnable: bool = True,
    ):
        super().__init__(state_dim=embed_dim)
        
        self.irreversibility_detector = irreversibility_detector or IrreversibilityDetector()
        self.winnability_tracker = winnability_tracker or WinnabilityTracker()
        
        if learnable:
            self.base_metric = nn.Parameter(torch.tensor(base_metric))
            self.unwinnability_severity = nn.Parameter(torch.tensor(unwinnability_severity))
            self.irreversibility_severity = nn.Parameter(torch.tensor(irreversibility_severity))
        else:
            self.register_buffer('base_metric', torch.tensor(base_metric))
            self.register_buffer('unwinnability_severity', torch.tensor(unwinnability_severity))
            self.register_buffer('irreversibility_severity', torch.tensor(irreversibility_severity))
        
        self._last_irreversibility_score = 0.0
        self._last_winnability_score = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute metric from embedded state.
        
        For text input, use compute_from_game_state() instead.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        metric = torch.ones(batch_size, 1, device=x.device) * self.base_metric
        
        if self._last_winnability_score < 0.5:
            margin = max(self._last_winnability_score, 0.01)
            metric = metric + self.unwinnability_severity / margin
        
        if self._last_irreversibility_score > 0.3:
            metric = metric + self.irreversibility_severity * self._last_irreversibility_score
        
        return metric
    
    def compute_from_game_state(
        self,
        state: GameState,
        action: str,
        observation: str,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute metric directly from game state.
        
        Returns:
            Tuple of (total_metric, component_scores)
        """
        irrev_score, irrev_action = self.irreversibility_detector.analyze_action(
            action, observation
        )
        self._last_irreversibility_score = irrev_score
        
        winnability = self.winnability_tracker.record_state(state)
        self._last_winnability_score = winnability
        
        total_metric = float(self.base_metric.detach())
        
        if winnability < 0.5:
            margin = max(winnability, 0.01)
            total_metric += float(self.unwinnability_severity) / margin
        
        if irrev_score > 0.3:
            total_metric += float(self.irreversibility_severity) * irrev_score
        
        scores = {
            'winnability': winnability,
            'irreversibility': irrev_score,
            'base_metric': float(self.base_metric),
            'total_metric': total_metric,
        }
        if irrev_action:
            scores['irreversible_action_type'] = irrev_action.irreversibility_type.value
        
        return total_metric, scores
    
    def get_black_hole_centers(self) -> List[np.ndarray]:
        """Return placeholder (text games don't have spatial centers)."""
        return [np.zeros(2)]
    
    def get_event_horizons(self) -> List[float]:
        return [0.2]


class TextWorldSGPOWrapper(SGPOWrapperBase):
    """
    SGPO wrapper for TextWorld/TALES environments.
    
    Tracks irreversible actions and winnability to guide
    Sheaf-Geodesic Policy Optimization.
    """
    
    def __init__(
        self,
        env,
        metric: Optional[TextGameRiemannianMetric] = None,
        track_irreversibility: bool = True,
        track_winnability: bool = True,
    ):
        self.track_irreversibility = track_irreversibility
        self.track_winnability = track_winnability
        
        self._current_state: Optional[GameState] = None
        self._last_action: str = ""
        self._last_observation: str = ""
        self._step_count: int = 0
        self._inventory: Set[str] = set()
        self._location: str = "unknown"
        self._score: int = 0
        
        super().__init__(env, metric)
    
    def _create_default_metric(self) -> TextGameRiemannianMetric:
        return TextGameRiemannianMetric(learnable=True)
    
    def _extract_hazards(self) -> Tuple[List[np.ndarray], List[float]]:
        return [np.zeros(2)], [0.2]
    
    def _parse_observation(self, obs: Any) -> Tuple[str, Set[str], str, int]:
        """Parse observation to extract game state components."""
        if isinstance(obs, str):
            observation = obs
        elif isinstance(obs, dict):
            observation = obs.get('observation', obs.get('text', str(obs)))
        else:
            observation = str(obs)
        
        inventory = self._inventory
        location = self._location
        score = self._score
        
        obs_lower = observation.lower()
        
        if 'carrying' in obs_lower or 'inventory' in obs_lower:
            pass
        
        location_keywords = ['you are in', 'you are at', 'you find yourself']
        for kw in location_keywords:
            if kw in obs_lower:
                idx = obs_lower.find(kw)
                end_idx = obs_lower.find('.', idx)
                if end_idx > idx:
                    location = observation[idx:end_idx][:50]
                break
        
        return observation, inventory, location, score
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        obs, info = self.env.reset(**kwargs)
        
        self._step_count = 0
        self._last_action = ""
        self._inventory = set()
        self._location = "start"
        self._score = info.get('score', 0)
        
        observation, inventory, location, score = self._parse_observation(obs)
        self._last_observation = observation
        
        self._current_state = GameState(
            observation=observation,
            inventory=inventory,
            location=location,
            score=score,
            step=0,
        )
        
        info['metric_value'] = 1.0
        info['winnability'] = 1.0
        info['irreversibility'] = 0.0
        
        return obs, info
    
    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]:
        if isinstance(action, str):
            action_text = action
        else:
            action_text = str(action)
        
        self._last_action = action_text
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        
        self._score = info.get('score', self._score + int(reward > 0))
        
        observation, inventory, location, score = self._parse_observation(obs)
        self._last_observation = observation
        self._inventory = inventory
        self._location = location
        
        self._current_state = GameState(
            observation=observation,
            inventory=inventory,
            location=location,
            score=self._score,
            step=self._step_count,
        )
        
        metric_value, scores = self.metric.compute_from_game_state(
            self._current_state,
            action_text,
            observation,
        )
        
        info['metric_value'] = metric_value
        info['winnability'] = scores['winnability']
        info['irreversibility'] = scores['irreversibility']
        info['in_black_hole'] = metric_value > 10.0
        
        if 'irreversible_action_type' in scores:
            info['irreversible_action_type'] = scores['irreversible_action_type']
        
        cost = 0.0
        if scores['winnability'] < 0.3:
            cost += 0.5
        if scores['irreversibility'] > 0.5:
            cost += scores['irreversibility']
        info['cost'] = cost
        
        return obs, reward, terminated, truncated, info
    
    def _compute_metric(self, obs: Any) -> float:
        if self._current_state:
            metric, _ = self.metric.compute_from_game_state(
                self._current_state,
                self._last_action,
                self._last_observation,
            )
            return metric
        return 1.0


def create_textworld_gpo_env(
    game_file_or_name: str,
    track_irreversibility: bool = True,
    track_winnability: bool = True,
    **env_kwargs
) -> TextWorldSGPOWrapper:
    """
    Create a TextWorld environment wrapped for SGPO training.
    
    Args:
        game_file_or_name: Path to .z8/.ulx game file or TextWorld game name
        track_irreversibility: Detect irreversible actions
        track_winnability: Track estimated winnability
        **env_kwargs: Additional environment arguments
        
    Returns:
        TextWorldSGPOWrapper ready for SGPO training
    """
    try:
        import textworld
        import textworld.gym
        
        if game_file_or_name.endswith(('.z8', '.ulx', '.z5')):
            env_id = textworld.gym.register_game(
                game_file_or_name,
                request_infos=textworld.EnvInfos(
                    inventory=True,
                    location=True,
                    score=True,
                    moves=True,
                ),
                **env_kwargs
            )
        else:
            env_id = game_file_or_name
        
        import gymnasium as gym
        env = gym.make(env_id)
        
    except ImportError:
        raise ImportError(
            "textworld not installed. Run: pip install textworld\n"
            "For TALES suite: clone https://github.com/microsoft/tale-suite"
        )
    
    return TextWorldSGPOWrapper(
        env,
        track_irreversibility=track_irreversibility,
        track_winnability=track_winnability,
    )


# =============================================================================
# PROPOSED UPSTREAM CONTRIBUTION
# =============================================================================

UPSTREAM_PR_PROPOSAL = """
# Proposed Pull Request: Safety Analysis Tools for TALES/TextWorld

## Summary
Add optional safety analysis tools to TALES/TextWorld that help identify:
1. Irreversible actions that could lead to unwinnable states
2. Game state winnability estimates
3. Action sequences approaching "point of no return"

## Motivation
Text adventure games are uniquely challenging for RL because:
- Single wrong action can make game unwinnable
- Irreversibility is often not obvious until too late
- Long-horizon dependencies require careful planning

These tools help:
- Researchers study safe exploration in text games
- Developers debug game design (find unfair traps)
- RL algorithms avoid catastrophic mistakes

## Proposed API

```python
from textworld.safety import IrreversibilityDetector, WinnabilityTracker

# Wrap any TextWorld environment
from textworld.safety import SafetyWrapper

env = textworld.start("my_game.ulx")
safe_env = SafetyWrapper(
    env,
    track_irreversibility=True,
    track_winnability=True,
)

obs, infos = safe_env.step("eat the key")
print(infos['irreversibility_warning'])  # "Warning: 'eat' is irreversible"
print(infos['winnability_estimate'])     # 0.3 (likely stuck)
```

## Implementation
- `textworld/safety/irreversibility.py`: Action analysis
- `textworld/safety/winnability.py`: State tracking
- `textworld/safety/wrapper.py`: Environment wrapper
- `textworld/safety/visualize.py`: State graph visualization

## Use Cases
1. **Safe RL Research**: Cost signal for constrained optimization
2. **Game Design**: Find unfair/frustrating game mechanics
3. **Hint Systems**: Warn players about risky actions
4. **Curriculum Learning**: Start with low-irreversibility games

## Testing
- Unit tests with synthetic games
- Integration tests with Zork, Enchanter
- Benchmark on TALES suite games
"""
