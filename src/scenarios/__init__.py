"""
Scenario Generators for Semantic MDP Training

This module provides diverse training scenarios:
1. Alignment scenarios (helpful vs harmful)
2. Strategic games (chess, go) with verbal constraints
3. Coding/problem-solving tasks

Each scenario generates (state, action, next_state, reward, cost) trajectories
suitable for training the Hodge Critic and SGPO.
"""

from .alignment import AlignmentScenarioGenerator
from .strategic_games import ChessScenarioGenerator, GoScenarioGenerator
from .coding import CodingScenarioGenerator

__all__ = [
    "AlignmentScenarioGenerator",
    "ChessScenarioGenerator",
    "GoScenarioGenerator",
    "CodingScenarioGenerator",
]
