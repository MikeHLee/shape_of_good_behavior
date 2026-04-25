"""
SGPO Environment Adapters

This package provides Gymnasium-compatible wrappers that integrate
Sheaf-Geodesic Policy Optimization (SGPO) with various RL benchmarks.

Environments:
- Safety-Gymnasium: Direct black hole paradigm (hazards as singularities)
- LMRL-Gym: LLM multi-turn RL with partial observability
- Robust-Gymnasium: Adversarial perturbations with LLM attacks
- TALES/TextWorld: Text adventure with irreversible state black holes

Each adapter includes proposed upstream contributions that could be
submitted as pull requests to the respective open-source projects.
"""

from .base import (
    SGPOWrapperBase,
    RiemannianMetricBase,
)

from .safety_gymnasium_adapter import (
    SafetyGymnasiumSGPOWrapper,
    MultiHazardRiemannianMetric,
    create_safety_gpo_env,
    SGPOTrainer,
)

from .lmrl_gym_adapter import (
    LMRLGymSGPOWrapper,
    ConversationalRiemannianMetric,
    ConversationBlackHoleType,
    SafetyConstraint,
    ConversationBlackHoleDetectors,
    DEFAULT_SAFETY_CONSTRAINTS,
    create_lmrl_gpo_env,
)

from .robust_gymnasium_adapter import (
    RobustGymnasiumSGPOWrapper,
    AdversarialRiemannianMetric,
    AdversarialBlackHoleTracker,
    DisturbanceMode,
    DisturbanceConfig,
    RiemannianAdversary,
    create_robust_gpo_env,
)

from .textworld_adapter import (
    TextWorldSGPOWrapper,
    TextGameRiemannianMetric,
    IrreversibilityDetector,
    WinnabilityTracker,
    IrreversibleAction,
    IrreversibilityType,
    create_textworld_gpo_env,
)

from .storytelling_machine import (
    StorytellingMachine,
    StorytellingSGPOWrapper,
    WorldOracle,
    LLMOracle,
    SemanticEmbeddingMetric,
    MCPAction,
    ActionType,
    BeliefState,
    Page,
    Transition,
    create_storytelling_env,
)

__all__ = [
    # Base classes
    "SGPOWrapperBase",
    "RiemannianMetricBase",
    # Safety-Gymnasium (Priority 1)
    "SafetyGymnasiumSGPOWrapper",
    "MultiHazardRiemannianMetric",
    "create_safety_gpo_env",
    "SGPOTrainer",
    # LMRL-Gym (Priority 2)
    "LMRLGymSGPOWrapper",
    "ConversationalRiemannianMetric",
    "ConversationBlackHoleType",
    "SafetyConstraint",
    "ConversationBlackHoleDetectors",
    "DEFAULT_SAFETY_CONSTRAINTS",
    "create_lmrl_gpo_env",
    # Robust-Gymnasium (Priority 3)
    "RobustGymnasiumSGPOWrapper",
    "AdversarialRiemannianMetric",
    "AdversarialBlackHoleTracker",
    "DisturbanceMode",
    "DisturbanceConfig",
    "RiemannianAdversary",
    "create_robust_gpo_env",
    # TextWorld/TALES (Priority 4)
    "TextWorldSGPOWrapper",
    "TextGameRiemannianMetric",
    "IrreversibilityDetector",
    "WinnabilityTracker",
    "IrreversibleAction",
    "IrreversibilityType",
    "create_textworld_gpo_env",
    # Storytelling Machine (Semantic Turing Machine)
    "StorytellingMachine",
    "StorytellingSGPOWrapper",
    "WorldOracle",
    "LLMOracle",
    "SemanticEmbeddingMetric",
    "MCPAction",
    "ActionType",
    "BeliefState",
    "Page",
    "Transition",
    "create_storytelling_env",
]
