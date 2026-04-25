"""
SGPO Process Supervision Module

Implements Sheaf-Geodesic Policy Optimization with process-level human feedback
for detecting topological anomalies in reward spaces.
"""

from .environment import AnomalyNavigationEnv, AnomalyType
from .feedback import (
    StepFeedback,
    TrajectoryFeedback,
    AnomalyCandidate,
    ProcessSupervisor,
)
from .models import (
    Actor,
    Critic,
    StepValueNetwork,
    OutcomeNetwork,
    LearnedAggregator,
    AnomalyAwareRewardLearning,
    AnomalyAwareMetric,
)
from .trainer import SGPOTrainer, TrainingConfig

__all__ = [
    "AnomalyNavigationEnv",
    "AnomalyType",
    "StepFeedback",
    "TrajectoryFeedback",
    "AnomalyCandidate",
    "ProcessSupervisor",
    "Actor",
    "Critic",
    "StepValueNetwork",
    "OutcomeNetwork",
    "LearnedAggregator",
    "AnomalyAwareRewardLearning",
    "AnomalyAwareMetric",
    "SGPOTrainer",
    "TrainingConfig",
]
