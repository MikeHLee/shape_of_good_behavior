"""Cross-track shared pipeline for reward hacking experiments.

Serves all three tracks of the Shape of Good Behavior series:
- Track 1 (Feedback Geometry): Hodge decomposition, H1 detection
- Track 2 (Constraint Geometry): SGPO, conformal safety
- Track 3 (Constitutional Alignment Geometry): Geometric analysis
"""

from .src.config import PipelineConfig
from .src.data_ingest import ExploitRecord, IngestResult, ingest_all
from .src.counterfactual_gen import CounterfactualPair, CounterfactualGenerator
from .src.preference_mapper import (
    PreferenceEdge,
    EmbeddingPair,
    DangerRegionSpec,
    PreferenceMapper,
)
from .src.hodge_analysis import HodgeRewardHackingAnalyzer
from .src.reward_hacking_eval import ExploitResistanceReport, ExploitResistanceEvaluator

__all__ = [
    "PipelineConfig",
    "ExploitRecord",
    "IngestResult",
    "ingest_all",
    "CounterfactualPair",
    "CounterfactualGenerator",
    "PreferenceEdge",
    "EmbeddingPair",
    "DangerRegionSpec",
    "PreferenceMapper",
    "HodgeRewardHackingAnalyzer",
    "ExploitResistanceReport",
    "ExploitResistanceEvaluator",
]
