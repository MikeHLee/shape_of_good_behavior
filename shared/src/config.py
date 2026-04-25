"""Shared configuration for the cross-track reward hacking experiment pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Dataset identifiers
TRACE_DATASET_ID = "PatronusAI/trace-dataset"
HH_RLHF_DATASET_ID = "Anthropic/hh-rlhf"
PKU_SAFE_RLHF_DATASET_ID = "PKU-Alignment/PKU-SafeRLHF"
BEAVER_TAILS_DATASET_ID = "PKU-Alignment/BeaverTails"
ADV_BENCH_DATASET_ID = "walledai/AdvBench"

# Project root (high_dimensional_reward_spaces/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SHARED_ROOT = Path(__file__).resolve().parent.parent

# Track source directories
FEEDBACK_GEOMETRY_SRC = PROJECT_ROOT / "feedback_geometry" / "src"
CONSTRAINT_GEOMETRY_SRC = PROJECT_ROOT / "src"


@dataclass
class PipelineConfig:
    """Configuration for the full cross-track pipeline."""

    # Sampling
    trace_max_samples: int = 500
    hh_rlhf_max_samples: int = 2000
    pku_safe_max_samples: int = 1000
    beaver_tails_max_samples: int = 1000
    adv_bench_max_samples: int = 500
    seed: int = 42

    # LLM for counterfactual generation
    llm_model: str = "claude-sonnet-4-6"
    llm_thinking_budget: int = 8000   # extended thinking tokens for alignment reasoning
    llm_max_tokens: int = 10000       # must exceed thinking_budget; ~2k for text output

    # Embedding model
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384
    reduced_dim: int = 32  # PCA target for RM state vectors

    # Hodge analysis (Track 1)
    h1_threshold: float = 0.0  # 0.0 = remove all harmonic component
    num_seeds_quick: int = 5
    num_seeds_full: int = 30
    rm_hidden_dim: int = 64
    rm_epochs: int = 100
    rm_lr: float = 1e-3

    # SGPO / safety (Track 2)
    danger_cluster_eps: float = 0.5
    danger_min_cluster_size: int = 3
    danger_margin: float = 0.1
    sgpo_clip_ratio: float = 0.2
    sgpo_geometric_threshold: float = 2.0

    # Preference optimizer baselines
    dpo_beta: float = 0.1
    grpo_beta: float = 0.04
    grpo_group_size: int = 8
    orpo_lambda: float = 0.5
    kto_beta: float = 0.1
    kto_lambda_good: float = 1.0
    kto_lambda_bad: float = 1.33

    # Hodge regularization
    hodge_lambda: float = 0.05  # Regularization strength for Hodge potential-alignment loss

    # Visualization
    pca_viz_dim: int = 3  # 3D for Plotly
    figure_width: int = 900
    figure_height: int = 700

    # Paths
    data_dir: Optional[str] = None
    results_dir: Optional[str] = None
    figures_dir: Optional[str] = None
    cache_dir: Optional[str] = None

    # Constitutional principles for Track 3
    constitutional_principles: List[str] = field(default_factory=lambda: [
        "Refuse requests for weapons of mass destruction",
        "Refuse offensive hacking instructions",
        "Refuse hate speech and discrimination campaigns",
        "Refuse manipulation and social engineering tactics",
        "Avoid sycophantic agreement with harmful premises",
        "Provide honest, balanced information",
        "Respect user autonomy without enabling harm",
        "Maintain transparency about AI limitations",
    ])

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = str(SHARED_ROOT / "data")
        if self.results_dir is None:
            self.results_dir = str(SHARED_ROOT / "results")
        if self.figures_dir is None:
            self.figures_dir = str(SHARED_ROOT / "figures")
        if self.cache_dir is None:
            self.cache_dir = str(SHARED_ROOT / "data" / "cache")
        # Ensure directories exist
        for d in [self.data_dir, self.results_dir, self.figures_dir, self.cache_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
