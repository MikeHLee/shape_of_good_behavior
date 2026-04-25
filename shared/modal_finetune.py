"""Modal GPU runner for the SFT → Hodge-RM → PPO fine-tuning pipeline.

Target model : Qwen/Qwen2.5-1.5B-Instruct (fits on L4 in bf16 with LoRA)
Pipeline:
  Stage 1 — SFT on ideal responses from counterfactuals cache
  Stage 2 — Reward model training (standard or Hodge-weighted)
  Stage 3 — PPO policy optimization against the RM
  Stage 4 — Evaluation: base / SFT / PPO / Hodge-PPO exploit resistance

Usage:
    # Full pipeline (standard RM)
    modal run shared/modal_finetune.py

    # Full pipeline with Hodge-weighted RM (Hodge-PPO)
    modal run shared/modal_finetune.py --hodge

    # Single stage
    modal run shared/modal_finetune.py --stage sft
    modal run shared/modal_finetune.py --stage rm --hodge
    modal run shared/modal_finetune.py --stage ppo --hodge
    modal run shared/modal_finetune.py --stage eval

    # Fire-and-forget
    modal run shared/modal_finetune.py --hodge --detach
"""

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image — reads requirements-finetune.txt at build time
# ---------------------------------------------------------------------------

_req_path = Path(__file__).parent / "requirements-finetune.txt"
try:
    _requirements = [
        line.strip()
        for line in _req_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
except FileNotFoundError:
    _requirements = []

_project_root = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_requirements)
    # Embed shared pipeline code + track-1 Hodge utilities
    .add_local_dir(str(_project_root / "shared"),                    remote_path="/app/shared")
    .add_local_dir(str(_project_root / "feedback_geometry" / "src"), remote_path="/app/feedback_geometry/src")
    .add_local_dir(str(_project_root / "src"),                       remote_path="/app/src")
)

app = modal.App("reward-hacking-finetune")

# Two volumes: one for model checkpoints, one shared with the main pipeline
ckpt_vol    = modal.Volume.from_name("reward-hacking-checkpoints", create_if_missing=True)
results_vol = modal.Volume.from_name("reward-hacking-results",     create_if_missing=True)

_SECRETS = [modal.Secret.from_name("huggingface-token")]

# Checkpoint layout (inside ckpt_vol at /checkpoints/)
# ├── hf_hub_cache/       HuggingFace model cache (shared across functions)
# ├── sft/                SFT policy adapter
# ├── rm/                 Standard reward model
# ├── rm_hodge/           Hodge-weighted reward model
# ├── ppo/                PPO-trained policy
# └── ppo_hodge/          Hodge-PPO-trained policy


def _setup_paths():
    """Configure module-level path overrides for the Modal container."""
    import sys
    from pathlib import Path as P

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/feedback_geometry/src")
    sys.path.insert(0, "/app/src")

    import shared.src.config as cfg_mod
    cfg_mod.PROJECT_ROOT            = P("/app")
    cfg_mod.SHARED_ROOT             = P("/app/shared")
    cfg_mod.FEEDBACK_GEOMETRY_SRC   = P("/app/feedback_geometry/src")
    cfg_mod.CONSTRAINT_GEOMETRY_SRC = P("/app/src")


def _hf_cache_dir() -> str:
    """Point HuggingFace cache into the persistent checkpoint volume."""
    import os
    cache = "/checkpoints/hf_hub_cache"
    os.environ["HF_HOME"]             = cache
    os.environ["TRANSFORMERS_CACHE"]  = cache
    os.environ["HF_DATASETS_CACHE"]   = "/results/hf_datasets_cache"
    return cache


def _load_pairs(config):
    """Load TRACE hacked records and match to cached counterfactuals."""
    from shared.src.data_ingest import ingest_all
    from shared.src.counterfactual_gen import CounterfactualGenerator, _cache_key

    result = ingest_all(config, sources=["trace"])
    hacked = result.filter_hacked()

    gen = CounterfactualGenerator(config)
    pairs = []
    for record in hacked:
        key = _cache_key(record)
        if key not in gen._cache:
            continue
        cached = gen._cache[key]
        from shared.src.counterfactual_gen import CounterfactualPair
        pairs.append(CounterfactualPair(
            exploit_text    = record.exploit_text,
            ideal_text      = cached.get("ideal_response", ""),
            context_text    = record.context_text,
            failure_analysis= cached.get("failure_analysis", ""),
            exploit_type    = cached.get("exploit_type", "other"),
            principles_violated = cached.get("principles_violated", []),
            confidence      = cached.get("confidence", 1.0),
            exploit_category= record.exploit_category,
            source          = "trace_llm",
        ))

    return pairs, hacked


# ---------------------------------------------------------------------------
# Stage 1 — SFT
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    memory=32768,
    volumes={"/checkpoints": ckpt_vol, "/results": results_vol},
    secrets=_SECRETS,
)
def train_sft() -> str:
    """Fine-tune Qwen2.5-1.5B-Instruct on ideal responses (SFT warm-start).

    Returns:
        Path to the SFT checkpoint inside /checkpoints/.
    """
    _setup_paths()
    _hf_cache_dir()
    ckpt_vol.reload()

    from shared.src.config import PipelineConfig
    from shared.src.lm_finetuning import FineTuneConfig, run_sft

    config    = PipelineConfig()
    config.trace_max_samples    = 517
    config.hh_rlhf_max_samples  = 0
    config.cache_dir            = "/app/shared/data/cache"

    ft_config = FineTuneConfig()
    ft_config.checkpoint_dir = "/checkpoints"

    pairs, _ = _load_pairs(config)
    print(f"SFT: {len(pairs)} (context, ideal) pairs loaded")

    output_dir = "/checkpoints/sft"
    run_sft(pairs, ft_config, output_dir)

    ckpt_vol.commit()
    print(f"SFT complete → {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Stage 2 — Reward model
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    memory=32768,
    volumes={"/checkpoints": ckpt_vol, "/results": results_vol},
    secrets=_SECRETS,
)
def train_reward_model(hodge: bool = False) -> str:
    """Train a scalar reward model on (exploit, ideal) preference pairs.

    Args:
        hodge: If True, apply Hodge cycle weights to the Bradley-Terry loss.

    Returns:
        Path to the RM checkpoint inside /checkpoints/.
    """
    _setup_paths()
    _hf_cache_dir()
    ckpt_vol.reload()

    from shared.src.config import PipelineConfig
    from shared.src.lm_finetuning import FineTuneConfig, run_reward_model_training

    config   = PipelineConfig()
    config.trace_max_samples   = 517
    config.hh_rlhf_max_samples = 0
    config.cache_dir           = "/app/shared/data/cache"

    ft_config = FineTuneConfig()
    ft_config.checkpoint_dir = "/checkpoints"

    pairs, _ = _load_pairs(config)
    print(f"RM training: {len(pairs)} pairs  hodge={hodge}")

    output_dir = "/checkpoints/rm_hodge" if hodge else "/checkpoints/rm"
    run_reward_model_training(
        pairs, ft_config, output_dir,
        pipeline_config=config if hodge else None,
        use_hodge=hodge,
    )

    ckpt_vol.commit()
    print(f"RM training complete → {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Stage 3 — PPO
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="L4",
    timeout=14400,
    memory=32768,
    volumes={"/checkpoints": ckpt_vol, "/results": results_vol},
    secrets=_SECRETS,
)
def train_ppo(hodge: bool = False) -> dict:
    """PPO policy optimization against the reward model.

    Loads the SFT checkpoint as the initial policy and frozen reference.
    Uses the standard RM (hodge=False) or Hodge-weighted RM (hodge=True).

    Args:
        hodge: If True, uses the Hodge-RM for reward scoring (Hodge-PPO).

    Returns:
        Dict with training stats saved to /results/finetune/.
    """
    _setup_paths()
    _hf_cache_dir()
    ckpt_vol.reload()
    results_vol.reload()

    import json
    from pathlib import Path as P

    from shared.src.config import PipelineConfig
    from shared.src.lm_finetuning import FineTuneConfig, run_ppo

    config   = PipelineConfig()
    config.trace_max_samples   = 517
    config.hh_rlhf_max_samples = 0
    config.cache_dir           = "/app/shared/data/cache"

    ft_config = FineTuneConfig()
    ft_config.checkpoint_dir = "/checkpoints"

    _, hacked_records = _load_pairs(config)
    print(f"PPO: {len(hacked_records)} exploit prompts as queries  hodge={hodge}")

    sft_ckpt = "/checkpoints/sft"
    rm_ckpt  = "/checkpoints/rm_hodge" if hodge else "/checkpoints/rm"
    out_dir  = "/checkpoints/ppo_hodge" if hodge else "/checkpoints/ppo"

    stats = run_ppo(
        records       = hacked_records,
        sft_checkpoint= sft_ckpt,
        rm_checkpoint = rm_ckpt,
        config        = ft_config,
        output_dir    = out_dir,
    )
    stats["hodge"] = hodge

    # Persist stats
    P("/results/finetune").mkdir(parents=True, exist_ok=True)
    tag = "hodge_ppo" if hodge else "ppo"
    stats_path = f"/results/finetune/{tag}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    ckpt_vol.commit()
    results_vol.commit()
    print(f"PPO complete → {out_dir}  mean_reward={stats['mean_reward_final']:.4f}")
    return stats


# ---------------------------------------------------------------------------
# Stage 4 — Evaluation
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    memory=32768,
    volumes={"/checkpoints": ckpt_vol, "/results": results_vol},
    secrets=_SECRETS,
)
def evaluate(n_eval: int = 50) -> dict:
    """Compare base / SFT / PPO / Hodge-PPO on held-out TRACE exploit prompts.

    Generates responses from each available checkpoint, scores with the
    Hodge-RM, and reports mean_reward and exploit_resistance (fraction > 0).

    Returns:
        Comparison table saved to /results/finetune/eval_comparison.json.
    """
    _setup_paths()
    _hf_cache_dir()
    ckpt_vol.reload()
    results_vol.reload()

    import json
    from pathlib import Path as P

    from shared.src.config import PipelineConfig
    from shared.src.lm_finetuning import FineTuneConfig, evaluate_exploit_resistance

    config = PipelineConfig()
    config.trace_max_samples   = 517
    config.hh_rlhf_max_samples = 0
    config.cache_dir           = "/app/shared/data/cache"

    ft_config = FineTuneConfig()
    ft_config.checkpoint_dir = "/checkpoints"

    _, hacked_records = _load_pairs(config)

    # Use Hodge-RM for scoring if available, else standard RM
    rm_ckpt = (
        "/checkpoints/rm_hodge"
        if P("/checkpoints/rm_hodge").exists()
        else "/checkpoints/rm"
    )

    # Collect available checkpoints (None = base model weights, no fine-tuning)
    checkpoints = {"base": None}
    for name, path in [
        ("sft",       "/checkpoints/sft"),
        ("ppo",       "/checkpoints/ppo"),
        ("hodge_ppo", "/checkpoints/ppo_hodge"),
    ]:
        if P(path).exists():
            checkpoints[name] = path

    print(f"Evaluating: {list(checkpoints.keys())}  n_eval={n_eval}")

    results = evaluate_exploit_resistance(
        records     = hacked_records,
        checkpoints = checkpoints,
        rm_checkpoint= rm_ckpt,
        config      = ft_config,
        n_eval      = n_eval,
    )

    P("/results/finetune").mkdir(parents=True, exist_ok=True)
    out_path = "/results/finetune/eval_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    results_vol.commit()

    # Pretty-print table
    print("\n=== Exploit Resistance Comparison ===")
    print(f"{'Model':<14}  {'Mean Reward':>12}  {'Exploit Resistance':>18}  {'N':>5}")
    print("-" * 56)
    for name, r in sorted(results.items()):
        print(
            f"{name:<14}  {r['mean_reward']:>12.4f}  "
            f"{r['exploit_resistance']:>17.2%}  {r['n']:>5}"
        )

    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    stage:  str  = "all",   # sft | rm | ppo | eval | all
    hodge:  bool = False,
    detach: bool = False,
    n_eval: int  = 50,
):
    """Orchestrate the fine-tuning pipeline on Modal GPU.

    Examples
    --------
    # Full pipeline, standard PPO
    modal run shared/modal_finetune.py

    # Full pipeline, Hodge-PPO
    modal run shared/modal_finetune.py --hodge

    # Single stages
    modal run shared/modal_finetune.py --stage sft
    modal run shared/modal_finetune.py --stage rm --hodge
    modal run shared/modal_finetune.py --stage ppo --hodge
    modal run shared/modal_finetune.py --stage eval --n-eval 100

    # Fire-and-forget (survives local disconnect)
    modal run shared/modal_finetune.py --hodge --detach
    """

    def _run(fn, *args, **kwargs):
        return fn.remote(*args, **kwargs)

    if stage in ("sft", "all"):
        print("--- Stage 1: SFT ---")
        result = _run(train_sft)
        print(f"  → {result}")

    if stage in ("rm", "all"):
        print(f"--- Stage 2: Reward Model (hodge={hodge}) ---")
        result = _run(train_reward_model, hodge=hodge)
        print(f"  → {result}")

    if stage in ("ppo", "all"):
        print(f"--- Stage 3: PPO (hodge={hodge}) ---")
        result = _run(train_ppo, hodge=hodge)
        print(f"  → {result}")

    if stage in ("eval", "all"):
        print("--- Stage 4: Evaluation ---")
        result = _run(evaluate, n_eval=n_eval)
        print(f"  → {result}")
