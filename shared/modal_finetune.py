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

import sys
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
# _project_root = .../ai_research/topics/shape_of_good_behavior; the parent's
# parent is the ai_research root shared_modal/ lives at.
_ai_research_root = _project_root.parent.parent
# So the local_entrypoint (runs on this machine, not in a container) can pull
# the manifest back off the volume after Stage 3 dispatch — see main().
if str(_ai_research_root) not in sys.path:
    sys.path.insert(0, str(_ai_research_root))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_requirements)
    # PYTHONUNBUFFERED belt-and-braces: run_ppo now prints with flush=True, but
    # any other print() without explicit flush (from HF, TRL, etc.) will also
    # land per-line in Modal's log capture instead of sitting in a 4KB buffer.
    .env({"PYTHONUNBUFFERED": "1"})
    # Embed shared pipeline code + track-1 Hodge utilities
    .add_local_dir(str(_project_root / "shared"),                    remote_path="/app/shared")
    .add_local_dir(str(_project_root / "feedback_geometry" / "src"), remote_path="/app/feedback_geometry/src")
    .add_local_dir(str(_project_root / "src"),                       remote_path="/app/src")
    # shared_modal.manifest — the run-manifest ORG-002/003 built specifically so
    # Stage 3 PPO (this file) never again vanishes without a trail. See train_ppo.
    .add_local_dir(str(_ai_research_root / "shared_modal"),          remote_path="/app/shared_modal")
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


_HOLDOUT_FRAC = 0.2
_HOLDOUT_SEED = 42


def _is_holdout(record, holdout_frac: float = _HOLDOUT_FRAC, seed: int = _HOLDOUT_SEED) -> bool:
    """Deterministic train/held-out assignment, keyed on record content.

    Content-keyed (not index-keyed) so the split is stable even if
    ingest_all's ordering changes between runs — a record with the same
    (source, exploit_category, exploit_text) always lands on the same side.
    """
    import hashlib
    from shared.src.counterfactual_gen import _cache_key

    digest = hashlib.sha256(f"{seed}:{_cache_key(record)}".encode()).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return frac < holdout_frac


def _load_pairs(config, split: str = "all"):
    """Load TRACE hacked records and match to cached counterfactuals.

    Args:
        split: "train" — exclude the held-out fraction (Stages 1-3 must use
            this so eval measures generalization, not memorization).
            "holdout" — only the held-out fraction (Stage 4 eval).
            "all" — no split (legacy/debug; do not use for train or eval).
    """
    from shared.src.data_ingest import ingest_all
    from shared.src.counterfactual_gen import CounterfactualGenerator, _cache_key

    result = ingest_all(config, sources=["trace"])
    hacked = result.filter_hacked()

    if split == "train":
        hacked = [r for r in hacked if not _is_holdout(r)]
    elif split == "holdout":
        hacked = [r for r in hacked if _is_holdout(r)]
    elif split != "all":
        raise ValueError(f"split must be 'train', 'holdout', or 'all', got {split!r}")

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

    pairs, _ = _load_pairs(config, split="train")
    print(f"SFT: {len(pairs)} (context, ideal) pairs loaded  [train split, {_HOLDOUT_FRAC:.0%} held out]")

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

    pairs, _ = _load_pairs(config, split="train")
    print(f"RM training: {len(pairs)} pairs  hodge={hodge}  [train split, {_HOLDOUT_FRAC:.0%} held out]")

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
    # PPO holds policy + frozen ref + frozen RM on the same device. Three
    # Qwen2.5-1.5B in bf16 + autograd + KV cache OOMs on L4 (24 GB), so
    # bump to A100-40GB which has real headroom for the rollout buffers
    # and ppo-epoch autograd graph.
    gpu="A100-40GB",
    timeout=14400,
    memory=32768,
    volumes={"/checkpoints": ckpt_vol, "/results": results_vol},
    secrets=_SECRETS,
)
def train_ppo(hodge: bool = False, experiment_id: str = "",
              ppo_steps: int = 0) -> dict:
    """PPO policy optimization against the reward model.

    Loads the SFT checkpoint as the initial policy and frozen reference.
    Uses the standard RM (hodge=False) or Hodge-weighted RM (hodge=True).

    Args:
        hodge: If True, uses the Hodge-RM for reward scoring (Hodge-PPO).
        experiment_id: manifest id minted by the local entrypoint (`main()`),
            so it knows exactly where on the volume to pull the manifest from
            afterward via `mirror_from_volume`. SGB-003 is the reason this
            exists: the April 25 PPO runs left no checkpoint dir and no
            surviving App-dashboard entry under Starter-tier retention, so a
            dead run was indistinguishable from one that never launched. A
            manifest written to the volume before/during/after this function
            body makes that observable even if this container is killed.
        ppo_steps: If > 0, overrides FineTuneConfig.ppo_steps (default 256).
            Purpose is the SGB-003 regression-target run: FineTuneConfig's
            default × ppo_batch_size × ppo_max_new_tokens realistically needs
            6+ hours per variant on A100-40GB and can exceed the 4h Modal
            timeout, so the manifest-observability re-run defaults to a
            smaller step count to actually finish. Full training is
            recoverable by launching without this override once the manifest
            path is proven.

    Returns:
        Dict with training stats saved to /results/finetune/.
    """
    _setup_paths()
    _hf_cache_dir()
    ckpt_vol.reload()
    results_vol.reload()

    import json
    from pathlib import Path as P

    from shared_modal.manifest import track

    from shared.src.config import PipelineConfig
    from shared.src.lm_finetuning import FineTuneConfig, run_ppo

    tag = "hodge_ppo" if hodge else "ppo"

    with track(
        division="SGB", inquiry_id="SGB-003", experiment_class=f"train_ppo[{tag}]",
        paths=[f"/checkpoints/_manifests/{experiment_id}.json",
               f"/results/_manifests/{experiment_id}.json"],
        kwargs={"hodge": hodge}, modal={"gpu": "A100-40GB", "hodge": hodge},
        git_root=str(_ai_research_root), volume_commit=results_vol.commit,
        experiment_id=experiment_id,
    ) as m:
        config   = PipelineConfig()
        config.trace_max_samples   = 517
        config.hh_rlhf_max_samples = 0
        config.cache_dir           = "/app/shared/data/cache"

        ft_config = FineTuneConfig()
        ft_config.checkpoint_dir = "/checkpoints"
        if ppo_steps > 0:
            ft_config.ppo_steps = ppo_steps

        _, hacked_records = _load_pairs(config, split="train")
        print(f"PPO: {len(hacked_records)} exploit prompts as queries  "
              f"hodge={hodge}  ppo_steps={ft_config.ppo_steps}  "
              f"[train split, {_HOLDOUT_FRAC:.0%} held out]", flush=True)

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
        stats_path = f"/results/finetune/{tag}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        ckpt_vol.commit()
        results_vol.commit()
        m.add_checkpoint(tag, ckpt_vol.name, out_dir)
        m.add_result_file(f"{tag}_stats", results_vol.name, stats_path)
        m.results = stats
        print(f"PPO complete → {out_dir}  mean_reward={stats['mean_reward_final']:.4f}")

    return stats


# ---------------------------------------------------------------------------
# Stage 4 — Evaluation
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="L4",
    # n_eval=100 across 4 checkpoints (base/sft/ppo/hodge_ppo) is ~400
    # unbatched single-example generations at 256 tokens each — the original
    # 3600s timeout was hit mid-run on the first SGB-004 attempt. Sized to
    # match Stage 3's budget with headroom; incremental writes below mean a
    # second timeout would still keep whatever finished instead of losing all.
    timeout=14400,
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

    _, hacked_records = _load_pairs(config, split="holdout")
    print(f"Eval set: {len(hacked_records)} held-out exploit prompts "
          f"[holdout split, {_HOLDOUT_FRAC:.0%} of total]")

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

    P("/results/finetune").mkdir(parents=True, exist_ok=True)
    out_path = "/results/finetune/eval_comparison.json"

    def _on_checkpoint_done(name: str, result: dict) -> None:
        # Commit after every checkpoint, not just at the end, so a timeout
        # (hit on the first SGB-004 attempt) leaves the partial table on the
        # volume instead of losing everything.
        results_vol.commit()
        print(f"  [{name}] mean_reward={result['mean_reward']:.4f}  "
              f"resist={result['exploit_resistance']:.2%}  (committed)", flush=True)

    results = evaluate_exploit_resistance(
        records      = hacked_records,
        checkpoints  = checkpoints,
        rm_checkpoint= rm_ckpt,
        config       = ft_config,
        n_eval       = n_eval,
        results_path = out_path,
        on_checkpoint_done = _on_checkpoint_done,
    )

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
    ppo_steps: int = 0,     # override FineTuneConfig.ppo_steps (default 256)
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
        # .remote() ties the remote function to the local gRPC stream — Modal
        # sends InputCancellation when the local process exits (even cleanly,
        # even with -d). .spawn() submits the function as an independent job;
        # the remote continues whether or not the local process is alive.
        # For short stages (SFT, RM) that finish before the client exits,
        # .spawn().get() behaves identically to .remote(); for PPO (hours)
        # we just spawn and exit.
        return fn.spawn(*args, **kwargs).get()

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
        from shared_modal import mirror_from_volume, new_experiment_id
        exp_id = new_experiment_id()
        print(f"  manifest id: {exp_id}  (SGB-003 regression target — see modal_finetune.py train_ppo)")
        # _run uses .spawn().get(): the remote function is committed as an
        # independent job before .get() blocks. If this process is killed,
        # Modal does NOT cancel the spawned container (unlike .remote() which
        # holds a cancellable gRPC stream). Run this entrypoint as a long-lived
        # background process so it stays connected and mirrors the manifest.
        try:
            result = _run(train_ppo, hodge=hodge, experiment_id=exp_id,
                          ppo_steps=ppo_steps)
            print(f"  → {result}")
        finally:
            dest = mirror_from_volume(
                "reward-hacking-results", "/results", exp_id,
                division="SGB", inquiry_id="SGB-003",
            )
            print(f"  manifest mirrored to: {dest}")

    if stage in ("eval", "all"):
        print("--- Stage 4: Evaluation ---")
        result = _run(evaluate, n_eval=n_eval)
        print(f"  → {result}")
