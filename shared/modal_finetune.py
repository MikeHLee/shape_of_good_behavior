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

    # Fire-and-forget -- NOTE: --detach is a flag to `modal run` ITSELF and
    # must come BEFORE the script path, not after it. `main()`'s own --detach
    # parameter below is unused dead code -- anything placed after the script
    # path is parsed as an argument to main()'s CLI, not to `modal run`, so
    # `modal run shared/modal_finetune.py --hodge --detach` silently does NOT
    # detach (found the hard way in SGB-006: two full PPO-7B runs died when
    # the local `modal run` process was killed by something in the
    # environment during a long unattended wait, because they weren't
    # actually detached despite passing what looked like the right flag).
    modal run --detach shared/modal_finetune.py --hodge

    # SGB-006: 7B scale-up (Qwen2.5-7B-Instruct), fully namespaced under
    # /checkpoints/7b/ -- cannot collide with the 1.5B checkpoints above.
    # PPO-7B stages are multi-hour; always launch with --detach BEFORE the
    # script path (see note above) so the Modal app survives a killed local
    # process instead of tearing down mid-run.
    modal run shared/modal_finetune.py --stage sft-7b
    modal run shared/modal_finetune.py --stage rm-7b --hodge
    modal run shared/modal_finetune.py --stage ppo-7b --hodge --ppo-steps 8   # smoke test
    modal run --detach shared/modal_finetune.py --stage ppo-7b --hodge       # full run
    modal run shared/modal_finetune.py --stage eval-7b --n-eval 100
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
    # expandable_segments reduces CUDA allocator fragmentation -- suggested by
    # the SGB-006 7B PPO OOM's own error message; pure allocator strategy, no
    # effect on results, safe for every stage/scale.
    .env({"PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
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
# ├── sft/                SFT policy adapter                    [1.5B, Qwen2.5-1.5B-Instruct]
# ├── rm/                 Standard reward model                 [1.5B]
# ├── rm_hodge/           Hodge-weighted reward model            [1.5B]
# ├── ppo/                PPO-trained policy                     [1.5B]
# ├── ppo_hodge/          Hodge-PPO-trained policy                [1.5B]
# └── 7b/                 SGB-006 scale-up, Qwen2.5-7B-Instruct — namespaced so it
#     ├── sft/            can never collide with the 1.5B checkpoints above.
#     ├── rm/
#     ├── rm_hodge/
#     ├── ppo/
#     └── ppo_hodge/
#
# SGB-006 model choice: Qwen2.5-7B-Instruct, not the queue's originally-suggested
# DeepSeek-R1-Distill-Qwen-7B or Llama-8B. Same family/version as the 1.5B
# baseline isolates scale as the only changed variable (a reasoning-distilled
# model would also change response format/verbosity; Llama needs gated HF access
# this repo's token has already been confirmed not to have -- see SGB-012 notes).

_MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"


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
# Stage 5 (SGB-005b) — Score reference pairs with the RM
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="L4",
    timeout=1800,
    memory=16384,
    volumes={"/checkpoints": ckpt_vol, "/results": results_vol},
    secrets=_SECRETS,
)
def score_pairs_rm(hodge: bool = False) -> dict:
    """Score every train+holdout (ideal_text, exploit_text) reference pair with a trained RM.

    Unlike `evaluate()`, this scores the fixed reference texts directly —
    no policy generation involved. Used for the SGB-005b "Hodge as
    featurizer" test: whether the Hodge potential-diff (computed separately,
    locally, from embeddings) adds ranking signal beyond the RM's own scalar
    score, without requiring a new PPO run.

    Returns:
        {"train": [...], "holdout": [...]} — each a list of
        {exploit_text, ideal_text, rm_exploit, rm_ideal} dicts.
    """
    _setup_paths()
    _hf_cache_dir()
    ckpt_vol.reload()

    import json
    import torch
    from pathlib import Path as P

    from shared.src.config import PipelineConfig
    from shared.src.lm_finetuning import FineTuneConfig, load_reward_model, SYSTEM_PROMPT

    config = PipelineConfig()
    config.trace_max_samples   = 517
    config.hh_rlhf_max_samples = 0
    config.cache_dir           = "/app/shared/data/cache"

    ft_config = FineTuneConfig()
    ft_config.checkpoint_dir = "/checkpoints"

    checkpoint = "/checkpoints/rm_hodge" if hodge else "/checkpoints/rm"
    rm_model, rm_tokenizer = load_reward_model(ft_config, checkpoint=checkpoint)
    rm_model.eval()
    # Match build_rm_dataset/_fmt exactly (chat-templated system+context+
    # response, left-truncated) -- scoring raw response text alone, or with
    # the tokenizer's default right-truncation, does not match what the RM
    # was trained on and gives meaningless numbers (caught via a sanity
    # check: hodge RM had *lower* train loss than standard but *worse*
    # holdout "accuracy" under the old raw-text scoring -- backwards).
    rm_tokenizer.truncation_side = "left"

    def _fmt(context_text: str, response: str) -> str:
        msgs = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": context_text.strip()},
            {"role": "assistant", "content": response.strip()},
        ]
        return rm_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    def _score(context_text: str, response: str) -> float:
        text = _fmt(context_text, response)
        enc = rm_tokenizer(
            text, truncation=True, max_length=ft_config.rm_max_length,
            return_tensors="pt",
        ).to(rm_model.device)
        with torch.no_grad():
            return rm_model(**enc).logits.squeeze(-1).item()

    out = {}
    for split in ("train", "holdout"):
        pairs, _ = _load_pairs(config, split=split)
        rows = []
        for p in pairs:
            rows.append({
                "exploit_text": p.exploit_text,
                "ideal_text":   p.ideal_text,
                "rm_exploit":   _score(p.context_text, p.exploit_text),
                "rm_ideal":     _score(p.context_text, p.ideal_text),
            })
        out[split] = rows
        print(f"scored {len(rows)} pairs [{split}]", flush=True)

    out_path = f"/results/finetune/sgb005b_rm_scores{'_hodge' if hodge else ''}.json"
    P(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    results_vol.commit()
    print(f"  → {out_path}", flush=True)

    return out


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    memory=32768,
    volumes={"/checkpoints": ckpt_vol, "/results": results_vol},
    secrets=_SECRETS,
)
def score_pairs_rm_7b(hodge: bool = False) -> dict:
    """SGB-006 pre-PPO gate: verify the 7B RM generalizes before spending an
    A100-80GB PPO run against it.

    Identical purpose to `score_pairs_rm` (SGB-005b) but reusable as the
    correctness gate SGB-005c established: both RM-7B training losses
    saturated to ~0 (0.046 standard, 0.082 hodge), the same shape as the 1.5B
    RMs before the truncation-bug fix -- so this must be checked BEFORE
    launching PPO, not inferred from train loss alone.
    """
    _setup_paths()
    _hf_cache_dir()
    ckpt_vol.reload()

    import json
    import torch
    from pathlib import Path as P

    from shared.src.config import PipelineConfig
    from shared.src.lm_finetuning import FineTuneConfig, load_reward_model, SYSTEM_PROMPT

    config = PipelineConfig()
    config.trace_max_samples   = 517
    config.hh_rlhf_max_samples = 0
    config.cache_dir           = "/app/shared/data/cache"

    ft_config = FineTuneConfig()
    ft_config.model_name     = _MODEL_7B
    ft_config.checkpoint_dir = "/checkpoints/7b"

    checkpoint = "/checkpoints/7b/rm_hodge" if hodge else "/checkpoints/7b/rm"
    rm_model, rm_tokenizer = load_reward_model(ft_config, checkpoint=checkpoint)
    rm_model.eval()
    # Same fix as SGB-005c's score_pairs_rm: match build_rm_dataset's exact
    # chat-template + left-truncation format, or the numbers are meaningless.
    rm_tokenizer.truncation_side = "left"

    def _fmt(context_text: str, response: str) -> str:
        msgs = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": context_text.strip()},
            {"role": "assistant", "content": response.strip()},
        ]
        return rm_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    def _score(context_text: str, response: str) -> float:
        text = _fmt(context_text, response)
        enc = rm_tokenizer(
            text, truncation=True, max_length=ft_config.rm_max_length,
            return_tensors="pt",
        ).to(rm_model.device)
        with torch.no_grad():
            return rm_model(**enc).logits.squeeze(-1).item()

    out = {}
    for split in ("train", "holdout"):
        pairs, _ = _load_pairs(config, split=split)
        rows = []
        for p in pairs:
            rows.append({
                "exploit_text": p.exploit_text,
                "ideal_text":   p.ideal_text,
                "rm_exploit":   _score(p.context_text, p.exploit_text),
                "rm_ideal":     _score(p.context_text, p.ideal_text),
            })
        out[split] = rows
        acc = sum(1 for r in rows if r["rm_ideal"] > r["rm_exploit"]) / len(rows)
        print(f"[7B] scored {len(rows)} pairs [{split}]  ranking_accuracy={acc:.2%}", flush=True)

    out_path = f"/results/finetune/7b_rm_verify_scores{'_hodge' if hodge else ''}.json"
    P(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    results_vol.commit()
    print(f"  → {out_path}", flush=True)

    return out


# ---------------------------------------------------------------------------
# SGB-006 — 7B scale-up (Qwen2.5-7B-Instruct)
# ---------------------------------------------------------------------------
# Same 4-stage pipeline as Stages 1-4 above, parameterized to a bigger base
# model and namespaced under /checkpoints/7b/ + /results/finetune/7b_*. PPO
# alone needs a bigger GPU: run_ppo holds three full model copies at once
# (trainable policy, frozen SFT reference, frozen reward model) -- at 7B
# that's ~42GB of bf16 weights before any activations/KV-cache, which does
# not fit on A100-40GB. SFT/RM/eval only ever hold one or two 7B copies at
# once, so A100-40GB has enough headroom there.

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=14400,
    memory=65536,
    volumes={"/checkpoints": ckpt_vol, "/results": results_vol},
    secrets=_SECRETS,
)
def train_sft_7b() -> str:
    """SGB-006: SFT warm-start at 7B (Qwen2.5-7B-Instruct)."""
    _setup_paths()
    _hf_cache_dir()
    ckpt_vol.reload()

    from shared.src.config import PipelineConfig
    from shared.src.lm_finetuning import FineTuneConfig, run_sft

    config = PipelineConfig()
    config.trace_max_samples   = 517
    config.hh_rlhf_max_samples = 0
    config.cache_dir           = "/app/shared/data/cache"

    ft_config = FineTuneConfig()
    ft_config.model_name     = _MODEL_7B
    ft_config.checkpoint_dir = "/checkpoints/7b"

    pairs, _ = _load_pairs(config, split="train")
    print(f"[7B] SFT: {len(pairs)} (context, ideal) pairs loaded  [train split, {_HOLDOUT_FRAC:.0%} held out]")

    output_dir = "/checkpoints/7b/sft"
    run_sft(pairs, ft_config, output_dir)

    ckpt_vol.commit()
    print(f"[7B] SFT complete → {output_dir}")
    return output_dir


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=14400,
    memory=65536,
    volumes={"/checkpoints": ckpt_vol, "/results": results_vol},
    secrets=_SECRETS,
)
def train_reward_model_7b(hodge: bool = False) -> str:
    """SGB-006: reward model training at 7B (standard or Hodge-weighted)."""
    _setup_paths()
    _hf_cache_dir()
    ckpt_vol.reload()

    from shared.src.config import PipelineConfig
    from shared.src.lm_finetuning import FineTuneConfig, run_reward_model_training

    config = PipelineConfig()
    config.trace_max_samples   = 517
    config.hh_rlhf_max_samples = 0
    config.cache_dir           = "/app/shared/data/cache"

    ft_config = FineTuneConfig()
    ft_config.model_name     = _MODEL_7B
    ft_config.checkpoint_dir = "/checkpoints/7b"

    pairs, _ = _load_pairs(config, split="train")
    print(f"[7B] RM training: {len(pairs)} pairs  hodge={hodge}  [train split, {_HOLDOUT_FRAC:.0%} held out]")

    output_dir = "/checkpoints/7b/rm_hodge" if hodge else "/checkpoints/7b/rm"
    run_reward_model_training(
        pairs, ft_config, output_dir,
        pipeline_config=config if hodge else None,
        use_hodge=hodge,
    )

    ckpt_vol.commit()
    print(f"[7B] RM training complete → {output_dir}")
    return output_dir


@app.function(
    image=image,
    # See module-level note above: 3 x 7B bf16 copies (~42GB) doesn't fit on
    # A100-40GB, so PPO alone gets the bigger GPU.
    gpu="A100-80GB",
    # First real 256-step attempt was killed by the previous 21600s (6h)
    # timeout at step 155/256 -- observed real throughput was ~139s/step
    # (not the ~30s/step a short window suggested), so 256 steps needs
    # ~9.9h. Bumped to 14h for margin; the periodic-checkpoint safeguard
    # below means even hitting this ceiling now loses at most
    # ppo_checkpoint_every steps of progress, not the whole run.
    timeout=50400,
    memory=131072,
    volumes={"/checkpoints": ckpt_vol, "/results": results_vol},
    secrets=_SECRETS,
)
def train_ppo_7b(hodge: bool = False, experiment_id: str = "",
                  ppo_steps: int = 0) -> dict:
    """SGB-006: PPO at 7B against the 7B reward model.

    Same manifest-tracked dispatch pattern as `train_ppo` (SGB-003); tagged
    under inquiry_id SGB-006 so its manifests don't mix with the 1.5B ones.
    `ppo_steps` override exists for the same reason it does in `train_ppo`:
    a cheap smoke test before committing to the full step count, which costs
    substantially more wall-clock per step at 7B than at 1.5B.
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
        division="SGB", inquiry_id="SGB-006", experiment_class=f"train_ppo_7b[{tag}]",
        paths=[f"/checkpoints/7b/_manifests/{experiment_id}.json",
               f"/results/_manifests/{experiment_id}.json"],
        kwargs={"hodge": hodge}, modal={"gpu": "A100-80GB", "hodge": hodge},
        git_root=str(_ai_research_root), volume_commit=results_vol.commit,
        experiment_id=experiment_id,
    ) as m:
        config = PipelineConfig()
        config.trace_max_samples   = 517
        config.hh_rlhf_max_samples = 0
        config.cache_dir           = "/app/shared/data/cache"

        ft_config = FineTuneConfig()
        ft_config.model_name     = _MODEL_7B
        ft_config.checkpoint_dir = "/checkpoints/7b"
        if ppo_steps > 0:
            ft_config.ppo_steps = ppo_steps
        # First smoke test OOM'd on A100-80GB at the default ppo_batch_size=8
        # (79.18/79.25 GiB in use before the backward pass finished one
        # batch) -- the backward-pass activation memory for a batch of 8
        # sequences through a 7B model was the real cost, not just the three
        # weight copies. First fix tried gradient checkpointing + batch=2
        # together; batch=2 turned out to be a SEPARATE bug: adv = (r-mean)/std
        # over exactly 2 samples is PROVABLY always +-0.7071 regardless of the
        # actual reward gap (verified numerically -- rewards differing by 4,
        # by 150, or by 0.003 all give the same +-0.7071), so PPO's gradient
        # carried only the sign of which response won, not the magnitude. This
        # is why ppo_loss printed the identical value at nearly every step of
        # both the discarded standard and hodge runs. Gradient checkpointing
        # alone (no batch reduction) is the fix that doesn't also break the
        # advantage estimate; keep batch_size at FineTuneConfig's default (8,
        # same as the 1.5B pipeline) unless a fresh smoke test shows it still
        # doesn't fit.
        ft_config.ppo_gradient_checkpointing = True
        # LoRA-only adapter (~40MB) -- cheap to write and commit often. Worst
        # case a kill lands right before a checkpoint: ~10 steps x ~139s/step
        # ~= 23min / ~$1 lost, not the whole run.
        ft_config.ppo_checkpoint_every = 10

        _, hacked_records = _load_pairs(config, split="train")
        print(f"[7B] PPO: {len(hacked_records)} exploit prompts as queries  "
              f"hodge={hodge}  ppo_steps={ft_config.ppo_steps}  "
              f"ppo_batch_size={ft_config.ppo_batch_size}  "
              f"[train split, {_HOLDOUT_FRAC:.0%} held out]", flush=True)

        sft_ckpt = "/checkpoints/7b/sft"
        rm_ckpt  = "/checkpoints/7b/rm_hodge" if hodge else "/checkpoints/7b/rm"
        out_dir  = "/checkpoints/7b/ppo_hodge" if hodge else "/checkpoints/7b/ppo"

        def _on_checkpoint(steps_done: int) -> None:
            ckpt_vol.commit()
            print(f"  [7B] checkpoint committed at step {steps_done}", flush=True)

        stats = run_ppo(
            records       = hacked_records,
            sft_checkpoint= sft_ckpt,
            rm_checkpoint = rm_ckpt,
            config        = ft_config,
            output_dir    = out_dir,
            on_checkpoint = _on_checkpoint,
        )
        stats["hodge"] = hodge

        P("/results/finetune").mkdir(parents=True, exist_ok=True)
        stats_path = f"/results/finetune/7b_{tag}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        ckpt_vol.commit()
        results_vol.commit()
        m.add_checkpoint(tag, ckpt_vol.name, out_dir)
        m.add_result_file(f"7b_{tag}_stats", results_vol.name, stats_path)
        m.results = stats
        print(f"[7B] PPO complete → {out_dir}  mean_reward={stats['mean_reward_final']:.4f}")

    return stats


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=21600,
    memory=65536,
    volumes={"/checkpoints": ckpt_vol, "/results": results_vol},
    secrets=_SECRETS,
)
def evaluate_7b(n_eval: int = 50) -> dict:
    """SGB-006: base / SFT / PPO / Hodge-PPO exploit resistance at 7B."""
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
    ft_config.model_name     = _MODEL_7B
    ft_config.checkpoint_dir = "/checkpoints/7b"

    _, hacked_records = _load_pairs(config, split="holdout")
    print(f"[7B] Eval set: {len(hacked_records)} held-out exploit prompts "
          f"[holdout split, {_HOLDOUT_FRAC:.0%} of total]")

    rm_ckpt = (
        "/checkpoints/7b/rm_hodge"
        if P("/checkpoints/7b/rm_hodge").exists()
        else "/checkpoints/7b/rm"
    )

    def _ppo_checkpoint_ready(path: str) -> bool:
        # PPO checkpoints (not SFT) can be a periodic, INCOMPLETE save left by
        # an interrupted run -- the first 7B standard-PPO attempt left exactly
        # this at /checkpoints/7b/ppo (discovered + deleted by hand, not
        # caught by any code check). A checkpoint written by run_ppo's
        # periodic save always carries this manifest; treat its absence as
        # "pre-dates this safeguard, assume complete" rather than block on it.
        progress_path = P(path) / "_ppo_progress.json"
        if not progress_path.exists():
            return True
        progress = json.loads(progress_path.read_text())
        return bool(progress.get("complete", False))

    checkpoints = {"base": None}
    for name, path in [
        ("sft",       "/checkpoints/7b/sft"),
        ("ppo",       "/checkpoints/7b/ppo"),
        ("hodge_ppo", "/checkpoints/7b/ppo_hodge"),
    ]:
        if not P(path).exists():
            continue
        if name != "sft" and not _ppo_checkpoint_ready(path):
            print(f"  [SKIP] {name}: checkpoint at {path} is an incomplete "
                  f"periodic save (_ppo_progress.json complete=False) -- "
                  f"not a valid PPO result, excluding from eval", flush=True)
            continue
        checkpoints[name] = path

    print(f"[7B] Evaluating: {list(checkpoints.keys())}  n_eval={n_eval}")

    P("/results/finetune").mkdir(parents=True, exist_ok=True)
    out_path = "/results/finetune/7b_eval_comparison.json"

    def _on_checkpoint_done(name: str, result: dict) -> None:
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

    print("\n=== [7B] Exploit Resistance Comparison ===")
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
    stage:  str  = "all",   # sft | rm | ppo | eval | score-pairs | all |
                            # sft-7b | rm-7b | ppo-7b | eval-7b (SGB-006, never
                            # part of "all" -- launch these one at a time)
    hodge:  bool = False,
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

    # SGB-005b: score reference pairs with the standard RM (featurizer test)
    modal run shared/modal_finetune.py --stage score-pairs

    # Fire-and-forget (survives local disconnect) -- --detach is a flag to
    # `modal run` itself, must precede the script path, not follow it
    modal run --detach shared/modal_finetune.py --hodge
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

    if stage == "score-pairs":
        print(f"--- Stage 5 (SGB-005b): Score reference pairs with RM (hodge={hodge}) ---")
        result = _run(score_pairs_rm, hodge=hodge)
        print(f"  → train={len(result['train'])} holdout={len(result['holdout'])} pairs scored")

    # --- SGB-006: 7B scale-up. Deliberately NOT folded into "all" -- these are
    # separate, much more expensive stages and should always be launched one
    # at a time so a bad launch doesn't burn an A100-80GB PPO run by accident.
    if stage == "sft-7b":
        print("--- SGB-006 Stage 1: SFT (7B) ---")
        result = _run(train_sft_7b)
        print(f"  → {result}")

    if stage == "rm-7b":
        print(f"--- SGB-006 Stage 2: Reward Model (7B, hodge={hodge}) ---")
        result = _run(train_reward_model_7b, hodge=hodge)
        print(f"  → {result}")

    if stage == "verify-rm-7b":
        print(f"--- SGB-006 pre-PPO gate: verify RM-7B generalization (hodge={hodge}) ---")
        result = _run(score_pairs_rm_7b, hodge=hodge)
        train_acc = sum(1 for r in result["train"] if r["rm_ideal"] > r["rm_exploit"]) / len(result["train"])
        hold_acc  = sum(1 for r in result["holdout"] if r["rm_ideal"] > r["rm_exploit"]) / len(result["holdout"])
        print(f"  → train_acc={train_acc:.2%}  holdout_acc={hold_acc:.2%}")

    if stage == "ppo-7b":
        print(f"--- SGB-006 Stage 3: PPO (7B, hodge={hodge}) ---")
        from shared_modal import mirror_from_volume, new_experiment_id
        exp_id = new_experiment_id()
        print(f"  manifest id: {exp_id}  (SGB-006 — see modal_finetune.py train_ppo_7b)")
        try:
            result = _run(train_ppo_7b, hodge=hodge, experiment_id=exp_id,
                          ppo_steps=ppo_steps)
            print(f"  → {result}")
        finally:
            dest = mirror_from_volume(
                "reward-hacking-results", "/results", exp_id,
                division="SGB", inquiry_id="SGB-006",
            )
            print(f"  manifest mirrored to: {dest}")

    if stage == "eval-7b":
        print("--- SGB-006 Stage 4: Evaluation (7B) ---")
        result = _run(evaluate_7b, n_eval=n_eval)
        print(f"  → {result}")
