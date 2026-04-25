"""Modal GPU runner for the cross-track reward hacking experiment pipeline.

Runs GPU-intensive stages (embedding, RM training, Hodge analysis) on Modal.
LLM counterfactual generation runs locally (uses Anthropic API key).

Usage:
    # Quick run (5 seeds, small data)
    modal run shared/modal_runner.py --quick

    # Full run (30 seeds, all data)
    modal run shared/modal_runner.py --full

    # Track-specific
    modal run shared/modal_runner.py --track 1
"""

from pathlib import Path

import modal

# Read requirements from the modal requirements file.
# This only needs to succeed locally (at image build time).
# On the container, __file__ resolves to /root/modal_runner.py where the
# requirements file doesn't exist — but deps are already installed.
_req_path = Path(__file__).parent / "requirements-modal.txt"
try:
    _requirements = [
        line.strip()
        for line in _req_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
except FileNotFoundError:
    _requirements = []

app = modal.App("reward-hacking-pipeline-v2")

_project_root = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_requirements)
    # Embed shared pipeline code + track source code
    .add_local_dir(str(_project_root / "shared"), remote_path="/app/shared")
    .add_local_dir(str(_project_root / "feedback_geometry" / "src"), remote_path="/app/feedback_geometry/src")
    .add_local_dir(str(_project_root / "src"), remote_path="/app/src")
)

vol = modal.Volume.from_name("reward-hacking-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,  # 2 hours — optimizer comparison only
    memory=16384,
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def run_optimizer_comparison(num_seeds: int = 30):
    """Run Stage 5 (optimizer comparison) standalone, using cached mapping from volume."""
    import sys, json, time, pickle
    import numpy as np

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/feedback_geometry/src")
    sys.path.insert(0, "/app/src")

    import shared.src.config as cfg_mod
    cfg_mod.PROJECT_ROOT = Path("/app")
    cfg_mod.SHARED_ROOT = Path("/app/shared")
    cfg_mod.FEEDBACK_GEOMETRY_SRC = Path("/app/feedback_geometry/src")
    cfg_mod.CONSTRAINT_GEOMETRY_SRC = Path("/app/src")

    from shared.src.config import PipelineConfig
    from shared.src.optimizer_comparison import OptimizerBenchmark

    config = PipelineConfig()
    config.results_dir = "/results/pipeline"
    config.figures_dir = "/results/figures"
    config.data_dir = "/results/data"
    config.cache_dir = "/results/cache"
    config.rm_epochs = 50

    # Load cached mapping from volume
    mapping_path = "/results/pipeline/mapping.pkl"
    if not Path(mapping_path).exists():
        raise FileNotFoundError(
            f"No cached mapping at {mapping_path}. Run the full pipeline first."
        )
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)

    print(f"Loaded mapping: {len(mapping.embedding_pairs)} pairs")

    # Subsample embedding_pairs for memory safety, but keep ALL preference_edges
    # (cross-pair k-NN edges are critical for Hodge decomposition — H1=0 without them)
    if len(mapping.embedding_pairs) > 500:
        from shared.src.preference_mapper import MappingResult
        idx = np.random.choice(len(mapping.embedding_pairs), 500, replace=False)
        # Collect node IDs referenced by retained pairs
        retained_nodes = set()
        for i in idx:
            e = mapping.preference_edges[i]  # direct edges come first
            retained_nodes.add(e[0])
            retained_nodes.add(e[1])
        # Keep all edges (direct + cross-pair) that reference retained nodes
        kept_edges = [
            e for e in mapping.preference_edges
            if e[0] in retained_nodes and e[1] in retained_nodes
        ]
        mapping = MappingResult(
            preference_edges=kept_edges,
            n_items=mapping.n_items,
            embedding_pairs=[mapping.embedding_pairs[i] for i in idx],
            danger_regions=mapping.danger_regions,
            constitutional_gradients=mapping.constitutional_gradients,
            exploit_embeddings_reduced=mapping.exploit_embeddings_reduced[idx],
            ideal_embeddings_reduced=mapping.ideal_embeddings_reduced[idx],
            exploit_embeddings=mapping.exploit_embeddings[idx],
            ideal_embeddings=mapping.ideal_embeddings[idx],
        )
        print(f"Subsampled to 500 pairs, {len(kept_edges)} edges (incl. cross-pair)")

    try:
        benchmark = OptimizerBenchmark(config, mapping)
        print(f"Starting {num_seeds}-seed benchmark...", flush=True)
        comp_table = benchmark.run(num_seeds=num_seeds)
        print(OptimizerBenchmark.print_table(comp_table), flush=True)
        path = benchmark.save_results(comp_table)
        print(f"Saved to {path}", flush=True)
        vol.commit()

        return {k: {kk: vv for kk, vv in v.items() if kk != "values"}
                for k, v in comp_table.method_stats.items()}
    except Exception as e:
        import traceback
        err_msg = f"FAILED: {e}\n{traceback.format_exc()}"
        print(err_msg, flush=True)
        # Write error to volume so we can diagnose detached failures
        Path("/results/pipeline").mkdir(parents=True, exist_ok=True)
        with open("/results/pipeline/error.log", "w") as f:
            f.write(err_msg)
        vol.commit()
        raise


@app.function(
    image=image,
    gpu="L4",
    timeout=14400,  # 4 hours
    memory=16384,
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def run_pipeline(
    quick: bool = True,
    full: bool = False,
    track: str = "all",
    num_seeds: int = 5,
):
    """Run the GPU-intensive pipeline stages on Modal.

    Stages run on GPU:
    - Stage 3: Embedding computation (sentence-transformers)
    - Stage 4: Hodge analysis + RM training (PyTorch)
    - Stage 5: Evaluation
    - Stage 6: Visualization

    Stages that run locally (NOT here):
    - Stage 1: Data ingestion (downloads from HuggingFace — can run here or locally)
    - Stage 2: LLM counterfactual generation (requires Anthropic API key)
    """
    import sys
    import json
    import time
    # Set up paths for embedded source code
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/feedback_geometry/src")
    sys.path.insert(0, "/app/src")

    # Override config module-level paths for container environment
    import shared.src.config as cfg_mod
    cfg_mod.PROJECT_ROOT = Path("/app")
    cfg_mod.SHARED_ROOT = Path("/app/shared")
    cfg_mod.FEEDBACK_GEOMETRY_SRC = Path("/app/feedback_geometry/src")
    cfg_mod.CONSTRAINT_GEOMETRY_SRC = Path("/app/src")

    import torch
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running pipeline on device: {device}")
    print(f"Config: quick={quick}, full={full}, track={track}, seeds={num_seeds}")

    # Import pipeline modules
    from shared.src.config import PipelineConfig
    from shared.src.data_ingest import ingest_all
    from shared.src.counterfactual_gen import CounterfactualGenerator
    from shared.src.preference_mapper import PreferenceMapper
    from shared.src.hodge_analysis import HodgeRewardHackingAnalyzer
    from shared.src.visualize import AlignmentVisualizer

    # Configure
    config = PipelineConfig()
    config.results_dir = "/results/pipeline"
    config.figures_dir = "/results/figures"
    config.data_dir = "/results/data"
    config.cache_dir = "/results/cache"

    # Always load full TRACE (517 rows) + scale HH-RLHF by mode
    config.trace_max_samples = 517   # all of TRACE
    config.pku_safe_max_samples = 0  # not yet — keep run focused on TRACE
    config.beaver_tails_max_samples = 0
    config.adv_bench_max_samples = 0
    if quick:
        config.hh_rlhf_max_samples = 500
        config.rm_epochs = 30
    elif full:
        config.hh_rlhf_max_samples = 2000
        config.rm_epochs = 100

    start = time.time()

    # Stage 1: Ingest — TRACE (gated, needs HF_TOKEN) + HH-RLHF
    print("=" * 60)
    print("STAGE 1: Data Ingestion")
    ingest_result = ingest_all(config, sources=["trace", "hh_rlhf"])
    print(
        f"Loaded {ingest_result.total} records: "
        f"{ingest_result.trace_count} TRACE + {ingest_result.hh_rlhf_count} HH-RLHF"
    )

    # Stage 2: Build preference pairs
    # TRACE records with is_hacked=True have no pre-cached ideal — generate direct pairs
    # from both the TRACE benign responses and HH-RLHF chosen/rejected.
    print("=" * 60)
    print("STAGE 2: Building preference pairs (skip-llm mode)")
    generator = CounterfactualGenerator(config)

    # HH-RLHF records carry chosen/rejected natively via skip_llm
    hh_pairs = generator.generate_batch(
        [r for r in ingest_result.records if r.source == "hh_rlhf"],
        skip_llm=True,
    )
    # TRACE hacked records: use trajectory final turn as exploit, skip_llm=True
    # (no ideal counterpart without LLM — pairs generated as exploit-only stubs
    # that CounterfactualGenerator handles as self-pairs with is_hacked flag)
    trace_pairs = generator.generate_batch(
        [r for r in ingest_result.records if r.source == "trace"],
        skip_llm=True,
    )
    pairs = hh_pairs + trace_pairs
    print(f"Built {len(pairs)} pairs (HH-RLHF: {len(hh_pairs)}, TRACE: {len(trace_pairs)})")

    # Stage 3: Preference mapping (GPU-accelerated embeddings)
    print("=" * 60)
    print("STAGE 3: Preference Mapping (GPU embeddings)")
    mapper = PreferenceMapper(config)
    mapping = mapper.map_pairs(pairs)
    print(f"Edges: {len(mapping.preference_edges)}, "
          f"Pairs: {len(mapping.embedding_pairs)}, "
          f"Danger regions: {len(mapping.danger_regions)}")

    # Cache mapping to volume for standalone optimizer comparison
    import pickle
    Path("/results/pipeline").mkdir(parents=True, exist_ok=True)
    with open("/results/pipeline/mapping.pkl", "wb") as f:
        pickle.dump(mapping, f)
    vol.commit()
    print("Mapping cached to volume.")

    results = {"stages_completed": []}

    # Stage 4: Hodge analysis (GPU RM training)
    h1_result = None
    multi_seed = None
    if track in ("all", "1"):
        print("=" * 60)
        print(f"STAGE 4: Hodge Analysis ({num_seeds} seeds)")
        analyzer = HodgeRewardHackingAnalyzer(config, mapping)
        h1_result = analyzer.analyze_h1_structure()
        multi_seed = analyzer.run_multi_seed(num_seeds=num_seeds)
        results["h1_overall"] = h1_result.h1_overall
        results["standard_mean"] = multi_seed.standard_stats.get("mean", 0)
        results["hodge_mean"] = multi_seed.hodge_stats.get("mean", 0)
        results["stages_completed"].append("hodge_analysis")

    # Stage 5: Optimizer comparison (DPO, GRPO, ORPO, KTO + Hodge variants)
    if track in ("all", "1"):
        print("=" * 60)
        print(f"STAGE 5: Optimizer Comparison ({num_seeds} seeds)")
        try:
            from shared.src.optimizer_comparison import OptimizerBenchmark

            # Cap samples and epochs for memory safety on L4 (16GB)
            # Full embedding dim (384) × 7 methods × 30 seeds is expensive
            comp_config = PipelineConfig()
            comp_config.__dict__.update(config.__dict__)
            comp_config.rm_epochs = min(config.rm_epochs, 50)
            comp_config.results_dir = config.results_dir

            # Subsample pairs but keep ALL preference edges (cross-pair k-NN
            # edges are critical for Hodge decomposition)
            sub_mapping = mapping
            if len(mapping.embedding_pairs) > 500:
                import numpy as np
                from shared.src.preference_mapper import MappingResult as MR
                idx = np.random.choice(len(mapping.embedding_pairs), 500, replace=False)
                retained_nodes = set()
                for i in idx:
                    e = mapping.preference_edges[i]
                    retained_nodes.add(e[0])
                    retained_nodes.add(e[1])
                kept_edges = [
                    e for e in mapping.preference_edges
                    if e[0] in retained_nodes and e[1] in retained_nodes
                ]
                sub_mapping = MR(
                    preference_edges=kept_edges,
                    n_items=mapping.n_items,
                    embedding_pairs=[mapping.embedding_pairs[i] for i in idx],
                    danger_regions=mapping.danger_regions,
                    constitutional_gradients=mapping.constitutional_gradients,
                    exploit_embeddings_reduced=mapping.exploit_embeddings_reduced[idx],
                    ideal_embeddings_reduced=mapping.ideal_embeddings_reduced[idx],
                    exploit_embeddings=mapping.exploit_embeddings[idx],
                    ideal_embeddings=mapping.ideal_embeddings[idx],
                )
                print(f"Subsampled to 500 pairs, {len(kept_edges)} edges")
            benchmark = OptimizerBenchmark(comp_config, sub_mapping)

            comp_table = benchmark.run(num_seeds=num_seeds)
            markdown = OptimizerBenchmark.print_table(comp_table)
            benchmark.save_results(comp_table)
            results["optimizer_comparison"] = {
                k: {kk: vv for kk, vv in v.items() if kk != "values"}
                for k, v in comp_table.method_stats.items()
            }
            results["stages_completed"].append("optimizer_comparison")
        except Exception as e:
            print(f"Optimizer comparison failed: {e}")
            import traceback
            traceback.print_exc()

    # Stage 6: Visualization
    print("=" * 60)
    print("STAGE 6: Visualization")
    tracks = [track] if track != "all" else ["all"]
    viz = AlignmentVisualizer(config, mapping, h1_result, multi_seed)
    fig_paths = viz.generate_all() if track == "all" else {}
    results["figures"] = {k: v for k, v in fig_paths.items() if v}
    results["stages_completed"].append("visualization")

    elapsed = time.time() - start
    results["elapsed_seconds"] = elapsed
    print(f"\nPipeline completed in {elapsed:.1f}s")

    # Save summary
    summary_path = "/results/pipeline/summary.json"
    Path("/results/pipeline").mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    vol.commit()
    print(f"Results saved to Modal volume: reward-hacking-results")
    return results


@app.local_entrypoint()
def main(
    quick: bool = True,
    full: bool = False,
    track: str = "all",
    seeds: int = 5,
    detach: bool = False,
    compare_only: bool = False,
):
    """Local entrypoint — dispatches to Modal GPU.

    Flags:
        --quick         5-seed small run (default)
        --full          30-seed full run
        --detach        Fire-and-forget (survives local disconnect)
        --compare-only  Run optimizer comparison only (needs cached mapping)
    """
    if compare_only:
        if detach:
            fc = run_optimizer_comparison.spawn(num_seeds=max(seeds, 30) if full else seeds)
            print(f"Spawned optimizer comparison: {fc.object_id}")
        else:
            result = run_optimizer_comparison.remote(num_seeds=max(seeds, 30) if full else seeds)
            print(f"Optimizer comparison result: {result}")
        return

    if full:
        seeds = max(seeds, 30)
        quick = False

    if detach:
        fc = run_pipeline.spawn(
            quick=quick,
            full=full,
            track=track,
            num_seeds=seeds,
        )
        print(f"Spawned detached job: {fc.object_id}")
        print("Check results later with:")
        print("  modal volume ls reward-hacking-results")
    else:
        result = run_pipeline.remote(
            quick=quick,
            full=full,
            track=track,
            num_seeds=seeds,
        )
        print(f"\nModal run complete: {result}")
