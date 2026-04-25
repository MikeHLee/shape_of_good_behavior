#!/usr/bin/env python3
"""End-to-end orchestration for the cross-track reward hacking experiment pipeline.

Usage:
    python -m shared.src.run_pipeline --quick
    python -m shared.src.run_pipeline --full --track all
    python -m shared.src.run_pipeline --skip-llm --viz-only
    python -m shared.src.run_pipeline --track 1  # Track 1 only (Hodge)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .data_ingest import ingest_all, IngestResult
from .counterfactual_gen import CounterfactualGenerator, CounterfactualPair
from .preference_mapper import PreferenceMapper, MappingResult
from .hodge_analysis import HodgeRewardHackingAnalyzer, H1AnalysisResult, MultiSeedResult
from .reward_hacking_eval import ExploitResistanceEvaluator
from .visualize import AlignmentVisualizer

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_ingest(config: PipelineConfig) -> IngestResult:
    """Stage 1: Load external benchmarks."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Ingestion")
    logger.info("=" * 60)

    result = ingest_all(config)

    logger.info(f"Total records: {result.total}")
    logger.info(f"  TRACE: {result.trace_count}")
    logger.info(f"  HH-RLHF: {result.hh_rlhf_count}")
    logger.info(f"  Categories: {result.category_counts}")

    return result


def run_counterfactual_gen(
    config: PipelineConfig,
    ingest_result: IngestResult,
    skip_llm: bool = False,
) -> list:
    """Stage 2: Generate counterfactual pairs."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Counterfactual Generation")
    logger.info("=" * 60)

    generator = CounterfactualGenerator(config)

    # Only process hacked records for exploit analysis
    hacked_records = ingest_result.filter_hacked()
    logger.info(f"Processing {len(hacked_records)} hacked records")

    pairs = generator.generate_batch(hacked_records, skip_llm=skip_llm)

    # Also include some benign records for contrast
    benign_records = ingest_result.filter_benign()[:100]
    if benign_records:
        benign_pairs = generator.generate_batch(benign_records, skip_llm=skip_llm)
        pairs.extend(benign_pairs)

    # Save pairs
    generator.save_pairs(pairs)

    logger.info(f"Generated {len(pairs)} total counterfactual pairs")
    return pairs


def run_preference_mapping(
    config: PipelineConfig,
    pairs: list,
) -> MappingResult:
    """Stage 3: Map to preference space + embedding space + danger regions."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Preference Mapping")
    logger.info("=" * 60)

    mapper = PreferenceMapper(config)
    mapping = mapper.map_pairs(pairs)

    logger.info(f"Preference edges: {len(mapping.preference_edges)}")
    logger.info(f"Embedding pairs: {len(mapping.embedding_pairs)}")
    logger.info(f"Danger regions: {len(mapping.danger_regions)}")
    logger.info(f"Constitutional gradients: {len(mapping.constitutional_gradients)}")
    logger.info(f"Items: {mapping.n_items}")

    return mapping


def run_hodge_analysis(
    config: PipelineConfig,
    mapping: MappingResult,
    num_seeds: int,
) -> tuple:
    """Stage 4: Hodge decomposition + RM training (Track 1)."""
    logger.info("=" * 60)
    logger.info("STAGE 4: Hodge Analysis (Track 1)")
    logger.info("=" * 60)

    analyzer = HodgeRewardHackingAnalyzer(config, mapping)

    # H1 structure analysis
    h1_result = analyzer.analyze_h1_structure()
    logger.info(f"Overall H1: {h1_result.h1_overall:.4f}")
    logger.info(f"Per-category H1: {h1_result.h1_per_category}")

    # Multi-seed training
    multi_seed = analyzer.run_multi_seed(num_seeds=num_seeds)

    return h1_result, multi_seed


def run_evaluation(
    config: PipelineConfig,
    mapping: MappingResult,
    multi_seed: MultiSeedResult,
):
    """Stage 5: Per-category exploit resistance evaluation."""
    logger.info("=" * 60)
    logger.info("STAGE 5: Exploit Resistance Evaluation")
    logger.info("=" * 60)

    evaluator = ExploitResistanceEvaluator(config, mapping)

    # Get detailed per-category breakdown from the best seed's models
    # (Full comparison requires retraining models, which run_multi_seed already did)
    logger.info("Detailed evaluation from multi-seed results:")
    logger.info(f"  Standard RM: {multi_seed.standard_stats}")
    logger.info(f"  Hodge RM:    {multi_seed.hodge_stats}")
    logger.info(f"  Comparison:  {multi_seed.comparison}")

    return multi_seed


def run_visualization(
    config: PipelineConfig,
    mapping: MappingResult,
    h1_result: H1AnalysisResult = None,
    multi_seed: MultiSeedResult = None,
    tracks: list = None,
) -> dict:
    """Stage 6: Generate 3D Plotly visualizations."""
    logger.info("=" * 60)
    logger.info("STAGE 6: Visualization")
    logger.info("=" * 60)

    viz = AlignmentVisualizer(config, mapping, h1_result, multi_seed)

    if tracks is None or "all" in tracks:
        return viz.generate_all()

    paths = {}
    if "1" in tracks or 1 in tracks:
        paths["fig4"] = viz.fig4_hodge_decomposition()
        paths["fig5"] = viz.fig5_hodge_filtering_comparison()
    if "2" in tracks or 2 in tracks:
        paths["fig3"] = viz.fig3_sgpo_trajectory_avoidance()
    if "3" in tracks or 3 in tracks:
        paths["fig1"] = viz.fig1_constitutional_gradient_field()
        paths["fig2"] = viz.fig2_alignment_boundary_topology()

    return paths


def save_results(
    config: PipelineConfig,
    h1_result: H1AnalysisResult = None,
    multi_seed: MultiSeedResult = None,
    figure_paths: dict = None,
):
    """Save all results to JSON."""
    results_dir = Path(config.results_dir)

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "trace_max_samples": config.trace_max_samples,
            "hh_rlhf_max_samples": config.hh_rlhf_max_samples,
            "embed_model": config.embed_model,
            "h1_threshold": config.h1_threshold,
            "reduced_dim": config.reduced_dim,
        },
    }

    if h1_result is not None:
        output["h1_analysis"] = {
            "h1_overall": h1_result.h1_overall,
            "h1_per_category": h1_result.h1_per_category,
            "n_edges": h1_result.n_edges,
            "n_items": h1_result.n_items,
        }

    if multi_seed is not None:
        output["multi_seed"] = {
            "standard_stats": multi_seed.standard_stats,
            "hodge_stats": multi_seed.hodge_stats,
            "comparison": multi_seed.comparison,
            "n_seeds": len(multi_seed.seed_results),
        }

    if figure_paths:
        output["figures"] = figure_paths

    output_path = results_dir / "pipeline_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-track reward hacking experiment pipeline"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run: 5 seeds, small data samples",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full run: 30 seeds, all data",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM calls, use cached counterfactuals only",
    )
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="Only run visualization (requires previous results)",
    )
    parser.add_argument(
        "--track",
        nargs="+",
        default=["all"],
        help="Which tracks to run: 1, 2, 3, or all (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Configure
    config = PipelineConfig()
    if args.quick:
        config.trace_max_samples = 50
        config.hh_rlhf_max_samples = 100
        config.rm_epochs = 30
        num_seeds = config.num_seeds_quick
    elif args.full:
        config.trace_max_samples = 500
        config.hh_rlhf_max_samples = 2000
        num_seeds = config.num_seeds_full
    else:
        num_seeds = config.num_seeds_quick

    logger.info(f"Pipeline config: quick={args.quick}, full={args.full}, "
                f"tracks={args.track}, skip_llm={args.skip_llm}")

    start_time = time.time()

    if args.viz_only:
        # Load previous results and regenerate visualizations
        logger.info("Viz-only mode: loading previous mapping results...")
        # For viz-only, we'd need serialized mapping — for now, re-run stages 1-3
        logger.warning("Viz-only requires re-running data stages (no serialized mapping)")
        ingest_result = run_ingest(config)
        pairs = run_counterfactual_gen(config, ingest_result, skip_llm=True)
        mapping = run_preference_mapping(config, pairs)
        figure_paths = run_visualization(config, mapping, tracks=args.track)
        save_results(config, figure_paths=figure_paths)
    else:
        # Full pipeline
        # Stage 1: Ingest
        ingest_result = run_ingest(config)

        # Stage 2: Counterfactual generation
        pairs = run_counterfactual_gen(config, ingest_result, skip_llm=args.skip_llm)

        # Stage 3: Preference mapping
        mapping = run_preference_mapping(config, pairs)

        # Stage 4: Hodge analysis (Track 1)
        h1_result = None
        multi_seed = None
        if "all" in args.track or "1" in args.track:
            h1_result, multi_seed = run_hodge_analysis(config, mapping, num_seeds)

        # Stage 5: Evaluation
        if multi_seed is not None:
            run_evaluation(config, mapping, multi_seed)

        # Stage 6: Visualization
        figure_paths = run_visualization(
            config, mapping, h1_result, multi_seed, tracks=args.track
        )

        # Save results
        save_results(config, h1_result, multi_seed, figure_paths)

    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
