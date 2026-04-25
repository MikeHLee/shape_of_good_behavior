"""Generate extended-thinking counterfactuals for all 268 hacked TRACE records.

Saves to shared/data/cache/counterfactuals.json (incremental — safe to interrupt
and resume; already-cached records are skipped).

Usage:
    ./venv/bin/python3 scripts/generate_trace_counterfactuals.py
"""
import logging
import sys
import time
from pathlib import Path

# Load .env before any anthropic import
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.src.config import PipelineConfig
from shared.src.data_ingest import ingest_all
from shared.src.counterfactual_gen import CounterfactualGenerator, _cache_key

config = PipelineConfig()
config.trace_max_samples = 517
config.hh_rlhf_max_samples = 0

logger.info("Loading TRACE dataset...")
result = ingest_all(config, sources=["trace"])
hacked = result.filter_hacked()
logger.info(f"{len(hacked)} hacked TRACE records to process")

gen = CounterfactualGenerator(config)
already_cached = sum(1 for r in hacked if _cache_key(r) in gen._cache)
to_generate = len(hacked) - already_cached
logger.info(f"Already cached: {already_cached}  |  To generate: {to_generate}")
logger.info(f"Model: {config.llm_model}  |  Thinking budget: {config.llm_thinking_budget} tokens")
logger.info(f"Cache: {gen.cache_path}")

if to_generate == 0:
    logger.info("All records already cached — nothing to do.")
    sys.exit(0)

errors = 0
generated = 0
t_start = time.time()

for i, record in enumerate(hacked):
    key = _cache_key(record)
    if key in gen._cache:
        continue  # skip cached

    traj_id = record.metadata.get("trajectory_id", f"record_{i}")
    logger.info(f"[{generated + 1}/{to_generate}] {traj_id} ({record.exploit_category})")

    pair = gen.generate_for_trace(record)

    if pair.confidence < 0.2:
        errors += 1
        logger.warning(f"  Low confidence ({pair.confidence:.2f}) — {pair.failure_analysis[:80]}")
    else:
        generated += 1
        elapsed = time.time() - t_start
        per_call = elapsed / max(generated, 1)
        remaining = (to_generate - generated - errors) * per_call
        logger.info(
            f"  exploit_type={pair.exploit_type!r}  "
            f"confidence={pair.confidence:.2f}  "
            f"thinking_words={pair.metadata.get('thinking_tokens', 0)}  "
            f"ETA={remaining/60:.1f}min"
        )

elapsed_total = time.time() - t_start
logger.info(
    f"\nDone. Generated: {generated}  Errors: {errors}  "
    f"Total time: {elapsed_total/60:.1f}min  "
    f"Cache: {gen.cache_path}"
)
