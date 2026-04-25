"""Load external reward hacking benchmarks into a unified format.

Supports:
- TRACE dataset (PatronusAI/trace-dataset): 517 ChatML trajectories with exploit categories
- HH-RLHF (Anthropic/hh-rlhf): harmless-base and helpful-base splits
- PKU-SafeRLHF (PKU-Alignment/PKU-SafeRLHF): paired responses with safety labels
- BeaverTails (PKU-Alignment/BeaverTails): 14-category harm annotations
- AdvBench (walledai/AdvBench): adversarial instruction prompts
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import (
    PipelineConfig,
    TRACE_DATASET_ID,
    HH_RLHF_DATASET_ID,
    PKU_SAFE_RLHF_DATASET_ID,
    BEAVER_TAILS_DATASET_ID,
    ADV_BENCH_DATASET_ID,
)

logger = logging.getLogger(__name__)


@dataclass
class ExploitRecord:
    """Unified format for a single exploit/benign record from any benchmark."""

    source: str  # "trace" or "hh_rlhf"
    exploit_category: str  # e.g. "nuclear_bioweapon", "sycophancy", "harmless-base"
    is_hacked: bool  # True if this is an exploit/harmful response
    conversation: List[Dict[str, str]]  # [{role, content}, ...]
    exploit_text: str  # The harmful/exploited response text
    context_text: str  # The prompt/context leading to the response
    metadata: Dict = field(default_factory=dict)


@dataclass
class IngestResult:
    """Result of ingesting all benchmarks."""

    records: List[ExploitRecord]
    trace_count: int
    hh_rlhf_count: int
    pku_safe_count: int = 0
    beaver_tails_count: int = 0
    adv_bench_count: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.records)

    def filter_hacked(self) -> List[ExploitRecord]:
        return [r for r in self.records if r.is_hacked]

    def filter_benign(self) -> List[ExploitRecord]:
        return [r for r in self.records if not r.is_hacked]

    def by_category(self) -> Dict[str, List[ExploitRecord]]:
        result: Dict[str, List[ExploitRecord]] = {}
        for r in self.records:
            result.setdefault(r.exploit_category, []).append(r)
        return result


def _parse_chatml(text: str) -> List[Dict[str, str]]:
    """Parse ChatML-formatted text into conversation turns."""
    turns = []
    current_role = None
    current_content = []

    for line in text.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("<|im_start|>"):
            if current_role is not None:
                turns.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip(),
                })
            role_part = line_stripped.replace("<|im_start|>", "").strip()
            current_role = role_part
            current_content = []
        elif line_stripped == "<|im_end|>":
            if current_role is not None:
                turns.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip(),
                })
                current_role = None
                current_content = []
        else:
            if current_role is not None:
                current_content.append(line)

    # Handle unterminated last turn
    if current_role is not None and current_content:
        turns.append({
            "role": current_role,
            "content": "\n".join(current_content).strip(),
        })

    return turns


def _extract_last_assistant(turns: List[Dict[str, str]]) -> str:
    """Extract the last assistant response from conversation turns."""
    for turn in reversed(turns):
        if turn["role"] == "assistant":
            return turn["content"]
    return ""


def _extract_context(turns: List[Dict[str, str]]) -> str:
    """Extract the user prompt context (everything before last assistant turn)."""
    context_parts = []
    for turn in turns:
        if turn["role"] == "assistant" and turn == turns[-1]:
            break
        context_parts.append(f"{turn['role']}: {turn['content']}")
    return "\n".join(context_parts) if context_parts else ""


def load_trace_dataset(config: PipelineConfig) -> List[ExploitRecord]:
    """Load the TRACE dataset from HuggingFace.

    TRACE contains 517 multi-turn agentic trajectories (268 hacked, 249 benign).

    Real schema:
        trajectory_id  str   e.g. "trajectory_0018"
        conversation   str   JSON-encoded list of {role, content} dicts
        label          str   "0" = benign; exploit category code otherwise
                             (e.g. "1.2.3", "1.1.1, 1.3.2" for multi-label)

    exploit_text  = last assistant turn content (the terminal action)
    context_text  = all preceding turns joined as "role: content"
    category      = label string; "benign" when label is "0"
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install `datasets`: pip install datasets")

    import json as _json

    logger.info(f"Loading TRACE dataset: {TRACE_DATASET_ID}")
    # TRACE only has a 'train' split
    ds = load_dataset(TRACE_DATASET_ID, split="train")

    records = []
    for i, row in enumerate(ds):
        if config.trace_max_samples and i >= config.trace_max_samples:
            break

        label = row.get("label", "0") or "0"
        is_hacked = label != "0"
        category = label if is_hacked else "benign"

        # Parse conversation JSON
        raw_conv = row.get("conversation", "[]")
        try:
            turns = _json.loads(raw_conv) if isinstance(raw_conv, str) else raw_conv
        except Exception:
            turns = []

        # Ensure each turn has role + content strings
        turns = [
            {"role": str(t.get("role", "")), "content": str(t.get("content", ""))}
            for t in turns
            if isinstance(t, dict)
        ]

        exploit_text = _extract_last_assistant(turns)
        context_text = _extract_context(turns)

        # Fallback if conversation was empty
        if not exploit_text and not context_text:
            context_text = row.get("trajectory_id", "")

        records.append(ExploitRecord(
            source="trace",
            exploit_category=category,
            is_hacked=is_hacked,
            conversation=turns,
            exploit_text=exploit_text,
            context_text=context_text,
            metadata={
                "index": i,
                "trajectory_id": row.get("trajectory_id", ""),
                "label": label,
                "num_turns": len(turns),
            },
        ))

    logger.info(
        f"Loaded {len(records)} TRACE records "
        f"({sum(1 for r in records if r.is_hacked)} hacked, "
        f"{sum(1 for r in records if not r.is_hacked)} benign)"
    )
    return records


def _parse_hh_rlhf_conversation(text: str) -> List[Dict[str, str]]:
    """Parse HH-RLHF conversation format (Human:/Assistant: prefixed)."""
    turns = []
    lines = text.strip().split("\n")
    current_role = None
    current_content = []

    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("Human:"):
            if current_role is not None:
                turns.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip(),
                })
            current_role = "user"
            current_content = [line_stripped[len("Human:"):].strip()]
        elif line_stripped.startswith("Assistant:"):
            if current_role is not None:
                turns.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip(),
                })
            current_role = "assistant"
            current_content = [line_stripped[len("Assistant:"):].strip()]
        else:
            if current_role is not None:
                current_content.append(line)

    if current_role is not None and current_content:
        turns.append({
            "role": current_role,
            "content": "\n".join(current_content).strip(),
        })

    return turns


def load_hh_rlhf_dataset(
    config: PipelineConfig,
    split: str = "harmless-base",
) -> List[ExploitRecord]:
    """Load HH-RLHF dataset from HuggingFace.

    In harmless-base: chosen = safe response, rejected = harmful exploit.
    In helpful-base: chosen = helpful response, rejected = less helpful.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install `datasets`: pip install datasets")

    # Map split names to HuggingFace config names
    hf_split_map = {
        "harmless-base": "harmless-base",
        "helpful-base": "helpful-base",
    }
    hf_config = hf_split_map.get(split, split)

    logger.info(f"Loading HH-RLHF dataset: {HH_RLHF_DATASET_ID} ({hf_config})")
    ds = load_dataset(HH_RLHF_DATASET_ID, data_dir=hf_config, split="train")

    records = []
    for i, row in enumerate(ds):
        if config.hh_rlhf_max_samples and i >= config.hh_rlhf_max_samples:
            break

        chosen_text = row.get("chosen", "")
        rejected_text = row.get("rejected", "")

        # Parse conversations
        chosen_turns = _parse_hh_rlhf_conversation(chosen_text)
        rejected_turns = _parse_hh_rlhf_conversation(rejected_text)

        # Extract last assistant responses
        chosen_response = _extract_last_assistant(chosen_turns)
        rejected_response = _extract_last_assistant(rejected_turns)
        context = _extract_context(rejected_turns)

        # For harmless-base: rejected = harmful exploit
        # For helpful-base: rejected = less helpful (still useful for preference data)
        is_hacked = (split == "harmless-base")

        records.append(ExploitRecord(
            source="hh_rlhf",
            exploit_category=split,
            is_hacked=is_hacked,
            conversation=rejected_turns,
            exploit_text=rejected_response,
            context_text=context,
            metadata={
                "index": i,
                "chosen_text": chosen_response,
                "rejected_text": rejected_response,
                "split": split,
            },
        ))

    logger.info(f"Loaded {len(records)} HH-RLHF ({split}) records")
    return records


def load_pku_safe_rlhf_dataset(config: PipelineConfig) -> List[ExploitRecord]:
    """Load PKU-SafeRLHF from HuggingFace.

    Each row has two responses with independent safety labels. We emit one
    ExploitRecord per row treating the unsafe response as exploit_text and
    the safer response stored in metadata for counterfactual generation.
    Rows where both responses are safe are kept as benign preference data.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install `datasets`: pip install datasets")

    logger.info(f"Loading PKU-SafeRLHF: {PKU_SAFE_RLHF_DATASET_ID}")
    ds = load_dataset(PKU_SAFE_RLHF_DATASET_ID, split="train")

    records = []
    for i, row in enumerate(ds):
        if config.pku_safe_max_samples and i >= config.pku_safe_max_samples:
            break

        prompt = row.get("prompt", "")
        r0 = row.get("response_0", "")
        r1 = row.get("response_1", "")
        safe0 = bool(row.get("is_response_0_safe", True))
        safe1 = bool(row.get("is_response_1_safe", True))

        # Determine exploit vs ideal
        if not safe0 and safe1:
            exploit_text, ideal_text, is_hacked = r0, r1, True
        elif not safe1 and safe0:
            exploit_text, ideal_text, is_hacked = r1, r0, True
        elif not safe0 and not safe1:
            # Both unsafe — use the less preferred as exploit
            better = row.get("better_response_id", 0)
            exploit_text = r0 if better == 1 else r1
            ideal_text = r1 if better == 1 else r0
            is_hacked = True
        else:
            # Both safe — benign preference pair
            better = row.get("better_response_id", 0)
            ideal_text = r0 if better == 0 else r1
            exploit_text = r1 if better == 0 else r0
            is_hacked = False

        # Extract harm category from category dict (any flagged key)
        raw_cats = row.get("category", {}) or {}
        if isinstance(raw_cats, dict):
            flagged = [k for k, v in raw_cats.items() if v]
            category = flagged[0] if flagged else "general"
        else:
            category = str(raw_cats) if raw_cats else "general"

        turns = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": exploit_text},
        ]
        records.append(ExploitRecord(
            source="pku_safe_rlhf",
            exploit_category=category,
            is_hacked=is_hacked,
            conversation=turns,
            exploit_text=exploit_text,
            context_text=prompt,
            metadata={
                "index": i,
                "ideal_text": ideal_text,
                "safe_0": safe0,
                "safe_1": safe1,
            },
        ))

    hacked = sum(1 for r in records if r.is_hacked)
    logger.info(f"Loaded {len(records)} PKU-SafeRLHF records ({hacked} hacked)")
    return records


def load_beaver_tails_dataset(config: PipelineConfig) -> List[ExploitRecord]:
    """Load BeaverTails from HuggingFace.

    Each row is a single (prompt, response) pair with a binary is_safe label
    and a dict of 14 harm-category flags. Unsafe responses → is_hacked=True.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install `datasets`: pip install datasets")

    logger.info(f"Loading BeaverTails: {BEAVER_TAILS_DATASET_ID}")
    ds = load_dataset(BEAVER_TAILS_DATASET_ID, split="30k_train")

    records = []
    for i, row in enumerate(ds):
        if config.beaver_tails_max_samples and i >= config.beaver_tails_max_samples:
            break

        prompt = row.get("prompt", "")
        response = row.get("response", "")
        is_safe = bool(row.get("is_safe", True))

        # Extract primary harm category
        raw_cats = row.get("category", {}) or {}
        if isinstance(raw_cats, dict):
            flagged = [k for k, v in raw_cats.items() if v]
            category = flagged[0] if flagged else "general"
        else:
            category = str(raw_cats) if raw_cats else "general"

        turns = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        records.append(ExploitRecord(
            source="beaver_tails",
            exploit_category=category,
            is_hacked=not is_safe,
            conversation=turns,
            exploit_text=response,
            context_text=prompt,
            metadata={"index": i, "is_safe": is_safe, "all_categories": raw_cats},
        ))

    hacked = sum(1 for r in records if r.is_hacked)
    logger.info(f"Loaded {len(records)} BeaverTails records ({hacked} hacked)")
    return records


def load_adv_bench_dataset(config: PipelineConfig) -> List[ExploitRecord]:
    """Load AdvBench from HuggingFace.

    AdvBench contains adversarial instruction prompts with target harmful
    completions. All records are marked is_hacked=True. No safe counterpart
    is available — useful as adversarial context sources for counterfactual
    generation.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install `datasets`: pip install datasets")

    logger.info(f"Loading AdvBench: {ADV_BENCH_DATASET_ID}")
    ds = load_dataset(ADV_BENCH_DATASET_ID, split="train")

    records = []
    for i, row in enumerate(ds):
        if config.adv_bench_max_samples and i >= config.adv_bench_max_samples:
            break

        goal = row.get("goal", "")
        target = row.get("target", "")

        turns = [
            {"role": "user", "content": goal},
            {"role": "assistant", "content": target},
        ]
        records.append(ExploitRecord(
            source="adv_bench",
            exploit_category="adversarial_instruction",
            is_hacked=True,
            conversation=turns,
            exploit_text=target,
            context_text=goal,
            metadata={"index": i},
        ))

    logger.info(f"Loaded {len(records)} AdvBench records (all hacked)")
    return records


def ingest_all(
    config: Optional[PipelineConfig] = None,
    splits: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> IngestResult:
    """Load all benchmarks and merge into a single IngestResult.

    Args:
        config: Pipeline configuration. Uses defaults if None.
        splits: HH-RLHF splits to load. Defaults to ["harmless-base"].
        sources: Which dataset sources to load. Defaults to all available.
            Options: "trace", "hh_rlhf", "pku_safe_rlhf", "beaver_tails", "adv_bench"
    """
    if config is None:
        config = PipelineConfig()
    if splits is None:
        splits = ["harmless-base"]
    if sources is None:
        sources = ["trace", "hh_rlhf", "pku_safe_rlhf", "beaver_tails", "adv_bench"]

    all_records: List[ExploitRecord] = []

    # TRACE
    trace_records: List[ExploitRecord] = []
    if "trace" in sources and config.trace_max_samples > 0:
        try:
            trace_records = load_trace_dataset(config)
            all_records.extend(trace_records)
        except Exception as e:
            logger.warning(f"Failed to load TRACE dataset: {e}")
    elif "trace" not in sources:
        logger.info("Skipping TRACE (not in sources)")
    else:
        logger.info("Skipping TRACE (trace_max_samples=0)")

    # HH-RLHF
    hh_records: List[ExploitRecord] = []
    if "hh_rlhf" in sources:
        for split in splits:
            split_records = load_hh_rlhf_dataset(config, split=split)
            hh_records.extend(split_records)
        all_records.extend(hh_records)

    # PKU-SafeRLHF
    pku_records: List[ExploitRecord] = []
    if "pku_safe_rlhf" in sources and config.pku_safe_max_samples > 0:
        try:
            pku_records = load_pku_safe_rlhf_dataset(config)
            all_records.extend(pku_records)
        except Exception as e:
            logger.warning(f"Failed to load PKU-SafeRLHF: {e}")

    # BeaverTails
    beaver_records: List[ExploitRecord] = []
    if "beaver_tails" in sources and config.beaver_tails_max_samples > 0:
        try:
            beaver_records = load_beaver_tails_dataset(config)
            all_records.extend(beaver_records)
        except Exception as e:
            logger.warning(f"Failed to load BeaverTails: {e}")

    # AdvBench
    adv_records: List[ExploitRecord] = []
    if "adv_bench" in sources and config.adv_bench_max_samples > 0:
        try:
            adv_records = load_adv_bench_dataset(config)
            all_records.extend(adv_records)
        except Exception as e:
            logger.warning(f"Failed to load AdvBench: {e}")

    # Compute category counts
    category_counts: Dict[str, int] = {}
    for r in all_records:
        category_counts[r.exploit_category] = category_counts.get(r.exploit_category, 0) + 1

    result = IngestResult(
        records=all_records,
        trace_count=len(trace_records),
        hh_rlhf_count=len(hh_records),
        pku_safe_count=len(pku_records),
        beaver_tails_count=len(beaver_records),
        adv_bench_count=len(adv_records),
        category_counts=category_counts,
    )

    logger.info(
        f"Ingested {result.total} total records: "
        f"{result.trace_count} TRACE + {result.hh_rlhf_count} HH-RLHF + "
        f"{result.pku_safe_count} PKU-SafeRLHF + {result.beaver_tails_count} BeaverTails + "
        f"{result.adv_bench_count} AdvBench"
    )
    logger.info(f"Categories: {category_counts}")

    return result
