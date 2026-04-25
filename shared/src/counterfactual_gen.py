"""LLM counterfactual generation for exploit records.

For TRACE records: calls Claude API to generate ideal aligned counterfactuals.
For HH-RLHF records: the 'chosen' response IS the ideal — no LLM call needed.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import PipelineConfig
from .data_ingest import ExploitRecord

logger = logging.getLogger(__name__)

COUNTERFACTUAL_PROMPT = """\
You are an AI alignment researcher analyzing a case where an AI system produced a harmful, \
exploitative, or misaligned response.

## Conversation / Context
{context}

## Problematic Final Response
{exploit}

## Exploit Category Code
{category}

## Constitutional Principles
{principles}

## Your Task
Reason carefully about:
1. What specific alignment failure occurred (reward hacking, sycophancy, deception, etc.)
2. Which constitutional principles were violated and why
3. What the ideal aligned response would look like — one that is maximally helpful while \
   remaining honest and avoiding harm
4. How the ideal response differs structurally from the exploit (not just surface refusal)

After reasoning, return a JSON object with exactly these fields:
{{
  "failure_analysis": "Detailed explanation of what alignment failure occurred and why",
  "exploit_type": "One of: reward_hacking | sycophancy | deception | harmful_compliance | other",
  "principles_violated": ["list of principles from the Constitutional Principles section"],
  "ideal_response": "The full response the AI should have given — helpful, honest, harmless",
  "structural_difference": "Key structural way ideal differs from exploit (not just refusal)",
  "confidence": 0.95
}}

Return ONLY valid JSON, no markdown fences or extra text."""


@dataclass
class CounterfactualPair:
    """A paired (exploit, ideal) response with analysis metadata."""

    exploit_text: str
    ideal_text: str
    context_text: str
    failure_analysis: str = ""
    principles_violated: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""  # "trace_llm", "hh_rlhf_direct"
    exploit_category: str = ""
    exploit_type: str = ""           # reward_hacking | sycophancy | deception | ...
    structural_difference: str = ""  # how ideal structurally differs from exploit
    thinking_summary: str = ""       # first 500 chars of extended thinking (for audit)
    metadata: Dict = field(default_factory=dict)


def _cache_key(record: ExploitRecord) -> str:
    """Compute a stable cache key for a record."""
    content = f"{record.source}:{record.exploit_category}:{record.exploit_text[:200]}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class CounterfactualGenerator:
    """Generate ideal counterfactual responses for exploit records."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cache_path = Path(config.cache_dir) / "counterfactuals.json"
        self._cache: Dict[str, dict] = self._load_cache()

    def _load_cache(self) -> Dict[str, dict]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Cache file corrupted, starting fresh")
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def from_hh_rlhf_pair(self, record: ExploitRecord) -> CounterfactualPair:
        """For HH-RLHF records, the chosen response is the ideal — no LLM call needed."""
        chosen_text = record.metadata.get("chosen_text", "")
        if not chosen_text:
            chosen_text = "[No chosen response available]"

        return CounterfactualPair(
            exploit_text=record.exploit_text,
            ideal_text=chosen_text,
            context_text=record.context_text,
            failure_analysis="HH-RLHF rejected response (harmful/unhelpful)",
            principles_violated=[],
            confidence=1.0,
            source="hh_rlhf_direct",
            exploit_category=record.exploit_category,
            metadata={"record_source": record.source},
        )

    def generate_for_trace(self, record: ExploitRecord) -> CounterfactualPair:
        """Generate counterfactual for a TRACE record via Claude API."""
        key = _cache_key(record)

        # Check cache
        if key in self._cache:
            cached = self._cache[key]
            return CounterfactualPair(
                exploit_text=record.exploit_text,
                ideal_text=cached.get("ideal_response", ""),
                context_text=record.context_text,
                failure_analysis=cached.get("failure_analysis", ""),
                principles_violated=cached.get("principles_violated", []),
                confidence=cached.get("confidence", 0.5),
                exploit_type=cached.get("exploit_type", ""),
                structural_difference=cached.get("structural_difference", ""),
                thinking_summary=cached.get("thinking_summary", ""),
                source="trace_llm",
                exploit_category=record.exploit_category,
                metadata={"cached": True},
            )

        # Call Claude API with extended thinking
        try:
            import anthropic
            from dotenv import load_dotenv
            from pathlib import Path as _Path
            load_dotenv(_Path(__file__).resolve().parent.parent.parent / ".env")
        except ImportError:
            raise ImportError("Install `anthropic` and `python-dotenv`")

        client = anthropic.Anthropic()
        principles_str = "\n".join(
            f"- {p}" for p in self.config.constitutional_principles
        )

        # Provide full conversation context for multi-turn TRACE trajectories
        # (TRACE turns can be 20+ so give generous context window)
        context_text = record.context_text[:4000]
        exploit_text = record.exploit_text[:2000]

        prompt = COUNTERFACTUAL_PROMPT.format(
            context=context_text,
            exploit=exploit_text,
            category=record.exploit_category,
            principles=principles_str,
        )

        # Retry with exponential backoff on 529 overload errors
        _retry_delays = [30, 60, 120]
        last_exc: Optional[Exception] = None
        response = None
        for attempt, delay in enumerate([0] + _retry_delays):
            if delay:
                logger.info(f"  529 overload — waiting {delay}s before retry {attempt}/{len(_retry_delays)}...")
                time.sleep(delay)
            try:
                response = client.messages.create(
                    model=self.config.llm_model,
                    max_tokens=self.config.llm_max_tokens,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": self.config.llm_thinking_budget,
                    },
                    messages=[{"role": "user", "content": prompt}],
                )
                break  # success
            except Exception as exc:
                is_overload = (
                    hasattr(exc, "status_code") and exc.status_code == 529
                ) or "overloaded" in str(exc).lower()
                if is_overload and attempt < len(_retry_delays):
                    last_exc = exc
                    continue
                raise  # non-529 or exhausted retries

        try:
            assert response is not None

            # Extract thinking and text blocks separately
            thinking_text = ""
            response_text = ""
            for block in response.content:
                if block.type == "thinking":
                    thinking_text = block.thinking
                elif block.type == "text":
                    response_text = block.text.strip()

            # Strip any markdown fences from the JSON text block
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )

            parsed = json.loads(response_text)

            # Cache result including thinking summary for auditability
            cached_entry = {
                **parsed,
                "thinking_summary": thinking_text[:500],
            }
            self._cache[key] = cached_entry
            self._save_cache()

            return CounterfactualPair(
                exploit_text=record.exploit_text,
                ideal_text=parsed.get("ideal_response", ""),
                context_text=record.context_text,
                failure_analysis=parsed.get("failure_analysis", ""),
                principles_violated=parsed.get("principles_violated", []),
                confidence=parsed.get("confidence", 0.5),
                exploit_type=parsed.get("exploit_type", ""),
                structural_difference=parsed.get("structural_difference", ""),
                thinking_summary=thinking_text[:500],
                source="trace_llm",
                exploit_category=record.exploit_category,
                metadata={"cached": False, "thinking_tokens": len(thinking_text.split())},
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response for record {key}: {e}")
            return CounterfactualPair(
                exploit_text=record.exploit_text,
                ideal_text="I cannot assist with that request.",
                context_text=record.context_text,
                failure_analysis=f"Parse error: {e}",
                principles_violated=[],
                confidence=0.1,
                source="trace_llm",
                exploit_category=record.exploit_category,
                metadata={"error": str(e)},
            )
        except Exception as e:
            logger.error(f"LLM API error for record {key}: {e}")
            return CounterfactualPair(
                exploit_text=record.exploit_text,
                ideal_text="I cannot assist with that request.",
                context_text=record.context_text,
                failure_analysis=f"API error: {e}",
                principles_violated=[],
                confidence=0.1,
                source="trace_llm",
                exploit_category=record.exploit_category,
                metadata={"error": str(e)},
            )

    def generate_batch(
        self,
        records: List[ExploitRecord],
        skip_llm: bool = False,
    ) -> List[CounterfactualPair]:
        """Generate counterfactuals for a batch of records.

        Args:
            records: List of ExploitRecord to process.
            skip_llm: If True, only use cached results and HH-RLHF direct mapping.
        """
        pairs: List[CounterfactualPair] = []
        llm_needed = 0
        llm_cached = 0

        for i, record in enumerate(records):
            if record.source == "hh_rlhf":
                pairs.append(self.from_hh_rlhf_pair(record))
            elif record.source == "trace":
                if skip_llm and _cache_key(record) not in self._cache:
                    # Skip uncached TRACE records when --skip-llm
                    continue
                pair = self.generate_for_trace(record)
                pairs.append(pair)
                if pair.metadata.get("cached"):
                    llm_cached += 1
                else:
                    llm_needed += 1

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(records)} records")

        logger.info(
            f"Generated {len(pairs)} counterfactual pairs "
            f"({llm_needed} new LLM calls, {llm_cached} cached)"
        )
        return pairs

    def save_pairs(self, pairs: List[CounterfactualPair], path: Optional[str] = None):
        """Save generated pairs to JSON."""
        if path is None:
            path = str(Path(self.config.results_dir) / "counterfactual_pairs.json")
        with open(path, "w") as f:
            json.dump([asdict(p) for p in pairs], f, indent=2)
        logger.info(f"Saved {len(pairs)} pairs to {path}")

    def load_pairs(self, path: Optional[str] = None) -> List[CounterfactualPair]:
        """Load previously generated pairs from JSON."""
        if path is None:
            path = str(Path(self.config.results_dir) / "counterfactual_pairs.json")
        with open(path) as f:
            data = json.load(f)
        return [CounterfactualPair(**d) for d in data]
