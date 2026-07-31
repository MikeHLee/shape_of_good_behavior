"""Cheap, CPU-only check: what fraction of TRACE context_text records exceed
the truncation budgets used at PPO-training time (sft_max_seq_length=512)
and at eval time (hardcoded 256)? Informs whether the eval truncation bug
found in sgb006_diagnose_7b_eval.py also corrupted PPO training data itself
(would require retraining), or is mostly confined to eval (fix + re-eval
suffices). No GPU needed -- tokenizer + dataset only.
"""
import sys
from pathlib import Path

import modal

_project_root = Path(__file__).parent.parent
_ai_research_root = _project_root.parent.parent
if str(_ai_research_root) not in sys.path:
    sys.path.insert(0, str(_ai_research_root))

_req_path = _project_root / "shared" / "requirements-finetune.txt"
try:
    _requirements = [
        line.strip() for line in _req_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
except FileNotFoundError:
    _requirements = []

image = modal.Image.debian_slim(python_version="3.11").pip_install(*_requirements).env({"PYTHONUNBUFFERED": "1"})
for _local_dir, _remote_dir in [
    (_project_root / "shared", "/app/shared"),
    (_project_root / "feedback_geometry" / "src", "/app/feedback_geometry/src"),
    (_project_root / "src", "/app/src"),
]:
    if _local_dir.is_dir():
        image = image.add_local_dir(str(_local_dir), remote_path=_remote_dir)

app = modal.App("sgb006-check-context-lengths")
_SECRETS = [modal.Secret.from_name("huggingface-token")]


@app.function(image=image, cpu=2, timeout=600, secrets=_SECRETS)
def check() -> dict:
    sys.path.insert(0, "/app")
    from transformers import AutoTokenizer
    from shared.src.config import PipelineConfig
    from shared.src.lm_finetuning import SYSTEM_PROMPT
    from shared.src.data_ingest import ingest_all

    config = PipelineConfig()
    config.trace_max_samples = 517
    config.hh_rlhf_max_samples = 0
    config.cache_dir = "/app/shared/data/cache"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    result = ingest_all(config, sources=["trace"])
    hacked = result.filter_hacked()

    lengths = []
    for record in hacked:
        if not record.context_text:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record.context_text.strip()},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        n_tok = len(tokenizer(prompt)["input_ids"])
        lengths.append(n_tok)

    import statistics
    n = len(lengths)
    over_256 = sum(1 for l in lengths if l > 256)
    over_512 = sum(1 for l in lengths if l > 512)

    return {
        "n_records": n,
        "mean_tokens": statistics.mean(lengths),
        "median_tokens": statistics.median(lengths),
        "max_tokens": max(lengths),
        "min_tokens": min(lengths),
        "pct_over_256": round(100 * over_256 / n, 1),
        "pct_over_512": round(100 * over_512 / n, 1),
        "n_over_256": over_256,
        "n_over_512": over_512,
    }


@app.local_entrypoint()
def main():
    result = check.remote()
    import json
    print(json.dumps(result, indent=2))
