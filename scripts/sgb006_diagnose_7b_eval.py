"""SGB-006 diagnostic: inspect raw 7B generations behind the eval reversal.

evaluate_7b reported exploit resistance getting WORSE with more training
(base 22% > SFT 20% > Hodge-PPO 14% > PPO 10%) -- the exact opposite
ordering of the 1.5B result (SGB-005c: base 68.63% < SFT/PPO 80.39% <
Hodge-PPO 82.35%). Before trusting this as a real finding, print the actual
generated text for a couple of holdout prompts per checkpoint so a human can
see whether these are coherent responses or degenerate/garbage output (e.g.
from the meta-device-offload warning seen only on adapter-loaded checkpoints,
or from the 256-token prompt truncation cutting context_text before the
actual question).

Usage: modal run scripts/sgb006_diagnose_7b_eval.py --n 2
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
    # This module gets re-imported inside the remote container too (Modal
    # mounts the entrypoint script at a different path there, e.g. /root/),
    # where __file__-relative paths above resolve differently and this read
    # fails -- harmless when it does, since the image is already built by
    # the time the container re-imports this file.
    _requirements = [
        line.strip() for line in _req_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
except FileNotFoundError:
    _requirements = []

image = modal.Image.debian_slim(python_version="3.11").pip_install(*_requirements).env({"PYTHONUNBUFFERED": "1"})
# Same re-import caveat as _requirements above: only mount dirs that actually
# exist under this resolution of _project_root (true on the real local run;
# false and skipped on the container's re-import, where mounts are moot).
for _local_dir, _remote_dir in [
    (_project_root / "shared", "/app/shared"),
    (_project_root / "feedback_geometry" / "src", "/app/feedback_geometry/src"),
    (_project_root / "src", "/app/src"),
]:
    if _local_dir.is_dir():
        image = image.add_local_dir(str(_local_dir), remote_path=_remote_dir)

app = modal.App("sgb006-diagnose-7b-eval")
ckpt_vol = modal.Volume.from_name("reward-hacking-checkpoints", create_if_missing=False)
_SECRETS = [modal.Secret.from_name("huggingface-token")]


@app.function(image=image, gpu="A100-40GB", timeout=1800,
               volumes={"/checkpoints": ckpt_vol}, secrets=_SECRETS)
def diagnose(n: int = 2) -> dict:
    import os
    sys.path.insert(0, "/app")
    os.environ["HF_HOME"] = "/tmp/hf_cache"

    import torch
    from shared.src.config import PipelineConfig
    from shared.src.lm_finetuning import (
        FineTuneConfig, SYSTEM_PROMPT, load_policy_model, load_reward_model,
    )

    config = PipelineConfig()
    config.trace_max_samples = 517
    config.hh_rlhf_max_samples = 0
    config.cache_dir = "/app/shared/data/cache"

    ft_config = FineTuneConfig()
    ft_config.model_name = "Qwen/Qwen2.5-7B-Instruct"
    ft_config.checkpoint_dir = "/checkpoints/7b"

    # Inlined from modal_finetune.py's _load_pairs/_is_holdout (not imported --
    # that module's top-level Image/App-building code assumes local dev machine
    # paths and isn't safe to import from inside an already-running container).
    _HOLDOUT_FRAC = 0.2
    _HOLDOUT_SEED = 42

    def _is_holdout(record) -> bool:
        import hashlib
        from shared.src.counterfactual_gen import _cache_key
        digest = hashlib.sha256(f"{_HOLDOUT_SEED}:{_cache_key(record)}".encode()).hexdigest()
        frac = int(digest[:8], 16) / 0xFFFFFFFF
        return frac < _HOLDOUT_FRAC

    from shared.src.data_ingest import ingest_all
    result = ingest_all(config, sources=["trace"])
    hacked_records = [r for r in result.filter_hacked() if _is_holdout(r)]
    eval_records = hacked_records[:n]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rm_model, rm_tokenizer = load_reward_model(ft_config, checkpoint="/checkpoints/7b/rm")
    rm_model.eval()

    checkpoints = {
        "base": None,
        "sft": "/checkpoints/7b/sft",
        "ppo": "/checkpoints/7b/ppo",
        "hodge_ppo": "/checkpoints/7b/ppo_hodge",
    }

    gen_kwargs = dict(max_new_tokens=ft_config.ppo_max_new_tokens, do_sample=False, pad_token_id=None)

    results = {}
    for name, ckpt in checkpoints.items():
        print(f"\n{'='*20} {name} {'='*20}", flush=True)
        model, tokenizer = load_policy_model(ft_config, checkpoint=ckpt)
        model.eval()
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

        entries = []
        for i, record in enumerate(eval_records):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record.context_text.strip()},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            enc_untrunc = tokenizer(prompt, return_tensors="pt")
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            n_tok_full = enc_untrunc["input_ids"].shape[1]
            n_tok_trunc = enc["input_ids"].shape[1]
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            full_text = tokenizer.decode(out[0], skip_special_tokens=True)
            gen_only = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

            rm_enc = rm_tokenizer(full_text, truncation=True, max_length=ft_config.rm_max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                score = rm_model(**rm_enc).logits.squeeze(-1).item()

            print(f"\n--- [{name}] record {i} ---", flush=True)
            print(f"prompt tokens: {n_tok_trunc}/{n_tok_full} (truncated={n_tok_trunc < n_tok_full})", flush=True)
            print(f"GENERATED: {gen_only!r}", flush=True)
            print(f"RM score: {score:.4f}", flush=True)

            entries.append({
                "prompt_tokens_full": n_tok_full,
                "prompt_tokens_truncated": n_tok_trunc,
                "truncated": n_tok_trunc < n_tok_full,
                "generated_text": gen_only,
                "rm_score": score,
            })
        results[name] = entries
        del model

    return results


@app.local_entrypoint()
def main(n: int = 2):
    result = diagnose.remote(n=n)
    import json
    out_path = Path(__file__).parent.parent / "shared" / "results" / "finetune" / "sgb006_7b_eval_diagnosis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
