"""SGB-044 eval cross-check: re-score the saved eval responses with BOTH RMs.

The eval scores every checkpoint with rm_hodge, which Hodge-PPO was trained
against. This re-scores the identical saved responses with rm and rm_hodge
(same format_rm_input + left truncation), no generation, read-only on
checkpoints; writes /results/finetune/eval_rescore_sgb044.json.
"""
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/Users/Michaellee/Documents/Runes/ai_research/topics/shape_of_good_behavior")

from shared.modal_finetune import app, image, ckpt_vol, results_vol, _SECRETS, _setup_paths, _hf_cache_dir, _load_pairs


@app.function(image=image, gpu="L4", timeout=3600, memory=32768,
              volumes={"/checkpoints": ckpt_vol, "/results": results_vol}, secrets=_SECRETS)
def rescore() -> dict:
    _setup_paths()
    _hf_cache_dir()
    ckpt_vol.reload(); results_vol.reload()
    import json, torch
    from shared.src.config import PipelineConfig
    from shared.src.lm_finetuning import FineTuneConfig, load_reward_model, format_rm_input

    config = PipelineConfig()
    config.trace_max_samples = 517
    config.hh_rlhf_max_samples = 0
    config.cache_dir = "/app/shared/data/cache"
    ft = FineTuneConfig(); ft.checkpoint_dir = "/checkpoints"
    _, recs = _load_pairs(config, split="holdout")

    ev = json.load(open("/results/finetune/eval_comparison_sgb044.json"))
    out = {}
    for rm_name in ["rm", "rm_hodge"]:
        rm, tok = load_reward_model(ft, checkpoint=f"/checkpoints/{rm_name}")
        rm.eval(); tok.truncation_side = "left"
        dev = next(rm.parameters()).device
        out[rm_name] = {}
        for ck, v in ev.items():
            scores = []
            for ex in v["per_example"]:
                text = format_rm_input(tok, recs[ex["idx"]].context_text, ex["response"])
                enc = tok(text, truncation=True, max_length=ft.rm_max_length, return_tensors="pt").to(dev)
                with torch.no_grad():
                    scores.append(rm(**enc).logits.squeeze(-1).item())
            out[rm_name][ck] = scores
            print(rm_name, ck, sum(s > 0 for s in scores), "/", len(scores), flush=True)
        del rm; torch.cuda.empty_cache()
    with open("/results/finetune/eval_rescore_sgb044.json", "w") as f:
        json.dump(out, f)
    results_vol.commit()
    return out


@app.local_entrypoint()
def run():
    import json
    r = rescore.spawn().get()
    with open("/private/tmp/claude-501/-Users-Michaellee-Documents-Runes-ai-research/6369dcb2-e936-474a-a5bf-aab86986c7d1/scratchpad/eval_rescore_sgb044.json", "w") as f:
        json.dump(r, f)
    print("saved")
