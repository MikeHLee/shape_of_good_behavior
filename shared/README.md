# Shared Pipeline

Cross-track infrastructure for the Shape of Good Behavior research series. Provides data ingestion, counterfactual generation, Hodge analysis, reward-hacking evaluation, visualization, and Modal dispatch used by all three tracks.

## Module Map

| File | Purpose |
|------|---------|
| `src/config.py` | Shared config (paths, seeds, hyperparams) |
| `src/data_ingest.py` | Load HH-RLHF and TRACE preference datasets |
| `src/counterfactual_gen.py` | Claude-based counterfactual generation (incremental, resumable) |
| `src/preference_mapper.py` | Build preference graphs from pairwise comparisons |
| `src/hodge_analysis.py` | Full Hodge decomposition (gradient / curl / harmonic) |
| `src/hodge_diagnostic.py` | `HodgeDiagnosticCritic`: selective cycle filtering (genuine vs exploitable) |
| `src/preference_optimizers.py` | DPO, GRPO, ORPO, KTO at embedding level |
| `src/hodge_preference_optimizers.py` | Hodge-DPO, Hodge-GRPO, Hodge-KTO variants |
| `src/optimizer_comparison.py` | 30-seed benchmark runner (Welch t + Cohen's d) |
| `src/lm_finetuning.py` | PPO / Hodge-PPO fine-tuning loop for LM checkpoints on Modal |
| `src/peer_sheaf.py` | Peer-consistency sheaf construction and restriction map fitting |
| `src/peer_hodge.py` | Cocycle violation ‖δ¹c‖ computation and AUC reporting |
| `src/reward_hacking_eval.py` | Exploit resistance evaluation |
| `src/visualize.py` | Plotting helpers |
| `src/run_pipeline.py` | End-to-end pipeline runner |
| `modal_finetune.py` | Modal entrypoint: SFT → reward model → PPO (Stages 1–3) |
| `modal_peer_sheaf.py` | Modal entrypoint: peer-sheaf experiments on GPU |
| `modal_runner.py` | Modal entrypoint: optimizer comparison experiments |

## Key Results

### Optimizer Comparison (30 seeds, 2026-07)

File: `results/optimizer_comparison_hodge_v3_30seed.json`

| Method | Exploit Resistance | Notes |
|--------|--------------------|-------|
| Hodge-DPO | 0.9999 ± 0.001 | +6.3% vs DPO, d=6.52, p<0.0001 |
| Hodge-KTO | 0.9964 ± 0.003 | +24.5% vs KTO, d=16.47, p<0.0001 |
| Hodge-GRPO | 1.000 ± 0.000 | = GRPO (ceiling) |
| GRPO | 1.000 ± 0.000 | |
| DPO | 0.940 ± 0.013 | |
| KTO | 0.800 ± 0.017 | |
| ORPO | 0.622 ± 0.080 | |

### PPO Fine-tuning — SGB-003 (2026-07-23, A100-40GB)

Both runs verified `status=success` with crash-safe manifest tracking. Manifests in `_runs/` at the repo root.

| Run | mean_reward_final | mean_reward_all | Steps |
|-----|------------------|-----------------|-------|
| Standard PPO (`hodge=False`) | 3.6897 | 3.3920 | 64 |
| Hodge-PPO (`hodge=True`) | -1.2170 | -0.8646 | 64 |

Hodge-PPO negative reward: training on exploit-suppression drives mean reward down on exploit queries — this is the intended signal, not a failure.

Three bugs fixed to get here (all in `src/lm_finetuning.py` and `modal_finetune.py`):
1. `do_sample=False` — greedy decoding eliminates `torch.multinomial` NaN crash from bf16 + top-p near-empty distribution
2. `log_ratio.clamp(-5, 5)` before `.exp()` — bounds ratio to [e⁻⁵, e⁵], prevents bf16 overflow at step ~45
3. `fn.spawn().get()` dispatch — decouples remote job lifetime from local gRPC stream; prevents `InputCancellation` on local process exit

### Peer Sheaf — SGB-012 (2026-06, 7–9B panel)

File: `results/peer_sheaf_modal_summary.json`

| Config | AUC (‖δ¹c‖ vs label) | p-value | Notes |
|--------|-----------------------|---------|-------|
| convincing-game (persuasion) | 0.661 | 2.5e-6 | length-matched |
| insider-trading (agentic) | 0.625 | 8e-8 | length-matched |
| alignment-faking reasoning | 0.391 | n.s. | sign-flipped vs hypothesis |
| instructed-deception | ~0.54 | n.s. | overt / cue-robust, not strategic |

Signal is **selective**: ‖δ¹c‖ detects strategy-as-deception (persuasion, insider trading) but not overt instructed lies. Gated by model scale — effect absent at 0.3–1B, present at 7–9B.

## Modal Cloud Setup

Requirements loaded dynamically from `requirements-modal.txt` (not inline `.pip_install()`). LLM counterfactual generation is done locally first (needs `anthropic` key); Modal runners use `--skip-llm` with pre-cached counterfactuals.

```bash
# Run optimizer comparison on Modal
modal run shared/modal_runner.py

# Run PPO fine-tuning (both variants)
modal run shared/modal_finetune.py -- --stage ppo --hodge false
modal run shared/modal_finetune.py -- --stage ppo --hodge true
```

Experiment manifests are written to the `reward-hacking-results` Modal volume under `/results/_manifests/` and mirrored locally to `_runs/` via `mirror_from_volume()`.
