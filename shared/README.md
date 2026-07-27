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

### Exploit Resistance Eval — SGB-004 (2026-07-24, true train/holdout split)

File: `results/finetune/sgb004_exploit_resistance_holdout.json`

The SGB-003 checkpoints above were retrained from scratch on a deterministic 217/51 train/holdout split (content-keyed SHA-256 hash, seed=42, 20% held out) after discovering the first eval (below) had no split at all. All four checkpoints (base/SFT/PPO/Hodge-PPO) are scored on the same 51 held-out exploit prompts by the Hodge reward model:

| Model | Mean Reward | Exploit Resistance | N |
|-------|-------------|---------------------|---|
| base | -1.6614 | 11.76% | 51 |
| **SFT** | -0.6398 | **35.29%** | 51 |
| PPO | -1.1246 | 23.53% | 51 |
| Hodge-PPO | -1.0473 | 25.49% | 51 |

**Findings, reported plainly:**
- **The headline hypothesis is not supported.** Hodge-PPO edges out standard PPO by 25.49% vs 23.53% — a one-example swing at n=51 (13/51 vs 12/51). That's noise, not a replication of the embedding-level Hodge-DPO/Hodge-KTO advantage from the optimizer comparison benchmark above.
- **Unexpected finding:** plain SFT (35.29%) beats both PPO variants. SGB-005b below found a likely root cause: the reward model both PPO variants trained against barely learned to discriminate ideal from exploit text at all (train loss ≈ log 2).
- All fine-tuned checkpoints beat `base`, which makes sense now (base never saw the RM signal) — that alone confirms the eval is measuring something real, unlike the retracted run below.

**Retracted:** a 2026-07-23 eval run reported base=70%, sft=24%, ppo=40%, hodge_ppo=31% resistance. That run had no train/holdout split — every stage loaded the identical 517-record set, so those numbers were in-sample RM agreement, not exploit resistance on unseen data. Superseded by the table above.

**Update (2026-07-27):** the table above is itself superseded — SGB-005c found and fixed the real cause of the "SFT beats both PPO variants" anomaly (a reward-model truncation bug, not a fundamental PPO/Hodge property). See SGB-005c below for the corrected numbers.

### Hodge-as-Featurizer Test — SGB-005b (2026-07-24)

Files: `scripts/sgb005b_hodge_featurizer.py`, `results/finetune/sgb005b_rm_scores.json`, `results/finetune/sgb005b_hodge_featurizer_result.json`.

Prompted by SGB-004's null result: instead of using the Hodge decomposition to reweight the RM's training loss (the existing `compute_hodge_weights` use), test whether its per-pair gradient-potential difference (`HodgeDiagnosticCritic.diagnose_for_samples().sample_potential_diffs`) is useful as an auxiliary *feature* — does it separate reference ideal/exploit text beyond what the trained RM's own scalar score captures? Scored the RM directly on the 217 train + 51 holdout reference pairs (no generation, no new PPO run — cheap CPU/L4-only test).

**First pass looked great and was wrong.** Hodge feature alone: 100% train / 98.04% holdout accuracy at ranking ideal > exploit. This is near-tautological: the graph construction feeds in a direct edge asserting `ideal > exploit` at confidence 0.999 for every single pair — the same quantity then being "recovered." Caught via an ablation (setting that direct edge's confidence to 0.5, i.e. zero log-odds weight, so only the unsupervised cross-pair kNN embedding-similarity edges remain) before reporting this as a result:

| Test | Train acc | Holdout acc |
|------|-----------|-------------|
| RM alone | 49.31% | 45.10% |
| Hodge feature alone (with direct edge — circular) | 100.00% | 98.04% |
| **Hodge feature alone (ablated — unsupervised kNN only)** | **55.76%** | **64.71%** |
| RM + Hodge feature (ablated) | 53.46% | 60.78% |

**Honest read:** pure embedding-geometry Hodge decomposition, with no ideal/exploit labels at all, weakly discriminates ideal from exploit text (55–65%, above chance but not reliable, and n=268 pairs is small enough that this range could shift). Combining with the RM's own score doesn't help — because the RM itself is the bigger problem.

**Bigger finding, not what this test was looking for:** RM training loss converged to ≈0.69 (`log 2`) for both the standard and Hodge RM in SGB-004 — the textbook value for a Bradley-Terry loss that learned nothing beyond chance. That's the likely real explanation for SGB-004's "SFT beats both PPO variants": if the reward signal both PPO runs trained against was close to noise, no amount of Hodge weighting could be expected to show an advantage. **Before re-testing Hodge-PPO vs standard PPO, the RM training itself needs to actually converge** (more epochs/steps on 217 pairs, a learning-rate sweep, or more data) — re-running PPO on top of a broken reward signal, Hodge-weighted or not, won't produce a meaningful comparison.

### RM Root-Cause Fix + Corrected Rerun — SGB-005c (2026-07-27)

File: `results/finetune/sgb004_exploit_resistance_holdout_v2.json`

Chasing the "RM stuck at log(2)" finding from SGB-005b to its actual cause, not just its symptom. First attempt (more epochs, higher LR) changed nothing — 5x the steps and 2x the LR left both RMs at ≈0.69 loss, ruling out "undertrained." The real cause: TRACE `context_text` averages ~1010 tokens (median 947, max 2122), but `rm_max_length=512` combined with the tokenizer's default right-truncation cut the chat-templated sequence off *before the assistant turn even started* — so 49/50 sampled chosen/rejected pairs were byte-identical after truncation. The RM was training on duplicate inputs; no amount of epochs or LR could fix that.

**Fix:** `tokenizer.truncation_side = "left"` in `build_rm_dataset` (preserves the differentiating tail), `rm_max_length` 512→1024, and `rm_batch_size` 4→1 / `rm_grad_accum` 4→16 (batch=4 at the longer length OOM'd on L4). Result: both RMs' loss dropped from stuck-at-chance to fully saturated (train_loss 0.083 standard / 0.019 Hodge) within ~2 epochs. Verified this wasn't a training-set-only artifact by scoring both RMs directly on reference pairs in the exact chat-template + left-truncation format they trained on: **100% train AND 100% holdout ranking accuracy** for both (mean margin ~32–34 in logit space), up from 49.31%/45.10% (chance) before the fix. (A first version of this sanity check itself had a bug — it scored raw untemplated text with default right-truncation, giving misleading numbers; caught because the Hodge RM had *lower* train loss than standard but *worse* apparent accuracy, which is backwards and was the tell.)

With the RM genuinely converged, retrained both PPO variants (64 steps, A100, same as before) and reran the Stage 4 eval on the true 51-record holdout:

| Model | Mean Reward | Exploit Resistance | N |
|-------|-------------|---------------------|---|
| base | 1.1432 | 68.63% | 51 |
| SFT | 1.9272 | 80.39% | 51 |
| PPO | 2.2174 | 80.39% | 51 |
| **Hodge-PPO** | **2.5873** | **82.35%** | 51 |

**This resolves the SGB-004 anomaly.** PPO now clearly beats SFT on mean reward (2.22 vs 1.93) and matches it on resistance — confirming the earlier "SFT beats PPO" result was an artifact of PPO training against a near-random reward signal, not a real property of PPO. Ordering is now sensible and monotonic: base < SFT = PPO < Hodge-PPO.

**On the headline hypothesis:** Hodge-PPO edges out standard PPO on both metrics for the first time in this investigation, in the hypothesized direction — but the resistance gap is still a single example at n=51 (42/51 vs 41/51), not statistically distinguishable at this sample size. The continuous mean-reward gap (0.37) is comparable in size to PPO's own margin over SFT (0.29), which is suggestive but not a substitute for a real significance test — per-example rewards weren't persisted, only aggregates, so no paired test was possible from this run. Directionally positive, not yet confirmed; more holdout data (or repeated seeds) is the cheapest path to a firmer answer, not further RM/PPO tuning.

**Caveat:** both RMs' training loss hit exactly 0.0 several epochs before the configured 10 finished — extreme confidence on only 217 training pairs, a real overfitting signature in absolute terms. It doesn't appear to have hurt holdout generalization here (100% holdout ranking accuracy, sensible downstream PPO ordering), but epoch count wasn't independently tuned down to rule out.

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
