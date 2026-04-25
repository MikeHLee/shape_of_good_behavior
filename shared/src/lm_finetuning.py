"""Token-level fine-tuning: SFT → Hodge-weighted RM → PPO / Hodge-PPO.

Pipeline for Qwen2.5-1.5B-Instruct on TRACE counterfactuals:
  1. SFT  — warm-start on ideal responses from the counterfactuals cache
  2. RM   — train reward model on (exploit, ideal) pairs; optionally Hodge-weighted
  3. PPO  — custom PPO-Clip loop against the RM (no TRL PPOTrainer dependency)

Hodge weighting: HodgeDiagnosticCritic assigns per-sample weights in [0, 1] based on
cycle participation — down-weights exploitable cycles, preserves genuine value tensions.

Requires: trl>=1.0.0, peft>=0.10.0, transformers>=4.40.0
PPO is implemented from scratch (TRL 1.x removed PPOTrainer in favour of online trainers).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorWithPadding
from trl import RewardConfig, RewardTrainer, SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = "You are a helpful, honest, and harmless AI assistant."


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FineTuneConfig:
    model_name: str = DEFAULT_MODEL

    # LoRA — keeps VRAM well within L4's 22 GB at 1.5 B params
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # SFT
    sft_epochs: int = 2
    sft_lr: float = 2e-4
    sft_batch_size: int = 4
    sft_grad_accum: int = 4          # effective batch = 16
    sft_max_seq_length: int = 512

    # Reward model
    rm_epochs: int = 2
    rm_lr: float = 1e-4
    rm_batch_size: int = 4
    rm_grad_accum: int = 4
    rm_max_length: int = 512

    # PPO (custom loop — PPO-Clip without value network)
    ppo_steps: int = 256             # gradient update steps
    ppo_lr: float = 1.4e-5
    ppo_batch_size: int = 8          # queries per step
    ppo_ppo_epochs: int = 4          # inner PPO epochs per batch
    ppo_epsilon: float = 0.2         # clip ratio
    ppo_kl_coeff: float = 0.05       # KL penalty coefficient
    ppo_max_new_tokens: int = 256

    # Checkpointing
    checkpoint_dir: str = "/checkpoints"


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_sft_dataset(pairs, tokenizer, config: FineTuneConfig) -> Dataset:
    """Format (context, ideal_response) pairs for SFT via the model's chat template."""
    texts = []
    for pair in pairs:
        if not pair.ideal_text or not pair.context_text:
            continue
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": pair.context_text.strip()},
            {"role": "assistant", "content": pair.ideal_text.strip()},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append({"text": text})
        except Exception as e:
            logger.warning(f"Skipping SFT pair (chat template error): {e}")

    logger.info(f"SFT dataset: {len(texts)} examples")
    return Dataset.from_list(texts)


def build_rm_dataset(
    pairs,
    tokenizer,
    config: FineTuneConfig,
    hodge_weights: Optional[np.ndarray] = None,
) -> Dataset:
    """Tokenize (chosen=ideal, rejected=exploit) pairs for RewardTrainer.

    Stores hodge_weight as a float column; HodgeRewardDataCollator batches it.
    """
    if hodge_weights is not None and len(hodge_weights) != len(pairs):
        raise ValueError(f"hodge_weights length {len(hodge_weights)} != pairs {len(pairs)}")

    rows = []
    for i, pair in enumerate(pairs):
        if not pair.ideal_text or not pair.exploit_text or not pair.context_text:
            continue

        def _fmt(response: str) -> str:
            msgs = [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": pair.context_text.strip()},
                {"role": "assistant", "content": response.strip()},
            ]
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )

        try:
            text_chosen   = _fmt(pair.ideal_text)
            text_rejected = _fmt(pair.exploit_text)
        except Exception as e:
            logger.warning(f"Skipping RM pair {i}: {e}")
            continue

        tok_c = tokenizer(text_chosen,   truncation=True, max_length=config.rm_max_length, padding=False)
        tok_r = tokenizer(text_rejected, truncation=True, max_length=config.rm_max_length, padding=False)

        rows.append({
            "input_ids_chosen":        tok_c["input_ids"],
            "attention_mask_chosen":   tok_c["attention_mask"],
            "input_ids_rejected":      tok_r["input_ids"],
            "attention_mask_rejected": tok_r["attention_mask"],
            "hodge_weight": float(hodge_weights[i]) if hodge_weights is not None else 1.0,
        })

    logger.info(f"RM dataset: {len(rows)} examples")
    return Dataset.from_list(rows)


def build_ppo_dataset(records, tokenizer, config: FineTuneConfig) -> Dataset:
    """Format TRACE exploit contexts as PPO query prompts (no response prefix)."""
    rows = []
    for record in records:
        if not record.context_text:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": record.context_text.strip()},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"{SYSTEM_PROMPT}\n\nUser: {record.context_text.strip()}\nAssistant:"

        enc = tokenizer(
            prompt, truncation=True, max_length=config.sft_max_seq_length,
            padding=False, return_tensors="pt",
        )
        rows.append({
            "input_ids": enc["input_ids"].squeeze(0),
            "query":     prompt,
        })

    logger.info(f"PPO dataset: {len(rows)} queries")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Hodge weights
# ---------------------------------------------------------------------------

def compute_hodge_weights(pairs, pipeline_config, ft_config: FineTuneConfig) -> np.ndarray:
    """Embed (exploit, ideal) pairs and compute per-sample Hodge cycle weights."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from .hodge_diagnostic import HodgeDiagnosticCritic
    from .preference_mapper import EmbeddingPair

    logger.info("Computing Hodge weights...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    exploit_embs = embedder.encode([p.exploit_text for p in pairs], normalize_embeddings=True, show_progress_bar=False)
    ideal_embs   = embedder.encode([p.ideal_text   for p in pairs], normalize_embeddings=True, show_progress_bar=False)

    n = len(pairs)
    # Direct edges: ideal > exploit for each pair
    preference_edges = [(2*i, 2*i+1, 1.0) for i in range(n)]

    # Cross-pair k-NN edges (critical for non-trivial Hodge H1)
    k = min(5, n - 1)
    all_embs = np.vstack([e for pair in zip(exploit_embs, ideal_embs) for e in pair])
    sim = cosine_similarity(all_embs)
    for a in range(len(all_embs)):
        s = sim[a].copy(); s[a] = -1.0
        for b in np.argsort(s)[-k:]:
            preference_edges.append((int(a), int(b), float(s[b])))

    fake_pairs = [
        EmbeddingPair(
            exploit_embedding=exploit_embs[i],
            ideal_embedding=ideal_embs[i],
            exploit_text=pairs[i].exploit_text,
            ideal_text=pairs[i].ideal_text,
            context_text=pairs[i].context_text,
            category=getattr(pairs[i], "exploit_category", "default"),
        )
        for i in range(n)
    ]

    critic   = HodgeDiagnosticCritic(pipeline_config)
    diagnosis = critic.diagnose_for_samples(
        preference_edges=preference_edges,
        n_items=2 * n,
        embedding_pairs=fake_pairs,
    )

    weights = diagnosis.per_sample_weights
    if weights is None:
        logger.warning("No per_sample_weights returned; using uniform.")
        weights = np.ones(n, dtype=np.float32)
    else:
        weights = np.array(weights, dtype=np.float32)

    logger.info(
        f"Hodge weights — min={weights.min():.3f}  mean={weights.mean():.3f}  "
        f"max={weights.max():.3f}  exploit_fraction={diagnosis.exploit_fraction:.2%}"
    )
    return weights


# ---------------------------------------------------------------------------
# Hodge-aware Reward Trainer
# ---------------------------------------------------------------------------

class _RewardCollator:
    """Pads chosen/rejected token sequences and stacks them into a batch dict."""

    def __init__(self, tokenizer, max_length: int):
        self.pad_id     = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.max_length = max_length

    def _pad(self, seqs):
        max_len = min(max(len(s) for s in seqs), self.max_length)
        input_ids = torch.full((len(seqs), max_len), self.pad_id, dtype=torch.long)
        attn_mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            s = s[:max_len]
            input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn_mask[i, :len(s)] = 1
        return input_ids, attn_mask

    def __call__(self, features):
        input_ids_c, attn_c = self._pad([f["input_ids_chosen"]   for f in features])
        input_ids_r, attn_r = self._pad([f["input_ids_rejected"]  for f in features])
        return {
            "input_ids_chosen":        input_ids_c,
            "attention_mask_chosen":   attn_c,
            "input_ids_rejected":      input_ids_r,
            "attention_mask_rejected": attn_r,
        }


class HodgeRewardDataCollator(_RewardCollator):
    """Extends _RewardCollator to also batch the scalar hodge_weight column."""

    def __call__(self, features):
        weights = [f.get("hodge_weight", 1.0) for f in features]
        batch   = super().__call__(features)
        batch["hodge_weight"] = torch.tensor(weights, dtype=torch.float)
        return batch


class HodgeRewardTrainer(RewardTrainer):
    """Bradley-Terry loss scaled by per-sample Hodge cycle weights."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        hodge_weights = inputs.pop("hodge_weight", None)

        rewards_chosen   = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
        ).logits.squeeze(-1)
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
        ).logits.squeeze(-1)

        margin = rewards_chosen - rewards_rejected

        if hodge_weights is not None:
            w    = hodge_weights.to(rewards_chosen.device)
            w    = w / w.mean().clamp(min=1e-8)
            loss = -(w * F.logsigmoid(margin)).mean()
        else:
            loss = -F.logsigmoid(margin).mean()

        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_policy_model(config: FineTuneConfig, checkpoint: Optional[str] = None):
    """Load Qwen2.5-1.5B-Instruct with LoRA for SFT / PPO policy."""
    model_path = checkpoint or config.model_name
    tokenizer  = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    if checkpoint is None:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules, bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def load_reward_model(config: FineTuneConfig, checkpoint: Optional[str] = None):
    """Load Qwen2.5-1.5B-Instruct with a scalar reward head + LoRA."""
    model_path = checkpoint or config.model_name
    tokenizer  = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )

    if checkpoint is None:
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_r, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules, bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Stage 1 — SFT
# ---------------------------------------------------------------------------

def run_sft(pairs, config: FineTuneConfig, output_dir: str) -> str:
    model, tokenizer = load_policy_model(config)
    dataset = build_sft_dataset(pairs, tokenizer, config)

    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=config.sft_batch_size,
        gradient_accumulation_steps=config.sft_grad_accum,
        learning_rate=config.sft_lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=config.sft_max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"SFT saved → {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Stage 2 — Reward model
# ---------------------------------------------------------------------------

def run_reward_model_training(
    pairs,
    config: FineTuneConfig,
    output_dir: str,
    pipeline_config=None,
    use_hodge: bool = False,
) -> str:
    model, tokenizer = load_reward_model(config)

    hodge_weights = None
    if use_hodge and pipeline_config is not None:
        hodge_weights = compute_hodge_weights(pairs, pipeline_config, config)

    dataset = build_rm_dataset(pairs, tokenizer, config, hodge_weights=hodge_weights)

    rm_cfg = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=config.rm_epochs,
        per_device_train_batch_size=config.rm_batch_size,
        gradient_accumulation_steps=config.rm_grad_accum,
        learning_rate=config.rm_lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=config.rm_max_length,
        report_to="none",
        remove_unused_columns=False,
    )

    TrainerCls  = HodgeRewardTrainer if use_hodge else RewardTrainer
    CollatorCls = HodgeRewardDataCollator if use_hodge else _RewardCollator

    trainer = TrainerCls(
        model=model,
        args=rm_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=CollatorCls(tokenizer, max_length=config.rm_max_length),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"RM saved → {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Stage 3 — PPO (custom PPO-Clip loop, no TRL PPOTrainer)
# ---------------------------------------------------------------------------

def _compute_log_probs(model, input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Sum log-probs of response tokens under the model."""
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # (B, T, V)
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_ids  = input_ids[:, 1:]
    token_lp   = log_probs.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)
    return (token_lp * response_mask[:, 1:]).sum(-1)       # (B,)


def run_ppo(
    records,
    sft_checkpoint: str,
    rm_checkpoint: str,
    config: FineTuneConfig,
    output_dir: str,
) -> Dict:
    """Custom PPO-Clip training loop against the reward model.

    Uses PPO-Clip without a value network — advantage = normalised reward.
    KL penalty keeps the policy close to the SFT reference.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Policy (trainable) and frozen SFT reference
    policy,     tokenizer = load_policy_model(config, checkpoint=sft_checkpoint)
    ref_policy, _         = load_policy_model(config, checkpoint=sft_checkpoint)
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    ref_policy.eval()

    # Frozen reward model
    rm_model, rm_tokenizer = load_reward_model(config, checkpoint=rm_checkpoint)
    for p in rm_model.parameters():
        p.requires_grad_(False)
    rm_model.eval()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=config.ppo_lr,
    )

    dataset = build_ppo_dataset(records, tokenizer, config)
    # Simple list-based dataloader — queries are variable length tensors
    queries  = [dataset[i]["input_ids"] for i in range(len(dataset))]

    gen_kwargs = dict(
        max_new_tokens=config.ppo_max_new_tokens,
        do_sample=True, temperature=0.7, top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    all_rewards, steps_done = [], 0

    while steps_done < config.ppo_steps:
        # Sample a batch of queries
        idx      = torch.randperm(len(queries))[:config.ppo_batch_size].tolist()
        q_batch  = [queries[i].to(device) for i in idx]

        # ── Generate responses ──────────────────────────────────────────────
        policy.eval()
        with torch.no_grad():
            responses = []
            for q in q_batch:
                out = policy.generate(q.unsqueeze(0), **gen_kwargs)
                responses.append(out[0])                   # full sequence

        # ── Score with reward model ─────────────────────────────────────────
        rewards = []
        for resp in responses:
            text = tokenizer.decode(resp, skip_special_tokens=True)
            enc  = rm_tokenizer(
                text, truncation=True, max_length=config.rm_max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                score = rm_model(**enc).logits.squeeze(-1).item()
            rewards.append(score)
        all_rewards.extend(rewards)

        rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
        # Normalize advantages
        adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

        # ── Pad sequences for batch ops ─────────────────────────────────────
        max_len = max(r.size(0) for r in responses)
        pad_id  = tokenizer.pad_token_id or tokenizer.eos_token_id
        padded  = torch.full((len(responses), max_len), pad_id, dtype=torch.long, device=device)
        for i, r in enumerate(responses):
            padded[i, :r.size(0)] = r

        # Response mask: tokens generated after the query
        q_lens   = [q.size(0) for q in q_batch]
        resp_mask = torch.zeros_like(padded, dtype=torch.float)
        for i, ql in enumerate(q_lens):
            resp_mask[i, ql:responses[i].size(0)] = 1.0

        # ── Compute old log-probs (reference) ───────────────────────────────
        with torch.no_grad():
            old_lp  = _compute_log_probs(ref_policy, padded, resp_mask)

        # ── PPO-Clip inner loop ─────────────────────────────────────────────
        policy.train()
        for _ in range(config.ppo_ppo_epochs):
            new_lp = _compute_log_probs(policy, padded, resp_mask)
            ratio  = (new_lp - old_lp).exp()
            clipped = ratio.clamp(1 - config.ppo_epsilon, 1 + config.ppo_epsilon)
            ppo_loss = -torch.min(ratio * adv, clipped * adv).mean()

            # KL penalty: keep policy close to SFT reference
            kl_loss  = (new_lp - old_lp).mean()
            loss     = ppo_loss + config.ppo_kl_coeff * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        steps_done += 1
        if steps_done % 20 == 0:
            logger.info(
                f"  step={steps_done}  mean_reward={np.mean(all_rewards[-20:]):.4f}  "
                f"ppo_loss={ppo_loss.item():.4f}"
            )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"PPO saved → {output_dir}")

    return {
        "mean_reward_final": float(np.mean(all_rewards[-50:]) if all_rewards else 0.0),
        "mean_reward_all":   float(np.mean(all_rewards) if all_rewards else 0.0),
        "steps": steps_done,
    }


# ---------------------------------------------------------------------------
# Stage 4 — Evaluation
# ---------------------------------------------------------------------------

def evaluate_exploit_resistance(
    records,
    checkpoints: Dict[str, Optional[str]],
    rm_checkpoint: str,
    config: FineTuneConfig,
    n_eval: int = 50,
) -> Dict[str, Dict]:
    """Generate from each checkpoint, score with RM, report exploit resistance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rm_model, rm_tokenizer = load_reward_model(config, checkpoint=rm_checkpoint)
    for p in rm_model.parameters():
        p.requires_grad_(False)
    rm_model.eval()

    eval_records = records[:n_eval]
    results      = {}

    gen_kwargs = dict(
        max_new_tokens=config.ppo_max_new_tokens,
        do_sample=True, temperature=0.7, top_p=0.9,
        pad_token_id=None,                           # set per model below
    )

    for name, ckpt in checkpoints.items():
        logger.info(f"Evaluating {name}...")
        model, tokenizer = load_policy_model(config, checkpoint=ckpt)
        model.eval()
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

        rewards = []
        for record in eval_records:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": record.context_text.strip()},
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = f"{SYSTEM_PROMPT}\n\nUser: {record.context_text.strip()}\nAssistant:"

            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            text = tokenizer.decode(out[0], skip_special_tokens=True)

            rm_enc = rm_tokenizer(
                text, truncation=True, max_length=config.rm_max_length, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                score = rm_model(**rm_enc).logits.squeeze(-1).item()
            rewards.append(score)

        mean_r  = float(np.mean(rewards))
        resist  = float(np.mean([r > 0 for r in rewards]))
        results[name] = {"mean_reward": mean_r, "exploit_resistance": resist, "n": len(rewards)}
        logger.info(f"  {name}: mean_reward={mean_r:.4f}  resist={resist:.2%}")
        del model

    return results
