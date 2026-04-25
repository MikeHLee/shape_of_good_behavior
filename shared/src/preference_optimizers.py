"""Preference optimization baselines: DPO, GRPO, ORPO, KTO.

All methods operate at the embedding level (not token-level) for fair comparison
on the same preference data without requiring an actual language model. The
mathematical formulations are preserved — just operating on embeddings instead
of token sequences.

"Policy" = network mapping (prompt_embed, response_embed) → scalar logit
"Reference policy" = frozen copy of initial network
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PreferenceSample:
    """A single preference sample for optimization."""
    prompt_embed: np.ndarray        # Context embedding
    chosen_embed: np.ndarray        # Preferred response embedding
    rejected_embed: np.ndarray      # Dispreferred response embedding
    is_desirable: Optional[bool] = None  # For KTO (unpaired binary)
    weight: float = 1.0             # Per-sample weight (for Hodge variants)


@dataclass
class OptimizerResult:
    """Result from training a preference optimizer."""
    method: str
    losses: List[float]
    exploit_resistance: float
    implicit_rewards_chosen: List[float]
    implicit_rewards_rejected: List[float]
    extra_metrics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Policy network (embedding-level)
# ---------------------------------------------------------------------------

class EmbeddingPolicy(nn.Module):
    """Policy network operating on embeddings.

    Maps (prompt_embed, response_embed) → scalar logit representing
    log-probability of the response given the prompt.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, prompt: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        """Return scalar logit for (prompt, response) pair.

        Args:
            prompt: (batch, embed_dim)
            response: (batch, embed_dim)
        Returns:
            logit: (batch, 1)
        """
        x = torch.cat([prompt, response], dim=-1)
        return self.net(x)

    def log_prob(
        self, prompt: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        """Log-probability proxy (tanh-squashed logit)."""
        return torch.tanh(self.forward(prompt, response))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PreferenceOptimizer(ABC):
    """Common interface for all preference optimization methods."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

        self.policy = EmbeddingPolicy(embed_dim, hidden_dim).to(self.device)
        self.ref_policy = deepcopy(self.policy)
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    @abstractmethod
    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute method-specific loss on a mini-batch."""
        ...

    def train(
        self, samples: List[PreferenceSample]
    ) -> OptimizerResult:
        """Train on preference samples."""
        losses = []
        n = len(samples)

        for epoch in range(self.epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch = self._make_batch([samples[i] for i in idx])
                batch["batch_indices"] = torch.tensor(idx, dtype=torch.long, device=self.device)

                loss, metrics = self.compute_loss(batch)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            losses.append(epoch_loss / max(n_batches, 1))

        # Evaluate
        resistance, chosen_rewards, rejected_rewards = self.evaluate_exploit_resistance(
            samples
        )

        return OptimizerResult(
            method=self.__class__.__name__,
            losses=losses,
            exploit_resistance=resistance,
            implicit_rewards_chosen=chosen_rewards,
            implicit_rewards_rejected=rejected_rewards,
        )

    def evaluate_exploit_resistance(
        self, samples: List[PreferenceSample]
    ) -> Tuple[float, List[float], List[float]]:
        """Fraction of samples where implicit reward(chosen) > implicit reward(rejected)."""
        correct = 0
        total = 0
        chosen_rewards = []
        rejected_rewards = []

        self.policy.eval()
        with torch.no_grad():
            for s in samples:
                prompt = torch.tensor(s.prompt_embed, dtype=torch.float32, device=self.device).unsqueeze(0)
                chosen = torch.tensor(s.chosen_embed, dtype=torch.float32, device=self.device).unsqueeze(0)
                rejected = torch.tensor(s.rejected_embed, dtype=torch.float32, device=self.device).unsqueeze(0)

                r_c = self.get_implicit_reward(prompt, chosen).item()
                r_r = self.get_implicit_reward(prompt, rejected).item()

                chosen_rewards.append(r_c)
                rejected_rewards.append(r_r)

                if r_c > r_r:
                    correct += 1
                total += 1

        self.policy.train()
        resistance = correct / max(total, 1)
        return resistance, chosen_rewards, rejected_rewards

    def get_implicit_reward(
        self, prompt: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        """Implicit reward = log π(y|x) - log π_ref(y|x)."""
        return self.policy.log_prob(prompt, response) - self.ref_policy.log_prob(
            prompt, response
        )

    def _make_batch(
        self, samples: List[PreferenceSample]
    ) -> Dict[str, torch.Tensor]:
        """Convert list of samples to tensor batch."""
        return {
            "prompt": torch.tensor(
                np.array([s.prompt_embed for s in samples]),
                dtype=torch.float32,
                device=self.device,
            ),
            "chosen": torch.tensor(
                np.array([s.chosen_embed for s in samples]),
                dtype=torch.float32,
                device=self.device,
            ),
            "rejected": torch.tensor(
                np.array([s.rejected_embed for s in samples]),
                dtype=torch.float32,
                device=self.device,
            ),
            "weights": torch.tensor(
                np.array([s.weight for s in samples]),
                dtype=torch.float32,
                device=self.device,
            ),
        }


# ---------------------------------------------------------------------------
# DPO: Direct Preference Optimization
# ---------------------------------------------------------------------------

class DPOTrainer(PreferenceOptimizer):
    """DPO: Direct Preference Optimization (Rafailov et al., 2023).

    Loss: -E[log σ(β(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
    """

    def __init__(self, beta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        prompt = batch["prompt"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        weights = batch["weights"]

        # Log-prob differences
        pi_chosen = self.policy.log_prob(prompt, chosen)
        pi_rejected = self.policy.log_prob(prompt, rejected)
        ref_chosen = self.ref_policy.log_prob(prompt, chosen)
        ref_rejected = self.ref_policy.log_prob(prompt, rejected)

        # DPO implicit reward difference
        logits = self.beta * (
            (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
        )

        # Weighted loss
        loss = -(weights.unsqueeze(-1) * F.logsigmoid(logits)).mean()

        with torch.no_grad():
            accuracy = (logits > 0).float().mean().item()

        return loss, {"accuracy": accuracy}


# ---------------------------------------------------------------------------
# GRPO: Group Relative Policy Optimization
# ---------------------------------------------------------------------------

class GRPOTrainer(PreferenceOptimizer):
    """GRPO: Group Relative Policy Optimization (Shao et al., 2024).

    Uses group-normalized advantages and PPO-style clipped loss.
    At embedding level: sample G responses per prompt, score with RM,
    normalize advantages within group, apply clipped surrogate.
    """

    def __init__(
        self,
        beta: float = 0.04,
        group_size: int = 8,
        clip_ratio: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta = beta
        self.group_size = group_size
        self.clip_ratio = clip_ratio

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        prompt = batch["prompt"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        weights = batch["weights"]

        # Simulate group scoring: chosen gets +1 advantage, rejected gets -1
        # Normalized within the "group" of (chosen, rejected)
        pi_chosen = self.policy.log_prob(prompt, chosen)
        pi_rejected = self.policy.log_prob(prompt, rejected)
        ref_chosen = self.ref_policy.log_prob(prompt, chosen)
        ref_rejected = self.ref_policy.log_prob(prompt, rejected)

        # Ratio for chosen
        ratio_c = torch.exp(pi_chosen - ref_chosen)
        ratio_r = torch.exp(pi_rejected - ref_rejected)

        # Group-normalized advantages: chosen = +1, rejected = -1 (simplified)
        adv_c = torch.ones_like(ratio_c)
        adv_r = -torch.ones_like(ratio_r)

        # Clipped surrogate for chosen
        clipped_c = torch.clamp(ratio_c, 1 - self.clip_ratio, 1 + self.clip_ratio)
        loss_c = -torch.min(ratio_c * adv_c, clipped_c * adv_c)

        # Clipped surrogate for rejected
        clipped_r = torch.clamp(ratio_r, 1 - self.clip_ratio, 1 + self.clip_ratio)
        loss_r = -torch.min(ratio_r * adv_r, clipped_r * adv_r)

        # KL penalty
        kl_chosen = (ref_chosen - pi_chosen).mean()
        kl_rejected = (ref_rejected - pi_rejected).mean()
        kl = (kl_chosen + kl_rejected) / 2

        loss = (weights.unsqueeze(-1) * (loss_c + loss_r)).mean() + self.beta * kl

        return loss, {"kl": kl.item()}


# ---------------------------------------------------------------------------
# ORPO: Odds Ratio Preference Optimization
# ---------------------------------------------------------------------------

class ORPOTrainer(PreferenceOptimizer):
    """ORPO: Odds Ratio Preference Optimization (Hong et al., 2024).

    Loss: L_SFT(y_w) + λ * log(odds(y_w|x) / odds(y_l|x))
    No reference model needed — single policy network.
    """

    def __init__(self, lambda_align: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.lambda_align = lambda_align

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        prompt = batch["prompt"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        weights = batch["weights"]

        # Log-probs (used as proxy for token-level NLL)
        log_p_chosen = self.policy.log_prob(prompt, chosen)
        log_p_rejected = self.policy.log_prob(prompt, rejected)

        # SFT loss on chosen (maximize log prob of chosen)
        sft_loss = -log_p_chosen.mean()

        # Odds ratio: odds = p / (1 - p), in log space
        # log_odds = log_p - log(1 - exp(log_p))
        # For tanh-squashed logits, use log_p directly as proxy
        odds_chosen = log_p_chosen - torch.log(1 - torch.exp(log_p_chosen).clamp(max=0.999) + 1e-10)
        odds_rejected = log_p_rejected - torch.log(1 - torch.exp(log_p_rejected).clamp(max=0.999) + 1e-10)

        # Log odds ratio
        log_odds_ratio = odds_chosen - odds_rejected

        # ORPO loss: SFT + alignment
        align_loss = -(weights.unsqueeze(-1) * F.logsigmoid(log_odds_ratio)).mean()
        loss = sft_loss + self.lambda_align * align_loss

        with torch.no_grad():
            accuracy = (log_odds_ratio > 0).float().mean().item()

        return loss, {"sft_loss": sft_loss.item(), "accuracy": accuracy}

    def get_implicit_reward(
        self, prompt: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        """ORPO has no reference model — implicit reward is just log odds."""
        return self.policy.log_prob(prompt, response)


# ---------------------------------------------------------------------------
# KTO: Kahneman-Tversky Optimization
# ---------------------------------------------------------------------------

class KTOTrainer(PreferenceOptimizer):
    """KTO: Kahneman-Tversky Optimization (Ethayarajh et al., 2024).

    Asymmetric loss on desirable/undesirable examples (unpaired binary).
    λ_bad > λ_good reflects loss aversion from prospect theory.
    """

    def __init__(
        self,
        beta: float = 0.1,
        lambda_good: float = 1.0,
        lambda_bad: float = 1.33,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta = beta
        self.lambda_good = lambda_good
        self.lambda_bad = lambda_bad
        self._r_ref = 0.0  # Running reference baseline

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        prompt = batch["prompt"]
        chosen = batch["chosen"]    # desirable
        rejected = batch["rejected"]  # undesirable
        weights = batch["weights"]

        # Implicit rewards
        r_good = self.beta * (
            self.policy.log_prob(prompt, chosen)
            - self.ref_policy.log_prob(prompt, chosen)
        )
        r_bad = self.beta * (
            self.policy.log_prob(prompt, rejected)
            - self.ref_policy.log_prob(prompt, rejected)
        )

        # Reference baseline (running average)
        with torch.no_grad():
            r_ref = torch.tensor(self._r_ref, device=self.device)
            # Update running average
            all_r = torch.cat([r_good, r_bad], dim=0)
            self._r_ref = 0.9 * self._r_ref + 0.1 * all_r.mean().item()

        # KTO losses (asymmetric)
        loss_good = self.lambda_good * (1 - torch.sigmoid(r_good - r_ref))
        loss_bad = self.lambda_bad * (1 - torch.sigmoid(r_ref - r_bad))

        loss = (weights.unsqueeze(-1) * (loss_good + loss_bad)).mean()

        with torch.no_grad():
            accuracy = (
                ((r_good > r_ref).float().mean() + (r_bad < r_ref).float().mean()) / 2
            ).item()

        return loss, {"accuracy": accuracy, "r_ref": self._r_ref}


# ---------------------------------------------------------------------------
# Utility: convert CounterfactualPairs + MappingResult to PreferenceSamples
# ---------------------------------------------------------------------------

def mapping_to_preference_samples(
    mapping,
    weights: Optional[np.ndarray] = None,
) -> List[PreferenceSample]:
    """Convert a MappingResult to PreferenceSamples for optimizer training.

    Args:
        mapping: MappingResult with embedding_pairs and reduced embeddings
        weights: Optional per-sample weights from Hodge diagnostic

    Returns:
        List of PreferenceSample
    """
    samples = []
    for i, ep in enumerate(mapping.embedding_pairs):
        w = weights[i] if weights is not None and i < len(weights) else 1.0
        samples.append(PreferenceSample(
            prompt_embed=ep.context_embed,
            chosen_embed=ep.ideal_embed,
            rejected_embed=ep.exploit_embed,
            weight=float(w),
        ))
    return samples
