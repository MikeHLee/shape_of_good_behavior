"""Hodge-aware preference optimization variants: Hodge-DPO, Hodge-GRPO, Hodge-KTO.

Each variant uses two Hodge integration mechanisms:
1. Per-sample weights from node-level cycle participation (diagnose_for_samples)
2. Hodge potential-alignment regularization: penalizes reward differences that
   deviate from the globally-consistent Hodge gradient potential.

Note: The original batch harmonic penalty (computing Hodge decomposition on
in-batch rewards) is mathematically always zero for scalar reward models,
because scalar predictions are always gradient-consistent. Instead, we use the
*precomputed* Hodge potential from the global preference graph as a
regularization target.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .hodge_diagnostic import CycleDiagnosis, HodgeDiagnosticCritic
from .preference_optimizers import (
    DPOTrainer,
    GRPOTrainer,
    KTOTrainer,
    OptimizerResult,
    PreferenceSample,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hodge-DPO
# ---------------------------------------------------------------------------

class HodgeDPOTrainer(DPOTrainer):
    """DPO with Hodge-aware preference weighting + potential-alignment loss.

    1. Per-sample weights from node-level cycle participation
    2. Hodge potential-alignment: regularize model reward diffs toward
       globally-consistent Hodge potential diffs
    """

    def __init__(
        self,
        diagnosis: Optional[CycleDiagnosis] = None,
        hodge_lambda: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diagnosis = diagnosis
        self.hodge_lambda = hodge_lambda
        self._potential_diffs: Optional[torch.Tensor] = None

    def train(
        self, samples: List[PreferenceSample]
    ) -> OptimizerResult:
        if self.diagnosis is not None:
            weights = self.diagnosis.per_sample_weights
            if weights is None:
                weights = self.diagnosis.per_preference_weights
            for i, s in enumerate(samples):
                if i < len(weights):
                    s.weight = float(weights[i])

            # Store potential diffs as tensor for loss computation
            if self.diagnosis.sample_potential_diffs is not None:
                pd = self.diagnosis.sample_potential_diffs[:len(samples)]
                self._potential_diffs = torch.tensor(
                    pd, dtype=torch.float32, device=self.device,
                ).unsqueeze(-1)

        result = super().train(samples)
        result.method = "Hodge-DPO"
        if self.diagnosis is not None:
            result.extra_metrics["exploit_fraction"] = self.diagnosis.exploit_fraction
            result.extra_metrics["genuine_h1"] = self.diagnosis.genuine_h1
            if self.diagnosis.per_sample_weights is not None:
                result.extra_metrics["weight_std"] = float(
                    self.diagnosis.per_sample_weights.std()
                )
        return result

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss, metrics = super().compute_loss(batch)

        # Hodge potential-alignment regularization
        if (
            self.hodge_lambda > 0
            and self._potential_diffs is not None
            and "batch_indices" in batch
        ):
            idx = batch["batch_indices"]
            target_diffs = self._potential_diffs[idx]  # (B, 1)

            r_c = self.get_implicit_reward(batch["prompt"], batch["chosen"])
            r_r = self.get_implicit_reward(batch["prompt"], batch["rejected"])
            model_diffs = r_c - r_r  # (B, 1)

            # MSE between model reward diffs and Hodge-consistent diffs
            hodge_loss = F.mse_loss(model_diffs, target_diffs)
            loss = loss + self.hodge_lambda * hodge_loss
            metrics["hodge_potential_loss"] = hodge_loss.item()

        return loss, metrics


# ---------------------------------------------------------------------------
# Hodge-GRPO
# ---------------------------------------------------------------------------

class HodgeGRPOTrainer(GRPOTrainer):
    """GRPO with Hodge potential-based advantages + alignment loss.

    1. Per-sample weights from node-level cycle participation
    2. Hodge potential-alignment regularization
    """

    def __init__(
        self,
        diagnosis: Optional[CycleDiagnosis] = None,
        hodge_lambda: float = 0.5,
        hodge_potential: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diagnosis = diagnosis
        self.hodge_lambda = hodge_lambda
        self._potential = hodge_potential
        self._potential_diffs: Optional[torch.Tensor] = None

    def train(
        self, samples: List[PreferenceSample]
    ) -> OptimizerResult:
        if self.diagnosis is not None:
            weights = self.diagnosis.per_sample_weights
            if weights is None:
                weights = self.diagnosis.per_preference_weights
            for i, s in enumerate(samples):
                if i < len(weights):
                    s.weight = float(weights[i])

            if self.diagnosis.sample_potential_diffs is not None:
                pd = self.diagnosis.sample_potential_diffs[:len(samples)]
                self._potential_diffs = torch.tensor(
                    pd, dtype=torch.float32, device=self.device,
                ).unsqueeze(-1)

        result = super().train(samples)
        result.method = "Hodge-GRPO"
        if self.diagnosis is not None:
            result.extra_metrics["exploit_fraction"] = self.diagnosis.exploit_fraction
        return result

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss, metrics = super().compute_loss(batch)

        if (
            self.hodge_lambda > 0
            and self._potential_diffs is not None
            and "batch_indices" in batch
        ):
            idx = batch["batch_indices"]
            target_diffs = self._potential_diffs[idx]

            r_c = self.get_implicit_reward(
                batch["prompt"], batch["chosen"]
            )
            r_r = self.get_implicit_reward(
                batch["prompt"], batch["rejected"]
            )
            model_diffs = r_c - r_r

            hodge_loss = F.mse_loss(model_diffs, target_diffs)
            loss = loss + self.hodge_lambda * hodge_loss
            metrics["hodge_potential_loss"] = hodge_loss.item()

        return loss, metrics


# ---------------------------------------------------------------------------
# Hodge-KTO
# ---------------------------------------------------------------------------

class HodgeKTOTrainer(KTOTrainer):
    """KTO with Hodge-aware reference baseline + alignment loss.

    1. Per-sample weights from node-level cycle participation
    2. Hodge potential-alignment regularization
    3. Uses Hodge potential as reference baseline (optional)
    """

    def __init__(
        self,
        diagnosis: Optional[CycleDiagnosis] = None,
        hodge_potential_baseline: Optional[float] = None,
        hodge_lambda: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diagnosis = diagnosis
        self._hodge_baseline = hodge_potential_baseline
        self.hodge_lambda = hodge_lambda
        self._potential_diffs: Optional[torch.Tensor] = None

    def train(
        self, samples: List[PreferenceSample]
    ) -> OptimizerResult:
        if self.diagnosis is not None:
            weights = self.diagnosis.per_sample_weights
            if weights is None:
                weights = self.diagnosis.per_preference_weights
            for i, s in enumerate(samples):
                if i < len(weights):
                    s.weight = float(weights[i])

            if self.diagnosis.sample_potential_diffs is not None:
                pd = self.diagnosis.sample_potential_diffs[:len(samples)]
                self._potential_diffs = torch.tensor(
                    pd, dtype=torch.float32, device=self.device,
                ).unsqueeze(-1)

        if self._hodge_baseline is not None:
            self._r_ref = self._hodge_baseline

        result = super().train(samples)
        result.method = "Hodge-KTO"
        if self.diagnosis is not None:
            result.extra_metrics["exploit_fraction"] = self.diagnosis.exploit_fraction
        return result

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss, metrics = super().compute_loss(batch)

        if (
            self.hodge_lambda > 0
            and self._potential_diffs is not None
            and "batch_indices" in batch
        ):
            idx = batch["batch_indices"]
            target_diffs = self._potential_diffs[idx]

            r_c = self.get_implicit_reward(batch["prompt"], batch["chosen"])
            r_r = self.get_implicit_reward(batch["prompt"], batch["rejected"])
            model_diffs = r_c - r_r

            hodge_loss = F.mse_loss(model_diffs, target_diffs)
            loss = loss + self.hodge_lambda * hodge_loss
            metrics["hodge_potential_loss"] = hodge_loss.item()

        return loss, metrics


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ALL_HODGE_OPTIMIZERS = {
    "Hodge-DPO": HodgeDPOTrainer,
    "Hodge-GRPO": HodgeGRPOTrainer,
    "Hodge-KTO": HodgeKTOTrainer,
}


def create_hodge_optimizers(
    embed_dim: int,
    diagnosis: CycleDiagnosis,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    epochs: int = 100,
    hodge_lambda: float = 0.5,
    hodge_potential: Optional[np.ndarray] = None,
    hodge_potential_baseline: Optional[float] = None,
) -> Dict[str, "PreferenceOptimizer"]:
    """Create all Hodge-aware optimizer instances."""
    common = dict(embed_dim=embed_dim, hidden_dim=hidden_dim, lr=lr, epochs=epochs)

    return {
        "Hodge-DPO": HodgeDPOTrainer(
            diagnosis=diagnosis, hodge_lambda=hodge_lambda,
            beta=0.1, **common,
        ),
        "Hodge-GRPO": HodgeGRPOTrainer(
            diagnosis=diagnosis, hodge_lambda=hodge_lambda,
            hodge_potential=hodge_potential,
            beta=0.04, group_size=8, **common,
        ),
        "Hodge-KTO": HodgeKTOTrainer(
            diagnosis=diagnosis, hodge_lambda=hodge_lambda,
            hodge_potential_baseline=hodge_potential_baseline,
            beta=0.1, **common,
        ),
    }
