"""
Loss functions for H-Net training.

This module provides:
- Cross-entropy loss for language modeling
- Load balancing loss for dynamic chunking
- Combined loss computation
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from hnet.modules.dc import RoutingModuleOutput


class LanguageModelingLoss:
    """
    Cross-entropy loss for language modeling.
    """

    def __init__(self, ignore_index: int = -100):
        """
        Initialize language modeling loss.

        Args:
            ignore_index: Index to ignore in loss computation (e.g., padding)
        """
        self.ignore_index = ignore_index

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model predictions (B, L, vocab_size)
            targets: Target token ids (B, L)

        Returns:
            Cross-entropy loss
        """
        # Reshape logits and targets for cross-entropy
        # logits: (B, L, vocab_size) -> (B*L, vocab_size)
        # targets: (B, L) -> (B*L,)
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        loss = F.cross_entropy(
            logits_flat, targets_flat, ignore_index=self.ignore_index, reduction="mean"
        )

        return loss


class LoadBalancingLoss:
    """
    Load balancing loss for hierarchical routing.

    Encourages balanced usage of dynamic chunking across the sequence.
    """

    def __init__(self, downsampling_factor: float = 2.0):
        """
        Initialize load balancing loss.

        Args:
            downsampling_factor: Expected downsampling factor (N)
        """
        self.downsampling_factor = downsampling_factor

    def __call__(
        self,
        router_output: RoutingModuleOutput,
    ) -> torch.Tensor:
        """
        Compute load balancing loss for a single routing module.

        Args:
            router_output: Output from routing module

        Returns:
            Load balancing loss
        """
        # Import here to avoid circular dependency
        from hnet.utils.train import load_balancing_loss as lb_loss

        return lb_loss(router_output, self.downsampling_factor)


class HierarchicalLoadBalancingLoss:
    """
    Load balancing loss for hierarchical models with multiple routing stages.
    """

    def __init__(
        self,
        downsampling_factors: Optional[List[float]] = None,
        stage_weights: Optional[List[float]] = None,
    ):
        """
        Initialize hierarchical load balancing loss.

        Args:
            downsampling_factors: Expected downsampling per stage
            stage_weights: Weight for each stage's loss
        """
        self.downsampling_factors = downsampling_factors
        self.stage_weights = stage_weights

    def __call__(
        self,
        bpred_outputs: List[RoutingModuleOutput],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute load balancing loss across all stages.

        Args:
            bpred_outputs: List of routing module outputs from all stages

        Returns:
            Dictionary with per-stage losses and total loss
        """
        from hnet.utils.train import load_balancing_loss as lb_loss

        if len(bpred_outputs) == 0:
            return {"total": torch.tensor(0.0), "per_stage": {}}

        # Use the first output to place tensors on the right device
        ref = bpred_outputs[0]
        if hasattr(ref, "boundary_prob") and ref.boundary_prob is not None:
            ref_device = ref.boundary_prob.device
        else:
            ref_device = ref.boundary_mask.device

        stage_losses = {}
        total_loss = torch.zeros((), device=ref_device)

        for stage_idx, router_output in enumerate(bpred_outputs):
            # Determine downsampling factor for this stage
            if self.downsampling_factors is not None and stage_idx < len(self.downsampling_factors):
                N = self.downsampling_factors[stage_idx]
            else:
                # No default: paper explicitly uses [6.0] for 1-stage or [3.0, 3.0] for 2-stage
                raise ValueError(
                    f"downsampling_factors must be provided for stage {stage_idx}. "
                    f"Paper uses [6.0] for 1-stage models ('6-DC') or [3.0, 3.0] for 2-stage models ('(3,3)-DC'). "
                    f"See Table 1 in the paper."
                )

            # Compute loss for this stage
            stage_loss = lb_loss(router_output, N)

            # Apply stage weight
            if self.stage_weights is not None and stage_idx < len(self.stage_weights):
                weight = self.stage_weights[stage_idx]
            else:
                weight = 1.0

            weighted_loss = weight * stage_loss
            stage_losses[f"stage_{stage_idx}"] = stage_loss
            total_loss = total_loss + weighted_loss

        return {"total": total_loss, "per_stage": stage_losses}


class CombinedLoss:
    """
    Combined loss for H-Net training.

    Combines cross-entropy loss with load balancing losses.
    """

    def __init__(
        self,
        lm_loss: LanguageModelingLoss,
        lb_loss: HierarchicalLoadBalancingLoss,
        lb_weight: float = 0.03,
    ):
        """
        Initialize combined loss.

        Args:
            lm_loss: Language modeling loss
            lb_loss: Load balancing loss
            lb_weight: Weight for load balancing loss
        """
        self.lm_loss = lm_loss
        self.lb_loss = lb_loss
        self.lb_weight = lb_weight

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        bpred_outputs: List[RoutingModuleOutput],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Model predictions
            targets: Target token ids
            bpred_outputs: Routing module outputs

        Returns:
            Dictionary with all loss components and total loss
        """
        # Compute cross-entropy loss
        ce_loss = self.lm_loss(logits, targets)

        # Compute load balancing loss
        lb_result = self.lb_loss(bpred_outputs)
        lb_total = lb_result["total"]
        lb_per_stage = lb_result["per_stage"]

        # Combine losses
        total_loss = ce_loss + self.lb_weight * lb_total

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "lb_loss": lb_total,
            "lb_losses_per_stage": lb_per_stage,
        }


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity from cross-entropy loss.

    Args:
        loss: Cross-entropy loss

    Returns:
        Perplexity
    """
    return torch.exp(loss)


def compute_bpb(loss: torch.Tensor) -> torch.Tensor:
    """
    Compute bits-per-byte from cross-entropy loss.

    CrossEntropyLoss in PyTorch uses natural log (nats).
    This converts nats to bits for byte-level models.

    Args:
        loss: Cross-entropy loss in nats

    Returns:
        Bits-per-byte (technically bits-per-token)
    """
    return loss / math.log(2.0)


def compute_token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute token-level accuracy.

    Args:
        logits: Model predictions (B, L, vocab_size)
        targets: Target token ids (B, L)
        ignore_index: Index to ignore

    Returns:
        Token accuracy
    """
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)

    # Create mask for valid tokens
    mask = targets != ignore_index
    valid = mask.sum()

    # Guard against empty mask
    if valid == 0:
        return torch.tensor(0.0, device=logits.device)

    # Compute accuracy only on valid tokens
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / valid.float()

    return accuracy


def get_routing_statistics(
    bpred_outputs: List[RoutingModuleOutput],
    downsampling_factors: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute statistics about routing decisions.

    This tracks F_s (actual selection rate from hard boundaries) and G_s (average
    boundary probability) as described in the paper's ratio loss (Eq. 10).
    During training, F_s and G_s should converge to ~1/N_s where N_s is the target
    compression ratio.

    Args:
        bpred_outputs: List of routing module outputs
        downsampling_factors: Optional target compression ratios per stage

    Returns:
        Dictionary with routing statistics (boundary rates, compression ratios, etc.)
    """
    stats = {}

    for stage_idx, router_output in enumerate(bpred_outputs):
        # F_s: actual fraction selected (from hard boundaries, non-differentiable)
        F_s = router_output.boundary_mask.float().mean()
        stats[f"stage_{stage_idx}_F_actual_selection"] = F_s

        # G_s: average boundary probability (differentiable)
        if hasattr(router_output, "boundary_prob") and router_output.boundary_prob is not None:
            # boundary_prob is (B, L, 2) with [no-boundary, boundary] probs
            # Extract boundary probability (last channel) matching official implementation
            tokenized_prob = router_output.boundary_prob[..., -1]
            G_s = tokenized_prob.float().mean()
            stats[f"stage_{stage_idx}_G_avg_prob"] = G_s

            # Compute actual compression ratio (1/F_s)
            actual_compression = 1.0 / (F_s + 1e-8)
            stats[f"stage_{stage_idx}_actual_compression"] = actual_compression

            # If target is provided, compute target vs actual
            if downsampling_factors is not None and stage_idx < len(downsampling_factors):
                target_N = downsampling_factors[stage_idx]
                stats[f"stage_{stage_idx}_target_compression"] = torch.tensor(
                    target_N, device=F_s.device
                )
                stats[f"stage_{stage_idx}_compression_error"] = actual_compression - target_N
        else:
            # Fallback: only hard mask available
            stats[f"stage_{stage_idx}_G_avg_prob"] = F_s

    return stats
