"""Group-fairness disparities over per-user recommendation quality.

Per-user accuracy metrics live in metrics_accuracy.py; these measure the gap
between two groups rather than the quality itself.
"""

from __future__ import annotations

import torch


def brier_rmse_per_user(
    scores: torch.Tensor,
    pos_mask: torch.Tensor,
    cand_mask: torch.Tensor,
    balanced: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-user RMSE between calibrated relevance and binary test relevance.

    :param scores: [U, I] raw LightGCN scores (dot products).
    :param pos_mask: [U, I] 1.0 where the item is a test positive for that user.
    :param cand_mask: [U, I] 1.0 where the item is a candidate, i.e. not already
        seen by that user in train or validation.
    :return: [U] per-user RMSE in [0, 1]. Lower is better, so a *larger* value
        for a group means that group is served worse.
    """
    prob = torch.sigmoid(scores)
    err2 = (prob - pos_mask) ** 2

    if balanced:
        neg_mask = cand_mask * (1.0 - pos_mask)
        n_pos = pos_mask.sum(dim=1).clamp_min(eps)
        n_neg = neg_mask.sum(dim=1).clamp_min(eps)
        mse = 0.5 * (err2 * pos_mask).sum(dim=1) / n_pos \
            + 0.5 * (err2 * neg_mask).sum(dim=1) / n_neg
    else:
        n = cand_mask.sum(dim=1).clamp_min(eps)
        mse = (err2 * cand_mask).sum(dim=1) / n

    return torch.sqrt(mse + eps)


def group_disparity(rmse_u: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor):
    """Demographic parity on the RMSE, eq. (5) specialised to one scalar metric.

    Returns (signed difference A - B, magnitude). The attacker maximises the
    magnitude; the sign is reported so a sign flip is visible rather than hidden
    behind the absolute value.
    """
    signed = rmse_u.index_select(0, idx_a).mean() - rmse_u.index_select(0, idx_b).mean()
    return signed, signed.abs()
