"""Orthogonal molecular branch projection used by released NOSE weights."""

from __future__ import annotations

import torch


def hard_orthogonalize(
    vector: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    if vector.shape != reference.shape:
        raise ValueError("vector and reference must have the same shape")
    coefficient = (vector * reference).sum(dim=-1, keepdim=True)
    denominator = reference.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    return vector - coefficient / denominator * reference
