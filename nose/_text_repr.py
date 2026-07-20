"""Internal text-representation helpers."""

from __future__ import annotations

import torch


def summarize_reference(states: torch.Tensor) -> torch.Tensor:
    if states.ndim != 3:
        raise ValueError("Expected states with shape [batch, sequence, hidden]")
    return states.select(1, 0)


def summarize_nose(states: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    if states.ndim != 3 or attention.ndim != 2:
        raise ValueError("Expected states [B,L,D] and attention [B,L]")
    if states.shape[:2] != attention.shape:
        raise ValueError("States and attention shapes are incompatible")
    if torch.any(attention.sum(dim=1) == 0):
        raise ValueError("Cannot summarize an empty sequence")
    lengths = attention.sum(dim=1).to(torch.long) - 1
    rows = torch.arange(states.shape[0], device=states.device)
    return states[rows, lengths]
