"""Exact cosine retrieval and rank utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RetrievalResult:
    scores: torch.Tensor
    indices: torch.Tensor
    candidate_ids: list[list[Any]] | None = None


def retrieve(
    query_embeddings: Any,
    candidate_embeddings: Any,
    *,
    top_k: int = 10,
    candidate_ids: Sequence[Any] | None = None,
) -> RetrievalResult:
    queries = torch.as_tensor(query_embeddings).float()
    candidates = torch.as_tensor(candidate_embeddings).float()
    if queries.ndim == 1:
        queries = queries.unsqueeze(0)
    if candidates.ndim != 2 or queries.ndim != 2:
        raise ValueError("query and candidate embeddings must be 2D")
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError("query and candidate dimensions differ")
    if candidates.shape[0] == 0 or top_k <= 0:
        raise ValueError("candidates and top_k must be non-empty/positive")
    if candidate_ids is not None and len(candidate_ids) != candidates.shape[0]:
        raise ValueError("candidate_ids length must match candidates")
    scores = F.normalize(queries, dim=-1) @ F.normalize(candidates, dim=-1).T
    values, indices = torch.topk(
        scores, min(int(top_k), candidates.shape[0]), dim=1, sorted=True
    )
    resolved = None
    if candidate_ids is not None:
        ids = list(candidate_ids)
        resolved = [[ids[index] for index in row.tolist()] for row in indices.cpu()]
    return RetrievalResult(values.cpu(), indices.cpu(), resolved)


def rank_percentile(
    query_embedding: Any,
    candidate_embeddings: Any,
    target_index: int,
) -> tuple[int, float]:
    """Return one-based cosine rank and ``rank / candidate_count * 100``."""
    query = F.normalize(torch.as_tensor(query_embedding).float().reshape(1, -1), dim=-1)
    candidates = F.normalize(torch.as_tensor(candidate_embeddings).float(), dim=-1)
    scores = (query @ candidates.T).squeeze(0)
    order = torch.argsort(scores, descending=True)
    matches = (order == int(target_index)).nonzero(as_tuple=False)
    if not len(matches):
        raise IndexError(target_index)
    rank = int(matches[0, 0].item()) + 1
    return rank, rank / candidates.shape[0] * 100.0
