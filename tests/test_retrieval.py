import pytest
import torch

from nose.retrieval import rank_percentile, retrieve


def test_cosine_retrieval_and_ids() -> None:
    result = retrieve(
        [[1.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]],
        top_k=2,
        candidate_ids=["y", "x", "-x"],
    )
    assert result.indices.tolist() == [[1, 0]]
    assert result.candidate_ids == [["x", "y"]]


def test_rank_percentile_definition() -> None:
    rank, percentile = rank_percentile(
        torch.tensor([1.0, 0.0]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.5, 0.5]]),
        3,
    )
    assert rank == 2
    assert percentile == pytest.approx(50.0)
