import pytest
import torch

from nose._text_repr import summarize_nose, summarize_reference


def test_nose_summary_handles_variable_lengths() -> None:
    states = torch.arange(2 * 3 * 2).reshape(2, 3, 2)
    attention = torch.tensor([[1, 1, 0], [1, 1, 1]])
    summary = summarize_nose(states, attention)
    assert torch.equal(summary[0], states[0, 1])
    assert torch.equal(summary[1], states[1, 2])


def test_nose_summary_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize_nose(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long))


def test_reference_summary_shape() -> None:
    states = torch.arange(2 * 3 * 2).reshape(2, 3, 2)
    assert summarize_reference(states).shape == (2, 2)
