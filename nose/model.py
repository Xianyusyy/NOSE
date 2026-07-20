"""Inference-only NOSE model compatible with released adapter checkpoints."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from ._text_repr import summarize_nose, summarize_reference
from .config import ModelConfig
from .orthogonal import hard_orthogonalize
from .projection import DeepProjectionAdapter, SimCLRProjectionHead


class NOSEModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        text_encoder: nn.Module,
        *,
        hard_orthogonal: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.hard_orthogonal = hard_orthogonal
        self.LLM_Model = text_encoder
        self.adapter_desc = DeepProjectionAdapter(
            config.molecular_hidden_size,
            config.embedding_dim,
            d_model=config.descriptor_adapter.d_model,
            hidden_dim=config.descriptor_adapter.hidden_dim,
            layers=config.descriptor_adapter.layers,
            dropout=config.descriptor_adapter.dropout,
        )
        self.adapter_rec = DeepProjectionAdapter(
            config.molecular_hidden_size,
            config.embedding_dim,
            d_model=config.receptor_adapter.d_model,
            hidden_dim=config.receptor_adapter.hidden_dim,
            layers=config.receptor_adapter.layers,
            dropout=config.receptor_adapter.dropout,
        )
        self.descriptor_head = SimCLRProjectionHead(
            config.text_hidden_size, config.embedding_dim, config.embedding_dim
        )
        self.receptor_head = SimCLRProjectionHead(
            config.receptor_hidden_size, config.embedding_dim, config.embedding_dim
        )
        self.molecule_projection: nn.Module = (
            nn.Identity()
            if config.molecular_hidden_size == config.embedding_dim
            else nn.Linear(config.molecular_hidden_size, config.embedding_dim)
        )

    @property
    def text_encoder(self) -> nn.Module:
        return self.LLM_Model

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def _aligned(self, vector: torch.Tensor, molecule: torch.Tensor) -> torch.Tensor:
        return hard_orthogonalize(vector, molecule) if self.hard_orthogonal else vector

    def encode_smiles_from_repr(
        self,
        molecule: torch.Tensor,
        *,
        branch: str = "descriptor",
        fusion_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> torch.Tensor:
        descriptor = self.adapter_desc(molecule)
        receptor = self.adapter_rec(molecule)
        molecule = self.molecule_projection(molecule)
        if branch == "molecule":
            return molecule
        if branch == "descriptor":
            return self._aligned(descriptor, molecule)
        if branch == "receptor":
            return self._aligned(receptor, molecule)
        if branch != "fused":
            raise ValueError(f"Unsupported molecular branch: {branch}")
        weights = molecule.new_tensor(fusion_weights)
        weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        return weights[0] * molecule + weights[1] * descriptor + weights[2] * receptor

    def _text_states(
        self,
        text_inputs: dict[str, torch.Tensor],
        *,
        adapted: bool,
    ) -> torch.Tensor:
        disable_adapter = getattr(self.LLM_Model, "disable_adapter", None)
        context = nullcontext() if adapted or not callable(disable_adapter) else disable_adapter()
        with context:
            outputs: Any = self.LLM_Model(**text_inputs)
        return (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )

    def encode_native(self, text_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        states = self._text_states(text_inputs, adapted=False)
        return summarize_reference(states)

    def encode_descriptors(
        self,
        text_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        representation = summarize_nose(
            self._text_states(text_inputs, adapted=True),
            text_inputs["attention_mask"],
        )
        representation = representation.to(self.descriptor_head.dense.weight.dtype)
        return self.descriptor_head(representation)

    def encode_receptors(self, receptor_embedding: torch.Tensor) -> torch.Tensor:
        return self.receptor_head(receptor_embedding)
