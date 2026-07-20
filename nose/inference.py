"""High-level inference API for public and local NOSE checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoint import (
    checkpoint_files,
    load_nose_components,
    resolve_checkpoint,
    verify_checkpoint,
)
from .config import (
    is_local_reference,
    load_config,
    load_local_environment,
    resolve_model_reference,
)
from .encoders import UniMolEncoder
from .model import NOSEModel


def _torch_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[value.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {value}") from error


class NOSEPipeline:
    def __init__(
        self,
        model: NOSEModel,
        tokenizer: object,
        config: object,
        checkpoint_dir: Path,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self._unimol: UniMolEncoder | None = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path | None = None,
        *,
        qwen_model: str | Path | None = None,
        device: str | torch.device | None = None,
        torch_dtype: str | torch.dtype = "bfloat16",
    ) -> "NOSEPipeline":
        load_local_environment()
        checkpoint = resolve_checkpoint(checkpoint_dir)
        verify_checkpoint(checkpoint)
        config_path, _, lora_dir = checkpoint_files(checkpoint)
        config = load_config(config_path)
        qwen = resolve_model_reference(
            qwen_model, "NOSE_QWEN_MODEL", config.qwen_model
        )
        target_device = torch.device(
            device or os.getenv("NOSE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = _torch_dtype(torch_dtype)
        if target_device.type == "cpu" and dtype == torch.float16:
            dtype = torch.float32

        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer

        local_only = is_local_reference(qwen)
        tokenizer = AutoTokenizer.from_pretrained(
            qwen,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        base = AutoModel.from_pretrained(
            qwen,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=local_only,
            low_cpu_mem_usage=True,
        )
        text_encoder = PeftModel.from_pretrained(
            base,
            str(lora_dir),
            is_trainable=False,
            local_files_only=True,
        )
        if hasattr(text_encoder, "config"):
            text_encoder.config.use_cache = False
        model = NOSEModel(
            config.model,
            text_encoder,
            hard_orthogonal=config.hard_orthogonal,
        )
        load_nose_components(model, checkpoint)
        model.to(target_device)
        model.eval()
        return cls(model, tokenizer, config, checkpoint, target_device)

    def _tokenize(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        values = [str(value).strip() for value in texts]
        if not values or any(not value for value in values):
            raise ValueError("texts must contain non-empty strings")
        encoded = self.tokenizer(
            values,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def encode_native(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> torch.Tensor:
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                representation = self.model.encode_native(
                    self._tokenize(texts[start : start + batch_size])
                ).float()
                outputs.append(
                    F.normalize(representation, dim=-1) if normalize else representation
                )
        return torch.cat(outputs).cpu()

    def encode_descriptors(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> torch.Tensor:
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                embeddings = self.model.encode_descriptors(
                    self._tokenize(texts[start : start + batch_size]),
                ).float()
                outputs.append(F.normalize(embeddings, dim=-1) if normalize else embeddings)
        return torch.cat(outputs).cpu()

    def encode_smiles_from_repr(
        self,
        representations: np.ndarray | torch.Tensor,
        *,
        branch: str = "descriptor",
        batch_size: int = 256,
        normalize: bool = True,
    ) -> torch.Tensor:
        values = torch.as_tensor(representations, dtype=torch.float32)
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                embeddings = self.model.encode_smiles_from_repr(
                    values[start : start + batch_size].to(self.device),
                    branch=branch,
                ).float()
                outputs.append(F.normalize(embeddings, dim=-1) if normalize else embeddings)
        return torch.cat(outputs).cpu()

    def encode_smiles(
        self,
        smiles: Sequence[str],
        *,
        branch: str = "descriptor",
        unimol_weight_dir: str | Path | None = None,
    ) -> torch.Tensor:
        if self._unimol is None:
            self._unimol = UniMolEncoder(
                weight_dir=unimol_weight_dir,
                repository=self.config.unimol_repository,
                checkpoint=self.config.unimol_checkpoint,
                dictionary=self.config.unimol_dictionary,
            )
        return self.encode_smiles_from_repr(self._unimol.encode(smiles), branch=branch)
