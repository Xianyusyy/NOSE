"""Inference-only configuration for released NOSE checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdapterConfig:
    d_model: int
    hidden_dim: int
    layers: int
    dropout: float


@dataclass(frozen=True)
class ModelConfig:
    text_hidden_size: int
    embedding_dim: int
    molecular_hidden_size: int
    receptor_hidden_size: int
    descriptor_adapter: AdapterConfig
    receptor_adapter: AdapterConfig


@dataclass(frozen=True)
class InferenceConfig:
    model: ModelConfig
    qwen_model: str
    unimol_repository: str
    unimol_checkpoint: str
    unimol_dictionary: str
    esm2_model: str
    max_text_length: int = 64
    hard_orthogonal: bool = True


def load_config(path: str | Path) -> InferenceConfig:
    """Load a public config or adapt a compatible development checkpoint."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("format") == "nose-inference-config":
        if raw.get("format_version") != 1:
            raise ValueError(f"Unsupported NOSE inference config version: {source}")
        bases = raw["base_models"]
        inference = raw.get("inference", {})
    elif {"model", "paths", "training", "loss"}.issubset(raw):
        bases = {
            "qwen": raw["paths"]["qwen_model"],
            "unimol": "dptech/Uni-Mol-Models",
            "unimol_checkpoint": "mol_pre_no_h_220816.pt",
            "unimol_dictionary": "mol.dict.txt",
            "esm2": "facebook/esm2_t33_650M_UR50D",
        }
        inference = {
            "max_text_length": raw["training"].get("max_text_length", 64),
            "hard_orthogonal": raw["loss"].get("hard_orthogonal", True),
        }
    else:
        raise ValueError(f"Unsupported NOSE config: {source}")
    source_model = raw["model"]
    model = {
        key: source_model[key]
        for key in (
            "text_hidden_size",
            "embedding_dim",
            "molecular_hidden_size",
            "receptor_hidden_size",
        )
    }
    model["descriptor_adapter"] = AdapterConfig(**source_model["descriptor_adapter"])
    model["receptor_adapter"] = AdapterConfig(**source_model["receptor_adapter"])
    return InferenceConfig(
        model=ModelConfig(**model),
        qwen_model=str(bases["qwen"]),
        unimol_repository=str(bases["unimol"]),
        unimol_checkpoint=str(bases["unimol_checkpoint"]),
        unimol_dictionary=str(bases["unimol_dictionary"]),
        esm2_model=str(bases["esm2"]),
        max_text_length=int(inference.get("max_text_length", 64)),
        hard_orthogonal=bool(inference.get("hard_orthogonal", True)),
    )


def resolve_model_reference(
    explicit: str | Path | None,
    environment_variable: str,
    default: str,
) -> str:
    """Resolve explicit path, environment override, then public model ID."""
    value = explicit or os.getenv(environment_variable) or default
    return str(Path(value).expanduser()) if Path(str(value)).expanduser().exists() else str(value)


def is_local_reference(value: str | Path) -> bool:
    return Path(value).expanduser().exists()


def load_local_environment() -> Path | None:
    """Load optional untracked ``.env.local`` values without overriding the shell."""
    package_root = Path(__file__).resolve().parents[1]
    candidates = [package_root / ".env.local"]
    current = Path.cwd().resolve()
    candidates.extend(parent / ".env.local" for parent in (current, *current.parents))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                raise ValueError(f"Invalid .env.local line in {candidate}: {raw_line!r}")
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                os.environ.setdefault(key, value)
        return candidate
    return None


def config_to_dict(config: InferenceConfig) -> dict[str, Any]:
    """Small serializable summary useful in notebooks and diagnostics."""
    return {
        "qwen_model": config.qwen_model,
        "unimol_repository": config.unimol_repository,
        "esm2_model": config.esm2_model,
        "embedding_dim": config.model.embedding_dim,
        "max_text_length": config.max_text_length,
    }
