import hashlib
import json

import torch
from safetensors.torch import save_file

from nose.checkpoint import resolve_checkpoint, verify_checkpoint


def test_verify_public_checkpoint(tmp_path) -> None:
    (tmp_path / "text_encoder_lora").mkdir()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "text_encoder_lora/adapter_config.json").write_text("{}", encoding="utf-8")
    save_file({"weight": torch.ones(1)}, tmp_path / "nose_components.safetensors")
    save_file({"weight": torch.ones(1)}, tmp_path / "text_encoder_lora/adapter_model.safetensors")
    files = [
        "config.json",
        "nose_components.safetensors",
        "text_encoder_lora/adapter_config.json",
        "text_encoder_lora/adapter_model.safetensors",
    ]
    metadata = {}
    for relative in files:
        path = tmp_path / relative
        metadata[relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "nose-inference-checkpoint",
                "format_version": 1,
                "files": metadata,
            }
        ),
        encoding="utf-8",
    )
    assert verify_checkpoint(tmp_path)["format_version"] == 1


def test_explicit_checkpoint_never_triggers_download(tmp_path) -> None:
    assert resolve_checkpoint(tmp_path) == tmp_path
