"""Safe loading for public and native NOSE checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from safetensors.torch import load_file

RELEASE_VERSION = "v1.0.0"
RELEASE_DIRECTORY = "nose-8b-inference-v1.0.0"
RELEASE_ARCHIVE = f"{RELEASE_DIRECTORY}.tar.gz"
RELEASE_URL = (
    f"https://github.com/Xianyusyy/NOSE/releases/download/{RELEASE_VERSION}/"
    f"{RELEASE_ARCHIVE}"
)
RELEASE_SHA256 = "9af2ad071bb3eb4c73c4e9501f2d9b9b2bca239ae79033728d3d119138b7896f"


class CheckpointError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise CheckpointError(f"Unsafe path in release archive: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise CheckpointError(f"Unsupported link/device in release archive: {member.name}")
        handle.extractall(destination)


def resolve_checkpoint(checkpoint_dir: str | Path | None = None) -> Path:
    """Resolve an explicit/local checkpoint or download the public GitHub Release."""
    explicit = checkpoint_dir or os.getenv("NOSE_CHECKPOINT")
    if explicit:
        return Path(explicit).expanduser()

    package_root = Path(__file__).resolve().parents[1]
    local_candidates = (
        package_root / "model" / "nose-8b",
        package_root / ".release" / RELEASE_DIRECTORY,
    )
    for candidate in local_candidates:
        if (candidate / "manifest.json").is_file():
            return candidate

    cache_root = Path(
        os.getenv("NOSE_CACHE_DIR", Path.home() / ".cache" / "nose")
    ).expanduser()
    destination = cache_root / RELEASE_DIRECTORY
    if (destination / "manifest.json").is_file():
        return destination

    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nose-download-", dir=cache_root) as temporary:
        temporary_root = Path(temporary)
        archive = temporary_root / RELEASE_ARCHIVE
        try:
            with urlopen(RELEASE_URL) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        except (OSError, URLError) as error:
            raise CheckpointError(
                "Could not download the NOSE 8B GitHub Release. "
                "Set NOSE_CHECKPOINT to an extracted checkpoint directory or download "
                f"{RELEASE_URL} manually. Original error: {error}"
            ) from error
        actual = _sha256(archive)
        if actual != RELEASE_SHA256:
            raise CheckpointError(
                f"Release SHA256 mismatch: expected {RELEASE_SHA256}, got {actual}"
            )
        extracted = temporary_root / "extracted"
        extracted.mkdir()
        _safe_extract(archive, extracted)
        source = extracted / RELEASE_DIRECTORY
        if not (source / "manifest.json").is_file():
            raise CheckpointError(
                f"Release archive does not contain {RELEASE_DIRECTORY}/manifest.json"
            )
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(source), str(destination))
    return destination


def read_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_dir)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Invalid checkpoint manifest under {root}: {error}") from error
    checkpoint_format = manifest.get("format")
    if checkpoint_format not in {"nose-inference-checkpoint", "nose-training-checkpoint"}:
        raise CheckpointError(f"Unsupported checkpoint format: {checkpoint_format!r}")
    if manifest.get("format_version") != 1:
        raise CheckpointError(f"Unsupported checkpoint version: {manifest.get('format_version')!r}")
    return manifest


def checkpoint_files(checkpoint_dir: str | Path) -> tuple[Path, Path, Path]:
    root = Path(checkpoint_dir)
    manifest = read_manifest(root)
    files = manifest.get("files", {})
    if manifest["format"] == "nose-inference-checkpoint":
        config = root / "config.json"
        components = root / "nose_components.safetensors"
        lora = root / "text_encoder_lora"
    else:
        config = root / str(files.get("config", "config.json"))
        components = root / str(files.get("nose_components", "nose_components.safetensors"))
        lora = root / str(files.get("text_encoder_lora", "text_encoder_lora"))
    for path in (config, components, lora / "adapter_config.json", lora / "adapter_model.safetensors"):
        if not path.exists():
            raise CheckpointError(f"Missing checkpoint asset: {path}")
    return config, components, lora


def verify_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_dir)
    manifest = read_manifest(root)
    if manifest["format"] == "nose-inference-checkpoint":
        for relative, metadata in manifest.get("files", {}).items():
            path = root / relative
            if not path.is_file():
                raise CheckpointError(f"Missing checkpoint asset: {path}")
            if _sha256(path) != metadata.get("sha256"):
                raise CheckpointError(f"SHA256 mismatch: {path}")
            if path.stat().st_size != metadata.get("size_bytes"):
                raise CheckpointError(f"Size mismatch: {path}")
    checkpoint_files(root)
    return manifest


def load_nose_components(
    model: Any,
    checkpoint_dir: str | Path,
    *,
    device: str = "cpu",
) -> None:
    """Load adapter and projection tensors while excluding the base Qwen model."""
    _, components, _ = checkpoint_files(checkpoint_dir)
    state = load_file(str(components), device=device)
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [key for key in result.missing_keys if not key.startswith("LLM_Model.")]
    if missing or unexpected:
        raise CheckpointError(
            f"NOSE component mismatch: missing={missing}, unexpected={unexpected}"
        )
