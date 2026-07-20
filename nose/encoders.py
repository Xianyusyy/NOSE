"""Wrappers for external base encoders that are not redistributed by NOSE."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


class UniMolEncoder:
    """Encode SMILES with the released Uni-Mol no-hydrogen checkpoint."""

    def __init__(
        self,
        *,
        weight_dir: str | Path | None = None,
        repository: str = "dptech/Uni-Mol-Models",
        checkpoint: str = "mol_pre_no_h_220816.pt",
        dictionary: str = "mol.dict.txt",
        batch_size: int = 64,
    ) -> None:
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        resolved = Path(
            weight_dir
            or os.getenv("NOSE_UNIMOL_WEIGHT_DIR")
            or Path.home() / ".cache" / "nose" / "unimol"
        ).expanduser()
        resolved.mkdir(parents=True, exist_ok=True)
        missing = [name for name in (checkpoint, dictionary) if not (resolved / name).is_file()]
        if missing:
            from huggingface_hub import hf_hub_download

            for name in missing:
                hf_hub_download(repo_id=repository, filename=name, local_dir=resolved)
        os.environ["UNIMOL_WEIGHT_DIR"] = str(resolved)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        vendor = os.getenv("NOSE_UNIMOL_TOOLS_PATH")
        if vendor:
            sys.path.insert(0, str(Path(vendor).expanduser()))
        try:
            from unimol_tools import UniMolRepr
        except ImportError as error:
            raise ImportError(
                "SMILES encoding requires `pip install 'nose-olfaction[demo]'` "
                "or NOSE_UNIMOL_TOOLS_PATH pointing to a compatible unimol_tools package."
            ) from error
        self._encoder = UniMolRepr(data_type="molecule", remove_hs=True)

    def encode(self, smiles: Sequence[str]) -> np.ndarray:
        values = [str(value).strip() for value in smiles]
        if not values or any(not value for value in values):
            raise ValueError("smiles must contain non-empty strings")
        chunks: list[np.ndarray] = []
        for start in range(0, len(values), self.batch_size):
            result = self._encoder.get_repr(values[start : start + self.batch_size])
            chunks.append(np.asarray(result["cls_repr"], dtype=np.float32))
        representations = np.concatenate(chunks, axis=0)
        if representations.shape != (len(values), 512):
            raise RuntimeError(
                f"Expected Uni-Mol representations [{len(values)},512], "
                f"got {representations.shape}"
            )
        return representations
