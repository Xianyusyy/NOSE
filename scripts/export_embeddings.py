#!/usr/bin/env python3
"""Export NOSE descriptor or SMILES embeddings to a NumPy archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nose import NOSEPipeline


def _read_lines(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [value for value in values if value and not value.startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="UTF-8 file with one item per line")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--modality", choices=["descriptor", "smiles"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    values = _read_lines(args.input)
    pipeline = NOSEPipeline.from_pretrained(
        args.checkpoint,
        qwen_model=args.qwen_model,
        device=args.device,
    )
    embeddings = (
        pipeline.encode_descriptors(values)
        if args.modality == "descriptor"
        else pipeline.encode_smiles(values)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        values=np.asarray(values, dtype=str),
        embeddings=embeddings.numpy(),
        modality=np.asarray(args.modality),
    )
    print(f"Wrote {len(values)} {args.modality} embeddings to {args.output}")


if __name__ == "__main__":
    main()
