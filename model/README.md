# NOSE 8B inference checkpoint

The GitHub Release asset `nose-8b-inference-v1.0.0.tar.gz` contains only the
components trained by NOSE:

```text
nose-8b-inference-v1.0.0/
├── config.json
├── manifest.json
├── nose_components.safetensors
└── text_encoder_lora/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

`nose_components.safetensors` contains the molecular descriptor/receptor
adapters and descriptor/receptor projection heads. The LoRA archive contains
Qwen q/k/v adapter tensors. Optimizer state, random-number state, training data,
splits, logs, and base-model parameters are not included.

## Required base models

Download the base encoders from their original publishers:

```bash
huggingface-cli download Qwen/Qwen3-Embedding-8B \
  --local-dir weights/qwen/Qwen3-Embedding-8B

huggingface-cli download dptech/Uni-Mol-Models \
  mol_pre_no_h_220816.pt mol.dict.txt \
  --local-dir weights/unimol
```

The receptor projection head expects 1,280-dimensional ESM-2 representations
from [`facebook/esm2_t33_650M_UR50D`](https://huggingface.co/facebook/esm2_t33_650M_UR50D).
The public demos do not distribute receptor sequences.

## Loading

```python
from nose import NOSEPipeline

pipeline = NOSEPipeline.from_pretrained(device="cuda:0")
embeddings = pipeline.encode_descriptors(["rose", "smoky", "odorless"])
```

Without local overrides, the pipeline downloads and verifies this checkpoint
from GitHub Releases and lets Transformers obtain Qwen from Hugging Face.

For local experiments, copy `.env.local.example` to the ignored `.env.local`
file and point it to any compatible native checkpoint:

```text
NOSE_CHECKPOINT=/path/to/compatible/checkpoint
NOSE_QWEN_MODEL=/path/to/Qwen3-Embedding-8B
NOSE_UNIMOL_WEIGHT_DIR=/path/to/unimol/weights
```

Local checkpoints must retain the current model architecture, tensor names,
and configuration fields. Architecture-changing experiments require a
corresponding loader update.

## Integrity

The locally prepared release archive has SHA256:

```text
9af2ad071bb3eb4c73c4e9501f2d9b9b2bca239ae79033728d3d119138b7896f
```

Regenerate this value if any release file changes.