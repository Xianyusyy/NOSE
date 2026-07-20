# NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning

<p align="center">
  <b>ACL 2026 Main Conference</b>
</p>

<p align="center">
  Yanyi Su<sup>1,2</sup> &nbsp;
  Hongshuai Wang<sup>2</sup> &nbsp;
  Zhifeng Gao<sup>2*</sup> &nbsp;
  Jun Cheng<sup>1,3,4*</sup>
</p>

<p align="center">
  <sup>1</sup>College of Chemistry and Chemical Engineering, Xiamen University &nbsp;
  <sup>2</sup>DP Technology &nbsp;
  <sup>3</sup>AI4EC, IKKEM &nbsp;
  <sup>4</sup>Institute of Artificial Intelligence, Xiamen University
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.10452"><img alt="Paper" src="https://img.shields.io/badge/Paper-ACL%202026-red"></a>
  <a href="https://github.com/Xianyusyy/NOSE"><img alt="Code" src="https://img.shields.io/badge/Code-GitHub-blue"></a>
  <a href="https://github.com/Xianyusyy/NOSE/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

## Abstract

Olfaction lies at the intersection of chemical structure, neural encoding, and linguistic perception, yet existing representation methods fail to fully capture this pathway. Current approaches typically model only isolated segments of the olfactory pathway, overlooking the complete chain from molecule to receptors to linguistic descriptions. Such fragmentation yields learned embeddings that lack both biological grounding and semantic interpretability. We propose **NOSE** (Neural Olfactory-Semantic Embedding), a representation learning framework that aligns three modalities along the olfactory pathway: **molecular structure**, **receptor sequence**, and **natural language description**. Rather than simply fusing these signals, we decouple their contributions via orthogonal constraints, preserving the unique encoded information of each modality. To address the sparsity of olfactory language, we introduce a weak positive sample strategy to calibrate semantic similarity, preventing erroneous repulsion of similar odors in the feature space. Extensive experiments demonstrate that NOSE achieves state-of-the-art (SOTA) performance and excellent zero-shot generalization, confirming the strong alignment between its representation space and human olfactory intuition.

## Updates

- **[2026/07]** Inference-only 8B release prepared with adapters, projection
  heads, Qwen LoRA, the LLM-augmented pair dataset, and three runnable demos.
- **[2026/04]** Paper accepted at ACL 2026 Main Conference.

## Release scope

This repository provides:

- [x] NOSE molecular/receptor adapters and projection heads
- [x] Qwen3-Embedding-8B LoRA weights
- [x] LLM-augmented SMILES–descriptor data (2,567,558 rows)
- [x] Embedding extraction and cosine-retrieval APIs
- [x] Three real-inference demo notebooks

## Installation

```bash
git clone https://github.com/Xianyusyy/NOSE.git
cd NOSE
pip install -e .

# Notebook, plotting and live Uni-Mol SMILES encoding dependencies
pip install -e ".[demo]"
```

Python 3.10+ and a CUDA GPU are recommended. The 8B Qwen base model requires
substantial GPU memory.

## Download base models

NOSE Release assets contain only trained NOSE components. Download the base
encoders from their publishers:

```bash
huggingface-cli download Qwen/Qwen3-Embedding-8B \
  --local-dir weights/qwen/Qwen3-Embedding-8B

huggingface-cli download dptech/Uni-Mol-Models \
  mol_pre_no_h_220816.pt mol.dict.txt \
  --local-dir weights/unimol
```

For receptor embeddings, use
[`facebook/esm2_t33_650M_UR50D`](https://huggingface.co/facebook/esm2_t33_650M_UR50D).
NOSE does not redistribute any of these base weights.

## Quickstart

The first call downloads and verifies the NOSE adapter checkpoint from GitHub
Releases and downloads Qwen from Hugging Face:

```python
from nose import NOSEPipeline, retrieve

pipeline = NOSEPipeline.from_pretrained(
    device="cuda:0",
)

odor_terms = ["rose", "smoky", "odorless"]
embeddings = pipeline.encode_descriptors(odor_terms)
result = retrieve(embeddings[:1], embeddings, top_k=3, candidate_ids=odor_terms)
print(result.candidate_ids)
```

For a newly trained compatible local checkpoint, copy
`.env.local.example` to the ignored `.env.local` file and set local paths once:

```bash
# .env.local (never committed)
NOSE_CHECKPOINT=/path/to/compatible/checkpoint
NOSE_QWEN_MODEL=/path/to/Qwen3-Embedding-8B
NOSE_UNIMOL_WEIGHT_DIR=/path/to/unimol/weights
NOSE_UNIMOL_TOOLS_PATH=/path/to/directory/containing/unimol_tools
NOSE_DEVICE=cuda:0
```

Then launch normally:

```bash
jupyter lab demos/
```

The loader accepts the public inference checkpoint and compatible local
checkpoints. The notebooks contain no machine-specific paths.

## Demos

- [`demos/01_odor_space.ipynb`](demos/01_odor_space.ipynb): fixed-category
  Qwen representation before/after NOSE LoRA.
- [`demos/02_odor_algebra.ipynb`](demos/02_odor_algebra.ipynb): seven
  registered vector-algebra queries over 1,086 descriptors.
- [`demos/03_smiles_to_odor.ipynb`](demos/03_smiles_to_odor.ipynb): real
  Uni-Mol → NOSE retrieval for three sourced PubChem structures.

## Dataset

The dataset Release asset has two columns, `SMILES` and `TARGET_Descriptor`.
It has 2,567,558 rows including duplicates and 1,513,528 unique pairs.

## Project structure

```
NOSE/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── data/                      # Dataset card and release statistics
├── model/                     # Checkpoint model card and download instructions
├── nose/                      # Inference, model, checkpoint and retrieval API
├── scripts/                   # Embedding export and demo validation
├── tests/                     # CPU unit tests
└── demos/                     # Three public notebooks and fixed assets
```

## Pre-trained Encoders

NOSE builds upon the following pre-trained models:

| Modality | Encoder | Source |
|----------|---------|--------|
| Molecular Structure | [Uni-Mol](https://huggingface.co/dptech/Uni-Mol-Models) | 3D molecular conformations |
| Receptor Sequence | [ESM-2 (650M)](https://huggingface.co/facebook/esm2_t33_650M_UR50D) | Protein language model |
| Odor Description | [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | Text embedding model |

## Downstream Task Data

Downstream evaluation datasets are publicly available via [pyrfume-data](https://github.com/pyrfume/pyrfume-data) (MIT License).

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{su-etal-2026-nose,
    title = "{NOSE}: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning",
    author = "Su, Yanyi  and
      Wang, Hongshuai  and
      Gao, Zhifeng  and
      Cheng, Jun",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.898/",
    doi = "10.18653/v1/2026.acl-long.898",
    pages = "19615--19647",
    ISBN = "979-8-89176-390-6",
}
```
