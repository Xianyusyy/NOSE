# NOSE LLM-Augmented SMILES–Descriptor Dataset

The release asset `nose-smiles-descriptor-v1.0.0.csv.gz` contains two columns:

- `SMILES`: molecular structure string.
- `TARGET_Descriptor`: an odor descriptor associated with that molecule.

## Dataset statistics

| Quantity | Value | Meaning |
|---|---:|---|
| Rows | 2,567,558 | Records in the compressed CSV, including repeated pairs |
| Unique pairs | 1,513,528 | Distinct `(SMILES, TARGET_Descriptor)` combinations |
| Repeated rows | 1,054,030 | Rows duplicating a combination that appears elsewhere |
| Unique SMILES | 9,513 | Distinct molecular strings |
| Descriptor vocabulary | 1,086 | Distinct odor-description terms |

Machine-readable values and the release checksum are in
[`dataset_stats.json`](dataset_stats.json).

## Integrity

Download the dataset from the GitHub Release and verify:

```bash
sha256sum -c nose-smiles-descriptor-v1.0.0.csv.gz.sha256
```

The current checksum is recorded in `dataset_stats.json`. The audit found no
values beginning with spreadsheet formula prefixes (`=`, `+`, `-`, or `@`).

## Licensing and provenance

The dataset is a compilation and augmentation of olfactory records from sources
described in the NOSE paper. The repository's MIT license applies to NOSE source
code; it does not automatically replace third-party dataset terms. Users are
responsible for complying with the terms of the underlying sources and for
validating the data before commercial use.