# 3D Facial Preprocessing Pipeline

An inference-only, auditable pipeline for mapping raw triangular facial OBJ meshes to a fixed 7,906-vertex template. The release packages the selected crop and landmark models, nine-landmark coarse alignment, MeshMonk-style non-rigid mapping, GPA normalization, optional symmetry, and optional phenotype-landmark export.

Chinese documentation: [README_zh.md](README_zh.md)

## Processing flow

```text
raw OBJ (millimetres)
  -> PointNeXt crop model
  -> nine-point landmark model
  -> nine-landmark similarity GPA
  -> MeshMonk mapping
  -> 7,906-point GPA normalization
  -> optional symmetry
  -> optional phenotype-landmark CSV and PickedPoints export
```

The phenotype stage is off by default. When requested, it runs on the final mesh: after symmetry when `--symmetry` is enabled, otherwise after the second GPA.

## Privacy scope

This repository contains inference code, two selected inference checkpoints, fixed geometry assets, configuration, tests, and documentation. It contains no raw scans, training or validation data, sample manifests, individual predictions, subject labels, or run outputs.

The crop checkpoint is an inference-only derivative containing only `model_state` and the five architecture parameters required by the loader. Training paths, optimizer state, scheduler state, metrics, and run metadata were removed. Run the release audit before publishing a change:

```bash
python scripts/audit_public_release.py --root . --json
```

## Installation

Python 3.10 is recommended. Install a locked CPU environment on Linux:

```bash
python -m pip install -r environments/requirements-linux-cpu.lock.txt
python -m pip install --no-deps .
```

For CUDA 12.1, use `environments/requirements-linux-cuda121.lock.txt`. Windows lock files are provided with the corresponding names under `environments/`.

## Validate the runtime

```bash
face-preprocess --json doctor \
  --config configs/pipeline_server.yaml \
  --device-profile configs/device_scanner_1p2mm.yaml \
  --qc-config configs/qc_exploratory.yaml \
  --device cpu
```

The doctor validates asset hashes and loads both checkpoints strictly when PyTorch is available.

## Run the complete pipeline

```bash
face-preprocess --json run \
  --input /path/to/raw.obj \
  --output /path/to/run_output \
  --config configs/pipeline_server.yaml \
  --device-profile configs/device_scanner_1p2mm.yaml \
  --qc-config configs/qc_exploratory.yaml \
  --device cpu \
  --workers 1 \
  --qc-level standard \
  --output-mode full \
  --symmetry \
  --phenotype-landmarks
```

`--input` accepts one OBJ or a flat directory of OBJ files. Use `--input-manifest` instead for an explicit CSV manifest. Omit `--symmetry` or `--phenotype-landmarks` to disable those optional stages.

## Export phenotype landmarks from existing normalized meshes

```bash
face-preprocess --json phenotype-landmarks \
  --input /path/to/final_obj \
  --output /path/to/phenotype_output \
  --config configs/pipeline_server.yaml
```

This writes `<stem>_landmarks.csv` and `<stem>_landmarks.pp`. Add `--no-pp` to produce CSV only. Inputs must have exactly the configured template face topology; the command fails rather than projecting onto an incompatible mesh.

## Main outputs

- `final_obj/<output_key>.obj`: passed or warning final meshes.
- `phenotype_landmarks/<output_key>.csv`: optional named phenotype coordinates.
- `phenotype_landmarks/<output_key>.pp`: optional PickedPoints representation.
- `metadata/samples/<output_key>.json`: hashes, model identities, transforms, QC, and optional phenotype provenance.
- `artifacts/<output_key>/`: intermediate meshes and diagnostic arrays in `full` mode.

See [model assets](docs/model_assets.md) and the [output contract](docs/output_contract.md) for the fixed scientific interfaces.

## Tests

```bash
python -m pytest -q
python scripts/audit_public_release.py --root . --json
```

Real model smoke tests are marked `model_smoke` and require PyTorch plus the packaged checkpoints.

## License

No license has been assigned to this repository. Public visibility does not itself grant reuse rights.
