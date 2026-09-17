#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 4 )); then
  echo "Usage: bash scripts/verify_server.sh INPUT_OBJ OUTPUT_DIR [ENV_NAME] [cuda|cpu]" >&2
  exit 2
fi

INPUT_OBJ="$1"
OUTPUT_DIR="$2"
ENV_NAME="${3:-face-preprocess}"
DEVICE="${4:-cuda}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${DEVICE}" in
  cuda|cpu) ;;
  *)
    echo "DEVICE must be cuda or cpu." >&2
    exit 2
    ;;
esac

if [[ ! -f "${INPUT_OBJ}" ]]; then
  echo "Input OBJ does not exist: ${INPUT_OBJ}" >&2
  exit 2
fi

if command -v mamba >/dev/null 2>&1; then
  CONDA_EXE="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXE="conda"
else
  echo "Conda or Mamba is required." >&2
  exit 4
fi

cd "${PROJECT_DIR}"

"${CONDA_EXE}" run -n "${ENV_NAME}" face-preprocess --json doctor \
  --config configs/pipeline_server.yaml \
  --device-profile configs/device_scanner_1p2mm.yaml \
  --qc-config configs/qc_exploratory.yaml \
  --device "${DEVICE}"

"${CONDA_EXE}" run -n "${ENV_NAME}" python -m pytest -m model_smoke -q

"${CONDA_EXE}" run -n "${ENV_NAME}" face-preprocess --json validate-run \
  --input "${INPUT_OBJ}" \
  --config configs/pipeline_server.yaml \
  --device-profile configs/device_scanner_1p2mm.yaml \
  --qc-config configs/qc_exploratory.yaml \
  --device "${DEVICE}" \
  --workers 1 \
  --qc-level standard \
  --output-mode full

"${CONDA_EXE}" run -n "${ENV_NAME}" face-preprocess --json run \
  --input "${INPUT_OBJ}" \
  --output "${OUTPUT_DIR}" \
  --config configs/pipeline_server.yaml \
  --device-profile configs/device_scanner_1p2mm.yaml \
  --qc-config configs/qc_exploratory.yaml \
  --device "${DEVICE}" \
  --workers 1 \
  --qc-level standard \
  --output-mode full

echo "Server acceptance passed."
