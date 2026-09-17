#!/usr/bin/env bash
#SBATCH --job-name=faceprep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/faceprep-%j.out
#SBATCH --error=logs/faceprep-%j.err

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${INPUT:?Set INPUT to one OBJ or a flat OBJ directory}"
: "${OUTPUT:?Set OUTPUT to a new run directory}"
ENV_NAME="${ENV_NAME:-face-preprocess}"
QC_LEVEL="${QC_LEVEL:-standard}"
OUTPUT_MODE="${OUTPUT_MODE:-compact}"
DEVICE="${DEVICE:-cuda}"

cd "${PROJECT_DIR}"
mkdir -p logs

conda run -n "${ENV_NAME}" face-preprocess --json run \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --config configs/pipeline_server.yaml \
  --device-profile configs/device_scanner_1p2mm.yaml \
  --qc-config configs/qc_exploratory.yaml \
  --device "${DEVICE}" \
  --workers 1 \
  --qc-level "${QC_LEVEL}" \
  --output-mode "${OUTPUT_MODE}"

