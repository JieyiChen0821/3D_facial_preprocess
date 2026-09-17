#!/usr/bin/env bash
#SBATCH --job-name=faceprep-doctor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/doctor-%j.out
#SBATCH --error=logs/doctor-%j.err

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR with sbatch --export=ALL,PROJECT_DIR=/absolute/project/path}"
ENV_NAME="${ENV_NAME:-face-preprocess}"
cd "${PROJECT_DIR}"
mkdir -p logs

conda run -n "${ENV_NAME}" face-preprocess --json doctor \
  --config configs/pipeline_server.yaml \
  --device-profile configs/device_scanner_1p2mm.yaml \
  --qc-config configs/qc_exploratory.yaml \
  --device cuda

