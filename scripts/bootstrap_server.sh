#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-cuda121}"
ENV_NAME="${2:-face-preprocess}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${BACKEND}" in
  cuda121|cpu) ;;
  *)
    echo "Usage: bash scripts/bootstrap_server.sh [cuda121|cpu] [env_name]" >&2
    exit 2
    ;;
esac

if command -v mamba >/dev/null 2>&1; then
  CONDA_EXE="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXE="conda"
else
  echo "Conda or Mamba is required." >&2
  exit 4
fi

"${CONDA_EXE}" create -y -n "${ENV_NAME}" python=3.10 pip=24.3
"${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install \
  -r "${PROJECT_DIR}/environments/requirements-linux-${BACKEND}.lock.txt"
"${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install --no-deps -e "${PROJECT_DIR}"
"${CONDA_EXE}" run -n "${ENV_NAME}" face-preprocess --json doctor

echo "Environment ${ENV_NAME} is ready."

