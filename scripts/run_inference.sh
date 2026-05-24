#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="python3"

NORMALIZED_LABEL_DIR="${REPO_ROOT}/outputs/label_normalization/mineru"
MODEL_PATH="popo_model"
OUTPUT_DIR="${REPO_ROOT}/outputs/inference/mineru"

LIMIT="5"
MAX_NEW_TOKENS="8192"

echo "[run_inference] repo=${REPO_ROOT}"
echo "[run_inference] normalized_label_dir=${NORMALIZED_LABEL_DIR}"
echo "[run_inference] model_path=${MODEL_PATH}"
echo "[run_inference] output_dir=${OUTPUT_DIR}"

if [[ ! -d "${NORMALIZED_LABEL_DIR}" ]]; then
  echo "missing normalized label dir: ${NORMALIZED_LABEL_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[run_inference] start inference"
POPO_INFERENCE_BACKEND="transformers" \
POPO_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
"${PYTHON_BIN}" "${REPO_ROOT}/post_processing/run_inference.py" \
  --input-dir "${NORMALIZED_LABEL_DIR}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --raw-output-root "" \
  --limit "${LIMIT}" \
  "$@"
