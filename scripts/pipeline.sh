#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Usage:
#   bash scripts/pipeline.sh
#   LIMIT=10 MODELS="glm-ocr,mineru,monkeyocr,dolphin,PaddleOCR-VL-1.5" bash scripts/pipeline.sh
#   RUN_INFERENCE=0 RUN_BUILD_TREE=1 bash scripts/pipeline.sh
#
# Extra arguments are passed through to run_inference.py, for example:
#   bash scripts/pipeline.sh --dry-run
#   bash scripts/pipeline.sh --resume

PYTHON_BIN="${PYTHON_BIN:-/data/conda/envs/qwen3vl/bin/python}"
PYTHON_LABEL_BIN="${PYTHON_LABEL_BIN:-python3}"
PYTHON_TREE_BIN="${PYTHON_TREE_BIN:-python3}"

MODELS="${MODELS:-glm-ocr,mineru,monkeyocr,dolphin,PaddleOCR-VL-1.5}"
LIMIT="${LIMIT:-10}"
BBOX_SCALE="${BBOX_SCALE:-source}"

POST_PROCESS_ROOT="${POST_PROCESS_ROOT:-${REPO_ROOT}/post-process}"
PDF_DIR="${PDF_DIR:-eval_pdf_dir}"

LABEL_OUTPUT_DIR="${LABEL_OUTPUT_DIR:-${REPO_ROOT}/outputs/label_normalization}"
INFERENCE_OUTPUT_ROOT="${INFERENCE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/inference}"
INFERENCE_RAW_OUTPUT_ROOT="${INFERENCE_RAW_OUTPUT_ROOT:-${REPO_ROOT}/outputs/inference_raw}"
TREE_OUTPUT_ROOT="${TREE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/build_tree}"
TREE_TXT_OUTPUT_ROOT="${TREE_TXT_OUTPUT_ROOT:-${REPO_ROOT}/outputs/build_tree_txt}"

RUN_LABEL_NORMALIZATION="${RUN_LABEL_NORMALIZATION:-1}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_BUILD_TREE="${RUN_BUILD_TREE:-1}"

export POPO_INFERENCE_BACKEND="${POPO_INFERENCE_BACKEND:-transformers}"
export POPO_MODEL_PATH="${POPO_MODEL_PATH:-popo_model}"
export POPO_MAX_NEW_TOKENS="${POPO_MAX_NEW_TOKENS:-1024}"

IFS=',' read -r -a MODEL_ARRAY <<< "${MODELS}"

normalize_model_name() {
  local model="$1"
  python3 - "$model" <<'PY'
import re
import sys
print(re.sub(r"[^0-9A-Za-z_.-]+", "_", sys.argv[1]))
PY
}

log_step() {
  printf '\n[pipeline] %s\n' "$*"
}

mkdir -p "${LABEL_OUTPUT_DIR}" "${INFERENCE_OUTPUT_ROOT}" "${INFERENCE_RAW_OUTPUT_ROOT}" "${TREE_OUTPUT_ROOT}" "${TREE_TXT_OUTPUT_ROOT}"

log_step "repo=${REPO_ROOT}"
log_step "models=${MODELS}"
log_step "limit=${LIMIT}"
log_step "pdf_dir=${PDF_DIR}"
log_step "label_output=${LABEL_OUTPUT_DIR}"
log_step "inference_output=${INFERENCE_OUTPUT_ROOT}"
log_step "inference_raw_output=${INFERENCE_RAW_OUTPUT_ROOT}"
log_step "tree_output=${TREE_OUTPUT_ROOT}"
log_step "tree_txt_output=${TREE_TXT_OUTPUT_ROOT}"

if [[ "${RUN_LABEL_NORMALIZATION}" == "1" ]]; then
  log_step "stage 1/3: label normalization"
  for model in "${MODEL_ARRAY[@]}"; do
    model="${model//[[:space:]]/}"
    [[ -z "${model}" ]] && continue

    input_model_dir="${POST_PROCESS_ROOT}/${model}"
    if [[ ! -d "${input_model_dir}" ]]; then
      echo "[pipeline] missing model input dir: ${input_model_dir}" >&2
      exit 1
    fi

    log_step "normalize model=${model}"
    "${PYTHON_LABEL_BIN}" "${REPO_ROOT}/post_processing/label_normalization.py" \
      --model "${model}" \
      --input-dir "${input_model_dir}" \
      --output-dir "${LABEL_OUTPUT_DIR}" \
      --pdf-dir "${PDF_DIR}" \
      --bbox-scale "${BBOX_SCALE}" \
      --doc-limit "${LIMIT}"
  done
else
  log_step "skip label normalization"
fi

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  log_step "stage 2/3: inference"
  "${PYTHON_BIN}" "${REPO_ROOT}/post_processing/run_inference.py" \
    --models "${MODELS}" \
    --input-dir "${LABEL_OUTPUT_DIR}" \
    --output-root "${INFERENCE_OUTPUT_ROOT}" \
    --raw-output-root "${INFERENCE_RAW_OUTPUT_ROOT}" \
    --limit "${LIMIT}" \
    "$@"
else
  log_step "skip inference"
fi

if [[ "${RUN_BUILD_TREE}" == "1" ]]; then
  log_step "stage 3/3: build tree"
  for model in "${MODEL_ARRAY[@]}"; do
    model="${model//[[:space:]]/}"
    [[ -z "${model}" ]] && continue

    safe_model="$(normalize_model_name "${model}")"
    model_inference_dir="${INFERENCE_OUTPUT_ROOT}/${safe_model}"
    model_tree_dir="${TREE_OUTPUT_ROOT}/${safe_model}"
    model_tree_txt_dir="${TREE_TXT_OUTPUT_ROOT}/${safe_model}"

    if [[ ! -d "${model_inference_dir}" ]]; then
      echo "[pipeline] missing inference dir: ${model_inference_dir}" >&2
      exit 1
    fi

    mkdir -p "${model_tree_dir}" "${model_tree_txt_dir}"
    log_step "build tree model=${model} input=${model_inference_dir}"
    "${PYTHON_TREE_BIN}" "${REPO_ROOT}/post_processing/get_json_tree.py" \
      --input-dir "${model_inference_dir}" \
      --output-dir "${model_tree_dir}" \
      --txt-dir "${model_tree_txt_dir}"
  done
else
  log_step "skip build tree"
fi

log_step "done"
