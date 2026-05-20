#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Full inference entrypoint.
#
# Outputs:
#   normalized labels: outputs/label_normalization/<model>/<doc_id>.json
#   parsed inference:  outputs/inference/<model>/<doc_id>.json
#   raw LLM records:   outputs/inference_raw/<model>/<doc_id>/*.json
#
# Common usage:
#   bash scripts/run_inference.sh
#   LIMIT=10 MODELS="mineru,dolphin" bash scripts/run_inference.sh --resume
#   RUN_LABEL_NORMALIZATION=0 bash scripts/run_inference.sh --resume

PYTHON_BIN="${PYTHON_BIN:-/data/conda/envs/qwen3vl/bin/python}"
MODELS="${MODELS:-glm-ocr,mineru,monkeyocr,dolphin,PaddleOCR-VL-1.5}"
POST_PROCESS_ROOT="${POST_PROCESS_ROOT:-${REPO_ROOT}/post-process}"
PDF_DIR="${PDF_DIR:-eval_pdf_dir}"
INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/outputs/label_normalization}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/inference}"
RAW_OUTPUT_ROOT="${RAW_OUTPUT_ROOT:-${REPO_ROOT}/outputs/inference_raw}"
LIMIT="${LIMIT:-10}"
RUN_LABEL_NORMALIZATION="${RUN_LABEL_NORMALIZATION:-1}"
BBOX_SCALE="${BBOX_SCALE:-source}"

export POPO_INFERENCE_BACKEND="${POPO_INFERENCE_BACKEND:-transformers}"
export POPO_MODEL_PATH="${POPO_MODEL_PATH:-popo_model}"
export POPO_MAX_NEW_TOKENS="${POPO_MAX_NEW_TOKENS:-1024}"

mkdir -p "${INPUT_DIR}" "${OUTPUT_ROOT}" "${RAW_OUTPUT_ROOT}"

echo "[run_inference] repo=${REPO_ROOT}"
echo "[run_inference] models=${MODELS}"
echo "[run_inference] limit=${LIMIT}"
echo "[run_inference] run_label_normalization=${RUN_LABEL_NORMALIZATION}"
echo "[run_inference] label_output=${INPUT_DIR}"
echo "[run_inference] inference_output=${OUTPUT_ROOT}"
echo "[run_inference] raw_output=${RAW_OUTPUT_ROOT}"
echo "[run_inference] pdf_dir=${PDF_DIR}"

if [[ "${RUN_LABEL_NORMALIZATION}" == "1" ]]; then
  IFS=',' read -r -a MODEL_ARRAY <<< "${MODELS}"
  for model in "${MODEL_ARRAY[@]}"; do
    model="${model//[[:space:]]/}"
    [[ -z "${model}" ]] && continue
    input_model_dir="${POST_PROCESS_ROOT}/${model}"
    if [[ ! -d "${input_model_dir}" ]]; then
      echo "missing model input dir: ${input_model_dir}" >&2
      exit 1
    fi
    echo "[run_inference] normalize model=${model}"
    python3 "${REPO_ROOT}/post_processing/label_normalization.py" \
      --model "${model}" \
      --input-dir "${input_model_dir}" \
      --output-dir "${INPUT_DIR}" \
      --pdf-dir "${PDF_DIR}" \
      --bbox-scale "${BBOX_SCALE}" \
      --doc-limit "${LIMIT}"
  done
else
  echo "[run_inference] skip label normalization"
fi

echo "[run_inference] start inference"
"${PYTHON_BIN}" "${REPO_ROOT}/post_processing/run_inference.py" \
  --models "${MODELS}" \
  --input-dir "${INPUT_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --raw-output-root "${RAW_OUTPUT_ROOT}" \
  --limit "${LIMIT}" \
  "$@"
