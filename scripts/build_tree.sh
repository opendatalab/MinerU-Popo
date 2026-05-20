#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Build JSON trees and text previews from parsed inference outputs.
#
# Inputs:
#   outputs/inference/<model>/<doc_id>.json
#
# Outputs:
#   outputs/build_tree/<model>/<doc_id>.json
#   outputs/build_tree_txt/<model>/<doc_id>.txt
#
# Usage:
#   bash scripts/build_tree.sh
#   MODELS="mineru,dolphin" bash scripts/build_tree.sh
#   CLEAN=1 bash scripts/build_tree.sh

PYTHON_TREE_BIN="${PYTHON_TREE_BIN:-python3}"
INFERENCE_OUTPUT_ROOT="${INFERENCE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/inference}"
TREE_OUTPUT_ROOT="${TREE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/build_tree}"
TREE_TXT_OUTPUT_ROOT="${TREE_TXT_OUTPUT_ROOT:-${REPO_ROOT}/outputs/build_tree_txt}"
MODELS="${MODELS:-}"
CLEAN="${CLEAN:-0}"
FAIL_ON_MISSING="${FAIL_ON_MISSING:-1}"

safe_model_name() {
  local model="$1"
  python3 - "$model" <<'PY'
import re
import sys
print(re.sub(r"[^0-9A-Za-z_.-]+", "_", sys.argv[1]))
PY
}

count_json_files() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo 0
    return
  fi
  find "${dir}" -maxdepth 1 -type f -name "*.json" | wc -l | tr -d ' '
}

count_txt_files() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo 0
    return
  fi
  find "${dir}" -maxdepth 1 -type f -name "*.txt" | wc -l | tr -d ' '
}

echo "[build_tree] repo=${REPO_ROOT}"
echo "[build_tree] inference_input=${INFERENCE_OUTPUT_ROOT}"
echo "[build_tree] tree_output=${TREE_OUTPUT_ROOT}"
echo "[build_tree] tree_txt_output=${TREE_TXT_OUTPUT_ROOT}"
echo "[build_tree] clean=${CLEAN}"

if [[ ! -d "${INFERENCE_OUTPUT_ROOT}" ]]; then
  echo "[build_tree] missing inference root: ${INFERENCE_OUTPUT_ROOT}" >&2
  exit 1
fi

if [[ -z "${MODELS}" ]]; then
  mapfile -t MODEL_ARRAY < <(find "${INFERENCE_OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
else
  IFS=',' read -r -a MODEL_ARRAY <<< "${MODELS}"
fi

if [[ "${#MODEL_ARRAY[@]}" -eq 0 ]]; then
  echo "[build_tree] no model directories found under ${INFERENCE_OUTPUT_ROOT}" >&2
  exit 1
fi

mkdir -p "${TREE_OUTPUT_ROOT}" "${TREE_TXT_OUTPUT_ROOT}"

for model in "${MODEL_ARRAY[@]}"; do
  model="${model//[[:space:]]/}"
  [[ -z "${model}" ]] && continue

  safe_model="$(safe_model_name "${model}")"
  model_inference_dir="${INFERENCE_OUTPUT_ROOT}/${safe_model}"
  model_tree_dir="${TREE_OUTPUT_ROOT}/${safe_model}"
  model_tree_txt_dir="${TREE_TXT_OUTPUT_ROOT}/${safe_model}"

  if [[ ! -d "${model_inference_dir}" ]]; then
    message="[build_tree] missing inference dir: ${model_inference_dir}"
    if [[ "${FAIL_ON_MISSING}" == "1" ]]; then
      echo "${message}" >&2
      exit 1
    fi
    echo "${message}; skip"
    continue
  fi

  input_count="$(count_json_files "${model_inference_dir}")"
  if [[ "${input_count}" == "0" ]]; then
    echo "[build_tree] no inference JSON files for model=${safe_model}; skip"
    continue
  fi

  mkdir -p "${model_tree_dir}" "${model_tree_txt_dir}"
  if [[ "${CLEAN}" == "1" ]]; then
    find "${model_tree_dir}" -maxdepth 1 -type f -name "*.json" -delete
    find "${model_tree_txt_dir}" -maxdepth 1 -type f -name "*.txt" -delete
  fi

  echo "[build_tree] model=${safe_model} input_json=${input_count}"
  "${PYTHON_TREE_BIN}" "${REPO_ROOT}/post_processing/get_json_tree.py" \
    --input-dir "${model_inference_dir}" \
    --output-dir "${model_tree_dir}" \
    --txt-dir "${model_tree_txt_dir}"

  tree_count="$(count_json_files "${model_tree_dir}")"
  txt_count="$(count_txt_files "${model_tree_txt_dir}")"
  echo "[build_tree] done model=${safe_model} tree_json=${tree_count} tree_txt=${txt_count}"
done

echo "[build_tree] all done"
