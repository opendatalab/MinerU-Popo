#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="mineru"
INPUT_DIR="${REPO_ROOT}/post-process/${MODEL}"
OUTPUT_DIR="${REPO_ROOT}/outputs/label_normalization"
PDF_DIR="${REPO_ROOT}/eval_pdf_dir"
DOC_LIMIT="10"

mkdir -p "${OUTPUT_DIR}"

python3 "${REPO_ROOT}/post_processing/label_normalization.py" \
  --model "${MODEL}" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --pdf-dir "${PDF_DIR}" \
  --doc-limit "${DOC_LIMIT}"
