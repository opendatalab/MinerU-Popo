#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="mineru"
INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/post-process/${MODEL}}"
OUTPUT_DIR="${REPO_ROOT}/outputs/eval/mineru"
BBOX_SCALE="source"
DOC_LIMIT="5"

mkdir -p "${OUTPUT_DIR}"

python3 "${REPO_ROOT}/eval/evaluate.py" \
  --model "${MODEL}" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --bbox-scale "${BBOX_SCALE}" \
  --doc-limit "${DOC_LIMIT}"
