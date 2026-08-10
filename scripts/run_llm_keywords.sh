#!/usr/bin/env bash
set -euo pipefail

# Run Abstract keyword extraction with a local Hugging Face causal LLM.
#
# Usage:
#   bash scripts/run_llm_keywords.sh [input_csv] [output_dir]
#
# Examples:
#   bash scripts/run_llm_keywords.sh
#   bash scripts/run_llm_keywords.sh "data/Quantum/filtered/awards_start_2024.csv"
#   bash scripts/run_llm_keywords.sh "data/digital twins/filtered/awards_start_2025_2026.csv"
#   bash scripts/run_llm_keywords.sh "data/llm/filtered/awards_start_2024.csv" output-custom
#
# Environment overrides:
#   KG_EXTRACT_BIN             Command used to run kg-extract. Default: kg-extract
#   LLM_MODEL_PATH             Local model directory. Default: /data/cyang314/kg
#   LLM_REPO_ID                Hugging Face repo used when model files are missing.
#   LLM_REVISION               Optional Hugging Face revision, branch, or commit.
#   LLM_NO_DOWNLOAD            Set to 1 to fail instead of downloading missing model files.
#   LLM_MAX_NEW_TOKENS         Maximum generated tokens per abstract. Default: 256
#   LLM_TEMPERATURE            Generation temperature. Default: 0.0
#   LLM_DEVICE_MAP             Transformers device_map. Default: auto
#   KEYWORD_TOP_K              Maximum keywords per award. Default: 8
#   KEYWORD_MIN_SCORE          Minimum keyword score. Default: 0.0
#   KEYWORD_NGRAM_MIN          Minimum n-gram length. Default: 2
#   KEYWORD_NGRAM_MAX          Maximum n-gram length. Default: 3
#   KEYWORD_LIMIT              Optional number of abstracts to process.
#   KEYWORD_CLUSTER            Set to 0 to disable embedding clustering. Default: 1
#   KEYWORD_CLUSTER_MODEL      Embedding model for clustering.
#   KEYWORD_CLUSTER_THRESHOLD  Embedding clustering threshold. Default: 0.88

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_CSV="${1:-data/llm/filtered/awards_start_2024.csv}"

KG_EXTRACT_BIN="${KG_EXTRACT_BIN:-kg-extract}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-/data/cyang314/kg}"
LLM_REPO_ID="${LLM_REPO_ID:-deepseek-ai/dspark_qwen3_8b_block7}"
LLM_MAX_NEW_TOKENS="${LLM_MAX_NEW_TOKENS:-256}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
LLM_DEVICE_MAP="${LLM_DEVICE_MAP:-auto}"
KEYWORD_TOP_K="${KEYWORD_TOP_K:-8}"
KEYWORD_MIN_SCORE="${KEYWORD_MIN_SCORE:-0.0}"
KEYWORD_NGRAM_MIN="${KEYWORD_NGRAM_MIN:-2}"
KEYWORD_NGRAM_MAX="${KEYWORD_NGRAM_MAX:-3}"
KEYWORD_CLUSTER="${KEYWORD_CLUSTER:-1}"
KEYWORD_CLUSTER_MODEL="${KEYWORD_CLUSTER_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
KEYWORD_CLUSTER_THRESHOLD="${KEYWORD_CLUSTER_THRESHOLD:-0.88}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "Input CSV does not exist: ${INPUT_CSV}" >&2
  echo "Pass an input path as the first argument, for example:" >&2
  echo "  bash scripts/run_llm_keywords.sh 'data/Quantum/filtered/awards_start_2024.csv'" >&2
  exit 1
fi

slugify() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  value="$(printf '%s' "${value}" | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
  printf '%s' "${value:-dataset}"
}

infer_dataset_name() {
  local path="$1"
  local relative_path="${path#./}"
  if [[ "${relative_path}" == data/* ]]; then
    relative_path="${relative_path#data/}"
    printf '%s' "${relative_path%%/*}"
    return
  fi
  basename "$(dirname "$(dirname "${path}")")"
}

infer_year_label() {
  local filename
  filename="$(basename "$1")"
  if [[ "${filename}" =~ ([0-9]{4}(_[0-9]{4})?) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return
  fi
  printf 'unknown-year'
}

if [[ $# -ge 2 ]]; then
  OUTPUT_DIR="$2"
else
  DATASET_NAME="$(infer_dataset_name "${INPUT_CSV}")"
  DATASET_SLUG="$(slugify "${DATASET_NAME}")"
  YEAR_LABEL="$(infer_year_label "${INPUT_CSV}")"
  OUTPUT_DIR="output-llm-keywords/${DATASET_SLUG}/${YEAR_LABEL}"
fi

command_args=(
  "${INPUT_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --keyword-backend llm
  --keyword-llm-model-path "${LLM_MODEL_PATH}"
  --keyword-llm-repo-id "${LLM_REPO_ID}"
  --keyword-llm-max-new-tokens "${LLM_MAX_NEW_TOKENS}"
  --keyword-llm-temperature "${LLM_TEMPERATURE}"
  --keyword-llm-device-map "${LLM_DEVICE_MAP}"
  --keyword-top-k "${KEYWORD_TOP_K}"
  --keyword-ngram-min "${KEYWORD_NGRAM_MIN}"
  --keyword-ngram-max "${KEYWORD_NGRAM_MAX}"
  --keyword-min-score "${KEYWORD_MIN_SCORE}"
  --keyword-noun-filter
)

if [[ -n "${LLM_REVISION:-}" ]]; then
  command_args+=(--keyword-llm-revision "${LLM_REVISION}")
fi

if [[ "${LLM_NO_DOWNLOAD:-0}" == "1" ]]; then
  command_args+=(--keyword-llm-no-download)
fi

if [[ -n "${KEYWORD_LIMIT:-}" ]]; then
  command_args+=(--keyword-limit "${KEYWORD_LIMIT}")
fi

if [[ "${KEYWORD_CLUSTER}" != "0" ]]; then
  command_args+=(
    --keyword-cluster
    --keyword-cluster-model "${KEYWORD_CLUSTER_MODEL}"
    --keyword-cluster-threshold "${KEYWORD_CLUSTER_THRESHOLD}"
  )
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Input CSV: ${INPUT_CSV}"
echo "Output directory: ${OUTPUT_DIR}"
echo "LLM model path: ${LLM_MODEL_PATH}"
echo "LLM repo id: ${LLM_REPO_ID}"
echo
echo "Running LLM keyword extraction..."

"${KG_EXTRACT_BIN}" "${command_args[@]}"
