#!/usr/bin/env bash
set -euo pipefail

# Run Abstract keyword extraction with a Tinker OpenAI-compatible chat endpoint.
#
# Usage:
#   TINKER_MODEL="tinker://.../sampler_weights/..." bash scripts/run_tinker_keywords.sh [input_csv] [output_dir]
#
# Required environment:
#   TINKER_API_KEY             Tinker API key.
#   TINKER_MODEL               Tinker sampler checkpoint path, for example tinker://.../sampler_weights/...
#
# Environment overrides:
#   KG_EXTRACT_BIN             Command used to run kg-extract. Default: kg-extract
#   TINKER_API_BASE            Tinker API base URL.
#   TINKER_API_KEY_ENV         Environment variable containing the Tinker API key. Default: TINKER_API_KEY
#   TINKER_MAX_TOKENS          Maximum generated tokens per abstract. Default: 256
#   TINKER_TEMPERATURE         Generation temperature. Default: 0.0
#   TINKER_REASONING_EFFORT    Optional reasoning effort: none, minimal, low, medium, high, or xhigh.
#   KEYWORD_TOP_K              Maximum keywords per award. Default: 8
#   KEYWORD_MIN_SCORE          Minimum keyword score. Default: 0.0
#   KEYWORD_NGRAM_MIN          Minimum n-gram length. Default: 1
#   KEYWORD_NGRAM_MAX          Maximum n-gram length. Default: 2
#   KEYWORD_LIMIT              Optional number of abstracts to process.
#   KEYWORD_CLUSTER            Set to 0 to disable embedding clustering. Default: 1
#   KEYWORD_CLUSTER_MODEL      Embedding model for clustering.
#   KEYWORD_CLUSTER_THRESHOLD  Embedding clustering threshold. Default: 0.88

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_CSV="${1:-data/llm/filtered/awards_start_2024.csv}"

KG_EXTRACT_BIN="${KG_EXTRACT_BIN:-kg-extract}"
TINKER_MODEL="${TINKER_MODEL:-}"
TINKER_API_BASE="${TINKER_API_BASE:-https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1}"
TINKER_API_KEY_ENV="${TINKER_API_KEY_ENV:-TINKER_API_KEY}"
TINKER_MAX_TOKENS="${TINKER_MAX_TOKENS:-256}"
TINKER_TEMPERATURE="${TINKER_TEMPERATURE:-0.0}"
KEYWORD_TOP_K="${KEYWORD_TOP_K:-8}"
KEYWORD_MIN_SCORE="${KEYWORD_MIN_SCORE:-0.0}"
KEYWORD_NGRAM_MIN="${KEYWORD_NGRAM_MIN:-1}"
KEYWORD_NGRAM_MAX="${KEYWORD_NGRAM_MAX:-2}"
KEYWORD_CLUSTER="${KEYWORD_CLUSTER:-1}"
KEYWORD_CLUSTER_MODEL="${KEYWORD_CLUSTER_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
KEYWORD_CLUSTER_THRESHOLD="${KEYWORD_CLUSTER_THRESHOLD:-0.88}"

cd "${PROJECT_ROOT}"

if [[ -z "${TINKER_MODEL}" ]]; then
  echo "TINKER_MODEL is required, for example:" >&2
  echo "  export TINKER_MODEL='tinker://.../sampler_weights/...'" >&2
  exit 1
fi

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "Input CSV does not exist: ${INPUT_CSV}" >&2
  echo "Pass an input path as the first argument, for example:" >&2
  echo "  bash scripts/run_tinker_keywords.sh 'data/Quantum/filtered/awards_start_2024.csv'" >&2
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
  OUTPUT_DIR="output-tinker-keywords/${DATASET_SLUG}/${YEAR_LABEL}"
fi

command_args=(
  "${INPUT_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --keyword-backend tinker
  --keyword-tinker-model "${TINKER_MODEL}"
  --keyword-tinker-api-base "${TINKER_API_BASE}"
  --keyword-tinker-api-key-env "${TINKER_API_KEY_ENV}"
  --keyword-tinker-max-tokens "${TINKER_MAX_TOKENS}"
  --keyword-tinker-temperature "${TINKER_TEMPERATURE}"
  --keyword-top-k "${KEYWORD_TOP_K}"
  --keyword-ngram-min "${KEYWORD_NGRAM_MIN}"
  --keyword-ngram-max "${KEYWORD_NGRAM_MAX}"
  --keyword-min-score "${KEYWORD_MIN_SCORE}"
  --keyword-noun-filter
)

if [[ -n "${TINKER_REASONING_EFFORT:-}" ]]; then
  command_args+=(--keyword-tinker-reasoning-effort "${TINKER_REASONING_EFFORT}")
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
echo "Tinker API base: ${TINKER_API_BASE}"
echo "Tinker model: ${TINKER_MODEL}"
echo
echo "Running Tinker keyword extraction..."

"${KG_EXTRACT_BIN}" "${command_args[@]}"
