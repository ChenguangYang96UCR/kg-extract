#!/usr/bin/env bash
set -euo pipefail

# Run the recommended Abstract keyword extraction pipeline.
#
# Usage:
#   bash scripts/run_keybert_keywords.sh [input_csv] [output_dir]
#
# Examples:
#   bash scripts/run_keybert_keywords.sh
#   bash scripts/run_keybert_keywords.sh "data/Quantum/filtered/awards_start_2024.csv" output-keybert-clean
#   bash scripts/run_keybert_keywords.sh "data/digital twins/filtered/awards_start_2025_2026.csv"
#
# Environment overrides:
#   KG_EXTRACT_BIN             Command used to run kg-extract. Default: kg-extract
#   KEYWORD_MODEL              KeyBERT sentence-transformers model.
#   KEYWORD_TOP_K              Maximum keywords per award. Default: 8
#   KEYWORD_MIN_SCORE          Minimum keyword score. Default: 0.35
#   KEYWORD_NGRAM_MIN          Minimum n-gram length. Default: 2
#   KEYWORD_NGRAM_MAX          Maximum n-gram length. Default: 3
#   KEYWORD_CLUSTER_THRESHOLD  Embedding clustering threshold. Default: 0.88
#   KEYWORD_LIMIT              Optional number of abstracts to process.
#   KEYWORD_CLUSTER            Set to 0 to disable embedding clustering. Default: 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_CSV="${1:-data/llm/filtered/awards_start_2024.csv}"

KG_EXTRACT_BIN="${KG_EXTRACT_BIN:-kg-extract}"
KEYWORD_MODEL="${KEYWORD_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
KEYWORD_TOP_K="${KEYWORD_TOP_K:-50}"
KEYWORD_MIN_SCORE="${KEYWORD_MIN_SCORE:-0.15}"
KEYWORD_NGRAM_MIN="${KEYWORD_NGRAM_MIN:-2}"
KEYWORD_NGRAM_MAX="${KEYWORD_NGRAM_MAX:-3}"
KEYWORD_CLUSTER_THRESHOLD="${KEYWORD_CLUSTER_THRESHOLD:-0.88}"
KEYWORD_CLUSTER="${KEYWORD_CLUSTER:-1}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "Input CSV does not exist: ${INPUT_CSV}" >&2
  echo "Pass an input path as the first argument, for example:" >&2
  echo "  bash scripts/run_keybert_keywords.sh 'data/llm/filtered/awards_start_2024.csv' output-llm-keywords" >&2
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
  OUTPUT_DIR="output-keybert-clean/${DATASET_SLUG}/${YEAR_LABEL}"
fi

command_args=(
  "${INPUT_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --keyword-backend keybert
  --keyword-model "${KEYWORD_MODEL}"
  --keyword-top-k "${KEYWORD_TOP_K}"
  --keyword-ngram-min "${KEYWORD_NGRAM_MIN}"
  --keyword-ngram-max "${KEYWORD_NGRAM_MAX}"
  --keyword-min-score "${KEYWORD_MIN_SCORE}"
  --keyword-noun-filter
)

if [[ -n "${KEYWORD_LIMIT:-}" ]]; then
  command_args+=(--keyword-limit "${KEYWORD_LIMIT}")
fi

if [[ "${KEYWORD_CLUSTER}" != "0" ]]; then
  command_args+=(
    --keyword-cluster
    --keyword-cluster-threshold "${KEYWORD_CLUSTER_THRESHOLD}"
  )
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Input CSV: ${INPUT_CSV}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Keyword model: ${KEYWORD_MODEL}"
echo
echo "Running keyword extraction..."

"${KG_EXTRACT_BIN}" "${command_args[@]}"
