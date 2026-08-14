#!/usr/bin/env bash
set -euo pipefail

# Run KGGen Abstract triple extraction with an optional LLM node cleaner.
#
# Usage:
#   bash scripts/run_kggen_node_cleaner.sh [input_csv] [output_dir]
#
# Examples:
#   ABSTRACT_LIMIT=5 bash scripts/run_kggen_node_cleaner.sh
#   bash scripts/run_kggen_node_cleaner.sh "data/Quantum/filtered/awards_start_2024.csv"
#   bash scripts/run_kggen_node_cleaner.sh "data/digital twins/filtered/awards_start_2025_2026.csv" output-custom
#
# Ollama example:
#   ollama pull deepseek-r1:32b
#   ollama pull deepseek-r1:14b
#   ABSTRACT_LIMIT=5 bash scripts/run_kggen_node_cleaner.sh
#
# Tinker node-cleaner example:
#   export TINKER_API_KEY="..."
#   export TINKER_MODEL="tinker://.../sampler_weights/..."
#   NODE_CLEANER_MODEL="openai/${TINKER_MODEL}" \
#   NODE_CLEANER_API_BASE="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1" \
#   NODE_CLEANER_API_KEY_ENV="TINKER_API_KEY" \
#   ABSTRACT_LIMIT=5 bash scripts/run_kggen_node_cleaner.sh
#
# Environment overrides:
#   PYTHON                        Python executable. Default: python
#   KG_EXTRACT_BIN                Command used to run kg-extract. Default: python -m kg_extract
#   KGGEN_MODEL                   KGGen model. Default: ollama_chat/deepseek-r1:32b
#   KGGEN_API_BASE                Optional KGGen provider base URL.
#   KGGEN_API_KEY_ENV             Optional environment variable containing the KGGen provider API key.
#   KGGEN_CHUNK_SIZE              KGGen chunk size in characters. Default: 5000
#   KGGEN_DEDUPLICATE             Set to 1 to enable KGGen built-in deduplication. Default: 0
#   NODE_CLEANER                  Set to 0 to disable the LLM node cleaner. Default: 1
#   NODE_CLEANER_MODEL            Node cleaner model. Default: ollama_chat/deepseek-r1:14b
#   NODE_CLEANER_API_BASE         Node cleaner LiteLLM API base. Default: http://localhost:11434
#   NODE_CLEANER_API_KEY_ENV      Optional environment variable containing the node cleaner API key.
#   NODE_CLEANER_MAX_TOKENS       Maximum generated tokens for node cleaning. Default: 1024
#   NODE_CLEANER_TEMPERATURE      Node cleaner sampling temperature. Default: 0.0
#   ABSTRACT_LIMIT                Optional number of abstracts to process.
#   MIN_CONFIDENCE                Minimum relation confidence. Default: 0.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_CSV="${1:-data/llm/filtered/awards_start_2024.csv}"

PYTHON="${PYTHON:-python}"
KG_EXTRACT_BIN="${KG_EXTRACT_BIN:-${PYTHON} -m kg_extract}"
KGGEN_MODEL="${KGGEN_MODEL:-ollama_chat/deepseek-r1:32b}"
KGGEN_CHUNK_SIZE="${KGGEN_CHUNK_SIZE:-5000}"
KGGEN_DEDUPLICATE="${KGGEN_DEDUPLICATE:-0}"
NODE_CLEANER="${NODE_CLEANER:-1}"
NODE_CLEANER_MODEL="${NODE_CLEANER_MODEL:-ollama_chat/deepseek-r1:14b}"
NODE_CLEANER_API_BASE="${NODE_CLEANER_API_BASE:-http://localhost:11434}"
NODE_CLEANER_MAX_TOKENS="${NODE_CLEANER_MAX_TOKENS:-1024}"
NODE_CLEANER_TEMPERATURE="${NODE_CLEANER_TEMPERATURE:-0.0}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.0}"

cd "${PROJECT_ROOT}"

# Prefer the current checkout over any previously installed console script.
# This matters because kg-extract and the vendored src/kg-gen package are both
# under active local development in this project.
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/src/kg-gen/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "Input CSV does not exist: ${INPUT_CSV}" >&2
  echo "Pass an input path as the first argument, for example:" >&2
  echo "  bash scripts/run_kggen_node_cleaner.sh 'data/Quantum/filtered/awards_start_2024.csv'" >&2
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
  OUTPUT_DIR="output-kggen-node-cleaner/${DATASET_SLUG}/${YEAR_LABEL}"
fi

command_args=(
  "${INPUT_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --abstract-backend kggen
  --abstract-model "${KGGEN_MODEL}"
  --chunk-size "${KGGEN_CHUNK_SIZE}"
  --min-confidence "${MIN_CONFIDENCE}"
)

if [[ -n "${KGGEN_API_BASE:-}" ]]; then
  command_args+=(--base-url "${KGGEN_API_BASE}")
fi

if [[ -n "${KGGEN_API_KEY_ENV:-}" ]]; then
  command_args+=(--api-key-env "${KGGEN_API_KEY_ENV}")
fi

if [[ -n "${ABSTRACT_LIMIT:-}" ]]; then
  command_args+=(--abstract-limit "${ABSTRACT_LIMIT}")
fi

if [[ "${KGGEN_DEDUPLICATE}" == "1" ]]; then
  command_args+=(--kggen-deduplicate)
fi

if [[ "${NODE_CLEANER}" != "0" ]]; then
  command_args+=(
    --abstract-node-cleaner llm
    --abstract-node-cleaner-model "${NODE_CLEANER_MODEL}"
    --abstract-node-cleaner-api-base "${NODE_CLEANER_API_BASE}"
    --abstract-node-cleaner-max-tokens "${NODE_CLEANER_MAX_TOKENS}"
    --abstract-node-cleaner-temperature "${NODE_CLEANER_TEMPERATURE}"
  )
  if [[ -n "${NODE_CLEANER_API_KEY_ENV:-}" ]]; then
    command_args+=(--abstract-node-cleaner-api-key-env "${NODE_CLEANER_API_KEY_ENV}")
  fi
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Input CSV: ${INPUT_CSV}"
echo "Output directory: ${OUTPUT_DIR}"
echo "KGGen model: ${KGGEN_MODEL}"
echo "KGGen chunk size: ${KGGEN_CHUNK_SIZE}"
echo "Python: ${PYTHON}"
echo "PYTHONPATH head: ${PROJECT_ROOT}/src:${PROJECT_ROOT}/src/kg-gen/src"
if [[ -n "${ABSTRACT_LIMIT:-}" ]]; then
  echo "Abstract limit: ${ABSTRACT_LIMIT}"
else
  echo "Abstract limit: all"
fi
if [[ "${NODE_CLEANER}" != "0" ]]; then
  echo "Node cleaner model: ${NODE_CLEANER_MODEL}"
  echo "Node cleaner API base: ${NODE_CLEANER_API_BASE}"
else
  echo "Node cleaner: disabled"
fi
echo

"${PYTHON}" -c 'import inspect
import kg_extract
from kg_gen import KGGen
print(f"kg_extract import: {kg_extract.__file__}")
print(f"KGGen.generate supports no_dspy: {'no_dspy' in inspect.signature(KGGen.generate).parameters}")' || {
  echo "Could not inspect kg_extract/kg_gen imports. Continuing to run extraction..." >&2
}

read -r -a kg_extract_command <<< "${KG_EXTRACT_BIN}"

echo "Running KGGen Abstract extraction..."

"${kg_extract_command[@]}" "${command_args[@]}"
