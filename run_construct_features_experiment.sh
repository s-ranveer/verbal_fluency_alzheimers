#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

INPUT_ROOT="${INPUT_ROOT:-outputs/new/transcriptions_wo_speakers/year_1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/new/features/year_1}"
AOA_PATH="${AOA_PATH:-data/age_of_acquisition.xlsx}"
AOA_SEC_PATH="${AOA_SEC_PATH:-data/age_of_acquisition_secondary.xlsx}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-0 1 2 3 4}"

read -r -a SEED_ARRAY <<< "$SEEDS"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-${#SEED_ARRAY[@]}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"

if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || (( MAX_PARALLEL_JOBS < 1 )); then
    echo "MAX_PARALLEL_JOBS must be a positive integer. Got: ${MAX_PARALLEL_JOBS}" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

run_seed() {
    local seed="$1"
    local input_dir="${INPUT_ROOT}/seed_${seed}"
    local output_path="${OUTPUT_ROOT}/features_seed_${seed}.csv"
    local binned_output_path="${OUTPUT_ROOT}/features_seed_${seed}_binned.csv"
    local invalidated_words_output_path="${OUTPUT_ROOT}/features_seed_${seed}_invalidated_words.json"

    if [[ ! -d "$input_dir" ]]; then
        echo "Skipping seed ${seed}: input directory not found: ${input_dir}" >&2
        return 0
    fi

    echo "Running construct_features.py with seed ${seed}"
    echo "  input_dir:          ${input_dir}"
    echo "  aoa_path:           ${AOA_PATH}"
    echo "  aoa_sec_path:       ${AOA_SEC_PATH}"
    echo "  output_path:        ${output_path}"
    echo "  binned_output_path: ${binned_output_path}"
    echo "  invalidated_words:  ${invalidated_words_output_path}"

    "$PYTHON_BIN" construct_features.py \
        --input_dir "$input_dir" \
        --aoa_path "$AOA_PATH" \
        --aoa_sec_path "$AOA_SEC_PATH" \
        --output_path "$output_path" \
        --binned_output_path "$binned_output_path" \
        --invalidated_words_output_path "$invalidated_words_output_path"
}

running_jobs=0
failed_jobs=0

echo "Processing seeds in parallel: ${SEEDS}"
echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}"
echo "Per-seed logs: ${LOG_ROOT}"

for seed in "${SEED_ARRAY[@]}"; do
    echo "Starting seed ${seed}"
    run_seed "$seed" > "${LOG_ROOT}/features_seed_${seed}.log" 2>&1 &
    running_jobs=$((running_jobs + 1))

    if (( running_jobs >= MAX_PARALLEL_JOBS )); then
        if ! wait -n; then
            failed_jobs=1
        fi
        running_jobs=$((running_jobs - 1))
    fi
done

while (( running_jobs > 0 )); do
    if ! wait -n; then
        failed_jobs=1
    fi
    running_jobs=$((running_jobs - 1))
done

if (( failed_jobs != 0 )); then
    echo "One or more seeds failed. Check logs in ${LOG_ROOT}." >&2
    exit 1
fi

echo "All seeds completed successfully."
