#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TRANSCRIPT_DIR="${TRANSCRIPT_DIR:-data/transcriptions_wo_speakers/year_1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/new/transcriptions_wo_speakers/year_1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_TOKENS="${MAX_TOKENS:-6000}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-0 1 2 3 4}"

read -r -a SEED_ARRAY <<< "$SEEDS"

for seed in "${SEED_ARRAY[@]}"; do
    output_dir="${OUTPUT_ROOT}/seed_${seed}"

    echo "Running process_data_batching.py with seed ${seed}"
    echo "  transcript_dir: ${TRANSCRIPT_DIR}"
    echo "  output_dir:     ${output_dir}"

    cmd=(
        "$PYTHON_BIN" process_data_batching.py
        --transcript_dir "$TRANSCRIPT_DIR"
        --output_dir "$output_dir"
        --batch_size "$BATCH_SIZE"
        --max_tokens "$MAX_TOKENS"
        --seed "$seed"
    )

    if [[ -n "$MAX_MODEL_LEN" ]]; then
        cmd+=(--max_model_len "$MAX_MODEL_LEN")
    fi

    "${cmd[@]}"
done
