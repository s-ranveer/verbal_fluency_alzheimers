#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SEED="${SEED:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLM_OUTPUT_DIR="${LLM_OUTPUT_DIR:-outputs/new/transcriptions_wo_speakers/year_1/seed_${SEED}}"
SYMBOLIC_FEEDBACK_FILE="${SYMBOLIC_FEEDBACK_FILE:-outputs/new/features/year_1/features_seed_${SEED}_invalidated_words.json}"
CORRECTED_OUTPUT_DIR="${CORRECTED_OUTPUT_DIR:-outputs/new/corrections/year_1/seed_${SEED}}"
PROMPT_TEMPLATE_FILE="${PROMPT_TEMPLATE_FILE:-prompts/reprompt_for_corrections.md}"
DEBUG_SINGLE_CASE="${DEBUG_SINGLE_CASE:-0}"
DEBUG_PATIENT_ID="${DEBUG_PATIENT_ID:-}"
BUILD_DATASET_ONLY="${BUILD_DATASET_ONLY:-0}"

if [[ ! -d "$LLM_OUTPUT_DIR" ]]; then
    echo "LLM_OUTPUT_DIR not found: $LLM_OUTPUT_DIR" >&2
    exit 1
fi

if [[ ! -f "$SYMBOLIC_FEEDBACK_FILE" ]]; then
    echo "SYMBOLIC_FEEDBACK_FILE not found: $SYMBOLIC_FEEDBACK_FILE" >&2
    exit 1
fi

if [[ ! -f "$PROMPT_TEMPLATE_FILE" ]]; then
    echo "PROMPT_TEMPLATE_FILE not found: $PROMPT_TEMPLATE_FILE" >&2
    exit 1
fi

echo "Testing corrections_using_symbolic_feedback.py"
echo "  seed:                   ${SEED}"
echo "  llm_output_dir:         ${LLM_OUTPUT_DIR}"
echo "  symbolic_feedback_file: ${SYMBOLIC_FEEDBACK_FILE}"
echo "  corrected_output_dir:   ${CORRECTED_OUTPUT_DIR}"
echo "  prompt_template_file:   ${PROMPT_TEMPLATE_FILE}"

cmd=(
    "$PYTHON_BIN" corrections_using_symbolic_feedback.py
    --llm_output_dir "$LLM_OUTPUT_DIR"
    --symbolic_feedback_file "$SYMBOLIC_FEEDBACK_FILE"
    --corrected_output_dir "$CORRECTED_OUTPUT_DIR"
    --prompt_template_file "$PROMPT_TEMPLATE_FILE"
)

if [[ "$DEBUG_SINGLE_CASE" == "1" ]]; then
    cmd+=(--debug_single_case)
    if [[ -n "$DEBUG_PATIENT_ID" ]]; then
        cmd+=(--debug_patient_id "$DEBUG_PATIENT_ID")
    fi
fi

if [[ "$BUILD_DATASET_ONLY" == "1" ]]; then
    cmd+=(--build_dataset_only)
fi

"${cmd[@]}"
