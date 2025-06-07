#!/bin/bash

# Preamble
set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error.
set -o pipefail # Return value of a pipeline is the value of the last command to exit with a non-zero status.

# --- Model Name Argument ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_name>"
    echo "Supported models: gpt-4o, claude-3-5-sonnet-20240620, deepseek-ai/deepseek-r1"
    exit 1
fi

MODEL_NAME="$1"
SUPPORTED_MODELS=("gpt-4o" "claude-3-5-sonnet-20240620" "deepseek-ai/deepseek-r1")

# Validate model name
model_is_supported=false
for supported_model in "${SUPPORTED_MODELS[@]}"; do
    if [ "$MODEL_NAME" == "$supported_model" ]; then
        model_is_supported=true
        break
    fi
done

if [ "$model_is_supported" = false ]; then
    echo "Error: Model '$MODEL_NAME' is not supported."
    echo "Supported models are: ${SUPPORTED_MODELS[*]}"
    exit 1
fi

# --- API Key Prerequisite Message ---
echo "-----------------------------------------------------------------------"
echo "IMPORTANT:"
echo "This script requires API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY,"
echo "REPLICATE_API_TOKEN) to be set as environment variables for the"
echo "selected model. Ensure the relevant key for '$MODEL_NAME' is configured."
echo "-----------------------------------------------------------------------"
# Optional: Add a small delay or a read prompt if you want to ensure user sees this.
# read -p "Press [Enter] to continue if API keys are set..."

# --- Data Setup ---
echo "INFO: Running data setup scripts if necessary..."
python src/domain_scripts/aut_setup.py
echo "INFO: AUT setup script run."

# TruthfulQA setup - this script might not exist yet, which is okay,
# but its output file is checked.
if [ -f "src/domain_scripts/truthfulqa_setup.py" ]; then
    python src/domain_scripts/truthfulqa_setup.py
    echo "INFO: TruthfulQA setup script run."
else
    echo "INFO: src/domain_scripts/truthfulqa_setup.py not found, assuming data exists or is not needed by this script."
fi

TRUTHFULQA_DATA_FILE="data/truthfulqa/truthfulqa_questions.jsonl"
if [ ! -f "$TRUTHFULQA_DATA_FILE" ]; then
    echo "Error: Crucial data file $TRUTHFULQA_DATA_FILE does not exist after setup."
    echo "Please ensure src/domain_scripts/truthfulqa_setup.py runs successfully and creates this file."
    exit 1
fi
echo "INFO: GSM8K and SocialChem data are expected to be present in data/ directory."


# --- Define Directories ---
GENERATED_DATA_DIR="outputs/generated_data"
EVAL_OUTPUT_DIR="outputs/evaluation_results"
SCORERS_DIR="src/eval_scorers" # Used by evaluate.py, not directly here

# Create directories if they don't exist
mkdir -p "$GENERATED_DATA_DIR"
mkdir -p "$EVAL_OUTPUT_DIR"
echo "INFO: Ensured directories $GENERATED_DATA_DIR and $EVAL_OUTPUT_DIR exist."

# --- Run Generation Scripts ---
echo "INFO: Running data generation for model: $MODEL_NAME..."

# AUT
echo "INFO: Generating outputs for AUT..."
python src/domain_scripts/generate_outputs_aut.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$GENERATED_DATA_DIR" \
    --data_file "data/aut_objects.json"

# GSM8K
echo "INFO: Generating outputs for GSM8K..."
python src/domain_scripts/generate_outputs_gsm8k.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$GENERATED_DATA_DIR" \
    --data_file "data/gsm8k/gsm8k_test_1k.jsonl"

# SocialChem
echo "INFO: Generating outputs for SocialChem..."
python src/domain_scripts/generate_outputs_socialchem.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$GENERATED_DATA_DIR" \
    --data_file "data/socialchem/socialchem_sampled_30.jsonl"

# TruthfulQA
echo "INFO: Generating outputs for TruthfulQA..."
python src/domain_scripts/generate_outputs_truthfulqa.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$GENERATED_DATA_DIR" \
    --data_file "$TRUTHFULQA_DATA_FILE"

# --- Run Evaluation Script ---
echo "INFO: Running evaluation for model: $MODEL_NAME..."
python src/evaluate.py \
    --model_name "$MODEL_NAME" \
    --generated_data_dir "$GENERATED_DATA_DIR" \
    --eval_output_dir "$EVAL_OUTPUT_DIR" \
    --scorers_dir "$SCORERS_DIR"

# --- Completion Message ---
# Sanitize model name for filenames (replace / with _)
SANITIZED_MODEL_NAME=$(echo "$MODEL_NAME" | sed 's/\//_/g')

echo "-----------------------------------------------------------------------"
echo "INFO: Experiment run completed for model: $MODEL_NAME."
echo "INFO: Generated data is in $GENERATED_DATA_DIR."
echo "INFO: Evaluation results (including scorer CSVs and summary tables) are in $EVAL_OUTPUT_DIR."
echo "INFO: Main summary files:"
echo "      CSV: $EVAL_OUTPUT_DIR/results_summary_${SANITIZED_MODEL_NAME}.csv"
echo "      MD:  $EVAL_OUTPUT_DIR/results_summary_${SANITIZED_MODEL_NAME}.md"
echo "-----------------------------------------------------------------------"

# Make the script executable by the user who runs this.
# The tool environment might not persist chmod changes effectively across tool calls,
# but it's good practice to include if this script were to be checked out and run manually.
# chmod +x $0
# For the subtask, the file is created. The calling environment would handle execution.

exit 0
