#!/bin/bash

# Default model name
MODEL="google/flan-t5-large"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) MODEL="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Using model: $MODEL"

# Create output directory
mkdir -p outputs

# Run setup scripts for all domains
echo "Setting up datasets..."
python src/domain_scripts/aut_setup.py
python src/domain_scripts/gsm8k_setup.py
python src/domain_scripts/socialchem_setup.py
python src/domain_scripts/truthfulqa_setup.py
echo "Setup complete."
echo "--------------------"


# Run AUT experiment
echo "Running AUT experiment..."
python src/domain_scripts/generate_outputs_aut.py --model_name "$MODEL" --output_file outputs/aut_outputs.jsonl
python src/evaluate.py --domain aut --input_file outputs/aut_outputs.jsonl
echo "--------------------"

# Run GSM8K experiment
echo "Running GSM8K experiment..."
python src/domain_scripts/generate_outputs_gsm8k.py --model_name "$MODEL" --output_file outputs/gsm8k_outputs.jsonl
python src/evaluate.py --domain gsm8k --input_file outputs/gsm8k_outputs.jsonl
echo "--------------------"

# Run SocialChem experiment
echo "Running SocialChem experiment..."
python src/domain_scripts/generate_outputs_socialchem.py --model_name "$MODEL" --output_file outputs/socialchem_outputs.jsonl
python src/evaluate.py --domain social --input_file outputs/socialchem_outputs.jsonl
echo "--------------------"

# Run TruthfulQA experiment
echo "Running TruthfulQA experiment..."
python src/domain_scripts/generate_outputs_truthfulqa.py --model_name "$MODEL" --output_file outputs/truthfulqa_outputs.jsonl
python src/evaluate.py --domain truthfulqa --input_file outputs/truthfulqa_outputs.jsonl --questions_file data/TruthfulQA.csv
echo "--------------------"

echo "All experiments complete."
