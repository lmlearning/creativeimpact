# Creative Impact of Language Models

This repository contains code to evaluate the "creative impact" of language models across a variety of tasks. The framework is designed to be extensible to new models, datasets, and evaluation metrics.

## Setup

1.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Download datasets:**
    The setup scripts will download the necessary datasets.
    ```bash
    python src/domain_scripts/aut_setup.py
    python src/domain_scripts/gsm8k_setup.py
    python src/domain_scripts/socialchem_setup.py
    python src/domain_scripts/truthfulqa_setup.py
    ```

## Running Experiments

The main script to run all experiments is `run_experiments.sh`. You can specify a model to use. The default is `google/flan-t5-large`.

```bash
bash run_experiments.sh --model_name google/flan-t5-base
```

This will:
1.  Generate model outputs for each task and save them to the `outputs/` directory.
2.  Evaluate the generated outputs using the corresponding scorer.
3.  Print the evaluation scores to the console.

### Individual Tasks

You can also run generation and evaluation for each task individually.

#### 1. Alternate Uses Test (AUT) - Divergent Thinking

This task evaluates a model's ability to come up with creative, alternative uses for common objects.

* **Generate:**
    ```bash
    python src/domain_scripts/generate_outputs_aut.py --model_name <your_model> --output_file outputs/aut_outputs.jsonl
    ```
* **Evaluate:** The score is the average number of unique uses generated per object.
    ```bash
    python src/evaluate.py --domain aut --input_file outputs/aut_outputs.jsonl
    ```

#### 2. GSM8K - Mathematical Reasoning

This task tests grade-school mathematical reasoning.

* **Generate:**
    ```bash
    python src/domain_scripts/generate_outputs_gsm8k.py --model_name <your_model> --output_file outputs/gsm8k_outputs.jsonl
    ```
* **Evaluate:** The score is the exact match accuracy on the final numerical answer.
    ```bash
    python src/evaluate.py --domain gsm8k --input_file outputs/gsm8k_outputs.jsonl
    ```

#### 3. SocialChem 101 - Social Norms

This task evaluates the model's understanding of basic social norms.

* **Generate:**
    ```bash
    python src/domain_scripts/generate_outputs_socialchem.py --model_name <your_model> --output_file outputs/socialchem_outputs.jsonl
    ```
* **Evaluate:** The score is the classification accuracy.
    ```bash
    python src/evaluate.py --domain social --input_file outputs/socialchem_outputs.jsonl
    ```

#### 4. TruthfulQA - Truthfulness

This task measures whether a language model is truthful in generating answers to questions.

* **Generate:**
    ```bash
    python src/domain_scripts/generate_outputs_truthfulqa.py --model_name <your_model> --output_file outputs/truthfulqa_outputs.jsonl
    ```
* **Evaluate:** Uses the official TruthfulQA metrics (BLEU, ROUGE, etc.) to compare generated answers against known correct and incorrect answers.
    ```bash
    python src/evaluate.py --domain truthfulqa --input_file outputs/truthfulqa_outputs.jsonl --questions_file data/TruthfulQA.csv
    ```

## Project Structure

-   `data/`: Stores downloaded datasets.
-   `outputs/`: Default directory for model-generated outputs.
-   `src/`: Main source code.
    -   `domain_scripts/`: Scripts for data setup and model output generation for each task.
    -   `eval_scorers/`: Evaluation scripts for each task.
    -   `utils/`: Utility functions, including the LLM handler.
-   `run_experiments.sh`: Main script to run all experiments.
-   `evaluate.py`: Main script to run evaluation on generated outputs.
-   `predict.py`: Script for generating a prediction for a single input.
