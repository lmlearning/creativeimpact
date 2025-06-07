import argparse
import pandas as pd
import subprocess
import os
import tempfile

# Define the path to the official (placeholder) evaluation script
OFFICIAL_EVAL_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "truthfulqa_official", "evaluate.py"
)

def run_official_truthfulqa_eval(
    generated_answers_csv_path: str,
    metric: str,
    output_dir: str # Directory for official script's output
) -> pd.DataFrame | None:
    """
    Runs the (placeholder) official TruthfulQA evaluation script.

    Args:
        generated_answers_csv_path: Path to the CSV file with 'question_id' and 'generated_answer'.
        metric: The metric to evaluate (e.g., 'bleurt').
        output_dir: Directory where the official script will save its output.

    Returns:
        A pandas DataFrame with scores, or None if evaluation failed.
    """
    if not os.path.exists(generated_answers_csv_path):
        print(f"Wrapper: Input file for official script not found: {generated_answers_csv_path}")
        return None

    # Define where the official script should save its output
    official_output_filename = f"official_scores_{metric}.csv"
    official_output_path = os.path.join(output_dir, official_output_filename)

    cmd = [
        "python",
        OFFICIAL_EVAL_SCRIPT_PATH,
        "--input_path", generated_answers_csv_path,
        "--output_path", official_output_path,
        "--metrics", metric,
        # Add other necessary args for the official script, even if placeholders ignore them
        "--model_type", "custom_wrapper",
        "--preset", "qa", # common preset
    ]

    print(f"Wrapper: Running command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Wrapper: Official script stdout:")
        print(process.stdout)
        if process.stderr:
            print("Wrapper: Official script stderr:")
            print(process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Wrapper: Error running official TruthfulQA script.")
        print(f"Wrapper: Command: {' '.join(e.cmd)}")
        print(f"Wrapper: Return code: {e.returncode}")
        print(f"Wrapper: stdout: {e.stdout}")
        print(f"Wrapper: stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"Wrapper: Error: Python interpreter or official script not found at {OFFICIAL_EVAL_SCRIPT_PATH}")
        return None


    if os.path.exists(official_output_path):
        try:
            scores_df = pd.read_csv(official_output_path)
            print(f"Wrapper: Successfully read scores from {official_output_path}")
            return scores_df
        except Exception as e:
            print(f"Wrapper: Error reading official script's output CSV {official_output_path} - {e}")
            return None
    else:
        print(f"Wrapper: Official script output file not found: {official_output_path}")
        return None

def main_wrapper(generated_answers_file_path: str, metric: str, temp_base_dir: str) -> list[dict] | None:
    """
    Main wrapper function to evaluate generated answers using the (placeholder) official script.

    Args:
        generated_answers_file_path: Path to the input CSV containing generated answers.
                                     Must have 'question_id' and 'generated_answer'.
        metric: Metric to evaluate (e.g., "bleurt").
        temp_base_dir: Base directory for temporary files and official script outputs.

    Returns:
        A list of dictionaries, each with 'question_id' and 'score', or None.
    """
    print(f"Wrapper: Starting evaluation for file '{generated_answers_file_path}' with metric '{metric}'")

    # The generated_answers_file_path is already assumed to be in the correct format
    # for the placeholder official script (question_id, generated_answer).
    # No transformation needed for this subtask.

    # Directory for official script's output
    official_output_dir = os.path.join(temp_base_dir, "official_output")
    os.makedirs(official_output_dir, exist_ok=True)

    scores_df = run_official_truthfulqa_eval(generated_answers_file_path, metric, official_output_dir)

    if scores_df is not None and not scores_df.empty:
        # Assuming the official script outputs 'question_id' and '{metric}_score' (e.g., 'bleurt_score')
        # The placeholder script already outputs 'bleurt_score' for 'bleurt' metric.
        if f'{metric}_score' not in scores_df.columns:
            print(f"Wrapper: Error - Metric score column '{metric}_score' not found in official script output.")
            if 'question_id' in scores_df.columns and len(scores_df.columns) > 1:
                 # Try to use the first column after question_id if the specific metric score column is not found
                potential_score_col = scores_df.columns[1]
                print(f"Wrapper: Attempting to use column '{potential_score_col}' as score column.")
                scores_df.rename(columns={potential_score_col: f'{metric}_score'}, inplace=True)
            else:
                 return None


        # Convert to list of dicts
        results = []
        for _, row in scores_df.iterrows():
            results.append({
                "question_id": row["question_id"],
                "score": row[f"{metric}_score"]
            })
        print(f"Wrapper: Successfully processed scores. Number of results: {len(results)}")
        return results
    else:
        print("Wrapper: Failed to get scores from official script or scores DataFrame is empty.")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wrapper for TruthfulQA Evaluation")
    parser.add_argument("--input_csv", type=str, help="Path to generated answers CSV file (question_id, generated_answer).")
    parser.add_argument("--metric", type=str, default="bleurt", help="Metric to evaluate (e.g., bleurt).")
    parser.add_argument("--temp_dir", type=str, default=None, help="Directory for temporary files and outputs.")

    args = parser.parse_args()

    # Use a temporary directory if not specified
    if args.temp_dir:
        temp_base_dir = args.temp_dir
        os.makedirs(temp_base_dir, exist_ok=True)
        print(f"Using specified temp directory: {temp_base_dir}")
    else:
        # Create a temporary directory that will be automatically cleaned up
        # For testing, it's better to have a predictable path, so let's default to 'outputs/temp/truthfulqa_wrapper_test'
        # If this script were part of a larger system, a true tempfile.TemporaryDirectory() would be better for __main__
        temp_base_dir = os.path.join("outputs", "temp", "truthfulqa_wrapper_test")
        os.makedirs(temp_base_dir, exist_ok=True)
        print(f"Using default temp directory for this run: {temp_base_dir}")


    if args.input_csv:
        input_generated_answers_file = args.input_csv
        if not os.path.exists(input_generated_answers_file):
            print(f"Error: Provided input CSV '{input_generated_answers_file}' not found.")
            exit(1)
    else:
        print("No input_csv provided, creating a dummy generated_answers.csv for testing.")
        dummy_data = {
            'question_id': [1, 2, 3, 4],
            'generated_answer': [
                "The sky is blue.",
                "Water is wet.",
                "Birds can fly.",
                "The earth is flat." # A less truthful answer
            ]
        }
        dummy_df = pd.DataFrame(dummy_data)
        input_generated_answers_file = os.path.join(temp_base_dir, "dummy_generated_answers.csv")
        dummy_df.to_csv(input_generated_answers_file, index=False)
        print(f"Dummy generated answers saved to: {input_generated_answers_file}")


    print(f"\n--- Running TruthfulQA Wrapper Test (Metric: {args.metric}) ---")

    final_scores = main_wrapper(input_generated_answers_file, args.metric, temp_base_dir)

    if final_scores:
        print("\n--- Wrapper Test Results ---")
        for score_entry in final_scores:
            print(score_entry)
    else:
        print("\n--- Wrapper Test Failed to retrieve scores ---")

    print(f"\n--- TruthfulQA Wrapper Test Finished ---")
    # Note: if temp_base_dir was created with tempfile.TemporaryDirectory(), it would be cleaned up here.
    # Since we are creating it manually for predictability in tests, it will persist.
    # Consider adding cleanup if this path (outputs/temp/...) is not desired long-term.
    print(f"Temporary files (including dummy input and official script output) are in: {temp_base_dir}")
