import argparse
import json
import os

try:
    from datasets import load_dataset
except ImportError:
    print("Error: The 'datasets' library is not installed. Please install it by running:")
    print("pip install datasets")
    # It's better to raise an exception or sys.exit here if datasets is critical.
    # For this script, it is critical.
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download TruthfulQA questions and save them as JSONL.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite the output file if it already exists."
    )
    args = parser.parse_args()

    output_dir = "data/truthfulqa/"
    output_file = os.path.join(output_dir, "truthfulqa_questions.jsonl")

    # Attempt to create the directory. If it exists, no error will be raised.
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}")
        # Depending on the severity, you might want to exit.
        # For now, let's assume if it fails, subsequent file operations will also fail clearly.

    if os.path.exists(output_file) and not args.force:
        print(f"Output file {output_file} already exists. Use --force to overwrite.")
        return

    print("Loading TruthfulQA 'generation' dataset (validation split)...")
    try:
        # The 'generation' config of 'truthful_qa' has a 'validation' split
        # which contains the questions suitable for generation tasks.
        ds = load_dataset("truthful_qa", "generation", split='validation', trust_remote_code=True) # Added trust_remote_code
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        print("Please check your internet connection and if the dataset 'truthful_qa' with 'generation' config is available.")
        print("You might need to ensure 'trust_remote_code=True' is appropriate for this dataset if not set.")
        exit(1)

    print(f"Writing questions to {output_file}...")
    count = 0
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, item in enumerate(ds):
                if "Question" not in item or item["Question"] is None: # Check for None as well
                    print(f"Warning: Item at index {i} does not have a valid 'Question' field. Skipping: {item}")
                    continue

                record = {"Question": item["Question"], "idx": i}
                f.write(json.dumps(record) + '\n')
                count +=1
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")
        exit(1)

    if count > 0:
        print(f"Successfully wrote {count} questions to {output_file}.")
    else:
        print(f"No questions were written to {output_file}. Please check the dataset and input.")


if __name__ == "__main__":
    main()
