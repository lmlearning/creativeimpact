from datasets import load_dataset
import json
import os

def setup_gsm8k_dataset():
    output_dir = "data/gsm8k"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gsm8k_test_1k.jsonl")

    print("Loading GSM8K dataset from Hugging Face...")
    try:
        # Load the main GSM8K dataset, test split
        gsm8k_dataset = load_dataset("gsm8k", "main", split="test")
        print(f"Full GSM8K test set loaded. Number of examples: {len(gsm8k_dataset)}")

        # The issue specifies "1k problems".
        # The standard GSM8K test set has 1319 examples.
        # We will take the first 1000 examples for consistency with the "1k" requirement.
        # If a specific 1k subset is defined elsewhere or by the user, this might need adjustment.
        # For now, taking the first 1000.
        num_samples = 1000
        if len(gsm8k_dataset) < num_samples:
            print(f"Warning: Full test set has {len(gsm8k_dataset)} examples, which is less than the requested {num_samples}.")
            sampled_dataset = gsm8k_dataset
        else:
            sampled_dataset = gsm8k_dataset.select(range(num_samples))
            print(f"Selected {len(sampled_dataset)} examples for the 1k subset.")

        # Save the sampled dataset to a JSONL file
        # Each line in the file will be a JSON object representing a question-answer pair.
        print(f"Saving the {len(sampled_dataset)} GSM8K test examples to {output_path}...")
        with open(output_path, 'w') as f:
            for example in sampled_dataset:
                # The dataset contains 'question' and 'answer' fields.
                # The answer field often has a format like "The final answer is \boxed{1234}."
                # We will store them as is, parsing can be done by the scoring script.
                f.write(json.dumps({"question": example["question"], "answer": example["answer"]}) + '\n')

        print(f"Successfully saved GSM8K test subset to {output_path}")
        file_size = os.path.getsize(output_path)
        print(f"File size: {file_size} bytes")

    except Exception as e:
        print(f"Error loading or processing GSM8K dataset: {e}")
        raise

if __name__ == "__main__":
    setup_gsm8k_dataset()
