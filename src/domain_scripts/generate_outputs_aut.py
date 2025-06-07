import argparse
import json
import os
import sys

try:
    # Assuming scripts are run from the repository root or src is in PYTHONPATH
    from src.utils.llm_handler import generate_prompt, get_llm_response
except ImportError:
    # Fallback for direct script execution if src is not in PYTHONPATH
    # This allows running `python src/domain_scripts/generate_outputs_aut.py ...`
    # from the repo root.
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    try:
        from src.utils.llm_handler import generate_prompt, get_llm_response
    except ImportError:
        print("Error: llm_handler.py not found. Ensure 'src' is in PYTHONPATH or run from the repository root.")
        sys.exit(1)

SUPPORTED_MODELS = ["gpt-4o", "claude-3-5-sonnet-20240620", "deepseek-ai/deepseek-r1"]

def main():
    parser = argparse.ArgumentParser(description="Generate LLM outputs for AUT (Alternate Uses Task) items.")
    parser.add_argument(
        "--model_name",
        required=True,
        choices=SUPPORTED_MODELS,
        help="Name of the LLM to use."
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/generated_data",
        help="Directory to save the generated output files."
    )
    parser.add_argument(
        "--data_file",
        default="data/aut_objects.json",
        help="Path to the JSON file containing AUT objects (e.g., {'objects': ['pen', 'brick']})."
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Maximum number of items to process from the data file (optional)."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    try:
        with open(args.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "objects" not in data or not isinstance(data["objects"], list):
            print(f"Error: Data file {args.data_file} must contain a JSON object with an 'objects' key holding a list of strings.")
            sys.exit(1)
        items = data['objects']
    except FileNotFoundError:
        print(f"Error: Data file {args.data_file} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.data_file}.")
        sys.exit(1)

    if not items:
        print("No items found in the data file. Exiting.")
        return

    # Limit items if max_items is set
    if args.max_items is not None and args.max_items > 0:
        items_to_process = items[:args.max_items]
    else:
        items_to_process = items

    total_items_to_process = len(items_to_process)
    print(f"Processing {total_items_to_process} AUT items using model: {args.model_name}")

    results = []
    for index, item_text in enumerate(items_to_process):
        print(f"Processing item {index + 1} of {total_items_to_process}: '{item_text}'...")

        try:
            # Generate plain prompt and get response
            plain_prompt = generate_prompt(item_text, "AUT", "plain")
            plain_response = get_llm_response(plain_prompt, args.model_name)

            # Generate creative prompt and get response
            creative_prompt = generate_prompt(item_text, "AUT", "creative")
            creative_response = get_llm_response(creative_prompt, args.model_name)

            results.append({
                "item_id": item_text,  # Using the item text itself as a simple ID for AUT
                "domain": "AUT",
                "model_name": args.model_name,
                "original_item": item_text,
                "plain_prompt": plain_prompt,
                "plain_response": plain_response,
                "creative_prompt": creative_prompt,
                "creative_response": creative_response
            })
        except ValueError as ve:
            print(f"  Skipping item '{item_text}' due to API key/configuration error: {ve}")
            # Optionally, log this to a separate error file or add a partial result
            results.append({
                "item_id": item_text,
                "domain": "AUT",
                "model_name": args.model_name,
                "original_item": item_text,
                "error": str(ve)
            })
        except Exception as e:
            print(f"  Skipping item '{item_text}' due to an unexpected error: {e}")
            # Optionally, log this
            results.append({
                "item_id": item_text,
                "domain": "AUT",
                "model_name": args.model_name,
                "original_item": item_text,
                "error": f"Unexpected error: {str(e)}"
            })


    # Save results
    # Sanitize model name for filename (e.g., replace "/" with "_")
    sanitized_model_name = args.model_name.replace("/", "_")
    output_filename = f"generated_aut_{sanitized_model_name}.json"
    output_filepath = os.path.join(args.output_dir, output_filename)

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully generated outputs to {output_filepath}")
    except IOError as e:
        print(f"Error: Could not write results to {output_filepath}. Error: {e}")

if __name__ == "__main__":
    main()
