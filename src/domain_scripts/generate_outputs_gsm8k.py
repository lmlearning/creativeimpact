import argparse
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    # Assuming scripts are run from the repository root or src is in PYTHONPATH
    from src.utils.llm_handler import generate_prompt, get_llm_response
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    try:
        from src.utils.llm_handler import generate_prompt, get_llm_response
    except ImportError:
        print("Error: llm_handler.py not found. Ensure 'src' is in PYTHONPATH or run from the repository root.")
        sys.exit(1)

SUPPORTED_MODELS = ["o3", "claude-3-5-sonnet-20240620", "deepseek-ai/deepseek-r1"]
DOMAIN_NAME = "GSM8K"

def main():
    parser = argparse.ArgumentParser(description=f"Generate LLM outputs for {DOMAIN_NAME} items.")
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
        default="data/gsm8k/gsm8k_test_1k.jsonl",
        help=f"Path to the JSONL file containing {DOMAIN_NAME} data."
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Maximum number of items to process from the data file (optional)."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    items = []
    try:
        with open(args.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                items.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Data file {args.data_file} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from a line in {args.data_file}. Error: {e}")
        sys.exit(1)

    if not items:
        print("No items found in the data file. Exiting.")
        return

    items_to_process = items[:args.max_items] if args.max_items is not None and args.max_items > 0 else items
    total_items_to_process = len(items_to_process)
    print(f"Processing {total_items_to_process} {DOMAIN_NAME} items using model: {args.model_name}")

    results = []
    for index, item in enumerate(items_to_process):
        item_id = f"{DOMAIN_NAME.lower()}_item_{index}" # Default item_id using index
        # Potentially override with a specific ID field if available, e.g. item.get('id') or item.get('idx')
        # For GSM8K, 'question' is the primary text.
        if 'question' not in item:
            print(f"  Skipping item index {index} due to missing 'question' field: {item}")
            results.append({
                "item_id": item_id,
                "domain": DOMAIN_NAME,
                "model_name": args.model_name,
                "original_item_dict": item,
                "error": "Missing 'question' field in original item."
            })
            continue

        item_text = item['question']
        print(f"Processing item {index + 1} of {total_items_to_process}: (ID: {item_id}) '{item_text[:100]}...'")

        try:
            plain_prompt = generate_prompt(item_text, DOMAIN_NAME, "plain")
            plain_response = get_llm_response(plain_prompt, args.model_name)

            creative_prompt = generate_prompt(item_text, DOMAIN_NAME, "creative")
            creative_response = get_llm_response(creative_prompt, args.model_name)

            results.append({
                "item_id": item_id,
                "domain": DOMAIN_NAME,
                "model_name": args.model_name,
                "original_item_dict": item,
                "plain_prompt": plain_prompt,
                "plain_response": plain_response,
                "creative_prompt": creative_prompt,
                "creative_response": creative_response
            })
        except ValueError as ve:
            print(f"  Skipping item ID '{item_id}' due to API key/configuration error: {ve}")
            results.append({
                "item_id": item_id,
                "domain": DOMAIN_NAME,
                "model_name": args.model_name,
                "original_item_dict": item,
                "error": str(ve)
            })
        except Exception as e:
            print(f"  Skipping item ID '{item_id}' due to an unexpected error: {e}")
            results.append({
                "item_id": item_id,
                "domain": DOMAIN_NAME,
                "model_name": args.model_name,
                "original_item_dict": item,
                "error": f"Unexpected error: {str(e)}"
            })

    sanitized_model_name = args.model_name.replace("/", "_")
    output_filename = f"generated_{DOMAIN_NAME.lower()}_{sanitized_model_name}.json"
    output_filepath = os.path.join(args.output_dir, output_filename)

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully generated outputs for {DOMAIN_NAME} to {output_filepath}")
    except IOError as e:
        print(f"Error: Could not write results to {output_filepath}. Error: {e}")

if __name__ == "__main__":
    main()
