import json
import os
import sys
import time
import pandas as pd # For CSV/TSV handling

# Add src/utils to Python path to import llm_handler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))) # Adjusted path for src.utils
try:
    from src.utils import llm_handler # Assuming __init__.py in src and src/utils
except ImportError as e:
    print(f"Error importing llm_handler: {e}")
    print("Make sure llm_handler.py is in src/utils/ and src/utils contains __init__.py")
    print("Also ensure src contains __init__.py to be treated as a package.")
    sys.exit(1)

# --- Configuration ---
DOMAIN = "SocialChem"
DATA_FILE = "data/socialchem/socialchem_sampled_30.jsonl"
OUTPUT_FILE = "outputs/outputs_{domain_lc}.json" # domain_lc will be lowercase domain name

MODELS_TO_RUN = [
    "deepseek-ai/deepseek-r1",
    "o3",
    "claude-sonnet-3.7",
    "apple/OpenELM-3B-Instruct"
]

API_KEYS = {
    "replicate": os.environ.get("REPLICATE_API_TOKEN"),
    "openai": os.environ.get("OPENAI_API_KEY"),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY")
}

# --- Data Loading Function ---
def load_data(file_path):
    if file_path.endswith(".json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return [{"id": f"{DOMAIN.lower()}_{i+1}", "text": obj} for i, obj in enumerate(data.get("objects", []))]
    elif file_path.endswith(".jsonl"): # SocialChem data is .jsonl
        items = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                record = json.loads(line)
                # GSM8K: {"question": "...", "answer": "..."} -> text is question
                # SocialChem: {"situation_id": "...", "situation": "...", "rot": "..."} -> text is situation
                item_text = record.get("question") or record.get("situation")
                # Use situation_id if present, else generate one.
                item_id = record.get("situation_id") or record.get("id") or f"{DOMAIN.lower()}_{i+1}"
                if item_text:
                    items.append({"id": item_id, "text": item_text, "full_record": record})
        return items
    elif file_path.endswith(".csv") or file_path.endswith(".tsv"):
        sep = '\t' if file_path.endswith(".tsv") else ','
        df = pd.read_csv(file_path, sep=sep)
        items = []
        for i, row in df.iterrows():
            question_text = row.get("Question") or row.get("question")
            if question_text:
                items.append({"id": f"{DOMAIN.lower()}_q{i+1}", "text": question_text, "full_record": row.to_dict()})
        return items
    else:
        raise ValueError(f"Unsupported data file format: {file_path}")

# --- Main Script ---
def main():
    print(f"Starting output generation for domain: {DOMAIN}")
    print(f"Loading data from: {DATA_FILE}")

    output_file_name = OUTPUT_FILE.format(domain_lc=DOMAIN.lower())
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

    try:
        dataset_items = load_data(DATA_FILE)
        if not dataset_items:
            print(f"Error: No items found or loaded from {DATA_FILE}")
            return
    except Exception as e:
        print(f"Error loading data from {DATA_FILE}: {e}")
        return

    all_outputs = []
    # --- Process only a subset for testing in environment ---
    # SocialChem has 30 items, so processing all is fine / quick.
    dataset_items_subset = dataset_items[:5] # Process first 5 items for consistency and speed
    total_items_to_process = len(dataset_items_subset)
    print(f"Found {len(dataset_items)} total items. Processing subset of {total_items_to_process} for {DOMAIN}.")


    for item_idx, item_data in enumerate(dataset_items_subset): # Use subset
        item_text = item_data["text"]
        item_id = item_data["id"]
        print(f"\nProcessing item {item_idx + 1}/{total_items_to_process}: ID '{item_id}'")

        for model_name in MODELS_TO_RUN:
            print(f"  Model: {model_name}")
            for prompt_type in ["plain", "creative"]:
                print(f"    Prompt Type: {prompt_type}")
                try:
                    prompt = llm_handler.generate_prompt(item_text, DOMAIN, prompt_type)
                    time.sleep(0.1)
                    response = llm_handler.get_llm_response(prompt, model_name, api_keys=API_KEYS)

                    output_record = {
                        "item_id": item_id,
                        "domain_specific_data": item_data.get("full_record", {DOMAIN.lower()+"_text": item_text}),
                        "model_name": model_name,
                        "prompt_type": prompt_type,
                        "prompt": prompt,
                        "response": response
                    }
                    all_outputs.append(output_record)
                    # Reduced verbosity
                except Exception as e:
                    print(f"Error processing item ID {item_id} with {model_name} ({prompt_type}): {e}")
                    error_record = {
                        "item_id": item_id,
                        "domain_specific_data": item_data.get("full_record", {DOMAIN.lower()+"_text": item_text}),
                        "model_name": model_name,
                        "prompt_type": prompt_type,
                        "prompt": llm_handler.generate_prompt(item_text, DOMAIN, prompt_type) if 'prompt' not in locals() else prompt,
                        "response": f"ERROR: {str(e)}"
                    }
                    all_outputs.append(error_record)

    print(f"\nSaving all outputs to {output_file_name}...")
    try:
        with open(output_file_name, 'w') as f:
            json.dump(all_outputs, f, indent=4)
        print(f"Successfully saved outputs for {DOMAIN} to {output_file_name}")
    except IOError:
        print(f"Error: Could not write to {output_file_name}.")

if __name__ == "__main__":
    print(f"Environment check for {DOMAIN} script:")
    if not API_KEYS["replicate"]: print("  REPLICATE_API_TOKEN not set.")
    if not API_KEYS["openai"]: print("  OPENAI_API_KEY not set.")
    if not API_KEYS["anthropic"]: print("  ANTHROPIC_API_KEY not set.")
    print("  Note: apple/OpenELM-3B-Instruct will use mock if transformers not fully installed.\n")
    main()
