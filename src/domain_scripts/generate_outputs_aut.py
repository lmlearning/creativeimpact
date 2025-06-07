import json
import os
import sys
import time

# Ensure src is on path for utils.llm_handler, or run with PYTHONPATH including src
try:
    from src.utils import llm_handler # Changed import to absolute from src
except ImportError:
    print("Error: llm_handler from src.utils not found. Make sure src is on PYTHONPATH or script is run as part of a package.")
    sys.exit(1)

# --- Configuration ---
DOMAIN = "AUT"
DATA_FILE = "data/aut_objects.json"
OUTPUT_FILE = "outputs/outputs_aut.json"

# Define models to use - these should match identifiers in llm_handler
# The 'apple/OpenELM-3B-Instruct' will likely use mock responses in subtasks
# due to previous environment limitations for transformers.
MODELS_TO_RUN = [
    "deepseek-ai/deepseek-r1",
    "o3",  # OpenAI o3 model
    "claude-sonnet-3.7", # Anthropic Claude Sonnet 3.7
    "apple/OpenELM-3B-Instruct"
]

# API Key Management (Example: Load from environment variables)
# In a real execution, these would be set in the user's environment.
# llm_handler.get_llm_response will also check environment variables directly.
API_KEYS = {
    "replicate": os.environ.get("REPLICATE_API_TOKEN"),
    "openai": os.environ.get("OPENAI_API_KEY"),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY")
}

# --- Main Script ---
def main():
    print(f"Starting output generation for domain: {DOMAIN}")
    print(f"Loading data from: {DATA_FILE}")

    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        aut_objects = data.get("objects", [])
        if not aut_objects:
            print(f"Error: No objects found in {DATA_FILE}")
            return
    except FileNotFoundError:
        print(f"Error: Data file {DATA_FILE} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {DATA_FILE}.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    all_outputs = []
    total_items = len(aut_objects)
    print(f"Found {total_items} AUT objects to process.")

    for item_idx, item_text in enumerate(aut_objects):
        print(f"\nProcessing item {item_idx + 1}/{total_items}: '{item_text}'")
        item_id = f"aut_object_{item_idx + 1}" # Simple item ID

        for model_name in MODELS_TO_RUN:
            print(f"  Model: {model_name}")

            # Note: The llm_handler.get_llm_response is designed to use mock
            # responses if API keys are missing or if libraries are not installed (e.g. transformers).
            # This is especially relevant for 'apple/OpenELM-3B-Instruct' in limited environments.

            for prompt_type in ["plain", "creative"]:
                print(f"    Prompt Type: {prompt_type}")
                try:
                    prompt = llm_handler.generate_prompt(item_text, DOMAIN, prompt_type)

                    # Simulate a small delay to avoid overwhelming APIs if they were live
                    # and to make the script execution progress more observable.
                    time.sleep(0.1)

                    response = llm_handler.get_llm_response(prompt, model_name, api_keys=API_KEYS)

                    output_record = {
                        "item_id": item_id,
                        "object": item_text,
                        "model_name": model_name,
                        "prompt_type": prompt_type,
                        "prompt": prompt,
                        "response": response
                    }
                    all_outputs.append(output_record)
                    print(f"      -> Response (mocked/actual): {response[:100]}...") # Print snippet
                except Exception as e:
                    print(f"Error processing {item_text} with {model_name} ({prompt_type}): {e}")
                    # Optionally add an error record
                    error_record = {
                        "item_id": item_id,
                        "object": item_text,
                        "model_name": model_name,
                        "prompt_type": prompt_type,
                        "prompt": llm_handler.generate_prompt(item_text, DOMAIN, prompt_type) if 'prompt' not in locals() else prompt,
                        "response": f"ERROR: {str(e)}"
                    }
                    all_outputs.append(error_record)

    print(f"\nSaving all outputs to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_outputs, f, indent=4)
        print(f"Successfully saved outputs for {DOMAIN} to {OUTPUT_FILE}")
    except IOError:
        print(f"Error: Could not write to {OUTPUT_FILE}.")

if __name__ == "__main__":
    # This check is important if llm_handler itself tries to use API keys upon import,
    # which it currently doesn't in its global scope.
    print("Checking for API keys (will use mocks if not found):")
    if not API_KEYS["replicate"]: print("  REPLICATE_API_TOKEN not set.")
    if not API_KEYS["openai"]: print("  OPENAI_API_KEY not set.")
    if not API_KEYS["anthropic"]: print("  ANTHROPIC_API_KEY not set.")
    print("Note: apple/OpenELM-3B-Instruct will use mock if transformers not fully installed.\n")

    main()
