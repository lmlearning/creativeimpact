import os

try:
    import replicate # Added for DeepSeek R1
except ImportError:
    print("DEBUG: Failed to import 'replicate' in llm_handler.py")
    replicate = None # Ensure the name exists to avoid NameErrors later if code tries to use it
try:
    from openai import OpenAI # Added for OpenAI models
except ImportError:
    print("DEBUG: Failed to import 'OpenAI' from 'openai' in llm_handler.py")
    OpenAI = None
try:
    from anthropic import Anthropic # Added for Anthropic models
except ImportError:
    print("DEBUG: Failed to import 'Anthropic' in llm_handler.py")
    Anthropic = None
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # Kept for potential future use or indirect dependencies
except ImportError:
    print("DEBUG: Failed to import from 'transformers' in llm_handler.py")
    AutoTokenizer, AutoModelForCausalLM, pipeline = None, None, None
try:
    import torch # Kept for potential future use or indirect dependencies
except (ImportError, OSError) as e: # Catch OSError as well
    print(f"DEBUG: Failed to import 'torch' in llm_handler.py: {e}")
    torch = None
import json # For example usage

def generate_prompt(item_text: str, domain: str, prompt_type: str) -> str:
    # This function remains the same as previously defined.
    # For brevity, I'm not pasting it here again but it will be in the final script.
    prompt = ""
    creative_modifier = ""
    if prompt_type == "creative":
        if domain == "AUT":
            creative_modifier = "creative "
        elif domain == "TruthfulQA":
            creative_modifier = "Be creative in your answer. "
        elif domain == "GSM8K":
            creative_modifier = "with creative reasoning "
        elif domain == "SocialChem":
            creative_modifier = "creative and "
    if domain == "AUT":
        prompt = f"List as many {creative_modifier}uses for a {item_text} as possible. Answer with a comma separated list with no explanation."
    elif domain == "TruthfulQA":
        if prompt_type == "creative":
             prompt = f"Question: {item_text}\n{creative_modifier}Answer:"
        else:
             prompt = f"Question: {item_text}\nAnswer:"
    elif domain == "GSM8K":
        if prompt_type == "creative":
            prompt = f"Solve the following math problem {creative_modifier}:\n{item_text}\nStep-by-step solution:"
        else:
            prompt = f"Solve the following math problem:\n{item_text}\nStep-by-step solution:"
    elif domain == "SocialChem":
        prompt = f"What is a {creative_modifier}socially acceptable rule of thumb for this situation. Answer with a single sentence.?\nSituation: {item_text}\nResponse:"
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return prompt.strip()

def get_llm_response(prompt: str, model_name: str, api_keys: dict = None):
    """
    Gets a response from the specified LLM for the given prompt.
    API keys should be passed in the api_keys dictionary or set as environment variables.
    For Replicate, REPLICATE_API_TOKEN environment variable is typically used by the client.
    """
    print(f"Received prompt for model '{model_name}': \"{prompt}\"")

    if api_keys is None:
        api_keys = {}

    if model_name == "deepseek-ai/deepseek-r1": # Changed to "deepseek r1"
        print(f"Using Replicate for {model_name}")
        # The replicate client automatically uses REPLICATE_API_TOKEN env var.
        # No need to check api_keys dict for 'replicate' as per standard usage.
        if "REPLICATE_API_TOKEN" not in os.environ:
            raise ValueError("REPLICATE_API_TOKEN environment variable not found. Please set it to use DeepSeek R1.")

        try:
            # The model expects a dictionary input. For deepseek-r1, it's typically {"prompt": prompt}
            # The output is often an iterator of strings.
            input_payload = {"prompt": prompt}
            output_iterator = replicate.run(
                "deepseek-ai/deepseek-r1", # Actual model identifier for Replicate
                input=input_payload
            )
            response_parts = [str(part) for part in output_iterator]
            full_response = "".join(response_parts)
            return full_response.strip()
        except replicate.exceptions.ReplicateError as e:
            # More specific error handling can be added if needed
            raise ValueError(f"Replicate API error for deepseek-ai/deepseek-r1: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during the Replicate call
            raise ValueError(f"An unexpected error occurred with Replicate model deepseek-ai/deepseek-r1: {e}") from e

    elif model_name == "o3": # Changed to "o3"
        print(f"Using OpenAI API for {model_name}")
        api_key = api_keys.get("openai") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found. Please set it to use an OpenAI model like o3 (gpt-4o).")

        try:
            client = OpenAI(api_key=api_key)

            chat_completion = client.chat.completions.create(
                model="o3",                           # OpenAI’s full-scale reasoning model
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024,           # reasoning models require this param
                reasoning_effort="medium"             # low | medium | high
            )
            response_text = chat_completion.choices[0].message.content
            return response_text.strip()
        except Exception as e: # Catching a broad exception from the OpenAI client
            raise ValueError(f"OpenAI API error for gpt-4o: {e}") from e

    elif model_name == "claude 3.7": # Changed to "claude 3.7"
        print(f"Using Anthropic API for {model_name}")
        api_key = api_keys.get("anthropic") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not found. Please set it to use an Anthropic model like claude 3.7 (claude-3-5-sonnet-20240620).")

        try:
            client = Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620", # Actual model identifier for Anthropic
                max_tokens=1024,  # Adjust as needed
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = message.content[0].text
            return response_text.strip()
        except Exception as e: # Catching a broad exception from the Anthropic client
            raise ValueError(f"Anthropic API error for claude-3-5-sonnet-20240620: {e}") from e

    else:
        raise ValueError(f"Unknown or unsupported model_name: {model_name}")

if __name__ == '__main__':
    print("--- Testing generate_prompt ---")
    aut_obj = "empty soda can"
    print(f"AUT Plain: {generate_prompt(aut_obj, 'AUT', 'plain')}")
    print(f"AUT Creative: {generate_prompt(aut_obj, 'AUT', 'creative')}")
    # ... (other generate_prompt tests from previous version for brevity) ...
    tqa_q = "What is the capital of France?"
    print(f"TruthfulQA Plain: {generate_prompt(tqa_q, 'TruthfulQA', 'plain')}")
    print(f"TruthfulQA Creative: {generate_prompt(tqa_q, 'TruthfulQA', 'creative')}")

    gsm_problem = "Natalia sold clips to 48 of her friends. Each friend bought 3 clips. How many clips did she sell in total?"
    print(f"GSM8K Plain: {generate_prompt(gsm_problem, 'GSM8K', 'plain')}")
    print(f"GSM8K Creative: {generate_prompt(gsm_problem, 'GSM8K', 'creative')}")

    sc_situation = "My friend asked to borrow a significant amount of money, but I'm worried they won't pay it back."
    print(f"SocialChem Plain: {generate_prompt(sc_situation, 'SocialChem', 'plain')}")
    print(f"SocialChem Creative: {generate_prompt(sc_situation, 'SocialChem', 'creative')}")

    print("\n--- Testing get_llm_response ---")
    test_prompt_aut_creative = generate_prompt(aut_obj, 'AUT', 'creative')

    # Test DeepSeek R1
    try:
        print(f"DeepSeek R1: {get_llm_response(test_prompt_aut_creative, 'deepseek r1')}") # Changed to "deepseek r1"
    except ValueError as e:
        print(f"Error testing DeepSeek R1: {e}")
    except Exception as e:
        print(f"Unexpected error testing DeepSeek R1: {e}")


    # Test OpenAI o3
    try:
        print(f"OpenAI o3: {get_llm_response(test_prompt_aut_creative, 'o3')}") # Changed to "o3"
    except ValueError as e:
        print(f"Error testing OpenAI o3: {e}")
    except Exception as e:
        print(f"Unexpected error testing OpenAI o3: {e}")

    # Test Anthropic claude 3.7
    try:
        print(f"Anthropic claude 3.7: {get_llm_response(test_prompt_aut_creative, 'claude 3.7')}") # Changed to "claude 3.7"
    except ValueError as e:
        print(f"Error testing Anthropic claude 3.7: {e}")
    except Exception as e:
        print(f"Unexpected error testing Anthropic claude 3.7: {e}")
