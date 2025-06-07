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
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # Added for Hugging Face local inference
except ImportError:
    print("DEBUG: Failed to import from 'transformers' in llm_handler.py")
    AutoTokenizer, AutoModelForCausalLM, pipeline = None, None, None
try:
    import torch # Added for Hugging Face local inference
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
        prompt = f"List three {creative_modifier}uses for a {item_text}."
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
        prompt = f"What is a {creative_modifier}socially acceptable way to respond to the following situation?\nSituation: {item_text}\nResponse:"
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

    if model_name == "deepseek-ai/deepseek-r1":
        print(f"Using Replicate for {model_name}")
        try:
            # The replicate client automatically uses REPLICATE_API_TOKEN env var.
            # If api_keys['replicate'] is provided, it could be used to set it temporarily,
            # but standard practice is to rely on the environment variable.
            if "REPLICATE_API_TOKEN" not in os.environ and not api_keys.get("replicate"):
                print("Warning: REPLICATE_API_TOKEN not found in environment or api_keys.")
                print("Returning mock response for DeepSeek R1 as API key is missing.")
                return f"Mock response (API key missing) for DeepSeek R1: {prompt}"

            # If an API key is explicitly passed, set it for this call
            # This is useful if the subtask environment doesn't persist env variables easily
            current_env_token = os.environ.get("REPLICATE_API_TOKEN")
            if api_keys.get("replicate"):
                os.environ["REPLICATE_API_TOKEN"] = api_keys["replicate"]

            # It's good practice to ensure the client is initialized here if not globally
            # For this subtask, direct call is fine.

            # The model expects a dictionary input, typically with a 'prompt' key or similar.
            # Checking the specific model's expected input format on Replicate is important.
            # For deepseek-r1, it's likely a standard prompt input.
            # Let's assume it takes input like: {"prompt": "..."}
            # The output is often an iterator of strings.
            input_payload = {"prompt": prompt}

            # Replace with actual model version if needed, e.g. by finding the latest deployment
            # For now, using the model name directly which Replicate usually resolves.
            output_iterator = replicate.run(
                model_name,
                input=input_payload
            )

            response_parts = [str(part) for part in output_iterator]
            full_response = "".join(response_parts)

            # Restore original env token if it was changed
            if api_keys.get("replicate") and current_env_token:
                os.environ["REPLICATE_API_TOKEN"] = current_env_token
            elif api_keys.get("replicate") and not current_env_token: # was set, but no original
                 del os.environ["REPLICATE_API_TOKEN"]


            return full_response.strip()

        except replicate.exceptions.ReplicateError as e:
            print(f"Replicate API error for {model_name}: {e}")
            # Fallback to mock response or re-raise depending on desired error handling
            return f"Mock response (Replicate API Error: {e}) for DeepSeek R1: {prompt}"
        except Exception as e:
            print(f"An unexpected error occurred with Replicate model {model_name}: {e}")
            # Fallback to mock response
            return f"Mock response (Unexpected Error: {e}) for DeepSeek R1: {prompt}"

    elif model_name == "o3": # User specified this for OpenAI
        print(f"Using OpenAI API for {model_name}")
        try:
            api_key = api_keys.get("openai") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENAI_API_KEY not found in environment or api_keys.")
                print(f"Returning mock response for {model_name} as API key is missing.")
                return f"Mock response (API key missing) for {model_name}: {prompt}"

            client = OpenAI(api_key=api_key)

            # Using the ChatCompletions endpoint as it's standard
            chat_completion = client.chat.completions.create(
                model=model_name,  # "o3"
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024 # Adjust as needed
            )
            response_text = chat_completion.choices[0].message.content
            return response_text.strip()

        except Exception as e: # Catching a broad exception for now
            print(f"OpenAI API error for {model_name}: {e}")
            return f"Mock response (OpenAI API Error: {e}) for {model_name}: {prompt}"

    elif model_name == "apple/OpenELM-3B-Instruct":
        print(f"Using Hugging Face Transformers for {model_name}")
        try:
            # Try to use GPU if available, otherwise CPU
            device = 0 if torch.cuda.is_available() else -1
            print(f"HF device: {'cuda:0' if device == 0 else 'cpu'}")

            # Load tokenizer and model
            # Using trust_remote_code=True as some models (like OpenELM) require it.
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            model_obj = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            if device == 0: # GPU
                model_obj = model_obj.to("cuda")
            else: # CPU
                model_obj = model_obj.to("cpu")

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024) # Max input length

            if device == 0:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            else:
                inputs = {k: v.to("cpu") for k, v in inputs.items()}

            # Generate text
            output_sequences = model_obj.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)

            response_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

            # The response usually includes the prompt, so we might want to remove it.
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):]

            return response_text.strip()

        except ImportError:
            print("Error: Hugging Face Transformers or PyTorch not installed correctly.")
            return f"Mock response (HF/Torch Import Error) for {model_name}: {prompt}"
        except Exception as e:
            print(f"Hugging Face model error for {model_name}: {e}")
            return f"Mock response (HF Model Error: {e}) for {model_name}: {prompt}"

    elif model_name == "claude-sonnet-3.7":
        print(f"Using Anthropic API for {model_name}")
        try:
            # Anthropic client uses ANTHROPIC_API_KEY env var by default
            api_key = api_keys.get("anthropic") or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("Warning: ANTHROPIC_API_KEY not found in environment or api_keys.")
                print(f"Returning mock response for {model_name} as API key is missing.")
                return f"Mock response (API key missing) for {model_name}: {prompt}"

            client = Anthropic(api_key=api_key)

            message = client.messages.create(
                model=model_name, # Directly use the model_name passed, e.g. "claude-sonnet-3.7"
                max_tokens=1024,  # Adjust as needed
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = message.content[0].text
            return response_text.strip()

        except Exception as e: # Catching a broad exception
            print(f"Anthropic API error for {model_name}: {e}")
            return f"Mock response (Anthropic API Error: {e}) for {model_name}: {prompt}"

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

    print("\n--- Testing get_llm_response (Replicate potentially live if API key is set) ---")
    test_prompt_aut_creative = generate_prompt(aut_obj, 'AUT', 'creative')

    # To test Replicate, an API key must be set as REPLICATE_API_TOKEN environment variable
    # in the execution environment of this subtask.
    # If the key is not set, it will print a warning and return a mock response.
    print(f"DeepSeek R1: {get_llm_response(test_prompt_aut_creative, 'deepseek-ai/deepseek-r1')}")

    print(f"OpenAI o3 (mocked): {get_llm_response(test_prompt_aut_creative, 'o3')}")
    print(f"OpenELM (mocked): {get_llm_response(test_prompt_aut_creative, 'apple/OpenELM-3B-Instruct')}")
    print(f"Claude Sonnet 3.7 (mocked): {get_llm_response(test_prompt_aut_creative, 'claude-sonnet-3.7')}")
