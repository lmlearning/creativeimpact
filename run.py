import os
import subprocess
import argparse
from dotenv import load_dotenv

def run_command(command):
    """Executes a command and raises an exception on error."""
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, check=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(command)}")
        exit(1)
    print("-" * 20)

def main():
    """
    Main script to set up datasets and run all experiments.
    Configuration is handled via a .env file and command-line arguments.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run creative impact experiments.')
    parser.add_argument('--model_name', type=str,
                        default=os.getenv('MODEL_NAME', 'o3'),
                        help='The model to use. Overrides the MODEL_NAME in .env file.')
    
    args = parser.parse_args()
    model_name = args.model_name
    
    print(f"Using model: {model_name}")
    os.makedirs("outputs", exist_ok=True)

    # --- 1. Dataset Setup ---
    print("Setting up datasets...")
    setup_scripts = [
        "src/domain_scripts/aut_setup.py",
        "src/domain_scripts/gsm8k_setup.py",
        "src/domain_scripts/socialchem_setup.py",
        "src/domain_scripts/truthfulqa_setup.py",
    ]
    #for script in setup_scripts:
    #    run_command(["python", script])
    print("All datasets set up.")

    # --- 2. Define Experiments ---
    experiments = {
        "aut": {
            "generate": "src/domain_scripts/generate_outputs_aut.py",
            "evaluate": ["src/evaluate.py", "--domain", "aut"],
        },
        "gsm8k": {
            "generate": "src/domain_scripts/generate_outputs_gsm8k.py",
            "evaluate": ["src/evaluate.py", "--domain", "gsm8k"],
        },
        "social": {
            "generate": "src/domain_scripts/generate_outputs_socialchem.py",
            "evaluate": ["src/evaluate.py", "--domain", "social"],
        },
        "truthfulqa": {
            "generate": "src/domain_scripts/generate_outputs_truthfulqa.py",
            "evaluate": ["src/evaluate.py", "--domain", "truthfulqa", "--questions_file", "data/TruthfulQA.csv"],
        }
    }

    # --- 3. Run All Experiments ---
    for domain, scripts in experiments.items():
        print(f"--- Running Experiment: {domain.upper()} ---")
        output_file = f"outputs/{domain}_outputs.jsonl"
        
        # Generation step
        gen_command = [
            "python", scripts["generate"],
            "--model_name", model_name,
            "--output_file", output_file
        ]
        run_command(gen_command)
        
        # Evaluation step
        eval_command = scripts["evaluate"] + ["--input_file", output_file]
        run_command(eval_command)

    print("All experiments complete.")

if __name__ == '__main__':
    main()
