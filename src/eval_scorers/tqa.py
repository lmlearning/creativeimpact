import sys
import subprocess
import os
import argparse
import json
import importlib.util

# ===== Step 1: Check for compatible Python version =====
if sys.version_info >= (3, 12):
    print(f"❌ ERROR: Your Python version ({sys.version_info.major}.{sys.version_info.minor}) is not compatible.", file=sys.stderr)
    print("The dependencies for TruthfulQA require Python 3.11 or older.", file=sys.stderr)
    print("\nPlease create and activate a new Conda environment with an older Python version:", file=sys.stderr)
    print("  conda create -n tqa_eval python=3.9", file=sys.stderr)
    print("  conda activate tqa_eval", file=sys.stderr)
    print("\nThen, run this script again from within the new environment.", file=sys.stderr)
    sys.exit(1)

# A list of packages required for this script to run
REQUIRED_PACKAGES = ['pandas', 'gitpython']

def check_and_install_dependencies():
    """Checks for required packages and installs them if missing."""
    missing_packages = []
    for package_name in REQUIRED_PACKAGES:
        module_name = 'git' if package_name == 'gitpython' else package_name
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(package_name)

    if missing_packages:
        print(f"🚀 Missing required packages: {', '.join(missing_packages)}")
        print("   Attempting to install them now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("\n✅ Dependencies installed successfully.")
            print("👉 Please run the script again to continue.")
            sys.exit(0)
        except subprocess.CalledProcessError:
            print(f"\n❌ ERROR: Failed to install packages. Please install them manually:", file=sys.stderr)
            print(f"   pip install {' '.join(missing_packages)}", file=sys.stderr)
            sys.exit(1)

def setup_truthfulqa_repository(repo_url="https://github.com/sylinrl/TruthfulQA.git", repo_dir="TruthfulQA"):
    """Clones the TruthfulQA repository and installs its internal requirements."""
    from git import Repo
    if not os.path.exists(repo_dir):
        print(f"Cloning TruthfulQA repository from {repo_url}...")
        try:
            Repo.clone_from(repo_url, repo_dir)
            print("Repository cloned successfully.")
        except Exception as e:
            print(f"Fatal: Error cloning repository: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("✅ TruthfulQA repository already exists.")

    requirements_path = os.path.join(repo_dir, "requirements.txt")
    if os.path.exists(requirements_path):
        print("🚀 Preparing to install TruthfulQA's own dependencies...")
        try:
            print("   -> Step 1/3: Updating pip, setuptools, and wheel...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            
            # **THE FIX**: Manually install the compatible version of array-record BEFORE installing requirements.txt
            print("   -> Step 2/3: Forcing installation of compatible dependency (array-record==0.4.0)...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "array-record==0.4.0"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            print("   -> Step 3/3: Installing packages from requirements.txt...")
            install_command = [
                sys.executable, "-m", "pip", "install",
                "--no-build-isolation",
                "-r", requirements_path
            ]
            subprocess.check_call(install_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print("✅ TruthfulQA dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ ERROR: Failed to install dependencies from 'TruthfulQA/requirements.txt'.", file=sys.stderr)
            print(f"   Original error: {e}", file=sys.stderr)
            sys.exit(1)
    return repo_dir

def convert_json_to_csv(json_file_path, truthfulqa_csv_path, pd):
    """
    Converts a JSON file with model predictions to a CSV file formatted for evaluation.
    This version safely handles missing keys in the input data.
    """
    print(f"🔄 Converting '{os.path.basename(json_file_path)}' to evaluation format...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Fatal: Error reading input JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    model_name = data[0].get("model_name", "unnamed_model")
    
    answer_dict = {}
    for i, item in enumerate(data):
        question = item.get('original_item_dict', {}).get('question')
        if not question:
            print(f"⚠️ Warning: Item at index {i} is missing a question. Skipping.", file=sys.stderr)
            continue
        
        answer = item.get('plain_response', '')
        answer_dict[question] = answer

    df_truthfulqa = pd.read_csv(truthfulqa_csv_path)
    df_truthfulqa[model_name] = df_truthfulqa['Question'].map(answer_dict)
    
    unanswered_count = df_truthfulqa[model_name].isnull().sum()
    if unanswered_count > 0:
        print(f"⚠️ Warning: {unanswered_count} questions from TruthfulQA.csv were not found in the input JSON file. They will have blank answers.", file=sys.stderr)
        df_truthfulqa[model_name].fillna('', inplace=True)

    output_csv_path = "evaluation_input.csv"
    df_truthfulqa.to_csv(output_csv_path, index=False)
    
    print(f"✅ Created evaluation file for model '{model_name}' at: {output_csv_path}")
    return output_csv_path, model_name

def run_evaluation(repo_dir, input_csv_path, model_name):
    """Runs the official TruthfulQA evaluation script as a module."""
    abs_input_path = os.path.abspath(input_csv_path)
    abs_output_path = os.path.abspath("evaluation_results.csv")
    
    command = [
        sys.executable,
        "-m", "truthfulqa.evaluate",
        "--input_path", abs_input_path,
        "--output_path", abs_output_path,
        "--models", model_name,
        "--metrics", "judge", "info"
    ]

    print("\n🔬 Running official evaluation script...")
    print(f"   Command: {' '.join(command)}")
    print(f"   (Running from directory: {repo_dir})")
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            cwd=repo_dir
        )
        print(result.stdout)
        if result.stderr:
            # Filter out the known deprecation warning to avoid clutter
            filtered_stderr = "\n".join([line for line in result.stderr.splitlines() 
                                         if "pkg_resources is deprecated" not in line])
            if filtered_stderr:
                print(filtered_stderr, file=sys.stderr)

        print("\n🎉 Evaluation complete.")
        print(f"   -> Results saved to: {abs_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Fatal: The evaluation script failed with exit code {e.returncode}.", file=sys.stderr)
        print("\n--- STDOUT ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("\n--- STDERR ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A self-contained script to convert model outputs and run the official TruthfulQA evaluation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_json', type=str, help='Path to the input JSON file containing model predictions.')
    args = parser.parse_args()

    check_and_install_dependencies()
    
    import pandas as pd
    
    repo_directory = setup_truthfulqa_repository()
    original_csv_path = os.path.join(repo_directory, "TruthfulQA.csv")
    evaluation_csv, model_name_from_file = convert_json_to_csv(args.input_json, original_csv_path, pd)
    run_evaluation(repo_directory, evaluation_csv, model_name_from_file)