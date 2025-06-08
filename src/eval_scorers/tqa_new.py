#!/usr/bin/env python3
"""
TruthfulQA Official Evaluator
Uses the official TruthfulQA evaluation functions with minimal dependency setup.
"""

import sys
import json
import pandas as pd
import argparse
import os
from pathlib import Path
import subprocess

def setup_truthfulqa_repo():
    """Clone TruthfulQA repository if not present."""
    tqa_dir = Path("TruthfulQA")
    if not tqa_dir.exists():
        print("🔄 Cloning TruthfulQA repository...")
        try:
            subprocess.run([
                "git", "clone", "https://github.com/sylinrl/TruthfulQA.git"
            ], check=True)
            print("✅ TruthfulQA repository cloned")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Failed to clone repository. Please ensure git is installed and try:")
            print("   git clone https://github.com/sylinrl/TruthfulQA.git")
            sys.exit(1)
    else:
        print("✅ TruthfulQA repository already exists")
    
    # Add to Python path (Windows-compatible)
    tqa_abs_path = str(tqa_dir.absolute())
    if tqa_abs_path not in sys.path:
        sys.path.insert(0, tqa_abs_path)
    
    return tqa_dir

def convert_json_to_eval_format(json_file_path, truthfulqa_csv_path):
    """Convert model predictions JSON to evaluation format."""
    json_path = Path(json_file_path)
    print(f"🔄 Converting '{json_path.name}' to evaluation format...")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"❌ Error reading input JSON file: {e}")
        sys.exit(1)

    if not data:
        print("❌ Empty JSON file")
        sys.exit(1)

    model_name = data[0].get("model_name", "unnamed_model")
    print(f"📊 Found model: {model_name}")
    
    # Build answer dictionary
    answer_dict = {}
    skipped = 0
    for i, item in enumerate(data):
        question = item.get('original_item_dict', {}).get('question')
        if not question:
            skipped += 1
            continue
        
        answer = item.get('plain_response', '')
        answer_dict[question] = answer

    print(f"📝 Processed {len(answer_dict)} predictions, skipped {skipped} items")

    # Load TruthfulQA questions and add model answers
    print(f"📖 Loading TruthfulQA questions from {Path(truthfulqa_csv_path).name}")
    df_truthfulqa = pd.read_csv(truthfulqa_csv_path)
    df_truthfulqa[model_name] = df_truthfulqa['Question'].map(answer_dict)
    
    # Fill missing answers
    unanswered_count = df_truthfulqa[model_name].isnull().sum()
    if unanswered_count > 0:
        print(f"⚠️ Warning: {unanswered_count} questions from TruthfulQA.csv were not found in the input JSON file.")
        df_truthfulqa[model_name].fillna('', inplace=True)

    # Save evaluation input
    output_csv_path = f"evaluation_input_{model_name}.csv"
    df_truthfulqa.to_csv(output_csv_path, index=False)
    
    print(f"✅ Created evaluation file for model '{model_name}' at: {output_csv_path}")
    return output_csv_path, model_name

def run_official_evaluation(input_csv_path, model_name, output_dir="results"):
    """Run the official TruthfulQA evaluation using the actual functions."""
    
    try:
        # Import TruthfulQA functions
        print("📦 Importing TruthfulQA evaluation functions...")
        from truthfulqa.metrics import run_gpt_judge_forwarding
        from truthfulqa.utilities import split_multi_answer
        print("✅ Successfully imported TruthfulQA functions")
        
    except ImportError as e:
        print(f"❌ Failed to import TruthfulQA functions: {e}")
        print("💡 Make sure you've set up the environment correctly (see setup instructions)")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔬 Running official TruthfulQA evaluation...")
    
    try:
        # Load the data
        df = pd.read_csv(input_csv_path)
        
        # Run GPT judge evaluation
        print("🤖 Running GPT judge evaluation...")
        judge_results = run_gpt_judge_forwarding(
            df, 
            model_name, 
            info=True,  # Include informativeness
            mc=False    # Not multiple choice
        )
        
        # Save results
        output_path = os.path.join(output_dir, f"tqa_results_{model_name}.csv")
        judge_results.to_csv(output_path, index=False)
        
        # Calculate summary statistics
        truthful_col = f"{model_name} truthful"
        informative_col = f"{model_name} informative"
        
        if truthful_col in judge_results.columns:
            truthful_mean = judge_results[truthful_col].mean()
            print(f"📊 Average Truthfulness: {truthful_mean:.3f}")
        
        if informative_col in judge_results.columns:
            informative_mean = judge_results[informative_col].mean()
            print(f"📊 Average Informativeness: {informative_mean:.3f}")
        
        # Category breakdown
        if 'Category' in judge_results.columns and truthful_col in judge_results.columns:
            category_stats = judge_results.groupby('Category')[truthful_col].mean()
            print("\n📋 Truthfulness by Category:")
            for category, score in category_stats.items():
                print(f"  {category}: {score:.3f}")
        
        print(f"\n✅ Evaluation complete! Results saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="TruthfulQA Official Evaluator with clean environment setup",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_json', type=str, 
                       help='Path to the input JSON file containing model predictions.')
    parser.add_argument('--output_dir', type=str, default='tqa_results',
                       help='Output directory for results (default: tqa_results)')
    parser.add_argument('--truthfulqa_csv', type=str, default=None,
                       help='Path to TruthfulQA.csv (default: auto-detect from repo)')
    
    args = parser.parse_args()
    
    print("🚀 TruthfulQA Official Evaluation Starting...")
    
    # Setup TruthfulQA repository
    tqa_dir = setup_truthfulqa_repo()
    
    # Determine TruthfulQA.csv path (Windows-compatible)
    if args.truthfulqa_csv:
        truthfulqa_csv = Path(args.truthfulqa_csv)
    else:
        truthfulqa_csv = tqa_dir / "TruthfulQA.csv"
    
    if not truthfulqa_csv.exists():
        print(f"❌ TruthfulQA.csv not found at {truthfulqa_csv}")
        print("💡 Make sure the TruthfulQA repository was cloned correctly")
        sys.exit(1)
    
    # Convert input to evaluation format
    eval_csv, model_name = convert_json_to_eval_format(args.input_json, str(truthfulqa_csv))
    
    # Run official evaluation
    run_official_evaluation(eval_csv, model_name, args.output_dir)
    
    print("🎉 All done!")

if __name__ == "__main__":
    main()