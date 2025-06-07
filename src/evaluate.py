import argparse
import json
import os
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
try:
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Warning: statsmodels.stats.contingency_tables.mcnemar not found. McNemar test will not be performed.")
    STATSMODELS_AVAILABLE = False

SUPPORTED_MODELS = ["gpt-4o", "claude-3-5-sonnet-20240620", "deepseek-ai/deepseek-r1"]

def sanitize_model_name(model_name):
    return model_name.replace("/", "_")

def main():
    parser = argparse.ArgumentParser(description="Run evaluation scorers and compile results.")
    parser.add_argument(
        "--model_name",
        required=True,
        choices=SUPPORTED_MODELS,
        help="Name of the LLM to use for evaluation."
    )
    parser.add_argument(
        "--generated_data_dir",
        default="outputs/generated_data",
        help="Directory containing the generated outputs from domain scripts (JSON files)."
    )
    parser.add_argument(
        "--eval_output_dir",
        default="outputs/evaluation_results",
        help="Directory to save final evaluation tables and intermediate scorer outputs."
    )
    parser.add_argument(
        "--scorers_dir",
        default="src/eval_scorers",
        help="Directory containing the domain-specific scorer scripts."
    )
    args = parser.parse_args()

    sanitized_model_name_for_files = sanitize_model_name(args.model_name)

    # Directory setup
    os.makedirs(args.eval_output_dir, exist_ok=True)
    scorer_outputs_dir = os.path.join(args.eval_output_dir, "scorer_outputs")
    os.makedirs(scorer_outputs_dir, exist_ok=True)

    domains = [
        {
            "name": "AUT", "metrics": ["fluency", "originality"], "scorer_script": "aut_scorer.py",
            "input_file_template": f"generated_aut_{sanitized_model_name_for_files}.json",
            "score_file_template": f"scores_aut_{sanitized_model_name_for_files}.csv",
            "metric_type": "continuous",
            "score_columns": {
                "plain": ["plain_fluency", "plain_originality"],
                "creative": ["creative_fluency", "creative_originality"]
            }
        },
        {
            "name": "TruthfulQA", "metrics": ["is_truthful"], "scorer_script": "truthfulqa_evaluate.py",
            "input_file_template": f"generated_truthfulqa_{sanitized_model_name_for_files}.json",
            "score_file_template": f"scores_truthfulqa_{sanitized_model_name_for_files}.csv",
            "metric_type": "binary",
            "score_columns": {"plain": ["plain_is_truthful"], "creative": ["creative_is_truthful"]}
        },
        {
            "name": "GSM8K", "metrics": ["correct"], "scorer_script": "gsm8k_exact.py",
            "input_file_template": f"generated_gsm8k_{sanitized_model_name_for_files}.json",
            "score_file_template": f"scores_gsm8k_{sanitized_model_name_for_files}.csv",
            "metric_type": "binary",
            "score_columns": {"plain": ["plain_correct"], "creative": ["creative_correct"]}
        },
        {
            "name": "SocialChem", "metrics": ["pass"], "scorer_script": "socialchem_eval.py",
            "input_file_template": f"generated_socialchem_{sanitized_model_name_for_files}.json",
            "score_file_template": f"scores_socialchem_{sanitized_model_name_for_files}.csv",
            "metric_type": "binary",
            "score_columns": {"plain": ["plain_pass"], "creative": ["creative_pass"]}
        }
    ]

    domain_score_dfs = {}

    print(f"--- Running Scorer Scripts for Model: {args.model_name} ---")
    for domain_config in domains:
        domain_name = domain_config["name"]
        scorer_script_name = domain_config["scorer_script"]
        scorer_script_path = os.path.join(args.scorers_dir, scorer_script_name)

        input_json_filename = domain_config["input_file_template"]
        input_json_path = os.path.join(args.generated_data_dir, input_json_filename)

        scorer_csv_filename = domain_config["score_file_template"]
        scorer_csv_output_path = os.path.join(scorer_outputs_dir, scorer_csv_filename)

        print(f"Processing domain: {domain_name}")
        if not os.path.exists(input_json_path):
            print(f"  Warning: Input file {input_json_path} not found. Skipping {domain_name}.")
            domain_score_dfs[domain_name] = None # Mark as skipped
            continue

        command = [
            "python", scorer_script_path,
            "--input_file", input_json_path,
            "--output_file", scorer_csv_output_path
        ]

        try:
            print(f"  Running scorer: {' '.join(command)}")
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"  Scorer output saved to {scorer_csv_output_path}")
            df = pd.read_csv(scorer_csv_output_path)
            domain_score_dfs[domain_name] = df
        except FileNotFoundError:
            print(f"  Error: Scorer script {scorer_script_path} not found. Make sure it exists and is executable.")
            domain_score_dfs[domain_name] = None
        except subprocess.CalledProcessError as e:
            print(f"  Error running scorer for {domain_name}: {e}")
            print(f"  Stdout: {e.stdout}")
            print(f"  Stderr: {e.stderr}")
            domain_score_dfs[domain_name] = None
        except Exception as e:
            print(f"  An unexpected error occurred while processing {domain_name}: {e}")
            domain_score_dfs[domain_name] = None

    print("\n--- Performing Statistical Analysis & Building Results Table ---")
    results_summary_data = []

    for domain_config in domains:
        domain_name = domain_config["name"]
        df = domain_score_dfs.get(domain_name)

        if df is None:
            print(f"Skipping analysis for {domain_name} as its scores are unavailable.")
            for metric_idx, metric_name in enumerate(domain_config["metrics"]):
                 results_summary_data.append({
                    "Domain": domain_name, "Metric": metric_name,
                    "Plain": "N/A", "Creative": "N/A",
                    "Delta": "N/A", "P-value": "N/A",
                    "N_Paired": "N/A", "Test_Used": "N/A"
                })
            continue

        print(f"Analyzing domain: {domain_name}")
        for metric_idx, metric_name in enumerate(domain_config["metrics"]):
            plain_col_name = domain_config["score_columns"]["plain"][metric_idx]
            creative_col_name = domain_config["score_columns"]["creative"][metric_idx]

            plain_scores_raw = pd.to_numeric(df[plain_col_name], errors='coerce')
            creative_scores_raw = pd.to_numeric(df[creative_col_name], errors='coerce')

            paired_data = pd.DataFrame({'plain': plain_scores_raw, 'creative': creative_scores_raw}).dropna()
            plain_scores = paired_data['plain']
            creative_scores = paired_data['creative']
            n_paired = len(paired_data)

            if n_paired == 0:
                print(f"  No valid paired data for metric '{metric_name}'. Skipping.")
                results_summary_data.append({
                    "Domain": domain_name, "Metric": metric_name,
                    "Plain": "N/A", "Creative": "N/A",
                    "Delta": "N/A", "P-value": "N/A",
                    "N_Paired": n_paired, "Test_Used": "None"
                })
                continue

            mean_plain_score = plain_scores.mean()
            mean_creative_score = creative_scores.mean()
            p_value_str = "N/A"
            test_used = "N/A"

            if domain_config["metric_type"] == "binary":
                delta = (mean_creative_score - mean_plain_score) * 100
                plain_formatted = f"{mean_plain_score:.2%}"
                creative_formatted = f"{mean_creative_score:.2%}"
                delta_formatted = f"{delta:+.2f}pp"
                test_used = "McNemar"

                if STATSMODELS_AVAILABLE and n_paired > 0:
                    n_01 = np.sum((plain_scores == 0) & (creative_scores == 1))
                    n_10 = np.sum((plain_scores == 1) & (creative_scores == 0))

                    if n_01 + n_10 >= 10 :
                        table = [[np.sum((plain_scores == 1) & (creative_scores == 1)), n_10],
                                 [n_01, np.sum((plain_scores == 0) & (creative_scores == 0))]]
                        try:
                            mcnemar_result = mcnemar(table, exact=False)
                            p_value_str = f"{mcnemar_result.pvalue:.3f}"
                        except Exception as e:
                            p_value_str = f"Error" # Avoid printing full error in table
                            print(f"  Error during McNemar test for {domain_name}/{metric_name}: {e}")
                    else:
                        p_value_str = "N/A (low N)"
                        test_used += " (low N)"
                else:
                    p_value_str = "N/A (statsmodels)" if not STATSMODELS_AVAILABLE else "N/A (no data)"
                    test_used = "McNemar (unavailable)" if not STATSMODELS_AVAILABLE else "McNemar (no data)"

            elif domain_config["metric_type"] == "continuous":
                delta = mean_creative_score - mean_plain_score
                plain_formatted = f"{mean_plain_score:.2f}"
                creative_formatted = f"{mean_creative_score:.2f}"
                delta_formatted = f"{delta:+.2f}"
                test_used = "Wilcoxon"
                if n_paired > 0:
                    if np.all(plain_scores == creative_scores) or len(np.unique(creative_scores - plain_scores)) == 1 and np.unique(creative_scores - plain_scores)[0] == 0:
                        p_value_str = "1.000 (no diff)"
                    elif n_paired < 10 :
                        p_value_str = "N/A (low N)"
                        test_used += " (low N)"
                    else:
                        try:
                            # Ensure there's variance in differences for Wilcoxon
                            diffs = creative_scores - plain_scores
                            if np.all(diffs == 0):
                                 p_value_str = "1.000 (no diff)"
                            else:
                                stat, p_val = wilcoxon(creative_scores, plain_scores, zero_method='wilcox', correction=False, alternative='two-sided')
                                p_value_str = f"{p_val:.3f}"
                        except ValueError as ve:
                             p_value_str = "N/A (error)"
                             print(f"  Error during Wilcoxon test for {domain_name}/{metric_name}: {ve}")

            results_summary_data.append({
                "Domain": domain_name, "Metric": metric_name,
                "Plain": plain_formatted, "Creative": creative_formatted,
                "Delta": delta_formatted, "P-value": p_value_str,
                "N_Paired": n_paired, "Test_Used": test_used
            })

    summary_df = pd.DataFrame(results_summary_data)
    summary_csv_path = os.path.join(args.eval_output_dir, f"results_summary_{sanitized_model_name_for_files}.csv")
    summary_md_path = os.path.join(args.eval_output_dir, f"results_summary_{sanitized_model_name_for_files}.md")

    try:
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nResults summary table saved to {summary_csv_path}")
        summary_df.to_markdown(summary_md_path, index=False)
        print(f"Results summary table saved to {summary_md_path}")
    except Exception as e:
        print(f"Error saving summary tables: {e}")

    print("\n--- Creative-Gone-Wrong Snippets ---")
    print(f"Manual step: Review generated outputs (e.g., in {args.generated_data_dir}) "
          f"and scorer outputs (in {scorer_outputs_dir}) "
          "to select 2-3 'creative-gone-wrong' snippets per domain for your report.")

if __name__ == "__main__":
    main()
