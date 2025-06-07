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

# Import for TruthfulQA direct call
from eval_scorers.truthfulqa_evaluate import main_wrapper as truthfulqa_main_wrapper
# from sentence_transformers import SentenceTransformer # Only if model loaded here

SUPPORTED_MODELS = ["gpt-4o", "claude-3-5-sonnet-20240620", "deepseek-ai/deepseek-r1"]
# For SocialChem, if we were to load model here:
# SOCIALCHEM_MODEL = None
# SOCIALCHEM_MODEL_NAME = 'all-MiniLM-L6-v2'


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
            "name": "AUT", "metrics": ["fluency", "avg_originality"], "scorer_script": "aut_scorer.py", # Metric name changed
            "input_file_template": f"generated_aut_{sanitized_model_name_for_files}.json",
            "score_file_template": f"scores_aut_{sanitized_model_name_for_files}.csv",
            "metric_type": "continuous",
            "score_columns": { # Column names updated to match aut_scorer.py output
                "plain": ["plain_fluency", "plain_avg_originality"],
                "creative": ["creative_fluency", "creative_avg_originality"]
            }
        },
        {
            "name": "TruthfulQA", "metrics": ["bleurt_score"], "scorer_script": "truthfulqa_evaluate.py", # Metric name changed
            "input_file_template": f"generated_truthfulqa_{sanitized_model_name_for_files}.json",
            "score_file_template": f"scores_truthfulqa_{sanitized_model_name_for_files}.csv",
            "metric_type": "continuous", # Changed from binary
            "score_columns": {"plain": ["plain_bleurt_score"], "creative": ["creative_bleurt_score"]} # Column names updated
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
        scorer_script_path = os.path.join(args.scorers_dir, scorer_script_name) # Used for non-TruthfulQA

        input_json_filename = domain_config["input_file_template"]
        input_json_path = os.path.join(args.generated_data_dir, input_json_filename)

        scorer_csv_filename = domain_config["score_file_template"]
        scorer_csv_output_path = os.path.join(scorer_outputs_dir, scorer_csv_filename)

        print(f"Processing domain: {domain_name}")

        if not os.path.exists(input_json_path):
            print(f"  Warning: Input file {input_json_path} not found. Skipping {domain_name}.")
            domain_score_dfs[domain_name] = None
            continue

        if domain_name == "TruthfulQA":
            try:
                print(f"  Special handling for TruthfulQA: Calling wrapper function directly.")
                with open(input_json_path, 'r', encoding='utf-8') as f:
                    truthfulqa_data = json.load(f)

                # Prepare data for plain and creative responses
                # The wrapper expects 'question_id' and 'generated_answer'
                # 'item_id' from our JSON files can serve as 'question_id'

                temp_dir_for_truthfulqa = os.path.join(scorer_outputs_dir, "temp_truthfulqa_inputs")
                os.makedirs(temp_dir_for_truthfulqa, exist_ok=True)

                all_results_for_domain = {} # Store results by item_id

                for response_type in ["plain", "creative"]:
                    if f"error_{response_type}" in truthfulqa_data[0] or "error" in truthfulqa_data[0]: # Check first item for error structure
                         # This assumes if one item has error_plain, all might. A bit simplistic.
                        print(f"  Skipping {response_type} responses for TruthfulQA due to anticipated errors or missing fields.")
                        # Populate with Nones or error markers if needed for merging later
                        for item in truthfulqa_data:
                            item_id = item.get("item_id", "unknown_id")
                            if item_id not in all_results_for_domain:
                                all_results_for_domain[item_id] = {"item_id": item_id}
                            all_results_for_domain[item_id][domain_config["score_columns"][response_type][0]] = np.nan
                        continue


                    temp_csv_data = []
                    for item in truthfulqa_data:
                        item_id = item.get("item_id", "unknown_id")
                        # Ensure 'original_item_dict' and 'Question' exist, otherwise use item_id or a placeholder
                        question_text = item.get("original_item_dict", {}).get("Question", item_id) # Fallback for question text

                        response_key = f"{response_type}_response" # e.g. plain_response
                        generated_answer = item.get(response_key)

                        if generated_answer is not None: # Ensure there is an answer
                             temp_csv_data.append({
                                "question_id": item_id, # Using item_id as question_id
                                "question": question_text, # Official script might use this for context, placeholder doesn't
                                "generated_answer": generated_answer
                            })
                        else: # Handle missing answers by adding NaN scores later
                            if item_id not in all_results_for_domain:
                                all_results_for_domain[item_id] = {"item_id": item_id}
                            all_results_for_domain[item_id][domain_config["score_columns"][response_type][0]] = np.nan


                    if not temp_csv_data:
                        print(f"  No valid {response_type} data to evaluate for TruthfulQA.")
                        # Populate with NaNs for all items if no data for this response_type
                        for item in truthfulqa_data:
                            item_id = item.get("item_id", "unknown_id")
                            if item_id not in all_results_for_domain:
                                all_results_for_domain[item_id] = {"item_id": item_id}
                            all_results_for_domain[item_id][domain_config["score_columns"][response_type][0]] = np.nan
                        continue

                    temp_input_csv_path = os.path.join(temp_dir_for_truthfulqa, f"temp_input_{response_type}.csv")
                    pd.DataFrame(temp_csv_data).to_csv(temp_input_csv_path, index=False)

                    print(f"  Calling TruthfulQA wrapper for {response_type} responses...")
                    # Using "bleurt" as the default metric for now, as per TruthfulQA wrapper's test
                    scores = truthfulqa_main_wrapper(temp_input_csv_path, "bleurt", temp_dir_for_truthfulqa)

                    if scores:
                        for score_entry in scores:
                            item_id = score_entry["question_id"]
                            if item_id not in all_results_for_domain:
                                all_results_for_domain[item_id] = {"item_id": item_id}
                            # Store the score in the correct plain/creative column
                            score_col_name = domain_config["score_columns"][response_type][0] # e.g. plain_bleurt_score
                            all_results_for_domain[item_id][score_col_name] = score_entry["score"]
                    else:
                        print(f"  Warning: No scores returned from TruthfulQA wrapper for {response_type} responses.")
                        # Populate with NaNs if scores are None
                        for item_entry in temp_csv_data: # Use temp_csv_data to get relevant item_ids
                            item_id = item_entry["question_id"]
                            if item_id not in all_results_for_domain:
                                all_results_for_domain[item_id] = {"item_id": item_id}
                            all_results_for_domain[item_id][domain_config["score_columns"][response_type][0]] = np.nan

                # Consolidate all_results_for_domain into a DataFrame and save
                if all_results_for_domain:
                    final_df_data = []
                    # Ensure all items from original json are present and columns are ordered
                    for item in truthfulqa_data:
                        item_id = item.get("item_id", "unknown_id")
                        row_data = all_results_for_domain.get(item_id, {"item_id": item_id})
                        # Ensure all expected columns are present, fill with NaN if not
                        for rt in ["plain", "creative"]:
                            col = domain_config["score_columns"][rt][0]
                            if col not in row_data:
                                row_data[col] = np.nan
                        final_df_data.append(row_data)

                    df = pd.DataFrame(final_df_data)
                    # Reorder columns to be item_id, plain_score_col, creative_score_col
                    ordered_cols = ["item_id"] + [domain_config["score_columns"]["plain"][0]] + [domain_config["score_columns"]["creative"][0]]
                    # Filter out any columns not in df.columns before reindexing
                    ordered_cols = [col for col in ordered_cols if col in df.columns]
                    df = df[ordered_cols]

                else:
                    df = pd.DataFrame(columns=["item_id"] + domain_config["score_columns"]["plain"] + domain_config["score_columns"]["creative"])

                df.to_csv(scorer_csv_output_path, index=False)
                print(f"  TruthfulQA processed scores saved to {scorer_csv_output_path}")
                domain_score_dfs[domain_name] = df

            except Exception as e:
                print(f"  An unexpected error occurred while processing TruthfulQA for {domain_name}: {e}")
                domain_score_dfs[domain_name] = None
        else:
            # Standard subprocess call for other domains
            # Ensure scorer_script_path is absolute or correctly relative
            # os.path.abspath makes it absolute from the current CWD of evaluate.py
            # If evaluate.py is /app/src/evaluate.py, and scorer_script_path is src/eval_scorers/aut_scorer.py
            # then abspath would be /app/src/eval_scorers/aut_scorer.py.
            # This should be fine if subprocess inherits CWD as /app.
            # Let's try making it absolute to be certain.
            # Standard subprocess call for other domains
            abs_scorer_script_path = os.path.abspath(scorer_script_path)

            print(f"  Current CWD for evaluate.py: {os.getcwd()}")
            print(f"  Checking existence of scorer script at: {abs_scorer_script_path}")
            if not os.path.exists(abs_scorer_script_path):
                print(f"  FATAL: Scorer script {abs_scorer_script_path} confirmed NOT FOUND by os.path.exists().")
                domain_score_dfs[domain_name] = None
                continue

            command = [
                "python", abs_scorer_script_path,
                "--input_file", input_json_path,
                "--output_file", scorer_csv_output_path
            ]
            try:
                print(f"  Running scorer: {' '.join(command)}")
                # Pass cwd explicitly to ensure context, though it should inherit /app
                process_result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=os.getcwd())
                print(f"  Scorer output saved to {scorer_csv_output_path}")
                print(f"  Scorer STDOUT:\n{process_result.stdout}")
                if process_result.stderr:
                    print(f"  Scorer STDERR:\n{process_result.stderr}")
                df = pd.read_csv(scorer_csv_output_path)
                domain_score_dfs[domain_name] = df
            except FileNotFoundError as e_fnf:
                print(f"  Error: Output CSV file not found: {scorer_csv_output_path}. This might be due to an error in the scorer script itself. Original error: {e_fnf}")
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
