import os
import requests
import zipfile
import io
import pandas as pd
import json
import random

def setup_socialchem_dataset():
    output_dir = "data/socialchem"
    os.makedirs(output_dir, exist_ok=True)
    zip_url = "https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/social-chem-101.zip"
    tsv_filename = "social-chem-101.v1.0.tsv"
    sampled_output_path = os.path.join(output_dir, "socialchem_sampled_30.jsonl")

    print(f"Downloading {zip_url}...")
    try:
        response = requests.get(zip_url, timeout=60)
        response.raise_for_status()

        print(f"Extracting {tsv_filename} from zip file...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # The zip file might contain a directory structure, e.g., 'social-chem-101/social-chem-101.v1.0.tsv'
            # Let's find the correct path to the tsv file within the zip.
            correct_tsv_path = None
            for name in z.namelist():
                if name.endswith(tsv_filename):
                    correct_tsv_path = name
                    break

            if not correct_tsv_path:
                print(f"Error: Could not find {tsv_filename} in the zip file. Contents: {z.namelist()}")
                raise FileNotFoundError(f"{tsv_filename} not found in zip.")

            with z.open(correct_tsv_path) as tsv_file:
                # Read the TSV data into a pandas DataFrame
                # The README mentions it's tab-separated.
                df = pd.read_csv(tsv_file, sep='\t')

        print(f"Successfully loaded {tsv_filename} into a DataFrame. Number of rows: {len(df)}")

        # Filter out rows where 'rot' or 'situation' might be NaN, if any, though unlikely for core fields.
        df_filtered = df.dropna(subset=['rot', 'situation'])
        if len(df_filtered) < 30:
            print(f"Warning: After dropping NaN, only {len(df_filtered)} valid vignettes available. Sampling all of them.")
            num_to_sample = len(df_filtered)
        else:
            num_to_sample = 30

        # Randomly sample 30 vignettes
        # Ensure reproducibility if needed by setting random.seed() or using df.sample(n, random_state=...)
        sampled_df = df_filtered.sample(n=num_to_sample, random_state=42) # Using a fixed random_state for reproducibility

        print(f"Successfully sampled {len(sampled_df)} vignettes.")

        # Save the sampled data (situation and rot) to a JSONL file
        print(f"Saving sampled vignettes to {sampled_output_path}...")
        with open(sampled_output_path, 'w') as f:
            for _, row in sampled_df.iterrows():
                # We need 'situation' and 'rot' (Rule-of-Thumb)
                # The problem description also refers to it as "moral dilemmas" for situation
                # and "gold Rule-of-Thumb (ROT)" for rot.
                # The 'rot' column in the dataset is the "Rule of thumb written by the worker".
                # The 'situation' column is "Text of the situation".
                record = {
                    "situation_id": row.get("situation-short-id", ""), # Good to keep an ID if available
                    "situation": row["situation"],
                    "rot": row["rot"]
                }
                f.write(json.dumps(record) + '\n')

        print(f"Successfully saved {len(sampled_df)} sampled vignettes to {sampled_output_path}")
        file_size = os.path.getsize(sampled_output_path)
        print(f"File size: {file_size} bytes")

    except Exception as e:
        print(f"Error during Social-Chem-101 setup: {e}")
        raise

if __name__ == "__main__":
    setup_socialchem_dataset()
