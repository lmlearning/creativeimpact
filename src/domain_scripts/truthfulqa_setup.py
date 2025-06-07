import os
import requests

def setup_truthfulqa_dataset():
    output_dir = "data/truthfulqa"
    os.makedirs(output_dir, exist_ok=True)
    csv_url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
    output_path = os.path.join(output_dir, "TruthfulQA.csv")

    print(f"Downloading TruthfulQA.csv to {output_path}...")
    try:
        response = requests.get(csv_url, timeout=60)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded TruthfulQA.csv to {output_path}")
        # Check file size
        file_size = os.path.getsize(output_path)
        print(f"File size: {file_size} bytes")
        if file_size < 1000: # TruthfulQA.csv should be much larger
             print("Warning: Downloaded file is very small, check for issues.")
    except Exception as e:
        print(f"Error downloading TruthfulQA dataset: {e}")
        raise

if __name__ == "__main__":
    setup_truthfulqa_dataset()
