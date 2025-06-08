from datasets import load_dataset
import argparse, json, os, sys

def main():
    parser = argparse.ArgumentParser(
        description="Download TruthfulQA questions (generation/validation) as JSONL"
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output file")
    args = parser.parse_args()

    out_dir   = "data/truthfulqa"
    out_path  = os.path.join(out_dir, "truthfulqa_questions.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path) and not args.force:
        sys.exit(f"{out_path} exists. Use --force to overwrite.")

    print("Loading dataset …")
    ds = load_dataset("truthful_qa", "generation",
                      split="validation", trust_remote_code=True)

    print(f"Found {len(ds):,} rows. Writing to {out_path} …")
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            q = row.get("question")          # 🔹 1. use lower-case key
            if not q:
                print(f"Warning: no question at idx={idx}. Skipping.")
                continue
            json.dump({"idx": idx, "question": q}, f)  # 🔹 2. write it
            f.write("\n")
            written += 1

    print(f"Done: wrote {written} questions.")
    if written == 0:
        print("⚠️  Zero rows written – check that the split really contains data.")

if __name__ == "__main__":
    main()