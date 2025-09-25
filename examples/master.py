import os
import subprocess
import argparse

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Full OCR Pipeline")
    parser.add_argument("--input_dir", required=True, help="Input images dir")
    parser.add_argument("--preprocess_dir", default="output/preprocessed", help="Preprocessed images dir")
    args = parser.parse_args()

    # Step 1: Preprocess
    run_command(["python", "preprocess_for_detector.py", "--input_dir", args.input_dir, "--output_dir", args.preprocess_dir])

    # Step 2: OCR (sử dụng preprocessed images làm input)
    run_command(["python", "final_test.py", "--input_dir", args.preprocess_dir])

    # Step 3: Postprocess (batch)
    run_command(["python", "postprocess_with_llm_batch_parallel.py"])

if __name__ == "__main__":
    main()