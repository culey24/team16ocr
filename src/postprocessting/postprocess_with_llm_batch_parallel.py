#!/usr/bin/env python3
"""
Batch wrapper to run postprocess_with_llm.py on all JSON files in a folder
and aggregate results into:
 - final_cleaned_all.json
 - ocr_output_all.txt
 - review_queue_all.json
"""
import os
import sys
import json
import glob
import tempfile
import subprocess
from pathlib import Path
from typing import List

# ==== CONFIG ====
REC_DIR = "output/rec_sliding"   # folder chứa các rec json
OUT_DIR = "output/rec_sliding"   # nơi lưu aggregated outputs (cùng folder mặc định)
POSTPROCESS_SCRIPT = "postprocess_with_llm.py"  # nếu nằm ở repo root
# =================

def find_json_files(folder: str) -> List[str]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"{folder} không tồn tại")
    files = sorted([str(x) for x in p.glob("*.json")])
    return files

def run_postprocess_via_subprocess(in_json: str, out_json: str, out_txt: str) -> bool:
    """Call the script as a subprocess. Return True on OK."""
    cmd = [sys.executable, POSTPROCESS_SCRIPT, in_json, out_json, out_txt]
    try:
        r = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(r.stdout)
        if r.stderr:
            print("stderr:", r.stderr, file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERR] subprocess failed for {in_json}: {e}\n{e.stderr}", file=sys.stderr)
        return False

def aggregate_files(list_of_jsons: List[str], aggregate_json_path: str, aggregate_txt_path: str, aggregate_review_path: str):
    """Aggregate per-file outputs (json list, txt lines, review_queue) into single files."""
    agg_results = []
    agg_lines = []
    agg_review = []

    for jpath in list_of_jsons:
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # data expected to be a list of detection dicts (final_results from postprocess)
            if isinstance(data, list):
                agg_results.extend(data)
            elif isinstance(data, dict):
                agg_results.append(data)
        except Exception as e:
            print(f"[WARN] skip reading {jpath}: {e}", file=sys.stderr)
            continue

    # For each detection entry, extract final_text to lines and flags for review
    for det in agg_results:
        txt = det.get("final_text") or det.get("cleaned_text") or det.get("original_text") or ""
        txt = txt.replace("\n", " ").strip()
        if txt:
            agg_lines.append(txt)
        if det.get("flags"):
            agg_review.append(det)

    # Write aggregated outputs
    with open(aggregate_json_path, "w", encoding="utf-8") as f:
        json.dump(agg_results, f, ensure_ascii=False, indent=2)
    with open(aggregate_txt_path, "w", encoding="utf-8") as f:
        for line in agg_lines:
            f.write(line + "\n")
    with open(aggregate_review_path, "w", encoding="utf-8") as f:
        json.dump(agg_review, f, ensure_ascii=False, indent=2)

    print(f"[OK] Aggregated {len(agg_results)} detections -> {aggregate_json_path}")
    print(f"[OK] {len(agg_lines)} text lines -> {aggregate_txt_path}")
    print(f"[OK] {len(agg_review)} flagged items -> {aggregate_review_path}")

def main():
    json_files = find_json_files(REC_DIR)
    if not json_files:
        print(f"[!] Không tìm thấy file .json trong {REC_DIR}")
        return

    temp_outputs = []
    for j in json_files:
        basename = Path(j).stem
        # create per-file temp outputs inside OUT_DIR to make debugging easier
        out_json = os.path.join(OUT_DIR, f"{basename}__cleaned_temp.json")
        out_txt  = os.path.join(OUT_DIR, f"{basename}__ocr_temp.txt")

        # Try to import and call function directly if possible
        called = False
        try:
            # Try import postprocess module
            spec_path = Path(POSTPROCESS_SCRIPT)
            if spec_path.exists():
                # call via subprocess (safer because postprocess writes files and expects top-level __main__)
                called = run_postprocess_via_subprocess(j, out_json, out_txt)
            else:
                # try import as module (e.g., postprocess_with_llm in PYTHONPATH)
                from postprocess_with_llm import process_input_file  # type: ignore
                process_input_file(j, out_json, out_txt)
                called = True
        except Exception as e:
            print(f"[WARN] import/call direct failed: {e}. Falling back to subprocess.", file=sys.stderr)
            called = run_postprocess_via_subprocess(j, out_json, out_txt)

        if called:
            temp_outputs.append((out_json, out_txt))
        else:
            print(f"[ERR] Failed to process {j}. Skipping.", file=sys.stderr)

    # Now aggregate all per-file out_json into a single aggregated json and txt
    per_file_jsons = [p for p, t in temp_outputs if os.path.exists(p)]
    agg_json = os.path.join(OUT_DIR, "final_cleaned_all.json")
    agg_txt  = os.path.join(OUT_DIR, "ocr_output_all.txt")
    agg_review = os.path.join(OUT_DIR, "review_queue_all.json")

    aggregate_files(per_file_jsons, agg_json, agg_txt, agg_review)

    # Optional: cleanup temp files
    for p, t in temp_outputs:
        try:
            if os.path.exists(p):
                os.remove(p)
            if os.path.exists(t):
                os.remove(t)
        except Exception:
            pass

if __name__ == "__main__":
    main()
