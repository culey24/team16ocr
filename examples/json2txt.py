#!/usr/bin/env python3
import json
import sys
import os

def recjson_to_txt(json_path, txt_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # rec_json format: {"image": "...", "results": [{"box": ..., "text": "...", "conf": ...}, ...]}
    results = data.get("results", [])

    with open(txt_path, "w", encoding="utf-8") as out:
        for item in results:
            text = item.get("text", "").strip()
            if text:
                out.write(text + "\n")

    print(f"[OK] Saved {len(results)} lines to {txt_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python json2txt.py rec_result.json ocr_output.txt")
        sys.exit(1)

    json_path = sys.argv[1]
    txt_path = sys.argv[2]

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        sys.exit(1)

    recjson_to_txt(json_path, txt_path)
