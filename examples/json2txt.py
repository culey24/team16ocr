#!/usr/bin/env python3
import json
import sys
import os
import argparse  # THÊM

def recjson_to_txt(json_path, txt_path, min_conf=0.0):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # rec_json format: {"image": "...", "results": [{"box": ..., "text": "...", "conf": ...}, ...]}
    results = data.get("results", [])

    with open(txt_path, "w", encoding="utf-8") as out:
        for item in results:
            text = item.get("text", "").strip()
            conf = float(item.get("conf", 0.0))
            if text and conf >= min_conf:
                out.write(text + "\n")

    print(f"[OK] Saved {len(results)} lines (filtered by conf >= {min_conf}) to {txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Input JSON file")
    parser.add_argument("txt_path", help="Output TXT file")
    parser.add_argument("--min_conf", type=float, default=0.0, help="Minimum confidence to include text")
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: {args.json_path} not found")
        sys.exit(1)

    recjson_to_txt(args.json_path, args.txt_path, args.min_conf)