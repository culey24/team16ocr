#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process OCR JSON results with optional LLM correction (Gemini or OpenAI).
Outputs:
 - final_cleaned.json : structured output with original, cleaned, final_text, flags
 - ocr_output.txt      : one final_text per line (use with ocr_eval_20250903.py)
 - review_queue.json   : flagged items for manual review
"""

import os
import sys
import json
import time
import re
import unicodedata
from typing import Dict, List, Optional
from difflib import SequenceMatcher

# ---------------------------
# Config
# ---------------------------
CONFIDENCE_THRESHOLD = 0.75       # below this => consider LLM
LLM_CONFIDENCE_THRESHOLD = 0.6   # LLM's returned "confidence" threshold to accept
MIN_LEN_FOR_LLM = 20             # text length >= this => consider LLM
LLM_RATE_LIMIT_SEC = 1.0         # seconds between LLM calls (adjust to your quota)
BACKEND = os.environ.get("LLM_BACKEND", "").lower()  # "gemini" or "openai" or ""
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------
# Helpers
# ---------------------------
def normalize_text(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r'[\x00-\x1f\x7f]', '', s)  # ctrl chars
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def has_spacing_issues(text: str) -> bool:
    if not text: return False
    if len(text) > 15 and ' ' not in text:
        return True
    # Korean heuristic: many hangul but very few spaces -> spacing problem
    korean_chars = len(re.findall(r'[가-힣]', text))
    if korean_chars > 10 and len(text.split()) < 3:
        return True
    return False

def numbers_preserved(orig: str, sug: str) -> bool:
    o_nums = re.findall(r'\d+', orig)
    s_nums = re.findall(r'\d+', sug)
    return o_nums == s_nums

def simple_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# ---------------------------
# LLM call wrappers
# ---------------------------
import requests

def call_gemini(prompt: str, api_key: str) -> Optional[Dict]:
    """Call Gemini-like HTTP API (simple wrapper)."""
    if not api_key:
        return None
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.05, "maxOutputTokens": 300}
    }
    try:
        resp = requests.post(f"{url}?key={api_key}", json=payload, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        suggested = j["candidates"][0]["content"]["parts"][0]["text"].strip()
        # crude confidence estimation
        sim = simple_similarity(prompt.splitlines()[-1], suggested)
        confidence = 0.9 if 0.7 <= sim <= 0.95 else (0.8 if sim > 0.95 else (0.7 if sim > 0.5 else 0.45))
        return {"suggested": suggested, "confidence": confidence, "raw": j}
    except Exception as e:
        print("Gemini call failed:", e, file=sys.stderr)
        return None

def call_openai(prompt: str, api_key: str) -> Optional[Dict]:
    """Call OpenAI Chat API (compatible wrapper)."""
    if not api_key:
        return None
    try:
        import openai
        openai.api_key = api_key
        # Use chat completion
        system = "You are a helpful assistant that corrects OCR'd Korean text. Only output the corrected sentence."
        messages = [{"role":"system","content":system}, {"role":"user","content":prompt}]
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages, temperature=0.05, max_tokens=300)
        suggested = resp["choices"][0]["message"]["content"].strip()
        sim = simple_similarity(prompt.splitlines()[-1], suggested)
        confidence = 0.9 if 0.7 <= sim <= 0.95 else (0.8 if sim > 0.95 else (0.7 if sim > 0.5 else 0.45))
        return {"suggested": suggested, "confidence": confidence, "raw": resp}
    except Exception as e:
        print("OpenAI call failed:", e, file=sys.stderr)
        return None

# ---------------------------
# Main processing
# ---------------------------
def should_call_llm(text: str, conf: float) -> bool:
    if BACKEND == "":
        return False
    if conf < CONFIDENCE_THRESHOLD:
        return True
    if len(text.strip()) >= MIN_LEN_FOR_LLM:
        return True
    if has_spacing_issues(text):
        return True
    return False

def build_prompt_for_llm(text: str) -> str:
    # Prompt instructs LLM to correct spacing & OCR errors but preserve numbers.
    return (
        f"다음은 OCR로 인식된 한국어 문장입니다:\n\"{text}\"\n\n"
        "작업:\n"
        "1) 맞춤법 및 띄어쓰기를 교정하세요.\n"
        "2) OCR로 인한 문자 인식 오류를 정정하세요 (가능한 경우).\n"
        "3) 숫자나 날짜, 고유명사는 변경하지 마세요.\n"
        "4) 원문을 최대한 존중하여 수정된 문장만 출력하세요 (추가 설명 금지).\n"
        "출력: 수정된 문장만\n"
    )

def process_input_file(input_json_path: str, output_json_path: str, out_txt_path: str):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Data may be a list of image-results or single object; normalize
    if isinstance(data, dict) and "image" in data and "results" in data:
        images = [data]
    elif isinstance(data, list):
        images = data
    else:
        raise ValueError("Unsupported input format. Expected list or dict with image+results.")

    final_results = []
    review_queue = []
    all_lines = []

    last_llm_call = 0.0

    for img in images:
        img_path = img.get("image", "")
        output_img = {"image": img_path, "detections": []}
        for det in img.get("results", []):
            orig_text = det.get("text", "")
            conf = float(det.get("conf", det.get("score", 0.0))) if det.get("conf", None) is not None else float(det.get("score", 0.0))
            norm = normalize_text(orig_text)
            final_text = norm
            flags = []

            if should_call_llm(norm, conf):
                prompt = build_prompt_for_llm(norm)
                # rate limit
                wait = LLM_RATE_LIMIT_SEC - (time.time() - last_llm_call)
                if wait > 0:
                    time.sleep(wait)
                llm_resp = None
                if BACKEND == "gemini":
                    llm_resp = call_gemini(prompt, GEMINI_KEY)
                elif BACKEND == "openai":
                    llm_resp = call_openai(prompt, OPENAI_KEY)
                last_llm_call = time.time()

                if llm_resp and llm_resp.get("suggested"):
                    sug = llm_resp["suggested"].strip()
                    if numbers_preserved(norm, sug):
                        if llm_resp.get("confidence", 0.0) >= LLM_CONFIDENCE_THRESHOLD:
                            final_text = sug
                            flags.append("llm_corrected")
                        else:
                            flags.append("llm_low_confidence")
                    else:
                        flags.append("llm_rejected_numbers")
                else:
                    flags.append("llm_failed")

            # simple heuristic flags
            if conf < 0.5:
                flags.append("very_low_conf")
            elif conf < CONFIDENCE_THRESHOLD:
                flags.append("low_conf")

            if len(final_text.strip()) < 2:
                flags.append("too_short")

            # append
            det_out = {
                "poly": det.get("box", det.get("poly", [])),
                "original_text": orig_text,
                "cleaned_text": norm,
                "final_text": final_text,
                "confidence": conf,
                "flags": flags
            }
            output_img["detections"].append(det_out)
            final_results.append({"image": img_path, **det_out})
            all_lines.append(final_text)

            if flags:
                review_queue.append({"image": img_path, **det_out})

    # Save final json
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    # Save output txt (one line per detection). If you prefer one line per image, adapt easily.
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line.replace("\n", " ") + "\n")

    # Save review queue
    with open("review_queue.json", "w", encoding="utf-8") as f:
        json.dump(review_queue, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved {len(final_results)} detections to {output_json_path}")
    print(f"[OK] Saved text lines to {out_txt_path}")
    print(f"[OK] Review queue: {len(review_queue)} items -> review_queue.json")

# ---------------------------
# CLI
# ---------------------------
def usage_and_exit():
    print("Usage: python postprocess_with_llm.py input_rec.json final_cleaned.json ocr_output.txt")
    print("Set environment variables:")
    print("  LLM_BACKEND=gemini|openai  (optional)")
    print("  GEMINI_API_KEY=... or OPENAI_API_KEY=...")
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        usage_and_exit()
    input_json = sys.argv[1]
    out_json = sys.argv[2]
    out_txt = sys.argv[3]
    process_input_file(input_json, out_json, out_txt)
