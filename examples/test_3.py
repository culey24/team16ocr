# examples/test_3.py
# -*- coding: utf-8 -*-
"""
Test single-image OCR using KoreanEasyOcrAgent (EasyOCR, GPU preferred).
- Import từ: src.ocr_main.ocr_agent (class KoreanEasyOcrAgent)
- Chỉnh IMAGE_PATH rồi chạy: python -m examples.test_3
"""

import os
import json
import cv2

# Nếu bạn đã đặt agent EasyOCR theo hướng dẫn trước đó:
from src.ocr_main.ocr_agent import KoreanEasyOcrAgent
# Nếu file ocr_agent.py nằm cạnh file test, thay bằng:
# from ocr_agent import KoreanEasyOcrAgent

# =============================
# CONFIG ———> CHỈNH Ở ĐÂY
# =============================
IMAGE_PATH = "train/images/TAF20191_01.jpg" # ← ảnh đầu vào
OUT_DIR    = "output"
OUT_IMG    = os.path.join(OUT_DIR, "easy_annotated.jpg")
OUT_JSON   = os.path.join(OUT_DIR, "easy_results.json")

# =============================
# RUN (module-as-script friendly)
# =============================
if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(f"IMAGE_PATH not found: {IMAGE_PATH}")

# Ưu tiên GPU; nếu không khởi tạo được (thiếu torch CUDA v.v.), tự fallback về CPU.
try:
    agent = KoreanEasyOcrAgent(use_gpu=True)
except Exception as e:
    print("[WARN] GPU init failed, fallback to CPU:", e)
    agent = KoreanEasyOcrAgent(use_gpu=False)

# 1) OCR
payload = agent.ocr_image(IMAGE_PATH)  # {"image": str, "results": [...]}

# 2) Lưu JSON
os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

# 3) Vẽ & lưu ảnh
img = cv2.imread(payload["image"], cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError("OpenCV cannot read image: " + payload["image"])

vis = agent.annotate(
    img,
    payload["results"],
    draw_bbox=True,   # ô chữ nhật bao
    draw_quad=True,   # polygon 4 đỉnh theo detector
    box_color=(0, 200, 200),
    quad_color=(0, 200, 0),
    text_color=(0, 0, 255),
)

cv2.imwrite(OUT_IMG, vis)

print("Saved image  ->", OUT_IMG)
print("Saved JSON   ->", OUT_JSON)
print(f"Detected {len(payload['results'])} text boxes.")
