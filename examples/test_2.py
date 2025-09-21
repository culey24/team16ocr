# -*- coding: utf-8 -*-
"""
Simple test for KoreanOcrAgent
- Không cần argparse, không cần main().
- Chỉ chạy 1 lần: load model, predict trên ảnh mẫu.
"""

from src.ocr_main.ocr_agent import KoreanOcrAgent

# ảnh test
image_path = "train/images/TAF20161_00.png"

# tạo agent (nếu chưa có model fine-tune, tự fallback trocr-base-printed)
agent = KoreanOcrAgent()

print(f"[INFO] Using model source: {agent.used_source}")

# dự đoán text từ ảnh
pred = agent.predict_image(image_path)
print(f"[PRED] {pred}")
