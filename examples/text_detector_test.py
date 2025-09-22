# tests/test_text_detector_sliding_autorun.py
# -*- coding: utf-8 -*-
"""
Auto-run: Text detection bằng kỹ thuật "chia tile + cửa sổ trượt".
- Không main/CLI. Chỉ sửa hằng số bên dưới rồi chạy file.
"""

import os, sys, json

# ===== cấu hình nhanh =====
INPUT_IMAGE = "train/images/TCE20224_00.jpg"   # <<< sửa thành ảnh dài của bạn
OUTDIR      = "output/detect_sliding"
RESIZE_W    = 1400                          # 0 = không resize khi lưu ảnh annotate
# Sliding-window
TILE_H      = 1600
OVERLAP     = 0.15
MINW_UP     = 1000
MAX_UP      = 2.0
NMS_IOU     = 0.35
# ==========================

sys.path.append(os.path.abspath("."))
from src.ocr_main.text_detector import KoreanTextDetectorAgent  # noqa: E402

if not os.path.isfile(INPUT_IMAGE):
    print(f"[!] Không tìm thấy ảnh: {INPUT_IMAGE}")
    raise SystemExit(1)

os.makedirs(OUTDIR, exist_ok=True)

agent = KoreanTextDetectorAgent(
    use_gpu=None,                # auto GPU nếu có
    detector="craft",
    recognizer="standard",
    paragraph=False,             # chi tiết hơn (không gộp paragraph)
    text_threshold=0.7,
    low_text=0.4,
    link_threshold=0.4,
    contrast_ths=0.1,
    adjust_contrast=0.5,
    slope_ths=0.1,
    ycenter_ths=0.5,
    height_ths=0.5,
    width_ths=0.5,
    mag_ratio=1.5,
)

payload = agent.detect_image_sliding(
    INPUT_IMAGE,
    tile_height=TILE_H,
    overlap_ratio=OVERLAP,
    min_width_upscale=MINW_UP,
    max_upscale=MAX_UP,
    nms_iou=NMS_IOU,
)
regions = payload["regions"]

base = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]
out_img  = os.path.join(OUTDIR, f"{base}_sliding.jpg")
out_json = os.path.join(OUTDIR, f"{base}_sliding.json")

agent.export_annotated(
    INPUT_IMAGE, regions, out_img,
    resize_to_width=(RESIZE_W if RESIZE_W and RESIZE_W > 0 else None),
    draw_bbox=True, draw_quad=True
)
agent.save_json({"image": INPUT_IMAGE, "regions": regions}, out_json)

print(f"[*] Sliding detect -> regions: {len(regions)} | {out_img}")
