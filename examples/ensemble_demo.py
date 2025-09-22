# tests/test_text_detector_ensemble_compare.py
# -*- coding: utf-8 -*-
"""
Auto-run: So sánh detect_image_sliding (baseline) vs detect_image_sliding_ensemble (recall boost)
- Không cần main/CLI. Sửa hằng số bên dưới rồi chạy file.

Kết quả:
- Ảnh annotate & JSON cho từng phương pháp
- log so sánh số lượng region & số region "mới" (khác biệt đáng kể theo IoU)
"""

import os, sys, json
from typing import List, Tuple

# ================== CẤU HÌNH NHANH ==================
# Có thể điền 1 file, nhiều file, hoặc 1 thư mục.
INPUTS = [
    "train/images/TCE20224_00.jpg",  # nếu là thư mục, script sẽ quét hết ảnh bên trong
    # "samples/long_vertical.jpg",
]
OUTDIR = "output/compare_sliding_ensemble"
RESIZE_W = 1400  # 0 = không resize khi lưu ảnh annotate

# ---- cấu hình cho baseline (detect_image_sliding) ----
BASE_TILE_H      = 1600
BASE_OVERLAP     = 0.15
BASE_MINW_UP     = 1000
BASE_MAX_UP      = 2.0
BASE_NMS_IOU     = 0.35

# ---- cấu hình cho ensemble (detect_image_sliding_ensemble) ----
ENS_TILE_HEIGHTS = (1600, 2000)
ENS_OVERLAPS     = (0.15, 0.22)
ENS_MIN_WIDTHS   = (1000, 1300)
ENS_ROTATIONS    = (-3, 0, 3)
ENS_NMS_IOU_MERGE= 0.50
ENS_TEXT_TH      = 0.58
ENS_LOW_TEXT     = 0.30
ENS_LINK_TH      = 0.30
ENS_UPSCALE_LIM  = 2.0

# ---- cấu hình EasyOCR (khởi tạo agent) ----
USE_GPU       = None     # None: auto; True/False
PARAGRAPH     = False    # False để chi tiết hơn
TEXT_TH       = 0.6      # nhẹ nhàng ưu tiên recall so với mặc định 0.7
LOW_TEXT      = 0.3
LINK_TH       = 0.3
CONTRAST_THS  = 0.05
ADJ_CONTRAST  = 0.7
MAG_RATIO     = 2.0
# ====================================================

# Cho phép import src/*
sys.path.append(os.path.abspath("."))
from src.ocr_main.text_detector import KoreanTextDetectorAgent  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_img(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXTS


def _collect_images(inputs: List[str]) -> List[str]:
    bucket: List[str] = []
    for path in inputs:
        if os.path.isfile(path) and _is_img(path):
            bucket.append(path)
        elif os.path.isdir(path):
            for d, _, files in os.walk(path):
                for f in files:
                    p = os.path.join(d, f)
                    if _is_img(p):
                        bucket.append(p)
    return sorted(set(bucket))


def _poly_to_bbox(poly) -> Tuple[float, float, float, float]:
    import numpy as np
    pts = np.asarray(poly, dtype=np.float32)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = a
    X1, Y1, X2, Y2 = b
    iw = max(0.0, min(x2, X2) - max(x1, X1))
    ih = max(0.0, min(y2, Y2) - max(y1, Y1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    ub = max(0.0, (X2 - X1)) * max(0.0, (Y2 - Y1))
    return inter / max(ua + ub - inter, 1e-6)


def _count_new_regions(ens_regions, base_regions, iou_thr=0.5) -> int:
    """
    Đếm số region trong ensemble KHÔNG bị "trùng" (IoU≥thr) với bất kỳ region baseline nào.
    """
    base_bboxes = [_poly_to_bbox(r["box"]) for r in base_regions]
    cnt = 0
    for r in ens_regions:
        bb = _poly_to_bbox(r["box"])
        if not base_bboxes:
            cnt += 1
            continue
        overlap = max(_iou(bb, b) for b in base_bboxes)
        if overlap < iou_thr:
            cnt += 1
    return cnt


# ---------------------- AUTO RUN ----------------------
images = _collect_images(INPUTS)
if not images:
    print("[!] Không tìm thấy ảnh hợp lệ trong INPUTS. Hãy kiểm tra lại.")
    raise SystemExit(1)

os.makedirs(OUTDIR, exist_ok=True)

agent = KoreanTextDetectorAgent(
    use_gpu=USE_GPU,
    detector="craft",
    recognizer="standard",
    paragraph=PARAGRAPH,
    text_threshold=TEXT_TH,
    low_text=LOW_TEXT,
    link_threshold=LINK_TH,
    contrast_ths=CONTRAST_THS,
    adjust_contrast=ADJ_CONTRAST,
    slope_ths=0.1,
    ycenter_ths=0.5,
    height_ths=0.5,
    width_ths=0.5,
    mag_ratio=MAG_RATIO,
)

print(f"[*] Compare sliding vs ensemble | #images={len(images)} | OUTDIR={OUTDIR}")

summary = []

for i, img_path in enumerate(images, 1):
    base = os.path.splitext(os.path.basename(img_path))[0]

    # ---- Baseline: sliding ----
    payload_base = agent.detect_image_sliding(
        img_path,
        tile_height=BASE_TILE_H,
        overlap_ratio=BASE_OVERLAP,
        min_width_upscale=BASE_MINW_UP,
        max_upscale=BASE_MAX_UP,
        nms_iou=BASE_NMS_IOU,
    )
    regs_base = payload_base["regions"]

    out_img_base = os.path.join(OUTDIR, f"{base}_sliding.jpg")
    out_json_base = os.path.join(OUTDIR, f"{base}_sliding.json")

    agent.export_annotated(
        img_path,
        regs_base,
        out_img_base,
        resize_to_width=(RESIZE_W if RESIZE_W and RESIZE_W > 0 else None),
        draw_bbox=True,
        draw_quad=True,
    )
    agent.save_json({"image": img_path, "regions": regs_base}, out_json_base)

    # ---- Ensemble ----
    payload_ens = agent.detect_image_sliding_ensemble(
        img_path,
        tile_heights=ENS_TILE_HEIGHTS,
        overlaps=ENS_OVERLAPS,
        min_widths=ENS_MIN_WIDTHS,
        rotations_deg=ENS_ROTATIONS,
        nms_iou_merge=ENS_NMS_IOU_MERGE,
        rt_text_threshold=ENS_TEXT_TH,
        rt_low_text=ENS_LOW_TEXT,
        rt_link_threshold=ENS_LINK_TH,
        upscale_limit=ENS_UPSCALE_LIM,
    )
    regs_ens = payload_ens["regions"]

    out_img_ens = os.path.join(OUTDIR, f"{base}_ensemble.jpg")
    out_json_ens = os.path.join(OUTDIR, f"{base}_ensemble.json")

    agent.export_annotated(
        img_path,
        regs_ens,
        out_img_ens,
        resize_to_width=(RESIZE_W if RESIZE_W and RESIZE_W > 0 else None),
        draw_bbox=True,
        draw_quad=True,
    )
    agent.save_json({"image": img_path, "regions": regs_ens}, out_json_ens)

    # ---- So sánh nhanh ----
    nb = len(regs_base)
    ne = len(regs_ens)
    new_from_ens = _count_new_regions(regs_ens, regs_base, iou_thr=0.5)

    print(f"[{i:03d}/{len(images)}] {os.path.basename(img_path)} -> "
          f"sliding={nb}, ensemble={ne}, +new≈{new_from_ens} | "
          f"img: {os.path.basename(out_img_base)} vs {os.path.basename(out_img_ens)}")

    summary.append({
        "image": img_path,
        "sliding_regions": nb,
        "ensemble_regions": ne,
        "ensemble_new_regions_est": new_from_ens,
        "sliding_out_img": out_img_base,
        "sliding_out_json": out_json_base,
        "ensemble_out_img": out_img_ens,
        "ensemble_out_json": out_json_ens,
    })

# Lưu tổng kết
sum_path = os.path.join(OUTDIR, "summary_compare.json")
with open(sum_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"[*] Summary saved: {sum_path}")
