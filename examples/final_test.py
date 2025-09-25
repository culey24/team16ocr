# # tests/test_ocr_pipeline_sliding_autorun.py
# # -*- coding: utf-8 -*-
# """
# Auto-run OCR pipeline (SLIDING ONLY): Detector (EasyOCR) -> Recognizer (EasyOCR)
# - Không cần main/CLI. Chỉnh hằng số bên dưới rồi chạy.

# Kết quả:
# - output/det_sliding : ảnh + JSON vùng detect
# - output/rec_sliding : ảnh + JSON nhận dạng
# """

# import os, sys, json
# from typing import List

# # ================== CẤU HÌNH NHANH ==================
# INPUTS = [
#     "train/images/TAF20253_00.jpg",            # thư mục hoặc file ảnh đơn lẻ
#     # "samples/long_vertical.jpg",
# ]
# OUT_DET_DIR = "output/det_sliding"
# OUT_REC_DIR = "output/rec_sliding"
# RESIZE_W    = 1400  # 0 = không resize khi lưu ảnh annotate

# # ---- cấu hình detector: SLIDING ----
# SL_TILE_H      = 1600
# SL_OVERLAP     = 0.18    # tăng nhẹ overlap để giảm miss khi cắt ngang chữ
# SL_MINW_UP     = 1200    # upscale nếu ảnh quá hẹp
# SL_MAX_UP      = 2.0
# SL_NMS_IOU     = 0.40    # nới IoU để giữ box sát nhau

# # ---- cấu hình EasyOCR chung ----
# USE_GPU       = None     # None: auto; True/False
# LANGS         = ["ko"]   # thêm "en" nếu ảnh có trộn tiếng Anh: ["ko","en"]
# PARAGRAPH     = False
# TEXT_TH       = 0.6
# LOW_TEXT      = 0.3
# LINK_TH       = 0.3
# CONTRAST_THS  = 0.05
# ADJ_CONTRAST  = 0.7
# MAG_RATIO     = 2.0

# # ---- recognizer extra ----
# ALLOWLIST = None  # ví dụ: "0123456789-+/%."
# BLOCKLIST = None
# PAD_RATIO  = 0.08
# MIN_CROP_H = 12
# # ====================================================

# # Cho phép import src/*
# sys.path.append(os.path.abspath("."))
# from src.ocr_main.text_detector import KoreanTextDetectorAgent   # noqa: E402
# from src.ocr_main.text_recognizer import KoreanTextRecognizerAgent  # noqa: E402

# IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# def _is_img(p: str) -> bool:
#     return os.path.splitext(p)[1].lower() in IMG_EXTS

# def _collect_images(inputs: List[str]) -> List[str]:
#     bucket: List[str] = []
#     for path in inputs:
#         if os.path.isfile(path) and _is_img(path):
#             bucket.append(path)
#         elif os.path.isdir(path):
#             for d, _, files in os.walk(path):
#                 for f in files:
#                     p = os.path.join(d, f)
#                     if _is_img(p):
#                         bucket.append(p)
#     return sorted(set(bucket))

# # ---------------------- AUTO RUN ----------------------
# images = _collect_images(INPUTS)
# if not images:
#     print("[!] Không tìm thấy ảnh hợp lệ trong INPUTS.")
#     raise SystemExit(1)

# os.makedirs(OUT_DET_DIR, exist_ok=True)
# os.makedirs(OUT_REC_DIR, exist_ok=True)

# # Init agents (dùng cùng tham số để hành vi nhất quán)
# det = KoreanTextDetectorAgent(
#     use_gpu=USE_GPU,
#     detector="craft",
#     recognizer="standard",
#     paragraph=PARAGRAPH,
#     text_threshold=TEXT_TH,
#     low_text=LOW_TEXT,
#     link_threshold=LINK_TH,
#     contrast_ths=CONTRAST_THS,
#     adjust_contrast=ADJ_CONTRAST,
#     slope_ths=0.1,
#     ycenter_ths=0.5,
#     height_ths=0.5,
#     width_ths=0.5,
#     mag_ratio=MAG_RATIO,
# )

# rec = KoreanTextRecognizerAgent(
#     use_gpu=USE_GPU,
#     languages=LANGS,
#     detector="craft",
#     recognizer="standard",
#     paragraph=PARAGRAPH,
#     text_threshold=TEXT_TH,
#     low_text=LOW_TEXT,
#     link_threshold=LINK_TH,
#     contrast_ths=CONTRAST_THS,
#     adjust_contrast=ADJ_CONTRAST,
#     slope_ths=0.1,
#     ycenter_ths=0.5,
#     height_ths=0.5,
#     width_ths=0.5,
#     mag_ratio=MAG_RATIO,
#     allowlist=ALLOWLIST,
#     blocklist=BLOCKLIST,
# )

# print(f"[*] OCR pipeline (SLIDING) | #images={len(images)} | OUT_DET={OUT_DET_DIR} | OUT_REC={OUT_REC_DIR}")

# summary = []

# for i, img_path in enumerate(images, 1):
#     base = os.path.splitext(os.path.basename(img_path))[0]

#     # ---------- Detect (SLIDING) ----------
#     payload_det = det.detect_image_sliding(
#         img_path,
#         tile_height=SL_TILE_H,
#         overlap_ratio=SL_OVERLAP,
#         min_width_upscale=SL_MINW_UP,
#         max_upscale=SL_MAX_UP,
#         nms_iou=SL_NMS_IOU,
#     )
#     regions = payload_det["regions"]

#     det_img  = os.path.join(OUT_DET_DIR, f"{base}__sliding.jpg")
#     det_json = os.path.join(OUT_DET_DIR, f"{base}__sliding.json")
#     det.export_annotated(
#         img_path, regions, det_img,
#         resize_to_width=(RESIZE_W if RESIZE_W and RESIZE_W > 0 else None),
#         draw_bbox=True, draw_quad=True
#     )
#     det.save_json({"image": img_path, "regions": regions}, det_json)

#     # ---------- Recognize ----------
#     payload_rec = rec.recognize_with_regions(
#         image_path=img_path,
#         regions=regions,
#         pad_ratio=PAD_RATIO,
#         min_crop_h=MIN_CROP_H,
#     )
#     results = payload_rec["results"]

#     rec_img  = os.path.join(OUT_REC_DIR, f"{base}__sliding_rec.jpg")
#     rec_json = os.path.join(OUT_REC_DIR, f"{base}__sliding_rec.json")
#     rec.annotate_to_file(img_path, results, out_img=rec_img)

#     # resize preview nếu cần
#     if RESIZE_W and RESIZE_W > 0:
#         import cv2
#         vis = cv2.imread(rec_img, cv2.IMREAD_COLOR)
#         if vis is not None:
#             h, w = vis.shape[:2]
#             scale = RESIZE_W / max(w, 1)
#             vis2 = cv2.resize(vis, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
#             cv2.imwrite(rec_img, vis2)

#     rec.save_json(payload_rec, out_json=rec_json)

#     print(f"[{i:03d}/{len(images)}] {os.path.basename(img_path)} -> "
#           f"regions: {len(regions)} | lines: {len(results)} | "
#           f"DET: {os.path.basename(det_img)} | REC: {os.path.basename(rec_img)}")

#     summary.append({
#         "image": img_path,
#         "regions": len(regions),
#         "recognized": len(results),
#         "det_img": det_img,
#         "det_json": det_json,
#         "rec_img": rec_img,
#         "rec_json": rec_json,
#         "method": "sliding",
#     })

# # Lưu tổng kết
# sum_path = os.path.join(OUT_REC_DIR, "summary_pipeline_sliding.json")
# with open(sum_path, "w", encoding="utf-8") as f:
#     json.dump(summary, f, ensure_ascii=False, indent=2)
# print(f"[*] Done. Summary saved: {sum_path}")

# tests/test_ocr_pipeline_sliding_autorun.py
# -*- coding: utf-8 -*-
"""
Auto-run OCR pipeline (SLIDING ONLY): Detector (EasyOCR) -> Recognizer (EasyOCR)
- Không cần main/CLI. Chỉnh hằng số bên dưới rồi chạy.

Kết quả:
- output/det_sliding : ảnh + JSON vùng detect
- output/rec_sliding : ảnh + JSON nhận dạng
"""

import os, sys, json
import argparse  # THÊM
from typing import List

# ================== CẤU HÌNH NHANH ==================
OUT_DET_DIR = "output/det_sliding"
OUT_REC_DIR = "output/rec_sliding"
RESIZE_W    = 1400  # 0 = không resize khi lưu ảnh annotate

# ---- cấu hình detector: SLIDING ----
SL_TILE_H      = 1600
SL_OVERLAP     = 0.18    # tăng nhẹ overlap để giảm miss khi cắt ngang chữ
SL_MINW_UP     = 1200    # upscale nếu ảnh quá hẹp
SL_MAX_UP      = 2.0
SL_NMS_IOU     = 0.40    # nới IoU để giữ box sát nhau

# ---- cấu hình EasyOCR chung ----
USE_GPU       = None     # None: auto; True/False
LANGS         = ["ko"]   # thêm "en" nếu ảnh có trộn tiếng Anh: ["ko","en"]
PARAGRAPH     = False
TEXT_TH       = 0.6
LOW_TEXT      = 0.3
LINK_TH       = 0.3
CONTRAST_THS  = 0.05
ADJ_CONTRAST  = 0.7
MAG_RATIO     = 2.0

# ---- recognizer extra ----
ALLOWLIST = None  # ví dụ: "0123456789-+/%."
BLOCKLIST = None
PAD_RATIO  = 0.08
MIN_CROP_H = 12
# ====================================================

# Cho phép import src/*
sys.path.append(os.path.abspath("."))
from src.ocr_main.text_detector import KoreanTextDetectorAgent   # noqa: E402
from src.ocr_main.text_recognizer import KoreanTextRecognizerAgent  # noqa: E402

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

# ---------------------- AUTO RUN ----------------------
def main():
    parser = argparse.ArgumentParser(description="OCR Pipeline")
    parser.add_argument("--input_dir", type=str, default="train/images", help="Input directory or file")
    args = parser.parse_args()

    INPUTS = [args.input_dir]  # Sử dụng input từ args
    images = _collect_images(INPUTS)
    if not images:
        print("[!] Không tìm thấy ảnh hợp lệ trong INPUTS.")
        raise SystemExit(1)

    os.makedirs(OUT_DET_DIR, exist_ok=True)
    os.makedirs(OUT_REC_DIR, exist_ok=True)

    # Init agents (dùng cùng tham số để hành vi nhất quán)
    det = KoreanTextDetectorAgent(
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

    rec = KoreanTextRecognizerAgent(
        use_gpu=USE_GPU,
        languages=LANGS,
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
        allowlist=ALLOWLIST,
        blocklist=BLOCKLIST,
    )

    print(f"[*] OCR pipeline (SLIDING) | #images={len(images)} | OUT_DET={OUT_DET_DIR} | OUT_REC={OUT_REC_DIR}")

    summary = []

    for i, img_path in enumerate(images, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]

        # ---------- Detect (SLIDING) ----------
        payload_det = det.detect_image_sliding(
            img_path,
            tile_height=SL_TILE_H,
            overlap_ratio=SL_OVERLAP,
            min_width_upscale=SL_MINW_UP,
            max_upscale=SL_MAX_UP,
            nms_iou=SL_NMS_IOU,
        )
        regions = payload_det["regions"]

        det_img  = os.path.join(OUT_DET_DIR, f"{base}__sliding.jpg")
        det_json = os.path.join(OUT_DET_DIR, f"{base}__sliding.json")
        det.export_annotated(
            img_path, regions, det_img,
            resize_to_width=(RESIZE_W if RESIZE_W and RESIZE_W > 0 else None),
            draw_bbox=True, draw_quad=True
        )
        # THÊM: Resize det_img nếu cần
        if RESIZE_W and RESIZE_W > 0:
            import cv2
            vis = cv2.imread(det_img, cv2.IMREAD_COLOR)
            if vis is not None:
                h, w = vis.shape[:2]
                scale = RESIZE_W / max(w, 1)
                vis2 = cv2.resize(vis, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(det_img, vis2)
        det.save_json({"image": img_path, "regions": regions}, det_json)

        # ---------- Recognize ----------
        payload_rec = rec.recognize_with_regions(
            image_path=img_path,
            regions=regions,
            pad_ratio=PAD_RATIO,
            min_crop_h=MIN_CROP_H,
        )
        results = payload_rec["results"]

        rec_img  = os.path.join(OUT_REC_DIR, f"{base}__sliding_rec.jpg")
        rec_json = os.path.join(OUT_REC_DIR, f"{base}__sliding_rec.json")
        rec.annotate_to_file(img_path, results, out_img=rec_img)

        # resize preview nếu cần
        if RESIZE_W and RESIZE_W > 0:
            import cv2
            vis = cv2.imread(rec_img, cv2.IMREAD_COLOR)
            if vis is not None:
                h, w = vis.shape[:2]
                scale = RESIZE_W / max(w, 1)
                vis2 = cv2.resize(vis, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(rec_img, vis2)

        rec.save_json(payload_rec, out_json=rec_json)

        print(f"[{i:03d}/{len(images)}] {os.path.basename(img_path)} -> "
              f"regions: {len(regions)} | lines: {len(results)} | "
              f"DET: {os.path.basename(det_img)} | REC: {os.path.basename(rec_img)}")

        summary.append({
            "image": img_path,
            "regions": len(regions),
            "recognized": len(results),
            "det_img": det_img,
            "det_json": det_json,
            "rec_img": rec_img,
            "rec_json": rec_json,
            "method": "sliding",
        })

    # Lưu tổng kết
    sum_path = os.path.join(OUT_REC_DIR, "summary_pipeline_sliding.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[*] Done. Summary saved: {sum_path}")

if __name__ == "__main__":
    main()