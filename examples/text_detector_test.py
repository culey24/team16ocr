# tests/test_text_detector_autorun.py
# -*- coding: utf-8 -*-
"""
Auto-run text detection test (no main, no CLI args).
- Quét ảnh trong INPUT_ROOT (mặc định "samples")
- Gọi KoreanTextDetectorAgent.detect_image(...)
- Xuất ảnh annotate (.jpg) + JSON kết quả
- Không mở cửa sổ hiển thị

Chạy:
  python tests/test_text_detector_autorun.py
Muốn đổi thư mục input/output, sửa các hằng số ngay bên dưới.
"""

import os, sys, json
from typing import List

# ================== CẤU HÌNH NHANH ==================
INPUT_ROOT = "train/images/TAF20251_00.jpg"              # Ảnh đơn hoặc thư mục ảnh
OUTDIR      = "output/detect_vis"   # Thư mục xuất ảnh annotate + JSON
RESIZE_W    = 1400                  # 0 = không resize; >0 = resize theo chiều rộng trước khi lưu
USE_GPU     = None                  # None: auto; True: ép GPU; False: ép CPU
DRAW_BBOX   = True                  # Vẽ bbox axis-aligned
DRAW_QUAD   = True                  # Vẽ polygon theo 4 đỉnh
# ====================================================

# Cho phép import src/*
sys.path.append(os.path.abspath("."))
from src.ocr_main.text_detector import KoreanTextDetectorAgent  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_img(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXTS


def _collect_images(root: str) -> List[str]:
    if os.path.isfile(root):
        return [root] if _is_img(root) else []
    found: List[str] = []
    if os.path.isdir(root):
        for d, _, files in os.walk(root):
            for f in files:
                p = os.path.join(d, f)
                if _is_img(p):
                    found.append(p)
    return sorted(found)


# ---------------------- AUTO RUN ----------------------
images = _collect_images(INPUT_ROOT)

# Fallback nhanh nếu rỗng
if not images:
    for candidate in ["sample.jpg", "sample.png"]:
        if os.path.isfile(candidate) and _is_img(candidate):
            images = [candidate]
            break

if not images:
    print(f"[!] Không tìm thấy ảnh hợp lệ trong '{INPUT_ROOT}'. "
          f"Đặt ảnh vào '{INPUT_ROOT}/' hoặc sửa INPUT_ROOT ở đầu file.")
    raise SystemExit(1)

os.makedirs(OUTDIR, exist_ok=True)

agent = KoreanTextDetectorAgent(
    use_gpu=USE_GPU,
    detector="craft",
    recognizer="standard",  # không dùng text, nhưng giữ giống ocr_agent để đồng nhất hành vi
    text_threshold=0.7,
    low_text=0.4,
    link_threshold=0.4,
    paragraph=False,
    contrast_ths=0.1,
    adjust_contrast=0.5,
    slope_ths=0.1,
    ycenter_ths=0.5,
    height_ths=0.5,
    width_ths=0.5,
    mag_ratio=1.5,
)

print(f"[*] Auto-run | Tổng ảnh: {len(images)} | OUTDIR: {OUTDIR}")

summary = []
for i, img_path in enumerate(images, 1):
    try:
        payload = agent.detect_image(img_path)
        regions = payload.get("regions", [])
        base = os.path.splitext(os.path.basename(img_path))[0]

        out_img  = os.path.join(OUTDIR, f"{base}_det.jpg")
        out_json = os.path.join(OUTDIR, f"{base}_det.json")

        agent.export_annotated(
            image_path=img_path,
            regions=regions,
            out_img=out_img,
            resize_to_width=(RESIZE_W if RESIZE_W and RESIZE_W > 0 else None),
            draw_bbox=DRAW_BBOX,
            draw_quad=DRAW_QUAD,
        )
        agent.save_json({"image": img_path, "regions": regions}, out_json=out_json)

        print(f"[{i:04d}/{len(images)}] {os.path.basename(img_path)} -> "
              f"regions: {len(regions)} | img: {out_img} | json: {out_json}")

        summary.append({
            "image": img_path,
            "regions_count": len(regions),
            "out_img": out_img,
            "out_json": out_json
        })
    except Exception as e:
        print(f"[ERR] {img_path}: {e}")

# Lưu tổng hợp
sum_path = os.path.join(OUTDIR, "summary.json")
with open(sum_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"[*] Summary saved: {sum_path}")
# ------------------------------------------------------
