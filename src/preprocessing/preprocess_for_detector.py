# # #Preprocessing hình ảnh: grayscale, clahe, denoise, crop
# # #Usage:     python preprocess_for_detector.py \
# # #           --input_dir - vị trí file ảnh \
# # #           --output_dir - vị trí lưu ảnh sau khi xử lý \
# # #           -- limit - mặc định 0 (tất cả)

# # from __future__ import annotations
# # import cv2, numpy as np, json
# # from pathlib import Path
# # import argparse

# # IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}

# # # ======== IO ========
# # def imread_any(path: Path):
# #     return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

# # def imwrite_png(path: Path, img):
# #     path.parent.mkdir(parents=True, exist_ok=True)
# #     cv2.imencode(".png", img)[1].tofile(str(path))

# # # ======== ops (fixed settings) ========
# # def to_gray(img) -> np.ndarray:
# #     if img is None: return img
# #     if len(img.shape) == 2: return img
# #     if img.shape[2] == 4:
# #         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
# #     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # def clahe_light(gray: np.ndarray) -> np.ndarray:
# #     clahe = cv2.createCLAHE(clipLimit=10.5, tileGridSize=(8,8))
# #     return clahe.apply(gray)

# # def denoise_light(gray: np.ndarray) -> np.ndarray:
# #     # Gaussian 3x3 để không làm bệt nét mảnh
# #     return cv2.GaussianBlur(gray, (3,3), 0)

# # def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
# #     # dùng để tạo mask cho crop (không dùng làm ảnh đầu ra)
# #     block, C = 31, 15
# #     return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #                                  cv2.THRESH_BINARY, block, C)

# # def crop_text_region(gray: np.ndarray) -> np.ndarray:
# #     """Crop nhẹ vùng chữ bằng morphology; không xoay."""
# #     thr = adaptive_binarize(gray)
# #     mask = 255 - thr
# #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
# #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
# #     cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     if not cnts: return gray
# #     x0,y0,w0,h0 = cv2.boundingRect(np.vstack(cnts))
# #     H,W = gray.shape[:2]
# #     if w0*h0 < 0.03*H*W:  # vùng quá nhỏ -> bỏ crop
# #         return gray
# #     pad = int(round(0.02 * max(W,H)))  # padding nhẹ 2%
# #     x1,y1 = max(0,x0-pad), max(0,y0-pad)
# #     x2,y2 = min(W,x0+w0+pad), min(H,y0+h0+pad)
# #     return gray[y1:y2, x1:x2]

# # # ======== pipeline ========
# # def process_one(img_path: Path):
# #     raw = imread_any(img_path)
# #     if raw is None:
# #         return None, {"status":"read_error"}

# #     # giữ kích thước gốc
# #     gray = to_gray(raw)
# #     gray = clahe_light(gray)
# #     gray = denoise_light(gray)
# #     gray = crop_text_region(gray)   # crop nhẹ

# #     return gray, {"status":"ok"}

# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--input_dir", required=True)
# #     ap.add_argument("--output_dir", required=True)
# #     ap.add_argument("--limit", type=int, default=0, help="Số ảnh tối đa để xử lý (0 = tất cả)")
# #     args = ap.parse_args()

# #     in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
# #     out_dir.mkdir(parents=True, exist_ok=True)

# #     files = [p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
# #     files.sort()
# #     if args.limit > 0:
# #         files = files[:args.limit]

# #     manifest = []
# #     for p in files:
# #         img, meta = process_one(p)
# #         if img is None:
# #             manifest.append({"src":str(p), "dst":None, "status":"error"}); continue
# #         rel = p.relative_to(in_dir)
# #         dst = (out_dir / rel).with_suffix(".png")
# #         imwrite_png(dst, img)
# #         item = {"src":str(p), "dst":str(dst)}
# #         item.update(meta)
# #         manifest.append(item)

# #     (out_dir/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
# #     print(f"Processed {len(manifest)} images → {out_dir}")

# # if __name__ == "__main__":
# #     main()

# # Preprocessing hình ảnh: grayscale, clahe, denoise, crop
# # Usage: python preprocess_for_easyocr.py \
# #        --input_dir <vị trí folder ảnh gốc> \
# #        --output_dir <vị trí lưu ảnh sau xử lý> \
# #        --limit 0 (tất cả ảnh)

# # Preprocessing hình ảnh: grayscale, clahe, denoise, crop
# # Usage: python preprocess_for_easyocr.py \
# #        --input_dir <vị trí folder ảnh gốc> \
# #        --output_dir <vị trí lưu ảnh sau xử lý> \
# #        --limit 0 (tất cả ảnh)

# from __future__ import annotations
# import cv2, numpy as np, json
# from pathlib import Path
# import argparse

# IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}

# # ======== IO ========
# def imread_any(path: Path):
#     return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

# def imwrite_png(path: Path, img):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imencode(".png", img)[1].tofile(str(path))

# # ======== ops (fixed settings) ========
# def to_gray(img) -> np.ndarray:
#     if img is None: return img
#     if len(img.shape) == 2: return img
#     if img.shape[2] == 4:
#         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
#     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# def clahe_light(gray: np.ndarray) -> np.ndarray:
#     clahe = cv2.createCLAHE(clipLimit=10.5, tileGridSize=(8,8))
#     return clahe.apply(gray)

# def denoise_light(gray: np.ndarray) -> np.ndarray:
#     # Gaussian 3x3 để không làm bệt nét mảnh
#     return cv2.GaussianBlur(gray, (3,3), 0)

# def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
#     # dùng để tạo mask cho crop (không dùng làm ảnh đầu ra)
#     block, C = 31, 15
#     return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                  cv2.THRESH_BINARY, block, C)

# def crop_text_region(gray: np.ndarray) -> np.ndarray:
#     """Crop nhẹ vùng chữ bằng morphology; không xoay."""
#     thr = adaptive_binarize(gray)
#     mask = 255 - thr
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
#     cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts: return gray
#     x0,y0,w0,h0 = cv2.boundingRect(np.vstack(cnts))
#     H,W = gray.shape[:2]
#     if w0*h0 < 0.03*H*W:  # vùng quá nhỏ -> bỏ crop
#         return gray
#     pad = int(round(0.02 * max(W,H)))  # padding nhẹ 2%
#     x1,y1 = max(0,x0-pad), max(0,y0-pad)
#     x2,y2 = min(W,x0+w0+pad), min(H,y0+h0+pad)
#     return gray[y1:y2, x1:x2]

# # ======== pipeline ========
# def process_one(img_path: Path):
#     raw = imread_any(img_path)
#     if raw is None:
#         return None, {"status":"read_error"}

#     # giữ kích thước gốc
#     gray = to_gray(raw)
#     gray = clahe_light(gray)
#     gray = denoise_light(gray)
#     gray = crop_text_region(gray)   # crop nhẹ

#     return gray, {"status":"ok"}

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--input_dir", required=True)
#     ap.add_argument("--output_dir", required=True)
#     ap.add_argument("--limit", type=int, default=0, help="Số ảnh tối đa để xử lý (0 = tất cả)")
#     args = ap.parse_args()

#     in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     files = [p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
#     files.sort()
#     if args.limit > 0:
#         files = files[:args.limit]

#     manifest = []
#     for p in files:
#         img, meta = process_one(p)
#         if img is None:
#             manifest.append({"src":str(p), "dst":None, "status":"error"}); continue
#         rel = p.relative_to(in_dir)
#         dst = (out_dir / rel).with_suffix(".png")
#         imwrite_png(dst, img)
#         item = {"src":str(p), "dst":str(dst)}
#         item.update(meta)
#         manifest.append(item)

#     (out_dir/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
#     print(f"Processed {len(manifest)} images → {out_dir}")

# if __name__ == "__main__":
#     main()

# Preprocessing hình ảnh: grayscale, clahe, denoise, crop
# Usage: python preprocess_for_detector.py \
#        --input_dir <vị trí folder ảnh gốc> \
#        --output_dir <vị trí lưu ảnh sau xử lý> \
#        --limit 0 (tất cả ảnh)

from __future__ import annotations
import cv2, numpy as np, json
from pathlib import Path
import argparse

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}

# ======== IO ========
def imread_any(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

def imwrite_png(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", img)[1].tofile(str(path))

# ======== ops (fixed settings) ========
def to_gray(img) -> np.ndarray:
    if img is None: return img
    if len(img.shape) == 2: return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def clahe_light(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=10.5, tileGridSize=(8,8))
    return clahe.apply(gray)

def denoise_light(gray: np.ndarray) -> np.ndarray:
    # Gaussian 3x3 để không làm bệt nét mảnh
    return cv2.GaussianBlur(gray, (3,3), 0)

def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    # dùng để tạo mask cho crop (không dùng làm ảnh đầu ra)
    block, C = 31, 15
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, C)

def crop_text_region(gray: np.ndarray) -> np.ndarray:
    """Crop nhẹ vùng chữ bằng morphology; không xoay."""
    thr = adaptive_binarize(gray)
    mask = 255 - thr
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return gray
    x0,y0,w0,h0 = cv2.boundingRect(np.vstack(cnts))
    H,W = gray.shape[:2]
    if w0*h0 < 0.03*H*W:  # vùng quá nhỏ -> bỏ crop
        return gray
    pad = int(round(0.02 * max(W,H)))  # padding nhẹ 2%
    x1,y1 = max(0,x0-pad), max(0,y0-pad)
    x2,y2 = min(W,x0+w0+pad), min(H,y0+h0+pad)
    return gray[y1:y2, x1:x2]

# ======== pipeline ========
def process_one(img_path: Path):
    raw = imread_any(img_path)
    if raw is None:
        return None, {"status":"read_error"}

    # giữ kích thước gốc
    gray = to_gray(raw)
    gray = clahe_light(gray)
    gray = denoise_light(gray)
    gray = crop_text_region(gray)   # crop nhẹ

    return gray, {"status":"ok"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="Số ảnh tối đa để xử lý (0 = tất cả)")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    files.sort()
    if args.limit > 0:
        files = files[:args.limit]

    manifest = []
    for p in files:
        img, meta = process_one(p)
        if img is None:
            manifest.append({"src":str(p), "dst":None, "status":"error"}); continue
        rel = p.relative_to(in_dir)
        dst = (out_dir / rel).with_suffix(".png")
        imwrite_png(dst, img)
        item = {"src":str(p), "dst":str(dst)}
        item.update(meta)
        manifest.append(item)

    (out_dir/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Processed {len(manifest)} images → {out_dir}")

if __name__ == "__main__":
    main()