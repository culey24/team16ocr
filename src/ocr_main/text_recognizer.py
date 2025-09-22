# src/ocr_main/text_recognizer_agent.py
# -*- coding: utf-8 -*-
"""
Korean Text Recognizer Agent (EasyOCR - recognize only)

Mục tiêu:
- NHẬN DẠNG văn bản từ:
  (A) Ảnh gốc + list vùng (quad) đã detect (từ TextDetectorAgent),
  (B) Hoặc các crop ảnh dòng/word riêng lẻ.
- Trả về cấu trúc giống ocr_agent.py: [{"box": quad, "text": str, "conf": float}, ...]

API mẫu:
    rec = KoreanTextRecognizerAgent(use_gpu=True)

    # 1) Nhận dạng từ ảnh + regions (quad)
    det_payload = detector.detect_image_sliding_ensemble("samples/long.jpg", ...)
    rec_payload = rec.recognize_with_regions(det_payload["image"], det_payload["regions"])
    rec.annotate_to_file(rec_payload["image"], rec_payload["results"], out_img="output/rec_annotated.jpg")
    rec.save_json(rec_payload, out_json="output/rec_results.json")

    # 2) Nhận dạng 1 ảnh crop (một dòng)
    text = rec.predict_line("samples/crops/line_01.png")

    # 3) Nhận dạng nhiều crop (batch)
    lines = rec.predict_lines(["samples/crops/1.png", "samples/crops/2.png"])
"""

import os
import json
import unicodedata
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import easyocr


class KoreanTextRecognizerAgent:
    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        languages: Optional[List[str]] = None,
        detector: str = "craft",          # vẫn khởi tạo Reader “đủ bộ” để readtext chạy ổn định
        recognizer: str = "standard",
        # Các tham số ảnh hưởng chất lượng nhận dạng (đồng bộ với ocr_agent)
        paragraph: bool = False,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        contrast_ths: float = 0.1,
        adjust_contrast: float = 0.5,
        slope_ths: float = 0.1,
        ycenter_ths: float = 0.5,
        height_ths: float = 0.5,
        width_ths: float = 0.5,
        mag_ratio: float = 1.5,
        # Tùy chọn lọc ký tự (EasyOCR)
        allowlist: Optional[str] = None,
        blocklist: Optional[str] = None,
    ):
        """
        - use_gpu: None -> auto GPU; True/False để ép.
        - languages: list mã ngôn ngữ EasyOCR; mặc định ["ko"].
        - allowlist/blocklist: lọc ký tự cho recognizer (ví dụ allowlist="0123456789-+/%").
        """
        if use_gpu is None:
            use_gpu = False
            try:
                import torch  # noqa
                import torch.cuda  # noqa
                use_gpu = bool(torch.cuda.is_available())
            except Exception:
                use_gpu = False

        if languages is None:
            languages = ["ko"]

        self.reader = easyocr.Reader(
            languages, gpu=use_gpu, detector=detector, recognizer=recognizer
        )

        # Cấu hình chuyển cho readtext (như ocr_agent để hành vi nhất quán)
        self.read_cfg = dict(
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            paragraph=paragraph,
            contrast_ths=contrast_ths,
            adjust_contrast=adjust_contrast,
            slope_ths=slope_ths,
            ycenter_ths=ycenter_ths,
            height_ths=height_ths,
            width_ths=width_ths,
            mag_ratio=mag_ratio,
            # lọc ký tự:
            allowlist=allowlist,
            blocklist=blocklist,
        )

    # ---------------------------------------------------------------------
    # Core 1: Recognize with regions (từ ảnh gốc + quad list)
    # ---------------------------------------------------------------------
    def recognize_with_regions(
        self,
        image_path: str,
        regions: List[Dict[str, Any]],
        pad_ratio: float = 0.08,
        min_crop_h: int = 12,
    ) -> Dict[str, Any]:
        """
        Nhận dạng text bằng cách:
        - Warping mỗi quad -> crop “thẳng” (perspective transform).
        - Chạy readtext trên crop để lấy text + conf.
        - Trả về list kết quả theo cùng thứ tự regions.

        pad_ratio: đệm thêm quanh quad (tính theo kích thước bbox của quad) để giữ dấu/diacritics.
        min_crop_h: bỏ qua crop quá nhỏ (px) để tránh noise.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"OpenCV cannot read image: {image_path}")

        H, W = img.shape[:2]
        results: List[Dict[str, Any]] = []

        for r in regions:
            quad = r.get("box")
            if not quad or len(quad) < 4:
                continue

            crop = self._warp_quad(img, quad, pad_ratio=pad_ratio)
            if crop is None:
                continue

            # Bỏ crop quá nhỏ (khó nhận dạng)
            if crop.shape[0] < min_crop_h and crop.shape[1] < min_crop_h:
                continue

            # Gọi readtext trên crop (small patch)
            try:
                # detail=0 -> chỉ text; detail=1 -> (box,text,conf) trên crop
                # Dùng detail=1 để lấy conf; sau đó chọn mục có conf cao nhất (thường chỉ 1 vùng)
                res = self.reader.readtext(crop, detail=1, **self.read_cfg)
            except Exception:
                res = []

            if not res:
                # Không nhận ra gì -> text rỗng, conf None (vẫn giữ box cũ)
                results.append({"box": self._to_float_quad(quad), "text": "", "conf": None})
                continue

            # Lấy best line trong crop theo conf
            best_text, best_conf = "", 0.0
            for item in res:
                try:
                    _box_c, text_c, conf_c = item
                except Exception:
                    continue
                if text_c is None:
                    continue
                t = unicodedata.normalize("NFC", str(text_c)).strip()
                try:
                    c = float(conf_c)
                except Exception:
                    c = 0.0
                if c >= best_conf and t:
                    best_conf = c
                    best_text = t

            results.append({
                "box": self._to_float_quad(quad),
                "text": best_text,
                "conf": (float(best_conf) if best_text else None)
            })

        return {"image": image_path, "results": results}

    # ---------------------------------------------------------------------
    # Core 2: Predict line(s) từ crop trực tiếp
    # ---------------------------------------------------------------------
    def predict_line(self, image_path: str) -> str:
        """
        Nhận dạng 1 ảnh đã crop (một dòng/cụm).
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
        res = self.reader.readtext(image_path, detail=1, **self.read_cfg)
        best_text, best_conf = "", 0.0
        for item in res:
            try:
                _box, text, conf = item
            except Exception:
                continue
            if text is None:
                continue
            t = unicodedata.normalize("NFC", str(text)).strip()
            try:
                c = float(conf)
            except Exception:
                c = 0.0
            if c >= best_conf and t:
                best_conf, best_text = c, t
        return best_text

    def predict_lines(self, image_paths: List[str]) -> List[str]:
        """
        Nhận dạng nhiều crop; trả mảng text theo thứ tự input.
        """
        out: List[str] = []
        for p in image_paths:
            try:
                out.append(self.predict_line(p))
            except Exception:
                out.append("")
        return out

    # ---------------------------------------------------------------------
    # Utils: annotate & IO
    # ---------------------------------------------------------------------
    @staticmethod
    def annotate(
        img_bgr: np.ndarray,
        results: List[Dict[str, Any]],
        draw_bbox: bool = True,
        draw_quad: bool = True,
        box_color=(0, 200, 200),
        quad_color=(0, 200, 0),
        text_color=(0, 0, 255),
    ) -> np.ndarray:
        """
        Vẽ kết quả nhận dạng (giống ocr_agent.py): box + text + conf.
        """
        vis = img_bgr.copy()
        for idx, r in enumerate(results, 1):
            pts = np.asarray(r.get("box", []), dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 4:
                continue
            pts_i = pts.astype(np.int32)
            x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())

            if draw_quad:
                cv2.polylines(vis, [pts_i], isClosed=True, color=quad_color, thickness=2)
            if draw_bbox:
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

            label = f"#{idx}"
            if r.get("text"):
                label += f": {unicodedata.normalize('NFC', r['text'])}"
            if r.get("conf") is not None:
                try:
                    label += f" ({float(r['conf']):.2f})"
                except Exception:
                    pass

            cv2.putText(
                vis, label, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA
            )
        return vis

    def annotate_to_file(
        self,
        image_path: str,
        results: List[Dict[str, Any]],
        out_img: str = "output/rec_annotated.jpg",
        **annotate_kwargs,
    ) -> str:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"OpenCV cannot read image: {image_path}")
        vis = self.annotate(img, results, **annotate_kwargs)
        os.makedirs(os.path.dirname(out_img) or ".", exist_ok=True)
        cv2.imwrite(out_img, vis)
        return out_img

    @staticmethod
    def save_json(payload: Dict[str, Any], out_json: str = "output/rec_results.json") -> str:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_json

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _order_quad(quad: List[List[float]]) -> np.ndarray:
        """
        Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left.
        """
        pts = np.asarray(quad, dtype=np.float32)
        if pts.shape[0] > 4:
            pts = pts[:4]
        # dựa trên tổng & hiệu toạ độ
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _warp_quad(self, img: np.ndarray, quad: List[List[float]], pad_ratio: float = 0.08) -> Optional[np.ndarray]:
        """
        Perspective transform quad -> ảnh “thẳng” để recognizer dễ đọc.
        pad_ratio: đệm quanh bbox trước khi warp để giữ dấu/diacritics.
        """
        H, W = img.shape[:2]
        pts = self._order_quad(quad)

        # kích thước mục tiêu (định theo khoảng cách 2 cạnh đối diện)
        def _dist(a, b): return float(np.linalg.norm(a - b))
        w1 = _dist(pts[0], pts[1]); w2 = _dist(pts[3], pts[2]); Wt = max(int(max(w1, w2)), 1)
        h1 = _dist(pts[0], pts[3]); h2 = _dist(pts[1], pts[2]); Ht = max(int(max(h1, h2)), 1)

        src = pts.astype(np.float32)
        dst = np.array([[0, 0], [Wt - 1, 0], [Wt - 1, Ht - 1], [0, Ht - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (Wt, Ht), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        if pad_ratio and pad_ratio > 0:
            pad_x = int(Wt * pad_ratio)
            pad_y = int(Ht * pad_ratio)
            warped = cv2.copyMakeBorder(
                warped, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
            )

        return warped

    @staticmethod
    def _to_float_quad(quad: List[List[Any]]) -> List[List[float]]:
        out = []
        for p in quad[:4]:
            try:
                out.append([float(p[0]), float(p[1])])
            except Exception:
                pass
        return out
