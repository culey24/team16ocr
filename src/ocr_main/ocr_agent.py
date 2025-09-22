# src/ocr_main/ocr_agent.py
# -*- coding: utf-8 -*-
"""
Korean OCR Agent (EasyOCR)
- Dùng EasyOCR (ko) cho detect + recognize.
- API:
    agent = KoreanEasyOcrAgent(use_gpu=True)  # auto GPU nếu có, hoặc False để dùng CPU
    payload = agent.ocr_image("/path/to/img.jpg")
    agent.annotate_to_file(payload["image"], payload["results"], out_img="output/easy_annotated.jpg")
    agent.save_json(payload, out_json="output/easy_results.json")
"""

import os
import json
import unicodedata
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import easyocr


class KoreanEasyOcrAgent:
    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        detector: str = "craft",
        recognizer: str = "standard",  # giữ mặc định EasyOCR
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        paragraph: bool = False,
        contrast_ths: float = 0.1,
        adjust_contrast: float = 0.5,
        slope_ths: float = 0.1,
        ycenter_ths: float = 0.5,
        height_ths: float = 0.5,
        width_ths: float = 0.5,
        mag_ratio: float = 1.5,
    ):
        """
        Ghi chú:
        - use_gpu: None -> tự động dùng GPU nếu có; True/False để ép.
        - các ngưỡng trên map tới tham số của easyocr.readtext(...) để tinh chỉnh detect/recall.
        """
        if use_gpu is None:
            use_gpu = False
            try:
                import torch  # noqa
                import torch.cuda  # noqa
                use_gpu = bool(torch.cuda.is_available())
            except Exception:
                use_gpu = False

        self.reader = easyocr.Reader(
            ["ko"], gpu=use_gpu, detector=detector, recognizer=recognizer
        )

        # lưu config readtext
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
        )

    # -------- Core OCR --------
    def ocr_image(self, image_path: str) -> Dict[str, Any]:
        """
        Detect + recognize cả ảnh.
        Trả về: {"image": str, "results": [{"box": [[x,y]x4], "text": str, "conf": float}, ...]}
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

        # EasyOCR trả list các tuple: (box, text, conf)
        # box: list 4 điểm [[x1,y1],...]
        res = self.reader.readtext(image_path, detail=1, **self.read_cfg)

        results: List[Dict[str, Any]] = []
        for item in res:
            try:
                box, text, conf = item
            except Exception:
                # Một số version có thể trả khác dạng, cứ bỏ qua phần tử không đúng
                continue
            text = unicodedata.normalize("NFC", str(text)).strip()
            # ép float
            try:
                box = [[float(p[0]), float(p[1])] for p in box]
                conf = float(conf)
            except Exception:
                continue
            results.append({"box": box, "text": text, "conf": conf})

        return {"image": image_path, "results": results}

    def predict_line(self, image_path: str) -> str:
        """
        Nhận dạng ảnh đã crop 1 dòng (vẫn dùng readtext và ghép text).
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
        res = self.reader.readtext(image_path, detail=1, **self.read_cfg)
        parts: List[str] = []
        for item in res:
            try:
                parts.append(str(item[1]))
            except Exception:
                pass
        out = " ".join(p.strip() for p in parts if p).strip()
        return unicodedata.normalize("NFC", out)

    # -------- Visualization & IO --------
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
        vis = img_bgr.copy()
        for idx, r in enumerate(results, 1):
            pts = np.asarray(r["box"], dtype=np.float32)
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
            if "conf" in r and r["conf"] is not None:
                try:
                    label += f" ({float(r['conf']):.2f})"
                except Exception:
                    pass
            cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)
        return vis

    def annotate_to_file(
        self,
        image_path: str,
        results: List[Dict[str, Any]],
        out_img: str = "output/easy_annotated.jpg",
        **annotate_kwargs,
    ) -> str:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"OpenCV cannot read image: {image_path}")
        vis = self.annotate(img, results, **annotate_kwargs)
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        cv2.imwrite(out_img, vis)
        return out_img

    @staticmethod
    def save_json(payload: Dict[str, Any], out_json: str = "output/easy_results.json") -> str:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_json
