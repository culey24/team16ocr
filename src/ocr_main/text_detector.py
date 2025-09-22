# src/ocr_main/text_detector_agent.py
# -*- coding: utf-8 -*-
"""
Korean Text Detector Agent (EasyOCR - detect-only via readtext)

Logic:
- Dùng EasyOCR.readtext(detail=1) giống hệt ocr_agent.py (ổn định theo version).
- BỎ nhận dạng: không trả "text", chỉ lấy box + conf (đổi tên thành score).
- Trả về:
  {
    "image": str,
    "regions": [
      {"box": [[x,y],[x,y],[x,y],[x,y]], "score": float},
      ...
    ]
  }

API mẫu:
    agent = KoreanTextDetectorAgent(use_gpu=True)
    payload = agent.detect_image("/path/to/img.jpg")
    agent.annotate_to_file(payload["image"], payload["regions"], out_img="output/det_annotated.jpg")
    agent.save_json(payload, out_json="output/det_regions.json")
"""

import os
import json
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import easyocr


class KoreanTextDetectorAgent:
    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        detector: str = "craft",
        recognizer: str = "standard",  # giữ mặc định EasyOCR, nhưng ta không dùng text
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
        - use_gpu: None -> auto GPU nếu có; True/False để ép.
        - Các ngưỡng map trực tiếp sang readtext(...) như ocr_agent.py để tinh chỉnh detect/recall.
        """
        if use_gpu is None:
            use_gpu = False
            try:
                import torch  # noqa
                import torch.cuda  # noqa
                use_gpu = bool(torch.cuda.is_available())
            except Exception:
                use_gpu = False

        # Giữ tham số như ocr_agent.py để đảm bảo hành vi giống nhau
        self.reader = easyocr.Reader(
            ["ko"], gpu=use_gpu, detector=detector, recognizer=recognizer
        )

        # Lưu cấu hình dùng lại cho readtext
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

    # -------- Core (detect-only, via readtext) --------
    def detect_image(self, image_path: str) -> Dict[str, Any]:
        """
        Dùng readtext(detail=1) giống ocr_agent.py để lấy (box, text, conf),
        nhưng bỏ text -> chỉ trả regions (box + score).
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

        res = self.reader.readtext(image_path, detail=1, **self.read_cfg)
        # res: list of tuples (box, text, conf)

        regions: List[Dict[str, Any]] = []
        for item in res:
            try:
                box, _text, conf = item  # _text bỏ qua
            except Exception:
                # Một số version có thể trả khác dạng, cứ bỏ qua phần tử không đúng
                continue

            # ép kiểu an toàn
            try:
                box = [[float(p[0]), float(p[1])] for p in box]
                score = float(conf)
            except Exception:
                continue

            regions.append({"box": box, "score": score})

        return {"image": image_path, "regions": regions}

    # -------- Visualization & IO --------
    @staticmethod
    def annotate(
        img_bgr: np.ndarray,
        regions: List[Dict[str, Any]],
        draw_bbox: bool = True,
        draw_quad: bool = True,
        box_color=(0, 200, 200),
        quad_color=(0, 200, 0),
        text_color=(0, 0, 255),
    ) -> np.ndarray:
        """
        Vẽ khung detect. Nếu draw_quad=True vẽ đường bao tứ giác; draw_bbox=True vẽ bbox bao ngoài.
        """
        vis = img_bgr.copy()
        for idx, r in enumerate(regions, 1):
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
            score = r.get("score")
            if score is not None:
                try:
                    label += f" ({float(score):.2f})"
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
        regions: List[Dict[str, Any]],
        out_img: str = "output/det_annotated.jpg",
        **annotate_kwargs,
    ) -> str:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"OpenCV cannot read image: {image_path}")
        vis = self.annotate(img, regions, **annotate_kwargs)
        os.makedirs(os.path.dirname(out_img) or ".", exist_ok=True)
        cv2.imwrite(out_img, vis)
        return out_img

    def export_annotated(
        self,
        image_path: str,
        regions: List[Dict[str, Any]],
        out_img: str,
        resize_to_width: Optional[int] = None,
        **annotate_kwargs,
    ) -> str:
        """
        Vẽ box/quadrilateral lên ảnh và GHI RA FILE (không hiển thị).
        - out_img: đường dẫn file đầu ra (.jpg/.png)
        - resize_to_width: nếu set (vd 1280), ảnh sẽ được co theo chiều rộng này trước khi lưu.
        - annotate_kwargs: tham số của self.annotate (draw_bbox, draw_quad, ...)
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"OpenCV cannot read image: {image_path}")

        vis = self.annotate(img, regions, **annotate_kwargs)

        if resize_to_width is not None and resize_to_width > 0:
            h, w = vis.shape[:2]
            scale = resize_to_width / max(w, 1)
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        os.makedirs(os.path.dirname(out_img) or ".", exist_ok=True)
        cv2.imwrite(out_img, vis)
        return out_img

    @staticmethod
    def save_json(payload: Dict[str, Any], out_json: str = "output/det_regions.json") -> str:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_json
