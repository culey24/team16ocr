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

    # ---------- Helpers (put inside KoreanTextDetectorAgent) ----------
    # ====== Sliding-window detect via EasyOCR.readtext, with overlap & dedup ======
    @staticmethod
    def _poly_to_bbox(poly):
        import numpy as np
        pts = np.asarray(poly, dtype=np.float32)
        return float(pts[:,0].min()), float(pts[:,1].min()), float(pts[:,0].max()), float(pts[:,1].max())

    @staticmethod
    def _iou_bbox(a, b):
        # a,b: (x1,y1,x2,y2)
        x1,y1,x2,y2 = a
        X1,Y1,X2,Y2 = b
        iw = max(0.0, min(x2, X2) - max(x1, X1))
        ih = max(0.0, min(y2, Y2) - max(y1, Y1))
        inter = iw * ih
        if inter <= 0: return 0.0
        ua = max(0.0, (x2-x1)) * max(0.0, (y2-y1))
        ub = max(0.0, (X2-X1)) * max(0.0, (Y2-Y1))
        union = ua + ub - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _greedy_nms(cls, boxes, scores, iou_thr=0.3):
        idxs = list(range(len(boxes)))
        idxs.sort(key=lambda i: scores[i], reverse=True)
        keep = []
        while idxs:
            i = idxs.pop(0)
            keep.append(i)
            idxs = [j for j in idxs if cls._iou_bbox(boxes[i], boxes[j]) < iou_thr]
        return keep

    def detect_image_sliding(
        self,
        image_path: str,
        tile_height: int = 1600,
        overlap_ratio: float = 0.15,
        min_width_upscale: int = 1000,
        max_upscale: float = 2.0,
        nms_iou: float = 0.3,
    ):
        """
        Chia ảnh theo CHIỀU DỌC thành các tile cao `tile_height` (có chồng lấn `overlap_ratio`),
        chạy EasyOCR.readtext trên từng tile, rồi GHÉP lại và NMS khử trùng lặp.

        - Nếu ảnh quá hẹp (width < min_width_upscale) sẽ upscale trước (≤ max_upscale).
        - Tọa độ box được MAP NGƯỢC về ảnh gốc.
        """
        import os, cv2, numpy as np

        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"OpenCV cannot read image: {image_path}")

        H, W = img.shape[:2]

        # 1) Upscale nhẹ theo chiều ngang nếu quá hẹp
        scale = 1.0
        if W < min_width_upscale:
            scale = min(max_upscale, float(min_width_upscale) / max(W, 1))
        work = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_CUBIC) if scale > 1.0 else img
        h2, w2 = work.shape[:2]

        # 2) Lập lịch các cửa sổ dọc có overlap
        step = max(1, int(tile_height * (1.0 - overlap_ratio)))
        starts = list(range(0, max(1, h2 - tile_height + 1), step))
        if not starts or starts[-1] + tile_height < h2:
            starts.append(max(0, h2 - tile_height))

        all_polys, all_scores = [], []

        # 3) Quét từng tile
        for y0 in starts:
            y1 = min(h2, y0 + tile_height)
            tile = work[y0:y1, 0:w2]

            try:
                res = self.reader.readtext(tile, detail=1, **self.read_cfg)  # (box, text, conf)
            except Exception:
                res = []

            for item in res:
                try:
                    poly, _text, conf = item
                    # Map tile->work->original
                    mapped = [[float(px)/scale, (float(py)+y0)/scale] for (px,py) in poly]
                    all_polys.append(mapped)
                    all_scores.append(float(conf))
                except Exception:
                    continue

        if not all_polys:
            return {"image": image_path, "regions": []}

        # 4) Dedup bằng NMS (trên bbox của polygon)
        bboxes = [self._poly_to_bbox(p) for p in all_polys]
        keep = self._greedy_nms(bboxes, all_scores or [0.0]*len(bboxes), iou_thr=nms_iou)

        regions = [{"box": all_polys[i], "score": all_scores[i]} for i in keep]
        return {"image": image_path, "regions": regions}


    def detect_image_sliding_ensemble(
        self,
        image_path: str,
        # danh sách cấu hình sẽ thử lần lượt
        tile_heights=(1600, 2000),
        overlaps=(0.15, 0.22),
        min_widths=(1000, 1300),
        rotations_deg=(-3, 0, 3),
        nms_iou_merge=0.5,
        # bộ tham số readtext “aggressive” để tăng recall
        rt_text_threshold=0.58,
        rt_low_text=0.30,
        rt_link_threshold=0.30,
        upscale_limit=2.0,
    ):
        """
        Ensemble: chạy sliding nhiều cấu hình + xoay nhẹ; gộp box bằng NMS để giảm bỏ sót.
        """
        import os, cv2, numpy as np

        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
        img0 = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img0 is None:
            raise FileNotFoundError(f"OpenCV cannot read image: {image_path}")

        H0, W0 = img0.shape[:2]

        # backup & tạm thời nới tham số readtext
        old_cfg = dict(self.read_cfg)
        self.read_cfg.update(dict(
            text_threshold=rt_text_threshold,
            low_text=rt_low_text,
            link_threshold=rt_link_threshold,
        ))

        all_polys, all_scores = [], []

        try:
            for angle in rotations_deg:
                # xoay nhẹ quanh tâm (để bắt chữ hơi nghiêng)
                if angle == 0:
                    img_rot, M = img0, None
                    inv_scale = 1.0
                else:
                    center = (W0/2, H0/2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    cos = abs(M[0,0]); sin = abs(M[0,1])
                    # tính canvas quay
                    newW = int(H0*sin + W0*cos)
                    newH = int(H0*cos + W0*sin)
                    M[0,2] += (newW/2) - center[0]
                    M[1,2] += (newH/2) - center[1]
                    img_rot = cv2.warpAffine(img0, M, (newW, newH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

                Hr, Wr = img_rot.shape[:2]

                for mw in min_widths:
                    scale = 1.0 if Wr >= mw else min(upscale_limit, float(mw)/max(Wr,1))
                    work = cv2.resize(img_rot, (int(Wr*scale), int(Hr*scale)), interpolation=cv2.INTER_CUBIC) if scale>1.0 else img_rot
                    h2, w2 = work.shape[:2]

                    for th in tile_heights:
                        for ov in overlaps:
                            step = max(1, int(th*(1.0-ov)))
                            starts = list(range(0, max(1, h2 - th + 1), step))
                            if not starts or starts[-1] + th < h2:
                                starts.append(max(0, h2 - th))
                            for y0 in starts:
                                y1 = min(h2, y0 + th)
                                tile = work[y0:y1, 0:w2]

                                try:
                                    res = self.reader.readtext(tile, detail=1, **self.read_cfg)
                                except Exception:
                                    res = []

                                for item in res:
                                    try:
                                        poly, _text, conf = item
                                        # map tile->rotated->original
                                        pts = np.asarray(poly, dtype=np.float32)
                                        pts[:,0] = pts[:,0]                # x trên work
                                        pts[:,1] = pts[:,1] + float(y0)    # y cộng offset tile
                                        pts /= float(scale)

                                        if angle != 0:
                                            # map từ rotated về original qua nghịch đảo affine
                                            Minv = cv2.invertAffineTransform(M)
                                            ones = np.ones((pts.shape[0],1), dtype=np.float32)
                                            pts = np.hstack([pts, ones])
                                            pts = (pts @ Minv.T).astype(np.float32)

                                        all_polys.append([[float(px), float(py)] for px,py in pts])
                                        all_scores.append(float(conf))
                                    except Exception:
                                        continue
        finally:
            # khôi phục tham số readtext
            self.read_cfg = old_cfg

        if not all_polys:
            return {"image": image_path, "regions": []}

        # NMS gộp (trên bbox của polygon)
        def _poly_to_bbox(poly):
            pts = np.asarray(poly, dtype=np.float32)
            return float(pts[:,0].min()), float(pts[:,1].min()), float(pts[:,0].max()), float(pts[:,1].max())

        def _iou(a,b):
            x1,y1,x2,y2=a; X1,Y1,X2,Y2=b
            iw=max(0,min(x2,X2)-max(x1,X1)); ih=max(0,min(y2,Y2)-max(y1,Y1))
            inter=iw*ih
            if inter<=0: return 0.0
            ua=max(0,x2-x1)*max(0,y2-y1); ub=max(0,X2-X1)*max(0,Y2-Y1)
            return inter/max(ua+ub-inter,1e-6)

        boxes=[_poly_to_bbox(p) for p in all_polys]
        idxs=list(range(len(boxes)))
        idxs.sort(key=lambda i: all_scores[i], reverse=True)
        kept=[]
        while idxs:
            i=idxs.pop(0)
            kept.append(i)
            idxs=[j for j in idxs if _iou(boxes[i], boxes[j]) < nms_iou_merge]

        regions=[{"box": all_polys[i], "score": all_scores[i]} for i in kept]
        return {"image": image_path, "regions": regions}
