# -*- coding: utf-8 -*-
"""
OCR Agent for Korean (TrOCR)
- Module only (no main). Import và dùng API.
- Hỗ trợ:
  1) Train TrOCR ở mức text-line với CER metric.
  2) Inference ảnh đã crop (text-line) hoặc ảnh lớn (A4/banner) qua detector(DB)->crop->recognizer.
- Detector: PaddleOCR (tuỳ chọn). Nếu không cài, bạn vẫn dùng được predict trên ảnh đã crop.

Author: you + gpt :)
"""
import numpy as np
import os, re, glob, unicodedata
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, TrainingArguments, Trainer

# =========================
# CER utilities (rapidfuzz; fallback pure-Python)
# =========================
try:
    from rapidfuzz.distance import Levenshtein as _lev
    def _cer(a: str, b: str) -> float:
        return _lev.distance(a, b) / max(1, len(b))
except Exception:
    def _levenshtein(a: str, b: str) -> int:
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                cur = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(
                    dp[j] + 1,      # deletion
                    dp[j - 1] + 1,  # insertion
                    prev + cost     # substitution
                )
                prev = cur
        return dp[n]
    def _cer(a: str, b: str) -> float:
        return _levenshtein(a, b) / max(1, len(b))

# =========================
# Text normalization & IO
# =========================
def _normalize_ko(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    return s

def _read_pairs(label_file: str, root: str = ".") -> List[Tuple[str, str]]:
    pairs = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"Label line missing TAB: '{line}'")
            p, t = line.split("\t", 1)
            pairs.append((os.path.join(root, p), _normalize_ko(t)))
    return pairs

# =========================
# Dataset & Collator
# =========================
class _LineOCRDataset(Dataset):
    def __init__(self, label_file: str, processor: TrOCRProcessor, root: str = "."):
        self.samples = _read_pairs(label_file, root=root)
        self.processor = processor

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return {"pixel_values": image, "labels": text}

class _DataCollator:
    def __init__(self, processor: TrOCRProcessor):
        self.processor = processor

    def __call__(self, batch):
        images = [x["pixel_values"] for x in batch]
        texts  = [x["labels"] for x in batch]
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

# =========================
# Public training config
# =========================
@dataclass
class TrainConfig:
    # data
    train_labels: str = "labels/train.txt"
    val_labels: str   = "labels/val.txt"
    data_root: str    = "."
    # model
    checkpoint: str   = "microsoft/trocr-base-printed"  # hoặc 'microsoft/trocr-large-printed'
    output_dir: str   = "runs/trocr-ko"
    # training
    batch_size: int   = 8
    eval_batch: int   = 8
    accum_steps: int  = 2
    lr: float         = 3e-5
    epochs: int       = 10
    fp16: bool        = True
    save_total: int   = 3
    eval_steps: int   = 500
    save_steps: int   = 500
    logging_steps: int = 100
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    # generation
    max_length: int   = 64
    num_beams: int    = 4

# =========================
# OCR Agent (Recognizer + optional Detector)
# =========================
class KoreanOcrAgent:
    """
    Recognizer (TrOCR) + optional big-image pipeline via PaddleOCR detector.
    """
    def __init__(self, model_dir_or_ckpt: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_dir_or_ckpt)
        self.model     = VisionEncoderDecoderModel.from_pretrained(model_dir_or_ckpt)
        self.model.to(self.device).eval()
        # Detector lazy-loading
        self._detector = None
        self._cv2 = None
        self._np = None

    # ---------- Core recognition (single cropped line) ----------
    @torch.inference_mode()
    def predict_image(self, image_path: str, num_beams: int = 4, max_length: int = 64) -> str:
        img = Image.open(image_path).convert("RGB")
        enc = self.processor(images=[img], return_tensors="pt").pixel_values.to(self.device)
        gen = self.model.generate(enc, num_beams=num_beams, max_length=max_length)
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]

    @torch.inference_mode()
    def predict_glob(self, pattern: str, num_beams: int = 4, max_length: int = 64) -> List[Tuple[str, str]]:
        out = []
        for p in sorted(glob.glob(pattern)):
            try:
                out.append((p, self.predict_image(p, num_beams=num_beams, max_length=max_length)))
            except Exception as e:
                out.append((p, f"[ERROR] {e}"))
        return out

    # ---------- Detector setup ----------
    def enable_detector(self, lang: str = "korean", use_angle_cls: bool = True, show_log: bool = False):
        """
        Bật detector của PaddleOCR để xử lý ảnh lớn (A4/banner).
        Cần: paddleocr, opencv-python, numpy
        """
        try:
            from paddleocr import PaddleOCR  # lazy imports
            import cv2
        except Exception as e:
            raise ImportError(
                "Detector requires 'paddleocr', 'opencv-python', 'numpy'. "
                "Install: pip install paddleocr opencv-python numpy"
            ) from e

        # cache libs
        self._cv2, self._np = cv2, np
        # init detector
        self._detector = PaddleOCR(lang=lang, det=True, rec=False, use_angle_cls=use_angle_cls, show_log=show_log)

    # ---------- Big image OCR (detector -> crop -> recognizer) ----------
    @torch.inference_mode()
    def ocr_big_image(self,
                      image_path: str,
                      beams: int = 4,
                      max_length: int = 64,
                      min_box: int = 8) -> Dict[str, Any]:
        """
        Xử lý ảnh lớn:
          - detect text boxes
          - crop (perspective)
          - recognize bằng TrOCR
        Trả về: {"image": str, "num_boxes": int, "results": [{"box": [[x,y],...], "text": str}, ...]}
        """
        if self._detector is None:
            raise RuntimeError("Detector not enabled. Call enable_detector(lang='korean') before ocr_big_image().")

        cv2, np = self._cv2, self._np

        # 1) detect
        det_res = self._detector.ocr(image_path, det=True, rec=False)
        boxes = []
        if det_res and det_res[0]:
            for item in det_res[0]:
                pts = np.array(item[0], dtype=np.float32)  # (4,2)
                xs, ys = pts[:, 0], pts[:, 1]
                if (xs.max() - xs.min()) >= min_box and (ys.max() - ys.min()) >= min_box:
                    boxes.append(pts)

        # sort reading order (top-to-bottom, left-to-right)
        def _key_fn(b):
            c = b.mean(axis=0)
            return (float(c[1]), float(c[0]))
        boxes = sorted(boxes, key=_key_fn)

        # 2) crop & recognize
        img_bgr = cv2.imread(image_path)
        results = []
        for quad in boxes:
            crop_bgr = _quad_to_perspective(cv2, quad, img_bgr)
            pil = _bgr_to_pil(cv2, crop_bgr)
            enc = self.processor(images=[pil], return_tensors="pt").pixel_values.to(self.device)
            gen = self.model.generate(enc, num_beams=beams, max_length=max_length)
            text = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
            results.append({"box": quad.astype(float).tolist(), "text": text})

        return {"image": image_path, "num_boxes": len(boxes), "results": results}

# =========================
# Perspective helpers
# =========================
def _order_quad(cv2, pts):
    rect = pts.copy().astype("float32")
    s = pts.sum(axis=1)
    diff = cv2.subtract(pts[:, 0], pts[:, 1])  # x - y
    rect_out = cv2.convexHull(pts).reshape(-1, 2).astype("float32")  # not strictly needed
    # more robust ordering:
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return cv2.convertPointsToHomogeneous(
        cv2.UMat(np.array([tl, tr, br, bl], dtype=np.float32))
    ).get().reshape(-1, 3)[:, :2].astype(np.float32)

def _quad_to_perspective(cv2, quad, img_bgr, pad: int = 2):
    rect = _order_quad(cv2, quad)

    w1 = _l2(rect[1], rect[0])
    w2 = _l2(rect[2], rect[3])
    h1 = _l2(rect[3], rect[0])
    h2 = _l2(rect[2], rect[1])
    width = int(max(w1, w2))
    height = int(max(h1, h2))
    width = max(width, 1)
    height = max(height, 1)

    dst = _np_array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32", cv2=cv2)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (width, height), flags=cv2.INTER_CUBIC)

    if pad > 0:
        warped = cv2.copyMakeBorder(warped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return warped

def _bgr_to_pil(cv2, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _np_array(data, dtype, cv2):
    import numpy as _np
    return _np.array(data, dtype=getattr(_np, dtype) if isinstance(dtype, str) else dtype)

def _l2(a, b):
    import numpy as _np
    return float(_np.linalg.norm(a - b))

# =========================
# Public train entry
# =========================
def train(cfg: TrainConfig) -> str:
    """
    Fine-tune TrOCR theo cấu hình cfg.
    Return: đường dẫn thư mục final model (vd: runs/trocr-ko/final)
    """
    processor = TrOCRProcessor.from_pretrained(cfg.checkpoint)
    model     = VisionEncoderDecoderModel.from_pretrained(cfg.checkpoint)

    # cấu hình decode
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.max_length             = cfg.max_length
    model.config.num_beams              = cfg.num_beams
    model.config.early_stopping         = True

    train_ds  = _LineOCRDataset(cfg.train_labels, processor, root=cfg.data_root)
    val_ds    = _LineOCRDataset(cfg.val_labels,   processor, root=cfg.data_root)
    collator  = _DataCollator(processor)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch,
        gradient_accumulation_steps=cfg.accum_steps,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        fp16=cfg.fp16,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total,
        logging_steps=cfg.logging_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds_t = torch.tensor(preds)
        pred_str = processor.batch_decode(preds_t, skip_special_tokens=True)
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        cers = [_cer(p, l) for p, l in zip(pred_str, label_str)]
        return {"cer": float(np.mean(cers))}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=processor.feature_extractor,  # để Trainer không complain
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    return final_dir
