# -*- coding: utf-8 -*-
"""
Korean OCR Agent (TrOCR) - module only (no main)
- Line-level OCR (mỗi ảnh = 1 dòng)
- Dùng được cả train & inference qua API Python
- Metrics CER dùng rapidfuzz (fallback pure-Python nếu thiếu)
"""

import os, re, unicodedata, glob
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, TrainingArguments, Trainer

# ====== CER utilities (Cách 1: rapidfuzz; fallback pure-Python) ======
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

# -------- utils --------
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

# -------- dataset --------
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

# -------- public config --------
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

# -------- public agent --------
class KoreanOCRAgent:
    def __init__(self, model_dir_or_ckpt: str, device: Optional[str] = None):
        """
        model_dir_or_ckpt:
          - thư mục model đã fine-tune (vd: runs/trocr-ko/final)
          - hoặc checkpoint HF (vd: microsoft/trocr-base-printed)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_dir_or_ckpt)
        self.model     = VisionEncoderDecoderModel.from_pretrained(model_dir_or_ckpt)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_image(self, image_path: str, num_beams: int = 4, max_length: int = 64) -> str:
        img = Image.open(image_path).convert("RGB")
        enc = self.processor(images=[img], return_tensors="pt").pixel_values.to(self.device)
        gen = self.model.generate(enc, num_beams=num_beams, max_length=max_length)
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]

    @torch.inference_mode()
    def predict_glob(self, pattern: str, num_beams: int = 4, max_length: int = 64) -> List[Tuple[str,str]]:
        out = []
        for p in sorted(glob.glob(pattern)):
            try:
                out.append((p, self.predict_image(p, num_beams=num_beams, max_length=max_length)))
            except Exception as e:
                out.append((p, f"[ERROR] {e}"))
        return out

# -------- public train entry --------
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
        import numpy as np
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
