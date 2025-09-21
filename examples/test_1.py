# -*- coding: utf-8 -*-
"""
Test runner cho korean_ocr_agent (CLI nhẹ)
- Train:  python test_korean_ocr_agent.py train --train labels/train.txt --val labels/val.txt
- Infer:  python test_korean_ocr_agent.py infer --model runs/trocr-ko/final --image images/test/zzz1.jpg
- Batch:  python test_korean_ocr_agent.py batch --model runs/trocr-ko/final --glob "images/test/*.jpg"
"""

import argparse, json
from src.ocr_main.ocr_agent import TrainConfig, train, KoreanOCRAgent

def main():
    p = argparse.ArgumentParser("KOCR tester")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="Fine-tune TrOCR")
    pt.add_argument("--train", required=True, help="labels/train.txt")
    pt.add_argument("--val",   required=True, help="labels/val.txt")
    pt.add_argument("--root",  default=".")
    pt.add_argument("--ckpt",  default="microsoft/trocr-base-printed")
    pt.add_argument("--out",   default="runs/trocr-ko")
    pt.add_argument("--batch", type=int, default=8)
    pt.add_argument("--accum", type=int, default=2)
    pt.add_argument("--epochs", type=int, default=10)
    pt.add_argument("--lr", type=float, default=3e-5)

    # infer single
    pi = sub.add_parser("infer", help="Infer 1 ảnh")
    pi.add_argument("--model", required=True)
    pi.add_argument("--image", required=True)
    pi.add_argument("--beams", type=int, default=4)
    pi.add_argument("--maxlen", type=int, default=64)

    # batch infer
    pb = sub.add_parser("batch", help="Infer nhiều ảnh (glob)")
    pb.add_argument("--model", required=True)
    pb.add_argument("--glob",  required=True)
    pb.add_argument("--out",   default="predictions.jsonl")
    pb.add_argument("--beams", type=int, default=4)
    pb.add_argument("--maxlen", type=int, default=64)

    args = p.parse_args()

    if args.cmd == "train":
        cfg = TrainConfig(
            train_labels=args.train,
            val_labels=args.val,
            data_root=args.root,
            checkpoint=args.ckpt,
            output_dir=args.out,
            batch_size=args.batch,
            accum_steps=args.accum,
            epochs=args.epochs,
            lr=args.lr,
        )
        final_dir = train(cfg)
        print("Saved model to:", final_dir)

    elif args.cmd == "infer":
        agent = KoreanOCRAgent(args.model)
        text = agent.predict_image(args.image, num_beams=args.beams, max_length=args.maxlen)
        print(text)

    elif args.cmd == "batch":
        agent = KoreanOCRAgent(args.model)
        results = agent.predict_glob(args.glob, num_beams=args.beams, max_length=args.maxlen)
        with open(args.out, "w", encoding="utf-8") as f:
            for p, t in results:
                f.write(json.dumps({"path": p, "text": t}, ensure_ascii=False) + "\n")
        print("Saved:", args.out)

if __name__ == "__main__":
    main()
