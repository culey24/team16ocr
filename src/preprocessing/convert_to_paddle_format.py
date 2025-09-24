# #!Chuyển từ file mapping thành file txt có định dạng của paddle
# # Usage:
# #   python convert_to_paddle.py \
# #       --mapping (Enter ur *.csv) \
# #       --out (Enter where u want to save - *.txt) 

# from pathlib import Path
# from PIL import Image
# import csv, re, unicodedata, argparse

# def read_text(path: Path) -> str:
#     txt = None
#     for enc in ("utf-8","cp949","euc-kr","latin-1"):
#         try:
#             txt = path.read_text(encoding=enc); break
#         except: pass
#     if txt is None: return ""
#     txt = unicodedata.normalize("NFC", txt)
#     txt = re.sub(r"\s+"," ", txt).strip()
#     return txt

# def main(mapping="mapping_clean.csv", out_file="merged_dataset/train.txt"):
#     mapping = Path(mapping)
#     outp = Path(out_file); outp.parent.mkdir(parents=True, exist_ok=True)

#     kept, skipped = 0, 0
#     with mapping.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as wf:
#         rdr = csv.DictReader(f)
#         assert {"stem","image_path","text_path"} <= set(rdr.fieldnames)
#         for r in rdr:
#             ip, tp = Path(r["image_path"]), Path(r["text_path"])
#             if not ip.exists() or not tp.exists(): 
#                 skipped += 1; continue
#             try:
#                 w, h = Image.open(ip).size
#             except Exception:
#                 skipped += 1; continue
#             text = read_text(tp)
#             if not text: 
#                 skipped += 1; continue

#             box = f"0,0,{w},0,{w},{h},0,{h}"
#             wf.write(f"{ip}\t{box},{text}\n")
#             kept += 1

#     print(f"Done. Wrote: {outp}")
#     print(f"OK: {kept} | Skipped: {skipped}")

# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--mapping", default="mapping_preprocessed_clean.csv")
#     ap.add_argument("--out", default="merged_dataset/train_preprocessed.txt")
#     args = ap.parse_args()
#     main(args.mapping, args.out)