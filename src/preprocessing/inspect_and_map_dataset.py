# #Trả về file mapping ảnh và text
# #Usage: python inspect_and_map_dataset.py

# #Có thể chỉnh nơi lưu thư mục ở dưới nhe
# #dòng 72
# # def main(
# #     img_dir="DATA/PREPROCESSED_PNG", (này là anh preprocess xong rồi anh mới mapping á)
# #     txt_dir="DATA/ORIGINAL_TEXT",
# #     out_csv="mapping.csv",
# #     out_csv_clean="mapping_preprocessed_clean.csv", (clean ở đây là xóa lập với rỗng á)
# # ):

# import os, re, csv, unicodedata
# from pathlib import Path

# IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
# TXT_EXTS = {".txt", ".text"}

# def stem_from_text_name(p: Path):
#     """
#     Trả về 'stem chuẩn' để khớp với ảnh.
#     Ví dụ: 'TAF20161_00.txt.text' -> 'TAF20161_00'
#     """
#     name = p.name
#     # bỏ đuôi .txt hoặc .text nếu có dạng kép
#     name = re.sub(r"\.txt(\.text)?$", "", name, flags=re.IGNORECASE)
#     name = re.sub(r"\.text$", "", name, flags=re.IGNORECASE)
#     return name

# def find_files(root: Path, exts):
#     files = []
#     for p in root.rglob("*"):
#         if p.is_file() and (p.suffix.lower() in exts or p.name.endswith(".txt.text")):
#             files.append(p)
#     return files

# # ---- NEW: utils cho lọc dữ liệu ----
# def read_text(path: Path, max_chars=None) -> str:
#     txt = None
#     for enc in ("utf-8", "cp949", "euc-kr", "latin-1"):
#         try:
#             txt = path.read_text(encoding=enc)
#             break
#         except Exception:
#             continue
#     if txt is None:
#         return ""
#     txt = unicodedata.normalize("NFC", txt)
#     txt = re.sub(r"\s+", " ", txt).strip()
#     if max_chars and len(txt) > max_chars:
#         txt = txt[:max_chars] + " …"
#     return txt

# def pick_best_text(paths):
#     """
#     Chọn 1 file text 'đẹp' nhất trong các bản trùng (cùng stem):
#     - Dài hơn (file size lớn hơn) ưu tiên trước
#     - Nếu bằng nhau: ưu tiên .txt.text > .txt > .text
#     """
#     def suf_rank(p: Path):
#         if p.name.endswith(".txt.text"): return 0
#         if p.suffix.lower()==".txt":     return 1
#         if p.suffix.lower()==".text":    return 2
#         return 3
#     def key(p: Path):
#         try: size = p.stat().st_size
#         except: size = -1
#         return (size, - (3 - suf_rank(p)))
#     return sorted(paths, key=key, reverse=True)[0]

# # ---- main ----
# def main(
#     img_dir="DATA/PREPROCESSED_PNG",
#     txt_dir="DATA/ORIGINAL_TEXT",
#     out_csv="mapping.csv",
#     out_csv_clean="mapping_preprocessed_clean.csv",
# ):
#     img_dir = Path(img_dir)
#     txt_dir = Path(txt_dir)

#     imgs = find_files(img_dir, IMG_EXTS)
#     txts = find_files(txt_dir, TXT_EXTS)  # sẽ kèm cả *.txt.text nhờ find_files

#     # Lập chỉ mục theo stem
#     img_map = {}
#     for p in imgs:
#         img_map.setdefault(p.stem, []).append(p)

#     txt_map = {}
#     for p in txts:
#         stem = stem_from_text_name(p)
#         txt_map.setdefault(stem, []).append(p)

#     # Khớp cặp (bản thô)
#     all_stems = sorted(set(img_map.keys()) | set(txt_map.keys()))
#     pairs, missing_img, missing_txt, dup_img, dup_txt = [], [], [], [], []

#     for s in all_stems:
#         im = img_map.get(s, [])
#         tx = txt_map.get(s, [])
#         if not im and tx:
#             missing_img.append((s, tx))
#         elif im and not tx:
#             missing_txt.append((s, im))
#         elif im and tx:
#             if len(im) > 1: dup_img.append((s, im))
#             if len(tx) > 1: dup_txt.append((s, tx))
#             # lấy 1 file ảnh đại diện (ưu tiên png/jpg/jpeg)
#             im_sorted = sorted(im, key=lambda p: (p.suffix.lower() not in [".png", ".jpg", ".jpeg"], str(p)))
#             # tạm thời cứ lấy text đầu tiên (bản CLEAN ở dưới sẽ chọn tốt nhất)
#             tx_sorted = sorted(tx, key=lambda p: (len(p.suffix), str(p)))
#             pairs.append((s, im_sorted[0], tx_sorted[0]))

#     # Ghi mapping.csv (thô)
#     with open(out_csv, "w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(["stem", "image_path", "text_path"])
#         for s, ip, tp in pairs:
#             w.writerow([s, str(ip), str(tp)])

#     # ------ NEW: sinh bản CLEAN (lọc thiếu / rỗng / trùng text) ------
#     kept_clean = []
#     dropped = {"no_img": 0, "no_txt": 0, "empty": 0, "dup_txt": 0}

#     # ảnh còn dùng được (chọn 1 file ảnh nếu trùng)
#     img_one = {}
#     for s, plist in img_map.items():
#         ip = sorted(plist, key=lambda p: (p.suffix.lower() not in [".png",".jpg",".jpeg"], str(p)))[0]
#         img_one[s] = ip

#     # chọn best text cho mỗi stem có text
#     txt_best = {}
#     for s, plist in txt_map.items():
#         if len(plist) > 1:
#             dropped["dup_txt"] += 1
#         txt_best[s] = pick_best_text(plist)

#     # duyệt theo các stem có ảnh
#     for s, ip in img_one.items():
#         tp = txt_best.get(s)
#         if tp is None:
#             dropped["no_txt"] += 1
#             continue
#         if not ip.exists():
#             dropped["no_img"] += 1
#             continue
#         content = read_text(tp)
#         if not content:
#             dropped["empty"] += 1
#             continue
#         kept_clean.append((s, ip, tp))

#     # thêm thống kê: text-only (không có ảnh)
#     txt_only = [s for s in txt_best.keys() if s not in img_one]
#     dropped["no_img"] += len(txt_only)

#     # ghi mapping_clean.csv
#     with open(out_csv_clean, "w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(["stem", "image_path", "text_path"])
#         for s, ip, tp in kept_clean:
#             w.writerow([s, str(ip), str(tp)])

#     # In thống kê nhanh
#     print("=== DATASET SUMMARY ===")
#     print(f"Images found     : {len(imgs)}")
#     print(f"Text files found : {len(txts)}")
#     print(f"Matched pairs    : {len(pairs)}")
#     print(f"Missing image    : {len(missing_img)} stems")
#     print(f"Missing text     : {len(missing_txt)} stems")
#     print(f"Duplicate images : {len(dup_img)} stems")
#     print(f"Duplicate texts  : {len(dup_txt)} stems")
#     print()
#     if missing_img:
#         print(">> Stems lacking IMAGE (up to 5):", [s for s,_ in missing_img[:5]])
#     if missing_txt:
#         print(">> Stems lacking TEXT  (up to 5):", [s for s,_ in missing_txt[:5]])
#     if dup_img:
#         print(">> Stems with DUPLICATE IMAGES (up to 3):", [s for s,_ in dup_img[:3]])
#     if dup_txt:
#         print(">> Stems with DUPLICATE TEXTS  (up to 3):", [s for s,_ in dup_txt[:3]])

#     print("\n=== CLEAN SUMMARY ===")
#     print(f"Pairs kept (clean): {len(kept_clean)}")
#     print(f"Dropped no_txt    : {dropped['no_txt']}")
#     print(f"Dropped no_img    : {dropped['no_img']}")
#     print(f"Dropped empty     : {dropped['empty']}")
#     print(f"Duplicate texts   : {dropped['dup_txt']} (resolved by best-pick)")
#     print(f"\nWrote: {out_csv}")
#     print(f"Wrote: {out_csv_clean}")
#     print("\nNext: dùng mapping_clean.csv cho bước convert_to_paddle_format (fake bbox toàn ảnh).")

# if __name__ == "__main__":
#     main()

# Trả về file mapping ảnh và text
# Usage: python inspect_and_map_dataset.py
# (Dùng để map ảnh đã preprocess với text ground-truth, sau đó dùng cho EasyOCR verify)

import os, re, csv, unicodedata
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
TXT_EXTS = {".txt", ".text"}

def stem_from_text_name(p: Path):
    name = p.name
    name = re.sub(r"\.txt(\.text)?$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.text$", "", name, flags=re.IGNORECASE)
    return name

def find_files(root: Path, exts):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and (p.suffix.lower() in exts or p.name.endswith(".txt.text")):
            files.append(p)
    return files

def read_text(path: Path, max_chars=None) -> str:
    txt = None
    for enc in ("utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            txt = path.read_text(encoding=enc)
            break
        except Exception:
            continue
    if txt is None:
        return ""
    txt = unicodedata.normalize("NFC", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    if max_chars and len(txt) > max_chars:
        txt = txt[:max_chars] + " …"
    return txt

def pick_best_text(paths):
    def suf_rank(p: Path):
        if p.name.endswith(".txt.text"): return 0
        if p.suffix.lower()==".txt":     return 1
        if p.suffix.lower()==".text":    return 2
        return 3
    def key(p: Path):
        try: size = p.stat().st_size
        except: size = -1
        return (size, - (3 - suf_rank(p)))
    return sorted(paths, key=key, reverse=True)[0]

def main(
    img_dir="DATA/PREPROCESSED_PNG",  # Ảnh sau preprocess
    txt_dir="DATA/ORIGINAL_TEXT",
    out_csv="mapping.csv",
    out_csv_clean="mapping_preprocessed_clean.csv",
):
    img_dir = Path(img_dir)
    txt_dir = Path(txt_dir)

    imgs = find_files(img_dir, IMG_EXTS)
    txts = find_files(txt_dir, TXT_EXTS)

    img_map = {}
    for p in imgs:
        img_map.setdefault(p.stem, []).append(p)

    txt_map = {}
    for p in txts:
        stem = stem_from_text_name(p)
        txt_map.setdefault(stem, []).append(p)

    all_stems = sorted(set(img_map.keys()) | set(txt_map.keys()))
    pairs, missing_img, missing_txt, dup_img, dup_txt = [], [], [], [], []

    for s in all_stems:
        im = img_map.get(s, [])
        tx = txt_map.get(s, [])
        if not im and tx:
            missing_img.append((s, tx))
        elif im and not tx:
            missing_txt.append((s, im))
        elif im and tx:
            if len(im) > 1: dup_img.append((s, im))
            if len(tx) > 1: dup_txt.append((s, tx))
            im_sorted = sorted(im, key=lambda p: (p.suffix.lower() not in [".png", ".jpg", ".jpeg"], str(p)))
            tx_sorted = sorted(tx, key=lambda p: (len(p.suffix), str(p)))
            pairs.append((s, im_sorted[0], tx_sorted[0]))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "image_path", "text_path"])
        for s, ip, tp in pairs:
            w.writerow([s, str(ip), str(tp)])

    kept_clean = []
    dropped = {"no_img": 0, "no_txt": 0, "empty": 0, "dup_txt": 0}

    img_one = {}
    for s, plist in img_map.items():
        ip = sorted(plist, key=lambda p: (p.suffix.lower() not in [".png",".jpg",".jpeg"], str(p)))[0]
        img_one[s] = ip

    txt_best = {}
    for s, plist in txt_map.items():
        if len(plist) > 1:
            dropped["dup_txt"] += 1
        txt_best[s] = pick_best_text(plist)

    for s, ip in img_one.items():
        tp = txt_best.get(s)
        if tp is None:
            dropped["no_txt"] += 1
            continue
        if not ip.exists():
            dropped["no_img"] += 1
            continue
        content = read_text(tp)
        if not content:
            dropped["empty"] += 1
            continue
        kept_clean.append((s, ip, tp))

    txt_only = [s for s in txt_best.keys() if s not in img_one]
    dropped["no_img"] += len(txt_only)

    with open(out_csv_clean, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "image_path", "text_path"])
        for s, ip, tp in kept_clean:
            w.writerow([s, str(ip), str(tp)])

    print("=== DATASET SUMMARY ===")
    print(f"Images found     : {len(imgs)}")
    print(f"Text files found : {len(txts)}")
    print(f"Matched pairs    : {len(pairs)}")
    print(f"Missing image    : {len(missing_img)} stems")
    print(f"Missing text     : {len(missing_txt)} stems")
    print(f"Duplicate images : {len(dup_img)} stems")
    print(f"Duplicate texts  : {len(dup_txt)} stems")
    print()
    if missing_img:
        print(">> Stems lacking IMAGE (up to 5):", [s for s,_ in missing_img[:5]])
    if missing_txt:
        print(">> Stems lacking TEXT  (up to 5):", [s for s,_ in missing_txt[:5]])
    if dup_img:
        print(">> Stems with DUPLICATE IMAGES (up to 3):", [s for s,_ in dup_img[:3]])
    if dup_txt:
        print(">> Stems with DUPLICATE TEXTS  (up to 3):", [s for s,_ in dup_txt[:3]])

    print("\n=== CLEAN SUMMARY ===")
    print(f"Pairs kept (clean): {len(kept_clean)}")
    print(f"Dropped no_txt    : {dropped['no_txt']}")
    print(f"Dropped no_img    : {dropped['no_img']}")
    print(f"Dropped empty     : {dropped['empty']}")
    print(f"Duplicate texts   : {dropped['dup_txt']} (resolved by best-pick)")
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_csv_clean}")
    print("\nNext: Dùng mapping_clean.csv để chạy EasyOCR inference.")

if __name__ == "__main__":
    main()