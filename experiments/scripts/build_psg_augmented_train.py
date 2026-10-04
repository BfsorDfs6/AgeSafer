#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERSION: PSG-Dim5-v4-augment-train
FILE NAME ON SERVER: scripts/build_psg_augmented_train.py

Purpose
-------
Build augmented GMF training data from PSG predictions. This uses psg_pred=1
pseudo positives selected by p1/margin and rho per user.

Common command
--------------
cd .
STORY=outputs/psg_dim5_v4
python -u scripts/build_psg_augmented_train.py \
  --base_data_dir Data \
  --base_dataset ml-1m_safe \
  --pred_jsonl ${STORY}/predictions/psg_pred_dim5_top100.jsonl \
  --out_dir ${STORY}/gmf_data/ml-1m_safe_dim5_top100_rho020 \
  --out_dataset ml-1m_safe_dim5_top100_rho020 \
  --rho 0.20 \
  --min_p1 0.0 \
  --min_margin -999
"""
"""
build_psg_augmented_train_dim5.py

Build augmented NCF/GMF train.rating from v2 PSG predictions.
Select PSG-positive pseudo items per user and append them to the original train.rating.
Also copies valid/test rating/negative files into the output directory with the new dataset prefix.
"""
import argparse, json, os, re, shutil
from collections import defaultdict

def safe_int(x, default=-1):
    try: return int(float(str(x).strip()))
    except Exception: return default

def safe_float(x, default=0.0):
    try:
        if x is None or str(x).strip() == "": return default
        return float(str(x).strip())
    except Exception: return default

def split_line(line):
    s = line.strip()
    if not s: return []
    if "::" in s: return s.split("::")
    if "\t" in s: return s.split("\t")
    if "," in s and not s.startswith("("): return [x.strip() for x in s.split(",")]
    return re.split(r"\s+", s)

def read_train(path):
    rows, seen = [], set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            arr = split_line(line)
            if len(arr) < 2: continue
            u, i = safe_int(arr[0]), safe_int(arr[1])
            if u < 0 or i < 0: continue
            rows.append((u, i, line.strip()))
            seen.add((u, i))
    print(f"[load] train rows={len(rows)} pairs={len(seen)} path={path}")
    return rows, seen

def read_preds(path, min_p1, min_margin, pred_value=1):
    by_user = defaultdict(list)
    n = keep = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            n += 1
            obj = json.loads(line)
            if int(obj.get("psg_pred", 0)) != int(pred_value): continue
            p1 = safe_float(obj.get("p1", 1.0), 1.0)
            margin = safe_float(obj.get("margin_1_minus_0", 0.0), 0.0)
            if p1 < min_p1 or margin < min_margin: continue
            meta = obj.get("metadata", {})
            u = safe_int(meta.get("inner_user_id", meta.get("user_id", -1)), -1)
            i = safe_int(meta.get("inner_item_id", meta.get("item_id", -1)), -1)
            if u < 0 or i < 0: continue
            by_user[u].append((p1, margin, i, obj))
            keep += 1
    print(f"[load] pred rows={n} psg_positive_candidates={keep} users={len(by_user)} path={path}")
    return by_user

def copy_split(src_dir, src_dataset, out_dir, out_dataset, split, suffix):
    src = os.path.join(src_dir, f"{src_dataset}.{split}.{suffix}")
    dst = os.path.join(out_dir, f"{out_dataset}.{split}.{suffix}")
    if os.path.exists(src):
        shutil.copyfile(src, dst); print(f"[copy] {src} -> {dst}")
    else:
        print(f"[warn] missing split file: {src}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_data_dir", default="Data")
    ap.add_argument("--base_dataset", default="ml-1m_safe")
    ap.add_argument("--train_rating", default="")
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_dataset", required=True)
    ap.add_argument("--rho", type=float, default=0.20, help="per-user pseudo count = ceil(rho * real_train_count)")
    ap.add_argument("--max_per_user", type=int, default=0)
    ap.add_argument("--min_p1", type=float, default=0.0)
    ap.add_argument("--min_margin", type=float, default=-999.0)
    ap.add_argument("--rating_value", default="1")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train_path = args.train_rating or os.path.join(args.base_data_dir, f"{args.base_dataset}.train.rating")
    train_rows, seen = read_train(train_path)
    user_real = defaultdict(int)
    for u, i, _ in train_rows: user_real[u] += 1
    preds = read_preds(args.pred_jsonl, args.min_p1, args.min_margin)
    added, added_by_user = [], defaultdict(int)
    for u, arr in preds.items():
        arr.sort(key=lambda x: (x[0], x[1]), reverse=True)
        quota = max(1, int(round(args.rho * max(1, user_real.get(u, 1)))))
        if args.max_per_user > 0: quota = min(quota, args.max_per_user)
        for p1, margin, i, obj in arr:
            if (u, i) in seen: continue
            added.append((u, i)); seen.add((u, i)); added_by_user[u] += 1
            if added_by_user[u] >= quota: break
    out_train = os.path.join(args.out_dir, f"{args.out_dataset}.train.rating")
    with open(out_train, "w", encoding="utf-8") as f:
        for _, _, line in train_rows: f.write(line + "\n")
        for u, i in added: f.write(f"{u}\t{i}\t{args.rating_value}\n")
    print(f"[write] train original={len(train_rows)} added={len(added)} total={len(train_rows)+len(added)} path={out_train}")
    for split in ["valid", "test"]:
        copy_split(args.base_data_dir, args.base_dataset, args.out_dir, args.out_dataset, split, "rating")
        copy_split(args.base_data_dir, args.base_dataset, args.out_dir, args.out_dataset, split, "negative")
    summary = {"base_dataset":args.base_dataset, "out_dataset":args.out_dataset, "rho":args.rho, "original_train":len(train_rows), "added":len(added), "users_with_added":len(added_by_user), "min_p1":args.min_p1, "min_margin":args.min_margin}
    with open(os.path.join(args.out_dir, "summary_augmented_train.json"), "w", encoding="utf-8") as f: json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
