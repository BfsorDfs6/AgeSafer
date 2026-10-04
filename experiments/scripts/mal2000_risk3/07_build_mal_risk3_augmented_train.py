#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build MAL risk3 augmented train.rating from PSG LoRA predictions.

Logic aligned with ML-1M build_psg_augmented_train.py:
1. Read original train.rating.
2. Read LoRA prediction JSONL.
3. Keep psg_pred=1 pseudo-positive candidates.
4. Sort candidates per user by p1 and margin.
5. Add ceil/round rho * real_train_count pseudo positives per user.
6. Copy original valid/test rating files to the augmented dataset directory.
7. Optionally enforce MAL risk3 safety:
   - minor: R17/RPLUS/RX unsafe
   - adult: RPLUS/RX unsafe
   - adult R17 is allowed
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def safe_int(x, default=-1):
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(str(x).strip()))
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(str(x).strip())
    except Exception:
        return default


def split_line(line):
    s = line.strip()
    if not s:
        return []
    if "::" in s:
        return s.split("::")
    if "\t" in s:
        return s.split("\t")
    if "," in s and not s.startswith("("):
        return [x.strip() for x in s.split(",")]
    return re.split(r"\s+", s)


def get_first(row, names, default=""):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return str(row[n])
    return default


def is_one(x):
    return str(x).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def read_train_rating(path):
    rows = []
    seen = set()
    user_real = defaultdict(int)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            arr = split_line(line)
            if len(arr) < 2:
                continue
            u = safe_int(arr[0], -1)
            i = safe_int(arr[1], -1)
            if u < 0 or i < 0:
                continue
            raw = line.strip()
            rows.append((u, i, raw))
            seen.add((u, i))
            user_real[u] += 1

    print(f"[load] train rows={len(rows)} users={len(user_real)} path={path}")
    return rows, seen, user_real


def item_id_from_item_safe(row):
    return safe_int(
        get_first(row, ["inner_item_id", "\ufeffinner_item_id", "item_id", "iid", "item"], "-1"),
        -1,
    )


def bucket_from_item_safe(row):
    rating = " ".join([
        str(row.get("rating_bucket", "")),
        str(row.get("rating", "")),
        str(row.get("rating_name", "")),
        str(row.get("content_rating", "")),
    ])
    t = rating.upper().replace(" ", "")

    if is_one(row.get("rx_code", 0)) or "RX" in t or "HENTAI" in t:
        return "RX"
    if is_one(row.get("rplus_code", 0)) or "R+" in t or "RPLUS" in t or "MILDNUDITY" in t:
        return "RPLUS"
    if is_one(row.get("r17_code", 0)) or "R-17" in t or "17+" in t or "R17" in t:
        return "R17"
    if "PG-13" in t or "PG13" in t:
        return "PG13"
    if "PG-CHILDREN" in t or t.startswith("PG"):
        return "PG"
    if "ALLAGES" in t or t.startswith("G"):
        return "G"
    return "OTHER"


def load_item_buckets(path):
    out = {}
    if not path:
        return out

    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = item_id_from_item_safe(row)
            if iid >= 0:
                out[iid] = bucket_from_item_safe(row)

    print(f"[load] item buckets={len(out)} path={path}")
    print("[bucket_counts]", dict(Counter(out.values())))
    return out


def mal_risk3_safe(is_minor, bucket):
    bucket = str(bucket or "").upper()
    if is_minor:
        return bucket not in {"R17", "RPLUS", "RX"}
    return bucket not in {"RPLUS", "RX"}


def meta_user_id(meta):
    return safe_int(
        get_first(meta, ["user_id", "inner_user_id", "uid", "user"], "-1"),
        -1,
    )


def meta_item_id(meta):
    return safe_int(
        get_first(meta, ["item_id", "candidate_inner_item_id", "inner_item_id", "iid", "item"], "-1"),
        -1,
    )


def read_predictions(path, item_buckets, min_p1, min_margin, enforce_safe):
    by_user = defaultdict(list)
    stats = Counter()
    pred1_bucket = Counter()
    filtered_bucket = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stats["pred_rows"] += 1
            obj = json.loads(line)

            if int(obj.get("psg_pred", 0)) != 1:
                stats["pred0_skip"] += 1
                continue

            p1 = safe_float(obj.get("p1", 1.0), 1.0)
            margin = safe_float(obj.get("margin_1_minus_0", 0.0), 0.0)

            if p1 < min_p1:
                stats["min_p1_skip"] += 1
                continue
            if margin < min_margin:
                stats["min_margin_skip"] += 1
                continue

            meta = obj.get("metadata", {})
            u = meta_user_id(meta)
            i = meta_item_id(meta)
            if u < 0 or i < 0:
                stats["bad_uid_iid_skip"] += 1
                continue

            is_minor = bool(meta.get("is_minor", False))
            bucket = str(meta.get("candidate_bucket", "") or "")
            if not bucket:
                bucket = item_buckets.get(i, "UNKNOWN")
            bucket = bucket.upper()

            pred1_bucket[bucket] += 1

            if enforce_safe and not mal_risk3_safe(is_minor, bucket):
                stats["unsafe_filtered"] += 1
                filtered_bucket[bucket] += 1
                continue

            by_user[u].append({
                "user_id": u,
                "item_id": i,
                "p1": p1,
                "margin": margin,
                "is_minor": is_minor,
                "candidate_bucket": bucket,
                "rank": safe_int(meta.get("candidate_rank", 999999), 999999),
                "metadata": meta,
            })
            stats["kept_psg_positive"] += 1

    print(f"[load] pred rows={stats['pred_rows']} kept={stats['kept_psg_positive']} users={len(by_user)} path={path}")
    print("[pred1_bucket]", dict(pred1_bucket))
    print("[filtered_bucket]", dict(filtered_bucket))
    return by_user, stats, pred1_bucket, filtered_bucket


def copy_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print(f"[copy] {src} -> {dst}")
        return True
    print(f"[warn] missing: {src}")
    return False


def copy_splits(base_data_dir, base_dataset, out_dir, out_dataset):
    copied = []
    for split in ["valid", "test"]:
        for suffix in ["rating", "negative"]:
            src = os.path.join(base_data_dir, f"{base_dataset}.{split}.{suffix}")
            dst = os.path.join(out_dir, f"{out_dataset}.{split}.{suffix}")
            if copy_if_exists(src, dst):
                copied.append(dst)
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_data_dir", default="Data")
    ap.add_argument("--base_dataset", required=True)
    ap.add_argument("--train_rating", default="")
    ap.add_argument("--item_safe", default="")
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_dataset", required=True)
    ap.add_argument("--summary_json", default="")
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--max_per_user", type=int, default=0)
    ap.add_argument("--min_p1", type=float, default=0.0)
    ap.add_argument("--min_margin", type=float, default=-999.0)
    ap.add_argument("--rating_value", default="1")
    ap.add_argument("--quota_mode", choices=["round", "ceil", "floor"], default="round")
    ap.add_argument("--enforce_mal_risk3_safe", action="store_true")
    ap.add_argument("--write_pseudo_jsonl", default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_path = args.train_rating or os.path.join(args.base_data_dir, f"{args.base_dataset}.train.rating")
    train_rows, seen, user_real = read_train_rating(train_path)

    item_buckets = load_item_buckets(args.item_safe) if args.item_safe else {}

    preds, pred_stats, pred1_bucket, filtered_bucket = read_predictions(
        path=args.pred_jsonl,
        item_buckets=item_buckets,
        min_p1=args.min_p1,
        min_margin=args.min_margin,
        enforce_safe=args.enforce_mal_risk3_safe,
    )

    added = []
    added_meta = []
    added_by_user = Counter()
    shortage_users = 0
    duplicate_skip = 0

    all_users = sorted(user_real.keys())

    for u in all_users:
        real_n = user_real.get(u, 0)
        raw_quota = float(args.rho) * max(1, real_n)

        if args.quota_mode == "ceil":
            quota = int(math.ceil(raw_quota))
        elif args.quota_mode == "floor":
            quota = int(math.floor(raw_quota))
        else:
            quota = int(round(raw_quota))

        quota = max(1, quota)

        if args.max_per_user > 0:
            quota = min(quota, args.max_per_user)

        arr = preds.get(u, [])
        arr.sort(key=lambda x: (x["p1"], x["margin"], -x["rank"]), reverse=True)

        used = 0
        for rec in arr:
            i = rec["item_id"]
            if (u, i) in seen:
                duplicate_skip += 1
                continue

            seen.add((u, i))
            added.append((u, i))
            added_by_user[u] += 1
            used += 1
            added_meta.append(rec)

            if used >= quota:
                break

        if used < quota:
            shortage_users += 1

    out_train = os.path.join(args.out_dir, f"{args.out_dataset}.train.rating")
    with open(out_train, "w", encoding="utf-8") as f:
        for _, _, raw in train_rows:
            f.write(raw + "\n")
        for u, i in added:
            f.write(f"{u}\t{i}\t{args.rating_value}\n")

    copied = copy_splits(args.base_data_dir, args.base_dataset, args.out_dir, args.out_dataset)

    pseudo_jsonl = args.write_pseudo_jsonl or os.path.join(args.out_dir, f"{args.out_dataset}.pseudo_added.jsonl")
    with open(pseudo_jsonl, "w", encoding="utf-8") as f:
        for rec in added_meta:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    added_bucket = Counter(rec["candidate_bucket"] for rec in added_meta)
    added_minor = Counter("minor" if rec["is_minor"] else "adult" for rec in added_meta)

    summary = {
        "version": "mal_risk3_augmented_train_v1",
        "base_dataset": args.base_dataset,
        "out_dataset": args.out_dataset,
        "rho": args.rho,
        "quota_mode": args.quota_mode,
        "max_per_user": args.max_per_user,
        "min_p1": args.min_p1,
        "min_margin": args.min_margin,
        "enforce_mal_risk3_safe": bool(args.enforce_mal_risk3_safe),
        "original_train": len(train_rows),
        "users_in_train": len(user_real),
        "pred_rows": pred_stats["pred_rows"],
        "kept_psg_positive_after_filters": pred_stats["kept_psg_positive"],
        "unsafe_filtered": pred_stats["unsafe_filtered"],
        "duplicate_skip": duplicate_skip,
        "added": len(added),
        "total_train": len(train_rows) + len(added),
        "actual_added_ratio": len(added) / max(1, len(train_rows)),
        "users_with_added": len(added_by_user),
        "quota_shortage_users": shortage_users,
        "pred1_bucket_counts_before_safe_filter": dict(pred1_bucket),
        "unsafe_filtered_bucket_counts": dict(filtered_bucket),
        "added_bucket_counts": dict(added_bucket),
        "added_scope_counts": dict(added_minor),
        "train_path": out_train,
        "pseudo_added_jsonl": pseudo_jsonl,
        "copied_splits": copied,
    }

    summary_path = args.summary_json or os.path.join(args.out_dir, "summary_augmented_train.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[write] train original={len(train_rows)} added={len(added)} total={len(train_rows) + len(added)} path={out_train}")
    print(f"[write] pseudo_added={pseudo_jsonl}")
    print(f"[write] summary={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
