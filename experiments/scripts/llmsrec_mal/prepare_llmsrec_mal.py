#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare MyAnimeList / MAL rating files for LLM-SRec.

Output format expected by LLM-SRec:
  <llmsrec_root>/SeqRec/data_<out_name>/<out_name>_train.txt
  <llmsrec_root>/SeqRec/data_<out_name>/<out_name>_valid.txt
  <llmsrec_root>/SeqRec/data_<out_name>/<out_name>_test.txt
  <llmsrec_root>/SeqRec/data_<out_name>/text_name_dict.json.gz

Protocol:
  - base mode: train = original train
  - augmented mode: train = original train + pseudo positives from aug_train
  - valid/test pairs are removed from pseudo positives to avoid leakage
  - IDs are shifted by +1 because LLM-SRec uses 1-based ids internally
"""

import argparse
import csv
import json
import os
import pickle
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def split_line(line: str) -> List[str]:
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


def safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return default


def safe_float(x: Any, default=None):
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(str(x).strip())
    except Exception:
        return default


def read_rating(path: str) -> List[Tuple[int, int, float, int]]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for order, line in enumerate(f):
            parts = split_line(line)
            if len(parts) < 2:
                continue
            if order == 0 and safe_int(parts[0], -999999) == -999999:
                continue
            u = safe_int(parts[0], -1)
            i = safe_int(parts[1], -1)
            if u < 0 or i < 0:
                continue
            ts = None
            if len(parts) >= 4:
                ts = safe_float(parts[3], None)
            rows.append((u, i, ts, order))
    return rows


def group_by_user(rows, sort_by_ts=False):
    d = defaultdict(list)
    seen = defaultdict(set)
    if sort_by_ts:
        rows = sorted(rows, key=lambda x: (x[0], x[2] if x[2] is not None else x[3], x[3]))
    for u, i, ts, order in rows:
        if i in seen[u]:
            continue
        d[u].append((i, ts, order))
        seen[u].add(i)
    return d


def read_csv_dict(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path or not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = []
        for row in reader:
            rows.append({str(k).replace("\ufeff", "").strip(): v for k, v in row.items()})
        return rows, reader.fieldnames or []


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def pick_col(fields: List[str], candidates: List[str], required=False) -> str:
    norm_to_real = {norm_name(f): f for f in fields}
    for c in candidates:
        if norm_name(c) in norm_to_real:
            return norm_to_real[norm_name(c)]
    for c in candidates:
        key = norm_name(c)
        for nf, real in norm_to_real.items():
            if key and key in nf:
                return real
    if required:
        raise ValueError(f"Cannot find column from candidates={candidates}. fields={fields}")
    return ""


def read_item_map(path: str) -> Dict[int, int]:
    rows, fields = read_csv_dict(path)
    if not rows:
        return {}
    inner_col = pick_col(fields, ["inner_item_id", "item_id", "iid", "inner_id", "new_item_id"], False)
    raw_col = pick_col(fields, ["raw_item_id", "anime_id", "animeId", "raw_anime_id", "old_item_id"], False)
    if not inner_col or not raw_col:
        return {}
    out = {}
    for r in rows:
        inner = safe_int(r.get(inner_col), -1)
        raw = safe_int(r.get(raw_col), -1)
        if inner >= 0 and raw >= 0:
            out[inner] = raw
    return out


def read_anime_csv(path: str):
    rows, fields = read_csv_dict(path)
    if not rows:
        return {}, {}
    id_col = pick_col(fields, ["anime_id", "animeId", "raw_anime_id", "id"], False)
    title_col = pick_col(fields, ["name", "title", "anime_title"], False)
    genre_col = pick_col(fields, ["genre", "genres"], False)
    type_col = pick_col(fields, ["type"], False)
    rating_col = pick_col(fields, ["rating", "age_rating", "rated"], False)
    episodes_col = pick_col(fields, ["episodes", "episode"], False)
    source_col = pick_col(fields, ["source"], False)
    title, desc = {}, {}
    for idx, r in enumerate(rows):
        raw = safe_int(r.get(id_col), -1) if id_col else idx
        if raw < 0:
            continue
        t = str(r.get(title_col, "")).strip() if title_col else ""
        if not t:
            t = f"Anime {raw}"
        parts = []
        if genre_col and str(r.get(genre_col, "")).strip():
            parts.append(f"Genres: {r.get(genre_col)}")
        if type_col and str(r.get(type_col, "")).strip():
            parts.append(f"Type: {r.get(type_col)}")
        if episodes_col and str(r.get(episodes_col, "")).strip():
            parts.append(f"Episodes: {r.get(episodes_col)}")
        if source_col and str(r.get(source_col, "")).strip():
            parts.append(f"Source: {r.get(source_col)}")
        if rating_col and str(r.get(rating_col, "")).strip():
            parts.append(f"Rating bucket: {r.get(rating_col)}")
        title[raw] = t
        desc[raw] = "; ".join(parts) if parts else "No description"
    return title, desc


def read_item_risk_csv(path: str) -> Dict[int, str]:
    rows, fields = read_csv_dict(path)
    if not rows:
        return {}
    id_col = pick_col(fields, ["inner_item_id", "item_id", "iid", "anime_id", "id"], False)
    if not id_col:
        return {}
    risk_cols = [
        "rating_bucket", "risk_bucket", "risk_label", "age_rating",
        "PG", "PG13", "R17", "RPLUS", "RX", "R-17", "R+", "Rx",
        "unsafe_any", "minor_unsafe", "adult_unsafe",
    ]
    risk_norms = {norm_name(x) for x in risk_cols}
    real_cols = [c for c in fields if norm_name(c) in risk_norms]
    out = {}
    for r in rows:
        iid = safe_int(r.get(id_col), -1)
        if iid < 0:
            continue
        parts = []
        for c in real_cols:
            v = str(r.get(c, "")).strip()
            if v != "":
                parts.append(f"{c}={v}")
        if parts:
            out[iid] = "Safety/Risk attributes: " + ", ".join(parts)
    return out


def write_split(path: str, user_to_seq):
    with open(path, "w", encoding="utf-8") as f:
        for u in sorted(user_to_seq):
            llm_u = u + 1
            for i, _, _ in user_to_seq[u]:
                llm_i = i + 1
                f.write(f"{llm_u} {llm_i}\n")


def make_time_ms(ts, fallback_ms):
    if ts is None:
        return int(fallback_ms)
    if ts < 10**12:
        return int(ts * 1000)
    return int(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmsrec_root", required=True)
    ap.add_argument("--out_name", required=True)
    ap.add_argument("--base_train", required=True)
    ap.add_argument("--base_valid", required=True)
    ap.add_argument("--base_test", required=True)
    ap.add_argument("--aug_train", default="")
    ap.add_argument("--anime_csv", default="")
    ap.add_argument("--item_map_csv", default="")
    ap.add_argument("--item_risk_csv", default="")
    ap.add_argument("--sort_by_ts", action="store_true")
    ap.add_argument("--synthetic_ts_start", type=int, default=978300000000)
    ap.add_argument("--synthetic_ts_step", type=int, default=86400000)
    args = ap.parse_args()

    base_train_rows = read_rating(args.base_train)
    valid_rows = read_rating(args.base_valid)
    test_rows = read_rating(args.base_test)

    base_train = group_by_user(base_train_rows, args.sort_by_ts)
    valid = group_by_user(valid_rows, args.sort_by_ts)
    test = group_by_user(test_rows, args.sort_by_ts)

    heldout_pairs = {(u, i) for u, i, _, _ in valid_rows + test_rows}
    base_pairs = {(u, i) for u, i, _, _ in base_train_rows}

    added_pseudo = 0
    removed_heldout = 0
    duplicate_removed = 0

    if args.aug_train:
        aug_rows = read_rating(args.aug_train)
        aug_by_user = group_by_user(aug_rows, args.sort_by_ts)
        train = defaultdict(list)
        for u in sorted(set(base_train.keys()) | set(aug_by_user.keys())):
            seq = list(base_train.get(u, []))
            existing_items = {i for i, _, _ in seq}
            for i, ts, order in aug_by_user.get(u, []):
                if (u, i) in base_pairs:
                    continue
                if (u, i) in heldout_pairs:
                    removed_heldout += 1
                    continue
                if i in existing_items:
                    duplicate_removed += 1
                    continue
                seq.append((i, None, 10**12 + order))
                existing_items.add(i)
                added_pseudo += 1
            train[u] = seq
    else:
        train = base_train

    for u, _, _, _ in valid_rows + test_rows:
        train.setdefault(u, [])

    all_users = set(train.keys()) | set(valid.keys()) | set(test.keys())
    all_items = set()
    for d in [train, valid, test]:
        for seq in d.values():
            for i, _, _ in seq:
                all_items.add(i)

    if not all_users or not all_items:
        raise RuntimeError("Empty users/items after loading data.")

    max_user = max(all_users)
    max_item = max(all_items)

    inner_to_raw = read_item_map(args.item_map_csv)
    raw_title, raw_desc = read_anime_csv(args.anime_csv)
    item_risk = read_item_risk_csv(args.item_risk_csv)

    title, description = {}, {}
    for inner_i in range(max_item + 1):
        llm_i = inner_i + 1
        raw_i = inner_to_raw.get(inner_i, inner_i)
        t = raw_title.get(raw_i, raw_title.get(inner_i, f"Anime {inner_i}"))
        d = raw_desc.get(raw_i, raw_desc.get(inner_i, "No description"))
        parts = []
        if d:
            parts.append(d)
        if inner_i in item_risk:
            parts.append(item_risk[inner_i])
        title[llm_i] = t
        description[llm_i] = "; ".join(parts) if parts else "No description"

    time_dict = defaultdict(dict)
    for u in sorted(all_users):
        llm_u = u + 1
        pos = 0
        for split_dict in [train, valid, test]:
            for i, ts, _ in split_dict.get(u, []):
                llm_i = i + 1
                fallback = args.synthetic_ts_start + pos * args.synthetic_ts_step
                time_dict[llm_i][llm_u] = make_time_ms(ts, fallback)
                pos += 1

    out_dir = os.path.join(args.llmsrec_root, "SeqRec", f"data_{args.out_name}")
    os.makedirs(out_dir, exist_ok=True)

    write_split(os.path.join(out_dir, f"{args.out_name}_train.txt"), train)
    write_split(os.path.join(out_dir, f"{args.out_name}_valid.txt"), valid)
    write_split(os.path.join(out_dir, f"{args.out_name}_test.txt"), test)

    text_name_dict = {
        "title": title,
        "description": description,
        "time": {int(k): dict(v) for k, v in time_dict.items()},
    }
    with open(os.path.join(out_dir, "text_name_dict.json.gz"), "wb") as f:
        pickle.dump(text_name_dict, f)

    summary = {
        "out_name": args.out_name,
        "out_dir": out_dir,
        "mode": "augmented" if args.aug_train else "base",
        "num_users": max_user + 1,
        "num_items": max_item + 1,
        "train_lines": sum(len(v) for v in train.values()),
        "valid_lines": sum(len(v) for v in valid.values()),
        "test_lines": sum(len(v) for v in test.values()),
        "added_pseudo": added_pseudo,
        "removed_heldout_overlap": removed_heldout,
        "duplicate_removed": duplicate_removed,
        "anime_csv": args.anime_csv,
        "item_map_csv": args.item_map_csv,
        "item_risk_csv": args.item_risk_csv,
    }
    with open(os.path.join(out_dir, "prepare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
