#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import pickle
import re
from collections import defaultdict


def split_line(line):
    return re.split(r"[\s,]+", line.strip())


def read_rating(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for order, line in enumerate(f):
            if not line.strip():
                continue
            sp = split_line(line)
            if len(sp) < 2:
                continue
            u = int(float(sp[0]))
            i = int(float(sp[1]))

            ts = None
            if len(sp) >= 4:
                try:
                    ts = float(sp[3])
                except Exception:
                    ts = None

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


def read_csv_dict(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        out = []
        for row in reader:
            out.append({str(k).replace("\ufeff", "").strip(): v for k, v in row.items()})
        return out


def pick_col(row, candidates):
    for c in candidates:
        if c in row:
            return c
    return None


def read_item_map(path):
    """
    Return inner_item_id -> raw_movie_id when possible.
    Be tolerant to different column names.
    """
    rows = read_csv_dict(path)
    mp = {}
    if not rows:
        return mp

    first = rows[0]
    inner_col = pick_col(first, ["inner_item_id", "item_id", "inner_id", "new_item_id"])
    raw_col = pick_col(first, ["raw_item_id", "movie_id", "movieId", "raw_movie_id", "old_item_id"])

    if inner_col is None or raw_col is None:
        return mp

    for r in rows:
        try:
            inner = int(float(r[inner_col]))
            raw = int(float(r[raw_col]))
            mp[inner] = raw
        except Exception:
            continue

    return mp


def read_movies_dat(path):
    title = {}
    desc = {}

    if not path or not os.path.exists(path):
        return title, desc

    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            sp = line.rstrip("\n").split("::")
            if len(sp) < 3:
                continue
            try:
                raw_mid = int(sp[0])
            except Exception:
                continue
            title[raw_mid] = sp[1]
            desc[raw_mid] = f"Genres: {sp[2]}"

    return title, desc


def read_item_safe_csv(path):
    safe = {}

    rows = read_csv_dict(path)
    if not rows:
        return safe

    risk_cols = ["sex", "violence", "profanity", "drug", "intense", "adult_content", "unsafe_any"]

    first = rows[0]
    id_col = pick_col(first, ["inner_item_id", "item_id", "movie_id", "movieId", "id"])
    if id_col is None:
        return safe

    for r in rows:
        try:
            i = int(float(r[id_col]))
        except Exception:
            continue

        parts = []
        for c in risk_cols:
            if c in r and str(r[c]).strip() != "":
                parts.append(f"{c}={r[c]}")
        if parts:
            safe[i] = "Safety attributes: " + ", ".join(parts)

    return safe


def write_split(path, user_to_seq):
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
    ap.add_argument("--item_map_csv", default="")
    ap.add_argument("--movies_dat", default="")
    ap.add_argument("--item_safe_csv", default="")

    ap.add_argument("--sort_by_ts", action="store_true")
    ap.add_argument("--synthetic_ts_start", type=int, default=978300000000)
    ap.add_argument("--synthetic_ts_step", type=int, default=86400000)
    args = ap.parse_args()

    base_train_rows = read_rating(args.base_train)
    valid_rows = read_rating(args.base_valid)
    test_rows = read_rating(args.base_test)

    base_train = group_by_user(base_train_rows, sort_by_ts=args.sort_by_ts)
    valid = group_by_user(valid_rows, sort_by_ts=args.sort_by_ts)
    test = group_by_user(test_rows, sort_by_ts=args.sort_by_ts)

    heldout_pairs = {(u, i) for u, i, _, _ in valid_rows + test_rows}
    base_pairs = {(u, i) for u, i, _, _ in base_train_rows}

    added_pseudo = 0
    removed_heldout = 0

    if args.aug_train:
        aug_rows = read_rating(args.aug_train)
        aug_by_user = group_by_user(aug_rows, sort_by_ts=args.sort_by_ts)
        train = defaultdict(list)

        for u in sorted(set(list(base_train.keys()) + list(aug_by_user.keys()))):
            seq = list(base_train.get(u, []))
            existing_items = {i for i, _, _ in seq}

            for i, ts, order in aug_by_user.get(u, []):
                if (u, i) in base_pairs:
                    continue
                if (u, i) in heldout_pairs:
                    removed_heldout += 1
                    continue
                if i in existing_items:
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

    max_user = max(all_users)
    max_item = max(all_items)

    inner_to_raw = read_item_map(args.item_map_csv)
    movie_title, movie_desc = read_movies_dat(args.movies_dat)
    item_safe = read_item_safe_csv(args.item_safe_csv)

    title = {}
    description = {}

    for inner_i in range(max_item + 1):
        llm_i = inner_i + 1
        raw_i = inner_to_raw.get(inner_i, inner_i)

        t = movie_title.get(raw_i, movie_title.get(inner_i, f"Movie {inner_i}"))
        d = movie_desc.get(raw_i, movie_desc.get(inner_i, ""))

        parts = []
        if d:
            parts.append(d)
        if inner_i in item_safe:
            parts.append(item_safe[inner_i])

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
        "item_map_csv": args.item_map_csv,
        "movies_dat": args.movies_dat,
        "item_safe_csv": args.item_safe_csv,
    }

    with open(os.path.join(out_dir, "prepare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
