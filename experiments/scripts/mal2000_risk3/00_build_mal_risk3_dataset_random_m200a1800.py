#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="Data/myanimelist_raw")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_minor", type=int, default=200)
    ap.add_argument("--n_adult", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--min_total_rated", type=int, default=50)
    ap.add_argument("--max_total_rated", type=int, default=300)
    ap.add_argument("--min_pos", type=int, default=10)
    ap.add_argument("--pos_th", type=int, default=6)
    ap.add_argument("--chunksize", type=int, default=1000000)
    return ap.parse_args()


def norm_col(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def read_csv_header(path):
    return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig", encoding_errors="replace").columns)


def choose_cols(path, wanted):
    fields = read_csv_header(path)
    norm_map = {norm_col(c): c for c in fields}
    out = []
    for w in wanted:
        key = norm_col(w)
        if key in norm_map:
            out.append(norm_map[key])
    return out


def parse_dt(x):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if not s or s in {"0000-00-00", "0000-00-00 00:00:00"}:
        return pd.NaT
    return pd.to_datetime(s, errors="coerce")


def calc_age_at_join(birth, join):
    b = parse_dt(birth)
    j = parse_dt(join)
    if pd.isna(b) or pd.isna(j):
        return np.nan
    age = j.year - b.year - int((j.month, j.day) < (b.month, b.day))
    if age < 0 or age > 100:
        return np.nan
    return age


def map_rating_bucket(x):
    s = str(x).strip()
    m = {
        "G - All Ages": "G",
        "PG - Children": "PG",
        "PG-13 - Teens 13 or older": "PG13",
        "R - 17+ (violence & profanity)": "R17",
        "R+ - Mild Nudity": "RPLUS",
        "Rx - Hentai": "RX",
    }
    return m.get(s, "UNKNOWN")


def rating_code(bucket):
    return {
        "G": 0,
        "PG": 1,
        "PG13": 2,
        "R17": 3,
        "RPLUS": 4,
        "RX": 5,
    }.get(bucket, -1)


def make_timestamp(df):
    ts = pd.Series(pd.NaT, index=df.index)

    if "my_last_updated" in df.columns:
        raw = df["my_last_updated"]
        num = pd.to_numeric(raw, errors="coerce")
        # MAL 这个字段通常是 Unix 秒级时间戳
        ts1 = pd.to_datetime(num, unit="s", errors="coerce")
        ts = ts.fillna(ts1)

    for col in ["my_finish_date", "my_start_date"]:
        if col in df.columns:
            ts2 = pd.to_datetime(df[col], errors="coerce")
            ts = ts.fillna(ts2)

    return ts


def main():
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    anime_path = raw_dir / "anime_filtered.csv"
    users_path = raw_dir / "users_filtered.csv"
    inter_path = raw_dir / "animelists_filtered.csv"

    out_data = Path("Data")
    out_root = Path("outputs") / args.dataset
    tables_dir = out_root / "tables"
    safe_dir = out_root / "safe_features"

    out_data.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    safe_dir.mkdir(parents=True, exist_ok=True)

    print("[load] anime:", anime_path)
    anime_cols = choose_cols(anime_path, [
        "anime_id", "title", "title_english", "type", "source", "episodes",
        "rating", "score", "scored_by", "rank", "popularity", "members",
        "favorites", "genre", "studio", "producer"
    ])
    anime = pd.read_csv(anime_path, usecols=anime_cols, encoding="utf-8-sig", encoding_errors="replace", low_memory=False)
    anime["anime_id"] = pd.to_numeric(anime["anime_id"], errors="coerce").astype("Int64")
    anime = anime.dropna(subset=["anime_id"]).copy()
    anime["anime_id"] = anime["anime_id"].astype(int)
    anime = anime.drop_duplicates("anime_id")
    anime_ids = set(anime["anime_id"].tolist())

    anime["rating_bucket"] = anime["rating"].map(map_rating_bucket) if "rating" in anime.columns else "UNKNOWN"
    anime["rating_code"] = anime["rating_bucket"].map(rating_code)

    print("[load] users:", users_path)
    user_cols = choose_cols(users_path, [
        "username", "user_id", "gender", "location", "birth_date", "join_date",
        "stats_mean_score", "user_completed", "user_days_spent_watching"
    ])
    users = pd.read_csv(users_path, usecols=user_cols, encoding="utf-8-sig", encoding_errors="replace", low_memory=False)
    users = users.dropna(subset=["username"]).copy()
    users["username"] = users["username"].astype(str)

    if "user_id" in users.columns:
        users["user_id"] = pd.to_numeric(users["user_id"], errors="coerce").astype("Int64")

    users["age_at_join"] = users.apply(
        lambda r: calc_age_at_join(r.get("birth_date", None), r.get("join_date", None)),
        axis=1
    )
    users = users.dropna(subset=["age_at_join"]).copy()
    users["age_at_join"] = users["age_at_join"].astype(int)
    users["is_minor"] = users["age_at_join"] < 18

    print("[pass1] count interactions by user")
    inter_cols = choose_cols(inter_path, [
        "username", "anime_id", "my_score", "my_status",
        "my_watched_episodes", "my_start_date", "my_finish_date", "my_last_updated"
    ])

    if not {"username", "anime_id", "my_score"}.issubset(set(inter_cols)):
        raise ValueError(f"Missing required interaction columns. Got={inter_cols}")

    total_count = Counter()
    pos_count = Counter()

    for chunk in pd.read_csv(
        inter_path,
        usecols=inter_cols,
        chunksize=args.chunksize,
        encoding="utf-8-sig", encoding_errors="replace",
        low_memory=False
    ):
        chunk = chunk.dropna(subset=["username", "anime_id", "my_score"]).copy()
        chunk["username"] = chunk["username"].astype(str)
        chunk["anime_id"] = pd.to_numeric(chunk["anime_id"], errors="coerce")
        chunk["my_score"] = pd.to_numeric(chunk["my_score"], errors="coerce")
        chunk = chunk.dropna(subset=["anime_id", "my_score"])
        chunk["anime_id"] = chunk["anime_id"].astype(int)
        chunk["my_score"] = chunk["my_score"].astype(int)

        chunk = chunk[(chunk["my_score"] > 0) & (chunk["anime_id"].isin(anime_ids))]
        if chunk.empty:
            continue

        vc_total = chunk.groupby("username").size()
        vc_pos = chunk[chunk["my_score"] > args.pos_th].groupby("username").size()

        total_count.update(vc_total.to_dict())
        pos_count.update(vc_pos.to_dict())

    cnt = pd.DataFrame({
        "username": list(total_count.keys()),
        "total_rated": [total_count[u] for u in total_count.keys()],
        "pos_count": [pos_count.get(u, 0) for u in total_count.keys()],
    })

    users2 = users.merge(cnt, on="username", how="inner")
    eligible = users2[
        (users2["total_rated"] >= args.min_total_rated) &
        (users2["total_rated"] <= args.max_total_rated) &
        (users2["pos_count"] >= args.min_pos)
    ].copy()

    minor_pool = eligible[eligible["is_minor"]].copy()
    adult_pool = eligible[~eligible["is_minor"]].copy()

    print("[eligible] minor_pool =", len(minor_pool))
    print("[eligible] adult_pool =", len(adult_pool))

    if len(minor_pool) < args.n_minor:
        raise ValueError(f"minor_pool={len(minor_pool)} < n_minor={args.n_minor}")
    if len(adult_pool) < args.n_adult:
        raise ValueError(f"adult_pool={len(adult_pool)} < n_adult={args.n_adult}")

    rng = np.random.default_rng(args.seed)

    minor_sel = minor_pool.sample(n=args.n_minor, random_state=args.seed)
    adult_sel = adult_pool.sample(n=args.n_adult, random_state=args.seed + 1)
    selected_users = pd.concat([minor_sel, adult_sel], ignore_index=True)
    selected_users = selected_users.sample(frac=1.0, random_state=args.seed + 2).reset_index(drop=True)

    selected_usernames = set(selected_users["username"].astype(str).tolist())
    selected_users["inner_user_id"] = np.arange(len(selected_users), dtype=int)

    print("[select] users =", len(selected_users))
    print("[select] minor =", int(selected_users["is_minor"].sum()))
    print("[select] adult =", int((~selected_users["is_minor"]).sum()))

    print("[pass2] load selected user interactions")
    chunks = []
    for chunk in pd.read_csv(
        inter_path,
        usecols=inter_cols,
        chunksize=args.chunksize,
        encoding="utf-8-sig", encoding_errors="replace",
        low_memory=False
    ):
        chunk = chunk.dropna(subset=["username", "anime_id", "my_score"]).copy()
        chunk["username"] = chunk["username"].astype(str)
        chunk = chunk[chunk["username"].isin(selected_usernames)]
        if chunk.empty:
            continue

        chunk["anime_id"] = pd.to_numeric(chunk["anime_id"], errors="coerce")
        chunk["my_score"] = pd.to_numeric(chunk["my_score"], errors="coerce")
        chunk = chunk.dropna(subset=["anime_id", "my_score"])
        chunk["anime_id"] = chunk["anime_id"].astype(int)
        chunk["my_score"] = chunk["my_score"].astype(int)

        chunk = chunk[(chunk["my_score"] > 0) & (chunk["anime_id"].isin(anime_ids))]
        if chunk.empty:
            continue

        chunks.append(chunk)

    inter = pd.concat(chunks, ignore_index=True)
    inter = inter.drop_duplicates(["username", "anime_id"], keep="last").copy()
    inter["timestamp"] = make_timestamp(inter)
    inter["timestamp"] = inter["timestamp"].fillna(pd.Timestamp("1970-01-01"))
    inter["timestamp_unix"] = (inter["timestamp"].astype("int64") // 10**9).astype("int64")
    inter["is_liked"] = inter["my_score"] > args.pos_th

    # 只保留选中用户完整交互中出现过的 items
    selected_item_ids = sorted(inter["anime_id"].unique().tolist())
    item_map = {rid: idx for idx, rid in enumerate(selected_item_ids)}
    user_map = dict(zip(selected_users["username"].astype(str), selected_users["inner_user_id"]))

    inter["inner_user_id"] = inter["username"].map(user_map).astype(int)
    inter["inner_item_id"] = inter["anime_id"].map(item_map).astype(int)

    # items 表
    items = anime[anime["anime_id"].isin(selected_item_ids)].copy()
    items["inner_item_id"] = items["anime_id"].map(item_map).astype(int)
    items = items.sort_values("inner_item_id").reset_index(drop=True)
    items = items.rename(columns={"anime_id": "raw_item_id"})
    items["anime_id"] = items["raw_item_id"]

    if "title" not in items.columns:
        items["title"] = items["raw_item_id"].astype(str)
    if "title_english" not in items.columns:
        items["title_english"] = ""
    if "genre" not in items.columns:
        items["genre"] = ""
    if "type" not in items.columns:
        items["type"] = ""
    if "source" not in items.columns:
        items["source"] = ""
    if "rating" not in items.columns:
        items["rating"] = ""

    # users 表
    users_out = selected_users.copy()
    if "user_id" in users_out.columns:
        users_out = users_out.rename(columns={"user_id": "raw_user_id"})
        users_out["user_id"] = users_out["raw_user_id"]
    else:
        users_out["raw_user_id"] = users_out["username"]
        users_out["user_id"] = users_out["username"]

    users_out["age_group"] = np.where(users_out["is_minor"], "minor", "adult")
    users_out["is_minor"] = users_out["is_minor"].astype(int)
    users_out = users_out.sort_values("inner_user_id").reset_index(drop=True)

    # interactions 表
    item_meta = items[["inner_item_id", "raw_item_id", "rating_bucket", "rating_code"]].copy()
    inter_out = inter.merge(item_meta, on="inner_item_id", how="left")
    inter_out = inter_out.rename(columns={"anime_id": "raw_item_id_from_inter"})
    inter_out["raw_item_id"] = inter_out["raw_item_id_from_inter"]
    inter_out["split"] = "non_liked_or_unused"

    train_rows, valid_rows, test_rows = [], [], []
    split_records = []

    for u, g in inter_out[inter_out["is_liked"]].groupby("inner_user_id"):
        g = g.sort_values(["timestamp_unix", "inner_item_id"]).copy()
        if len(g) < args.min_pos:
            raise ValueError(f"user {u} liked count < min_pos after selection.")

        train_g = g.iloc[:-2]
        valid_g = g.iloc[-2:-1]
        test_g = g.iloc[-1:]

        for _, r in train_g.iterrows():
            train_rows.append((int(r["inner_user_id"]), int(r["inner_item_id"]), 1))
            split_records.append({**r.to_dict(), "split": "train"})
        for _, r in valid_g.iterrows():
            valid_rows.append((int(r["inner_user_id"]), int(r["inner_item_id"]), 1))
            split_records.append({**r.to_dict(), "split": "valid"})
        for _, r in test_g.iterrows():
            test_rows.append((int(r["inner_user_id"]), int(r["inner_item_id"]), 1))
            split_records.append({**r.to_dict(), "split": "test"})

    split_df = pd.DataFrame(split_records)

    def write_rating(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for u, i, r in rows:
                f.write(f"{u}\t{i}\t{r}\n")

    train_path = out_data / f"{args.dataset}.train.rating"
    valid_path = out_data / f"{args.dataset}.valid.rating"
    test_path = out_data / f"{args.dataset}.test.rating"

    write_rating(train_path, train_rows)
    write_rating(valid_path, valid_rows)
    write_rating(test_path, test_rows)

    # Public exports retain numeric IDs and derived age, not profile identifiers.
    users_out["user_id"] = users_out["inner_user_id"]
    # 输出 tables
    users_cols = [
        "inner_user_id", "user_id", "raw_user_id", "username",
        "age_at_join", "is_minor", "age_group",
        "birth_date", "join_date", "gender", "location",
        "total_rated", "pos_count"
    ]
    users_cols = [c for c in users_cols if c in users_out.columns and c not in {"raw_user_id", "username", "birth_date", "join_date", "gender", "location"}]
    users_out[users_cols].to_csv(tables_dir / f"{args.dataset}.users.csv", index=False, encoding="utf-8-sig")

    items_cols = [
        "inner_item_id", "anime_id", "raw_item_id", "title", "title_english",
        "type", "source", "episodes", "rating", "rating_bucket", "rating_code",
        "score", "scored_by", "rank", "popularity", "members", "favorites",
        "genre", "studio", "producer"
    ]
    items_cols = [c for c in items_cols if c in items.columns]
    items[items_cols].to_csv(tables_dir / f"{args.dataset}.items.csv", index=False, encoding="utf-8-sig")

    inter_cols_out = [
        "inner_user_id", "inner_item_id", "username", "raw_item_id",
        "my_score", "is_liked", "timestamp", "timestamp_unix",
        "rating_bucket", "rating_code"
    ]
    inter_cols_out = [c for c in inter_cols_out if c in inter_out.columns and c not in {"username", "raw_user_id"}]
    inter_out[inter_cols_out].to_csv(tables_dir / f"{args.dataset}.interactions.csv", index=False, encoding="utf-8-sig")

    split_cols_out = [
        "inner_user_id", "inner_item_id", "username", "raw_item_id",
        "my_score", "is_liked", "timestamp", "timestamp_unix",
        "rating_bucket", "rating_code", "split"
    ]
    split_cols_out = [c for c in split_cols_out if c in split_df.columns and c not in {"username", "raw_user_id"}]
    split_df[split_cols_out].to_csv(tables_dir / f"{args.dataset}.chronological_split.csv", index=False, encoding="utf-8-sig")

    # item safe
    item_safe = items.copy()
    item_safe["item_id"] = item_safe["inner_item_id"]
    item_safe["R17"] = (item_safe["rating_bucket"] == "R17").astype(int)
    item_safe["RPLUS"] = (item_safe["rating_bucket"] == "RPLUS").astype(int)
    item_safe["RX"] = (item_safe["rating_bucket"] == "RX").astype(int)
    item_safe["is_r17"] = item_safe["R17"]
    item_safe["is_rplus"] = item_safe["RPLUS"]
    item_safe["is_rx"] = item_safe["RX"]
    item_safe["risk3_code"] = item_safe["rating_code"]

    safe_cols = [
        "inner_item_id", "item_id", "anime_id", "raw_item_id",
        "title", "title_english", "type", "source", "genre",
        "rating", "rating_bucket", "rating_code", "risk3_code",
        "R17", "RPLUS", "RX", "is_r17", "is_rplus", "is_rx"
    ]
    safe_cols = [c for c in safe_cols if c in item_safe.columns]
    item_safe[safe_cols].to_csv(safe_dir / f"{args.dataset}_features.item_safe.csv", index=False, encoding="utf-8-sig")

    # user tolerance 兼容旧流程
    user_tol = users_out.copy()
    user_tol["user_id"] = user_tol["inner_user_id"]
    user_tol["minor"] = user_tol["is_minor"]
    user_tol["isMinor"] = user_tol["is_minor"]
    user_tol["under18"] = user_tol["is_minor"]
    tol_cols = [
        "inner_user_id", "user_id", "raw_user_id", "username",
        "age_at_join", "is_minor", "minor", "isMinor", "under18", "age_group"
    ]
    tol_cols = [c for c in tol_cols if c in user_tol.columns and c not in {"username", "raw_user_id"}]
    user_tol[tol_cols].to_csv(safe_dir / f"{args.dataset}_features.user_tolerance_p75.csv", index=False, encoding="utf-8-sig")

    # 检查 pair overlap
    train_set = {(u, i) for u, i, _ in train_rows}
    valid_set = {(u, i) for u, i, _ in valid_rows}
    test_set = {(u, i) for u, i, _ in test_rows}

    assert len(train_set & valid_set) == 0
    assert len(train_set & test_set) == 0
    assert len(valid_set & test_set) == 0

    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "n_minor": args.n_minor,
        "n_adult": args.n_adult,
        "users": int(len(users_out)),
        "minor_users": int(users_out["is_minor"].sum()),
        "adult_users": int((1 - users_out["is_minor"]).sum()),
        "items": int(len(items)),
        "complete_interactions_score_gt0": int(len(inter_out)),
        "liked_interactions_score_gt_pos_th": int(inter_out["is_liked"].sum()),
        "train": int(len(train_rows)),
        "valid": int(len(valid_rows)),
        "test": int(len(test_rows)),
        "non_liked_score_le_pos_th": int((~inter_out["is_liked"]).sum()),
        "pos_th": args.pos_th,
        "min_total_rated": args.min_total_rated,
        "max_total_rated": args.max_total_rated,
        "min_pos": args.min_pos,
        "train_valid_overlap": len(train_set & valid_set),
        "train_test_overlap": len(train_set & test_set),
        "valid_test_overlap": len(valid_set & test_set),
        "rating_bucket_counts_items": dict(Counter(items["rating_bucket"])),
        "rating_bucket_counts_interactions": dict(Counter(inter_out["rating_bucket"])),
        "paths": {
            "train": str(train_path),
            "valid": str(valid_path),
            "test": str(test_path),
            "users": str(tables_dir / f"{args.dataset}.users.csv"),
            "items": str(tables_dir / f"{args.dataset}.items.csv"),
            "interactions": str(tables_dir / f"{args.dataset}.interactions.csv"),
            "chronological_split": str(tables_dir / f"{args.dataset}.chronological_split.csv"),
            "item_safe": str(safe_dir / f"{args.dataset}_features.item_safe.csv"),
            "user_tolerance": str(safe_dir / f"{args.dataset}_features.user_tolerance_p75.csv"),
        }
    }

    summary_path = out_root / "dataset_build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOK")


if __name__ == "__main__":
    main()
