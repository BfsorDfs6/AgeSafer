#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE:
  scripts/mal2000_risk3/01_build_profile_history.py

PURPOSE:
  Build MAL profile-history CSV for user profile generation.

RUN COMMAND:

cd .

DATASET=mal2000_risk3_seq50_300

python -u scripts/mal2000_risk3/01_build_profile_history.py \
  --dataset ${DATASET}

OUTPUT:
  outputs/${DATASET}/profile_inputs/${DATASET}.train_profile_history.csv

NOTES:
  - Uses only Data/${DATASET}.train.rating positive interactions.
  - Does not use valid/test interactions.
  - Keeps continuous inner_user_id / inner_item_id for model-side use.
  - Keeps raw_user_id / username / raw_item_id for traceability.
"""

import argparse
from pathlib import Path
import pandas as pd


def read_rating(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["inner_user_id", "inner_item_id", "rating"],
        engine="python",
    )
    df["inner_user_id"] = pd.to_numeric(df["inner_user_id"], errors="coerce").astype("Int64")
    df["inner_item_id"] = pd.to_numeric(df["inner_item_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["inner_user_id", "inner_item_id"])
    df["inner_user_id"] = df["inner_user_id"].astype(int)
    df["inner_item_id"] = df["inner_item_id"].astype(int)
    return df


def choose_col(df: pd.DataFrame, candidates, required=False):
    cols = list(df.columns)
    norm = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if str(c).lower() in norm:
            return norm[str(c).lower()]
    for c in candidates:
        key = str(c).lower()
        for real in cols:
            if key in str(real).lower():
                return real
    if required:
        raise KeyError(f"Cannot find any column among {candidates}; available={cols}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mal2000_risk3_seq50_300")
    ap.add_argument("--data_dir", default="Data")
    ap.add_argument("--out_root", default="outputs")
    args = ap.parse_args()

    dataset = args.dataset
    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root) / dataset
    table_dir = out_root / "tables"
    out_dir = out_root / "profile_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / f"{dataset}.train.rating"
    users_path = table_dir / f"{dataset}.users.csv"
    items_path = table_dir / f"{dataset}.items.csv"
    inter_path = table_dir / f"{dataset}.interactions.csv"

    train = read_rating(train_path)
    users = pd.read_csv(users_path)
    items = pd.read_csv(items_path)
    inter = pd.read_csv(inter_path)

    # Normalize key columns.
    for df, cols in [(users, ["inner_user_id"]), (items, ["inner_item_id"]), (inter, ["inner_user_id", "inner_item_id"])]:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.dropna(subset=[c], inplace=True)
            df[c] = df[c].astype(int)

    # Only positive train interactions.
    if "pref_label" in inter.columns:
        inter = inter[inter["pref_label"] == 1].copy()

    # Keep only train pairs so valid/test are not used for profile.
    hist = train[["inner_user_id", "inner_item_id"]].drop_duplicates().merge(
        inter,
        on=["inner_user_id", "inner_item_id"],
        how="inner",
    )

    # Merge user metadata. Rename first to avoid suffix collisions.
    user_cols = ["inner_user_id"]
    for c in [
        "raw_user_id", "username",
        "age", "age_at_join", "is_minor", "age_group",
        "gender", "location",
        "n_rated", "n_pos", "n_neg",
        "total_rated", "pos_count"
    ]:
        if c in users.columns and c not in user_cols:
            user_cols.append(c)
    hist = hist.merge(users[user_cols], on="inner_user_id", how="left", suffixes=("", "_user"))

    # Guarantee age column for downstream profile generation.
    # New MAL dataset builder uses age_at_join, while profile generator expects age.
    if "age" not in hist.columns and "age_at_join" in hist.columns:
        hist["age"] = hist["age_at_join"]
    if "age" in hist.columns:
        hist["age"] = pd.to_numeric(hist["age"], errors="coerce").fillna(-1).astype(int)
    else:
        hist["age"] = -1

    # Guarantee is_minor column.
    if "is_minor" in hist.columns:
        hist["is_minor"] = pd.to_numeric(hist["is_minor"], errors="coerce").fillna(0).astype(int)
    elif "age" in hist.columns:
        hist["is_minor"] = (hist["age"] < 18).astype(int)
    else:
        hist["is_minor"] = 0

    # Merge item metadata. Rename raw_item_id before merge if history already has one.
    item_cols = ["inner_item_id"]
    for c in [
        "raw_item_id", "title", "rating", "rating_bucket", "rating_code",
        "r17_code", "rplus_code", "rx_code", "genre", "type", "source", "anime_score"
    ]:
        if c in items.columns and c not in item_cols:
            item_cols.append(c)

    # If hist already has raw_item_id from interactions, keep it as raw_item_id_inter.
    if "raw_item_id" in hist.columns:
        hist = hist.rename(columns={"raw_item_id": "raw_item_id_inter"})
    hist = hist.merge(items[item_cols], on="inner_item_id", how="left", suffixes=("", "_item"))

    # Create a guaranteed raw_item_id column.
    if "raw_item_id" not in hist.columns:
        if "raw_item_id_item" in hist.columns:
            hist["raw_item_id"] = hist["raw_item_id_item"]
        elif "raw_item_id_inter" in hist.columns:
            hist["raw_item_id"] = hist["raw_item_id_inter"]
        else:
            hist["raw_item_id"] = -1

    # Make raw_user_id and username guaranteed too.
    if "raw_user_id" not in hist.columns:
        hist["raw_user_id"] = hist["inner_user_id"]
    if "username" not in hist.columns:
        hist["username"] = ""

    # Robust time handling for MAL generated tables.
    # Prefer explicit interaction_time if present; otherwise fall back to
    # timestamp, timestamp_unix, my_last_updated, my_finish_date, my_start_date.
    if "interaction_time" in hist.columns:
        hist["interaction_time"] = pd.to_datetime(hist["interaction_time"], errors="coerce")
    elif "timestamp" in hist.columns:
        hist["interaction_time"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    elif "timestamp_unix" in hist.columns:
        hist["interaction_time"] = pd.to_datetime(
            pd.to_numeric(hist["timestamp_unix"], errors="coerce"),
            unit="s",
            errors="coerce"
        )
    elif "my_last_updated" in hist.columns:
        hist["interaction_time"] = pd.to_datetime(
            pd.to_numeric(hist["my_last_updated"], errors="coerce"),
            unit="s",
            errors="coerce"
        )
    elif "my_finish_date" in hist.columns:
        hist["interaction_time"] = pd.to_datetime(hist["my_finish_date"], errors="coerce")
    elif "my_start_date" in hist.columns:
        hist["interaction_time"] = pd.to_datetime(hist["my_start_date"], errors="coerce")
    else:
        hist["interaction_time"] = pd.NaT

    if "interaction_time_raw" not in hist.columns:
        if "timestamp" in hist.columns:
            hist["interaction_time_raw"] = hist["timestamp"].astype(str)
        elif "timestamp_unix" in hist.columns:
            hist["interaction_time_raw"] = hist["timestamp_unix"].astype(str)
        else:
            hist["interaction_time_raw"] = ""

    hist["_time_sort"] = hist["interaction_time"].fillna(pd.Timestamp("1900-01-01"))

    row_order_col = choose_col(hist, ["_row_order", "row_order"], required=False)
    if row_order_col is None:
        hist["_row_order"] = range(len(hist))
        row_order_col = "_row_order"

    hist = hist.sort_values(["inner_user_id", "_time_sort", row_order_col, "inner_item_id"]).copy()
    hist["history_rank_in_train"] = hist.groupby("inner_user_id").cumcount() + 1
    hist["history_len_train"] = hist.groupby("inner_user_id")["inner_item_id"].transform("size")
    hist["reverse_rank_before_valid"] = hist["history_len_train"] - hist["history_rank_in_train"] + 1

    out_cols = [
        "inner_user_id", "raw_user_id", "username", "age", "is_minor",
        "inner_item_id", "raw_item_id", "title", "rating", "rating_bucket",
        "rating_code", "r17_code", "rplus_code", "rx_code",
        "genre", "type", "source", "anime_score",
        "score", "interaction_time_raw", "interaction_time",
        "history_rank_in_train", "history_len_train", "reverse_rank_before_valid",
    ]
    out_cols = [c for c in out_cols if c in hist.columns]

    out_path = out_dir / f"{dataset}.train_profile_history.csv"
    hist[out_cols].to_csv(out_path, index=False, encoding="utf-8-sig")

    print("[write]", out_path)
    print("rows =", len(hist))
    print("users =", hist["inner_user_id"].nunique())
    print("raw_item_id missing =", int((pd.to_numeric(hist["raw_item_id"], errors="coerce").fillna(-1) < 0).sum()))
    print("username empty =", int((hist["username"].fillna("").astype(str).str.len() == 0).sum()))
    print("history length describe:")
    print(hist.groupby("inner_user_id").size().describe())
    last5 = hist[hist["reverse_rank_before_valid"] <= 5]
    print("last5 rows =", len(last5))
    print("last5 users =", last5["inner_user_id"].nunique())


if __name__ == "__main__":
    main()
