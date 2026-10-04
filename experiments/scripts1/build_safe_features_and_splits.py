# scripts/build_safe_features_and_splits.py
# 生成safe文件
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np


RISK_DIMS = [
    "sex_code",
    "violence_code",
    "profanity_code",
    "drug_code",
    "intense_code",
]

MINOR_CAP = {
    "isAdult": 0,
    "sex_code": 2,
    "drug_code": 2,
    "violence_code": 3,
    "intense_code": 3,
    "profanity_code": 3,
}

AGE_DESC = {
    "1": "under 18",
    "18": "18-24",
    "25": "25-34",
    "35": "35-44",
    "45": "45-49",
    "50": "50-55",
    "56": "56+",
}


def to_int(x, default=0):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def to_float(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def read_rating(path, split):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) < 4:
                continue
            rows.append({
                "inner_user_id": int(arr[0]),
                "inner_item_id": int(arr[1]),
                "rating": float(arr[2]),
                "timestamp": int(arr[3]),
                "split": split,
            })
    return rows


def read_user_map(path):
    users = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = int(row["inner_user_id"])
            age = str(row.get("age", ""))
            users[u] = {
                "inner_user_id": u,
                "raw_user_id": row.get("raw_user_id", ""),
                "gender": row.get("gender", ""),
                "age": age,
                "age_desc": AGE_DESC.get(age, age),
                "is_minor": age == "1",
                "occupation": row.get("occupation", ""),
                "zip": row.get("zip", ""),
            }
    return users


def read_item_map(path):
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i = int(row["inner_item_id"])
            items[i] = {
                "inner_item_id": i,
                "raw_movie_id": int(row["raw_movie_id"]),
                "title": row.get("title", ""),
                "genres": row.get("genres", ""),
            }
    return items


def read_links(path):
    movie_to_tconst = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = row.get("movieId", "")
            imdb_id = row.get("imdbId", "")
            if not movie_id or not imdb_id:
                continue

            movie_id = int(movie_id)
            imdb_id = str(imdb_id).strip()

            if imdb_id.startswith("tt"):
                tconst = imdb_id
            else:
                tconst = "tt" + imdb_id.zfill(7)

            movie_to_tconst[movie_id] = tconst

    return movie_to_tconst


def read_movie_details(path):
    """
    movie_details_en.json:
    {
      "1": {
        "title": "Toy Story",
        "year": "1995",
        "overview": "...",
        "adult": false
      }
    }
    """
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for raw_mid, info in data.items():
        try:
            raw_mid_int = int(raw_mid)
        except Exception:
            continue

        out[raw_mid_int] = {
            "overview": info.get("overview", ""),
            "detail_title": info.get("title", ""),
            "detail_year": info.get("year", ""),
            "detail_adult": bool(info.get("adult", False)),
        }

    return out


def read_imdb_parental(path):
    """
    读取 IMDB_parental_guide.csv。
    需要字段：
    tconst,isAdult,sex_code,violence_code,profanity_code,drug_code,intense_code
    """
    risks = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tconst = row.get("tconst", "")
            if not tconst:
                continue

            risks[tconst] = {
                "tconst": tconst,
                "imdb_title": row.get("primaryTitle", ""),
                "imdb_original_title": row.get("originalTitle", ""),
                "titleType": row.get("titleType", ""),
                "isAdult_imdb": to_int(row.get("isAdult", 0), 0),
                "startYear": row.get("startYear", ""),
                "runtimeMinutes": row.get("runtimeMinutes", ""),
                "imdb_genres": row.get("genres", ""),
                "averageRating": row.get("averageRating", ""),
                "numVotes": row.get("numVotes", ""),

                "sex": row.get("sex", ""),
                "violence": row.get("violence", ""),
                "profanity": row.get("profanity", ""),
                "drugs": row.get("drugs", ""),
                "intense": row.get("intense", ""),

                "sex_code": to_int(row.get("sex_code", 1), 1),
                "violence_code": to_int(row.get("violence_code", 1), 1),
                "profanity_code": to_int(row.get("profanity_code", 1), 1),
                "drug_code": to_int(row.get("drug_code", 1), 1),
                "intense_code": to_int(row.get("intense_code", 1), 1),

                "mpaa": row.get("mpaa", ""),
                "certificate": row.get("certificate", ""),
            }

    return risks


def attach_item_features(items, movie_to_tconst, imdb_risk, movie_details):
    for iid, item in items.items():
        raw_mid = item["raw_movie_id"]
        tconst = movie_to_tconst.get(raw_mid, "")

        detail = movie_details.get(raw_mid, {})
        risk = imdb_risk.get(tconst, None)

        if risk is None:
            risk = {
                "tconst": tconst,
                "isAdult_imdb": 0,
                "sex_code": 1,
                "violence_code": 1,
                "profanity_code": 1,
                "drug_code": 1,
                "intense_code": 1,
                "sex": "",
                "violence": "",
                "profanity": "",
                "drugs": "",
                "intense": "",
                "mpaa": "",
                "certificate": "",
                "risk_missing": True,
            }
        else:
            risk = dict(risk)
            risk["risk_missing"] = False

        detail_adult = bool(detail.get("detail_adult", False))
        is_adult_final = 1 if (risk.get("isAdult_imdb", 0) == 1 or detail_adult) else 0

        item.update(risk)
        item.update(detail)
        item["isAdult"] = is_adult_final

        values = [to_int(item.get(d, 1), 1) for d in RISK_DIMS]
        item["risk_max"] = max(values)
        item["risk_mean"] = round(sum(values) / len(values), 4)
        item["risk_sum"] = sum(values)

    return items


def discrete_p75(values, default=1):
    """
    风险是 1-4 的离散等级。
    用线性 p75 再 ceil 成整数。
    """
    if not values:
        return default
    q = np.percentile(values, 75)
    return int(math.ceil(float(q)))


def compute_population_default_tolerance(train_rows, users, items):
    minor_values = {d: [] for d in RISK_DIMS}
    adult_values = {d: [] for d in RISK_DIMS}
    minor_adult = []
    adult_adult = []

    for row in train_rows:
        u = row["inner_user_id"]
        i = row["inner_item_id"]
        if i not in items:
            continue
        group = minor_values if users[u]["is_minor"] else adult_values
        for d in RISK_DIMS:
            group[d].append(to_int(items[i].get(d, 1), 1))

        if users[u]["is_minor"]:
            minor_adult.append(to_int(items[i].get("isAdult", 0), 0))
        else:
            adult_adult.append(to_int(items[i].get("isAdult", 0), 0))

    defaults = {
        "minor": {},
        "adult": {},
    }

    for d in RISK_DIMS:
        defaults["minor"][d] = discrete_p75(minor_values[d], default=1)
        defaults["adult"][d] = discrete_p75(adult_values[d], default=1)

    defaults["minor"]["isAdult"] = discrete_p75(minor_adult, default=0)
    defaults["adult"]["isAdult"] = discrete_p75(adult_adult, default=0)

    return defaults


def compute_user_tolerance(train_rows, users, items, min_hist=5):
    """
    只用 train 计算用户历史风险接受度。
    valid/test 绝不参与。
    """
    user_dim_values = defaultdict(lambda: {d: [] for d in RISK_DIMS})
    user_adult_values = defaultdict(list)

    for row in train_rows:
        u = row["inner_user_id"]
        i = row["inner_item_id"]
        if i not in items:
            continue

        for d in RISK_DIMS:
            user_dim_values[u][d].append(to_int(items[i].get(d, 1), 1))
        user_adult_values[u].append(to_int(items[i].get("isAdult", 0), 0))

    defaults = compute_population_default_tolerance(train_rows, users, items)

    user_tol = {}

    for u, info in users.items():
        is_minor = info["is_minor"]
        group_key = "minor" if is_minor else "adult"

        total_hist = len(user_adult_values[u])
        tol_raw = {}
        tol_final = {}

        for d in RISK_DIMS:
            vals = user_dim_values[u][d]
            if len(vals) >= min_hist:
                raw = discrete_p75(vals, default=defaults[group_key][d])
            else:
                raw = defaults[group_key][d]

            tol_raw[d] = raw

            if is_minor:
                tol_final[d] = min(raw, MINOR_CAP[d])
            else:
                tol_final[d] = raw

        if len(user_adult_values[u]) >= min_hist:
            adult_raw = discrete_p75(user_adult_values[u], default=defaults[group_key]["isAdult"])
        else:
            adult_raw = defaults[group_key]["isAdult"]

        tol_raw["isAdult"] = adult_raw

        if is_minor:
            tol_final["isAdult"] = min(adult_raw, MINOR_CAP["isAdult"])
        else:
            tol_final["isAdult"] = adult_raw

        user_tol[u] = {
            **info,
            "train_history_count": total_hist,
            "used_population_default": total_hist < min_hist,

            "tol_raw_isAdult": tol_raw["isAdult"],
            "tol_isAdult": tol_final["isAdult"],

            "tol_raw_sex_code": tol_raw["sex_code"],
            "tol_raw_violence_code": tol_raw["violence_code"],
            "tol_raw_profanity_code": tol_raw["profanity_code"],
            "tol_raw_drug_code": tol_raw["drug_code"],
            "tol_raw_intense_code": tol_raw["intense_code"],

            "tol_sex_code": tol_final["sex_code"],
            "tol_violence_code": tol_final["violence_code"],
            "tol_profanity_code": tol_final["profanity_code"],
            "tol_drug_code": tol_final["drug_code"],
            "tol_intense_code": tol_final["intense_code"],

            "minor_cap_sex_code": MINOR_CAP["sex_code"] if is_minor else "",
            "minor_cap_violence_code": MINOR_CAP["violence_code"] if is_minor else "",
            "minor_cap_profanity_code": MINOR_CAP["profanity_code"] if is_minor else "",
            "minor_cap_drug_code": MINOR_CAP["drug_code"] if is_minor else "",
            "minor_cap_intense_code": MINOR_CAP["intense_code"] if is_minor else "",
        }

    return user_tol, defaults


def is_item_within_user_tolerance(item, tol):
    if to_int(item.get("isAdult", 0), 0) > to_int(tol.get("tol_isAdult", 0), 0):
        return False

    for d in RISK_DIMS:
        if to_int(item.get(d, 1), 1) > to_int(tol.get(f"tol_{d}", 1), 1):
            return False

    return True


def enrich_split(rows, users, items, user_tol):
    out = []
    for row in rows:
        u = row["inner_user_id"]
        i = row["inner_item_id"]

        user = users[u]
        item = items[i]
        tol = user_tol[u]

        within_tol = is_item_within_user_tolerance(item, tol)

        new_row = {
            "split": row["split"],

            "inner_user_id": u,
            "raw_user_id": user.get("raw_user_id", ""),
            "gender": user.get("gender", ""),
            "age": user.get("age", ""),
            "age_desc": user.get("age_desc", ""),
            "is_minor": user.get("is_minor", False),

            "inner_item_id": i,
            "raw_movie_id": item.get("raw_movie_id", ""),
            "tconst": item.get("tconst", ""),
            "title": item.get("title", ""),
            "genres": item.get("genres", ""),
            "overview": item.get("overview", ""),

            "rating": row["rating"],
            "timestamp": row["timestamp"],

            "isAdult": item.get("isAdult", 0),
            "sex_code": item.get("sex_code", 1),
            "violence_code": item.get("violence_code", 1),
            "profanity_code": item.get("profanity_code", 1),
            "drug_code": item.get("drug_code", 1),
            "intense_code": item.get("intense_code", 1),
            "risk_max": item.get("risk_max", 1),
            "risk_mean": item.get("risk_mean", 1),
            "risk_missing": item.get("risk_missing", False),

            "tol_isAdult": tol.get("tol_isAdult", 0),
            "tol_sex_code": tol.get("tol_sex_code", 1),
            "tol_violence_code": tol.get("tol_violence_code", 1),
            "tol_profanity_code": tol.get("tol_profanity_code", 1),
            "tol_drug_code": tol.get("tol_drug_code", 1),
            "tol_intense_code": tol.get("tol_intense_code", 1),

            "within_user_tolerance": int(within_tol),
            "safe_observed_label": 1 if within_tol else 0,
        }
        out.append(new_row)

    out = sorted(out, key=lambda x: (x["inner_user_id"], x["timestamp"], x["inner_item_id"]))
    return out


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_item_safe(path, items):
    rows = []
    for i in sorted(items.keys()):
        item = items[i]
        rows.append({
            "inner_item_id": i,
            "raw_movie_id": item.get("raw_movie_id", ""),
            "tconst": item.get("tconst", ""),
            "title": item.get("title", ""),
            "genres": item.get("genres", ""),
            "overview": item.get("overview", ""),

            "isAdult": item.get("isAdult", 0),
            "isAdult_imdb": item.get("isAdult_imdb", 0),
            "sex": item.get("sex", ""),
            "violence": item.get("violence", ""),
            "profanity": item.get("profanity", ""),
            "drugs": item.get("drugs", ""),
            "intense": item.get("intense", ""),

            "sex_code": item.get("sex_code", 1),
            "violence_code": item.get("violence_code", 1),
            "profanity_code": item.get("profanity_code", 1),
            "drug_code": item.get("drug_code", 1),
            "intense_code": item.get("intense_code", 1),

            "risk_max": item.get("risk_max", 1),
            "risk_mean": item.get("risk_mean", 1),
            "risk_sum": item.get("risk_sum", 5),
            "mpaa": item.get("mpaa", ""),
            "certificate": item.get("certificate", ""),
            "risk_missing": item.get("risk_missing", False),
        })

    write_csv(path, rows)


def write_user_tol(path, user_tol):
    rows = [user_tol[u] for u in sorted(user_tol.keys())]
    write_csv(path, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Data")
    parser.add_argument("--prefix", default="ml-1m_safe")
    parser.add_argument("--ml1m_dir", default="../ml-1m")
    parser.add_argument("--imdb_file", default="../ml-1m/IMDB_parental_guide.csv")
    parser.add_argument("--movie_details", default="../ml-1m/movie_details_en.json")
    parser.add_argument("--out_prefix", default="ml-1m_safe_features")
    parser.add_argument("--out_dir", default="outputs/psg/safe_features")
    parser.add_argument("--min_hist", type=int, default=5)
    args = parser.parse_args()

    train = read_rating(f"{args.data_dir}/{args.prefix}.train.rating", "train")
    valid = read_rating(f"{args.data_dir}/{args.prefix}.valid.rating", "valid")
    test = read_rating(f"{args.data_dir}/{args.prefix}.test.rating", "test")

    users = read_user_map(f"{args.data_dir}/{args.prefix}.user_map.csv")
    items = read_item_map(f"{args.data_dir}/{args.prefix}.item_map.csv")

    links = read_links(os.path.join(args.ml1m_dir, "links.csv"))
    movie_details = read_movie_details(args.movie_details)
    imdb_risk = read_imdb_parental(args.imdb_file)

    items = attach_item_features(items, links, imdb_risk, movie_details)

    user_tol, defaults = compute_user_tolerance(
        train_rows=train,
        users=users,
        items=items,
        min_hist=args.min_hist,
    )

    train_safe = enrich_split(train, users, items, user_tol)
    valid_safe = enrich_split(valid, users, items, user_tol)
    test_safe = enrich_split(test, users, items, user_tol)

    os.makedirs(args.out_dir, exist_ok=True)

    write_item_safe(f"{args.out_dir}/{args.out_prefix}.item_safe.csv", items)
    write_user_tol(f"{args.out_dir}/{args.out_prefix}.user_tolerance_p75.csv", user_tol)

    write_csv(f"{args.out_dir}/{args.out_prefix}.train.safe.csv", train_safe)
    write_csv(f"{args.out_dir}/{args.out_prefix}.valid.safe.csv", valid_safe)
    write_csv(f"{args.out_dir}/{args.out_prefix}.test.safe.csv", test_safe)

    with open(f"{args.out_dir}/{args.out_prefix}.population_defaults.json", "w", encoding="utf-8") as f:
        json.dump(defaults, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(f"  {args.out_dir}/{args.out_prefix}.item_safe.csv")
    print(f"  {args.out_dir}/{args.out_prefix}.user_tolerance_p75.csv")
    print(f"  {args.out_dir}/{args.out_prefix}.train.safe.csv")
    print(f"  {args.out_dir}/{args.out_prefix}.valid.safe.csv")
    print(f"  {args.out_dir}/{args.out_prefix}.test.safe.csv")
    print(f"  {args.out_dir}/{args.out_prefix}.population_defaults.json")
    print()
    print("Rows:")
    print("  train:", len(train_safe))
    print("  valid:", len(valid_safe))
    print("  test :", len(test_safe))

    # 简单统计
    for name, rows in [("train", train_safe), ("valid", valid_safe), ("test", test_safe)]:
        n = len(rows)
        unsafe = sum(1 for r in rows if int(r["within_user_tolerance"]) == 0)
        minor_n = sum(1 for r in rows if str(r["is_minor"]) == "True")
        minor_unsafe = sum(1 for r in rows if str(r["is_minor"]) == "True" and int(r["within_user_tolerance"]) == 0)
        print(f"{name}: unsafe_by_user_tol={unsafe}/{n}, minor_unsafe={minor_unsafe}/{minor_n}")


if __name__ == "__main__":
    main()
