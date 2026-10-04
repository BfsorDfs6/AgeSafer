# scripts/build_ml1m_safe_split.py
# 划分训练、验证、测试集
import argparse
import csv
import random
from collections import defaultdict


def read_movies(path):
    movies = {}
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            arr = line.rstrip("\n").split("::")
            if len(arr) < 3:
                continue
            raw_mid = int(arr[0])
            title = arr[1]
            genres = arr[2]
            movies[raw_mid] = {
                "raw_movie_id": raw_mid,
                "title": title,
                "genres": genres,
            }
    return movies


def read_users(path):
    users = {}
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            arr = line.rstrip("\n").split("::")
            if len(arr) < 5:
                continue
            raw_uid = int(arr[0])
            users[raw_uid] = {
                "raw_user_id": raw_uid,
                "gender": arr[1],
                "age": arr[2],
                "occupation": arr[3],
                "zip": arr[4],
            }
    return users


def read_ratings(path):
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            arr = line.rstrip("\n").split("::")
            if len(arr) < 4:
                continue
            raw_uid = int(arr[0])
            raw_mid = int(arr[1])
            rating = int(arr[2])
            timestamp = int(arr[3])
            rows.append((raw_uid, raw_mid, rating, timestamp))
    return rows


def write_rating(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for u, i, r, t in rows:
            f.write(f"{u}\t{i}\t{r}\t{t}\n")


def write_negative(path, positives, user_all_items, num_items, num_neg=99, seed=42):
    rng = random.Random(seed)
    all_items = list(range(num_items))

    with open(path, "w", encoding="utf-8") as f:
        for u, pos_i, r, t in positives:
            interacted = user_all_items[u]
            negs = []
            while len(negs) < num_neg:
                j = rng.choice(all_items)
                if j not in interacted and j not in negs:
                    negs.append(j)

            f.write(f"({u},{pos_i})\t" + "\t".join(map(str, negs)) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Data")
    parser.add_argument("--out_prefix", default="ml-1m_safe")
    parser.add_argument("--num_neg", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratings_path = f"{args.data_dir}/ratings.dat"
    movies_path = f"{args.data_dir}/movies.dat"
    users_path = f"{args.data_dir}/users.dat"

    movies = read_movies(movies_path)
    users = read_users(users_path)
    ratings = read_ratings(ratings_path)

    # 只保留有评分记录里的 user/movie，重新映射成连续 ID
    raw_users = sorted({u for u, m, r, t in ratings})
    raw_items = sorted({m for u, m, r, t in ratings})

    user2inner = {u: idx for idx, u in enumerate(raw_users)}
    item2inner = {m: idx for idx, m in enumerate(raw_items)}

    inner2user = {v: k for k, v in user2inner.items()}
    inner2item = {v: k for k, v in item2inner.items()}

    mapped = []
    user_seq = defaultdict(list)
    user_all_items = defaultdict(set)

    for raw_u, raw_m, rating, ts in ratings:
        u = user2inner[raw_u]
        i = item2inner[raw_m]
        mapped.append((u, i, rating, ts, raw_u, raw_m))
        user_seq[u].append((u, i, rating, ts))
        user_all_items[u].add(i)

    train_rows, valid_rows, test_rows = [], [], []

    for u, seq in user_seq.items():
        seq = sorted(seq, key=lambda x: x[3])
        if len(seq) < 3:
            # ML-1M 一般每个用户都够，这里只是兜底
            continue

        train_rows.extend(seq[:-2])
        valid_rows.append(seq[-2])
        test_rows.append(seq[-1])

    num_users = len(raw_users)
    num_items = len(raw_items)

    # 保存 train / valid / test
    write_rating(f"{args.data_dir}/{args.out_prefix}.train.rating", train_rows)
    write_rating(f"{args.data_dir}/{args.out_prefix}.valid.rating", valid_rows)
    write_rating(f"{args.data_dir}/{args.out_prefix}.test.rating", test_rows)

    # 注意：negative 只从该用户从未交互过的 item 中采样，不会采到 train/valid/test 正样本
    write_negative(
        f"{args.data_dir}/{args.out_prefix}.valid.negative",
        valid_rows,
        user_all_items,
        num_items,
        num_neg=args.num_neg,
        seed=args.seed,
    )
    write_negative(
        f"{args.data_dir}/{args.out_prefix}.test.negative",
        test_rows,
        user_all_items,
        num_items,
        num_neg=args.num_neg,
        seed=args.seed + 1,
    )

    # 保存 user 映射
    with open(f"{args.data_dir}/{args.out_prefix}.user_map.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["inner_user_id", "raw_user_id", "gender", "age", "occupation", "zip"])
        for inner_u in range(num_users):
            raw_u = inner2user[inner_u]
            info = users.get(raw_u, {})
            writer.writerow([
                inner_u,
                raw_u,
                info.get("gender", ""),
                info.get("age", ""),
                info.get("occupation", ""),
                info.get("zip", ""),
            ])

    # 保存 item 映射
    with open(f"{args.data_dir}/{args.out_prefix}.item_map.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["inner_item_id", "raw_movie_id", "title", "genres"])
        for inner_i in range(num_items):
            raw_m = inner2item[inner_i]
            info = movies.get(raw_m, {})
            writer.writerow([
                inner_i,
                raw_m,
                info.get("title", ""),
                info.get("genres", ""),
            ])

    # 简单防泄露检查：同一用户 train/valid/test item 不重复
    train_by_user = defaultdict(set)
    for u, i, r, t in train_rows:
        train_by_user[u].add(i)

    for u, i, r, t in valid_rows:
        assert i not in train_by_user[u], f"valid leakage: user={u}, item={i}"

    for u, i, r, t in test_rows:
        assert i not in train_by_user[u], f"test leakage: user={u}, item={i}"

    print("Done.")
    print(f"users = {num_users}")
    print(f"items = {num_items}")
    print(f"train interactions = {len(train_rows)}")
    print(f"valid interactions = {len(valid_rows)}")
    print(f"test interactions  = {len(test_rows)}")
    print("Saved:")
    print(f"  {args.data_dir}/{args.out_prefix}.train.rating")
    print(f"  {args.data_dir}/{args.out_prefix}.valid.rating")
    print(f"  {args.data_dir}/{args.out_prefix}.valid.negative")
    print(f"  {args.data_dir}/{args.out_prefix}.test.rating")
    print(f"  {args.data_dir}/{args.out_prefix}.test.negative")
    print(f"  {args.data_dir}/{args.out_prefix}.user_map.csv")
    print(f"  {args.data_dir}/{args.out_prefix}.item_map.csv")


if __name__ == "__main__":
    main()
