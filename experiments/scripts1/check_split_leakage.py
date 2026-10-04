# scripts/check_split_leakage.py
# 检查数据泄露
import argparse
from collections import defaultdict

def read_rating(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) < 4:
                continue
            u, i, r, t = int(arr[0]), int(arr[1]), float(arr[2]), int(arr[3])
            rows.append((u, i, r, t))
    return rows

def read_negative(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            head = arr[0].replace("(", "").replace(")", "")
            u, pos = map(int, head.split(","))
            negs = list(map(int, arr[1:]))
            data[u] = (pos, negs)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Data")
    parser.add_argument("--prefix", default="ml-1m_safe")
    args = parser.parse_args()

    train = read_rating(f"{args.data_dir}/{args.prefix}.train.rating")
    valid = read_rating(f"{args.data_dir}/{args.prefix}.valid.rating")
    test = read_rating(f"{args.data_dir}/{args.prefix}.test.rating")
    valid_neg = read_negative(f"{args.data_dir}/{args.prefix}.valid.negative")
    test_neg = read_negative(f"{args.data_dir}/{args.prefix}.test.negative")

    train_items = defaultdict(set)
    train_ts = defaultdict(list)
    all_pos_items = defaultdict(set)

    for u, i, r, t in train:
        train_items[u].add(i)
        train_ts[u].append(t)
        all_pos_items[u].add(i)

    valid_pos = {}
    valid_ts = {}
    for u, i, r, t in valid:
        valid_pos[u] = i
        valid_ts[u] = t
        all_pos_items[u].add(i)

    test_pos = {}
    test_ts = {}
    for u, i, r, t in test:
        test_pos[u] = i
        test_ts[u] = t
        all_pos_items[u].add(i)

    n_user = len(test_pos)

    leak_train_valid = []
    leak_train_test = []
    leak_valid_test_same = []
    neg_leak_valid = []
    neg_leak_test = []
    time_order_bad = []
    same_ts_boundary = []

    for u in test_pos:
        if valid_pos[u] in train_items[u]:
            leak_train_valid.append(u)
        if test_pos[u] in train_items[u]:
            leak_train_test.append(u)
        if valid_pos[u] == test_pos[u]:
            leak_valid_test_same.append(u)

        # negative 是否包含该用户任意真实交互
        v_pos, v_negs = valid_neg[u]
        t_pos, t_negs = test_neg[u]

        if v_pos != valid_pos[u]:
            print("valid negative head mismatch:", u, v_pos, valid_pos[u])
        if t_pos != test_pos[u]:
            print("test negative head mismatch:", u, t_pos, test_pos[u])

        if any(x in all_pos_items[u] for x in v_negs):
            neg_leak_valid.append(u)
        if any(x in all_pos_items[u] for x in t_negs):
            neg_leak_test.append(u)

        max_train_ts = max(train_ts[u]) if train_ts[u] else -1
        if not (max_train_ts <= valid_ts[u] <= test_ts[u]):
            time_order_bad.append((u, max_train_ts, valid_ts[u], test_ts[u]))

        # 不是硬泄露，只是看看边界同时间戳情况
        if max_train_ts == valid_ts[u] or max_train_ts == test_ts[u] or valid_ts[u] == test_ts[u]:
            same_ts_boundary.append((u, max_train_ts, valid_ts[u], test_ts[u]))

    print("users:", n_user)
    print("train rows:", len(train))
    print("valid rows:", len(valid))
    print("test rows:", len(test))
    print()
    print("train-valid positive leakage:", len(leak_train_valid))
    print("train-test positive leakage :", len(leak_train_test))
    print("valid-test same positive    :", len(leak_valid_test_same))
    print("valid negative leakage      :", len(neg_leak_valid))
    print("test negative leakage       :", len(neg_leak_test))
    print("time order bad              :", len(time_order_bad))
    print("same timestamp boundary     :", len(same_ts_boundary))
    print()
    if same_ts_boundary:
        print("Example same timestamp boundary:", same_ts_boundary[:10])

if __name__ == "__main__":
    main()
