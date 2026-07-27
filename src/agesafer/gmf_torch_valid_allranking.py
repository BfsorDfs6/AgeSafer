#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train GMF and select the checkpoint using validation full ranking.

This file is part of the lightweight AgeSafer reference implementation
for GMF on ML-1M. Paths and execution are controlled by run_ml1m_gmf.py.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Set

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def safe_int(x, default=-1):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return default


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


def read_rating(path: str) -> Tuple[List[Tuple[int, int]], Dict[int, Set[int]], int, int]:
    rows: List[Tuple[int, int]] = []
    user_items: Dict[int, Set[int]] = defaultdict(set)
    max_user, max_item = -1, -1

    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Rating file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            arr = split_line(line)
            if len(arr) < 2:
                continue
            u = safe_int(arr[0])
            i = safe_int(arr[1])
            if u < 0 or i < 0:
                continue
            rows.append((u, i))
            user_items[u].add(i)
            max_user = max(max_user, u)
            max_item = max(max_item, i)

    return rows, user_items, max_user + 1, max_item + 1


def read_eval_rating(path: str) -> Tuple[List[Tuple[int, int]], int, int]:
    rows, _, nu, ni = read_rating(path)
    return rows, nu, ni


read_train_rating = read_rating
read_test_rating = read_eval_rating


def read_test_negative(path: str) -> Tuple[List[List[int]], int, int]:
    negatives: List[List[int]] = []
    max_user, max_item = -1, -1
    if not path or not os.path.exists(path):
        return negatives, 0, 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            nums = [safe_int(x) for x in re.findall(r"-?\d+", s)]
            nums = [x for x in nums if x >= 0]
            if len(nums) < 2:
                continue

            first_token = s.split("\t", 1)[0].strip()
            if first_token.startswith("(") and len(nums) >= 3:
                u = nums[0]
                pos_i = nums[1]
                negs = nums[2:]
                max_item = max(max_item, pos_i)
            else:
                u = nums[0]
                negs = nums[1:]

            negatives.append(negs)
            max_user = max(max_user, u)
            if negs:
                max_item = max(max_item, max(negs))

    return negatives, max_user + 1, max_item + 1


def parse_topks(s: str) -> List[int]:
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        k = int(x)
        if k > 0:
            vals.append(k)
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("--valid_topks cannot be empty")
    return vals


class GMFDataset(Dataset):
    def __init__(self, positives, user_pos, num_items, num_neg=4, seed=42):
        self.positives = list(positives)
        self.user_pos = user_pos
        self.num_items = int(num_items)
        self.num_neg = int(num_neg)
        self.seed = int(seed)
        self.instances: List[Tuple[int, int, float]] = []

    def negative_sampling(self, epoch=0):
        rng = random.Random(self.seed + int(epoch))
        self.instances = []
        for u, i in self.positives:
            self.instances.append((u, i, 1.0))
            for _ in range(self.num_neg):
                j = rng.randrange(self.num_items)
                while j in self.user_pos[u]:
                    j = rng.randrange(self.num_items)
                self.instances.append((u, j, 0.0))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        u, i, y = self.instances[idx]
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )


class GMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors=8):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)
        self.output = nn.Linear(num_factors, 1)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, users, items):
        p_u = self.user_embedding(users)
        q_i = self.item_embedding(items)
        x = p_u * q_i
        return self.output(x).squeeze(-1)


@torch.no_grad()
def score_items(model, u: int, items: Sequence[int], device, batch_size=8192) -> np.ndarray:
    model.eval()
    scores_all: List[float] = []
    for start in range(0, len(items), batch_size):
        end = min(start + batch_size, len(items))
        users_t = torch.full((end - start,), int(u), dtype=torch.long, device=device)
        items_t = torch.tensor(items[start:end], dtype=torch.long, device=device)
        logits = model(users_t, items_t)
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        scores_all.extend(scores.tolist())
    return np.asarray(scores_all, dtype=np.float32)


@torch.no_grad()
def evaluate_sampled(model, eval_ratings, eval_negatives, device, topks=(10,), batch_size=8192):
    model.eval()
    topks = sorted(set(int(k) for k in topks if int(k) > 0))
    max_topk = max(topks)
    hits = {k: [] for k in topks}
    ndcgs = {k: [] for k in topks}

    if len(eval_ratings) != len(eval_negatives):
        print(f"[warn] rating rows={len(eval_ratings)} negative rows={len(eval_negatives)}; using min length")
    n = min(len(eval_ratings), len(eval_negatives))

    for idx in range(n):
        u, gt_item = eval_ratings[idx]
        items = [gt_item] + list(eval_negatives[idx])
        scores = score_items(model, u, items, device, batch_size=batch_size)
        order = np.argsort(-scores)[:max_topk]
        ranklist = [items[int(t)] for t in order]

        for k in topks:
            cur = ranklist[:k]
            if gt_item in cur:
                rank = cur.index(gt_item)
                hits[k].append(1.0)
                ndcgs[k].append(math.log(2.0) / math.log(rank + 2.0))
            else:
                hits[k].append(0.0)
                ndcgs[k].append(0.0)

    return {
        k: {
            "HR": float(np.mean(hits[k])) if hits[k] else 0.0,
            "NDCG": float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0,
        }
        for k in topks
    }


@torch.no_grad()
def evaluate_allranking(model, eval_ratings, train_user_pos, num_items: int, device, topks=(10,), batch_size=8192):
    model.eval()
    topks = sorted(set(int(k) for k in topks if int(k) > 0))
    max_topk = max(topks)
    hits = {k: [] for k in topks}
    ndcgs = {k: [] for k in topks}
    all_items = np.arange(int(num_items), dtype=np.int64)

    for u, gt_item in eval_ratings:
        if gt_item < 0 or gt_item >= num_items:
            for k in topks:
                hits[k].append(0.0)
                ndcgs[k].append(0.0)
            continue

        blocked = train_user_pos.get(u, set())
        if blocked:
            mask = np.ones(num_items, dtype=bool)
            for b in blocked:
                if 0 <= b < num_items and b != gt_item:
                    mask[b] = False
            items = all_items[mask]
        else:
            items = all_items

        scores = score_items(model, int(u), items.tolist(), device, batch_size=batch_size)
        if len(items) <= max_topk:
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, max_topk - 1)[:max_topk]
            order = part[np.argsort(-scores[part])]

        ranklist = [int(items[int(t)]) for t in order[:max_topk]]

        for k in topks:
            cur = ranklist[:k]
            if gt_item in cur:
                rank = cur.index(gt_item)
                hits[k].append(1.0)
                ndcgs[k].append(math.log(2.0) / math.log(rank + 2.0))
            else:
                hits[k].append(0.0)
                ndcgs[k].append(0.0)

    return {
        k: {
            "HR": float(np.mean(hits[k])) if hits[k] else 0.0,
            "NDCG": float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0,
        }
        for k in topks
    }


def monitor_value(metrics, monitor: str, topk: int) -> float:
    m = metrics[int(topk)]
    key = monitor.upper()
    if key == "HR":
        return float(m["HR"])
    if key == "NDCG":
        return float(m["NDCG"])
    if key == "SUM":
        return float(m["HR"] + m["NDCG"])
    raise ValueError(f"Unknown monitor: {monitor}")


def should_validate_epoch(epoch: int, epochs: int, valid_every: int) -> bool:
    valid_every = max(1, int(valid_every))
    return (epoch % valid_every == 0) or (epoch == epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Data")
    parser.add_argument("--dataset", default="ml-1m_safe")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=8192)
    parser.add_argument("--num_factors", type=int, default=8)
    parser.add_argument("--num_neg", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--valid_topks", default="1,5,10,20")
    parser.add_argument("--monitor", choices=["hr", "ndcg", "sum"], default="ndcg")
    parser.add_argument("--valid_protocol", choices=["allranking", "sampled"], default="allranking")
    parser.add_argument("--valid_every", type=int, default=1,
                        help="Run validation every N epochs. The final epoch is always validated.")
    parser.add_argument("--eval_block_train_rating", default="",
                        help="Optional train file used only to block candidates during valid all-ranking. Use original train here when training with pseudo samples but pseudo items should remain recommendable.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=int, default=1)
    parser.add_argument("--save_dir", default="Pretrain_torch")
    parser.add_argument("--num_items_override", type=int, default=0)
    parser.add_argument("--num_users_override", type=int, default=0)
    args = parser.parse_args()

    args.valid_every = max(1, int(args.valid_every))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_path = os.path.join(args.data_dir, f"{args.dataset}.train.rating")
    valid_path = os.path.join(args.data_dir, f"{args.dataset}.valid.rating")
    valid_neg_path = os.path.join(args.data_dir, f"{args.dataset}.valid.negative")

    train_pos, train_user_pos, n_user1, n_item1 = read_rating(train_path)

    eval_block_user_pos = train_user_pos
    if args.eval_block_train_rating:
        _, eval_block_user_pos, _, _ = read_rating(args.eval_block_train_rating)
        print(f"[valid-block] using eval_block_train_rating={args.eval_block_train_rating}")

    valid_ratings, n_user2, n_item2 = read_eval_rating(valid_path)

    n_user3 = n_item3 = 0
    valid_negs: List[List[int]] = []
    if args.valid_protocol == "sampled":
        valid_negs, n_user3, n_item3 = read_test_negative(valid_neg_path)

    num_users = max(n_user1, n_user2, n_user3, args.num_users_override)
    num_items = max(n_item1, n_item2, n_item3, args.num_items_override)

    valid_topks = parse_topks(args.valid_topks)
    if int(args.topk) not in valid_topks:
        valid_topks = sorted(set(valid_topks + [int(args.topk)]))

    print(f"Users: {num_users}, Items: {num_items}")
    print(f"Train positives: {len(train_pos)}")
    print(f"Valid positives: {len(valid_ratings)}")
    print(f"Valid protocol: {args.valid_protocol}")
    print(f"Valid every: {args.valid_every} epoch(s)")
    if args.valid_protocol == "sampled":
        print(f"Valid negatives: {len(valid_negs)}")
    print(f"Monitor: {args.monitor.upper()}@{args.topk}")
    print(f"Valid topks: {valid_topks}")
    print("Test split is NOT read by this script.")

    model = GMF(num_users, num_items, args.num_factors).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataset = GMFDataset(
        positives=train_pos,
        user_pos=train_user_pos,
        num_items=num_items,
        num_neg=args.num_neg,
        seed=args.seed,
    )

    best_score = -1e18
    best_epoch = -1
    best_metrics = None
    log_rows = []
    save_path = os.path.join(args.save_dir, f"{args.dataset}_GMF_{args.num_factors}_torch_best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        dataset.negative_sampling(epoch)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        model.train()
        total_loss = 0.0
        total_count = 0

        for users, items, labels in loader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(users, items)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)

        avg_loss = total_loss / max(total_count, 1)
        elapsed_train = time.time() - t0

        do_valid = should_validate_epoch(epoch, args.epochs, args.valid_every)

        row = {
            "epoch": epoch,
            "loss": avg_loss,
            "valid_ran": int(do_valid),
            "monitor": "",
            "monitor_name": args.monitor,
            "monitor_topk": args.topk,
            "valid_protocol": args.valid_protocol,
            "time_sec": elapsed_train,
        }

        if not do_valid:
            print(
                f"Epoch {epoch:03d}: loss={avg_loss:.4f} | "
                f"valid=skipped(valid_every={args.valid_every}) | "
                f"time={elapsed_train:.1f}s",
                flush=True,
            )
            log_rows.append(row)
            continue

        t_valid = time.time()
        if args.valid_protocol == "allranking":
            metrics = evaluate_allranking(
                model=model,
                eval_ratings=valid_ratings,
                train_user_pos=eval_block_user_pos,
                num_items=num_items,
                device=device,
                topks=valid_topks,
                batch_size=args.eval_batch_size,
            )
        else:
            metrics = evaluate_sampled(
                model=model,
                eval_ratings=valid_ratings,
                eval_negatives=valid_negs,
                device=device,
                topks=valid_topks,
                batch_size=args.eval_batch_size,
            )

        cur_score = monitor_value(metrics, args.monitor, args.topk)
        elapsed_total = time.time() - t0
        elapsed_valid = time.time() - t_valid

        msg = [f"Epoch {epoch:03d}: loss={avg_loss:.4f}"]
        for k in sorted(metrics):
            msg.append(f"Valid HR@{k}={metrics[k]['HR']:.4f} NDCG@{k}={metrics[k]['NDCG']:.4f}")
        msg.append(f"monitor={cur_score:.6f} train={elapsed_train:.1f}s valid={elapsed_valid:.1f}s total={elapsed_total:.1f}s")
        print(" | ".join(msg), flush=True)

        row["monitor"] = cur_score
        row["time_sec"] = elapsed_total
        row["valid_time_sec"] = elapsed_valid
        for k, v in metrics.items():
            row[f"HR@{k}"] = v["HR"]
            row[f"NDCG@{k}"] = v["NDCG"]
        log_rows.append(row)

        if cur_score > best_score:
            best_score = cur_score
            best_epoch = epoch
            best_metrics = metrics
            if args.out:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "num_users": num_users,
                        "num_items": num_items,
                        "num_factors": args.num_factors,
                        "epoch": epoch,
                        "valid_protocol": args.valid_protocol,
                        "valid_every": args.valid_every,
                        "valid_metrics": metrics,
                        "hr": metrics[int(args.topk)]["HR"],
                        "ndcg": metrics[int(args.topk)]["NDCG"],
                        "monitor": args.monitor,
                        "monitor_topk": args.topk,
                        "monitor_value": cur_score,
                        "args": vars(args),
                    },
                    save_path,
                )
                print(f"[save-best-valid-{args.valid_protocol}] {save_path}")

    log_path = os.path.join(args.save_dir, "training_log.csv")
    if log_rows:
        fieldnames = sorted({k for r in log_rows for k in r.keys()})
        with open(log_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)

    summary = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "save_dir": args.save_dir,
        "best_epoch": best_epoch,
        "best_monitor_name": args.monitor,
        "best_monitor_topk": args.topk,
        "best_monitor_value": best_score,
        "valid_protocol": args.valid_protocol,
        "valid_every": args.valid_every,
        "best_valid_metrics": best_metrics,
        "checkpoint": save_path,
        "test_split_used_for_checkpoint_selection": False,
        "args": vars(args),
    }
    with open(os.path.join(args.save_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Best epoch={best_epoch}, monitor={best_score:.6f}, protocol={args.valid_protocol}, valid_every={args.valid_every}")
    if best_metrics:
        print(json.dumps(best_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
