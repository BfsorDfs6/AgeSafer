#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lightgcn_torch_valid_allranking.py

Self-contained LightGCN training with the same valid-allranking checkpoint protocol.
Input uses the same *.train.rating / *.valid.rating files as GMF.
No KG triples are used; this is user-item bipartite graph CF.
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
from typing import Dict, List, Sequence, Set, Tuple

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
    rows = []
    user_items = defaultdict(set)
    max_user, max_item = -1, -1
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Rating file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            arr = split_line(line)
            if len(arr) < 2:
                continue
            u, i = safe_int(arr[0]), safe_int(arr[1])
            if u < 0 or i < 0:
                continue
            rows.append((u, i))
            user_items[u].add(i)
            max_user = max(max_user, u)
            max_item = max(max_item, i)
    return rows, dict(user_items), max_user + 1, max_item + 1


def read_eval_rating(path: str):
    rows, _, nu, ni = read_rating(path)
    return rows, nu, ni


def parse_topks(s: str) -> List[int]:
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    vals = sorted(set(k for k in vals if k > 0))
    if not vals:
        raise ValueError("--valid_topks cannot be empty")
    return vals


class BPRDataset(Dataset):
    def __init__(self, positives, user_pos, num_items, seed=42):
        self.positives = list(positives)
        self.user_pos = user_pos
        self.num_items = int(num_items)
        self.seed = int(seed)
        self.triples = []

    def negative_sampling(self, epoch=0):
        rng = random.Random(self.seed + int(epoch))
        self.triples = []
        for u, pos in self.positives:
            neg = rng.randrange(self.num_items)
            while neg in self.user_pos.get(u, set()):
                neg = rng.randrange(self.num_items)
            self.triples.append((u, pos, neg))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        u, p, n = self.triples[idx]
        return torch.tensor(u, dtype=torch.long), torch.tensor(p, dtype=torch.long), torch.tensor(n, dtype=torch.long)


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64, n_layers=3):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.emb_dim = int(emb_dim)
        self.n_layers = int(n_layers)
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def propagate(self, norm_adj):
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_emb]
        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(norm_adj, x)
            embs.append(x)
        out = torch.stack(embs, dim=0).mean(dim=0)
        users, items = torch.split(out, [self.num_users, self.num_items], dim=0)
        return users, items

    def bpr_loss(self, users, pos_items, neg_items, norm_adj, reg_weight=1e-4):
        user_g, item_g = self.propagate(norm_adj)
        u = user_g[users]
        p = item_g[pos_items]
        n = item_g[neg_items]
        pos_scores = (u * p).sum(dim=-1)
        neg_scores = (u * n).sum(dim=-1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        u0 = self.user_embedding(users)
        p0 = self.item_embedding(pos_items)
        n0 = self.item_embedding(neg_items)
        reg = (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / (2.0 * users.shape[0])
        return loss + reg_weight * reg, loss.detach(), reg.detach()


def build_norm_adj(num_users, num_items, train_pairs, device):
    n_nodes = num_users + num_items
    rows = []
    cols = []
    for u, i in train_pairs:
        rows.append(u)
        cols.append(num_users + i)
        rows.append(num_users + i)
        cols.append(u)
    if not rows:
        raise ValueError("No train edges for LightGCN")
    rows_t = torch.tensor(rows, dtype=torch.long)
    cols_t = torch.tensor(cols, dtype=torch.long)
    deg = torch.zeros(n_nodes, dtype=torch.float32)
    deg.index_add_(0, rows_t, torch.ones_like(rows_t, dtype=torch.float32))
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    vals = deg_inv_sqrt[rows_t] * deg_inv_sqrt[cols_t]
    idx = torch.stack([rows_t, cols_t], dim=0)
    adj = torch.sparse_coo_tensor(idx, vals, (n_nodes, n_nodes)).coalesce().to(device)
    return adj


@torch.no_grad()
def evaluate_allranking(model, norm_adj, eval_ratings, train_user_pos, num_items, device, topks=(10,), batch_size=8192):
    model.eval()
    topks = sorted(set(int(k) for k in topks if int(k) > 0))
    max_topk = max(topks)
    hits = {k: [] for k in topks}
    ndcgs = {k: [] for k in topks}
    user_g, item_g = model.propagate(norm_adj)
    all_items = np.arange(int(num_items), dtype=np.int64)

    for u, gt_item in eval_ratings:
        if u < 0 or u >= model.num_users or gt_item < 0 or gt_item >= num_items:
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
        scores_all = []
        uemb = user_g[u]
        for start in range(0, len(items), batch_size):
            batch = torch.tensor(items[start:start + batch_size], dtype=torch.long, device=device)
            scores = (item_g[batch] * uemb).sum(dim=-1)
            scores_all.extend(scores.detach().cpu().tolist())
        scores = np.asarray(scores_all, dtype=np.float32)
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
    return {k: {"HR": float(np.mean(hits[k])) if hits[k] else 0.0, "NDCG": float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0} for k in topks}


def monitor_value(metrics, monitor, topk):
    m = metrics[int(topk)]
    key = monitor.upper()
    if key == "HR":
        return float(m["HR"])
    if key == "NDCG":
        return float(m["NDCG"])
    if key == "SUM":
        return float(m["HR"] + m["NDCG"])
    raise ValueError(monitor)


def should_validate_epoch(epoch, epochs, valid_every):
    valid_every = max(1, int(valid_every))
    return (epoch % valid_every == 0) or (epoch == epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="Data")
    ap.add_argument("--dataset", default="ml-1m_safe")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--eval_batch_size", type=int, default=32768)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--decay", type=float, default=1e-4)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--valid_topks", default="1,5,10,20")
    ap.add_argument("--monitor", choices=["hr", "ndcg", "sum"], default="ndcg")
    ap.add_argument("--valid_every", type=int, default=10)
    ap.add_argument("--eval_block_train_rating", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", default="Pretrain_torch_lightgcn")
    ap.add_argument("--num_items_override", type=int, default=0)
    ap.add_argument("--num_users_override", type=int, default=0)
    args = ap.parse_args()

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
    train_pos, train_user_pos, n_user1, n_item1 = read_rating(train_path)
    eval_block_user_pos = train_user_pos
    if args.eval_block_train_rating:
        _, eval_block_user_pos, _, _ = read_rating(args.eval_block_train_rating)
        print(f"[valid-block] using eval_block_train_rating={args.eval_block_train_rating}")
    valid_ratings, n_user2, n_item2 = read_eval_rating(valid_path)
    num_users = max(n_user1, n_user2, args.num_users_override)
    num_items = max(n_item1, n_item2, args.num_items_override)
    valid_topks = parse_topks(args.valid_topks)
    if int(args.topk) not in valid_topks:
        valid_topks = sorted(set(valid_topks + [int(args.topk)]))
    print(f"Model: LightGCN")
    print(f"Users: {num_users}, Items: {num_items}, Train positives: {len(train_pos)}, Valid positives: {len(valid_ratings)}")
    print(f"emb_dim={args.emb_dim}, layers={args.layers}, monitor={args.monitor.upper()}@{args.topk}")
    print("Test split is NOT read by this script.")

    norm_adj = build_norm_adj(num_users, num_items, train_pos, device)
    model = LightGCN(num_users, num_items, emb_dim=args.emb_dim, n_layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataset = BPRDataset(train_pos, train_user_pos, num_items, seed=args.seed)
    best_score = -1e18
    best_epoch = -1
    best_metrics = None
    log_rows = []
    save_path = os.path.join(args.save_dir, f"{args.dataset}_LightGCN_torch_best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        dataset.negative_sampling(epoch)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        model.train()
        total_loss = 0.0
        total_bpr = 0.0
        total_reg = 0.0
        total_count = 0
        for users, pos, neg in loader:
            users, pos, neg = users.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()
            loss, bpr, reg = model.bpr_loss(users, pos, neg, norm_adj, reg_weight=args.decay)
            loss.backward()
            optimizer.step()
            bs = users.size(0)
            total_loss += loss.item() * bs
            total_bpr += bpr.item() * bs
            total_reg += reg.item() * bs
            total_count += bs
        avg_loss = total_loss / max(total_count, 1)
        avg_bpr = total_bpr / max(total_count, 1)
        avg_reg = total_reg / max(total_count, 1)
        elapsed_train = time.time() - t0
        do_valid = should_validate_epoch(epoch, args.epochs, args.valid_every)
        row = {"epoch": epoch, "loss": avg_loss, "bpr": avg_bpr, "reg": avg_reg, "valid_ran": int(do_valid), "monitor": "", "time_sec": elapsed_train}
        if not do_valid:
            print(f"Epoch {epoch:03d}: loss={avg_loss:.4f} bpr={avg_bpr:.4f} reg={avg_reg:.4f} | valid=skipped | time={elapsed_train:.1f}s", flush=True)
            log_rows.append(row)
            continue
        t_valid = time.time()
        metrics = evaluate_allranking(model, norm_adj, valid_ratings, eval_block_user_pos, num_items, device, topks=valid_topks, batch_size=args.eval_batch_size)
        cur_score = monitor_value(metrics, args.monitor, args.topk)
        elapsed_total = time.time() - t0
        elapsed_valid = time.time() - t_valid
        msg = [f"Epoch {epoch:03d}: loss={avg_loss:.4f} bpr={avg_bpr:.4f}"]
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
            torch.save({
                "model_state_dict": model.state_dict(),
                "model": "LightGCN",
                "num_users": num_users,
                "num_items": num_items,
                "emb_dim": args.emb_dim,
                "layers": args.layers,
                "epoch": epoch,
                "valid_metrics": metrics,
                "monitor": args.monitor,
                "monitor_topk": args.topk,
                "monitor_value": cur_score,
                "train_edges": train_pos,
                "args": vars(args),
            }, save_path)
            print(f"[save-best-valid-allranking] {save_path}")

    log_path = os.path.join(args.save_dir, "training_log.csv")
    if log_rows:
        fieldnames = sorted({k for r in log_rows for k in r.keys()})
        with open(log_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)
    summary = {"dataset": args.dataset, "data_dir": args.data_dir, "save_dir": args.save_dir, "model": "LightGCN",
               "best_epoch": best_epoch, "best_monitor_value": best_score, "best_valid_metrics": best_metrics,
               "checkpoint": save_path, "test_split_used_for_checkpoint_selection": False, "args": vars(args)}
    with open(os.path.join(args.save_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Best epoch={best_epoch}, monitor={best_score:.6f}")
    if best_metrics:
        print(json.dumps(best_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
