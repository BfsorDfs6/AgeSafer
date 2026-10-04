#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAL-Risk3 all-ranking evaluator for GMF / NeuMF / LightGCN checkpoints.

Metrics:
  HR@K, NDCG@K
  R17@K, RPLUS@K, RX@K, unsafe_any@K

Risk3 policy:
  minor unsafe: R17 / RPLUS / RX
  adult unsafe: RPLUS / RX
  adult R17 is allowed and is only counted in raw_R17_exposure@K, not R17@K.

This script follows the ML-1M all-ranking evaluation protocol:
  candidate pool = all_items - original train positives
  if --split test --exclude_valid_for_test: also remove valid positives
  then add held-out positives back if needed.
"""

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


def safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(str(x).strip()))
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(str(x).strip())
    except Exception:
        return default


def norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower().replace("\ufeff", ""))


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


def looks_like_header(parts: Sequence[str]) -> bool:
    return (not parts) or safe_int(parts[0], default=-999999) == -999999


def parse_topks(s: str) -> List[int]:
    vals = []
    for x in re.split(r"[,;\s]+", str(s).strip()):
        if x:
            vals.append(int(x))
    vals = sorted(set(k for k in vals if k > 0))
    return vals or [10]


def read_rating_by_user(path: str, name: str) -> Dict[int, List[int]]:
    by_user = defaultdict(list)
    rows = skipped = 0
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            parts = split_line(line)
            if not parts:
                continue
            if line_no == 1 and looks_like_header(parts):
                continue
            if len(parts) < 2:
                skipped += 1
                continue
            u, i = safe_int(parts[0]), safe_int(parts[1])
            if u < 0 or i < 0:
                skipped += 1
                continue
            by_user[u].append(i)
            rows += 1
    print(f"[load] {name}: rows={rows}, users={len(by_user)}, skipped={skipped}, path={path}")
    return dict(by_user)


def read_train_pairs(path: str) -> List[Tuple[int, int]]:
    pairs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            parts = split_line(line)
            if not parts:
                continue
            if line_no == 1 and looks_like_header(parts):
                continue
            if len(parts) < 2:
                continue
            u, i = safe_int(parts[0]), safe_int(parts[1])
            if u >= 0 and i >= 0:
                pairs.append((u, i))
    print(f"[load] graph train pairs: rows={len(pairs)}, path={path}")
    return pairs


def unique_int_list(xs: Sequence[int]) -> List[int]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def read_csv_rows(path: str):
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields


def find_col(fields: List[str], candidates: Sequence[str], required: bool = False, label: str = "") -> str:
    norm_to_real = {norm_col(f): f for f in fields}
    for c in candidates:
        nc = norm_col(c)
        if nc in norm_to_real:
            return norm_to_real[nc]
    for c in candidates:
        nc = norm_col(c)
        for nf, real in norm_to_real.items():
            if nc and nc in nf:
                return real
    if required:
        raise ValueError(f"Cannot find column for {label or candidates}. Available={fields}")
    return ""


def is_one(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def bucket_from_item_row(row: Dict[str, Any]) -> str:
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
    if "PG-13" in t or "PG13" in t or "TEENS13" in t:
        return "PG13"
    if "PG-CHILDREN" in t or t.startswith("PG") or "CHILDREN" in t:
        return "PG"
    if "ALLAGES" in t or t.startswith("G"):
        return "G"
    return "OTHER"


def load_item_buckets(path: str) -> Dict[int, str]:
    rows, fields = read_csv_rows(path)
    item_col = find_col(fields, ["inner_item_id", "item_id", "iid", "item"], required=True, label="item id")
    out = {}
    for row in rows:
        iid = safe_int(row.get(item_col), -1)
        if iid >= 0:
            out[iid] = bucket_from_item_row(row)
    print(f"[load] item_safe rows={len(rows)}, items={len(out)}, item_col={item_col}, path={path}")
    print("[bucket_counts]", dict(Counter(out.values())))
    return out


def load_user_is_minor(path: str) -> Dict[int, bool]:
    rows, fields = read_csv_rows(path)
    user_col = find_col(fields, ["inner_user_id", "user_id", "uid", "user"], required=True, label="user id")
    minor_col = find_col(fields, ["is_minor", "minor", "isMinor", "user_is_minor", "under18", "is_under_18"], required=False)
    age_col = find_col(fields, ["age_at_join", "age", "user_age"], required=False)
    out = {}
    for row in rows:
        uid = safe_int(row.get(user_col), -1)
        if uid < 0:
            continue
        if minor_col:
            v = str(row.get(minor_col, "")).strip().lower()
            out[uid] = v in {"1", "1.0", "true", "yes", "y", "minor", "under18", "under_18"}
        elif age_col:
            out[uid] = safe_float(row.get(age_col), 999.0) < 18.0
        else:
            out[uid] = False
    print(f"[load] users rows={len(rows)}, users={len(out)}, user_col={user_col}, minor_col={minor_col or 'N/A'}, age_col={age_col or 'N/A'}, path={path}")
    return out


class GMFModel(nn.Module):
    def __init__(self, num_users, num_items, num_factors=8):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)
        self.output = nn.Linear(num_factors, 1)

    def forward(self, users, items):
        return self.output(self.user_embedding(users) * self.item_embedding(items)).squeeze(-1)


class MLPModel(nn.Module):
    def __init__(self, num_users, num_items, layers=(64, 32, 16, 8), dropout=0.0):
        super().__init__()
        layers = list(layers)
        emb_dim = layers[0] // 2
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        blocks = []
        in_dim = layers[0]
        for out_dim in layers[1:]:
            blocks.append(nn.Linear(in_dim, out_dim))
            blocks.append(nn.ReLU())
            if dropout and dropout > 0:
                blocks.append(nn.Dropout(float(dropout)))
            in_dim = out_dim
        self.mlp = nn.Sequential(*blocks)
        self.output = nn.Linear(in_dim, 1)

    def forward(self, users, items):
        x = torch.cat([self.user_embedding(users), self.item_embedding(items)], dim=-1)
        return self.output(self.mlp(x)).squeeze(-1)


class NeuMFModel(nn.Module):
    def __init__(self, num_users, num_items, num_factors=8, layers=(64, 32, 16, 8), dropout=0.0):
        super().__init__()
        layers = list(layers)
        mlp_emb_dim = layers[0] // 2
        self.gmf_user_embedding = nn.Embedding(num_users, num_factors)
        self.gmf_item_embedding = nn.Embedding(num_items, num_factors)
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_emb_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_emb_dim)
        blocks = []
        in_dim = layers[0]
        for out_dim in layers[1:]:
            blocks.append(nn.Linear(in_dim, out_dim))
            blocks.append(nn.ReLU())
            if dropout and dropout > 0:
                blocks.append(nn.Dropout(float(dropout)))
            in_dim = out_dim
        self.mlp = nn.Sequential(*blocks)
        self.output = nn.Linear(num_factors + in_dim, 1)

    def forward(self, users, items):
        gmf = self.gmf_user_embedding(users) * self.gmf_item_embedding(items)
        mlp = torch.cat([self.mlp_user_embedding(users), self.mlp_item_embedding(items)], dim=-1)
        mlp = self.mlp(mlp)
        return self.output(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64, n_layers=3):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.emb_dim = int(emb_dim)
        self.n_layers = int(n_layers)
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)

    def propagate(self, norm_adj):
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_emb]
        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(norm_adj, x)
            embs.append(x)
        out = torch.stack(embs, dim=0).mean(dim=0)
        return torch.split(out, [self.num_users, self.num_items], dim=0)


def parse_layers(x) -> List[int]:
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(v) for v in re.split(r"[,;\s]+", str(x).strip().strip("[]")) if v]


def build_norm_adj(num_users, num_items, train_pairs, device):
    n_nodes = int(num_users) + int(num_items)
    rows, cols = [], []
    for u, i in train_pairs:
        if 0 <= int(u) < num_users and 0 <= int(i) < num_items:
            rows.append(int(u)); cols.append(num_users + int(i))
            rows.append(num_users + int(i)); cols.append(int(u))
    if not rows:
        raise ValueError("No graph edges for LightGCN")
    rows_t = torch.tensor(rows, dtype=torch.long)
    cols_t = torch.tensor(cols, dtype=torch.long)
    deg = torch.zeros(n_nodes, dtype=torch.float32)
    deg.index_add_(0, rows_t, torch.ones_like(rows_t, dtype=torch.float32))
    inv = torch.pow(deg.clamp(min=1.0), -0.5)
    vals = inv[rows_t] * inv[cols_t]
    idx = torch.stack([rows_t, cols_t], dim=0)
    return torch.sparse_coo_tensor(idx, vals, (n_nodes, n_nodes)).coalesce().to(device)


def infer_model_type(ckpt: Dict[str, Any], checkpoint_path: str) -> str:
    m = str(ckpt.get("model", ckpt.get("args", {}).get("model", ""))).lower()
    name = checkpoint_path.lower()
    if "lightgcn" in m or "lightgcn" in name or "emb_dim" in ckpt:
        return "LightGCN"
    if "neumf" in m or "neumf" in name:
        return "NeuMF"
    if "mlp" == m or "mlp" in name:
        return "MLP"
    return "GMF"


def load_model_or_embeddings(checkpoint: str, graph_rating: str, train_rating: str, device: str):
    print(f"[load] checkpoint={checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")
    model_type = infer_model_type(ckpt, checkpoint)
    num_users = int(ckpt["num_users"])
    num_items = int(ckpt["num_items"])
    state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if state is None:
        raise ValueError("Cannot find model_state_dict/state_dict in checkpoint")

    if model_type == "LightGCN":
        emb_dim = int(ckpt.get("emb_dim", ckpt.get("args", {}).get("emb_dim", 64)))
        layers = int(ckpt.get("layers", ckpt.get("args", {}).get("layers", 3)))
        model = LightGCN(num_users, num_items, emb_dim=emb_dim, n_layers=layers)
        model.load_state_dict(state, strict=True)
        model.to(device); model.eval()
        train_edges = ckpt.get("train_edges", None)
        if train_edges is None:
            gr = graph_rating or train_rating
            print(f"[graph] checkpoint has no train_edges; use graph_rating={gr}")
            train_edges = read_train_pairs(gr)
        else:
            print(f"[graph] use checkpoint train_edges, edges={len(train_edges)}")
        adj = build_norm_adj(num_users, num_items, train_edges, device)
        with torch.no_grad():
            user_g, item_g = model.propagate(adj)
        print(f"[model] LightGCN users={num_users}, items={num_items}, emb_dim={emb_dim}, layers={layers}")
        return {"type": "LightGCN", "user_g": user_g, "item_g": item_g, "num_users": num_users, "num_items": num_items}

    layers = parse_layers(ckpt.get("layers", ckpt.get("args", {}).get("layers", [64, 32, 16, 8])))
    num_factors = int(ckpt.get("num_factors", ckpt.get("args", {}).get("num_factors", 8)))
    dropout = float(ckpt.get("dropout", ckpt.get("args", {}).get("dropout", 0.0)))
    if model_type == "NeuMF":
        model = NeuMFModel(num_users, num_items, num_factors=num_factors, layers=layers, dropout=dropout)
    elif model_type == "MLP":
        model = MLPModel(num_users, num_items, layers=layers, dropout=dropout)
    else:
        model = GMFModel(num_users, num_items, num_factors=num_factors)
        model_type = "GMF"
    model.load_state_dict(state, strict=True)
    model.to(device); model.eval()
    print(f"[model] {model_type} users={num_users}, items={num_items}, factors={num_factors}, layers={layers}")
    return {"type": model_type, "model": model, "num_users": num_users, "num_items": num_items}


@torch.no_grad()
def score_items(loaded, uid: int, item_ids: List[int], device: str, batch_size: int) -> List[float]:
    out = []
    if loaded["type"] == "LightGCN":
        uemb = loaded["user_g"][int(uid)]
        item_g = loaded["item_g"]
        for start in range(0, len(item_ids), batch_size):
            batch = item_ids[start:start + batch_size]
            idx = torch.tensor(batch, dtype=torch.long, device=device)
            scores = (item_g[idx] * uemb).sum(dim=-1)
            out.extend(scores.detach().cpu().tolist())
        return out

    model = loaded["model"]
    for start in range(0, len(item_ids), batch_size):
        batch = item_ids[start:start + batch_size]
        users = torch.full((len(batch),), int(uid), dtype=torch.long, device=device)
        items = torch.tensor(batch, dtype=torch.long, device=device)
        logits = model(users, items)
        out.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return out


def risk3_flags(is_minor: bool, bucket: str) -> Dict[str, int]:
    b = str(bucket or "UNKNOWN").upper()
    r17 = int(is_minor and b == "R17")
    rplus = int(b == "RPLUS")
    rx = int(b == "RX")
    return {
        "R17": r17,
        "RPLUS": rplus,
        "RX": rx,
        "unsafe_any": int(r17 or rplus or rx),
        "raw_R17_exposure": int(b == "R17"),
        "raw_RPLUS_exposure": int(b == "RPLUS"),
        "raw_RX_exposure": int(b == "RX"),
    }


def init_acc(topks):
    return {
        "users": 0,
        "hr_sum": {k: 0.0 for k in topks},
        "ndcg_sum": {k: 0.0 for k in topks},
        "top_items": {k: 0 for k in topks},
        "risk_items": {name: {k: 0 for k in topks} for name in ["R17", "RPLUS", "RX", "unsafe_any", "raw_R17_exposure", "raw_RPLUS_exposure", "raw_RX_exposure"]},
    }


def update_acc(acc, topks, top_infos, pos_set):
    acc["users"] += 1
    for k in topks:
        topk = [r for r in top_infos if r["rank"] <= k]
        hit, dcg = 0.0, 0.0
        for r in topk:
            if r["iid"] in pos_set:
                hit = 1.0
                dcg += 1.0 / math.log2(r["rank"] + 1.0)
        ideal_len = min(len(pos_set), k)
        idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_len + 1)) if ideal_len > 0 else 1.0
        acc["hr_sum"][k] += hit
        acc["ndcg_sum"][k] += dcg / idcg if idcg > 0 else 0.0
        acc["top_items"][k] += len(topk)
        for name in acc["risk_items"]:
            acc["risk_items"][name][k] += sum(int(r.get(name, 0)) for r in topk)


def finalize(acc, topks):
    out = {"users": acc["users"]}
    users = max(acc["users"], 1)
    for k in topks:
        denom = max(acc["top_items"][k], 1)
        out[f"HR@{k}"] = acc["hr_sum"][k] / users
        out[f"NDCG@{k}"] = acc["ndcg_sum"][k] / users
        for name in acc["risk_items"]:
            out[f"{name}@{k}"] = acc["risk_items"][name][k] / denom
    return out


def round_tree(obj, digits):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_tree(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_tree(v, digits) for v in obj]
    return obj


def write_csv(path: str, rows: List[Dict[str, Any]]):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="Data")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", choices=["valid", "test"], default="test")
    ap.add_argument("--train_rating", default="", help="Original train.rating used for candidate masking.")
    ap.add_argument("--valid_rating", default="")
    ap.add_argument("--graph_rating", default="", help="Fallback graph rating for LightGCN if checkpoint has no train_edges.")
    ap.add_argument("--item_safe", required=True)
    ap.add_argument("--user_info", required=True)
    ap.add_argument("--topks", default="1,5,10,20")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--score_batch_size", type=int, default=32768)
    ap.add_argument("--max_users", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=100)
    ap.add_argument("--exclude_valid_for_test", action="store_true")
    ap.add_argument("--output_json", default="")
    ap.add_argument("--output_csv", default="")
    ap.add_argument("--output_topk_csv", default="")
    ap.add_argument("--round_digits", type=int, default=6)
    args = ap.parse_args()

    topks = parse_topks(args.topks)
    max_topk = max(topks)
    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    print(f"[info] device={device}, topks={topks}, split={args.split}")

    train_rating = args.train_rating or os.path.join(args.data_dir, f"{args.dataset}.train.rating")
    eval_rating = os.path.join(args.data_dir, f"{args.dataset}.{args.split}.rating")
    valid_rating = args.valid_rating or os.path.join(args.data_dir, f"{args.dataset}.valid.rating")

    train_by_user = read_rating_by_user(train_rating, "train-mask")
    eval_by_user = read_rating_by_user(eval_rating, args.split)
    valid_by_user = read_rating_by_user(valid_rating, "valid") if (args.split == "test" and args.exclude_valid_for_test and os.path.exists(valid_rating)) else {}
    item_bucket = load_item_buckets(args.item_safe)
    user_minor = load_user_is_minor(args.user_info)
    loaded = load_model_or_embeddings(args.checkpoint, args.graph_rating, train_rating, device)

    num_users, num_items = loaded["num_users"], loaded["num_items"]
    all_items = list(range(num_items))
    users = sorted(eval_by_user.keys())
    if args.max_users > 0:
        users = users[: args.max_users]

    acc = {"all": init_acc(topks), "minor": init_acc(topks), "adult": init_acc(topks)}
    topk_rows = []

    for n_done, uid in enumerate(users, 1):
        if uid < 0 or uid >= num_users:
            continue
        pos_items = [i for i in unique_int_list(eval_by_user.get(uid, [])) if 0 <= i < num_items]
        if not pos_items:
            continue
        pos_set = set(pos_items)
        block = set(i for i in train_by_user.get(uid, []) if 0 <= i < num_items)
        if args.split == "test" and args.exclude_valid_for_test:
            block.update(i for i in valid_by_user.get(uid, []) if 0 <= i < num_items and i not in pos_set)
        candidates = [i for i in all_items if i not in block]
        cset = set(candidates)
        for p in pos_items:
            if p not in cset:
                candidates.append(p)
                cset.add(p)
        scores = score_items(loaded, uid, candidates, device, args.score_batch_size)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:max_topk]
        is_minor = bool(user_minor.get(uid, False))
        group = "minor" if is_minor else "adult"
        top_infos = []
        for rank, (iid, score) in enumerate(ranked, 1):
            bucket = item_bucket.get(int(iid), "UNKNOWN")
            flags = risk3_flags(is_minor, bucket)
            info = {"rank": rank, "iid": int(iid), "score": float(score), "bucket": bucket, "is_positive": int(iid in pos_set), **flags}
            top_infos.append(info)
            if args.output_topk_csv:
                topk_rows.append({"group": group, "user_id": uid, "rank": rank, "item_id": iid, "score": f"{float(score):.8f}", "bucket": bucket, "is_positive": int(iid in pos_set), **flags})
        update_acc(acc["all"], topks, top_infos, pos_set)
        update_acc(acc[group], topks, top_infos, pos_set)
        if n_done == 1 or n_done % args.progress_every == 0:
            m = finalize(acc["all"], topks)
            msg = [f"[progress] users_seen={n_done}, eval_users={acc['all']['users']}, uid={uid}, group={group}, candidates={len(candidates)}"]
            for k in topks:
                msg.append(f"HR@{k}={m[f'HR@{k}']:.4f} NDCG@{k}={m[f'NDCG@{k}']:.4f} R17@{k}={m[f'R17@{k}']:.4f} RPLUS@{k}={m[f'RPLUS@{k}']:.4f} RX@{k}={m[f'RX@{k}']:.4f}")
            print(" | ".join(msg), flush=True)

    metrics = {g: finalize(a, topks) for g, a in acc.items()}
    metrics = round_tree(metrics, args.round_digits)
    summary = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "dataset": args.dataset,
        "split": args.split,
        "train_rating": train_rating,
        "valid_rating": valid_rating,
        "exclude_valid_for_test": bool(args.exclude_valid_for_test),
        "model_type": loaded["type"],
        "num_users": num_users,
        "num_items": num_items,
        "topks": topks,
        "metrics": metrics,
        "risk_policy": "minor: R17/RPLUS/RX unsafe; adult: RPLUS/RX unsafe; adult R17 allowed",
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[write] {args.output_json}")

    if args.output_csv:
        rows = []
        for group, vals in metrics.items():
            row = {"group": group, "checkpoint": args.checkpoint, "dataset": args.dataset, "split": args.split, "model_type": loaded["type"]}
            row.update(vals)
            rows.append(row)
        write_csv(args.output_csv, rows)

    if args.output_topk_csv:
        write_csv(args.output_topk_csv, topk_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
