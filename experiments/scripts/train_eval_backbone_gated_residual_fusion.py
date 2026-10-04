#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified score-level gated residual fusion for GMF / NeuMF / MLP / LightGCN.

Purpose
-------
Train a small fusion gate on top of two frozen backbone checkpoints:
  s_final(u,i) = s_base(u,i) + gate(u,i) * (s_psg(u,i) - s_base(u,i))

The gate uses scalar features only, so it is architecture-agnostic:
  s_base, s_psg, delta, psg_pred, p1, margin,
  sex/violence/profanity/drug/intense violation flags, unsafe_any,
  adult_content, is_minor.

It supports the same Dim5-v4 rule as eval_gmf_allranking_safety.py:
  minor: item_dim >= minor_block_at is unsafe
  adult: item_dim >= adult_block_at is unsafe

For LightGCN, pass --base_graph_rating and --psg_graph_rating when checkpoints do not contain train_edges.
"""

import argparse
import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

RISK_DIMS = ["sex_code", "violence_code", "profanity_code", "drug_code", "intense_code"]
DIM_OUT = {
    "sex_code": "sex",
    "violence_code": "violence",
    "profanity_code": "profanity",
    "drug_code": "drug",
    "intense_code": "intense",
}
ADULT_DIM = "isAdult"
DIM_NAMES = ["sex", "violence", "profanity", "drug", "intense"]


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def safe_int(x: Any, default: int = -1) -> int:
    try:
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


def parse_boolish(x: Any) -> bool:
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "minor", "under18", "under_18"}:
        return True
    if s in {"0", "false", "no", "n", "adult"}:
        return False
    try:
        return float(s) > 0
    except Exception:
        return False


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
    out = []
    for x in re.split(r"[,;\s]+", str(s).strip()):
        if x:
            out.append(int(x))
    return sorted(set(k for k in out if k > 0)) or [10]


def read_rating_by_user(path: str, name: str) -> Dict[int, List[int]]:
    by_user = defaultdict(list)
    rows = skipped = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
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
    return {int(u): list(v) for u, v in by_user.items()}


def read_rating_pairs(path: str) -> List[Tuple[int, int]]:
    pairs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
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
    return pairs


def unique_int_list(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def read_csv_dict(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields


def find_col(fields: List[str], candidates: List[str], required=False, label="") -> str:
    norm_to_real = {norm_name(f): f for f in fields}
    for cand in candidates:
        key = norm_name(cand)
        if key in norm_to_real:
            return norm_to_real[key]
    for cand in candidates:
        key = norm_name(cand)
        for nf, real in norm_to_real.items():
            if key and key in nf:
                return real
    if required:
        raise ValueError(f"Cannot find column for {label or candidates}. Available={fields}")
    return ""


def find_item_id_col(fields):
    return find_col(fields, ["inner_item_id", "item_inner_id", "item_id", "iid", "movie_id", "candidate_inner_item_id"], True, "item id")


def find_user_id_col(fields):
    return find_col(fields, ["inner_user_id", "user_inner_id", "user_id", "uid"], True, "user id")


def find_item_risk_cols(fields):
    out = {}
    for dim in RISK_DIMS + [ADULT_DIM]:
        short = dim.replace("_code", "")
        names = [dim, dim.lower(), short, short + "_code", short + "_level", short + "_risk", short + "_score", "item_" + dim, "risk_" + dim]
        out[dim] = find_col(fields, names, False, f"item risk {dim}")
    return out


def find_minor_col(fields):
    return find_col(fields, ["is_minor", "minor", "isMinor", "user_is_minor", "under18", "is_under_18"], False, "minor flag")


def load_item_safe(path):
    rows, fields = read_csv_dict(path)
    item_col = find_item_id_col(fields)
    risk_cols = find_item_risk_cols(fields)
    item_safe = {}
    for r in rows:
        iid = safe_int(r.get(item_col), -1)
        if iid >= 0:
            item_safe[iid] = r
    print(f"[load] item_safe rows={len(rows)}, items={len(item_safe)}, path={path}")
    print(f"[detect] item_id_col={item_col}")
    print(f"[detect] item_risk_cols={risk_cols}")
    return item_safe, risk_cols


def load_user_info(path):
    rows, fields = read_csv_dict(path)
    user_col = find_user_id_col(fields)
    minor_col = find_minor_col(fields)
    user_info = {}
    for r in rows:
        uid = safe_int(r.get(user_col), -1)
        if uid >= 0:
            user_info[uid] = r
    print(f"[load] user_info rows={len(rows)}, users={len(user_info)}, path={path}")
    print(f"[detect] user_id_col={user_col}, minor_col={minor_col or 'N/A'}")
    return user_info, minor_col


def is_minor_user(uid, user_info, minor_col):
    return parse_boolish(user_info.get(int(uid), {}).get(minor_col, "")) if minor_col else False


def dim_violations(uid, iid, item_safe, item_risk_cols, user_info, minor_col, minor_block_at, adult_block_at, isadult_policy):
    row_i = item_safe.get(int(iid), {})
    is_minor = is_minor_user(uid, user_info, minor_col)
    block_at = float(minor_block_at if is_minor else adult_block_at)
    vios, values = {}, {}
    for dim in RISK_DIMS:
        col = item_risk_cols.get(dim, "")
        val = safe_float(row_i.get(col), 0.0) if col else 0.0
        vio = int(val >= block_at)
        vios[DIM_OUT[dim]] = vio
        values[dim] = val
    adult_col = item_risk_cols.get(ADULT_DIM, "")
    adult_val = safe_float(row_i.get(adult_col), 0.0) if adult_col else 0.0
    adult_content = int(adult_val >= 1)
    adult_hard = int(isadult_policy in {"minor_only", "all"} and adult_content and (is_minor or isadult_policy == "all"))
    any_unsafe = int(any(vios.values()) or adult_hard)
    return vios, any_unsafe, adult_content, is_minor, values


# ---------- NCF models ----------
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


def parse_layers(layers):
    if isinstance(layers, str):
        return [int(x) for x in re.split(r"[,;\s]+", layers.strip().strip("[]")) if x]
    return list(layers)


def build_norm_adj(num_users, num_items, train_pairs, device):
    n_nodes = num_users + num_items
    rows, cols = [], []
    for u, i in train_pairs:
        if 0 <= u < num_users and 0 <= i < num_items:
            rows.append(u); cols.append(num_users + i)
            rows.append(num_users + i); cols.append(u)
    if not rows:
        raise ValueError("No train edges for LightGCN graph")
    rows_t = torch.tensor(rows, dtype=torch.long)
    cols_t = torch.tensor(cols, dtype=torch.long)
    deg = torch.zeros(n_nodes, dtype=torch.float32)
    deg.index_add_(0, rows_t, torch.ones_like(rows_t, dtype=torch.float32))
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    vals = deg_inv_sqrt[rows_t] * deg_inv_sqrt[cols_t]
    idx = torch.stack([rows_t, cols_t], dim=0)
    return torch.sparse_coo_tensor(idx, vals, (n_nodes, n_nodes)).coalesce().to(device)


class FrozenScorer:
    def __init__(self, kind, checkpoint, device, graph_rating=""):
        self.kind = kind.lower()
        self.checkpoint = checkpoint
        self.device = device
        self.graph_rating = graph_rating
        self.model = None
        self.user_g = None
        self.item_g = None
        self.num_users = 0
        self.num_items = 0
        self._load()

    def _build_ncf_model(self, ckpt):
        model_name = str(ckpt.get("model", ckpt.get("args", {}).get("model", self.kind))).lower()
        if self.kind in {"gmf", "mlp", "neumf"}:
            model_name = self.kind
        num_users = int(ckpt["num_users"])
        num_items = int(ckpt["num_items"])
        num_factors = int(ckpt.get("num_factors", ckpt.get("args", {}).get("num_factors", 8)))
        layers = parse_layers(ckpt.get("layers", ckpt.get("args", {}).get("layers", [64, 32, 16, 8])))
        dropout = float(ckpt.get("dropout", ckpt.get("args", {}).get("dropout", 0.0)))
        if model_name == "gmf":
            model = GMFModel(num_users, num_items, num_factors)
        elif model_name == "mlp":
            model = MLPModel(num_users, num_items, layers, dropout)
        elif model_name == "neumf":
            model = NeuMFModel(num_users, num_items, num_factors, layers, dropout)
        else:
            raise ValueError(f"Unknown NCF model kind={model_name}")
        state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        if state is None:
            raise ValueError(f"Cannot find model_state_dict in {self.checkpoint}")
        model.load_state_dict(state, strict=True)
        return model, num_users, num_items

    def _load(self):
        print(f"[load scorer] kind={self.kind}, checkpoint={self.checkpoint}")
        ckpt = torch.load(self.checkpoint, map_location="cpu")
        if self.kind in {"gmf", "mlp", "neumf"}:
            self.model, self.num_users, self.num_items = self._build_ncf_model(ckpt)
            self.model.to(self.device).eval()
        elif self.kind == "lightgcn":
            self.num_users = int(ckpt["num_users"])
            self.num_items = int(ckpt["num_items"])
            emb_dim = int(ckpt.get("emb_dim", ckpt.get("args", {}).get("emb_dim", 64)))
            layers = int(ckpt.get("layers", ckpt.get("args", {}).get("layers", 3)))
            self.model = LightGCN(self.num_users, self.num_items, emb_dim, layers)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
            if state is None:
                raise ValueError(f"Cannot find model_state_dict in {self.checkpoint}")
            self.model.load_state_dict(state, strict=True)
            self.model.to(self.device).eval()
            train_edges = ckpt.get("train_edges", None)
            if train_edges is None:
                if not self.graph_rating:
                    raise ValueError("LightGCN checkpoint has no train_edges; pass --base_graph_rating/--psg_graph_rating")
                print(f"[graph] use graph_rating={self.graph_rating}")
                train_edges = read_rating_pairs(self.graph_rating)
            else:
                print("[graph] use train_edges stored in checkpoint")
            norm_adj = build_norm_adj(self.num_users, self.num_items, train_edges, self.device)
            with torch.no_grad():
                self.user_g, self.item_g = self.model.propagate(norm_adj)
        else:
            raise ValueError(f"Unsupported backbone={self.kind}")
        print(f"[scorer ready] kind={self.kind}, users={self.num_users}, items={self.num_items}")

    @torch.no_grad()
    def score_pairs(self, users: List[int], items: List[int], batch_size: int = 32768) -> torch.Tensor:
        scores = []
        for st in range(0, len(users), batch_size):
            us = users[st:st+batch_size]
            it = items[st:st+batch_size]
            u = torch.tensor(us, dtype=torch.long, device=self.device)
            i = torch.tensor(it, dtype=torch.long, device=self.device)
            if self.kind in {"gmf", "mlp", "neumf"}:
                logits = self.model(u, i)
                s = torch.sigmoid(logits)
            else:
                s = (self.user_g[u] * self.item_g[i]).sum(dim=-1)
            scores.append(s.detach())
        return torch.cat(scores, dim=0) if scores else torch.empty(0, device=self.device)


class GateNet(nn.Module):
    def __init__(self, input_dim, hidden=32, dropout=0.1, gate_max=1.0):
        super().__init__()
        self.gate_max = float(gate_max)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.gate_max * torch.sigmoid(self.net(x).squeeze(-1))


class FusionFeatureBuilder:
    def __init__(self, base_scorer, psg_scorer, pred_map, item_safe, item_risk_cols, user_info, minor_col, args):
        self.base = base_scorer
        self.psg = psg_scorer
        self.pred_map = pred_map
        self.item_safe = item_safe
        self.item_risk_cols = item_risk_cols
        self.user_info = user_info
        self.minor_col = minor_col
        self.args = args
        self.input_dim = 14

    def build(self, users: List[int], items: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        device = self.args.device_resolved
        bs = self.base.score_pairs(users, items, self.args.score_batch_size)
        ps = self.psg.score_pairs(users, items, self.args.score_batch_size)

        # Important speed fix:
        # Move branch scores from GPU to CPU once per candidate list.
        # The old version called .cpu() for every single item, which caused
        # thousands of GPU synchronizations per user during all-ranking eval.
        bs_cpu = bs.detach().cpu().tolist()
        ps_cpu = ps.detach().cpu().tolist()
        diff_cpu = (ps - bs).detach().cpu().tolist()

        feats = []
        infos = []
        for idx, (u, i) in enumerate(zip(users, items)):
            key = (int(u), int(i))
            pf = self.pred_map.get(key, (0.0, 0.0, 0.0))  # psg_pred, p1, margin
            vios, any_unsafe, adult_content, is_minor, _ = dim_violations(
                u, i, self.item_safe, self.item_risk_cols, self.user_info, self.minor_col,
                self.args.minor_block_at, self.args.adult_block_at, self.args.isadult_policy)
            row = [
                float(bs_cpu[idx]),
                float(ps_cpu[idx]),
                float(diff_cpu[idx]),
                float(pf[0]), float(pf[1]), float(pf[2]),
                float(vios["sex"]), float(vios["violence"]), float(vios["profanity"]), float(vios["drug"]), float(vios["intense"]),
                float(any_unsafe), float(adult_content), float(is_minor),
            ]
            feats.append(row)
            infos.append({"vios": vios, "unsafe_any": any_unsafe, "adult_content": adult_content, "is_minor": is_minor})
        return torch.tensor(feats, dtype=torch.float32, device=device), bs, ps, infos


def read_pred_jsonl(path: str) -> Dict[Tuple[int, int], Tuple[float, float, float]]:
    out = {}
    if not path or not os.path.exists(path):
        print(f"[warn] pred_jsonl missing or empty: {path}; use default zeros")
        return out
    user_keys = ["user_id", "uid", "user", "inner_user_id", "candidate_inner_user_id"]
    item_keys = ["item_id", "iid", "item", "inner_item_id", "candidate_inner_item_id"]
    rows = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            u = -1
            i = -1
            meta = obj.get("metadata", {}) if isinstance(obj.get("metadata", {}), dict) else {}

            # Try top-level fields first, then metadata fields.
            for src in (obj, meta):
                for k in user_keys + ["inner_user_id", "raw_user_id", "candidate_inner_user_id"]:
                    if k in src:
                        u = safe_int(src.get(k), -1)
                        break
                if u >= 0:
                    break

            for src in (obj, meta):
                for k in item_keys + ["inner_item_id", "candidate_inner_item_id", "candidate_item_id"]:
                    if k in src:
                        i = safe_int(src.get(k), -1)
                        break
                if i >= 0:
                    break

            if u < 0 or i < 0:
                continue
            psg_pred = safe_float(obj.get("psg_pred", obj.get("pred", obj.get("label", 0.0))), 0.0)
            p1 = safe_float(obj.get("p1", obj.get("prob_1", obj.get("prob1", 0.0))), 0.0)
            margin = safe_float(obj.get("margin_1_minus_0", obj.get("margin", obj.get("logit_margin", 0.0))), 0.0)
            out[(u, i)] = (psg_pred, p1, margin)
            rows += 1
    print(f"[load] pred_jsonl rows={rows}, map_size={len(out)}, path={path}")
    return out


def final_score(gate, feats, bs, ps):
    g = gate(feats)
    return bs + g * (ps - bs), g


def sample_negative(uid, num_items, block_set, rng, max_trials=100):
    for _ in range(max_trials):
        j = rng.randrange(num_items)
        if j not in block_set:
            return j
    return rng.randrange(num_items)


def sample_safe_unsafe(uid, num_items, block_set, fb, rng, max_trials=200):
    safe = unsafe = None
    for _ in range(max_trials):
        j = rng.randrange(num_items)
        if j in block_set:
            continue
        vios, any_unsafe, _, _, _ = dim_violations(uid, j, fb.item_safe, fb.item_risk_cols, fb.user_info, fb.minor_col,
                                                   fb.args.minor_block_at, fb.args.adult_block_at, fb.args.isadult_policy)
        if any_unsafe and unsafe is None:
            unsafe = j
        if (not any_unsafe) and safe is None:
            safe = j
        if safe is not None and unsafe is not None:
            return safe, unsafe
    return safe, unsafe


def init_acc(topks):
    return {
        "users": 0,
        "hr_sum": {k: 0.0 for k in topks},
        "ndcg_sum": {k: 0.0 for k in topks},
        "top_items": {k: 0 for k in topks},
        "dim_items": {d: {k: 0 for k in topks} for d in DIM_NAMES},
        "any_items": {k: 0 for k in topks},
        "adult_items": {k: 0 for k in topks},
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
        for d in DIM_NAMES:
            acc["dim_items"][d][k] += sum(int(r[d]) for r in topk)
        acc["any_items"][k] += sum(int(r["unsafe_any"]) for r in topk)
        acc["adult_items"][k] += sum(int(r["adult_content"]) for r in topk)


def finalize(acc, topks):
    out = {"users": acc["users"]}
    for k in topks:
        users = max(acc["users"], 1)
        denom = max(acc["top_items"][k], 1)
        out[f"HR@{k}"] = acc["hr_sum"][k] / users
        out[f"NDCG@{k}"] = acc["ndcg_sum"][k] / users
        for d in DIM_NAMES:
            out[f"{d}@{k}"] = acc["dim_items"][d][k] / denom
        out[f"unsafe_any@{k}"] = acc["any_items"][k] / denom
        out[f"adult_content@{k}"] = acc["adult_items"][k] / denom
    return out


def evaluate(gate, fb, users, eval_by_user, train_by_user, valid_by_user, split, topks, args, output_topk_csv=""):
    max_topk = max(topks)
    num_items = min(fb.base.num_items, fb.psg.num_items)
    all_items = list(range(num_items))
    acc = {"all": init_acc(topks), "minor": init_acc(topks), "adult": init_acc(topks)}
    topk_rows = []
    gate.eval()
    with torch.no_grad():
        for n_done, uid in enumerate(users, 1):
            if uid < 0 or uid >= fb.base.num_users or uid >= fb.psg.num_users:
                continue
            pos_items = [i for i in unique_int_list(eval_by_user.get(uid, [])) if 0 <= i < num_items]
            if not pos_items:
                continue
            pos_set = set(pos_items)
            block = set(i for i in train_by_user.get(uid, []) if 0 <= i < num_items)
            if split == "test" and args.exclude_valid_for_test:
                block.update(i for i in valid_by_user.get(uid, []) if 0 <= i < num_items and i not in pos_set)
            candidates = [i for i in all_items if i not in block]
            cset = set(candidates)
            for p in pos_items:
                if p not in cset:
                    candidates.append(p); cset.add(p)
            us = [uid] * len(candidates)
            feats, bs, ps, infos = fb.build(us, candidates)
            scores, gates = final_score(gate, feats, bs, ps)

            # Faster top-k selection. The old version sorted all candidate scores
            # in Python, which is unnecessary for all-ranking Top-K evaluation.
            topk_n = min(max_topk, scores.numel())
            top_vals, top_idx = torch.topk(scores, k=topk_n, largest=True, sorted=True)
            vals_top = top_vals.detach().cpu().tolist()
            order = top_idx.detach().cpu().tolist()
            group = "minor" if is_minor_user(uid, fb.user_info, fb.minor_col) else "adult"
            top_infos = []
            for rank, idx in enumerate(order, 1):
                iid = int(candidates[idx])
                score_val = float(vals_top[rank - 1])
                info = {"rank": rank, "iid": iid, "score": score_val, "is_positive": int(iid in pos_set),
                        "unsafe_any": infos[idx]["unsafe_any"], "adult_content": infos[idx]["adult_content"], **infos[idx]["vios"]}
                top_infos.append(info)
                if output_topk_csv:
                    row = {"split": split, "group": group, "user_id": uid, "rank": rank, "item_id": iid, "iid": iid,
                           "score": f"{score_val:.8f}", "gate": f"{float(gates[idx].detach().cpu()):.8f}",
                           "is_positive": int(iid in pos_set), **infos[idx]["vios"],
                           "adult_content": infos[idx]["adult_content"], "unsafe_any": infos[idx]["unsafe_any"]}
                    topk_rows.append(row)
            update_acc(acc["all"], topks, top_infos, pos_set)
            update_acc(acc[group], topks, top_infos, pos_set)
            if n_done == 1 or n_done % args.progress_every == 0:
                m = finalize(acc["all"], topks)
                print(f"[eval {split}] users_seen={n_done}, eval_users={acc['all']['users']}, HR@{args.monitor_k}={m.get(f'HR@{args.monitor_k}',0):.4f}, NDCG@{args.monitor_k}={m.get(f'NDCG@{args.monitor_k}',0):.4f}", flush=True)
    if output_topk_csv and topk_rows:
        with open(output_topk_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(topk_rows[0].keys()))
            w.writeheader(); w.writerows(topk_rows)
        print(f"[write] {output_topk_csv}")
    return {g: finalize(a, topks) for g, a in acc.items()}


def train_gate(gate, fb, train_users, holdout_users, valid_by_user, train_by_user, topks, args):
    device = args.device_resolved
    rng = random.Random(args.seed)
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_items = min(fb.base.num_items, fb.psg.num_items)
    best_metric = -1e18
    best_state = None
    log_rows = []
    users = [u for u in train_users if valid_by_user.get(u)]
    print(f"[train gate] calib_users={len(users)}, holdout_users={len(holdout_users)}, num_items={num_items}")
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(users)
        gate.train()
        loss_sum = rank_sum = safe_sum = n_batches = 0
        for st in range(0, len(users), args.batch_users):
            batch_users = users[st:st + args.batch_users]
            pos_users, pos_items, neg_users, neg_items = [], [], [], []
            safe_users, safe_items, unsafe_users, unsafe_items = [], [], [], []
            for u in batch_users:
                pos_list = [i for i in valid_by_user.get(u, []) if 0 <= i < num_items]
                if not pos_list:
                    continue
                pi = rng.choice(pos_list)
                block = set(i for i in train_by_user.get(u, []) if 0 <= i < num_items)
                block.update(pos_list)
                ni = sample_negative(u, num_items, block, rng)
                pos_users.append(u); pos_items.append(pi)
                neg_users.append(u); neg_items.append(ni)
                if args.safety_weight > 0:
                    s_i, u_i = sample_safe_unsafe(u, num_items, block, fb, rng, args.safety_trials)
                    if s_i is not None and u_i is not None:
                        safe_users.append(u); safe_items.append(s_i)
                        unsafe_users.append(u); unsafe_items.append(u_i)
            if not pos_users:
                continue
            p_feats, p_bs, p_ps, _ = fb.build(pos_users, pos_items)
            n_feats, n_bs, n_ps, _ = fb.build(neg_users, neg_items)
            p_score, _ = final_score(gate, p_feats, p_bs, p_ps)
            n_score, _ = final_score(gate, n_feats, n_bs, n_ps)
            loss_rank = -F.logsigmoid(p_score - n_score).mean()
            loss = loss_rank
            loss_safe_val = torch.tensor(0.0, device=device)
            if args.safety_weight > 0 and safe_users:
                s_feats, s_bs, s_ps, _ = fb.build(safe_users, safe_items)
                u_feats, u_bs, u_ps, _ = fb.build(unsafe_users, unsafe_items)
                s_score, _ = final_score(gate, s_feats, s_bs, s_ps)
                u_score, _ = final_score(gate, u_feats, u_bs, u_ps)
                loss_safe_val = F.relu(u_score - s_score + args.safety_margin).mean()
                loss = loss + args.safety_weight * loss_safe_val
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.detach().cpu())
            rank_sum += float(loss_rank.detach().cpu())
            safe_sum += float(loss_safe_val.detach().cpu())
            n_batches += 1
        row = {"epoch": epoch, "loss": loss_sum / max(n_batches, 1), "loss_rank": rank_sum / max(n_batches, 1), "loss_safe": safe_sum / max(n_batches, 1)}
        if epoch == 1 or epoch % args.valid_every == 0 or epoch == args.epochs:
            metrics = evaluate(gate, fb, holdout_users, valid_by_user, train_by_user, {}, "valid", topks, args)
            m = metrics["all"]
            ndcg = m.get(f"NDCG@{args.monitor_k}", 0.0)
            unsafe = m.get(f"unsafe_any@{args.monitor_k}", 0.0)
            monitor = ndcg - args.monitor_safety_weight * unsafe
            row.update({"valid_NDCG": ndcg, "valid_HR": m.get(f"HR@{args.monitor_k}", 0.0), "valid_unsafe_any": unsafe, "monitor": monitor})
            print(f"[epoch {epoch}] loss={row['loss']:.5f} valid_NDCG@{args.monitor_k}={ndcg:.5f} unsafe={unsafe:.5f} monitor={monitor:.5f}")
            if monitor > best_metric:
                best_metric = monitor
                best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
                print(f"[best] epoch={epoch}, monitor={monitor:.6f}")
        else:
            print(f"[epoch {epoch}] loss={row['loss']:.5f}")
        log_rows.append(row)
    if best_state is not None:
        gate.load_state_dict(best_state)
    return log_rows, best_metric


def fmt(x, digits):
    return f"{float(x):.{digits}f}"


def write_metrics_csv(path, metrics_by_split, topks, digits):
    fields = ["split", "group", "K", "HR", "NDCG", "sex", "violence", "profanity", "drug", "intense", "unsafe_any", "adult_content", "users"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for split, metrics in metrics_by_split.items():
            for group in ["all", "minor", "adult"]:
                m = metrics[group]
                for k in topks:
                    w.writerow({
                        "split": split, "group": group, "K": k,
                        "HR": fmt(m[f"HR@{k}"], digits),
                        "NDCG": fmt(m[f"NDCG@{k}"], digits),
                        "sex": fmt(m[f"sex@{k}"], digits),
                        "violence": fmt(m[f"violence@{k}"], digits),
                        "profanity": fmt(m[f"profanity@{k}"], digits),
                        "drug": fmt(m[f"drug@{k}"], digits),
                        "intense": fmt(m[f"intense@{k}"], digits),
                        "unsafe_any": fmt(m[f"unsafe_any@{k}"], digits),
                        "adult_content": fmt(m[f"adult_content@{k}"], digits),
                        "users": m["users"],
                    })
    print(f"[write] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["gmf", "mlp", "neumf", "lightgcn"])
    ap.add_argument("--base_checkpoint", required=True)
    ap.add_argument("--psg_checkpoint", required=True)
    ap.add_argument("--base_graph_rating", default="")
    ap.add_argument("--psg_graph_rating", default="")
    ap.add_argument("--pred_jsonl", default="")
    ap.add_argument("--train_rating", required=True)
    ap.add_argument("--valid_rating", required=True)
    ap.add_argument("--test_rating", required=True)
    ap.add_argument("--item_safe", required=True)
    ap.add_argument("--user_tol", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--topks", default="1,5,10,20")
    ap.add_argument("--minor_block_at", type=float, default=3.0)
    ap.add_argument("--adult_block_at", type=float, default=4.0)
    ap.add_argument("--isadult_policy", choices=["minor_only", "all", "none"], default="minor_only")
    ap.add_argument("--exclude_valid_for_test", action="store_true")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--gate_max", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_users", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--calib_train_frac", type=float, default=0.8)
    ap.add_argument("--safety_weight", type=float, default=0.005)
    ap.add_argument("--safety_margin", type=float, default=0.05)
    ap.add_argument("--safety_trials", type=int, default=200)
    ap.add_argument("--monitor_k", type=int, default=10)
    ap.add_argument("--monitor_safety_weight", type=float, default=0.0)
    ap.add_argument("--score_batch_size", type=int, default=32768)
    ap.add_argument("--valid_every", type=int, default=5)
    ap.add_argument("--progress_every", type=int, default=200)
    ap.add_argument("--max_holdout_users", type=int, default=0, help="Limit holdout valid users for faster gate selection. 0 means use all holdout users.")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output_topk", action="store_true")
    ap.add_argument("--round_digits", type=int, default=5)
    args = ap.parse_args()

    args.device_resolved = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    topks = parse_topks(args.topks)
    os.makedirs(args.output_dir, exist_ok=True)

    train_by_user = read_rating_by_user(args.train_rating, "train")
    valid_by_user_all = read_rating_by_user(args.valid_rating, "valid")
    test_by_user = read_rating_by_user(args.test_rating, "test")
    item_safe, item_risk_cols = load_item_safe(args.item_safe)
    user_info, minor_col = load_user_info(args.user_tol)
    pred_map = read_pred_jsonl(args.pred_jsonl)

    base_scorer = FrozenScorer(args.backbone, args.base_checkpoint, args.device_resolved, args.base_graph_rating)
    psg_scorer = FrozenScorer(args.backbone, args.psg_checkpoint, args.device_resolved, args.psg_graph_rating)
    if base_scorer.num_users != psg_scorer.num_users or base_scorer.num_items != psg_scorer.num_items:
        print(f"[warn] base/psg shape mismatch: base=({base_scorer.num_users},{base_scorer.num_items}), psg=({psg_scorer.num_users},{psg_scorer.num_items}); using min dimensions")

    fb = FusionFeatureBuilder(base_scorer, psg_scorer, pred_map, item_safe, item_risk_cols, user_info, minor_col, args)
    gate = GateNet(fb.input_dim, hidden=args.hidden, dropout=args.dropout, gate_max=args.gate_max).to(args.device_resolved)

    valid_users = sorted([u for u, xs in valid_by_user_all.items() if xs])
    rng = random.Random(args.seed)
    rng.shuffle(valid_users)
    n_train = max(1, int(round(args.calib_train_frac * len(valid_users))))
    calib_users = valid_users[:n_train]
    holdout_users = valid_users[n_train:] or valid_users[-max(1, len(valid_users)//5):]
    if args.max_holdout_users and args.max_holdout_users > 0 and len(holdout_users) > args.max_holdout_users:
        holdout_users = holdout_users[:args.max_holdout_users]
        print(f"[fast] limit holdout_users to {len(holdout_users)}")

    log_rows, best_monitor = train_gate(gate, fb, calib_users, holdout_users, valid_by_user_all, train_by_user, topks, args)

    # save model and training log
    ckpt_path = os.path.join(args.output_dir, "backbone_gated_residual_best.pt")
    torch.save({"gate_state_dict": gate.state_dict(), "args": vars(args), "input_dim": fb.input_dim, "best_monitor": best_monitor}, ckpt_path)
    print(f"[write] {ckpt_path}")
    log_path = os.path.join(args.output_dir, "training_log.csv")
    if log_rows:
        fields = sorted(set(k for r in log_rows for k in r.keys()))
        with open(log_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(log_rows)
        print(f"[write] {log_path}")

    valid_metrics = evaluate(gate, fb, holdout_users, valid_by_user_all, train_by_user, {}, "valid", topks, args,
                             os.path.join(args.output_dir, "topk_valid.csv") if args.output_topk else "")
    test_users = sorted(test_by_user.keys())
    test_metrics = evaluate(gate, fb, test_users, test_by_user, train_by_user, valid_by_user_all, "test", topks, args,
                            os.path.join(args.output_dir, "topk_test.csv") if args.output_topk else "")

    metrics_by_split = {"valid": valid_metrics, "test": test_metrics}
    write_metrics_csv(os.path.join(args.output_dir, "matrix_final_scorer_metrics.csv"), metrics_by_split, topks, args.round_digits)
    with open(os.path.join(args.output_dir, "matrix_final_scorer_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "metrics": metrics_by_split}, f, ensure_ascii=False, indent=2)
    print(f"[write] {os.path.join(args.output_dir, 'matrix_final_scorer_metrics.json')}")


if __name__ == "__main__":
    main()
