#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SERVER FILE:
  scripts/llmsrec_mal/fuse_llmsrec_mal_cached_scores.py

Cached-score gated residual fusion for MAL / MyAnimeList LLM-SRec.

It reads:
  base/scores.npy + rhoXXX/scores.npy

It does NOT load Qwen and does NOT rerun LLM-SRec.

Fusion:
  s_final(u,i) = s_base(u,i) + g(u,i) * (s_aug(u,i) - s_base(u,i))

MAL Risk3 policy:
  minor: R17 / RPLUS / RX are unsafe
  adult: RPLUS / RX are unsafe; R17 is allowed

Reported safety metrics:
  R17, RPLUS, RX, unsafe_any,
  raw_R17_exposure, raw_RPLUS_exposure, raw_RX_exposure
"""

import argparse
import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


RISK_NAMES = [
    "R17",
    "RPLUS",
    "RX",
    "unsafe_any",
    "raw_R17_exposure",
    "raw_RPLUS_exposure",
    "raw_RX_exposure",
]


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
    vals = []
    for x in re.split(r"[,;\s]+", str(s).strip()):
        if x:
            vals.append(int(x))
    return sorted(set(k for k in vals if k > 0)) or [10]


def read_rating_by_user(path: str, name: str = "") -> Dict[int, List[int]]:
    by_user = defaultdict(list)
    rows = skipped = 0
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
    print(
        f"[load] {name or path}: rows={rows}, users={len(by_user)}, "
        f"skipped={skipped}, path={path}",
        flush=True,
    )
    return {int(u): list(v) for u, v in by_user.items()}


def unique_int_list(xs: Sequence[int]) -> List[int]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(int(x))
    return out


def read_csv_dict(path: str):
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = [
            {str(k).replace("\ufeff", "").strip(): v for k, v in row.items()}
            for row in reader
        ]
        fields = list(rows[0].keys()) if rows else [
            str(x).replace("\ufeff", "").strip()
            for x in (reader.fieldnames or [])
        ]
    return rows, fields


def find_col(
    fields: List[str],
    candidates: List[str],
    required: bool = False,
    label: str = "",
) -> str:
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
        raise ValueError(
            f"Cannot find column for {label or candidates}. Available={fields}"
        )
    return ""


def normalize_bucket(x: Any) -> str:
    s = str(x or "").strip()
    low = s.lower()
    compact = norm_name(s)

    if low in {"g", "g - all ages", "0"} or compact == "allages":
        return "G"
    if low in {"pg", "pg - children", "1"} or "pgchildren" in compact:
        return "PG"
    if "pg-13" in low or compact in {"pg13", "pg13teens13orolder"} or low == "2":
        return "PG13"
    if (
        low.startswith("r - 17")
        or low.startswith("r-17")
        or compact in {"r17", "r17violenceprofanity"}
        or "17+" in low
        or low == "3"
    ):
        return "R17"
    if (
        low.startswith("r+")
        or compact in {"rplus", "rplusmildnudity"}
        or "mild nudity" in low
        or low == "4"
    ):
        return "RPLUS"
    if low.startswith("rx") or compact in {"rx", "rxhentai", "hentai"} or low == "5":
        return "RX"
    return s.upper() if s else "UNKNOWN"


def load_item_buckets(path: str) -> Dict[int, str]:
    rows, fields = read_csv_dict(path)
    item_col = find_col(
        fields,
        ["inner_item_id", "item_id", "iid", "candidate_inner_item_id"],
        True,
        "item id",
    )
    bucket_col = find_col(
        fields,
        [
            "rating_bucket",
            "candidate_rating_bucket",
            "content_rating",
            "rating",
            "bucket",
        ],
        False,
        "rating bucket",
    )
    code_col = find_col(
        fields,
        ["rating_code", "content_rating_code", "bucket_code"],
        False,
        "rating code",
    )

    out = {}
    for row in rows:
        iid = safe_int(row.get(item_col), -1)
        if iid < 0:
            continue
        raw = row.get(bucket_col, "") if bucket_col else ""
        if not str(raw).strip() and code_col:
            raw = row.get(code_col, "")
        out[iid] = normalize_bucket(raw)

    print(
        f"[load] items rows={len(rows)}, mapped={len(out)}, "
        f"item_col={item_col}, bucket_col={bucket_col or 'N/A'}, "
        f"code_col={code_col or 'N/A'}, path={path}",
        flush=True,
    )
    return out


def load_user_minor(path: str) -> Dict[int, bool]:
    rows, fields = read_csv_dict(path)
    user_col = find_col(
        fields,
        ["inner_user_id", "user_id", "uid"],
        True,
        "user id",
    )
    minor_col = find_col(
        fields,
        ["is_minor", "minor", "isMinor", "under18", "is_under_18"],
        False,
        "minor flag",
    )
    age_col = find_col(
        fields,
        ["age_at_join", "age", "user_age"],
        False,
        "age",
    )

    out = {}
    for row in rows:
        uid = safe_int(row.get(user_col), -1)
        if uid < 0:
            continue
        if minor_col:
            out[uid] = parse_boolish(row.get(minor_col, ""))
        elif age_col:
            out[uid] = safe_float(row.get(age_col), 999.0) < 18.0
        else:
            out[uid] = False

    print(
        f"[load] users rows={len(rows)}, mapped={len(out)}, "
        f"user_col={user_col}, minor_col={minor_col or 'N/A'}, "
        f"age_col={age_col or 'N/A'}, path={path}",
        flush=True,
    )
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


def read_pred_jsonl(path: str) -> Dict[Tuple[int, int], Tuple[float, float, float]]:
    out = {}
    if not path or not os.path.exists(path):
        print(f"[warn] pred_jsonl missing: {path}; psg_pred/p1/margin use zeros")
        return out

    user_keys = [
        "user_id",
        "uid",
        "user",
        "inner_user_id",
        "candidate_inner_user_id",
    ]
    item_keys = [
        "item_id",
        "iid",
        "item",
        "inner_item_id",
        "candidate_inner_item_id",
        "candidate_item_id",
    ]

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

            meta = obj.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}

            u = i = -1
            for src in (obj, meta):
                for key in user_keys:
                    if key in src:
                        u = safe_int(src.get(key), -1)
                        break
                if u >= 0:
                    break

            for src in (obj, meta):
                for key in item_keys:
                    if key in src:
                        i = safe_int(src.get(key), -1)
                        break
                if i >= 0:
                    break

            if u < 0 or i < 0:
                continue

            psg_pred = safe_float(
                obj.get("psg_pred", obj.get("pred", obj.get("label", 0.0))),
                0.0,
            )
            p1 = safe_float(
                obj.get("p1", obj.get("prob_1", obj.get("prob1", 0.0))),
                0.0,
            )
            margin = safe_float(
                obj.get(
                    "margin_1_minus_0",
                    obj.get("margin", obj.get("logit_margin", 0.0)),
                ),
                0.0,
            )
            out[(u, i)] = (psg_pred, p1, margin)
            rows += 1

    print(f"[load] pred_jsonl rows={rows}, map_size={len(out)}, path={path}")
    return out


class GateNet(nn.Module):
    """
    Feature order:
      0 base_score
      1 aug_score
      2 delta
      3 psg_pred
      4 p1
      5 margin
      6 R17
      7 RPLUS
      8 RX
      9 unsafe_any
     10 is_minor
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden: int = 32,
        dropout: float = 0.10,
        gate_max: float = 1.0,
    ):
        super().__init__()
        self.gate_max = float(gate_max)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pref_head = nn.Linear(hidden, 1)
        self.conf_head = nn.Linear(hidden, 1)
        self.safe_head = nn.Linear(hidden, 1)

        nn.init.constant_(self.pref_head.bias, 1.0)
        nn.init.constant_(self.conf_head.bias, 1.0)
        nn.init.constant_(self.safe_head.bias, 1.2)

        self.conf_p1_scale = nn.Parameter(torch.tensor(1.0))
        self.conf_margin_scale = nn.Parameter(torch.tensor(0.10))
        self.safe_unsafe_scale = nn.Parameter(torch.tensor(0.8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        p1 = x[:, 4].clamp(0, 1)
        margin = x[:, 5]
        unsafe = x[:, 9].clamp(0, 1)

        g_pref = torch.sigmoid(self.pref_head(h).squeeze(-1))
        g_conf = torch.sigmoid(
            self.conf_head(h).squeeze(-1)
            + self.conf_p1_scale * (p1 - 0.5)
            + self.conf_margin_scale * margin
        )
        g_safe = torch.sigmoid(
            self.safe_head(h).squeeze(-1)
            - F.softplus(self.safe_unsafe_scale) * unsafe
        )
        return self.gate_max * g_pref * g_conf * g_safe


class CachedFeatureBuilder:
    def __init__(
        self,
        base_scores,
        aug_scores,
        pred_map,
        item_bucket,
        user_minor,
        args,
    ):
        self.base = base_scores
        self.aug = aug_scores
        self.pred_map = pred_map
        self.item_bucket = item_bucket
        self.user_minor = user_minor
        self.args = args
        self.num_users, self.num_items = base_scores.shape

    def build(self, users: List[int], items: List[int]):
        bs = np.array(
            [self.base[u, i] for u, i in zip(users, items)],
            dtype=np.float32,
        )
        ps = np.array(
            [self.aug[u, i] for u, i in zip(users, items)],
            dtype=np.float32,
        )

        feats = []
        for b, p, u, i in zip(bs, ps, users, items):
            pred = self.pred_map.get((int(u), int(i)), (0.0, 0.0, 0.0))
            is_minor = bool(self.user_minor.get(int(u), False))
            flags = risk3_flags(
                is_minor,
                self.item_bucket.get(int(i), "UNKNOWN"),
            )
            feats.append(
                [
                    float(b),
                    float(p),
                    float(p - b),
                    float(pred[0]),
                    float(pred[1]),
                    float(pred[2]),
                    float(flags["R17"]),
                    float(flags["RPLUS"]),
                    float(flags["RX"]),
                    float(flags["unsafe_any"]),
                    float(is_minor),
                ]
            )

        device = self.args.device_resolved
        return (
            torch.tensor(feats, dtype=torch.float32, device=device),
            torch.tensor(bs, dtype=torch.float32, device=device),
            torch.tensor(ps, dtype=torch.float32, device=device),
        )


def final_score(gate, feats, bs, ps, mode: str = "gated", alpha: float = 0.5):
    if mode == "fixed":
        g = torch.full_like(bs, float(alpha))
    else:
        g = gate(feats)
    return bs + g * (ps - bs), g


def sample_negative(num_items: int, block: set, rng: random.Random) -> int:
    for _ in range(500):
        iid = rng.randrange(num_items)
        if iid not in block:
            return iid
    return rng.randrange(num_items)


def init_acc(topks: List[int]):
    return {
        "users": 0,
        "hr_sum": {k: 0.0 for k in topks},
        "ndcg_sum": {k: 0.0 for k in topks},
        "top_items": {k: 0 for k in topks},
        "risk_items": {
            name: {k: 0 for k in topks}
            for name in RISK_NAMES
        },
    }


def update_acc(acc, topks, top_infos, pos_set):
    acc["users"] += 1
    for k in topks:
        topk = [row for row in top_infos if row["rank"] <= k]

        hit = 0.0
        dcg = 0.0
        for row in topk:
            if row["iid"] in pos_set:
                hit = 1.0
                dcg += 1.0 / math.log2(row["rank"] + 1.0)

        ideal_len = min(len(pos_set), k)
        idcg = (
            sum(
                1.0 / math.log2(rank + 1.0)
                for rank in range(1, ideal_len + 1)
            )
            if ideal_len > 0
            else 1.0
        )

        acc["hr_sum"][k] += hit
        acc["ndcg_sum"][k] += dcg / idcg if idcg > 0 else 0.0
        acc["top_items"][k] += len(topk)

        for name in RISK_NAMES:
            acc["risk_items"][name][k] += sum(
                int(row.get(name, 0))
                for row in topk
            )


def finalize(acc, topks):
    out = {"users": acc["users"]}
    users = max(acc["users"], 1)

    for k in topks:
        denom = max(acc["top_items"][k], 1)
        out[f"HR@{k}"] = acc["hr_sum"][k] / users
        out[f"NDCG@{k}"] = acc["ndcg_sum"][k] / users
        for name in RISK_NAMES:
            out[f"{name}@{k}"] = acc["risk_items"][name][k] / denom

    return out


def evaluate(
    gate,
    fb,
    users,
    eval_by_user,
    train_by_user,
    valid_by_user,
    split,
    topks,
    args,
    topk_csv="",
):
    max_k = max(topks)
    acc = {
        "all": init_acc(topks),
        "minor": init_acc(topks),
        "adult": init_acc(topks),
    }
    topk_rows = []

    if gate is not None:
        gate.eval()

    with torch.no_grad():
        for n_done, uid in enumerate(users, 1):
            if uid < 0 or uid >= fb.num_users:
                continue

            pos_items = [
                i
                for i in unique_int_list(eval_by_user.get(uid, []))
                if 0 <= i < fb.num_items
            ]
            if not pos_items:
                continue

            pos_set = set(pos_items)
            mask = np.ones(fb.num_items, dtype=bool)

            block = {
                i
                for i in train_by_user.get(uid, [])
                if 0 <= i < fb.num_items
            }
            if split == "test" and args.exclude_valid_for_test:
                block.update(
                    i
                    for i in valid_by_user.get(uid, [])
                    if 0 <= i < fb.num_items and i not in pos_set
                )
            block.difference_update(pos_set)

            if block:
                mask[np.fromiter(block, dtype=np.int64)] = False

            candidates = np.nonzero(mask)[0].astype(np.int64)
            user_ids = [uid] * len(candidates)
            item_ids = candidates.tolist()

            feats, bs, ps = fb.build(user_ids, item_ids)
            scores_t, gates_t = final_score(
                gate,
                feats,
                bs,
                ps,
                args.fusion_mode,
                args.alpha,
            )

            top_vals, top_idx = torch.topk(
                scores_t,
                k=min(max_k, scores_t.numel()),
                largest=True,
                sorted=True,
            )

            top_vals_np = top_vals.detach().cpu().numpy()
            top_idx_np = top_idx.detach().cpu().numpy()
            gates_np = gates_t.detach().cpu().numpy()

            is_minor = bool(fb.user_minor.get(int(uid), False))
            group = "minor" if is_minor else "adult"
            top_infos = []

            for rank, local_idx in enumerate(top_idx_np.tolist(), 1):
                iid = int(candidates[local_idx])
                score_val = float(top_vals_np[rank - 1])
                bucket = fb.item_bucket.get(iid, "UNKNOWN")
                flags = risk3_flags(is_minor, bucket)

                info = {
                    "rank": rank,
                    "iid": iid,
                    "score": score_val,
                    "bucket": bucket,
                    "is_positive": int(iid in pos_set),
                    **flags,
                }
                top_infos.append(info)

                if topk_csv:
                    topk_rows.append(
                        {
                            "split": split,
                            "group": group,
                            "user_id": uid,
                            "rank": rank,
                            "item_id": iid,
                            "score": f"{score_val:.8f}",
                            "gate": f"{float(gates_np[local_idx]):.8f}",
                            "bucket": bucket,
                            "is_positive": int(iid in pos_set),
                            **flags,
                        }
                    )

            update_acc(acc["all"], topks, top_infos, pos_set)
            update_acc(acc[group], topks, top_infos, pos_set)

            if n_done == 1 or n_done % args.progress_every == 0:
                m = finalize(acc["all"], topks)
                print(
                    f"[eval {split}] seen={n_done}, eval_users={m['users']}, "
                    f"HR@{args.monitor_k}="
                    f"{m.get(f'HR@{args.monitor_k}', 0.0):.5f}, "
                    f"NDCG@{args.monitor_k}="
                    f"{m.get(f'NDCG@{args.monitor_k}', 0.0):.5f}, "
                    f"unsafe_any@{args.monitor_k}="
                    f"{m.get(f'unsafe_any@{args.monitor_k}', 0.0):.5f}",
                    flush=True,
                )

    if topk_csv and topk_rows:
        with open(topk_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(topk_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(topk_rows)
        print(f"[write] {topk_csv}")

    return {
        group: finalize(group_acc, topks)
        for group, group_acc in acc.items()
    }


def train_gate(gate, fb, valid_by_user, train_by_user, topks, args):
    if args.fusion_mode == "fixed":
        return [], None

    rng = random.Random(args.seed)
    users = sorted(
        u
        for u, positives in valid_by_user.items()
        if positives and 0 <= u < fb.num_users
    )
    rng.shuffle(users)

    n_train = max(1, int(round(args.calib_train_frac * len(users))))
    calib_users = users[:n_train]
    holdout_users = users[n_train:]
    if not holdout_users:
        holdout_users = users[-max(1, len(users) // 5):]

    if args.max_holdout_users and len(holdout_users) > args.max_holdout_users:
        holdout_users = holdout_users[: args.max_holdout_users]

    optimizer = torch.optim.Adam(
        gate.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_metric = -1e18
    best_state = None
    logs = []

    print(
        f"[train gate] calib_users={len(calib_users)}, "
        f"holdout_users={len(holdout_users)}, num_items={fb.num_items}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(calib_users)
        gate.train()
        total_loss = 0.0
        batches = 0

        for start in range(0, len(calib_users), args.batch_users):
            pos_u, pos_i, neg_u, neg_i = [], [], [], []

            for uid in calib_users[start : start + args.batch_users]:
                pos_list = [
                    i
                    for i in valid_by_user.get(uid, [])
                    if 0 <= i < fb.num_items
                ]
                if not pos_list:
                    continue

                pos_item = rng.choice(pos_list)
                block = {
                    i
                    for i in train_by_user.get(uid, [])
                    if 0 <= i < fb.num_items
                }
                block.update(pos_list)
                neg_item = sample_negative(fb.num_items, block, rng)

                pos_u.append(uid)
                pos_i.append(pos_item)
                neg_u.append(uid)
                neg_i.append(neg_item)

            if not pos_u:
                continue

            pos_feats, pos_bs, pos_ps = fb.build(pos_u, pos_i)
            neg_feats, neg_bs, neg_ps = fb.build(neg_u, neg_i)

            pos_scores, _ = final_score(
                gate,
                pos_feats,
                pos_bs,
                pos_ps,
                "gated",
                args.alpha,
            )
            neg_scores, _ = final_score(
                gate,
                neg_feats,
                neg_bs,
                neg_ps,
                "gated",
                args.alpha,
            )

            loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            batches += 1

        row = {
            "epoch": epoch,
            "loss": total_loss / max(batches, 1),
        }

        if (
            epoch == 1
            or epoch % args.valid_every == 0
            or epoch == args.epochs
        ):
            metrics = evaluate(
                gate,
                fb,
                holdout_users,
                valid_by_user,
                train_by_user,
                {},
                "valid",
                topks,
                args,
            )
            all_metrics = metrics["all"]
            ndcg = all_metrics.get(
                f"NDCG@{args.monitor_k}",
                0.0,
            )
            unsafe = all_metrics.get(
                f"unsafe_any@{args.monitor_k}",
                0.0,
            )
            monitor = ndcg - args.monitor_safety_weight * unsafe

            row.update(
                {
                    "valid_NDCG": ndcg,
                    "valid_HR": all_metrics.get(
                        f"HR@{args.monitor_k}",
                        0.0,
                    ),
                    "valid_unsafe_any": unsafe,
                    "monitor": monitor,
                }
            )

            print(
                f"[epoch {epoch}] loss={row['loss']:.5f} "
                f"valid_NDCG@{args.monitor_k}={ndcg:.5f} "
                f"unsafe={unsafe:.5f} monitor={monitor:.5f}",
                flush=True,
            )

            if monitor > best_metric:
                best_metric = monitor
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in gate.state_dict().items()
                }
                print(
                    f"[best] epoch={epoch}, monitor={monitor:.6f}",
                    flush=True,
                )
        else:
            print(
                f"[epoch {epoch}] loss={row['loss']:.5f}",
                flush=True,
            )

        logs.append(row)

    if best_state is not None:
        gate.load_state_dict(best_state)

    return logs, best_metric


def write_metrics_csv(path, metrics_by_split, topks, digits=5):
    fields = [
        "split",
        "group",
        "K",
        "HR",
        "NDCG",
        "R17",
        "RPLUS",
        "RX",
        "unsafe_any",
        "raw_R17_exposure",
        "raw_RPLUS_exposure",
        "raw_RX_exposure",
        "users",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for split, metrics in metrics_by_split.items():
            for group in ["all", "minor", "adult"]:
                group_metrics = metrics[group]
                for k in topks:
                    row = {
                        "split": split,
                        "group": group,
                        "K": k,
                        "HR": f"{group_metrics[f'HR@{k}']:.{digits}f}",
                        "NDCG": f"{group_metrics[f'NDCG@{k}']:.{digits}f}",
                        "users": group_metrics["users"],
                    }
                    for name in RISK_NAMES:
                        row[name] = (
                            f"{group_metrics[f'{name}@{k}']:.{digits}f}"
                        )
                    writer.writerow(row)

    print(f"[write] {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_scores", required=True)
    parser.add_argument("--aug_scores", required=True)
    parser.add_argument("--pred_jsonl", default="")

    parser.add_argument("--train_rating", required=True)
    parser.add_argument("--valid_rating", required=True)
    parser.add_argument("--test_rating", required=True)
    parser.add_argument("--item_table", required=True)
    parser.add_argument("--user_table", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--fusion_mode",
        choices=["fixed", "gated"],
        default="gated",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gate_max", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_users", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--calib_train_frac", type=float, default=0.8)
    parser.add_argument("--valid_every", type=int, default=5)
    parser.add_argument("--max_holdout_users", type=int, default=0)

    parser.add_argument("--monitor_k", type=int, default=10)
    parser.add_argument(
        "--monitor_safety_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--topks", default="1,5,10,20")
    parser.add_argument(
        "--exclude_valid_for_test",
        action="store_true",
    )
    parser.add_argument("--output_topk", action="store_true")
    parser.add_argument("--round_digits", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=2028)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    args.device_resolved = (
        "cuda"
        if (
            args.device == "cuda"
            or (args.device == "auto" and torch.cuda.is_available())
        )
        else "cpu"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    topks = parse_topks(args.topks)

    train_by_user = read_rating_by_user(
        args.train_rating,
        "train",
    )
    valid_by_user = read_rating_by_user(
        args.valid_rating,
        "valid",
    )
    test_by_user = read_rating_by_user(
        args.test_rating,
        "test",
    )
    item_bucket = load_item_buckets(args.item_table)
    user_minor = load_user_minor(args.user_table)
    pred_map = read_pred_jsonl(args.pred_jsonl)

    base_scores = np.load(args.base_scores, mmap_mode="r")
    aug_scores = np.load(args.aug_scores, mmap_mode="r")

    if base_scores.shape != aug_scores.shape:
        raise ValueError(
            f"score shape mismatch: base={base_scores.shape}, "
            f"aug={aug_scores.shape}"
        )

    print(
        f"[scores] shape={base_scores.shape}, "
        f"base={args.base_scores}, aug={args.aug_scores}",
        flush=True,
    )

    feature_builder = CachedFeatureBuilder(
        base_scores,
        aug_scores,
        pred_map,
        item_bucket,
        user_minor,
        args,
    )

    gate = (
        None
        if args.fusion_mode == "fixed"
        else GateNet(
            input_dim=11,
            hidden=args.hidden,
            dropout=args.dropout,
            gate_max=args.gate_max,
        ).to(args.device_resolved)
    )

    logs, best_monitor = train_gate(
        gate,
        feature_builder,
        valid_by_user,
        train_by_user,
        topks,
        args,
    )

    if gate is not None:
        gate_path = os.path.join(
            args.output_dir,
            "llmsrec_mal_cached_gate_best.pt",
        )
        torch.save(
            {
                "gate_state_dict": gate.state_dict(),
                "args": vars(args),
                "best_monitor": best_monitor,
            },
            gate_path,
        )
        print(f"[write] {gate_path}")

    if logs:
        log_path = os.path.join(
            args.output_dir,
            "training_log.csv",
        )
        fields = sorted(
            set(key for row in logs for key in row.keys())
        )
        with open(log_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(logs)
        print(f"[write] {log_path}")

    valid_metrics = evaluate(
        gate,
        feature_builder,
        sorted(valid_by_user.keys()),
        valid_by_user,
        train_by_user,
        {},
        "valid",
        topks,
        args,
        os.path.join(args.output_dir, "topk_valid.csv")
        if args.output_topk
        else "",
    )

    test_metrics = evaluate(
        gate,
        feature_builder,
        sorted(test_by_user.keys()),
        test_by_user,
        train_by_user,
        valid_by_user,
        "test",
        topks,
        args,
        os.path.join(args.output_dir, "topk_test.csv")
        if args.output_topk
        else "",
    )

    metrics = {
        "valid": valid_metrics,
        "test": test_metrics,
    }

    metrics_csv = os.path.join(
        args.output_dir,
        "fusion_metrics.csv",
    )
    write_metrics_csv(
        metrics_csv,
        metrics,
        topks,
        args.round_digits,
    )

    metrics_json = os.path.join(
        args.output_dir,
        "fusion_metrics.json",
    )
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "metrics": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[write] {metrics_json}")


if __name__ == "__main__":
    main()
