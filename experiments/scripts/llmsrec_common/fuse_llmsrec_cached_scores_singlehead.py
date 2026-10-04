#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SERVER FILE
-----------
scripts/llmsrec_common/fuse_llmsrec_cached_scores_singlehead.py

Single-head MLP gated residual fusion for cached LLM-SRec scores.

This version intentionally matches the conventional-backbone fusion form:

  gate = gate_max * sigmoid(MLP(features))
  s_final = s_base + gate * (s_aug - s_base)

There are no preference/confidence/safety output heads.

Training objective:
  L = L_rank + safety_weight * L_safe

  L_rank = -log sigmoid(s_positive - s_negative)
  L_safe = relu(s_unsafe - s_safe + safety_margin)

The best epoch is selected on a held-out subset of validation users:

  monitor = NDCG@monitor_k
            - monitor_safety_weight * unsafe_any@monitor_k

This script reads cached base/aug scores.npy files and never loads Qwen.
"""

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ML_DIMS = ["sex", "violence", "profanity", "drug", "intense"]
MAL_RISKS = [
    "R17",
    "RPLUS",
    "RX",
    "unsafe_any",
    "raw_R17_exposure",
    "raw_RPLUS_exposure",
    "raw_RX_exposure",
]


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


def norm_name(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower().replace("\ufeff", ""))


def parse_boolish(x: Any) -> bool:
    s = str(x).strip().lower()
    if s in {"1", "1.0", "true", "yes", "y", "minor", "under18", "under_18"}:
        return True
    if s in {"0", "0.0", "false", "no", "n", "adult"}:
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
    return (not parts) or safe_int(parts[0], -999999) == -999999


def parse_topks(value: str) -> List[int]:
    vals = []
    for x in re.split(r"[,;\s]+", str(value).strip()):
        if x:
            vals.append(int(x))
    return sorted(set(k for k in vals if k > 0)) or [10]


def unique_int_list(xs: Sequence[int]) -> List[int]:
    out, seen = [], set()
    for x in xs:
        x = int(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def read_rating_by_user(path: str, name: str) -> Dict[int, List[int]]:
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
            uid = safe_int(parts[0])
            iid = safe_int(parts[1])
            if uid < 0 or iid < 0:
                skipped += 1
                continue
            by_user[uid].append(iid)
            rows += 1
    print(
        f"[load] {name}: rows={rows}, users={len(by_user)}, "
        f"skipped={skipped}, path={path}",
        flush=True,
    )
    return {int(u): list(v) for u, v in by_user.items()}


def read_csv_rows(path: str):
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        fields = [
            str(x).replace("\ufeff", "").strip()
            for x in (reader.fieldnames or [])
        ]
        rows = []
        for raw in reader:
            rows.append({
                str(k).replace("\ufeff", "").strip(): v
                for k, v in raw.items()
            })
    return rows, fields


def find_col(
    fields: Sequence[str],
    candidates: Sequence[str],
    required: bool = False,
    label: str = "",
) -> str:
    norm_to_real = {norm_name(f): f for f in fields}
    for candidate in candidates:
        key = norm_name(candidate)
        if key in norm_to_real:
            return norm_to_real[key]
    for candidate in candidates:
        key = norm_name(candidate)
        for nf, real in norm_to_real.items():
            if key and key in nf:
                return real
    if required:
        raise ValueError(
            f"Cannot find column for {label or candidates}. Available={list(fields)}"
        )
    return ""


def load_user_minor(path: str) -> Dict[int, bool]:
    rows, fields = read_csv_rows(path)
    uid_col = find_col(
        fields,
        ["inner_user_id", "user_inner_id", "user_id", "uid", "user"],
        True,
        "user id",
    )
    minor_col = find_col(
        fields,
        ["is_minor", "minor", "isMinor", "user_is_minor", "under18", "is_under_18"],
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
        uid = safe_int(row.get(uid_col))
        if uid < 0:
            continue
        if minor_col:
            out[uid] = parse_boolish(row.get(minor_col, ""))
        elif age_col:
            out[uid] = safe_float(row.get(age_col), 999.0) < 18.0
        else:
            out[uid] = False

    print(
        f"[load] users={len(out)}, minor={sum(out.values())}, "
        f"adult={len(out)-sum(out.values())}, path={path}",
        flush=True,
    )
    return out


def load_ml_item_arrays(path: str, num_items: int):
    rows, fields = read_csv_rows(path)
    iid_col = find_col(
        fields,
        ["inner_item_id", "item_inner_id", "item_id", "iid", "movie_id"],
        True,
        "item id",
    )

    risk_cols = {}
    for dim in ML_DIMS:
        risk_cols[dim] = find_col(
            fields,
            [
                f"{dim}_code",
                dim,
                f"{dim}_level",
                f"{dim}_risk",
                f"{dim}_score",
                f"item_{dim}",
                f"risk_{dim}",
            ],
            False,
            dim,
        )
    adult_col = find_col(
        fields,
        ["isAdult", "is_adult", "adult_content", "adult", "isAdult_code"],
        False,
        "adult content",
    )

    risk = np.zeros((num_items, len(ML_DIMS)), dtype=np.float32)
    adult = np.zeros(num_items, dtype=np.float32)
    for row in rows:
        iid = safe_int(row.get(iid_col))
        if iid < 0 or iid >= num_items:
            continue
        for idx, dim in enumerate(ML_DIMS):
            col = risk_cols[dim]
            risk[iid, idx] = safe_float(row.get(col), 0.0) if col else 0.0
        adult[iid] = safe_float(row.get(adult_col), 0.0) if adult_col else 0.0

    print(
        f"[load] ML item safety: iid_col={iid_col}, "
        f"risk_cols={risk_cols}, adult_col={adult_col or 'N/A'}, path={path}",
        flush=True,
    )
    return risk, adult


def bucket_from_row(row: Dict[str, Any]) -> str:
    bucket = str(row.get("rating_bucket", "")).strip()
    code = safe_int(row.get("rating_code"), -1)
    if bucket:
        compact = norm_name(bucket)
        if compact in {"r17", "r17violenceprofanity"}:
            return "R17"
        if compact in {"rplus", "rplusmildnudity"}:
            return "RPLUS"
        if compact in {"rx", "rxhentai", "hentai"}:
            return "RX"
        if compact.startswith("pg13"):
            return "PG13"
        if compact.startswith("pg"):
            return "PG"
        if compact in {"g", "allages"}:
            return "G"

    code_map = {0: "G", 1: "PG", 2: "PG13", 3: "R17", 4: "RPLUS", 5: "RX"}
    if code in code_map:
        return code_map[code]

    text = " ".join(
        str(row.get(k, ""))
        for k in ["rating_bucket", "rating", "rating_name", "content_rating"]
    )
    upper = text.upper().replace(" ", "")
    if "RX" in upper or "HENTAI" in upper:
        return "RX"
    if "R+" in upper or "RPLUS" in upper or "MILDNUDITY" in upper:
        return "RPLUS"
    if "R-17" in upper or "R17" in upper or "17+" in upper:
        return "R17"
    if "PG-13" in upper or "PG13" in upper:
        return "PG13"
    if upper.startswith("PG") or "CHILDREN" in upper:
        return "PG"
    if upper.startswith("G") or "ALLAGES" in upper:
        return "G"
    return "OTHER"


def load_mal_item_buckets(path: str, num_items: int) -> np.ndarray:
    rows, fields = read_csv_rows(path)
    iid_col = find_col(
        fields,
        ["inner_item_id", "item_inner_id", "item_id", "iid", "item"],
        True,
        "item id",
    )
    buckets = np.full(num_items, "OTHER", dtype=object)
    counts = Counter()
    for row in rows:
        iid = safe_int(row.get(iid_col))
        if iid < 0 or iid >= num_items:
            continue
        bucket = bucket_from_row(row)
        buckets[iid] = bucket
        counts[bucket] += 1
    print(
        f"[load] MAL buckets: iid_col={iid_col}, "
        f"counts={dict(counts)}, path={path}",
        flush=True,
    )
    return buckets


def read_pred_jsonl(path: str) -> Dict[Tuple[int, int], Tuple[float, float, float]]:
    out = {}
    if not path or not os.path.isfile(path):
        print(
            f"[warn] pred_jsonl missing: {path}; "
            "psg_pred/p1/margin features use zeros",
            flush=True,
        )
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
            metadata = obj.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            uid = iid = -1
            for source in (obj, metadata):
                for key in user_keys:
                    if key in source:
                        uid = safe_int(source.get(key))
                        break
                if uid >= 0:
                    break
            for source in (obj, metadata):
                for key in item_keys:
                    if key in source:
                        iid = safe_int(source.get(key))
                        break
                if iid >= 0:
                    break

            if uid < 0 or iid < 0:
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
            out[(uid, iid)] = (psg_pred, p1, margin)
            rows += 1

    print(f"[load] pred_jsonl rows={rows}, unique_pairs={len(out)}, path={path}")
    return out


class SafetyAdapter:
    def __init__(self, args, num_items: int):
        self.dataset_type = args.dataset_type
        self.user_minor = load_user_minor(args.user_info)
        self.num_items = num_items

        if self.dataset_type == "ml1m_dim5":
            if not args.item_safe:
                raise ValueError("--item_safe is required for ml1m_dim5")
            self.ml_risk, self.ml_adult = load_ml_item_arrays(
                args.item_safe,
                num_items,
            )
        else:
            item_table = args.item_table or args.item_safe
            if not item_table:
                raise ValueError("--item_table is required for mal_risk3")
            self.mal_bucket = load_mal_item_buckets(item_table, num_items)

    def is_minor(self, uid: int) -> bool:
        return bool(self.user_minor.get(int(uid), False))

    def flags(self, uid: int, iid: int, args) -> Dict[str, int]:
        is_minor = self.is_minor(uid)

        if self.dataset_type == "ml1m_dim5":
            threshold = (
                float(args.minor_block_at)
                if is_minor
                else float(args.adult_block_at)
            )
            vals = self.ml_risk[int(iid)]
            vios = {
                dim: int(vals[idx] >= threshold)
                for idx, dim in enumerate(ML_DIMS)
            }
            adult_content = int(self.ml_adult[int(iid)] >= 1.0)
            adult_hard = int(
                args.isadult_policy in {"minor_only", "all"}
                and adult_content
                and (is_minor or args.isadult_policy == "all")
            )
            return {
                **vios,
                "unsafe_any": int(any(vios.values()) or adult_hard),
                "adult_content": adult_content,
            }

        bucket = str(self.mal_bucket[int(iid)]).upper()
        r17 = int(is_minor and bucket == "R17")
        rplus = int(bucket == "RPLUS")
        rx = int(bucket == "RX")
        return {
            "R17": r17,
            "RPLUS": rplus,
            "RX": rx,
            "unsafe_any": int(r17 or rplus or rx),
            "raw_R17_exposure": int(bucket == "R17"),
            "raw_RPLUS_exposure": int(bucket == "RPLUS"),
            "raw_RX_exposure": int(bucket == "RX"),
        }

    def feature_names(self) -> List[str]:
        if self.dataset_type == "ml1m_dim5":
            return ML_DIMS + ["unsafe_any", "adult_content", "is_minor"]
        return ["R17", "RPLUS", "RX", "unsafe_any", "is_minor"]

    def metric_names(self) -> List[str]:
        if self.dataset_type == "ml1m_dim5":
            return ML_DIMS + ["unsafe_any", "adult_content"]
        return MAL_RISKS

    def is_unsafe(self, uid: int, iid: int, args) -> bool:
        return bool(self.flags(uid, iid, args)["unsafe_any"])


class FeatureBuilder:
    def __init__(
        self,
        base_scores,
        aug_scores,
        pred_map,
        adapter: SafetyAdapter,
        args,
    ):
        self.base = base_scores
        self.aug = aug_scores
        self.pred_map = pred_map
        self.adapter = adapter
        self.args = args
        self.num_users, self.num_items = base_scores.shape

        self.input_dim = (
            6 + len(adapter.feature_names())
        )

    def build(
        self,
        users: Sequence[int],
        items: Sequence[int],
    ):
        bs_np = np.fromiter(
            (self.base[int(u), int(i)] for u, i in zip(users, items)),
            dtype=np.float32,
            count=len(users),
        )
        aug_np = np.fromiter(
            (self.aug[int(u), int(i)] for u, i in zip(users, items)),
            dtype=np.float32,
            count=len(users),
        )

        features = []
        for idx, (uid, iid) in enumerate(zip(users, items)):
            uid = int(uid)
            iid = int(iid)
            pred = self.pred_map.get((uid, iid), (0.0, 0.0, 0.0))
            flags = self.adapter.flags(uid, iid, self.args)

            row = [
                float(bs_np[idx]),
                float(aug_np[idx]),
                float(aug_np[idx] - bs_np[idx]),
                float(pred[0]),
                float(pred[1]),
                float(pred[2]),
            ]

            if self.args.dataset_type == "ml1m_dim5":
                row.extend(float(flags[name]) for name in ML_DIMS)
                row.extend([
                    float(flags["unsafe_any"]),
                    float(flags["adult_content"]),
                    float(self.adapter.is_minor(uid)),
                ])
            else:
                row.extend([
                    float(flags["R17"]),
                    float(flags["RPLUS"]),
                    float(flags["RX"]),
                    float(flags["unsafe_any"]),
                    float(self.adapter.is_minor(uid)),
                ])
            features.append(row)

        device = self.args.device_resolved
        return (
            torch.tensor(features, dtype=torch.float32, device=device),
            torch.tensor(bs_np, dtype=torch.float32, device=device),
            torch.tensor(aug_np, dtype=torch.float32, device=device),
        )


class GateNet(nn.Module):
    """Single-head MLP gate, matching the conventional backbone fusion."""

    def __init__(
        self,
        input_dim: int,
        hidden: int = 32,
        dropout: float = 0.10,
        gate_max: float = 1.0,
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate_max * torch.sigmoid(
            self.net(x).squeeze(-1)
        )


def final_score(gate, features, base_score, aug_score):
    gate_value = gate(features)
    score = base_score + gate_value * (aug_score - base_score)
    return score, gate_value


def sample_negative(
    num_items: int,
    blocked: set,
    rng: random.Random,
    max_trials: int = 300,
) -> int:
    for _ in range(max_trials):
        iid = rng.randrange(num_items)
        if iid not in blocked:
            return iid
    for iid in range(num_items):
        if iid not in blocked:
            return iid
    return 0


def sample_safe_unsafe(
    uid: int,
    num_items: int,
    blocked: set,
    adapter: SafetyAdapter,
    args,
    rng: random.Random,
):
    safe_iid = None
    unsafe_iid = None
    for _ in range(args.safety_trials):
        iid = rng.randrange(num_items)
        if iid in blocked:
            continue
        if adapter.is_unsafe(uid, iid, args):
            if unsafe_iid is None:
                unsafe_iid = iid
        else:
            if safe_iid is None:
                safe_iid = iid
        if safe_iid is not None and unsafe_iid is not None:
            break
    return safe_iid, unsafe_iid


def init_acc(topks: List[int], metric_names: List[str]):
    return {
        "users": 0,
        "hr_sum": {k: 0.0 for k in topks},
        "ndcg_sum": {k: 0.0 for k in topks},
        "top_items": {k: 0 for k in topks},
        "metric_items": {
            name: {k: 0 for k in topks}
            for name in metric_names
        },
    }


def update_acc(
    acc,
    topks,
    ranked_rows,
    pos_set,
    metric_names,
):
    acc["users"] += 1
    for k in topks:
        rows = ranked_rows[:k]
        hit = 0.0
        dcg = 0.0
        for row in rows:
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
        acc["top_items"][k] += len(rows)

        for name in metric_names:
            acc["metric_items"][name][k] += sum(
                int(row.get(name, 0))
                for row in rows
            )


def finalize_acc(acc, topks, metric_names):
    users = max(acc["users"], 1)
    out = {"users": acc["users"]}
    for k in topks:
        denom = max(acc["top_items"][k], 1)
        out[f"HR@{k}"] = acc["hr_sum"][k] / users
        out[f"NDCG@{k}"] = acc["ndcg_sum"][k] / users
        for name in metric_names:
            out[f"{name}@{k}"] = acc["metric_items"][name][k] / denom
    return out


def evaluate(
    gate,
    feature_builder: FeatureBuilder,
    users: Sequence[int],
    eval_by_user,
    train_by_user,
    valid_by_user,
    split: str,
    topks,
    args,
    topk_path: str = "",
):
    gate.eval()
    metric_names = feature_builder.adapter.metric_names()
    groups = {
        "all": init_acc(topks, metric_names),
        "minor": init_acc(topks, metric_names),
        "adult": init_acc(topks, metric_names),
    }
    max_k = max(topks)
    topk_rows = []

    with torch.no_grad():
        for seen, uid in enumerate(users, 1):
            uid = int(uid)
            if uid < 0 or uid >= feature_builder.num_users:
                continue

            positives = [
                iid
                for iid in unique_int_list(eval_by_user.get(uid, []))
                if 0 <= iid < feature_builder.num_items
            ]
            if not positives:
                continue
            pos_set = set(positives)

            mask = np.ones(feature_builder.num_items, dtype=bool)
            blocked = {
                iid
                for iid in train_by_user.get(uid, [])
                if 0 <= iid < feature_builder.num_items
            }
            if split == "test" and args.exclude_valid_for_test:
                blocked.update(
                    iid
                    for iid in valid_by_user.get(uid, [])
                    if 0 <= iid < feature_builder.num_items
                )
            blocked.difference_update(pos_set)
            if blocked:
                mask[np.fromiter(blocked, dtype=np.int64)] = False

            candidates = np.nonzero(mask)[0]
            if candidates.size == 0:
                continue

            users_batch = [uid] * len(candidates)
            item_batch = candidates.tolist()
            features, base_score, aug_score = feature_builder.build(
                users_batch,
                item_batch,
            )
            scores, gates = final_score(
                gate,
                features,
                base_score,
                aug_score,
            )

            take = min(max_k, scores.numel())
            top_values, top_local = torch.topk(
                scores,
                k=take,
                largest=True,
                sorted=True,
            )
            top_values_np = top_values.detach().cpu().numpy()
            top_local_np = top_local.detach().cpu().numpy()
            gates_np = gates.detach().cpu().numpy()

            is_minor = feature_builder.adapter.is_minor(uid)
            group = "minor" if is_minor else "adult"
            ranked_rows = []

            for rank, local_idx in enumerate(top_local_np.tolist(), 1):
                iid = int(candidates[local_idx])
                flags = feature_builder.adapter.flags(uid, iid, args)
                row = {
                    "rank": rank,
                    "iid": iid,
                    "score": float(top_values_np[rank - 1]),
                    "gate": float(gates_np[local_idx]),
                    "is_positive": int(iid in pos_set),
                    **flags,
                }
                ranked_rows.append(row)

                if args.output_topk:
                    topk_rows.append({
                        "split": split,
                        "group": group,
                        "user_id": uid,
                        "rank": rank,
                        "item_id": iid,
                        "score": f"{row['score']:.8f}",
                        "gate": f"{row['gate']:.8f}",
                        "is_positive": row["is_positive"],
                        **flags,
                    })

            update_acc(
                groups["all"],
                topks,
                ranked_rows,
                pos_set,
                metric_names,
            )
            update_acc(
                groups[group],
                topks,
                ranked_rows,
                pos_set,
                metric_names,
            )

            if seen == 1 or seen % args.progress_every == 0:
                now = finalize_acc(
                    groups["all"],
                    topks,
                    metric_names,
                )
                print(
                    f"[eval {split}] seen={seen}, eval_users={now['users']}, "
                    f"HR@{args.monitor_k}="
                    f"{now.get(f'HR@{args.monitor_k}', 0):.5f}, "
                    f"NDCG@{args.monitor_k}="
                    f"{now.get(f'NDCG@{args.monitor_k}', 0):.5f}, "
                    f"unsafe_any@{args.monitor_k}="
                    f"{now.get(f'unsafe_any@{args.monitor_k}', 0):.5f}",
                    flush=True,
                )

    if args.output_topk:
        fields = [
            "split",
            "group",
            "user_id",
            "rank",
            "item_id",
            "score",
            "gate",
            "is_positive",
        ] + metric_names
        with open(topk_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(topk_rows)
        print(f"[write] {topk_path}", flush=True)

    return {
        group: finalize_acc(acc, topks, metric_names)
        for group, acc in groups.items()
    }


def train_gate(
    gate,
    feature_builder,
    valid_by_user,
    train_by_user,
    topks,
    args,
):
    rng = random.Random(args.seed)
    users = sorted(
        uid
        for uid, positives in valid_by_user.items()
        if positives and 0 <= uid < feature_builder.num_users
    )
    rng.shuffle(users)

    n_train = max(1, int(round(args.calib_train_frac * len(users))))
    train_users = users[:n_train]
    holdout_users = users[n_train:]
    if not holdout_users:
        holdout_users = users[-max(1, len(users) // 5):]

    if args.max_holdout_users > 0:
        holdout_users = holdout_users[: args.max_holdout_users]

    optimizer = torch.optim.Adam(
        gate.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_monitor = -1e18
    best_epoch = -1
    best_state = None
    logs = []

    print(
        f"[train gate] train_users={len(train_users)}, "
        f"holdout_users={len(holdout_users)}, "
        f"input_dim={feature_builder.input_dim}, "
        f"gate=single_head_mlp, safety_weight={args.safety_weight}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_users)
        gate.train()

        sum_loss = 0.0
        sum_rank = 0.0
        sum_safe = 0.0
        batches = 0

        for start in range(0, len(train_users), args.batch_users):
            batch_users = train_users[start : start + args.batch_users]

            pos_u, pos_i = [], []
            neg_u, neg_i = [], []
            safe_u, safe_i = [], []
            unsafe_u, unsafe_i = [], []

            for uid in batch_users:
                positives = [
                    iid
                    for iid in valid_by_user.get(uid, [])
                    if 0 <= iid < feature_builder.num_items
                ]
                if not positives:
                    continue

                positive = rng.choice(positives)
                blocked = {
                    iid
                    for iid in train_by_user.get(uid, [])
                    if 0 <= iid < feature_builder.num_items
                }
                blocked.update(positives)

                negative = sample_negative(
                    feature_builder.num_items,
                    blocked,
                    rng,
                )
                pos_u.append(uid)
                pos_i.append(positive)
                neg_u.append(uid)
                neg_i.append(negative)

                if args.safety_weight > 0:
                    safe_item, unsafe_item = sample_safe_unsafe(
                        uid,
                        feature_builder.num_items,
                        blocked,
                        feature_builder.adapter,
                        args,
                        rng,
                    )
                    if safe_item is not None and unsafe_item is not None:
                        safe_u.append(uid)
                        safe_i.append(safe_item)
                        unsafe_u.append(uid)
                        unsafe_i.append(unsafe_item)

            if not pos_u:
                continue

            pos_features, pos_base, pos_aug = feature_builder.build(
                pos_u,
                pos_i,
            )
            neg_features, neg_base, neg_aug = feature_builder.build(
                neg_u,
                neg_i,
            )
            pos_score, _ = final_score(
                gate,
                pos_features,
                pos_base,
                pos_aug,
            )
            neg_score, _ = final_score(
                gate,
                neg_features,
                neg_base,
                neg_aug,
            )

            rank_loss = -F.logsigmoid(pos_score - neg_score).mean()
            safe_loss = torch.zeros(
                (),
                dtype=torch.float32,
                device=args.device_resolved,
            )

            if args.safety_weight > 0 and safe_u:
                safe_features, safe_base, safe_aug = feature_builder.build(
                    safe_u,
                    safe_i,
                )
                unsafe_features, unsafe_base, unsafe_aug = feature_builder.build(
                    unsafe_u,
                    unsafe_i,
                )
                safe_score, _ = final_score(
                    gate,
                    safe_features,
                    safe_base,
                    safe_aug,
                )
                unsafe_score, _ = final_score(
                    gate,
                    unsafe_features,
                    unsafe_base,
                    unsafe_aug,
                )
                safe_loss = F.relu(
                    unsafe_score - safe_score + args.safety_margin
                ).mean()

            loss = rank_loss + args.safety_weight * safe_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            optimizer.step()

            sum_loss += float(loss.detach().cpu())
            sum_rank += float(rank_loss.detach().cpu())
            sum_safe += float(safe_loss.detach().cpu())
            batches += 1

        row = {
            "epoch": epoch,
            "loss": sum_loss / max(batches, 1),
            "rank_loss": sum_rank / max(batches, 1),
            "safety_loss": sum_safe / max(batches, 1),
        }

        if (
            epoch == 1
            or epoch % args.valid_every == 0
            or epoch == args.epochs
        ):
            holdout_metrics = evaluate(
                gate,
                feature_builder,
                holdout_users,
                valid_by_user,
                train_by_user,
                {},
                "valid",
                topks,
                args,
            )
            all_metrics = holdout_metrics["all"]
            ndcg = all_metrics.get(
                f"NDCG@{args.monitor_k}",
                0.0,
            )
            hr = all_metrics.get(
                f"HR@{args.monitor_k}",
                0.0,
            )
            unsafe = all_metrics.get(
                f"unsafe_any@{args.monitor_k}",
                0.0,
            )
            monitor = ndcg - args.monitor_safety_weight * unsafe
            row.update({
                "valid_HR": hr,
                "valid_NDCG": ndcg,
                "valid_unsafe_any": unsafe,
                "monitor": monitor,
            })

            print(
                f"[epoch {epoch}] loss={row['loss']:.6f}, "
                f"rank={row['rank_loss']:.6f}, "
                f"safe={row['safety_loss']:.6f}, "
                f"valid_HR@{args.monitor_k}={hr:.5f}, "
                f"valid_NDCG@{args.monitor_k}={ndcg:.5f}, "
                f"unsafe={unsafe:.5f}, monitor={monitor:.5f}",
                flush=True,
            )

            if monitor > best_monitor:
                best_monitor = monitor
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in gate.state_dict().items()
                }
                print(
                    f"[best] epoch={best_epoch}, monitor={best_monitor:.6f}",
                    flush=True,
                )
        else:
            print(
                f"[epoch {epoch}] loss={row['loss']:.6f}, "
                f"rank={row['rank_loss']:.6f}, "
                f"safe={row['safety_loss']:.6f}",
                flush=True,
            )

        logs.append(row)

    if best_state is None:
        raise RuntimeError("No best gate state was selected")
    gate.load_state_dict(best_state)
    return logs, best_epoch, best_monitor


def write_metrics_csv(
    path,
    metrics,
    topks,
    metric_names,
    digits,
):
    fields = [
        "split",
        "group",
        "K",
        "HR",
        "NDCG",
        *metric_names,
        "users",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split, split_metrics in metrics.items():
            for group in ["all", "minor", "adult"]:
                m = split_metrics[group]
                for k in topks:
                    row = {
                        "split": split,
                        "group": group,
                        "K": k,
                        "HR": f"{m[f'HR@{k}']:.{digits}f}",
                        "NDCG": f"{m[f'NDCG@{k}']:.{digits}f}",
                        "users": m["users"],
                    }
                    for name in metric_names:
                        row[name] = f"{m[f'{name}@{k}']:.{digits}f}"
                    writer.writerow(row)
    print(f"[write] {path}", flush=True)


def validate_mal_consistency(metrics, topks):
    for split in metrics:
        for group in metrics[split]:
            m = metrics[split][group]
            for k in topks:
                if m[f"R17@{k}"] > m[f"raw_R17_exposure@{k}"] + 1e-12:
                    raise AssertionError("R17 exceeds raw_R17_exposure")
                if max(
                    m[f"R17@{k}"],
                    m[f"RPLUS@{k}"],
                    m[f"RX@{k}"],
                ) > m[f"unsafe_any@{k}"] + 1e-12:
                    raise AssertionError("Risk component exceeds unsafe_any")
    print("[check] MAL fusion metric consistency passed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_type",
        choices=["ml1m_dim5", "mal_risk3"],
        required=True,
    )
    ap.add_argument("--base_scores", required=True)
    ap.add_argument("--aug_scores", required=True)
    ap.add_argument("--pred_jsonl", default="")
    ap.add_argument("--train_rating", required=True)
    ap.add_argument("--valid_rating", required=True)
    ap.add_argument("--test_rating", required=True)
    ap.add_argument("--user_info", required=True)
    ap.add_argument("--item_safe", default="")
    ap.add_argument("--item_table", default="")
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--topks", default="1,5,10,20")
    ap.add_argument("--exclude_valid_for_test", action="store_true")
    ap.add_argument("--output_topk", action="store_true")

    ap.add_argument("--minor_block_at", type=float, default=3.0)
    ap.add_argument("--adult_block_at", type=float, default=4.0)
    ap.add_argument(
        "--isadult_policy",
        choices=["minor_only", "all", "none"],
        default="minor_only",
    )

    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--gate_max", type=float, default=0.10)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_users", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--calib_train_frac", type=float, default=0.8)

    ap.add_argument("--safety_weight", type=float, default=0.0)
    ap.add_argument("--safety_margin", type=float, default=0.05)
    ap.add_argument("--safety_trials", type=int, default=200)

    ap.add_argument("--monitor_k", type=int, default=10)
    ap.add_argument("--monitor_safety_weight", type=float, default=0.0)
    ap.add_argument("--valid_every", type=int, default=5)
    ap.add_argument("--max_holdout_users", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=200)

    ap.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--round_digits", type=int, default=5)
    args = ap.parse_args()

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

    base_scores = np.load(args.base_scores, mmap_mode="r")
    aug_scores = np.load(args.aug_scores, mmap_mode="r")
    if base_scores.shape != aug_scores.shape:
        raise ValueError(
            f"score shape mismatch: base={base_scores.shape}, "
            f"aug={aug_scores.shape}"
        )
    if base_scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got {base_scores.shape}")

    print(
        f"[scores] shape={base_scores.shape}, "
        f"base={args.base_scores}, aug={args.aug_scores}",
        flush=True,
    )

    train_by_user = read_rating_by_user(args.train_rating, "train")
    valid_by_user = read_rating_by_user(args.valid_rating, "valid")
    test_by_user = read_rating_by_user(args.test_rating, "test")

    adapter = SafetyAdapter(args, int(base_scores.shape[1]))
    pred_map = read_pred_jsonl(args.pred_jsonl)
    feature_builder = FeatureBuilder(
        base_scores,
        aug_scores,
        pred_map,
        adapter,
        args,
    )

    gate = GateNet(
        input_dim=feature_builder.input_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        gate_max=args.gate_max,
    ).to(args.device_resolved)

    logs, best_epoch, best_monitor = train_gate(
        gate,
        feature_builder,
        valid_by_user,
        train_by_user,
        topks,
        args,
    )

    torch.save(
        {
            "gate_type": "single_head_mlp",
            "gate_state_dict": gate.state_dict(),
            "input_dim": feature_builder.input_dim,
            "best_epoch": best_epoch,
            "best_monitor": best_monitor,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "llmsrec_singlehead_gate_best.pt"),
    )

    log_fields = sorted({
        key
        for row in logs
        for key in row.keys()
    })
    with open(
        os.path.join(args.output_dir, "training_log.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        writer.writerows(logs)

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
        os.path.join(args.output_dir, "topk_valid.csv"),
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
        os.path.join(args.output_dir, "topk_test.csv"),
    )
    metrics = {
        "valid": valid_metrics,
        "test": test_metrics,
    }

    if args.dataset_type == "mal_risk3":
        validate_mal_consistency(metrics, topks)

    metric_names = adapter.metric_names()
    write_metrics_csv(
        os.path.join(args.output_dir, "fusion_metrics.csv"),
        metrics,
        topks,
        metric_names,
        args.round_digits,
    )

    with open(
        os.path.join(args.output_dir, "fusion_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "gate_type": "single_head_mlp",
                "best_epoch": best_epoch,
                "best_monitor": best_monitor,
                "args": vars(args),
                "metrics": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[done] output_dir={args.output_dir}, "
        f"gate_type=single_head_mlp, best_epoch={best_epoch}",
        flush=True,
    )


if __name__ == "__main__":
    main()
