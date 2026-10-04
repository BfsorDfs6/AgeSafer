#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERSION: PSG-Dim5-v4-eval-five-dim
All-ranking evaluator for checkpoints saved by lightgcn_torch_valid_allranking.py.
Reports the same Dim5-v4 metrics as eval_gmf_allranking_safety.py.

Important for LightGCN:
  - test candidate masking uses --train_rating, normally original Data/ml-1m_safe.train.rating
  - graph propagation uses checkpoint train_edges if present;
    otherwise it uses --graph_rating if provided, else --train_rating.
For PSG no-fusion checkpoints, pass --graph_rating to the augmented train file if
train_edges are not stored in the checkpoint.
"""
import argparse, csv, json, math, os, re
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

RISK_DIMS = ["sex_code", "violence_code", "profanity_code", "drug_code", "intense_code"]
DIM_OUT = {
    "sex_code": "sex",
    "violence_code": "violence",
    "profanity_code": "profanity",
    "drug_code": "drug",
    "intense_code": "intense",
}
ADULT_DIM = "isAdult"


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
    by_user: Dict[int, List[int]] = defaultdict(list)
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
    return dict(by_user)


def read_train_pairs(path: str) -> List[Tuple[int, int]]:
    out = []
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
                out.append((u, i))
    print(f"[load] graph train pairs: rows={len(out)}, path={path}")
    return out


def unique_int_list(xs: Sequence[int]) -> List[int]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def read_csv_dict(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
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


def find_col(fields: List[str], candidates: List[str], required: bool = False, label: str = "") -> str:
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


def find_item_id_col(fields: List[str]) -> str:
    return find_col(fields, ["inner_item_id", "item_inner_id", "item_id", "iid", "movie_id", "candidate_inner_item_id"], True, "item id")


def find_user_id_col(fields: List[str]) -> str:
    return find_col(fields, ["inner_user_id", "user_inner_id", "user_id", "uid"], True, "user id")


def find_item_risk_cols(fields: List[str]) -> Dict[str, str]:
    out = {}
    for dim in RISK_DIMS + [ADULT_DIM]:
        short = dim.replace("_code", "")
        names = [dim, dim.lower(), short, short + "_code", short + "_level", short + "_risk", short + "_score", "item_" + dim, "risk_" + dim]
        out[dim] = find_col(fields, names, False, f"item risk {dim}")
    return out


def find_minor_col(fields: List[str]) -> str:
    return find_col(fields, ["is_minor", "minor", "isMinor", "user_is_minor", "under18", "is_under_18"], False, "minor flag")


def load_item_safe(path: str):
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


def load_user_info(path: str):
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


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64, n_layers: int = 3):
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
        users, items = torch.split(out, [self.num_users, self.num_items], dim=0)
        return users, items


def build_norm_adj(num_users: int, num_items: int, train_pairs: Sequence[Tuple[int, int]], device: str):
    n_nodes = num_users + num_items
    rows, cols = [], []
    for u, i in train_pairs:
        if 0 <= u < num_users and 0 <= i < num_items:
            rows.append(u)
            cols.append(num_users + i)
            rows.append(num_users + i)
            cols.append(u)
    if not rows:
        raise ValueError("No graph edges for LightGCN propagation.")
    rows_t = torch.tensor(rows, dtype=torch.long)
    cols_t = torch.tensor(cols, dtype=torch.long)
    deg = torch.zeros(n_nodes, dtype=torch.float32)
    deg.index_add_(0, rows_t, torch.ones_like(rows_t, dtype=torch.float32))
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    vals = deg_inv_sqrt[rows_t] * deg_inv_sqrt[cols_t]
    idx = torch.stack([rows_t, cols_t], dim=0)
    return torch.sparse_coo_tensor(idx, vals, (n_nodes, n_nodes)).coalesce().to(device)


@torch.no_grad()
def score_candidates(user_g, item_g, uid: int, item_ids: List[int], device: str, batch_size: int) -> List[float]:
    uemb = user_g[int(uid)]
    out = []
    for start in range(0, len(item_ids), batch_size):
        batch = item_ids[start:start + batch_size]
        idx = torch.tensor(batch, dtype=torch.long, device=device)
        scores = (item_g[idx] * uemb).sum(dim=-1)
        out.extend(scores.detach().cpu().tolist())
    return out


def is_minor_user(uid: int, user_info: Dict[int, Dict[str, str]], minor_col: str) -> bool:
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
    return vios, any_unsafe, adult_content, int(not row_i), is_minor, values


def init_acc(topks):
    return {
        "users": 0,
        "hr_sum": {k: 0.0 for k in topks},
        "ndcg_sum": {k: 0.0 for k in topks},
        "top_items": {k: 0 for k in topks},
        "dim_items": {d: {k: 0 for k in topks} for d in ["sex", "violence", "profanity", "drug", "intense"]},
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
        for d in acc["dim_items"]:
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
        for d in ["sex", "violence", "profanity", "drug", "intense"]:
            out[f"{d}@{k}"] = acc["dim_items"][d][k] / denom
        out[f"unsafe_any@{k}"] = acc["any_items"][k] / denom
        out[f"adult_content@{k}"] = acc["adult_items"][k] / denom
    return out


def round_tree(obj, digits):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_tree(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_tree(v, digits) for v in obj]
    return obj


def fmt(x, digits):
    return f"{float(x):.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="Data")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", choices=["valid", "test"], default="test")
    ap.add_argument("--train_rating", default="", help="Original train.rating used only for candidate masking.")
    ap.add_argument("--valid_rating", default="")
    ap.add_argument("--graph_rating", default="", help="Fallback graph rating for LightGCN propagation if checkpoint has no train_edges.")
    ap.add_argument("--item_safe", required=True)
    ap.add_argument("--user_info", default="", help="CSV containing user id and is_minor. If omitted, --user_tol is used.")
    ap.add_argument("--user_tol", default="", help="Backward-compatible alias for --user_info.")
    ap.add_argument("--topks", default="1,5,10,20")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--score_batch_size", type=int, default=8192)
    ap.add_argument("--max_users", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=100)
    ap.add_argument("--exclude_valid_for_test", action="store_true")
    ap.add_argument("--isadult_policy", choices=["minor_only", "all", "none"], default="minor_only")
    ap.add_argument("--minor_block_at", type=float, default=3.0)
    ap.add_argument("--adult_block_at", type=float, default=4.0)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--output_csv", default="")
    ap.add_argument("--output_topk_csv", default="")
    ap.add_argument("--round_digits", type=int, default=5)
    args = ap.parse_args()

    topks = parse_topks(args.topks)
    max_topk = max(topks)
    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    print(f"[info] device={device}, topks={topks}, minor_block_at={args.minor_block_at}, adult_block_at={args.adult_block_at}")

    train_rating = args.train_rating or os.path.join(args.data_dir, f"{args.dataset}.train.rating")
    eval_rating = os.path.join(args.data_dir, f"{args.dataset}.{args.split}.rating")
    valid_rating = args.valid_rating or os.path.join(args.data_dir, f"{args.dataset}.valid.rating")
    user_info_path = args.user_info or args.user_tol
    if not user_info_path:
        raise ValueError("Need --user_info or --user_tol to detect is_minor.")

    train_by_user = read_rating_by_user(train_rating, "train-mask")
    eval_by_user = read_rating_by_user(eval_rating, args.split)
    valid_by_user = read_rating_by_user(valid_rating, "valid") if (args.split == "test" and args.exclude_valid_for_test and os.path.exists(valid_rating)) else {}
    item_safe, item_risk_cols = load_item_safe(args.item_safe)
    user_info, minor_col = load_user_info(user_info_path)

    print(f"[load] checkpoint={args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    num_users = int(ckpt["num_users"])
    num_items = int(ckpt["num_items"])
    emb_dim = int(ckpt.get("emb_dim", ckpt.get("args", {}).get("emb_dim", 64)))
    layers = int(ckpt.get("layers", ckpt.get("args", {}).get("layers", 3)))
    state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if state is None:
        raise ValueError("Cannot find model_state_dict/state_dict in checkpoint.")
    model = LightGCN(num_users, num_items, emb_dim=emb_dim, n_layers=layers)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    train_edges = ckpt.get("train_edges", None)
    if train_edges is not None:
        print(f"[graph] use train_edges stored in checkpoint, edges={len(train_edges)}")
    else:
        graph_rating = args.graph_rating or train_rating
        print(f"[graph] checkpoint has no train_edges; use graph_rating={graph_rating}")
        train_edges = read_train_pairs(graph_rating)

    norm_adj = build_norm_adj(num_users, num_items, train_edges, device)
    with torch.no_grad():
        user_g, item_g = model.propagate(norm_adj)
    all_items = list(range(num_items))
    print(f"[info] model=LightGCN, users={num_users}, items={num_items}, emb_dim={emb_dim}, layers={layers}")
    print("[info] candidate_pool=all_items - original train positives" + (" - valid positives" if args.split == "test" and args.exclude_valid_for_test else ""))

    users = sorted(eval_by_user.keys())
    if args.max_users > 0:
        users = users[: args.max_users]

    acc = {"all": init_acc(topks), "minor": init_acc(topks), "adult": init_acc(topks)}
    topk_rows = []

    for n_done, uid in enumerate(users, start=1):
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
        cand_set = set(candidates)
        for p in pos_items:
            if p not in cand_set:
                candidates.append(p)
                cand_set.add(p)
        scores = score_candidates(user_g, item_g, uid, candidates, device=device, batch_size=args.score_batch_size)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:max_topk]
        group = "minor" if is_minor_user(uid, user_info, minor_col) else "adult"
        top_infos = []
        for rank, (iid, score) in enumerate(ranked, start=1):
            vios, any_unsafe, adult_content, unk_i, _is_minor, values = dim_violations(
                uid, iid, item_safe, item_risk_cols, user_info, minor_col,
                args.minor_block_at, args.adult_block_at, args.isadult_policy
            )
            info = {"rank": rank, "iid": int(iid), "score": float(score), "is_positive": int(iid in pos_set),
                    "unsafe_any": any_unsafe, "adult_content": adult_content, **vios}
            top_infos.append(info)
            if args.output_topk_csv:
                row = {"group": group, "user_id": uid, "rank": rank, "item_id": iid, "iid": iid,
                       "score": f"{float(score):.8f}", "is_positive": int(iid in pos_set),
                       "unsafe_any": any_unsafe, "adult_content": adult_content, "unknown_item_safe": unk_i}
                row.update(vios)
                for dim, val in values.items():
                    row[f"{dim}_item"] = val
                topk_rows.append(row)
        update_acc(acc["all"], topks, top_infos, pos_set)
        update_acc(acc[group], topks, top_infos, pos_set)
        if n_done == 1 or n_done % args.progress_every == 0:
            m = finalize(acc["all"], topks)
            msg = [f"[progress] users_seen={n_done}, eval_users={acc['all']['users']}, uid={uid}, group={group}, candidates={len(candidates)}"]
            for k in topks:
                msg.append(f"HR@{k}={m[f'HR@{k}']:.4f} NDCG@{k}={m[f'NDCG@{k}']:.4f} sex@{k}={m[f'sex@{k}']:.4f} violence@{k}={m[f'violence@{k}']:.4f}")
            print(" | ".join(msg), flush=True)

    metrics = {g: finalize(a, topks) for g, a in acc.items()}
    result = {"checkpoint": args.checkpoint, "dataset": args.dataset, "split": args.split, "topks": topks,
              "candidate_policy": "all_items_minus_train" + ("_minus_valid" if args.split == "test" and args.exclude_valid_for_test else ""),
              "minor_block_at": args.minor_block_at, "adult_block_at": args.adult_block_at,
              "isadult_policy": args.isadult_policy, "metrics": metrics}
    rounded = round_tree(result, args.round_digits)
    print("[done]")
    print(json.dumps(rounded, ensure_ascii=False, indent=2))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(rounded, f, ensure_ascii=False, indent=2)
        print(f"[write] {args.output_json}")
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        fields = ["group", "K", "HR", "NDCG", "sex", "violence", "profanity", "drug", "intense", "unsafe_any", "adult_content", "users"]
        with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for g in ["all", "minor", "adult"]:
                for k in topks:
                    m = metrics[g]
                    w.writerow({"group": g, "K": k,
                                "HR": fmt(m[f"HR@{k}"], args.round_digits),
                                "NDCG": fmt(m[f"NDCG@{k}"], args.round_digits),
                                "sex": fmt(m[f"sex@{k}"], args.round_digits),
                                "violence": fmt(m[f"violence@{k}"], args.round_digits),
                                "profanity": fmt(m[f"profanity@{k}"], args.round_digits),
                                "drug": fmt(m[f"drug@{k}"], args.round_digits),
                                "intense": fmt(m[f"intense@{k}"], args.round_digits),
                                "unsafe_any": fmt(m[f"unsafe_any@{k}"], args.round_digits),
                                "adult_content": fmt(m[f"adult_content@{k}"], args.round_digits),
                                "users": m["users"]})
        print(f"[write] {args.output_csv}")
    if args.output_topk_csv:
        os.makedirs(os.path.dirname(args.output_topk_csv) or ".", exist_ok=True)
        if topk_rows:
            with open(args.output_topk_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(topk_rows[0].keys()))
                w.writeheader()
                w.writerows(topk_rows)
        print(f"[write] {args.output_topk_csv}")


if __name__ == "__main__":
    main()
