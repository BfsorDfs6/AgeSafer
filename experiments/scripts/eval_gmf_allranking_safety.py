#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERSION: PSG-Dim5-v4-eval-five-dim
FILE NAME ON SERVER: scripts/eval_gmf_allranking_safety.py

Purpose
-------
All-ranking GMF evaluator using the new five-dimensional metrics instead of old
BC1/BC2. It reports HR/NDCG plus:
  sex@K, violence@K, profanity@K, drug@K, intense@K, any_unsafe@K, adult_content@K

Common command
--------------
cd .
CUDA_VISIBLE_DEVICES=3 python -u scripts/eval_gmf_allranking_safety.py \
  --checkpoint <GMF_CKPT> \
  --data_dir Data \
  --dataset ml-1m_safe \
  --split test \
  --train_rating Data/ml-1m_safe.train.rating \
  --valid_rating Data/ml-1m_safe.valid.rating \
  --item_safe outputs/psg/safe_features/ml-1m_safe_features.item_safe.csv \
  --user_tol outputs/psg/safe_features/ml-1m_safe_features.user_tolerance_p75.csv \
  --topks 1,5,10,20 \
  --exclude_valid_for_test \
  --isadult_policy minor_only \
  --minor_block_at 3 \
  --adult_block_at 4 \
  --output_csv <OUT>.csv
"""
"""
eval_gmf_dim5_allranking_safety.py

All-ranking evaluator for GMF checkpoints under the v2 age-aware five-dimension rule.

Metrics:
  HR@K, NDCG@K
  sex@K, violence@K, profanity@K, drug@K, intense@K

Dimension rule:
  minor unsafe if item_dim >= --minor_block_at  (default 3)
  adult unsafe if item_dim >= --adult_block_at  (default 4)
  isAdult is only reported as adult_content@K and can optionally be hard-blocked for minors,
  but it is not mixed into the five IMDb dimensions.

Candidate pool:
  all checkpoint items - train positives [- valid positives for test if requested]
"""
import argparse, csv, json, math, os, re
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple
import torch

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
            by_user[u].append(i); rows += 1
    print(f"[load] {name}: rows={rows}, users={len(by_user)}, skipped={skipped}, path={path}")
    return dict(by_user)

def unique_int_list(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def read_csv_dict(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(4096); f.seek(0)
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

def unwrap_checkpoint(obj):
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "module"]:
            if key in obj and isinstance(obj[key], dict):
                return unwrap_checkpoint(obj[key])
        if any(torch.is_tensor(v) for v in obj.values()):
            return obj
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise ValueError("Cannot find state_dict in checkpoint")

def strip_prefix(k):
    for p in ["module.", "model.", "net."]:
        if k.startswith(p): return k[len(p):]
    return k

def normalize_state_dict(sd):
    return {strip_prefix(k): v.detach().cpu() for k, v in sd.items() if torch.is_tensor(v)}

def find_embedding_keys(sd):
    emb_keys = [k for k, v in sd.items() if v.ndim == 2 and "weight" in k.lower()]
    user_patterns = ["user_embedding", "embedding_user", "embed_user_gmf", "user_emb", "embed_user", "user"]
    item_patterns = ["item_embedding", "embedding_item", "embed_item_gmf", "item_emb", "embed_item", "item"]
    def score(k, pats):
        kl = k.lower(); best = 0
        for idx, p in enumerate(pats):
            if p in kl: best = max(best, 100 - idx)
        return best
    u_cands = sorted([(score(k, user_patterns), k) for k in emb_keys], reverse=True)
    i_cands = sorted([(score(k, item_patterns), k) for k in emb_keys], reverse=True)
    if u_cands and i_cands and u_cands[0][0] > 0 and i_cands[0][0] > 0 and u_cands[0][1] != i_cands[0][1]:
        return u_cands[0][1], i_cands[0][1]
    raise ValueError(f"Cannot detect user/item embeddings. Candidate keys={emb_keys[:30]}")

def find_linear_predictor(sd, dim):
    candidates = []
    for k, v in sd.items():
        if v.ndim == 2 and v.shape[0] == 1 and v.shape[1] == dim:
            priority = sum(10 for p in ["output", "predict", "affine", "fc", "linear"] if p in k.lower())
            candidates.append((priority, k, v))
    if not candidates:
        return None, None, "dot_product_no_linear"
    candidates.sort(reverse=True)
    _, wk, w = candidates[0]
    prefix = wk.rsplit(".", 1)[0] if "." in wk else wk.replace("weight", "")
    bias = None
    for bk, bv in sd.items():
        if bk == prefix + ".bias" and bv.ndim == 1 and bv.numel() == 1:
            bias = bv; break
    return w.reshape(-1).float(), (bias.reshape(()) if bias is not None else torch.tensor(0.0)), wk

def load_gmf_weights(checkpoint, device):
    print(f"[load] checkpoint={checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")
    sd = normalize_state_dict(unwrap_checkpoint(ckpt))
    uk, ik = find_embedding_keys(sd)
    user_emb, item_emb = sd[uk].float(), sd[ik].float()
    if user_emb.shape[1] != item_emb.shape[1]:
        raise ValueError(f"Embedding dim mismatch: {uk}={user_emb.shape}, {ik}={item_emb.shape}")
    pred_w, pred_b, pred_key = find_linear_predictor(sd, user_emb.shape[1])
    print(f"[detect] user_embedding={uk}, shape={tuple(user_emb.shape)}")
    print(f"[detect] item_embedding={ik}, shape={tuple(item_emb.shape)}")
    print(f"[detect] predictor={pred_key}")
    return user_emb.to(device), item_emb.to(device), (pred_w.to(device) if pred_w is not None else None), (pred_b.to(device) if pred_b is not None else None)

@torch.no_grad()
def score_candidates(uid, item_ids, user_emb, item_emb, pred_w, pred_b, device, batch_size):
    u = user_emb[int(uid)]
    out = []
    for start in range(0, len(item_ids), batch_size):
        batch = item_ids[start:start+batch_size]
        idx = torch.tensor(batch, dtype=torch.long, device=device)
        v = item_emb[idx]
        if pred_w is not None:
            s = (v * u * pred_w).sum(dim=1)
            if pred_b is not None: s = s + pred_b
        else:
            s = (v * u).sum(dim=1)
        out.extend(s.detach().cpu().tolist())
    return out

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
    return vios, any_unsafe, adult_content, int(not row_i), is_minor, values

def init_acc(topks):
    return {
        "users": 0,
        "hr_sum": {k:0.0 for k in topks},
        "ndcg_sum": {k:0.0 for k in topks},
        "top_items": {k:0 for k in topks},
        "dim_items": {d:{k:0 for k in topks} for d in ["sex","violence","profanity","drug","intense"]},
        "any_items": {k:0 for k in topks},
        "adult_items": {k:0 for k in topks},
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
        idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_len+1)) if ideal_len > 0 else 1.0
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
        for d in ["sex","violence","profanity","drug","intense"]:
            out[f"{d}@{k}"] = acc["dim_items"][d][k] / denom
        out[f"unsafe_any@{k}"] = acc["any_items"][k] / denom
        out[f"adult_content@{k}"] = acc["adult_items"][k] / denom
    return out

def round_tree(obj, digits):
    if isinstance(obj, float): return round(obj, digits)
    if isinstance(obj, dict): return {k: round_tree(v, digits) for k,v in obj.items()}
    if isinstance(obj, list): return [round_tree(v, digits) for v in obj]
    return obj

def fmt(x, digits): return f"{float(x):.{digits}f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="Data")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", choices=["valid","test"], default="test")
    ap.add_argument("--train_rating", default="")
    ap.add_argument("--valid_rating", default="")
    ap.add_argument("--item_safe", required=True)
    ap.add_argument("--user_info", default="", help="CSV containing user id and is_minor. If omitted, --user_tol is used.")
    ap.add_argument("--user_tol", default="", help="Backward-compatible alias for --user_info.")
    ap.add_argument("--topks", default="1,5,10,20")
    ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"])
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

    topks = parse_topks(args.topks); max_topk = max(topks)
    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    print(f"[info] device={device}, topks={topks}, minor_block_at={args.minor_block_at}, adult_block_at={args.adult_block_at}")

    train_rating = args.train_rating or os.path.join(args.data_dir, f"{args.dataset}.train.rating")
    eval_rating = os.path.join(args.data_dir, f"{args.dataset}.{args.split}.rating")
    valid_rating = args.valid_rating or os.path.join(args.data_dir, f"{args.dataset}.valid.rating")
    user_info_path = args.user_info or args.user_tol
    if not user_info_path:
        raise ValueError("Need --user_info or --user_tol to detect is_minor")

    train_by_user = read_rating_by_user(train_rating, "train")
    eval_by_user = read_rating_by_user(eval_rating, args.split)
    valid_by_user = read_rating_by_user(valid_rating, "valid") if (args.split == "test" and args.exclude_valid_for_test and os.path.exists(valid_rating)) else {}
    item_safe, item_risk_cols = load_item_safe(args.item_safe)
    user_info, minor_col = load_user_info(user_info_path)
    user_emb, item_emb, pred_w, pred_b = load_gmf_weights(args.checkpoint, device)
    num_users, num_items = user_emb.shape[0], item_emb.shape[0]
    all_items = list(range(num_items))
    users = sorted(eval_by_user.keys())
    if args.max_users > 0: users = users[:args.max_users]

    acc = {"all": init_acc(topks), "minor": init_acc(topks), "adult": init_acc(topks)}
    topk_rows = []
    for n_done, uid in enumerate(users, 1):
        if uid < 0 or uid >= num_users: continue
        pos_items = [i for i in unique_int_list(eval_by_user.get(uid, [])) if 0 <= i < num_items]
        if not pos_items: continue
        pos_set = set(pos_items)
        block = set(i for i in train_by_user.get(uid, []) if 0 <= i < num_items)
        if args.split == "test" and args.exclude_valid_for_test:
            block.update(i for i in valid_by_user.get(uid, []) if 0 <= i < num_items and i not in pos_set)
        candidates = [i for i in all_items if i not in block]
        cset = set(candidates)
        for p in pos_items:
            if p not in cset:
                candidates.append(p); cset.add(p)
        scores = score_candidates(uid, candidates, user_emb, item_emb, pred_w, pred_b, device, args.score_batch_size)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:max_topk]
        group = "minor" if is_minor_user(uid, user_info, minor_col) else "adult"
        top_infos = []
        for rank, (iid, score) in enumerate(ranked, 1):
            vios, any_unsafe, adult_content, unk_i, is_minor, values = dim_violations(
                uid, iid, item_safe, item_risk_cols, user_info, minor_col,
                args.minor_block_at, args.adult_block_at, args.isadult_policy)
            info = {"rank": rank, "iid": int(iid), "score": float(score), "is_positive": int(iid in pos_set),
                    "unsafe_any": any_unsafe, "adult_content": adult_content, **vios}
            top_infos.append(info)
            if args.output_topk_csv:
                row = {"group": group, "user_id": uid, "rank": rank, "item_id": iid, "score": f"{float(score):.8f}",
                       "is_positive": int(iid in pos_set), "unsafe_any": any_unsafe, "adult_content": adult_content,
                       "unknown_item_safe": unk_i}
                row.update(vios)
                for dim, val in values.items(): row[f"{dim}_item"] = val
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
    print("[done]"); print(json.dumps(rounded, ensure_ascii=False, indent=2))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f: json.dump(rounded, f, ensure_ascii=False, indent=2)
        print(f"[write] {args.output_json}")
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        fields = ["group", "K", "HR", "NDCG", "sex", "violence", "profanity", "drug", "intense", "unsafe_any", "adult_content", "users"]
        with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for g in ["all", "minor", "adult"]:
                for k in topks:
                    m = metrics[g]
                    w.writerow({"group": g, "K": k, "HR": fmt(m[f"HR@{k}"], args.round_digits), "NDCG": fmt(m[f"NDCG@{k}"], args.round_digits),
                                "sex": fmt(m[f"sex@{k}"], args.round_digits), "violence": fmt(m[f"violence@{k}"], args.round_digits),
                                "profanity": fmt(m[f"profanity@{k}"], args.round_digits), "drug": fmt(m[f"drug@{k}"], args.round_digits),
                                "intense": fmt(m[f"intense@{k}"], args.round_digits), "unsafe_any": fmt(m[f"unsafe_any@{k}"], args.round_digits),
                                "adult_content": fmt(m[f"adult_content@{k}"], args.round_digits), "users": m["users"]})
        print(f"[write] {args.output_csv}")
    if args.output_topk_csv:
        os.makedirs(os.path.dirname(args.output_topk_csv) or ".", exist_ok=True)
        if topk_rows:
            with open(args.output_topk_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(topk_rows[0].keys())); w.writeheader(); w.writerows(topk_rows)
        print(f"[write] {args.output_topk_csv}")

if __name__ == "__main__":
    main()
