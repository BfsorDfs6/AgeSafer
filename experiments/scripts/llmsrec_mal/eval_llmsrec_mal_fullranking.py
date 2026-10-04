#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: scripts/llmsrec_mal/eval_llmsrec_mal_fullranking.py

Load one trained LLM-SRec-Qwen25 checkpoint for MAL/MyAnimeList, export full
all-item scores, and evaluate all-ranking HR/NDCG + risk3 exposure.

Protocol:
  - User/item embeddings are generated once.
  - scores.npy is a dense float32 matrix: [num_users, num_items].
  - Candidate pool for evaluation blocks ONLY original train interactions.
  - For test, valid items are also blocked when --exclude_valid_for_test is set.
  - Augmented LLM-SRec models use augmented train history internally, but their
    injected pseudo items are NOT removed from the evaluation candidate pool.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


RISK_NAMES = ["R17", "RPLUS", "RX"]


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
    print(f"[load] {name or path}: rows={rows}, users={len(by_user)}, skipped={skipped}, path={path}")
    return {int(u): list(v) for u, v in by_user.items()}


def read_llmsrec_split(path: str) -> Dict[int, List[int]]:
    d = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            sp = line.strip().split()
            if len(sp) >= 2:
                d[int(sp[0])].append(int(sp[1]))
    return {int(u): list(v) for u, v in d.items()}


def unique_int_list(xs: Sequence[int]) -> List[int]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(int(x))
    return out


def read_csv_dict(path: str):
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
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


def normalize_rating_bucket(x: Any) -> str:
    s = str(x).strip()
    if not s:
        return "UNKNOWN"
    n = norm_name(s)

    # Most reliable explicit buckets.
    if n in {"rx", "hentai"}:
        return "RX"
    if n in {"rplus", "r18plus", "r18", "rp", "r"}:
        return "RPLUS"
    if n in {"r17", "r17plus", "r17violenceprofanity"}:
        return "R17"
    if n in {"pg13", "pg13teens13orolder"}:
        return "PG13"
    if n in {"pg", "pgchildren"}:
        return "PG"
    if n in {"g", "allages"}:
        return "G"

    # Rating string examples:
    # "R - 17+ (violence & profanity)", "R+ - Mild Nudity",
    # "Rx - Hentai", "PG-13 - Teens 13 or older"
    low = s.lower()
    if "rx" in low or "hentai" in low:
        return "RX"
    if "r+" in low or "mild nudity" in low:
        return "RPLUS"
    if "r - 17" in low or "r-17" in low or "17+" in low:
        return "R17"
    if "pg-13" in low or "teens 13" in low:
        return "PG13"
    if "pg" in low:
        return "PG"
    if low == "g" or "all ages" in low:
        return "G"
    return s.upper().replace("-", "").replace("+", "PLUS").replace(" ", "")


def load_items(path: str):
    rows, fields = read_csv_dict(path)
    item_col = find_col(fields, ["inner_item_id", "item_id", "iid"], required=True, label="inner item id")
    bucket_col = find_col(fields, ["rating_bucket", "risk_bucket", "bucket", "rating_code", "rating"], required=False, label="rating bucket")
    code_col = find_col(fields, ["rating_code", "risk_code"], required=False, label="rating code")

    item_bucket = {}
    item_code = {}
    for r in rows:
        iid = safe_int(r.get(item_col), -1)
        if iid < 0:
            continue
        raw_bucket = r.get(bucket_col, "") if bucket_col else ""
        bucket = normalize_rating_bucket(raw_bucket)
        item_bucket[iid] = bucket
        item_code[iid] = safe_float(r.get(code_col), 0.0) if code_col else 0.0

    print(f"[load] items rows={len(rows)}, mapped_items={len(item_bucket)}, path={path}")
    print(f"[detect] item_col={item_col}, bucket_col={bucket_col or 'N/A'}, code_col={code_col or 'N/A'}")
    return item_bucket, item_code


def load_users(path: str):
    rows, fields = read_csv_dict(path)
    user_col = find_col(fields, ["inner_user_id", "user_inner_id", "user_id", "uid"], required=True, label="inner user id")
    minor_col = find_col(fields, ["is_minor", "minor", "isMinor", "user_is_minor", "under18", "is_under_18"], required=False, label="minor")
    age_col = find_col(fields, ["age", "user_age"], required=False, label="age")
    user_info = {}
    for r in rows:
        uid = safe_int(r.get(user_col), -1)
        if uid >= 0:
            user_info[uid] = r
    print(f"[load] users rows={len(rows)}, mapped_users={len(user_info)}, path={path}")
    print(f"[detect] user_col={user_col}, minor_col={minor_col or 'N/A'}, age_col={age_col or 'N/A'}")
    return user_info, minor_col, age_col


def is_minor_user(uid: int, user_info: Dict[int, Dict[str, Any]], minor_col: str, age_col: str) -> bool:
    row = user_info.get(int(uid), {})
    if minor_col:
        return parse_boolish(row.get(minor_col, ""))
    if age_col:
        age = safe_float(row.get(age_col), 999.0)
        return age < 18
    return False


def risk_flags(uid: int, iid: int, item_bucket, user_info, minor_col, age_col):
    bucket = item_bucket.get(int(iid), "UNKNOWN")
    is_minor = is_minor_user(uid, user_info, minor_col, age_col)

    r17 = int(bucket == "R17")
    rplus = int(bucket == "RPLUS")
    rx = int(bucket == "RX")

    # User-aware unsafe rule:
    # - minor: R17/RPLUS/RX are unsafe
    # - adult: RPLUS/RX are unsafe
    unsafe_any = int((r17 or rplus or rx) if is_minor else (rplus or rx))
    return {"R17": r17, "RPLUS": rplus, "RX": rx}, unsafe_any, is_minor


def init_acc(topks):
    return {
        "users": 0,
        "hr_sum": {k: 0.0 for k in topks},
        "ndcg_sum": {k: 0.0 for k in topks},
        "top_items": {k: 0 for k in topks},
        "risk_items": {r: {k: 0 for k in topks} for r in RISK_NAMES},
        "unsafe_any": {k: 0 for k in topks},
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
        for rname in RISK_NAMES:
            acc["risk_items"][rname][k] += sum(int(x[rname]) for x in topk)
        acc["unsafe_any"][k] += sum(int(x["unsafe_any"]) for x in topk)


def finalize(acc, topks):
    out = {"users": acc["users"]}
    for k in topks:
        users = max(acc["users"], 1)
        denom = max(acc["top_items"][k], 1)
        out[f"HR@{k}"] = acc["hr_sum"][k] / users
        out[f"NDCG@{k}"] = acc["ndcg_sum"][k] / users
        for rname in RISK_NAMES:
            out[f"{rname}@{k}"] = acc["risk_items"][rname][k] / denom
        out[f"unsafe_any@{k}"] = acc["unsafe_any"][k] / denom
    return out


def write_metrics_csv(path, metrics_by_split, topks, digits=5):
    fields = ["split", "group", "K", "HR", "NDCG", "R17", "RPLUS", "RX", "unsafe_any", "users"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for split, metrics in metrics_by_split.items():
            for group in ["all", "minor", "adult"]:
                m = metrics[group]
                for k in topks:
                    row = {
                        "split": split,
                        "group": group,
                        "K": k,
                        "HR": f"{m[f'HR@{k}']:.{digits}f}",
                        "NDCG": f"{m[f'NDCG@{k}']:.{digits}f}",
                        "users": m["users"],
                    }
                    for rname in RISK_NAMES:
                        row[rname] = f"{m[f'{rname}@{k}']:.{digits}f}"
                    row["unsafe_any"] = f"{m[f'unsafe_any@{k}']:.{digits}f}"
                    w.writerow(row)
    print(f"[write] {path}")


def find_best_epoch(llmsrec_root: str, out_name: str, llm: str) -> int:
    best_dir = Path(llmsrec_root) / "models" / f"{out_name}_{llm}_trainbest"
    pats = sorted(best_dir.glob(f"{out_name}_{llm}_*_item_proj.pt"))
    if not pats:
        raise FileNotFoundError(f"No best item_proj found under {best_dir}")
    epochs = []
    for p in pats:
        m = re.search(rf"{re.escape(out_name)}_{re.escape(llm)}_(\d+)_item_proj\.pt$", p.name)
        if m:
            epochs.append(int(m.group(1)))
    if not epochs:
        raise RuntimeError(f"Cannot parse best epoch from: {[p.name for p in pats]}")
    epoch = max(epochs)
    print(f"[detect] best_epoch={epoch}, best_dir={best_dir}")
    return epoch


def make_llmsrec_args(args, device: str, best_epoch: int):
    return SimpleNamespace(
        multi_gpu=False,
        device=device,
        world_size=1,
        llm=args.llm,
        recsys=args.recsys,
        rec_pre_trained_data=args.out_name,
        train=True,
        extract=False,
        token=False,
        save_dir=f"{args.out_name}_{args.llm}_trainbest",
        batch_size=1,
        batch_size_infer=args.user_batch_size,
        infer_epoch=best_epoch,
        maxlen=args.maxlen,
        num_epochs=1,
        stage2_lr=1e-5,
        nn_parameter=False,
    )


@torch.no_grad()
def build_item_embs(model, item_batch_size: int, device: str):
    model.eval()
    item_num = int(model.item_num)
    all_embs = []
    max_input_length = 1024
    print(f"[item emb] item_num={item_num}, batch={item_batch_size}")

    for st in tqdm(range(1, item_num + 1, item_batch_size), desc="item_emb"):
        ids = list(range(st, min(item_num + 1, st + item_batch_size)))
        texts = [
            "The item title and item embedding are as follows: "
            + model.find_item_text_single(i, title_flag=True, description_flag=False)
            + "[HistoryEmb], then generate item representation token:[ItemOut]"
            for i in ids
        ]
        tokens = model.llm.llm_tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(device)
        hist = model.item_emb_proj(model.get_item_emb(ids))
        embeds = model.llm.llm_model.get_input_embeddings()(tokens["input_ids"])
        embeds = model.llm.replace_out_token_all_infer(
            tokens,
            embeds,
            token=["[ItemOut]", "[HistoryEmb]"],
            embs={"[HistoryEmb]": hist},
        )
        ctx = torch.amp.autocast("cuda") if str(device).startswith("cuda") else nullcontext()
        with ctx:
            outputs = model.llm.llm_model.forward(inputs_embeds=embeds, output_hidden_states=True)
            idxs = model.llm.get_embeddings(tokens, "[ItemOut]")
            item_out = torch.cat([
                outputs.hidden_states[-1][b, idxs[b]].mean(axis=0).unsqueeze(0)
                for b in range(len(idxs))
            ])
            item_out = model.llm.pred_item(item_out)
        all_embs.append(item_out.detach())
        del outputs, item_out, embeds, tokens, hist
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def build_user_embs(model, train_1based: Dict[int, List[int]], user_batch_size: int, maxlen: int, device: str):
    model.eval()
    user_num = int(model.recsys.user_num)
    all_user = []
    max_input_length = 1024
    print(f"[user emb] user_num={user_num}, batch={user_batch_size}")

    for st in tqdm(range(1, user_num + 1, user_batch_size), desc="user_emb"):
        us = list(range(st, min(user_num + 1, st + user_batch_size)))
        text_input, interact_embs = [], []
        for u in us:
            seq = np.array(train_1based.get(u, [])[-maxlen:], dtype=np.int64)
            if seq.size == 0:
                seq = np.array([1], dtype=np.int64)
            interact_text, interact_ids = model.make_interact_text(seq[seq > 0], 10, u)
            text_input.append(
                "This user has watched a sequence of anime in the following order: "
                + interact_text
                + ". Based on this sequence, generate user representation token:[UserOut]"
            )
            interact_embs.append(model.item_emb_proj(model.get_item_emb(interact_ids)))

        tokens = model.llm.llm_tokenizer(
            text_input,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(device)
        embeds = model.llm.llm_model.get_input_embeddings()(tokens["input_ids"])
        embeds = model.llm.replace_out_token_all(
            tokens,
            embeds,
            token=["[UserOut]", "[HistoryEmb]"],
            embs={"[HistoryEmb]": interact_embs},
        )
        ctx = torch.amp.autocast("cuda") if str(device).startswith("cuda") else nullcontext()
        with ctx:
            outputs = model.llm.llm_model.forward(inputs_embeds=embeds, output_hidden_states=True)
            idxs = model.llm.get_embeddings(tokens, "[UserOut]")
            user_out = torch.cat([
                outputs.hidden_states[-1][b, idxs[b]].mean(axis=0).unsqueeze(0)
                for b in range(len(idxs))
            ])
            user_out = model.llm.pred_user(user_out)
        all_user.append(user_out.detach())
        del outputs, user_out, embeds, tokens, interact_embs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(all_user, dim=0)


@torch.no_grad()
def make_score_matrix(user_embs, item_embs, out_path: str, user_chunk: int = 256):
    num_users, num_items = user_embs.shape[0], item_embs.shape[0]
    print(f"[score matrix] users={num_users}, items={num_items}, out={out_path}")
    scores = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(num_users, num_items))
    item_t = item_embs.float().T.contiguous()
    for st in tqdm(range(0, num_users, user_chunk), desc="score_matmul"):
        ed = min(num_users, st + user_chunk)
        s = torch.matmul(user_embs[st:ed].float(), item_t)
        scores[st:ed, :] = s.detach().cpu().numpy().astype(np.float32)
        del s
    scores.flush()
    return scores


def evaluate_scores(
    score_mat,
    users,
    eval_by_user,
    train_by_user,
    valid_by_user,
    split,
    topks,
    item_bucket,
    user_info,
    minor_col,
    age_col,
    args,
    topk_csv="",
):
    num_users, num_items = score_mat.shape
    max_eval_k = max(max(topks), args.save_topk)
    acc = {"all": init_acc(topks), "minor": init_acc(topks), "adult": init_acc(topks)}
    topk_rows = []

    for n_done, uid in enumerate(users, 1):
        if uid < 0 or uid >= num_users:
            continue
        pos_items = [i for i in unique_int_list(eval_by_user.get(uid, [])) if 0 <= i < num_items]
        if not pos_items:
            continue

        pos_set = set(pos_items)
        s = np.array(score_mat[uid], dtype=np.float32, copy=True)

        block = set(i for i in train_by_user.get(uid, []) if 0 <= i < num_items)
        if split == "test" and args.exclude_valid_for_test:
            block.update(i for i in valid_by_user.get(uid, []) if 0 <= i < num_items and i not in pos_set)
        block.difference_update(pos_set)

        if block:
            s[np.fromiter(block, dtype=np.int64)] = -np.inf

        k = min(max_eval_k, num_items)
        top_idx = np.argpartition(-s, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-s[top_idx])]

        is_minor = is_minor_user(uid, user_info, minor_col, age_col)
        group = "minor" if is_minor else "adult"
        top_infos = []

        for rank, iid in enumerate(top_idx.tolist(), 1):
            flags, unsafe_any, _ = risk_flags(uid, iid, item_bucket, user_info, minor_col, age_col)
            info = {
                "rank": rank,
                "iid": iid,
                "score": float(s[iid]),
                "is_positive": int(iid in pos_set),
                "unsafe_any": unsafe_any,
                **flags,
            }
            if rank <= max(topks):
                top_infos.append(info)
            if topk_csv and rank <= args.save_topk:
                topk_rows.append({
                    "split": split,
                    "group": group,
                    "user_id": uid,
                    "rank": rank,
                    "item_id": iid,
                    "score": f"{float(s[iid]):.8f}",
                    "is_positive": int(iid in pos_set),
                    **flags,
                    "unsafe_any": unsafe_any,
                })

        update_acc(acc["all"], topks, top_infos, pos_set)
        update_acc(acc[group], topks, top_infos, pos_set)

        if n_done == 1 or n_done % args.progress_every == 0:
            m = finalize(acc["all"], topks)
            print(
                f"[eval {split}] seen={n_done}, eval_users={m['users']}, "
                f"HR@{args.monitor_k}={m.get(f'HR@{args.monitor_k}', 0):.5f}, "
                f"NDCG@{args.monitor_k}={m.get(f'NDCG@{args.monitor_k}', 0):.5f}",
                flush=True,
            )

    if topk_csv and topk_rows:
        with open(topk_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(topk_rows[0].keys()))
            w.writeheader()
            w.writerows(topk_rows)
        print(f"[write] {topk_csv}")

    return {g: finalize(a, topks) for g, a in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmsrec_root", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_name", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--train_rating", required=True)
    ap.add_argument("--valid_rating", required=True)
    ap.add_argument("--test_rating", required=True)
    ap.add_argument("--item_table", required=True)
    ap.add_argument("--user_table", required=True)

    ap.add_argument("--llm", default="qwen25")
    ap.add_argument("--recsys", default="sasrec")
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--item_batch_size", type=int, default=128)
    ap.add_argument("--user_batch_size", type=int, default=1)
    ap.add_argument("--score_user_chunk", type=int, default=256)

    ap.add_argument("--topks", default="1,5,10,20")
    ap.add_argument("--save_topk", type=int, default=100)
    ap.add_argument("--save_scores", action="store_true")
    ap.add_argument("--output_topk", action="store_true")
    ap.add_argument("--exclude_valid_for_test", action="store_true")

    ap.add_argument("--monitor_k", type=int, default=10)
    ap.add_argument("--progress_every", type=int, default=200)
    ap.add_argument("--round_digits", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    topks = parse_topks(args.topks)

    train_by_user = read_rating_by_user(args.train_rating, "base-train-block")
    valid_by_user = read_rating_by_user(args.valid_rating, "valid")
    test_by_user = read_rating_by_user(args.test_rating, "test")
    item_bucket, item_code = load_items(args.item_table)
    user_info, minor_col, age_col = load_users(args.user_table)

    llmsrec_root = Path(args.llmsrec_root).resolve()
    seq_train_path = llmsrec_root / "SeqRec" / f"data_{args.out_name}" / f"{args.out_name}_train.txt"
    if not seq_train_path.exists():
        raise FileNotFoundError(f"Cannot find LLM-SRec train split: {seq_train_path}")
    train_1based = read_llmsrec_split(str(seq_train_path))
    print(f"[load] llmsrec train split users={len(train_1based)}, path={seq_train_path}")

    os.environ["LLMSREC_LLM_MODEL_PATH"] = args.model_path
    sys.path.insert(0, str(llmsrec_root))
    os.chdir(str(llmsrec_root))

    from models.seqllm_model import llmrec_model  # noqa: E402

    best_epoch = find_best_epoch(str(llmsrec_root), args.out_name, args.llm)
    llm_args = make_llmsrec_args(args, args.device, best_epoch)
    print(f"[load model] out={args.out_name}, save_dir={llm_args.save_dir}, epoch={best_epoch}")

    model = llmrec_model(llm_args).to(args.device)
    model.load_model(llm_args, phase2_epoch=best_epoch)
    model.eval()

    item_embs = build_item_embs(model, args.item_batch_size, args.device)
    user_embs = build_user_embs(model, train_1based, args.user_batch_size, args.maxlen, args.device)

    score_path = os.path.join(args.output_dir, "scores.npy")
    score_mat = make_score_matrix(user_embs, item_embs, score_path, args.score_user_chunk)

    meta = {
        "tag": args.tag,
        "out_name": args.out_name,
        "best_epoch": best_epoch,
        "scores_npy": score_path,
        "num_users": int(score_mat.shape[0]),
        "num_items": int(score_mat.shape[1]),
        "llmsrec_root": str(llmsrec_root),
        "model_path": args.model_path,
        "item_table": args.item_table,
        "user_table": args.user_table,
    }
    with open(os.path.join(args.output_dir, "score_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[write] {os.path.join(args.output_dir, 'score_meta.json')}")

    valid_metrics = evaluate_scores(
        score_mat,
        sorted(valid_by_user.keys()),
        valid_by_user,
        train_by_user,
        {},
        "valid",
        topks,
        item_bucket,
        user_info,
        minor_col,
        age_col,
        args,
        os.path.join(args.output_dir, "top100_valid.csv") if args.output_topk else "",
    )

    test_metrics = evaluate_scores(
        score_mat,
        sorted(test_by_user.keys()),
        test_by_user,
        train_by_user,
        valid_by_user,
        "test",
        topks,
        item_bucket,
        user_info,
        minor_col,
        age_col,
        args,
        os.path.join(args.output_dir, "top100_test.csv") if args.output_topk else "",
    )

    metrics_by_split = {"valid": valid_metrics, "test": test_metrics}
    write_metrics_csv(os.path.join(args.output_dir, "metrics.csv"), metrics_by_split, topks, args.round_digits)

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "meta": meta, "metrics": metrics_by_split}, f, ensure_ascii=False, indent=2)
    print(f"[write] {os.path.join(args.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
