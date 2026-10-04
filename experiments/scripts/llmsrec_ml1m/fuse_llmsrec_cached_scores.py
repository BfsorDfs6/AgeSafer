#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: scripts/llmsrec_ml1m/fuse_llmsrec_cached_scores.py

Cached-score fusion for LLM-SRec.
It reads base scores.npy + one rho scores.npy. It never loads Qwen and never
re-runs LLM-SRec. It can evaluate fixed residual fusion or train a small gate.

s_final = s_base + g(u,i) * (s_aug - s_base)
"""
import argparse, csv, json, math, os, random, re
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RISK_DIMS = ["sex_code", "violence_code", "profanity_code", "drug_code", "intense_code"]
DIM_OUT = {"sex_code":"sex", "violence_code":"violence", "profanity_code":"profanity", "drug_code":"drug", "intense_code":"intense"}
DIM_NAMES = ["sex", "violence", "profanity", "drug", "intense"]
ADULT_DIM = "isAdult"


def norm_name(s: str) -> str: return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def safe_int(x: Any, default: int=-1) -> int:
    try: return int(float(str(x).strip()))
    except Exception: return default
def safe_float(x: Any, default: float=0.0) -> float:
    try:
        if x is None or str(x).strip() == "": return default
        return float(str(x).strip())
    except Exception: return default
def parse_boolish(x: Any) -> bool:
    s = str(x).strip().lower()
    if s in {"1","true","yes","y","minor","under18","under_18"}: return True
    if s in {"0","false","no","n","adult"}: return False
    try: return float(s) > 0
    except Exception: return False

def split_line(line: str) -> List[str]:
    s = line.strip()
    if not s: return []
    if "::" in s: return s.split("::")
    if "\t" in s: return s.split("\t")
    if "," in s and not s.startswith("("): return [x.strip() for x in s.split(",")]
    return re.split(r"\s+", s)

def looks_like_header(parts: Sequence[str]) -> bool:
    return (not parts) or safe_int(parts[0], default=-999999) == -999999

def parse_topks(s: str) -> List[int]:
    out = []
    for x in re.split(r"[,;\s]+", str(s).strip()):
        if x: out.append(int(x))
    return sorted(set(k for k in out if k > 0)) or [10]

def read_rating_by_user(path: str, name: str="") -> Dict[int, List[int]]:
    by_user = defaultdict(list); rows = skipped = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            parts = split_line(line)
            if not parts: continue
            if line_no == 1 and looks_like_header(parts): continue
            if len(parts) < 2:
                skipped += 1; continue
            u, i = safe_int(parts[0]), safe_int(parts[1])
            if u < 0 or i < 0:
                skipped += 1; continue
            by_user[u].append(i); rows += 1
    print(f"[load] {name or path}: rows={rows}, users={len(by_user)}, skipped={skipped}, path={path}")
    return {int(u): list(v) for u, v in by_user.items()}

def unique_int_list(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(int(x))
    return out

def read_csv_dict(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(4096); f.seek(0)
        try: dialect = csv.Sniffer().sniff(sample)
        except Exception: dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader); fields = reader.fieldnames or []
    return rows, fields

def find_col(fields: List[str], candidates: List[str], required=False, label="") -> str:
    norm_to_real = {norm_name(f): f for f in fields}
    for cand in candidates:
        key = norm_name(cand)
        if key in norm_to_real: return norm_to_real[key]
    for cand in candidates:
        key = norm_name(cand)
        for nf, real in norm_to_real.items():
            if key and key in nf: return real
    if required: raise ValueError(f"Cannot find column for {label or candidates}. Available={fields}")
    return ""

def find_item_id_col(fields): return find_col(fields, ["inner_item_id","item_inner_id","item_id","iid","movie_id","candidate_inner_item_id"], True, "item id")
def find_user_id_col(fields): return find_col(fields, ["inner_user_id","user_inner_id","user_id","uid"], True, "user id")
def find_minor_col(fields): return find_col(fields, ["is_minor","minor","isMinor","user_is_minor","under18","is_under_18"], False, "minor flag")
def find_item_risk_cols(fields):
    out = {}
    for dim in RISK_DIMS + [ADULT_DIM]:
        short = dim.replace("_code", "")
        names = [dim, dim.lower(), short, short+"_code", short+"_level", short+"_risk", short+"_score", "item_"+dim, "risk_"+dim]
        out[dim] = find_col(fields, names, False, f"item risk {dim}")
    return out

def load_item_safe(path: str):
    rows, fields = read_csv_dict(path)
    item_col = find_item_id_col(fields); risk_cols = find_item_risk_cols(fields); item_safe = {}
    for r in rows:
        iid = safe_int(r.get(item_col), -1)
        if iid >= 0: item_safe[iid] = r
    print(f"[load] item_safe rows={len(rows)}, items={len(item_safe)}, path={path}")
    print(f"[detect] item_id_col={item_col}")
    print(f"[detect] item_risk_cols={risk_cols}")
    return item_safe, risk_cols

def load_user_info(path: str):
    rows, fields = read_csv_dict(path)
    user_col = find_user_id_col(fields); minor_col = find_minor_col(fields); user_info = {}
    for r in rows:
        uid = safe_int(r.get(user_col), -1)
        if uid >= 0: user_info[uid] = r
    print(f"[load] user_info rows={len(rows)}, users={len(user_info)}, path={path}")
    print(f"[detect] user_id_col={user_col}, minor_col={minor_col or 'N/A'}")
    return user_info, minor_col

def is_minor_user(uid, user_info, minor_col): return parse_boolish(user_info.get(int(uid), {}).get(minor_col, "")) if minor_col else False

def dim_violations(uid, iid, item_safe, item_risk_cols, user_info, minor_col, minor_block_at, adult_block_at, isadult_policy):
    row_i = item_safe.get(int(iid), {})
    is_minor = is_minor_user(uid, user_info, minor_col)
    block_at = float(minor_block_at if is_minor else adult_block_at)
    vios = {}
    for dim in RISK_DIMS:
        col = item_risk_cols.get(dim, "")
        val = safe_float(row_i.get(col), 0.0) if col else 0.0
        vios[DIM_OUT[dim]] = int(val >= block_at)
    adult_col = item_risk_cols.get(ADULT_DIM, "")
    adult_val = safe_float(row_i.get(adult_col), 0.0) if adult_col else 0.0
    adult_content = int(adult_val >= 1)
    adult_hard = int(isadult_policy in {"minor_only", "all"} and adult_content and (is_minor or isadult_policy == "all"))
    unsafe_any = int(any(vios.values()) or adult_hard)
    return vios, unsafe_any, adult_content, is_minor

def read_pred_jsonl(path: str) -> Dict[Tuple[int,int], Tuple[float,float,float]]:
    out = {}
    if not path or not os.path.exists(path):
        print(f"[warn] pred_jsonl missing: {path}; use zeros")
        return out
    user_keys = ["user_id","uid","user","inner_user_id","candidate_inner_user_id"]
    item_keys = ["item_id","iid","item","inner_item_id","candidate_inner_item_id","candidate_item_id"]
    rows = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except Exception: continue
            meta = obj.get("metadata", {}) if isinstance(obj.get("metadata", {}), dict) else {}
            u = i = -1
            for src in (obj, meta):
                for k in user_keys:
                    if k in src: u = safe_int(src.get(k), -1); break
                if u >= 0: break
            for src in (obj, meta):
                for k in item_keys:
                    if k in src: i = safe_int(src.get(k), -1); break
                if i >= 0: break
            if u < 0 or i < 0: continue
            psg_pred = safe_float(obj.get("psg_pred", obj.get("pred", obj.get("label", 0.0))), 0.0)
            p1 = safe_float(obj.get("p1", obj.get("prob_1", obj.get("prob1", 0.0))), 0.0)
            margin = safe_float(obj.get("margin_1_minus_0", obj.get("margin", obj.get("logit_margin", 0.0))), 0.0)
            out[(u, i)] = (psg_pred, p1, margin); rows += 1
    print(f"[load] pred_jsonl rows={rows}, map_size={len(out)}, path={path}")
    return out

class GateNet(nn.Module):
    def __init__(self, input_dim=14, hidden=32, dropout=0.1, gate_max=1.0):
        super().__init__(); self.gate_max = float(gate_max)
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.pref_head = nn.Linear(hidden, 1); self.conf_head = nn.Linear(hidden, 1); self.safe_head = nn.Linear(hidden, 1)
        nn.init.constant_(self.pref_head.bias, 1.0); nn.init.constant_(self.conf_head.bias, 1.0); nn.init.constant_(self.safe_head.bias, 1.2)
        self.conf_p1_scale = nn.Parameter(torch.tensor(1.0)); self.conf_margin_scale = nn.Parameter(torch.tensor(0.10)); self.safe_unsafe_scale = nn.Parameter(torch.tensor(0.8))
    def forward(self, x):
        h = self.shared(x); p1 = x[:,4].clamp(0,1); margin = x[:,5]; unsafe = x[:,11].clamp(0,1)
        g_pref = torch.sigmoid(self.pref_head(h).squeeze(-1))
        g_conf = torch.sigmoid(self.conf_head(h).squeeze(-1) + self.conf_p1_scale*(p1-0.5) + self.conf_margin_scale*margin)
        g_safe = torch.sigmoid(self.safe_head(h).squeeze(-1) - F.softplus(self.safe_unsafe_scale)*unsafe)
        return self.gate_max * g_pref * g_conf * g_safe

class CachedFeatureBuilder:
    def __init__(self, base_scores, aug_scores, pred_map, item_safe, item_risk_cols, user_info, minor_col, args):
        self.base = base_scores; self.aug = aug_scores; self.pred_map = pred_map
        self.item_safe = item_safe; self.item_risk_cols = item_risk_cols; self.user_info = user_info; self.minor_col = minor_col; self.args = args
        self.num_users, self.num_items = base_scores.shape
    def build(self, users: List[int], items: List[int]):
        bs = np.array([self.base[u, i] for u, i in zip(users, items)], dtype=np.float32)
        ps = np.array([self.aug[u, i] for u, i in zip(users, items)], dtype=np.float32)
        feats = []
        for b, p, u, i in zip(bs, ps, users, items):
            pf = self.pred_map.get((int(u), int(i)), (0.0, 0.0, 0.0))
            vios, unsafe_any, adult_content, is_minor = dim_violations(u, i, self.item_safe, self.item_risk_cols, self.user_info, self.minor_col, self.args.minor_block_at, self.args.adult_block_at, self.args.isadult_policy)
            feats.append([float(b), float(p), float(p-b), float(pf[0]), float(pf[1]), float(pf[2]),
                          float(vios["sex"]), float(vios["violence"]), float(vios["profanity"]), float(vios["drug"]), float(vios["intense"]),
                          float(unsafe_any), float(adult_content), float(is_minor)])
        return torch.tensor(feats, dtype=torch.float32, device=self.args.device_resolved), torch.tensor(bs, dtype=torch.float32, device=self.args.device_resolved), torch.tensor(ps, dtype=torch.float32, device=self.args.device_resolved)

def final_score(gate, feats, bs, ps, mode="gated", alpha=0.5):
    if mode == "fixed":
        g = torch.full_like(bs, float(alpha))
    else:
        g = gate(feats)
    return bs + g * (ps - bs), g

def sample_negative(num_items, block, rng):
    for _ in range(200):
        i = rng.randrange(num_items)
        if i not in block: return i
    return rng.randrange(num_items)

def init_acc(topks):
    return {"users":0, "hr_sum":{k:0.0 for k in topks}, "ndcg_sum":{k:0.0 for k in topks}, "top_items":{k:0 for k in topks},
            "dim_items":{d:{k:0 for k in topks} for d in DIM_NAMES}, "unsafe_any":{k:0 for k in topks}, "adult_content":{k:0 for k in topks}}

def update_acc(acc, topks, top_infos, pos_set):
    acc["users"] += 1
    for k in topks:
        topk = [r for r in top_infos if r["rank"] <= k]
        hit, dcg = 0.0, 0.0
        for r in topk:
            if r["iid"] in pos_set:
                hit = 1.0; dcg += 1.0 / math.log2(r["rank"] + 1.0)
        ideal_len = min(len(pos_set), k)
        idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_len + 1)) if ideal_len > 0 else 1.0
        acc["hr_sum"][k] += hit; acc["ndcg_sum"][k] += dcg / idcg if idcg > 0 else 0.0; acc["top_items"][k] += len(topk)
        for d in DIM_NAMES: acc["dim_items"][d][k] += sum(int(r[d]) for r in topk)
        acc["unsafe_any"][k] += sum(int(r["unsafe_any"]) for r in topk); acc["adult_content"][k] += sum(int(r["adult_content"]) for r in topk)

def finalize(acc, topks):
    out = {"users":acc["users"]}
    for k in topks:
        users = max(acc["users"], 1); denom = max(acc["top_items"][k], 1)
        out[f"HR@{k}"] = acc["hr_sum"][k] / users; out[f"NDCG@{k}"] = acc["ndcg_sum"][k] / users
        for d in DIM_NAMES: out[f"{d}@{k}"] = acc["dim_items"][d][k] / denom
        out[f"unsafe_any@{k}"] = acc["unsafe_any"][k] / denom; out[f"adult_content@{k}"] = acc["adult_content"][k] / denom
    return out

def evaluate(gate, fb, users, eval_by_user, train_by_user, valid_by_user, split, topks, args, topk_csv=""):
    max_k = max(topks); acc = {"all":init_acc(topks), "minor":init_acc(topks), "adult":init_acc(topks)}; topk_rows = []
    if gate: gate.eval()
    with torch.no_grad():
        for n_done, uid in enumerate(users, 1):
            if uid < 0 or uid >= fb.num_users: continue
            pos_items = [i for i in unique_int_list(eval_by_user.get(uid, [])) if 0 <= i < fb.num_items]
            if not pos_items: continue
            pos_set = set(pos_items)
            mask = np.ones(fb.num_items, dtype=bool)
            block = set(i for i in train_by_user.get(uid, []) if 0 <= i < fb.num_items)
            if split == "test" and args.exclude_valid_for_test:
                block.update(i for i in valid_by_user.get(uid, []) if 0 <= i < fb.num_items and i not in pos_set)
            block.difference_update(pos_set)
            if block: mask[np.fromiter(block, dtype=np.int64)] = False
            candidates = np.nonzero(mask)[0].astype(np.int64)
            # Vectorized final score for a full candidate list. Gate features still need safety/pred features.
            us = [uid] * len(candidates); it = candidates.tolist()
            feats, bs, ps = fb.build(us, it)
            scores_t, gates_t = final_score(gate, feats, bs, ps, args.fusion_mode, args.alpha)
            top_vals, top_idx = torch.topk(scores_t, k=min(max_k, scores_t.numel()), largest=True, sorted=True)
            top_vals = top_vals.detach().cpu().numpy(); top_idx = top_idx.detach().cpu().numpy(); gates_np = gates_t.detach().cpu().numpy()
            group = "minor" if is_minor_user(uid, fb.user_info, fb.minor_col) else "adult"; top_infos = []
            for rank, local_idx in enumerate(top_idx.tolist(), 1):
                iid = int(candidates[local_idx]); score_val = float(top_vals[rank-1])
                vios, unsafe_any, adult_content, _ = dim_violations(uid, iid, fb.item_safe, fb.item_risk_cols, fb.user_info, fb.minor_col, args.minor_block_at, args.adult_block_at, args.isadult_policy)
                info = {"rank":rank, "iid":iid, "score":score_val, "is_positive":int(iid in pos_set), "unsafe_any":unsafe_any, "adult_content":adult_content, **vios}
                top_infos.append(info)
                if topk_csv:
                    topk_rows.append({"split":split, "group":group, "user_id":uid, "rank":rank, "item_id":iid, "score":f"{score_val:.8f}", "gate":f"{float(gates_np[local_idx]):.8f}", "is_positive":int(iid in pos_set), **vios, "adult_content":adult_content, "unsafe_any":unsafe_any})
            update_acc(acc["all"], topks, top_infos, pos_set); update_acc(acc[group], topks, top_infos, pos_set)
            if n_done == 1 or n_done % args.progress_every == 0:
                m = finalize(acc["all"], topks)
                print(f"[eval {split}] seen={n_done}, eval_users={m['users']}, HR@{args.monitor_k}={m.get(f'HR@{args.monitor_k}',0):.5f}, NDCG@{args.monitor_k}={m.get(f'NDCG@{args.monitor_k}',0):.5f}", flush=True)
    if topk_csv and topk_rows:
        with open(topk_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(topk_rows[0].keys())); w.writeheader(); w.writerows(topk_rows)
        print(f"[write] {topk_csv}")
    return {g: finalize(a, topks) for g, a in acc.items()}

def train_gate(gate, fb, valid_by_user, train_by_user, topks, args):
    if args.fusion_mode == "fixed": return [], None
    rng = random.Random(args.seed); device = args.device_resolved
    users = sorted([u for u, xs in valid_by_user.items() if xs and u < fb.num_users])
    rng.shuffle(users); n_train = max(1, int(round(args.calib_train_frac * len(users))))
    calib_users = users[:n_train]; holdout_users = users[n_train:] or users[-max(1, len(users)//5):]
    if args.max_holdout_users and len(holdout_users) > args.max_holdout_users: holdout_users = holdout_users[:args.max_holdout_users]
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_metric = -1e18; best_state = None; logs = []
    print(f"[train gate] calib_users={len(calib_users)}, holdout_users={len(holdout_users)}, num_items={fb.num_items}")
    for epoch in range(1, args.epochs+1):
        rng.shuffle(calib_users); gate.train(); total = batches = 0
        for st in range(0, len(calib_users), args.batch_users):
            pos_u=[]; pos_i=[]; neg_u=[]; neg_i=[]
            for u in calib_users[st:st+args.batch_users]:
                pos_list = [i for i in valid_by_user.get(u, []) if 0 <= i < fb.num_items]
                if not pos_list: continue
                pi = rng.choice(pos_list)
                block = set(i for i in train_by_user.get(u, []) if 0 <= i < fb.num_items); block.update(pos_list)
                ni = sample_negative(fb.num_items, block, rng)
                pos_u.append(u); pos_i.append(pi); neg_u.append(u); neg_i.append(ni)
            if not pos_u: continue
            pf, pbs, pps = fb.build(pos_u, pos_i); nf, nbs, nps = fb.build(neg_u, neg_i)
            ps, _ = final_score(gate, pf, pbs, pps, "gated", args.alpha); ns, _ = final_score(gate, nf, nbs, nps, "gated", args.alpha)
            loss = -F.logsigmoid(ps - ns).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0); opt.step()
            total += float(loss.detach().cpu()); batches += 1
        row = {"epoch":epoch, "loss": total / max(batches, 1)}
        if epoch == 1 or epoch % args.valid_every == 0 or epoch == args.epochs:
            metrics = evaluate(gate, fb, holdout_users, valid_by_user, train_by_user, {}, "valid", topks, args)
            m = metrics["all"]; ndcg = m.get(f"NDCG@{args.monitor_k}", 0.0); unsafe = m.get(f"unsafe_any@{args.monitor_k}", 0.0)
            monitor = ndcg - args.monitor_safety_weight * unsafe
            row.update({"valid_NDCG":ndcg, "valid_HR":m.get(f"HR@{args.monitor_k}",0.0), "valid_unsafe_any":unsafe, "monitor":monitor})
            print(f"[epoch {epoch}] loss={row['loss']:.5f} valid_NDCG@{args.monitor_k}={ndcg:.5f} unsafe={unsafe:.5f} monitor={monitor:.5f}")
            if monitor > best_metric:
                best_metric = monitor; best_state = {k:v.detach().cpu().clone() for k,v in gate.state_dict().items()}; print(f"[best] epoch={epoch}, monitor={monitor:.6f}")
        else:
            print(f"[epoch {epoch}] loss={row['loss']:.5f}")
        logs.append(row)
    if best_state is not None: gate.load_state_dict(best_state)
    return logs, best_metric

def write_metrics_csv(path, metrics_by_split, topks, digits=5):
    fields = ["split", "group", "K", "HR", "NDCG", "sex", "violence", "profanity", "drug", "intense", "unsafe_any", "adult_content", "users"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for split, metrics in metrics_by_split.items():
            for group in ["all", "minor", "adult"]:
                m = metrics[group]
                for k in topks:
                    row = {"split":split, "group":group, "K":k, "HR":f"{m[f'HR@{k}']:.{digits}f}", "NDCG":f"{m[f'NDCG@{k}']:.{digits}f}", "users":m["users"]}
                    for d in DIM_NAMES: row[d] = f"{m[f'{d}@{k}']:.{digits}f}"
                    row["unsafe_any"] = f"{m[f'unsafe_any@{k}']:.{digits}f}"; row["adult_content"] = f"{m[f'adult_content@{k}']:.{digits}f}"
                    w.writerow(row)
    print(f"[write] {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_scores", required=True); ap.add_argument("--aug_scores", required=True); ap.add_argument("--pred_jsonl", default="")
    ap.add_argument("--train_rating", required=True); ap.add_argument("--valid_rating", required=True); ap.add_argument("--test_rating", required=True)
    ap.add_argument("--item_safe", required=True); ap.add_argument("--user_tol", required=True); ap.add_argument("--output_dir", required=True)
    ap.add_argument("--fusion_mode", choices=["fixed","gated"], default="gated"); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gate_max", type=float, default=0.5); ap.add_argument("--hidden", type=int, default=32); ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch_users", type=int, default=64); ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--calib_train_frac", type=float, default=0.8); ap.add_argument("--valid_every", type=int, default=5); ap.add_argument("--max_holdout_users", type=int, default=0)
    ap.add_argument("--monitor_k", type=int, default=10); ap.add_argument("--monitor_safety_weight", type=float, default=0.0); ap.add_argument("--progress_every", type=int, default=200)
    ap.add_argument("--topks", default="1,5,10,20"); ap.add_argument("--minor_block_at", type=float, default=3.0); ap.add_argument("--adult_block_at", type=float, default=4.0)
    ap.add_argument("--isadult_policy", choices=["minor_only","all","none"], default="minor_only"); ap.add_argument("--exclude_valid_for_test", action="store_true")
    ap.add_argument("--output_topk", action="store_true"); ap.add_argument("--round_digits", type=int, default=5); ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto"); ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(); os.makedirs(args.output_dir, exist_ok=True)
    args.device_resolved = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    random.seed(args.seed); torch.manual_seed(args.seed); topks = parse_topks(args.topks)

    train_by_user = read_rating_by_user(args.train_rating, "train"); valid_by_user = read_rating_by_user(args.valid_rating, "valid"); test_by_user = read_rating_by_user(args.test_rating, "test")
    item_safe, item_risk_cols = load_item_safe(args.item_safe); user_info, minor_col = load_user_info(args.user_tol); pred_map = read_pred_jsonl(args.pred_jsonl)
    base_scores = np.load(args.base_scores, mmap_mode="r"); aug_scores = np.load(args.aug_scores, mmap_mode="r")
    if base_scores.shape != aug_scores.shape: raise ValueError(f"score shape mismatch: base={base_scores.shape}, aug={aug_scores.shape}")
    print(f"[scores] shape={base_scores.shape}, base={args.base_scores}, aug={args.aug_scores}")
    fb = CachedFeatureBuilder(base_scores, aug_scores, pred_map, item_safe, item_risk_cols, user_info, minor_col, args)
    gate = None if args.fusion_mode == "fixed" else GateNet(14, args.hidden, args.dropout, args.gate_max).to(args.device_resolved)
    logs, best_monitor = train_gate(gate, fb, valid_by_user, train_by_user, topks, args)
    if gate is not None:
        torch.save({"gate_state_dict":gate.state_dict(), "args":vars(args), "best_monitor":best_monitor}, os.path.join(args.output_dir, "llmsrec_cached_gate_best.pt"))
        print(f"[write] {os.path.join(args.output_dir, 'llmsrec_cached_gate_best.pt')}")
    if logs:
        with open(os.path.join(args.output_dir, "training_log.csv"), "w", encoding="utf-8", newline="") as f:
            fields = sorted(set(k for r in logs for k in r.keys())); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(logs)
    valid_users = sorted(valid_by_user.keys()); test_users = sorted(test_by_user.keys())
    valid_metrics = evaluate(gate, fb, valid_users, valid_by_user, train_by_user, {}, "valid", topks, args, os.path.join(args.output_dir, "topk_valid.csv") if args.output_topk else "")
    test_metrics = evaluate(gate, fb, test_users, test_by_user, train_by_user, valid_by_user, "test", topks, args, os.path.join(args.output_dir, "topk_test.csv") if args.output_topk else "")
    metrics = {"valid":valid_metrics, "test":test_metrics}
    write_metrics_csv(os.path.join(args.output_dir, "fusion_metrics.csv"), metrics, topks, args.round_digits)
    with open(os.path.join(args.output_dir, "fusion_metrics.json"), "w", encoding="utf-8") as f: json.dump({"args":vars(args), "metrics":metrics}, f, ensure_ascii=False, indent=2)
    print(f"[write] {os.path.join(args.output_dir, 'fusion_metrics.json')}")

if __name__ == "__main__":
    main()
