#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERSION: PSG-Dim5-v4.4-trainonly-direct-top100-pair-aware-RAG-label-leak-fix
FILE NAME ON SERVER: scripts/build_psg_candidate_inputs.py

Purpose
-------
Build PSG inference inputs under the new Dim5 safety metric.

This version is train-only: validation/test interactions are not read and are never used
for candidate filtering. Candidate pool = all_items - train_observed_items - profile_last5_items.

Only relevant changes from the old candidate pipeline:
  - safety rule uses Dim5 age thresholds, not p75 tolerance/exceed-count
  - candidate proposal is direct top100, no MMR reranking
  - no GMF/NCF/LightGCN score is used
  - every candidate input obtains rule text by calling scripts/psg_regulation_rag_full.py gen-rule
  - RAG uses concrete user-item pairs but reuses generated rules by Dim5 signature cache

Candidate score
---------------
candidate_score = pref_weight * content/profile genre similarity + pop_weight * train popularity score
Then take topn=100 directly.

Common command
--------------
cd .
STORY=outputs/psg_dim5_v4
CAND=${STORY}/candidate_inputs/direct_top100_dim5_pairaware_rag_trainonly
mkdir -p ${CAND} ${STORY}/logs ${STORY}/rag_vector
python -u scripts/build_psg_candidate_inputs.py \
  --profiles outputs/psg/profiles/user_profiles_train_last5.jsonl \
  --train_safe outputs/psg/safe_features/ml-1m_safe_features.train.safe.csv \
  --item_safe outputs/psg/safe_features/ml-1m_safe_features.item_safe.csv \
  --output ${CAND}/psg_infer_inputs_dim5_top100.jsonl \
  --candidates_output ${CAND}/psg_candidates_dim5_top100.jsonl \
  --summary_json ${CAND}/summary.json \
  --topn 100 \
  --history_window 5 \
  --pref_weight 0.85 \
  --pop_weight 0.15 \
  --minor_block_at 3 \
  --adult_block_at 4 \
  --isadult_policy minor_only \
  --rule_mode rag_cli \
  --rag_script scripts/psg_regulation_rag_full.py \
  --vector_dir outputs/psg/rag_vector \
  --model_path models/Qwen2.5-7B-Instruct \
  --rag_cache_path ${STORY}/rag_vector/rule_text_cache_dim5_pairaware_v4_2.jsonl \
  --rag_device cuda \
  --rag_dtype bf16 \
  --strict_no_leakage \
  2>&1 | tee ${STORY}/logs/build_candidate_dim5_top100_trainonly.log
"""

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Set, Tuple

RISK_DIMS_CODE = ["sex_code", "violence_code", "profanity_code", "drug_code", "intense_code"]
DIM_SHORT = {"sex_code": "sex", "violence_code": "violence", "profanity_code": "profanity", "drug_code": "drug", "intense_code": "intense"}
# Visible prompt leakage guard.
# Do NOT include broad English words such as "label" or "rating" here; they
# can appear naturally in policy text or movie descriptions and cause false
# positives. Only internal field names / hidden metadata strings are blocked.
FORBIDDEN_VISIBLE = [
    "sex_code", "violence_code", "profanity_code", "drug_code", "intense_code",
    "isAdult",
    "inner_user_id", "inner_item_id", "raw_movie_id", "raw_user_id",
    "rating_hidden", "preference_label", "safety_label", "psg_label",
    "safe_observed_label", "within_user_tolerance", "triggered_dims",
    "p75", "tol_", "risk score", "risk_score",
]


def clean_text(x: Any, max_chars: int = 800) -> str:
    s = str(x or "").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        cut = s[:max_chars]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        s = cut + "..."
    return s


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


def get_first(row: Dict[str, Any], names: Iterable[str], default: str = "") -> str:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return str(row[n])
    return default


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def row_user_id(row: Dict[str, Any]) -> int:
    return safe_int(get_first(row, ["inner_user_id", "user_id", "uid", "user"], "-1"), -1)


def row_item_id(row: Dict[str, Any]) -> int:
    return safe_int(get_first(row, ["inner_item_id", "item_id", "iid", "item", "movie_inner_id"], "-1"), -1)


def row_timestamp(row: Dict[str, Any]) -> int:
    return safe_int(get_first(row, ["timestamp", "time", "ts"], "0"), 0)


def load_profiles(path: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    for obj in read_jsonl(path):
        if obj.get("parse_error"):
            continue
        uid = safe_int(obj.get("user_id", obj.get("inner_user_id")), -1)
        if uid >= 0:
            out[uid] = obj
    return out


def group_by_user(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    d = defaultdict(list)
    for r in rows:
        u = row_user_id(r)
        if u >= 0:
            d[u].append(r)
    for u in d:
        d[u].sort(key=lambda r: (row_timestamp(r), row_item_id(r)))
    return dict(d)


def load_item_map(path: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    for r in read_csv_rows(path):
        iid = row_item_id(r)
        if iid >= 0:
            out[iid] = r
    return out


def parse_genres(x: Any) -> Set[str]:
    parts = x if isinstance(x, list) else re.split(r"[|,;/]", str(x or ""))
    out = set()
    for p in parts:
        p = clean_text(p, 80).lower().replace("children's", "children").replace("sci-fi", "sci fi")
        if p and p not in {"unknown", "not specified", "nan", "none"}:
            out.add(p)
    return out


def item_text(row: Dict[str, Any]) -> Tuple[str, str, str]:
    title = clean_text(get_first(row, ["title", "movie_title", "primaryTitle", "originalTitle", "name"], "Unknown title"), 180)
    genres = clean_text(get_first(row, ["genres", "genre"], "Unknown genres"), 220)
    overview = clean_text(get_first(row, ["overview", "plot", "description", "movie_overview"], "No overview available"), 900)
    return title, genres, overview


def risk_value(row: Dict[str, Any], dim: str) -> float:
    return safe_float(get_first(row, [dim, dim.replace("_code", ""), "item_" + dim, "risk_" + dim], "0"), 0.0)


def adult_value(row: Dict[str, Any]) -> float:
    return safe_float(get_first(row, ["isAdult", "is_adult", "isadult", "adult"], "0"), 0.0)


def is_minor_user(profile: Dict[str, Any]) -> bool:
    ui = profile.get("user_info", {}) if profile else {}
    if "is_minor" in ui:
        return bool(ui.get("is_minor"))
    age = str(ui.get("age_desc", ui.get("age_group", ""))).lower()
    return "under" in age or "minor" in age or "children" in age


def dim5_violations(profile: Dict[str, Any], item_row: Dict[str, Any], minor_block_at: float, adult_block_at: float, isadult_policy: str) -> Tuple[List[str], int, int]:
    minor = is_minor_user(profile)
    threshold = float(minor_block_at if minor else adult_block_at)
    dims = []
    for dim in RISK_DIMS_CODE:
        if risk_value(item_row, dim) >= threshold:
            dims.append(DIM_SHORT[dim])
    adult_flag = int(adult_value(item_row) >= 1)
    adult_hard = 0
    if adult_flag:
        if isadult_policy in {"all", "all_unsafe"}:
            adult_hard = 1
        elif isadult_policy == "minor_only" and minor:
            adult_hard = 1
    return sorted(set(dims)), adult_flag, adult_hard


def read_rating_file(path: str) -> Dict[int, Set[int]]:
    out = defaultdict(set)
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "::" in s:
                arr = s.split("::")
            elif "\t" in s:
                arr = s.split("\t")
            elif "," in s and not s.startswith("("):
                arr = [x.strip() for x in s.split(",")]
            else:
                arr = re.split(r"\s+", s)
            if len(arr) >= 2:
                u = safe_int(arr[0], -1)
                i = safe_int(arr[1], -1)
                if u >= 0 and i >= 0:
                    out[u].add(i)
    return dict(out)


def profile_history_item_ids(profile: Dict[str, Any], k: int) -> List[int]:
    ids = profile.get("history_item_ids", []) or profile.get("profile_item_ids", []) or []
    out = []
    for x in ids[-k:]:
        iid = safe_int(x, -1)
        if iid >= 0:
            out.append(iid)
    return out


def user_genres(profile: Dict[str, Any], train_rows: List[Dict[str, Any]], item_map: Dict[int, Dict[str, Any]], history_window: int) -> Set[str]:
    p = profile.get("profile", {})
    g = parse_genres(p.get("preferred_genres", []))
    hist_ids = profile_history_item_ids(profile, history_window)
    if not hist_ids:
        hist_ids = [row_item_id(r) for r in train_rows[-history_window:]]
    for iid in hist_ids:
        row = item_map.get(iid, {})
        if row:
            g.update(parse_genres(get_first(row, ["genres", "genre"], "")))
    return g


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def profile_text(profile: Dict[str, Any]) -> str:
    p = profile.get("profile", {})
    ui = profile.get("user_info", {})
    desc = clean_text(p.get("user_description", ""), 240)
    if not desc:
        age_desc = clean_text(ui.get("age_desc", ui.get("age_group", "unknown age")), 80)
        gender_desc = clean_text(ui.get("gender_desc", ui.get("gender", "viewer")), 80)
        desc = f"The user is a {gender_desc} viewer, age group: {age_desc}."

    def list_text(x, max_items=6):
        if isinstance(x, list):
            vals = [clean_text(v, 80) for v in x if clean_text(v, 80)]
        elif isinstance(x, str):
            vals = [clean_text(v, 80) for v in re.split(r"[,;/|]", x) if clean_text(v, 80)]
        else:
            vals = []
        return ", ".join(vals[:max_items]) if vals else "Not specified"

    return "\n".join([
        "User Information:",
        desc,
        "",
        "User Preference Profile:",
        f"- Summary: {clean_text(p.get('profile_text', ''), 750) or 'Not specified'}",
        f"- Preferred genres: {list_text(p.get('preferred_genres', []))}",
        f"- Preferred themes: {list_text(p.get('preferred_themes', []))}",
        f"- Preferred style: {list_text(p.get('preferred_style', []), 5)}",
    ])


def history_text(profile: Dict[str, Any], train_rows: List[Dict[str, Any]], item_map: Dict[int, Dict[str, Any]], history_window: int) -> str:
    hist_ids = profile_history_item_ids(profile, history_window)
    if not hist_ids:
        hist_ids = [row_item_id(r) for r in train_rows[-history_window:]]
    lines = ["Recent User History:"]
    for idx, iid in enumerate(hist_ids[-history_window:], 1):
        row = item_map.get(iid, {})
        title, genres, overview = item_text(row)
        lines.append(f"{idx}. Title: {title}\n   Genres: {genres}\n   Overview: {overview}")
    return "\n".join(lines)


def candidate_text(row: Dict[str, Any]) -> str:
    title, genres, overview = item_text(row)
    return "\n".join(["Candidate Movie:", f"Title: {title}", f"Genres: {genres}", f"Overview: {overview}"])


def template_rule(is_minor: bool, dims: List[str], adult_hard: bool) -> str:
    """Fallback rule text.

    This template intentionally avoids exposing hidden numeric thresholds or
    internal field names. In the main experiment we use --rule_mode rag_cli
    --require_rag_success, so this is only for debugging or non-strict runs.
    """
    aud = "minor users" if is_minor else "adult users"
    topics = ", ".join(dims) if dims else "general age-appropriateness"
    lines = [
        f"- For {aud}, evaluate whether the movie is age-appropriate and suitable as a pseudo-positive recommendation candidate.",
        "- Reject movies with clear explicit sexual content, severe graphic violence, explicit drug abuse, excessive abusive language, or highly disturbing content.",
        f"- Pay particular attention to {topics}, but do not reject ordinary mild dramatic or genre content without a clear age-safety violation.",
        "- Allow ordinary safe themes such as mild romance, family/social themes, historical tension, non-graphic action, and ordinary dramatic conflict.",
    ]
    if adult_hard:
        lines.append("- Adult-oriented items are hard-blocked for minors.")
    lines.append("- Use this as a safety filter; reject only clear age-safety violations, otherwise allow broadly preference-compatible candidates.")
    return "\n".join(lines)

def parse_rag_json(stdout: str) -> str:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return str(json.loads(line).get("rule_text", "")).strip()
    marker = "Generated English regulatory rules:"
    if marker in stdout:
        return stdout.split(marker, 1)[1].strip()
    return stdout.strip()


def get_rule(args, uid: int, iid: int, profile: Dict[str, Any], dims: List[str], adult_hard: int, local_cache: Dict[Tuple, str]) -> str:
    key = ("minor" if is_minor_user(profile) else "adult", tuple(sorted(dims)), int(adult_hard), args.minor_block_at, args.adult_block_at, args.isadult_policy)
    if key in local_cache:
        return local_cache[key]
    if args.rule_mode == "template":
        rule = template_rule(is_minor_user(profile), dims, bool(adult_hard))
    else:
        cmd = [
            sys.executable, args.rag_script, "gen-rule",
            "--profiles", args.profiles,
            "--item_safe", args.item_safe,
            "--user_id", str(uid),
            "--item_id", str(iid),
            "--vector_dir", args.vector_dir,
            "--model_path", args.model_path,
            "--cache_path", args.rag_cache_path,
            "--cache_strategy", args.rule_cache_strategy,
            "--topk", str(args.rag_topk),
            "--topn", str(args.rag_topn),
            "--minor_block_at", str(args.minor_block_at),
            "--adult_block_at", str(args.adult_block_at),
            "--isadult_policy", args.isadult_policy,
            "--device", args.rag_device,
            "--dtype", args.rag_dtype,
            "--json_only",
        ]
        if args.allow_template_fallback:
            cmd.append("--allow_template_fallback")
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            if args.require_rag_success:
                raise RuntimeError(f"RAG failed for uid={uid}, iid={iid}\nCMD={' '.join(cmd)}\nSTDERR={p.stderr}\nSTDOUT={p.stdout}")
            rule = template_rule(is_minor_user(profile), dims, bool(adult_hard))
        else:
            rule = parse_rag_json(p.stdout)
            if not rule and args.require_rag_success:
                raise RuntimeError(f"RAG returned empty rule for uid={uid}, iid={iid}\nSTDOUT={p.stdout}\nSTDERR={p.stderr}")
            if not rule:
                rule = template_rule(is_minor_user(profile), dims, bool(adult_hard))
    local_cache[key] = rule
    return rule


def build_prompt(profile: Dict[str, Any], train_rows: List[Dict[str, Any]], item_map: Dict[int, Dict[str, Any]], candidate: Dict[str, Any], rule_text: str, history_window: int) -> Tuple[str, str]:
    instruction = (
        "You are a preference-and-safety pseudo-labeler for a movie recommender. "
        "Output exactly one digit: 1 if the candidate movie is both likely to match the user's preference and age-appropriate under the safety policy; otherwise output 0."
    )
    input_text = "\n\n".join([
        profile_text(profile),
        history_text(profile, train_rows, item_map, history_window),
        candidate_text(candidate),
        "Age-aware Safety Policy:",
        rule_text,
        "Answer with exactly one digit: 0 or 1."
    ])
    return instruction, input_text


def check_no_leakage(text: str) -> List[str]:
    low = text.lower()
    hits = []
    for p in FORBIDDEN_VISIBLE:
        if p.lower() in low:
            hits.append(p)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--train_safe", required=True)
    ap.add_argument("--item_safe", required=True)
    ap.add_argument("--valid_interactions", default="")
    ap.add_argument("--test_interactions", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--candidates_output", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--history_window", type=int, default=5)
    ap.add_argument("--pref_weight", type=float, default=0.85)
    ap.add_argument("--pop_weight", type=float, default=0.15)
    ap.add_argument("--minor_block_at", type=float, default=3.0)
    ap.add_argument("--adult_block_at", type=float, default=4.0)
    ap.add_argument("--isadult_policy", choices=["minor_only", "all", "none"], default="minor_only")
    ap.add_argument("--rule_mode", choices=["template", "rag_cli"], default="rag_cli")
    ap.add_argument("--rag_script", default="scripts/psg_regulation_rag_full.py")
    ap.add_argument("--vector_dir", default="outputs/psg/rag_vector")
    ap.add_argument("--model_path", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--rag_cache_path", default="outputs/psg_dim5_v4/rag_vector/rule_text_cache_dim5_pairaware_v4_2.jsonl")
    ap.add_argument("--rule_cache_strategy", choices=["signature", "pair", "none"], default="signature")
    ap.add_argument("--rag_topk", type=int, default=5)
    ap.add_argument("--rag_topn", type=int, default=80)
    ap.add_argument("--rag_device", default="cuda")
    ap.add_argument("--rag_dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")
    ap.add_argument("--strict_no_leakage", action="store_true")
    ap.add_argument("--require_rag_success", action="store_true")
    ap.add_argument("--allow_template_fallback", action="store_true")
    ap.add_argument("--max_users", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    random.seed(args.seed)
    profiles = load_profiles(args.profiles)
    train_rows = read_csv_rows(args.train_safe)
    train_by_user = group_by_user(train_rows)
    item_map = load_item_map(args.item_safe)
    # Train-only candidate generation. We keep --valid_interactions and
    # --test_interactions as deprecated no-op arguments for compatibility, but
    # we intentionally do not read them and do not use them to filter candidates.
    if args.valid_interactions or args.test_interactions:
        print("[warn] --valid_interactions/--test_interactions are ignored in train-only mode; no held-out interactions are read.", flush=True)
    valid_pos: Dict[int, Set[int]] = {}
    test_pos: Dict[int, Set[int]] = {}
    print(f"[load] profiles={len(profiles)} train_rows={len(train_rows)} item_safe={len(item_map)} valid_users=0 test_users=0 train_only=True")

    pop = Counter(row_item_id(r) for r in train_rows if row_item_id(r) >= 0)
    max_pop = max(pop.values()) if pop else 1
    all_items = sorted(item_map.keys())

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.candidates_output) or ".", exist_ok=True)
    local_rule_cache: Dict[Tuple, str] = {}
    stats = Counter()
    users = sorted(profiles.keys())
    if args.max_users and args.max_users > 0:
        users = users[:args.max_users]

    with open(args.output, "w", encoding="utf-8") as fout, open(args.candidates_output, "w", encoding="utf-8") as fcand:
        for idx_u, uid in enumerate(users, 1):
            profile = profiles[uid]
            train_seen = set(row_item_id(r) for r in train_by_user.get(uid, []) if row_item_id(r) >= 0)
            hist_seen = set(profile_history_item_ids(profile, args.history_window))
            block = set(train_seen) | set(hist_seen)
            ug = user_genres(profile, train_by_user.get(uid, []), item_map, args.history_window)
            scored = []
            for iid in all_items:
                if iid in block:
                    continue
                row = item_map[iid]
                ig = parse_genres(get_first(row, ["genres", "genre"], ""))
                pref = jaccard(ug, ig)
                pop_score = math.log1p(pop.get(iid, 0)) / max(1e-9, math.log1p(max_pop))
                score = args.pref_weight * pref + args.pop_weight * pop_score
                scored.append((score, pref, pop_score, iid))
            scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            chosen = scored[:args.topn]
            stats["users"] += 1
            stats["candidate_pairs"] += len(chosen)
            for rank, (score, pref_score, pop_score, iid) in enumerate(chosen, 1):
                item_row = item_map[iid]
                dims, adult_flag, adult_hard = dim5_violations(profile, item_row, args.minor_block_at, args.adult_block_at, args.isadult_policy)
                rule_text = get_rule(args, uid, iid, profile, dims, adult_hard, local_rule_cache)
                instruction, input_text = build_prompt(profile, train_by_user.get(uid, []), item_map, item_row, rule_text, args.history_window)
                if args.strict_no_leakage:
                    hits = check_no_leakage(instruction + "\n" + input_text)
                    if hits:
                        raise RuntimeError(f"Visible prompt leakage for uid={uid}, iid={iid}: {hits}")
                meta = {
                    "version": "psg_dim5_v4_4_candidate_top100_trainonly_pairaware_rag",
                    "user_id": uid,
                    "item_id": iid,
                    "candidate_rank": rank,
                    "candidate_score": score,
                    "pref_score": pref_score,
                    "pop_score": pop_score,
                    "triggered_dims": dims,
                    "adult_flag": adult_flag,
                    "adult_hard": adult_hard,
                    "is_minor": is_minor_user(profile),
                    "candidate_policy": "direct_top100_no_mmr_content_pop_train_only",
                }
                fout.write(json.dumps({"instruction": instruction, "input": input_text, "metadata": meta}, ensure_ascii=False) + "\n")
                fcand.write(json.dumps(meta, ensure_ascii=False) + "\n")
            if idx_u == 1 or idx_u % 100 == 0:
                print(f"[progress] users={idx_u}/{len(users)} pairs={stats['candidate_pairs']} local_rule_signatures={len(local_rule_cache)}", flush=True)

    summary = {
        "version": "psg_dim5_v4_4_candidate_top100_trainonly_pairaware_rag",
        "users": stats["users"],
        "candidate_pairs": stats["candidate_pairs"],
        "topn": args.topn,
        "candidate_policy": "direct_top100_no_mmr_content_pop_train_only",
        "train_only": True,
        "heldout_filtering": "none; validation/test interactions are not read",
        "pref_weight": args.pref_weight,
        "pop_weight": args.pop_weight,
        "local_rule_signatures": len(local_rule_cache),
        "rule_mode": args.rule_mode,
        "rag_cache_path": args.rag_cache_path,
        "minor_block_at": args.minor_block_at,
        "adult_block_at": args.adult_block_at,
        "isadult_policy": args.isadult_policy,
    }
    write_json(args.summary_json, summary)
    print(f"[write] {args.output}")
    print(f"[write] {args.candidates_output}")
    print(f"[write] {args.summary_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
