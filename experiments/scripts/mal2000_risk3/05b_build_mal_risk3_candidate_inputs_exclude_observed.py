#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERSION: MAL-Risk3-v1.0-trainonly-direct-top100-pair-aware-RAG
SERVER FILE NAME: scripts/mal2000_risk3/05_build_mal_risk3_candidate_inputs.py

Purpose
-------
Build PSG inference inputs for MAL risk3, following the ML-1M candidate-input
pipeline as closely as possible:

  - train-only candidate construction
  - validation/test interactions are not read
  - candidate pool = all_items - train_observed_items - profile_last5_items
  - direct top100 candidate proposal, no MMR, no backbone score
  - candidate_score = pref_weight * profile/history genre similarity
                    + pop_weight  * train popularity score
  - every selected candidate obtains age-aware rule_text through the MAL RAG
    gen-rule interface, with local signature cache to avoid repeated calls
  - output JSONL contains instruction + input + metadata, no output label

MAL risk3 policy
----------------
  minor users:
    R17 / RPLUS / RX are safety-triggered
  adult users:
    RPLUS / RX are safety-triggered
    R17 is treated as general compliance, not automatically unsafe
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

FORBIDDEN_VISIBLE = [
    "inner_user_id", "inner_item_id", "raw_user_id", "raw_item_id",
    "score_hidden", "rating_hidden", "preference_label", "safety_label",
    "quadrant_hidden", "triggered_risks_hidden", "unsafe_reasons_hidden",
    "candidate_score", "pref_score", "pop_score",
]


def clean_text(x: Any, max_chars: int = 800) -> str:
    s = str(x or "").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if max_chars and len(s) > max_chars:
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


def str_to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "1.0", "true", "yes", "y", "t", "minor", "under18", "under 18"}


def get_first(row: Dict[str, Any], names: Iterable[str], default: str = "") -> str:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return str(row[n])
    return default


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    # utf-8-sig handles BOM in MAL generated CSVs.
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
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
    return safe_int(get_first(row, ["inner_item_id", "item_id", "iid", "item", "anime_inner_id"], "-1"), -1)


def row_timestamp(row: Dict[str, Any]) -> int:
    return safe_int(get_first(row, ["timestamp", "time", "ts", "my_last_updated"], "0"), 0)


def read_rating_file(path: str) -> Tuple[Dict[int, Set[int]], Counter]:
    seen: Dict[int, Set[int]] = defaultdict(set)
    pop = Counter()
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
            if len(arr) < 2:
                continue
            u = safe_int(arr[0], -1)
            i = safe_int(arr[1], -1)
            if u >= 0 and i >= 0:
                seen[u].add(i)
                pop[i] += 1
    return dict(seen), pop


def load_profiles(path: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    skipped = 0
    for obj in read_jsonl(path):
        if obj.get("parse_error"):
            skipped += 1
            continue
        uid = safe_int(obj.get("user_id", obj.get("inner_user_id")), -1)
        if uid >= 0:
            out[uid] = obj
    print(f"[load] profiles={len(out)} skipped={skipped} path={path}")
    return out


def merge_item_maps(items_csv: str, item_safe_csv: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}

    for row in read_csv_rows(items_csv):
        iid = row_item_id(row)
        if iid >= 0:
            out[iid] = dict(row)

    for row in read_csv_rows(item_safe_csv):
        iid = row_item_id(row)
        if iid < 0:
            continue
        base = out.get(iid, {})
        base.update(row)
        out[iid] = base

    print(f"[load] items={len(out)} from items_csv={items_csv} item_safe={item_safe_csv}")
    return out


def parse_list_or_split(x: Any) -> List[str]:
    if isinstance(x, list):
        raw = x
    else:
        raw = re.split(r"[|,;/]", str(x or ""))
    out = []
    seen = set()
    for v in raw:
        s = clean_text(v, 80)
        if not s:
            continue
        k = s.lower()
        if k in {"unknown", "not specified", "none", "nan"}:
            continue
        if k not in seen:
            out.append(s)
            seen.add(k)
    return out


def parse_genres(x: Any) -> Set[str]:
    out = set()
    for g in parse_list_or_split(x):
        k = g.lower().replace("sci-fi", "sci fi").replace("children's", "children")
        if k:
            out.add(k)
    return out


def profile_history_item_ids(profile: Dict[str, Any], k: int) -> List[int]:
    ids = profile.get("history_item_ids", []) or profile.get("profile_item_ids", []) or []
    out = []
    for x in ids[-k:]:
        iid = safe_int(x, -1)
        if iid >= 0:
            out.append(iid)
    return out


def is_minor_user(profile: Dict[str, Any]) -> bool:
    ui = profile.get("user_info", {}) if profile else {}
    if "is_minor" in ui:
        return bool(ui.get("is_minor"))
    for c in ["is_minor", "minor", "isMinor"]:
        if c in profile:
            return str_to_bool(profile[c])
    age = ui.get("age", None)
    if age is not None:
        return safe_int(age, 99) < 18
    age_desc = str(ui.get("age_desc", ui.get("age_group", ""))).lower()
    return "minor" in age_desc or "under" in age_desc


def user_genres(profile: Dict[str, Any], item_map: Dict[int, Dict[str, Any]], history_window: int) -> Set[str]:
    p = profile.get("profile", {})
    g = set()

    for key in ["preferred_genres", "preferred_themes"]:
        val = p.get(key, [])
        if isinstance(val, list):
            g.update(parse_genres("|".join(str(x) for x in val)))
        else:
            g.update(parse_genres(val))

    for iid in profile_history_item_ids(profile, history_window):
        row = item_map.get(iid, {})
        g.update(parse_genres(get_first(row, ["genres", "genre"], "")))

    return g


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def normalize_rating_bucket(row: Dict[str, Any]) -> str:
    rb = str(get_first(row, ["rating_bucket"], "")).strip().upper()
    rating = str(get_first(row, ["rating", "content_rating"], "")).strip().upper()
    blob = (rb + " " + rating).replace(" ", "")

    def is_one(x: Any) -> bool:
        return str(x).strip().lower() in {"1", "1.0", "true", "yes", "y"}

    if is_one(get_first(row, ["rx_code", "RX", "rx", "is_rx"], "")) or "RX" in blob or "HENTAI" in blob:
        return "RX"
    if is_one(get_first(row, ["rplus_code", "RPLUS", "rplus", "is_rplus"], "")) or "RPLUS" in blob or "R+" in blob or "MILDNUDITY" in blob:
        return "RPLUS"
    if is_one(get_first(row, ["r17_code", "R17", "r17", "is_r17"], "")) or "R17" in blob or "R-17" in blob or "17+" in blob:
        return "R17"
    if "PG-13" in rating or "PG13" in blob or "TEENS13" in blob:
        return "PG13"
    if rb == "PG" or "PG-CHILDREN" in blob or "CHILDREN" in blob:
        return "PG"
    if rb == "G" or "ALLAGES" in blob:
        return "G"
    return rb or "OTHER"


def triggered_risks(profile: Dict[str, Any], row: Dict[str, Any]) -> List[str]:
    bucket = normalize_rating_bucket(row)
    minor = is_minor_user(profile)
    risks = []
    if bucket == "R17" and minor:
        risks.append("R17")
    if bucket == "RPLUS":
        risks.append("RPLUS")
    if bucket == "RX":
        risks.append("RX")
    return risks


def profile_to_text(profile: Dict[str, Any]) -> str:
    p = profile.get("profile", {})
    ui = profile.get("user_info", {})

    desc = clean_text(p.get("user_description", ""), 240)
    if not desc:
        age = ui.get("age", "")
        if age != "":
            desc = f"The user is {'a minor' if is_minor_user(profile) else 'an adult'} anime viewer aged {age}."
        else:
            desc = f"The user is {'a minor' if is_minor_user(profile) else 'an adult'} anime viewer."

    def list_text(x: Any, max_items: int = 6) -> str:
        vals = parse_list_or_split(x)
        return ", ".join(vals[:max_items]) if vals else "Not specified"

    return "\n".join([
        "User Information:",
        desc,
        "",
        "User Preference Profile:",
        f"- Summary: {clean_text(p.get('profile_text', ''), 750) or 'Not specified'}",
        f"- Preferred genres: {list_text(p.get('preferred_genres', []), 6)}",
        f"- Preferred themes: {list_text(p.get('preferred_themes', []), 6)}",
        f"- Preferred style: {list_text(p.get('preferred_style', []), 5)}",
    ])


def item_title(row: Dict[str, Any]) -> str:
    return clean_text(get_first(row, ["title", "name", "anime_title", "title_english"], "Unknown title"), 180)


def item_genres(row: Dict[str, Any]) -> str:
    return clean_text(get_first(row, ["genres", "genre"], "Unknown genres"), 240)


def item_type(row: Dict[str, Any]) -> str:
    return clean_text(get_first(row, ["type"], "Unknown"), 80)


def item_source(row: Dict[str, Any]) -> str:
    return clean_text(get_first(row, ["source"], "Unknown"), 100)


def item_rating(row: Dict[str, Any]) -> str:
    return clean_text(get_first(row, ["rating", "content_rating"], "Unknown content rating"), 160)


def history_to_text(profile: Dict[str, Any], item_map: Dict[int, Dict[str, Any]], history_window: int) -> str:
    lines = ["Recent User History:"]
    hist_ids = profile_history_item_ids(profile, history_window)
    for idx, iid in enumerate(hist_ids[-history_window:], 1):
        row = item_map.get(iid, {})
        lines.append(
            f"{idx}. Title: {item_title(row)}\n"
            f"   Genres: {item_genres(row)}\n"
            f"   Type: {item_type(row)}\n"
            f"   Source: {item_source(row)}"
        )
    if not hist_ids:
        lines.append("No recent history available.")
    return "\n".join(lines)


def candidate_to_text(row: Dict[str, Any]) -> str:
    return "\n".join([
        "Candidate Anime:",
        f"Title: {item_title(row)}",
        f"Genres: {item_genres(row)}",
        f"Type: {item_type(row)}",
        f"Source: {item_source(row)}",
        f"Content rating: {item_rating(row)}",
    ])


def make_instruction() -> str:
    return (
        "You are a preference-and-safety pseudo sample generator for anime recommendation. "
        "Given the user's profile, recent history, retrieved age-aware safety policy, and a candidate anime, "
        "decide whether the candidate can be used as a pseudo-positive training sample for this user. "
        "Output only 1 or 0. Output 1 only if BOTH conditions are satisfied: "
        "(1) the candidate is broadly compatible with the user's anime preferences inferred from the profile and recent history; "
        "and (2) the candidate is acceptable under the age-aware safety policy. "
        "Output 0 if the candidate is preference-incompatible, unsafe, age-inappropriate, or violates the retrieved policy. "
        "Do not predict a numeric user score, and do not explain your answer. Answer only 1 or 0."
    )


def build_prompt(profile: Dict[str, Any], item_map: Dict[int, Dict[str, Any]], candidate: Dict[str, Any], rule_text: str, history_window: int) -> Tuple[str, str]:
    input_text = "\n\n".join([
        profile_to_text(profile),
        history_to_text(profile, item_map, history_window),
        "Retrieved Age-aware Safety Policy:",
        clean_text(rule_text, 1600),
        candidate_to_text(candidate),
        "Question: Is this candidate BOTH preference-compatible with the user and safe under the age-aware policy, so that it can be used as a pseudo-positive sample? Answer only 1 or 0.",
    ])
    return make_instruction(), input_text


def parse_rag_json(stdout: str) -> Dict[str, Any]:
    text = stdout.strip()
    for m in re.finditer(r"\{", text):
        s = text[m.start():].strip()
        try:
            return json.loads(s)
        except Exception:
            continue
    raise ValueError("Cannot parse JSON from RAG stdout:\n" + text[-3000:])


def get_rule(args: argparse.Namespace, uid: int, iid: int, profile: Dict[str, Any], row: Dict[str, Any], local_cache: Dict[Tuple[Any, ...], str]) -> Tuple[str, str, List[str]]:
    risks = triggered_risks(profile, row)
    bucket = normalize_rating_bucket(row)
    audience = "minor" if is_minor_user(profile) else "adult"
    key = (audience, tuple(sorted(risks)), bucket, args.rag_topk, args.rule_cache_strategy)

    if key in local_cache:
        return local_cache[key], "local_signature_cache", risks

    if args.rule_mode == "template":
        if risks:
            rule = "- Safety should reject this candidate if the content rating indicates age-inappropriate or adult-oriented anime content.\n- Use this policy as a safety filter; otherwise allow broadly preference-compatible candidates."
        else:
            rule = "- Safety should be used as a filter for clear age-safety violations.\n- Ordinary age-appropriate anime themes should not be rejected automatically."
        local_cache[key] = rule
        return rule, "template", risks

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
        "--topn", str(args.rag_topn),
        "--topk", str(args.rag_topk),
        "--device", args.rag_device,
        "--dtype", args.rag_dtype,
        "--json_only",
    ]

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.rag_timeout,
    )

    if p.returncode != 0:
        if args.require_rag_success:
            raise RuntimeError(f"RAG failed uid={uid} iid={iid} returncode={p.returncode}\n{p.stdout[-4000:]}")
        rule = "- Safety should be used as a filter for clear age-safety violations.\n- Otherwise allow broadly preference-compatible candidates."
        local_cache[key] = rule
        return rule, "template_after_rag_fail", risks

    obj = parse_rag_json(p.stdout)
    rule = str(obj.get("rule_text", "")).strip()
    if not rule and args.require_rag_success:
        raise RuntimeError(f"RAG returned empty rule uid={uid} iid={iid}\n{p.stdout[-4000:]}")
    if not rule:
        rule = "- Safety should be used as a filter for clear age-safety violations.\n- Otherwise allow broadly preference-compatible candidates."

    source = "rag_cli"
    local_cache[key] = rule
    return rule, source, risks


def check_no_leakage(text: str) -> List[str]:
    low = text.lower()
    return [x for x in FORBIDDEN_VISIBLE if x.lower() in low]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--items_csv", required=True)
    ap.add_argument("--item_safe", required=True)
    ap.add_argument("--train_rating", required=True)
    ap.add_argument("--interactions_csv", required=True, help="Full original user-item interactions. Used to exclude all observed pairs before top100 candidate selection.")
    ap.add_argument("--exclude_rating", action="append", default=[], help="Extra user-item rating file to exclude, e.g. valid/test. Can be passed multiple times.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--candidates_output", required=True)
    ap.add_argument("--summary_json", required=True)

    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--history_window", type=int, default=5)
    ap.add_argument("--pref_weight", type=float, default=0.85)
    ap.add_argument("--pop_weight", type=float, default=0.15)
    ap.add_argument("--candidate_order", choices=["score", "random"], default="score")
    ap.add_argument("--max_users", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--rule_mode", choices=["template", "rag_cli"], default="rag_cli")
    ap.add_argument("--rag_script", default="scripts/mal2000_risk3/03_psg_regulation_rag_full.py")
    ap.add_argument("--vector_dir", default="outputs/mal2000_risk3_seq50_300/rag_vector")
    ap.add_argument("--model_path", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--rag_cache_path", default="outputs/mal2000_risk3_seq50_300/rag_vector/rule_text_cache_mal_risk3_pairaware.jsonl")
    ap.add_argument("--rule_cache_strategy", choices=["signature", "pair", "none"], default="signature")
    ap.add_argument("--rag_topk", type=int, default=5)
    ap.add_argument("--rag_topn", type=int, default=80)
    ap.add_argument("--rag_timeout", type=int, default=600)
    ap.add_argument("--rag_device", default="cuda")
    ap.add_argument("--rag_dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")
    ap.add_argument("--require_rag_success", action="store_true")
    ap.add_argument("--strict_no_leakage", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    profiles = load_profiles(args.profiles)
    item_map = merge_item_maps(args.items_csv, args.item_safe)
    train_seen, pop = read_rating_file(args.train_rating)

    # Exclude all original observed interactions for each user.
    # This includes liked train/valid/test items and rejected/non-liked rated items.
    # Popularity is still computed only from train_rating, so held-out interactions
    # are not used as a ranking signal.
    observed_seen: Dict[int, Set[int]] = defaultdict(set)
    for r in read_csv_rows(args.interactions_csv):
        u = row_user_id(r)
        i = row_item_id(r)
        if u >= 0 and i >= 0:
            observed_seen[u].add(i)

    # Extra exclusion files, mainly valid/test, as a safety double-check.
    exclude_seen: Dict[int, Set[int]] = defaultdict(set)
    for ep in args.exclude_rating or []:
        ex_seen, _ = read_rating_file(ep)
        for u, items in ex_seen.items():
            exclude_seen[u].update(items)

    max_pop = max(pop.values()) if pop else 1
    all_items = sorted(item_map.keys())

    users = sorted(profiles.keys())
    if args.max_users and args.max_users > 0:
        users = users[:args.max_users]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.candidates_output) or ".", exist_ok=True)

    local_rule_cache: Dict[Tuple[Any, ...], str] = {}
    stats = Counter()
    risk_counts = Counter()
    rule_counts = Counter()

    print(f"[info] users={len(users)} all_items={len(all_items)} train_users={len(train_seen)} observed_users={len(observed_seen)}")
    print(f"[info] candidate_pool = all_items - full_user_observed_interactions - extra_exclude_rating - profile_last{args.history_window}")
    print(f"[info] interactions_csv={args.interactions_csv}")
    print(f"[info] exclude_rating={args.exclude_rating}")
    print(f"[info] score = {args.pref_weight} * genre_jaccard + {args.pop_weight} * train_popularity")
    print(f"[info] rule_mode={args.rule_mode}, require_rag_success={args.require_rag_success}")

    with open(args.output, "w", encoding="utf-8") as fout, open(args.candidates_output, "w", encoding="utf-8") as fcand:
        for pos_u, uid in enumerate(users, 1):
            profile = profiles[uid]
            seen = set(train_seen.get(uid, set()))
            observed = set(observed_seen.get(uid, set()))
            extra_excluded = set(exclude_seen.get(uid, set()))
            hist = set(profile_history_item_ids(profile, args.history_window))

            # Main change:
            # select top100 only after removing this user's full original observed interactions.
            blocked = seen | observed | extra_excluded | hist

            ug = user_genres(profile, item_map, args.history_window)

            scored = []
            for iid in all_items:
                if iid in blocked:
                    continue
                row = item_map[iid]
                ig = parse_genres(get_first(row, ["genres", "genre"], ""))
                pref = jaccard(ug, ig)
                pop_score = math.log1p(pop.get(iid, 0)) / max(1e-9, math.log1p(max_pop))
                score = args.pref_weight * pref + args.pop_weight * pop_score
                scored.append((score, pref, pop_score, iid))

            if args.candidate_order == "random":
                rng.shuffle(scored)
            else:
                scored.sort(key=lambda x: (x[0], x[1], x[2], -x[3]), reverse=True)

            chosen = scored[:args.topn]
            stats["users"] += 1
            stats["candidate_pairs"] += len(chosen)

            for rank, (score, pref_score, pop_score, iid) in enumerate(chosen, 1):
                row = item_map[iid]
                rule_text, rule_source, risks = get_rule(args, uid, iid, profile, row, local_rule_cache)
                instruction, input_text = build_prompt(profile, item_map, row, rule_text, args.history_window)

                warnings = check_no_leakage(instruction + "\n" + input_text)
                if warnings and args.strict_no_leakage:
                    raise RuntimeError(f"Visible leakage uid={uid} iid={iid}: {warnings}\n{input_text[:1200]}")

                bucket = normalize_rating_bucket(row)
                meta = {
                    "version": "mal_risk3_v1_candidate_top100_exclude_observed_pairaware_rag",
                    "user_id": int(uid),
                    "item_id": int(iid),
                    "candidate_rank": int(rank),
                    "candidate_score": float(score),
                    "pref_score": float(pref_score),
                    "pop_score": float(pop_score),
                    "is_minor": bool(is_minor_user(profile)),
                    "rating_bucket": bucket,
                    "triggered_risks": risks,
                    "rule_source": rule_source,
                    "visible_leakage_warnings": warnings,
                    "candidate_policy": "direct_top100_no_mmr_content_pop_exclude_user_observed",
                }
                fout.write(json.dumps({"instruction": instruction, "input": input_text, "metadata": meta}, ensure_ascii=False) + "\n")
                fcand.write(json.dumps(meta, ensure_ascii=False) + "\n")

                risk_counts[",".join(risks) if risks else "general"] += 1
                rule_counts[rule_source] += 1

            if pos_u == 1 or pos_u % 100 == 0 or pos_u == len(users):
                print(f"[progress] users={pos_u}/{len(users)} pairs={stats['candidate_pairs']} local_rule_signatures={len(local_rule_cache)}", flush=True)

    summary = {
        "version": "mal_risk3_v1_candidate_top100_exclude_observed_pairaware_rag",
        "users": int(stats["users"]),
        "candidate_pairs": int(stats["candidate_pairs"]),
        "topn": args.topn,
        "candidate_policy": "direct_top100_no_mmr_content_pop_exclude_user_observed",
        "train_only": True,
        "exclude_user_observed_interactions": True,
        "heldout_filtering": "full interactions_csv and explicit valid/test exclude_rating are removed before top100 selection",
        "pref_weight": args.pref_weight,
        "pop_weight": args.pop_weight,
        "history_window": args.history_window,
        "local_rule_signatures": len(local_rule_cache),
        "rule_mode": args.rule_mode,
        "rule_source_counts": dict(rule_counts),
        "risk_counts": dict(risk_counts),
        "rag_cache_path": args.rag_cache_path,
        "strict_no_leakage": bool(args.strict_no_leakage),
    }
    write_json(args.summary_json, summary)

    print(f"[write] output={args.output}")
    print(f"[write] candidates_output={args.candidates_output}")
    print(f"[write] summary_json={args.summary_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
