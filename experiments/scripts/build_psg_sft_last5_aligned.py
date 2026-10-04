#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERSION: PSG-Dim5-v4.3-SFT-target56k-pair-aware-RAG
SERVER FILE NAME: scripts/build_psg_sft_last5_aligned.py

Purpose
-------
Build PSG SFT data from TRAIN interactions only, excluding the last-K items used
in each user's profile/history. This v4.3 version keeps the original server file
name but updates the safety rule to the new Dim5 metric and restores the old
sample-target controls so the final SFT size stays around the previous 56k scale.

Label definition
----------------
Preference:
  preference_label = 1 if rating >= --pos_rating
  preference_label = 0 if rating <= --neg_rating
  otherwise dropped as neutral.

Safety under Dim5:
  risk dimensions = sex_code, violence_code, profanity_code, drug_code, intense_code
  minor: a dimension violates if item_risk >= --minor_block_at (default 3)
  adult: a dimension violates if item_risk >= --adult_block_at (default 4)
  any violated dimension makes safety_label = 0.
  isAdult is handled separately; default --isadult_policy minor_only makes it a
  hard block only for minor users.

PSG label:
  output = 1 iff preference_label == 1 and safety_label == 1, otherwise 0.

RAG rule generation
-------------------
If --rule_mode rag_cli, this script calls:
  scripts/psg_regulation_rag_full.py gen-rule
with concrete --user_id and --item_id. The RAG script builds a pair-aware hidden
query and reuses rules through a Dim5 signature cache.

Use --rag_after_sampling to avoid calling Qwen for all raw candidates. The script
first collects/samples SFT examples, then attaches RAG rules only to selected
examples.

Recommended run
---------------
cd .
STORY=outputs/psg_dim5_v4
mkdir -p ${STORY}/sft ${STORY}/logs ${STORY}/rag_vector
python -u scripts/build_psg_sft_last5_aligned.py \
  --profiles outputs/psg/profiles/user_profiles_train_last5.jsonl \
  --train_safe outputs/psg/safe_features/ml-1m_safe_features.train.safe.csv \
  --item_safe outputs/psg/safe_features/ml-1m_safe_features.item_safe.csv \
  --output ${STORY}/sft/psg_sft_dim5_v4_pref42_target56k.jsonl \
  --summary_json ${STORY}/sft/summary_dim5_v4_pref42_target56k.json \
  --pos_rating 4 \
  --neg_rating 2 \
  --history_window 5 \
  --candidate_order recent \
  --minor_pos_target -1 \
  --minor_neg_target -1 \
  --adult_pos_target 20000 \
  --adult_neg_target 20000 \
  --neg_mix 0.4,0.4,0.2 \
  --minor_block_at 3 \
  --adult_block_at 4 \
  --isadult_policy minor_only \
  --rule_mode rag_cli \
  --rag_script scripts/psg_regulation_rag_full.py \
  --vector_dir outputs/psg/rag_vector \
  --model_path models/Qwen2.5-7B-Instruct \
  --rag_cache_path ${STORY}/rag_vector/rule_text_cache_dim5_pairaware_v4_2.jsonl \
  --rule_cache_strategy signature \
  --rag_device cuda \
  --rag_dtype bf16 \
  --rag_after_sampling \
  --strict_no_leakage \
  --require_rag_success \
  2>&1 | tee ${STORY}/logs/build_sft_dim5_v4_pref42_target56k.log
"""

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

RISK_DIMS = ["sex_code", "violence_code", "profanity_code", "drug_code", "intense_code"]
ADULT_DIM = "isAdult"

FORBIDDEN_VISIBLE_PATTERNS = [
    "sex_code", "violence_code", "profanity_code", "drug_code", "intense_code", "isAdult",
    "tol_", "p75", "risk score", "risk_score", "safe_observed_label", "within_user_tolerance",
    "triggered_dims", "inner_item_id", "raw_movie_id", "raw_user_id", "rating_hidden",
    "preference_label", "safety_label", "risk level 3", "risk level 4", "five risk dimensions",
]


def clean_text(x: Any, max_chars: int = 800) -> str:
    if x is None:
        return ""
    s = str(x).replace("\r", " ").replace("\n", " ")
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
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t", "minor", "under 18", "under18"}


def get_first_existing(row: Dict[str, Any], names: Iterable[str], default: str = "") -> str:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return str(row[name])
    return default


def list_to_text(x: Any, max_items: int = 6) -> str:
    if isinstance(x, list):
        arr = [clean_text(v, 80) for v in x if clean_text(v, 80)]
    elif isinstance(x, str) and x.strip():
        arr = [clean_text(v, 80) for v in re.split(r"[,;/|]", x) if clean_text(v, 80)]
    else:
        arr = []
    out, seen = [], set()
    for v in arr:
        key = v.lower()
        if key not in seen:
            out.append(v)
            seen.add(key)
    return ", ".join(out[:max_items]) if out else "Not specified"


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found: {path}")
        for row in reader:
            rows.append(row)
    print(f"[load] csv rows={len(rows)} path={path}")
    return rows


def read_jsonl_profiles(path: str) -> Dict[int, Dict[str, Any]]:
    profiles: Dict[int, Dict[str, Any]] = {}
    skipped = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if obj.get("parse_error"):
                    skipped += 1
                    continue
                uid = safe_int(obj.get("user_id", obj.get("inner_user_id")), -1)
                if uid >= 0:
                    profiles[uid] = obj
            except Exception:
                skipped += 1
    print(f"[load] profiles={len(profiles)} skipped={skipped} path={path}")
    return profiles


def row_user_id(row: Dict[str, Any]) -> int:
    return safe_int(get_first_existing(row, ["inner_user_id", "user_id", "uid", "user"]), -1)


def row_item_id(row: Dict[str, Any]) -> int:
    return safe_int(get_first_existing(row, ["inner_item_id", "item_id", "iid", "item", "movie_inner_id"]), -1)


def row_timestamp(row: Dict[str, Any]) -> int:
    return safe_int(get_first_existing(row, ["timestamp", "time", "ts"]), 0)


def group_by_user(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_user: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        uid = row_user_id(row)
        if uid >= 0:
            by_user[uid].append(row)
    for uid in by_user:
        by_user[uid].sort(key=lambda r: (row_timestamp(r), row_item_id(r)))
    return dict(by_user)


def read_item_safe(path: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    for row in read_csv_rows(path):
        iid = row_item_id(row)
        if iid >= 0:
            out[iid] = row
    print(f"[load] item_safe items={len(out)}")
    return out


def get_user_is_minor(profile_obj: Dict[str, Any]) -> bool:
    ui = profile_obj.get("user_info", {}) if profile_obj else {}
    if "is_minor" in ui:
        return bool(ui.get("is_minor"))
    for c in ["is_minor", "minor", "isMinor"]:
        if c in profile_obj:
            return str_to_bool(profile_obj[c])
    age_desc = str(ui.get("age_desc", ui.get("age_group", ""))).lower()
    if "under" in age_desc or "minor" in age_desc:
        return True
    age = ui.get("age", None)
    if age is not None:
        a = safe_int(age, 99)
        return a < 18 or a == 1
    return False


def profile_history_item_ids(profile_obj: Dict[str, Any], k: int) -> List[int]:
    ids = profile_obj.get("history_item_ids", []) or profile_obj.get("profile_item_ids", []) or []
    out = []
    for x in ids[-k:]:
        iid = safe_int(x, -1)
        if iid >= 0:
            out.append(iid)
    return out


def item_text_from_row_or_map(row_or_item: Dict[str, Any], item_map: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    iid = row_item_id(row_or_item)
    base = item_map.get(iid, {})
    raw_movie_id = get_first_existing(row_or_item, ["raw_movie_id", "movie_id", "raw_item_id", "movieId"], "")
    if not raw_movie_id:
        raw_movie_id = get_first_existing(base, ["raw_movie_id", "movie_id", "raw_item_id", "movieId"], "")
    title = get_first_existing(row_or_item, ["title", "movie_title", "primaryTitle", "originalTitle", "name"], "")
    if not title:
        title = get_first_existing(base, ["title", "movie_title", "primaryTitle", "originalTitle", "name"], "")
    genres = get_first_existing(row_or_item, ["genres", "genre"], "")
    if not genres:
        genres = get_first_existing(base, ["genres", "genre"], "")
    overview = get_first_existing(row_or_item, ["overview", "plot", "description", "movie_overview"], "")
    if not overview:
        overview = get_first_existing(base, ["overview", "plot", "description", "movie_overview"], "")
    risk_row = dict(base) if base else dict(row_or_item)
    return {
        "inner_item_id": iid,
        "raw_movie_id": str(raw_movie_id),
        "title": clean_text(title, 180),
        "genres": clean_text(genres, 220),
        "overview": clean_text(overview, 900),
        "risk_row": risk_row,
    }


def get_risk_value(risk_row: Dict[str, Any], dim: str) -> float:
    if dim == ADULT_DIM:
        return safe_float(get_first_existing(risk_row, ["isAdult", "is_adult", "adult", "isadult"], "0"), 0.0)
    return safe_float(get_first_existing(risk_row, [dim, dim.replace("_code", ""), "item_" + dim, "risk_" + dim], "0"), 0.0)


def compute_safety_dim5(profile_obj: Dict[str, Any], candidate: Dict[str, Any], args: argparse.Namespace) -> Tuple[int, List[str], List[str]]:
    is_minor = get_user_is_minor(profile_obj)
    risk_row = candidate.get("risk_row", {})
    triggered: List[str] = []
    reasons: List[str] = []

    adult_flag = get_risk_value(risk_row, ADULT_DIM) >= 1
    if adult_flag and args.isadult_policy in {"all", "all_unsafe"}:
        triggered.append(ADULT_DIM)
        reasons.append("adult_content_hard_block")
        return 0, sorted(set(triggered)), reasons
    if adult_flag and is_minor and args.isadult_policy == "minor_only":
        triggered.append(ADULT_DIM)
        reasons.append("minor_isAdult_hard_block")
        return 0, sorted(set(triggered)), reasons

    block_at = float(args.minor_block_at if is_minor else args.adult_block_at)
    for dim in RISK_DIMS:
        if get_risk_value(risk_row, dim) >= block_at:
            triggered.append(dim)
    if triggered:
        reasons.append("dim5_any_dimension_violation")
        return 0, sorted(set(triggered)), reasons
    return 1, [], []


def get_rating(row: Dict[str, Any]) -> Optional[float]:
    val = get_first_existing(row, ["rating", "rate", "score"], "")
    if val == "":
        return None
    return safe_float(val, 0.0)


def get_preference_label(rating: Optional[float], pos_rating: float, neg_rating: float) -> Optional[int]:
    if rating is None:
        return None
    if rating >= pos_rating:
        return 1
    if rating <= neg_rating:
        return 0
    return None


def profile_to_prompt_text(profile_obj: Dict[str, Any]) -> str:
    p = profile_obj.get("profile", {})
    ui = profile_obj.get("user_info", {})
    desc = clean_text(p.get("user_description", ""), 240)
    if not desc:
        age_desc = clean_text(ui.get("age_desc", ui.get("age_group", "unknown age")), 80)
        gender_desc = clean_text(ui.get("gender_desc", ui.get("gender", "viewer")), 80)
        desc = f"The user is a {gender_desc} viewer, age group: {age_desc}."
    profile_text = clean_text(p.get("profile_text", ""), 750)
    genres = list_to_text(p.get("preferred_genres", []), 6)
    themes = list_to_text(p.get("preferred_themes", []), 6)
    style = list_to_text(p.get("preferred_style", []), 5)
    return "\n".join([
        "User Information:",
        desc,
        "",
        "User Preference Profile:",
        f"- Summary: {profile_text if profile_text else 'Not specified'}",
        f"- Preferred genres: {genres}",
        f"- Preferred themes: {themes}",
        f"- Preferred style: {style}",
    ]).strip()


def history_to_prompt_text(profile_obj: Dict[str, Any], train_rows: List[Dict[str, Any]], item_map: Dict[int, Dict[str, Any]], history_window: int) -> str:
    hist_ids = profile_history_item_ids(profile_obj, history_window)
    by_iid = {row_item_id(r): r for r in train_rows}
    lines = ["Recent User History:"]
    if not hist_ids:
        hist_ids = [row_item_id(r) for r in train_rows[-history_window:]]
    for idx, iid in enumerate(hist_ids[-history_window:], 1):
        row = by_iid.get(iid, {"inner_item_id": iid, "item_id": iid})
        it = item_text_from_row_or_map(row, item_map)
        lines.append(
            f"{idx}. Title: {it['title'] or 'Unknown title'}\n"
            f"   Genres: {it['genres'] or 'Unknown genres'}\n"
            f"   Overview: {it['overview'] or 'No overview available'}"
        )
    return "\n".join(lines)


def candidate_to_prompt_text(candidate: Dict[str, Any]) -> str:
    return "\n".join([
        "Candidate Movie:",
        f"Title: {candidate['title'] or 'Unknown title'}",
        f"Genres: {candidate['genres'] or 'Unknown genres'}",
        f"Overview: {candidate['overview'] or 'No overview available'}",
    ])


def make_instruction() -> str:
    return (
        "You are a preference-and-safety pseudo sample generator for movie recommendation. "
        "Given the user's profile, recent history, retrieved age-aware safety policy, and a candidate movie, "
        "decide whether the candidate can be used as a pseudo-positive training sample for this user. "
        "Output only 1 or 0. Output 1 only if BOTH conditions are satisfied: "
        "(1) the candidate is broadly compatible with the user's movie preferences inferred from the profile and recent history; "
        "and (2) the candidate is acceptable under the age-aware safety policy. "
        "Output 0 if the candidate is preference-incompatible, unsafe, age-inappropriate, or violates the retrieved policy. "
        "Do not predict a numeric rating, and do not explain your answer. Answer only 1 or 0."
    )


def make_template_rules(is_minor: bool, triggered_dims: List[str]) -> str:
    lines = [
        "Retrieved Age-aware Safety Policy:",
        "- Use the policy as a safety filter, not as a rating predictor.",
        "- Reject clear adult-only, explicit, severe, imitable, or highly disturbing harmful content.",
        "- Do not reject ordinary safe themes such as mild romance, family/social themes, ordinary conflict, historical tension, or non-graphic action automatically.",
    ]
    if is_minor:
        lines.append("- For minor users, apply stricter age-appropriateness and block clear adult-only content, but do not over-filter ordinary safe movies.")
    else:
        lines.append("- For adult users, do not apply minor-protection restrictions; reject only clear content-compliance or severe harmful-content violations.")
    if triggered_dims:
        pretty = ", ".join(d.replace("_code", "") for d in triggered_dims)
        lines.append(f"- Pay special attention to the triggered safety topics: {pretty}. Do not expand the rule to unrelated risks unless the movie text clearly supports them.")
    lines.append("- Final principle: output 0 only for clear safety violations or clear broad-preference conflict; otherwise output 1.")
    return "\n".join(lines)


def parse_rules_from_cli(stdout: str) -> Optional[str]:
    if not stdout:
        return None
    text = stdout.replace("\r\n", "\n").replace("\r", "\n")
    markers = ["Loaded English rules from cache:", "Generated English regulatory rules:", "Retrieved Age-aware Safety Policy:", "Retrieved Regulatory Rules:"]
    cand = None
    for m in markers:
        if m in text:
            cand = text.split(m, 1)[1]
            break
    if cand is None:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("- ")]
        if len(lines) >= 2:
            cand = "\n".join(lines)
        else:
            return None
    stop_markers = ["Saved cache:", "Loading Qwen:", "Retrieved Chinese chunks", "Traceback", "User:", "Dense vector RAG:"]
    for sm in stop_markers:
        if sm in cand:
            cand = cand.split(sm, 1)[0]
    lines = []
    for ln in cand.splitlines():
        x = ln.strip()
        if not x:
            continue
        if x.startswith("[") and "Chunk" in x:
            continue
        if x.startswith("-") or x.lower().startswith(("retrieved", "policy")):
            lines.append(x)
    if not lines:
        return None
    joined = "\n".join(lines[:8]).strip()
    if not joined.startswith("Retrieved Age-aware Safety Policy:"):
        joined = "Retrieved Age-aware Safety Policy:\n" + joined
    return joined


def get_rules_text(uid: int, candidate: Dict[str, Any], is_minor: bool, triggered_dims: List[str], args: argparse.Namespace, rule_cache: Dict[Tuple[Any, ...], Tuple[str, str]], rag_state: Counter) -> Tuple[str, str]:
    if args.rule_mode == "template":
        if args.require_rag_success:
            raise RuntimeError("--require_rag_success requires --rule_mode rag_cli")
        return make_template_rules(is_minor, triggered_dims), "template"

    cache_key = ("minor" if is_minor else "adult", tuple(sorted(triggered_dims)), args.minor_block_at, args.adult_block_at, args.isadult_policy)
    if args.rule_cache_strategy == "signature" and cache_key in rule_cache:
        return rule_cache[cache_key]

    if args.max_rag_calls >= 0 and rag_state["calls"] >= args.max_rag_calls:
        if args.require_rag_success:
            raise RuntimeError(f"RAG call limit reached: max_rag_calls={args.max_rag_calls}")
        return make_template_rules(is_minor, triggered_dims), "template_after_rag_limit"

    cmd = [
        sys.executable, args.rag_script, "gen-rule",
        "--profiles", args.profiles,
        "--item_safe", args.item_safe,
        "--user_id", str(uid),
        "--item_id", str(candidate["inner_item_id"]),
        "--vector_dir", args.vector_dir,
        "--model_path", args.model_path,
        "--cache_path", args.rag_cache_path,
        "--cache_strategy", args.rule_cache_strategy,
        "--minor_block_at", str(args.minor_block_at),
        "--adult_block_at", str(args.adult_block_at),
        "--isadult_policy", args.isadult_policy,
        "--topk", str(args.rag_topk),
        "--topn", str(args.rag_topn),
        "--device", args.rag_device,
        "--dtype", args.rag_dtype,
    ]
    if args.rag_no_cache:
        cmd.append("--no_cache")

    rag_state["calls"] += 1
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", timeout=args.rag_timeout, check=False)
        if p.returncode == 0:
            parsed = parse_rules_from_cli(p.stdout)
            if parsed:
                if args.rule_cache_strategy == "signature":
                    rule_cache[cache_key] = (parsed, "rag_cli")
                return parsed, "rag_cli"
            last_error = "RAG CLI returned 0 but parsing failed. Last stdout:\n" + (p.stdout[-2000:] if p.stdout else "")
        else:
            last_error = f"RAG CLI failed returncode={p.returncode}. Last stdout:\n" + (p.stdout[-2000:] if p.stdout else "")
    except Exception as e:
        last_error = repr(e)

    if args.require_rag_success:
        raise RuntimeError(f"RAG failed: uid={uid}, iid={candidate.get('inner_item_id')}, triggered={triggered_dims}. {last_error}")
    rules = make_template_rules(is_minor, triggered_dims)
    if args.cache_template_rules and args.rule_cache_strategy == "signature":
        rule_cache[cache_key] = (rules, "template_cache")
    return rules, "template_after_rag_fail"


def make_input_text(profile_obj: Dict[str, Any], train_rows: List[Dict[str, Any]], item_map: Dict[int, Dict[str, Any]], candidate: Dict[str, Any], rules_text: str, history_window: int) -> str:
    return "\n\n".join([
        profile_to_prompt_text(profile_obj),
        history_to_prompt_text(profile_obj, train_rows, item_map, history_window),
        rules_text,
        candidate_to_prompt_text(candidate),
        "Question: Is this candidate BOTH preference-compatible with the user and safe under the age-aware policy, so that it can be used as a pseudo-positive sample? Answer only 1 or 0.",
    ]).strip()


def leakage_check(text: str) -> List[str]:
    low = text.lower()
    return [p for p in FORBIDDEN_VISIBLE_PATTERNS if p.lower() in low]


def build_candidate_record(uid: int, row: Dict[str, Any], profile_obj: Dict[str, Any], train_rows: List[Dict[str, Any]], item_map: Dict[int, Dict[str, Any]], args: argparse.Namespace, rule_cache: Dict[Tuple[Any, ...], Tuple[str, str]], rag_state: Counter) -> Optional[Dict[str, Any]]:
    rating = get_rating(row)
    pref = get_preference_label(rating, args.pos_rating, args.neg_rating)
    if pref is None:
        return None
    cand = item_text_from_row_or_map(row, item_map)
    if cand["inner_item_id"] < 0:
        return None
    safety, triggered, unsafe_reasons = compute_safety_dim5(profile_obj, cand, args)
    label = 1 if (pref == 1 and safety == 1) else 0
    is_minor = get_user_is_minor(profile_obj)

    if args.rule_mode == "rag_cli" and args.rag_after_sampling:
        rules_text = make_template_rules(is_minor, triggered)
        rule_source = "preselect_only_not_final"
    else:
        rules_text, rule_source = get_rules_text(uid, cand, is_minor, triggered, args, rule_cache, rag_state)
    input_text = make_input_text(profile_obj, train_rows, item_map, cand, rules_text, args.history_window)
    found = leakage_check(input_text)
    if found and args.strict_no_leakage:
        raise RuntimeError(f"Visible input leakage for uid={uid}, iid={cand['inner_item_id']}: {found}\n{input_text[:1000]}")

    if pref == 1 and safety == 0:
        quad = "pref_pos_unsafe"
    elif pref == 0 and safety == 1:
        quad = "pref_neg_safe"
    elif pref == 0 and safety == 0:
        quad = "pref_neg_unsafe"
    else:
        quad = "pref_pos_safe"

    return {
        "instruction": make_instruction(),
        "input": input_text,
        "output": str(int(label)),
        "metadata": {
            "user_id": int(uid),
            "raw_user_id": str(profile_obj.get("user_info", {}).get("raw_user_id", "")),
            "is_minor": bool(is_minor),
            "candidate_inner_item_id": int(cand["inner_item_id"]),
            "candidate_raw_movie_id": str(cand.get("raw_movie_id", "")),
            "candidate_title": cand.get("title", ""),
            "source": "train_before_profile_last5",
            "label": int(label),
            "quadrant_hidden": quad,
            "rating_hidden": rating,
            "preference_label_hidden": int(pref),
            "safety_label_hidden": int(safety),
            "triggered_dims_hidden": triggered,
            "unsafe_reasons_hidden": unsafe_reasons,
            "minor_block_at_hidden": float(args.minor_block_at),
            "adult_block_at_hidden": float(args.adult_block_at),
            "rule_source": rule_source,
            "visible_leakage_warnings": found,
        },
    }


def sample_list(rng: random.Random, arr: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if n < 0 or n >= len(arr):
        return list(arr)
    return rng.sample(arr, n)


def balanced_select_group(rng: random.Random, group: Dict[str, List[Dict[str, Any]]], pos_target: int, neg_target: int, neg_mix: Tuple[float, float, float]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    pos_pool = group.get("pref_pos_safe", [])
    n_pos_avail = len(pos_pool)
    neg_pools = {
        "pref_pos_unsafe": group.get("pref_pos_unsafe", []),
        "pref_neg_safe": group.get("pref_neg_safe", []),
        "pref_neg_unsafe": group.get("pref_neg_unsafe", []),
    }
    n_neg_avail = sum(len(v) for v in neg_pools.values())

    n1 = n_pos_avail if pos_target < 0 else min(pos_target, n_pos_avail)
    n0 = n_neg_avail if neg_target < 0 else min(neg_target, n_neg_avail)
    if pos_target < 0 and neg_target < 0:
        n = min(n1, n0)
        n1 = n0 = n
    elif pos_target < 0:
        n1 = min(n0, n_pos_avail)
    elif neg_target < 0:
        n0 = min(n1, n_neg_avail)

    wanted = {
        "pref_pos_unsafe": int(round(n0 * neg_mix[0])),
        "pref_neg_safe": int(round(n0 * neg_mix[1])),
    }
    wanted["pref_neg_unsafe"] = max(0, n0 - wanted["pref_pos_unsafe"] - wanted["pref_neg_safe"])
    selected_counts = {k: min(wanted[k], len(neg_pools[k])) for k in wanted}
    shortage = n0 - sum(selected_counts.values())
    for k in ["pref_pos_unsafe", "pref_neg_safe", "pref_neg_unsafe"]:
        if shortage <= 0:
            break
        can_add = max(0, len(neg_pools[k]) - selected_counts[k])
        add = min(can_add, shortage)
        selected_counts[k] += add
        shortage -= add

    out = []
    out.extend(sample_list(rng, pos_pool, n1))
    for k, c in selected_counts.items():
        out.extend(sample_list(rng, neg_pools[k], c))
    stats = {"selected_pref_pos_safe": n1}
    for k, c in selected_counts.items():
        stats[f"selected_{k}"] = c
    stats["selected_total"] = len(out)
    return out, stats


def write_jsonl(path: str, rows: List[Dict[str, Any]], no_metadata: bool = False) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            x = dict(r)
            if no_metadata:
                x.pop("metadata", None)
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--train_safe", required=True)
    ap.add_argument("--item_safe", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--output_nometa", default="")
    ap.add_argument("--summary_json", default="")
    ap.add_argument("--history_window", type=int, default=5)
    ap.add_argument("--pos_rating", type=float, default=4.0)
    ap.add_argument("--neg_rating", type=float, default=2.0)
    ap.add_argument("--candidate_order", choices=["recent", "oldest", "random"], default="recent")
    ap.add_argument("--max_users", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--minor_pos_target", type=int, default=-1)
    ap.add_argument("--minor_neg_target", type=int, default=-1)
    ap.add_argument("--adult_pos_target", type=int, default=20000)
    ap.add_argument("--adult_neg_target", type=int, default=20000)
    ap.add_argument("--neg_mix", default="0.4,0.4,0.2")
    ap.add_argument("--balance_2x2", action="store_true", help="Optional strict 4-bucket balancing. Usually not used for target56k.")

    ap.add_argument("--minor_block_at", type=float, default=3.0)
    ap.add_argument("--adult_block_at", type=float, default=4.0)
    ap.add_argument("--isadult_policy", choices=["minor_only", "all", "all_unsafe", "none", "ignore"], default="minor_only")

    ap.add_argument("--rule_mode", choices=["template", "rag_cli"], default="template")
    ap.add_argument("--rag_script", default="scripts/psg_regulation_rag_full.py")
    ap.add_argument("--vector_dir", default="outputs/psg/rag_vector")
    ap.add_argument("--model_path", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--rag_cache_path", default="outputs/psg_dim5_v4/rag_vector/rule_text_cache_dim5_pairaware_v4_2.jsonl")
    ap.add_argument("--rag_topk", type=int, default=5)
    ap.add_argument("--rag_topn", type=int, default=80)
    ap.add_argument("--rag_timeout", type=int, default=300)
    ap.add_argument("--rag_device", default="cuda")
    ap.add_argument("--rag_dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")
    ap.add_argument("--rag_no_cache", action="store_true")
    ap.add_argument("--max_rag_calls", type=int, default=-1)
    ap.add_argument("--rule_cache_strategy", choices=["signature", "pair", "none"], default="signature")
    ap.add_argument("--cache_template_rules", action="store_true")
    ap.add_argument("--rag_after_sampling", action="store_true")
    ap.add_argument("--require_rag_success", action="store_true")
    ap.add_argument("--strict_no_leakage", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    neg_mix_vals = tuple(float(x) for x in args.neg_mix.split(","))
    if len(neg_mix_vals) != 3 or sum(neg_mix_vals) <= 0:
        raise ValueError("--neg_mix must be three values, e.g. 0.4,0.4,0.2")
    sm = sum(neg_mix_vals)
    neg_mix = (neg_mix_vals[0] / sm, neg_mix_vals[1] / sm, neg_mix_vals[2] / sm)

    profiles = read_jsonl_profiles(args.profiles)
    train_rows = read_csv_rows(args.train_safe)
    train_by_user = group_by_user(train_rows)
    item_map = read_item_safe(args.item_safe)

    users = sorted(set(profiles.keys()) & set(train_by_user.keys()))
    if args.max_users and args.max_users > 0:
        users = users[:args.max_users]
    print(f"[info] users={len(users)}")
    print(f"[info] preference: positive >= {args.pos_rating}, negative <= {args.neg_rating}, neutral dropped")
    print(f"[info] target exclusion: profile/history last-{args.history_window} items excluded")
    print(f"[info] safety: minor_block_at={args.minor_block_at}, adult_block_at={args.adult_block_at}, isadult_policy={args.isadult_policy}")
    print(f"[info] sampling: minor_pos={args.minor_pos_target}, minor_neg={args.minor_neg_target}, adult_pos={args.adult_pos_target}, adult_neg={args.adult_neg_target}, neg_mix={neg_mix}")
    print(f"[info] rule_mode={args.rule_mode}, rag_after_sampling={args.rag_after_sampling}, require_rag_success={args.require_rag_success}")

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {"minor": defaultdict(list), "adult": defaultdict(list)}
    stats = Counter()
    rag_state = Counter()
    rule_cache: Dict[Tuple[Any, ...], Tuple[str, str]] = {}
    t0 = time.time()

    for pos, uid in enumerate(users, 1):
        prof = profiles[uid]
        rows = train_by_user[uid]
        hist_ids = set(profile_history_item_ids(prof, args.history_window))
        if not hist_ids and len(rows) >= args.history_window:
            hist_ids = set(row_item_id(r) for r in rows[-args.history_window:])
        target_rows = [r for r in rows if row_item_id(r) not in hist_ids]
        if args.candidate_order == "recent":
            target_rows = sorted(target_rows, key=lambda r: (row_timestamp(r), row_item_id(r)), reverse=True)
        elif args.candidate_order == "oldest":
            target_rows = sorted(target_rows, key=lambda r: (row_timestamp(r), row_item_id(r)))
        else:
            target_rows = list(target_rows)
            rng.shuffle(target_rows)
        scope = "minor" if get_user_is_minor(prof) else "adult"
        for row in target_rows:
            rating = get_rating(row)
            pref = get_preference_label(rating, args.pos_rating, args.neg_rating)
            if pref is None:
                stats[f"{scope}_dropped_neutral_or_missing_rating"] += 1
                continue
            sample = build_candidate_record(uid, row, prof, rows, item_map, args, rule_cache, rag_state)
            if sample is None:
                stats[f"{scope}_skipped_bad_sample"] += 1
                continue
            quad = sample["metadata"]["quadrant_hidden"]
            grouped[scope][quad].append(sample)
            stats[f"{scope}_available_{quad}"] += 1
        stats[f"{scope}_users"] += 1
        stats[f"{scope}_candidate_rows_after_excluding_last5"] += len(target_rows)
        if pos % 200 == 0 or pos == len(users):
            n_avail = sum(len(v) for g in grouped.values() for v in g.values())
            print(f"[progress] {pos}/{len(users)} users, elapsed={(time.time()-t0)/60:.1f}min, samples_available={n_avail}, rag_calls={rag_state['calls']}", flush=True)

    if args.balance_2x2:
        all_by_quad: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for scope in ["minor", "adult"]:
            for q, arr in grouped[scope].items():
                all_by_quad[q].extend(arr)
        min_n = min(len(all_by_quad[q]) for q in ["pref_pos_safe", "pref_pos_unsafe", "pref_neg_safe", "pref_neg_unsafe"])
        samples = []
        for q in ["pref_pos_safe", "pref_pos_unsafe", "pref_neg_safe", "pref_neg_unsafe"]:
            samples.extend(sample_list(rng, all_by_quad[q], min_n))
        minor_sel_stats = {"strict_balance_2x2": True}
        adult_sel_stats = {"strict_balance_2x2": True}
    else:
        selected_minor, minor_sel_stats = balanced_select_group(rng, grouped["minor"], args.minor_pos_target, args.minor_neg_target, neg_mix)
        selected_adult, adult_sel_stats = balanced_select_group(rng, grouped["adult"], args.adult_pos_target, args.adult_neg_target, neg_mix)
        samples = selected_minor + selected_adult
    rng.shuffle(samples)

    if args.rule_mode == "rag_cli" and args.rag_after_sampling:
        print(f"[info] attaching RAG rules after sampling for selected samples={len(samples)}", flush=True)
        rule_cache = {}
        rag_state = Counter()
        t_rag = time.time()
        for idx, sample in enumerate(samples, 1):
            md = sample.get("metadata", {})
            uid = int(md.get("user_id", -1))
            iid = int(md.get("candidate_inner_item_id", -1))
            if uid < 0 or iid < 0:
                continue
            prof = profiles[uid]
            rows = train_by_user[uid]
            cand = item_text_from_row_or_map({"inner_item_id": iid, "item_id": iid}, item_map)
            triggered = md.get("triggered_dims_hidden", []) or []
            is_minor = get_user_is_minor(prof)
            rules_text, rule_source = get_rules_text(uid, cand, is_minor, triggered, args, rule_cache, rag_state)
            sample["input"] = make_input_text(prof, rows, item_map, cand, rules_text, args.history_window)
            sample["metadata"]["rule_source"] = rule_source
            found = leakage_check(sample["input"])
            if found and args.strict_no_leakage:
                raise RuntimeError(f"Visible input leakage after RAG attach for uid={uid}, iid={iid}: {found}\n{sample['input'][:1000]}")
            if idx % 1000 == 0 or idx == len(samples):
                print(f"[rag_attach] {idx}/{len(samples)} selected samples, elapsed={(time.time()-t_rag)/60:.1f}min, rag_calls={rag_state['calls']}, rule_cache_entries={len(rule_cache)}", flush=True)

    if args.rule_mode == "rag_cli" and args.require_rag_success:
        bad_sources = Counter(s.get("metadata", {}).get("rule_source", "") for s in samples if s.get("metadata", {}).get("rule_source", "") != "rag_cli")
        if bad_sources:
            raise RuntimeError(f"Strict RAG-only check failed. Non-RAG rule sources: {dict(bad_sources)}")

    write_jsonl(args.output, samples, no_metadata=False)
    print(f"[save] output={args.output} samples={len(samples)}")
    if args.output_nometa:
        write_jsonl(args.output_nometa, samples, no_metadata=True)
        print(f"[save] output_nometa={args.output_nometa}")

    label_counts = Counter(s["output"] for s in samples)
    scope_counts = Counter("minor" if s["metadata"].get("is_minor") else "adult" for s in samples)
    quad_counts = Counter(s["metadata"].get("quadrant_hidden") for s in samples)
    rule_counts = Counter(s["metadata"].get("rule_source") for s in samples)
    summary = {
        "version": "PSG-Dim5-v4.3-SFT-target56k-pair-aware-RAG",
        "config": vars(args),
        "label_definition": {
            "preference_positive": f"rating >= {args.pos_rating}",
            "preference_negative": f"rating <= {args.neg_rating}",
            "neutral": "dropped",
            "safety_minor": f"unsafe if any risk dimension >= {args.minor_block_at}; numeric threshold hidden from visible prompt",
            "safety_adult": f"unsafe if any risk dimension >= {args.adult_block_at}; numeric threshold hidden from visible prompt",
            "psg_label": "1 iff preference_label=1 and safety_label=1; otherwise 0",
            "last5_exclusion": f"target candidates exclude profile/history last-{args.history_window} items",
            "visible_prompt_excludes": FORBIDDEN_VISIBLE_PATTERNS,
        },
        "availability_stats": dict(stats),
        "selected_stats": {
            "minor": minor_sel_stats,
            "adult": adult_sel_stats,
            "label_counts": dict(label_counts),
            "scope_counts": dict(scope_counts),
            "quadrant_counts": dict(quad_counts),
            "rule_source_counts": dict(rule_counts),
            "total_samples": len(samples),
            "rag_cli_calls": int(rag_state["calls"]),
            "rule_cache_entries": len(rule_cache),
        },
    }
    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[save] summary_json={args.summary_json}")

    print("\nDone.")
    print(json.dumps(summary["selected_stats"], ensure_ascii=False, indent=2))
    if samples:
        preview = {k: samples[0][k] for k in ["instruction", "input", "output"]}
        print("\nPreview sample:")
        print(json.dumps(preview, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
