#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERSION: MAL-Risk3-SFT-last5-aligned-v1
SERVER FILE NAME: scripts/mal2000_risk3/04_build_mal_risk3_sft_last5_aligned.py

Purpose
-------
Build PSG/SFT data for MAL anime recommendation from pre-validation training
history, excluding the last-K items used for user-profile construction.

The output format follows the ML-1M PSG SFT style:
  {"instruction": ..., "input": ..., "output": "0|1", "metadata": {...}}

Label definition
----------------
Preference:
  preference_label = 1 if my_score >= --pos_score, default 7
  preference_label = 0 if 1 <= my_score <= --neg_score, default 6
  score 0 / missing is dropped.

Safety under MAL Risk3:
  risk buckets are R17, RPLUS, RX.
  minor user: unsafe if candidate is R17, RPLUS, or RX.
  adult user: unsafe if candidate is RPLUS or RX. R17 is not unsafe for adults.

PSG/SFT output:
  output = 1 iff preference_label == 1 and safety_label == 1.
  The other three quadrants output 0:
    pref_pos_safe   -> 1
    pref_pos_unsafe -> 0
    pref_neg_safe   -> 0
    pref_neg_unsafe -> 0

Leakage control
---------------
- The last-K profile-history items are excluded from candidate samples.
- Positive candidates are restricted to pairs in Data/<dataset>.train.rating.
- Negative candidates are taken from complete interactions with my_score<=6, but
  by default only at or before the user's latest profile-history timestamp.
- Valid/test positive pairs are never used as positive SFT targets.

RAG rule generation
-------------------
If --rule_mode rag_cli, this script calls:
  scripts/mal2000_risk3/03_psg_regulation_rag_full.py gen-rule --json_only
with concrete --user_id and --item_id. Rules are cached by safety signature:
  minor/general, minor/r17, minor/rplus, minor/rx, adult/general,
  adult/rplus, adult/rx, etc.

Recommended run
---------------
cd .

DATASET=mal2000_risk3_seq50_300
MODEL_PATH=models/Qwen2.5-7B-Instruct
STORY=outputs/${DATASET}

# 1) Build RAG vector index first if needed:
CUDA_VISIBLE_DEVICES=0 python -u scripts/mal2000_risk3/03_psg_regulation_rag_full.py build \
  --pdf_dir regulation_pdfs \
  --out_dir ${STORY}/rag_vector \
  --model_path ${MODEL_PATH} \
  --device cuda \
  --dtype bf16

# 2) Debug SFT construction on 20 users:
CUDA_VISIBLE_DEVICES=0 python -u scripts/mal2000_risk3/04_build_mal_risk3_sft_last5_aligned.py \
  --dataset ${DATASET} \
  --profiles ${STORY}/profiles/${DATASET}.user_profiles.train_last5.jsonl \
  --interactions_csv ${STORY}/tables/${DATASET}.interactions.csv \
  --items_csv ${STORY}/tables/${DATASET}.items.csv \
  --item_safe ${STORY}/safe_features/${DATASET}_features.item_safe.csv \
  --train_rating Data/${DATASET}.train.rating \
  --output ${STORY}/sft/${DATASET}.sft.risk3.last5.debug.jsonl \
  --summary_json ${STORY}/sft/${DATASET}.sft.risk3.last5.debug.summary.json \
  --output_nometa ${STORY}/sft/${DATASET}.sft.risk3.last5.debug.nometa.jsonl \
  --history_window 5 \
  --pos_score 7 \
  --neg_score 6 \
  --candidate_order recent \
  --per_user_quad_cap 10 \
  --balance_4types \
  --max_users 20 \
  --rule_mode rag_cli \
  --rag_script scripts/mal2000_risk3/03_psg_regulation_rag_full.py \
  --vector_dir ${STORY}/rag_vector \
  --model_path ${MODEL_PATH} \
  --rag_cache_path ${STORY}/rag_vector/rule_text_cache_mal_risk3_pairaware.jsonl \
  --rag_after_sampling \
  --require_rag_success \
  --strict_no_leakage \
  --seed 2026

# 3) Full SFT construction:
CUDA_VISIBLE_DEVICES=0 python -u scripts/mal2000_risk3/04_build_mal_risk3_sft_last5_aligned.py \
  --dataset ${DATASET} \
  --profiles ${STORY}/profiles/${DATASET}.user_profiles.train_last5.jsonl \
  --interactions_csv ${STORY}/tables/${DATASET}.interactions.csv \
  --items_csv ${STORY}/tables/${DATASET}.items.csv \
  --item_safe ${STORY}/safe_features/${DATASET}_features.item_safe.csv \
  --train_rating Data/${DATASET}.train.rating \
  --output ${STORY}/sft/${DATASET}.sft.risk3.last5.balanced4.jsonl \
  --summary_json ${STORY}/sft/${DATASET}.sft.risk3.last5.balanced4.summary.json \
  --output_nometa ${STORY}/sft/${DATASET}.sft.risk3.last5.balanced4.nometa.jsonl \
  --history_window 5 \
  --pos_score 7 \
  --neg_score 6 \
  --candidate_order recent \
  --per_user_quad_cap 10 \
  --balance_4types \
  --rule_mode rag_cli \
  --rag_script scripts/mal2000_risk3/03_psg_regulation_rag_full.py \
  --vector_dir ${STORY}/rag_vector \
  --model_path ${MODEL_PATH} \
  --rag_cache_path ${STORY}/rag_vector/rule_text_cache_mal_risk3_pairaware.jsonl \
  --rag_after_sampling \
  --require_rag_success \
  --strict_no_leakage \
  --seed 2026 \
  2>&1 | tee ${STORY}/logs/build_sft_risk3_last5_balanced4.log
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

FORBIDDEN_VISIBLE_PATTERNS = [
    "inner_item_id", "inner_user_id", "raw_item_id", "raw_user_id", "anime_id", "my_score",
    "preference_label", "safety_label", "quadrant", "rating_hidden", "risk_hidden",
    "r17_code", "rplus_code", "rx_code", "rating_code", "isAdult", "unsafe_reasons",
]

RISK_NAMES = ["R17", "RPLUS", "RX"]


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
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t", "minor", "under18", "under 18"}


def get_first_existing(row: Dict[str, Any], names: Iterable[str], default: str = "") -> str:
    for name in names:
        if name in row and row[name] not in (None, "", "nan", "NaN"):
            return str(row[name])
    return default


def split_list_text(x: Any, max_items: int = 8) -> str:
    if isinstance(x, list):
        arr = [clean_text(v, 80) for v in x if clean_text(v, 80)]
    elif isinstance(x, str) and x.strip():
        arr = [clean_text(v, 80) for v in re.split(r"[,;/|]", x) if clean_text(v, 80)]
    else:
        arr = []
    seen, out = set(), []
    for v in arr:
        k = v.lower()
        if k not in seen:
            seen.add(k)
            out.append(v)
    return ", ".join(out[:max_items]) if out else "Not specified"


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found: {path}")

        # Normalize possible BOM/whitespace in CSV headers.
        fieldnames = [str(x).replace("\ufeff", "").strip() for x in reader.fieldnames]

        for row in reader:
            clean_row = {}
            for k, v in row.items():
                kk = str(k).replace("\ufeff", "").strip()
                clean_row[kk] = v
            rows.append(clean_row)

    print(f"[load] csv rows={len(rows)} path={path}")
    if rows:
        print(f"[load] csv columns={list(rows[0].keys())[:30]}")
    return rows


def read_jsonl_profiles(path: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
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
                    out[uid] = obj
            except Exception:
                skipped += 1
    print(f"[load] profiles={len(out)} skipped={skipped} path={path}")
    return out


def read_rating_pairs(path: str) -> set:
    pairs = set()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            parts = re.split(r"[\s,]+", line.strip())
            if len(parts) < 2:
                continue
            u, i = safe_int(parts[0], -1), safe_int(parts[1], -1)
            if u >= 0 and i >= 0:
                pairs.add((u, i))
    print(f"[load] rating pairs={len(pairs)} path={path}")
    return pairs


def row_user_id(row: Dict[str, Any]) -> int:
    return safe_int(get_first_existing(row, ["inner_user_id", "user_id", "uid", "user", "inner_uid"]), -1)


def row_item_id(row: Dict[str, Any]) -> int:
    return safe_int(get_first_existing(row, ["inner_item_id", "item_id", "iid", "item", "anime_inner_id"]), -1)


def row_raw_item_id(row: Dict[str, Any]) -> str:
    return get_first_existing(row, ["raw_item_id", "anime_id", "raw_anime_id", "mal_anime_id"], "")


def row_score(row: Dict[str, Any]) -> Optional[float]:
    val = get_first_existing(row, ["my_score", "score", "rating", "rate"], "")
    if val == "":
        return None
    return safe_float(val, 0.0)


def parse_time_value(x: Any) -> Tuple[int, str]:
    s = clean_text(x, 80)
    if not s:
        return (0, "")
    # Numeric timestamp.
    if re.fullmatch(r"\d+(?:\.0)?", s):
        return (safe_int(s, 0), s)
    # ISO-like date: use digits as sortable key.
    nums = re.findall(r"\d+", s)
    if nums:
        key = "".join(n.zfill(2) for n in nums[:6])
        return (safe_int(key[:14], 0), s)
    return (0, s)


def row_time(row: Dict[str, Any]) -> Tuple[int, str]:
    val = get_first_existing(row, [
        "interaction_time", "my_last_updated", "timestamp", "time", "ts",
        "my_start_date", "my_finish_date", "date",
    ], "")
    return parse_time_value(val)


def group_by_user(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        u = row_user_id(r)
        i = row_item_id(r)
        if u >= 0 and i >= 0:
            by[u].append(r)
    for u in by:
        by[u].sort(key=lambda r: (row_time(r), row_item_id(r)))
    return dict(by)


def profile_history_item_ids(profile_obj: Dict[str, Any], k: int) -> List[int]:
    ids = profile_obj.get("history_item_ids", []) or profile_obj.get("profile_item_ids", []) or []
    out = []
    for x in ids[-k:]:
        iid = safe_int(x, -1)
        if iid >= 0:
            out.append(iid)
    return out


def get_user_is_minor(profile_obj: Dict[str, Any]) -> bool:
    ui = profile_obj.get("user_info", {}) if profile_obj else {}
    if "is_minor" in ui:
        return bool(ui.get("is_minor"))
    if "age" in ui:
        a = safe_int(ui.get("age"), 99)
        return a < 18
    return False


def merge_item_maps(items_csv: str, item_safe: str) -> Dict[int, Dict[str, Any]]:
    item_map: Dict[int, Dict[str, Any]] = {}
    if items_csv and Path(items_csv).exists():
        for r in read_csv_rows(items_csv):
            iid = row_item_id(r)
            if iid >= 0:
                item_map[iid] = dict(r)
    if item_safe and Path(item_safe).exists():
        for r in read_csv_rows(item_safe):
            iid = row_item_id(r)
            if iid >= 0:
                base = item_map.get(iid, {})
                base.update(r)
                item_map[iid] = base
    print(f"[load] merged item_map items={len(item_map)}")
    return item_map


def infer_risk_flags(item_row: Dict[str, Any]) -> Dict[str, bool]:
    def has_flag(names: List[str]) -> bool:
        for n in names:
            if n in item_row:
                v = item_row.get(n)
                if str(v).strip() == "":
                    continue
                if safe_float(v, 0.0) >= 1.0:
                    return True
                if str(v).strip().lower() in {"true", "yes", "y"}:
                    return True
        return False

    rating_bucket = get_first_existing(item_row, ["rating_bucket", "bucket", "content_bucket"], "").upper()
    rating_text = get_first_existing(item_row, ["rating", "content_rating", "age_rating"], "")
    rt_low = rating_text.lower()
    rating_code = safe_int(get_first_existing(item_row, ["rating_code", "content_rating_code"], ""), -1)

    r17 = has_flag(["r17_code", "R17", "r17", "risk_r17", "item_r17"])
    rplus = has_flag(["rplus_code", "RPLUS", "rplus", "risk_rplus", "item_rplus"])
    rx = has_flag(["rx_code", "RX", "rx", "risk_rx", "item_rx"])

    if "R17" in rating_bucket or "R-17" in rating_bucket:
        r17 = True
    if "RPLUS" in rating_bucket or "R+" in rating_bucket:
        rplus = True
    if "RX" in rating_bucket:
        rx = True

    if "r - 17" in rt_low or "r-17" in rt_low or "17+" in rt_low:
        r17 = True
    if "r+" in rt_low or "mild nudity" in rt_low:
        rplus = True
    if "rx" in rt_low or "hentai" in rt_low:
        rx = True

    if rating_code == 3:
        r17 = True
    elif rating_code == 4:
        rplus = True
    elif rating_code >= 5:
        rx = True

    return {"R17": bool(r17), "RPLUS": bool(rplus), "RX": bool(rx)}


def compute_safety(profile_obj: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[int, List[str], List[str]]:
    is_minor = get_user_is_minor(profile_obj)
    flags = infer_risk_flags(candidate["item_row"])
    if is_minor:
        triggered = [k for k in RISK_NAMES if flags.get(k)]
    else:
        triggered = [k for k in ["RPLUS", "RX"] if flags.get(k)]
    if triggered:
        return 0, triggered, ["mal_risk3_age_policy_violation"]
    return 1, [], []


def item_text(iid: int, row: Optional[Dict[str, Any]], item_map: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    base = dict(item_map.get(iid, {}))
    if row:
        # Keep row values for score/time, but item metadata from items/item_safe wins where available.
        tmp = dict(row)
        tmp.update(base)
        base = tmp
    raw_item = row_raw_item_id(base)
    title = get_first_existing(base, ["title", "title_english", "name", "anime_title"], "")
    title_jp = get_first_existing(base, ["title_japanese"], "")
    if not title and title_jp:
        title = title_jp
    genre = get_first_existing(base, ["genre", "genres"], "")
    typ = get_first_existing(base, ["type", "anime_type"], "")
    source = get_first_existing(base, ["source"], "")
    rating = get_first_existing(base, ["rating", "content_rating", "age_rating"], "")
    bucket = get_first_existing(base, ["rating_bucket", "bucket", "content_bucket"], "")
    background = get_first_existing(base, ["synopsis", "description", "overview"], "")
    # MAL background is often licensing/broadcast metadata. Keep it optional and short if available.
    return {
        "inner_item_id": iid,
        "raw_item_id": str(raw_item),
        "title": clean_text(title, 180),
        "genre": clean_text(genre, 240),
        "type": clean_text(typ, 80),
        "source": clean_text(source, 80),
        "content_rating": clean_text(rating, 160),
        "rating_bucket": clean_text(bucket, 80),
        "background": clean_text(background, 300),
        "item_row": base,
    }


def get_preference_label(score: Optional[float], pos_score: float, neg_score: float) -> Optional[int]:
    if score is None or score <= 0:
        return None
    if score >= pos_score:
        return 1
    if score <= neg_score:
        return 0
    return None


def profile_to_prompt_text(profile_obj: Dict[str, Any]) -> str:
    p = profile_obj.get("profile", {})
    desc = clean_text(p.get("user_description", ""), 240)
    profile_text = clean_text(p.get("profile_text", ""), 700)
    genres = split_list_text(p.get("preferred_genres", []), 6)
    themes = split_list_text(p.get("preferred_themes", []), 6)
    style = split_list_text(p.get("preferred_style", []), 5)
    return "\n".join([
        "User Information:",
        desc if desc else "The user is an anime viewer.",
        "",
        "User Preference Profile:",
        f"- Summary: {profile_text if profile_text else 'Not specified'}",
        f"- Preferred genres: {genres}",
        f"- Preferred themes: {themes}",
        f"- Preferred style: {style}",
    ]).strip()


def history_to_prompt_text(profile_obj: Dict[str, Any], user_rows: List[Dict[str, Any]], item_map: Dict[int, Dict[str, Any]], k: int) -> str:
    ids = profile_history_item_ids(profile_obj, k)
    by_iid = {row_item_id(r): r for r in user_rows}
    lines = ["Recent User History:"]
    for idx, iid in enumerate(ids[-k:], 1):
        it = item_text(iid, by_iid.get(iid), item_map)
        lines.append(
            f"{idx}. Title: {it['title'] or 'Unknown title'}\n"
            f"   Genres: {it['genre'] or 'Unknown genres'}\n"
            f"   Type: {it['type'] or 'Unknown type'}\n"
            f"   Source: {it['source'] or 'Unknown source'}"
        )
    if len(lines) == 1:
        lines.append("No recent history available.")
    return "\n".join(lines)


def candidate_to_prompt_text(cand: Dict[str, Any]) -> str:
    lines = [
        "Candidate Anime:",
        f"Title: {cand['title'] or 'Unknown title'}",
        f"Genres: {cand['genre'] or 'Unknown genres'}",
        f"Type: {cand['type'] or 'Unknown type'}",
        f"Source: {cand['source'] or 'Unknown source'}",
        f"Content rating: {cand['content_rating'] or cand['rating_bucket'] or 'Unknown content rating'}",
    ]
    # Do not use MAL background by default; it is usually licensing/broadcast metadata.
    return "\n".join(lines)


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


def make_template_rules(is_minor: bool, triggered_risks: List[str]) -> str:
    lines = [
        "Retrieved Age-aware Safety Policy:",
        "- Use the policy as a safety filter, not as a rating predictor.",
        "- For anime recommendation, judge candidate age-appropriateness from the visible content rating and policy context.",
    ]
    if is_minor:
        lines.extend([
            "- For minor users, block anime rated for 17+ audiences, mild nudity/adult themes, or explicit adult content.",
            "- Do not block ordinary teen-appropriate action, comedy, fantasy, school life, sports, or romance automatically.",
        ])
    else:
        lines.extend([
            "- For adult users, do not apply minor-only restrictions.",
            "- Reject candidates with mild nudity/adult-oriented or explicit adult content as pseudo-positive samples under this safety setting.",
            "- Anime rated 17+ for violence/profanity alone is not automatically unsafe for adults, unless the retrieved policy indicates severe harmful content.",
        ])
    if triggered_risks:
        pretty = ", ".join(triggered_risks)
        lines.append(f"- Pay special attention to the candidate's triggered age-rating topic(s): {pretty}.")
    lines.append("- Final principle: output 1 only when both preference compatibility and age-aware safety are satisfied; otherwise output 0.")
    return "\n".join(lines)


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    # Try direct JSONL / JSON object.
    for line in reversed(s.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                pass
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except Exception:
            return None
    return None


def parse_rules_from_cli(stdout: str) -> Optional[str]:
    obj = parse_json_from_text(stdout)
    if obj:
        for key in ["rule_text", "rules_text", "english_rules", "policy_text", "rules", "text"]:
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                txt = val.strip()
                if not txt.startswith("Retrieved Age-aware Safety Policy:"):
                    txt = "Retrieved Age-aware Safety Policy:\n" + txt
                return txt
    markers = ["Loaded English rules from cache:", "Generated English regulatory rules:", "Retrieved Age-aware Safety Policy:", "Retrieved Regulatory Rules:"]
    cand = None
    for m in markers:
        if m in stdout:
            cand = stdout.split(m, 1)[1]
            break
    if cand is None:
        lines = [ln.strip() for ln in stdout.splitlines() if ln.strip().startswith("- ")]
        if len(lines) >= 2:
            cand = "\n".join(lines)
    if cand is None:
        return None
    for sm in ["Saved cache:", "Loading Qwen:", "Retrieved Chinese chunks", "Traceback", "Dense vector RAG:"]:
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
    joined = "\n".join(lines[:10]).strip()
    if not joined.startswith("Retrieved Age-aware Safety Policy:"):
        joined = "Retrieved Age-aware Safety Policy:\n" + joined
    return joined


def signature_key(is_minor: bool, triggered: List[str]) -> Tuple[str, Tuple[str, ...]]:
    return ("minor" if is_minor else "adult", tuple(sorted(triggered)))


def get_rules_text(uid: int, cand: Dict[str, Any], is_minor: bool, triggered: List[str], args: argparse.Namespace, cache: Dict[Any, Tuple[str, str]], rag_state: Counter) -> Tuple[str, str]:
    if args.rule_mode == "template":
        if args.require_rag_success:
            raise RuntimeError("--require_rag_success requires --rule_mode rag_cli")
        return make_template_rules(is_minor, triggered), "template"

    key = signature_key(is_minor, triggered)
    if key in cache:
        return cache[key]

    cmd = [
        sys.executable, args.rag_script, "gen-rule",
        "--profiles", args.profiles,
        "--item_safe", args.item_safe,
        "--user_id", str(uid),
        "--item_id", str(cand["inner_item_id"]),
        "--vector_dir", args.vector_dir,
        "--model_path", args.model_path,
        "--cache_path", args.rag_cache_path,
        "--topn", str(args.rag_topn),
        "--topk", str(args.rag_topk),
        "--device", args.rag_device,
        "--dtype", args.rag_dtype,
        "--json_only",
    ]
    # Keep compatibility with RAG scripts that support explicit cache strategy.
    if args.rag_cache_strategy:
        cmd.extend(["--cache_strategy", args.rag_cache_strategy])

    rag_state["calls"] += 1
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", timeout=args.rag_timeout, check=False)
        if p.returncode == 0:
            parsed = parse_rules_from_cli(p.stdout)
            if parsed:
                cache[key] = (parsed, "rag_cli")
                return cache[key]
            last_error = "RAG returned 0 but parsing failed. Last stdout:\n" + (p.stdout[-2000:] if p.stdout else "")
        else:
            last_error = f"RAG failed returncode={p.returncode}. Last stdout:\n" + (p.stdout[-2000:] if p.stdout else "")
    except Exception as e:
        last_error = repr(e)

    if args.require_rag_success:
        raise RuntimeError(f"RAG failed: uid={uid}, iid={cand['inner_item_id']}, triggered={triggered}. {last_error}")
    rules = make_template_rules(is_minor, triggered)
    cache[key] = (rules, "template_after_rag_fail")
    return cache[key]


def make_input_text(profile_obj: Dict[str, Any], user_rows: List[Dict[str, Any]], item_map: Dict[int, Dict[str, Any]], cand: Dict[str, Any], rules_text: str, history_window: int) -> str:
    return "\n\n".join([
        profile_to_prompt_text(profile_obj),
        history_to_prompt_text(profile_obj, user_rows, item_map, history_window),
        rules_text,
        candidate_to_prompt_text(cand),
        "Question: Is this candidate BOTH preference-compatible with the user and safe under the age-aware policy, so that it can be used as a pseudo-positive sample? Answer only 1 or 0.",
    ]).strip()


def leakage_check(text: str) -> List[str]:
    low = text.lower()
    return [p for p in FORBIDDEN_VISIBLE_PATTERNS if p.lower() in low]


def quadrant_name(pref: int, safety: int) -> str:
    if pref == 1 and safety == 1:
        return "pref_pos_safe"
    if pref == 1 and safety == 0:
        return "pref_pos_unsafe"
    if pref == 0 and safety == 1:
        return "pref_neg_safe"
    return "pref_neg_unsafe"


def build_record(uid: int, row: Dict[str, Any], profile_obj: Dict[str, Any], user_rows: List[Dict[str, Any]], item_map: Dict[int, Dict[str, Any]], args: argparse.Namespace, preselect_only: bool = True) -> Optional[Dict[str, Any]]:
    iid = row_item_id(row)
    if iid < 0:
        return None
    score = row_score(row)
    pref = get_preference_label(score, args.pos_score, args.neg_score)
    if pref is None:
        return None
    cand = item_text(iid, row, item_map)
    safety, triggered, unsafe_reasons = compute_safety(profile_obj, cand)
    label = 1 if pref == 1 and safety == 1 else 0
    is_minor = get_user_is_minor(profile_obj)
    rules_text = make_template_rules(is_minor, triggered)
    input_text = make_input_text(profile_obj, user_rows, item_map, cand, rules_text, args.history_window)
    found = leakage_check(input_text)
    if found and args.strict_no_leakage:
        raise RuntimeError(f"Visible input leakage uid={uid}, iid={iid}: {found}\n{input_text[:1200]}")
    q = quadrant_name(pref, safety)
    return {
        "instruction": make_instruction(),
        "input": input_text,
        "output": str(int(label)),
        "metadata": {
            "user_id": int(uid),
            "raw_user_id": str(profile_obj.get("user_info", {}).get("raw_user_id", "")),
            "is_minor": bool(is_minor),
            "candidate_inner_item_id": int(iid),
            "candidate_raw_item_id": str(cand.get("raw_item_id", "")),
            "candidate_title": cand.get("title", ""),
            "source": "mal_complete_interactions_before_valid_excluding_profile_last5",
            "label": int(label),
            "quadrant_hidden": q,
            "score_hidden": float(score) if score is not None else None,
            "preference_label_hidden": int(pref),
            "safety_label_hidden": int(safety),
            "triggered_risks_hidden": triggered,
            "unsafe_reasons_hidden": unsafe_reasons,
            "rule_source": "preselect_template" if preselect_only else "template",
            "visible_leakage_warnings": found,
        },
    }


def sample_list(rng: random.Random, arr: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if n < 0 or n >= len(arr):
        return list(arr)
    return rng.sample(arr, n)


def write_jsonl(path: str, rows: List[Dict[str, Any]], no_metadata: bool = False) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            obj = dict(r)
            if no_metadata:
                obj.pop("metadata", None)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mal2000_risk3_seq50_300")
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--interactions_csv", required=True)
    ap.add_argument("--items_csv", required=True)
    ap.add_argument("--item_safe", required=True)
    ap.add_argument("--train_rating", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--output_nometa", default="")
    ap.add_argument("--summary_json", default="")
    ap.add_argument("--history_window", type=int, default=5)
    ap.add_argument("--pos_score", type=float, default=7.0)
    ap.add_argument("--neg_score", type=float, default=6.0)
    ap.add_argument("--candidate_order", choices=["recent", "oldest", "random"], default="recent")
    ap.add_argument("--max_users", type=int, default=-1)
    ap.add_argument("--per_user_quad_cap", type=int, default=10)
    ap.add_argument("--balance_4types", action="store_true")
    ap.add_argument("--samples_per_quadrant", type=int, default=-1, help="If >0, sample at most this many from each quadrant.")
    ap.add_argument("--use_profile_time_cutoff", action="store_true", default=True)
    ap.add_argument("--no_profile_time_cutoff", dest="use_profile_time_cutoff", action="store_false")
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--rule_mode", choices=["template", "rag_cli"], default="template")
    ap.add_argument("--rag_script", default="scripts/mal2000_risk3/03_psg_regulation_rag_full.py")
    ap.add_argument("--vector_dir", default="outputs/mal2000_risk3_seq50_300/rag_vector")
    ap.add_argument("--model_path", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--rag_cache_path", default="outputs/mal2000_risk3_seq50_300/rag_vector/rule_text_cache_mal_risk3_pairaware.jsonl")
    ap.add_argument("--rag_cache_strategy", default="", help="Optional; pass through only if the RAG script supports --cache_strategy.")
    ap.add_argument("--rag_topk", type=int, default=5)
    ap.add_argument("--rag_topn", type=int, default=80)
    ap.add_argument("--rag_timeout", type=int, default=300)
    ap.add_argument("--rag_device", default="cuda")
    ap.add_argument("--rag_dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")
    ap.add_argument("--rag_after_sampling", action="store_true")
    ap.add_argument("--require_rag_success", action="store_true")
    ap.add_argument("--strict_no_leakage", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()

    profiles = read_jsonl_profiles(args.profiles)
    interactions = read_csv_rows(args.interactions_csv)
    by_user = group_by_user(interactions)
    train_pairs = read_rating_pairs(args.train_rating)
    item_map = merge_item_maps(args.items_csv, args.item_safe)

    users = sorted(set(profiles) & set(by_user))
    if args.max_users > 0:
        users = users[:args.max_users]

    print(f"[info] users={len(users)}")
    print(f"[info] preference positive: score >= {args.pos_score}; negative: 1 <= score <= {args.neg_score}")
    print("[info] safety: minor unsafe=R17/RPLUS/RX; adult unsafe=RPLUS/RX")
    print(f"[info] exclude profile last-{args.history_window} items from candidate samples")
    print(f"[info] profile_time_cutoff={args.use_profile_time_cutoff}")
    print(f"[info] rule_mode={args.rule_mode}, rag_after_sampling={args.rag_after_sampling}")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    stats = Counter()

    for idx, uid in enumerate(users, 1):
        prof = profiles[uid]
        rows = by_user[uid]
        hist_ids = set(profile_history_item_ids(prof, args.history_window))

        cutoff = (0, "")
        if args.use_profile_time_cutoff and hist_ids:
            hist_times = [row_time(r) for r in rows if row_item_id(r) in hist_ids]
            hist_times = [t for t in hist_times if t[0] > 0]
            if hist_times:
                cutoff = max(hist_times)

        candidates = []
        for r in rows:
            iid = row_item_id(r)
            if iid < 0:
                continue
            if iid in hist_ids:
                stats["excluded_profile_history_item"] += 1
                continue
            score = row_score(r)
            pref = get_preference_label(score, args.pos_score, args.neg_score)
            if pref is None:
                stats["dropped_neutral_or_missing_score"] += 1
                continue
            if pref == 1 and (uid, iid) not in train_pairs:
                stats["dropped_positive_not_in_train_rating"] += 1
                continue
            if cutoff[0] > 0 and row_time(r)[0] > cutoff[0]:
                stats["dropped_after_profile_time_cutoff"] += 1
                continue
            candidates.append(r)

        if args.candidate_order == "recent":
            candidates.sort(key=lambda r: (row_time(r), row_item_id(r)), reverse=True)
        elif args.candidate_order == "oldest":
            candidates.sort(key=lambda r: (row_time(r), row_item_id(r)))
        else:
            rng.shuffle(candidates)

        per_user_counts = Counter()
        for r in candidates:
            rec = build_record(uid, r, prof, rows, item_map, args, preselect_only=True)
            if rec is None:
                stats["skipped_bad_record"] += 1
                continue
            q = rec["metadata"]["quadrant_hidden"]
            if args.per_user_quad_cap >= 0 and per_user_counts[q] >= args.per_user_quad_cap:
                stats[f"cap_skip_{q}"] += 1
                continue
            grouped[q].append(rec)
            per_user_counts[q] += 1
            stats[f"available_{q}"] += 1
        stats["users_seen"] += 1
        if get_user_is_minor(prof):
            stats["minor_users_seen"] += 1
        else:
            stats["adult_users_seen"] += 1

        if idx % 200 == 0 or idx == len(users):
            total_avail = sum(len(v) for v in grouped.values())
            print(f"[progress] {idx}/{len(users)} users, available={total_avail}, elapsed={(time.time()-t0)/60:.1f}min", flush=True)

    quads = ["pref_pos_safe", "pref_pos_unsafe", "pref_neg_safe", "pref_neg_unsafe"]
    print("[availability]")
    for q in quads:
        print(f"  {q}: {len(grouped[q])}")

    if args.balance_4types:
        n = min(len(grouped[q]) for q in quads)
        if args.samples_per_quadrant > 0:
            n = min(n, args.samples_per_quadrant)
        samples: List[Dict[str, Any]] = []
        for q in quads:
            samples.extend(sample_list(rng, grouped[q], n))
        selected_stats = {f"selected_{q}": n for q in quads}
        selected_stats["selected_per_quadrant"] = n
    else:
        samples = []
        for q in quads:
            n = args.samples_per_quadrant
            samples.extend(sample_list(rng, grouped[q], n))
        selected_stats = {f"selected_{q}": sum(1 for s in samples if s["metadata"]["quadrant_hidden"] == q) for q in quads}

    rng.shuffle(samples)

    rag_state = Counter()
    rule_cache: Dict[Any, Tuple[str, str]] = {}
    if args.rule_mode == "rag_cli" and args.rag_after_sampling:
        print(f"[info] attaching RAG rules after sampling; selected={len(samples)}", flush=True)
        for idx, s in enumerate(samples, 1):
            md = s["metadata"]
            uid = int(md["user_id"])
            iid = int(md["candidate_inner_item_id"])
            prof = profiles[uid]
            rows = by_user[uid]
            cand = item_text(iid, None, item_map)
            triggered = md.get("triggered_risks_hidden", []) or []
            rules_text, rule_source = get_rules_text(uid, cand, get_user_is_minor(prof), triggered, args, rule_cache, rag_state)
            s["input"] = make_input_text(prof, rows, item_map, cand, rules_text, args.history_window)
            s["metadata"]["rule_source"] = rule_source
            found = leakage_check(s["input"])
            s["metadata"]["visible_leakage_warnings"] = found
            if found and args.strict_no_leakage:
                raise RuntimeError(f"Visible input leakage after RAG attach uid={uid}, iid={iid}: {found}\n{s['input'][:1200]}")
            if idx % 1000 == 0 or idx == len(samples):
                print(f"[rag_attach] {idx}/{len(samples)}, rag_calls={rag_state['calls']}, rule_cache_entries={len(rule_cache)}", flush=True)

    if args.rule_mode == "rag_cli" and args.require_rag_success:
        bad = Counter(s["metadata"].get("rule_source") for s in samples if s["metadata"].get("rule_source") != "rag_cli")
        if bad:
            raise RuntimeError(f"Strict RAG-only check failed. Non-RAG sources: {dict(bad)}")

    write_jsonl(args.output, samples, no_metadata=False)
    print(f"[save] output={args.output} samples={len(samples)}")
    if args.output_nometa:
        write_jsonl(args.output_nometa, samples, no_metadata=True)
        print(f"[save] output_nometa={args.output_nometa}")

    label_counts = Counter(s["output"] for s in samples)
    quad_counts = Counter(s["metadata"]["quadrant_hidden"] for s in samples)
    scope_counts = Counter("minor" if s["metadata"].get("is_minor") else "adult" for s in samples)
    rule_counts = Counter(s["metadata"].get("rule_source") for s in samples)
    selected_stats.update({
        "total_samples": len(samples),
        "label_counts": dict(label_counts),
        "quadrant_counts": dict(quad_counts),
        "scope_counts": dict(scope_counts),
        "rule_source_counts": dict(rule_counts),
        "rag_cli_calls": int(rag_state["calls"]),
        "rule_cache_entries": len(rule_cache),
    })

    summary = {
        "version": "MAL-Risk3-SFT-last5-aligned-v1",
        "dataset": args.dataset,
        "config": vars(args),
        "label_definition": {
            "preference_positive": f"my_score >= {args.pos_score}",
            "preference_negative": f"1 <= my_score <= {args.neg_score}",
            "safety_minor": "unsafe if R17 or RPLUS or RX",
            "safety_adult": "unsafe if RPLUS or RX; R17 is not unsafe for adults",
            "output": "1 iff preference_label=1 and safety_label=1; otherwise 0",
            "candidate_exclusion": f"exclude profile last-{args.history_window} items",
            "positive_leakage_control": "positive samples must be in train.rating",
        },
        "availability_stats": dict(stats),
        "available_quadrants": {q: len(grouped[q]) for q in quads},
        "selected_stats": selected_stats,
    }
    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[save] summary_json={args.summary_json}")

    print("\nDone.")
    print(json.dumps(selected_stats, ensure_ascii=False, indent=2))
    if samples:
        preview = {k: samples[0][k] for k in ["instruction", "input", "output"]}
        print("\nPreview sample:")
        print(json.dumps(preview, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
