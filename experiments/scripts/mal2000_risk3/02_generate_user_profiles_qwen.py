#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE:
  scripts/mal2000_risk3/02_generate_user_profiles_qwen.py
VERSION: v4 robust profile sanitization

PURPOSE:
  Generate MAL user profiles from latest K train interactions before validation.
  It uses only training-history items and does not use valid/test interactions.

RUN COMMANDS:

1) Debug 5 users:
cd .

DATASET=mal2000_risk3_seq50_300
MODEL_PATH=models/Qwen2.5-7B-Instruct

CUDA_VISIBLE_DEVICES=0 python -u scripts/mal2000_risk3/02_generate_user_profiles_qwen.py \
  --model_path "${MODEL_PATH}" \
  --history_csv outputs/${DATASET}/profile_inputs/${DATASET}.train_profile_history.csv \
  --output outputs/${DATASET}/profiles/${DATASET}.user_profiles.train_last5.debug.jsonl \
  --k 5 \
  --max_users 5 \
  --device_map auto \
  --dtype bf16 \
  --max_new_tokens 512

2) Full generation:
cd .

DATASET=mal2000_risk3_seq50_300
MODEL_PATH=models/Qwen2.5-7B-Instruct

rm -f outputs/${DATASET}/profiles/${DATASET}.user_profiles.train_last5.jsonl

CUDA_VISIBLE_DEVICES=0 python -u scripts/mal2000_risk3/02_generate_user_profiles_qwen.py \
  --model_path "${MODEL_PATH}" \
  --history_csv outputs/${DATASET}/profile_inputs/${DATASET}.train_profile_history.csv \
  --output outputs/${DATASET}/profiles/${DATASET}.user_profiles.train_last5.jsonl \
  --k 5 \
  --resume \
  --device_map auto \
  --dtype bf16 \
  --max_new_tokens 512 \
  2>&1 | tee outputs/${DATASET}/profiles/generate_user_profiles_train_last5.log

Input:
  outputs/<dataset>/profile_inputs/<dataset>.train_profile_history.csv

Output:
  outputs/<dataset>/profiles/<dataset>.user_profiles.train_last5.jsonl

No valid/test leakage:
  The profile history CSV only contains train.rating positives.
  Since valid/test are held out as the latest two liked interactions, the
  latest K train positives are the recent history before validation.

Policy:
  This script generates preference profiles only. It sanitizes adult/risk words
  such as Hentai, Ecchi, Harem, nudity, RX/RPLUS/R17, and mature content.
  Item safety is handled later by RAG rules and item rating labels, not by the
  user profile text.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd


def clean_text(x: Any, max_chars: int = 600) -> str:
    if x is None:
        return ""
    s = str(x).replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0] + "..."
    return s


def str_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def read_done_users(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                done.add(int(obj["user_id"]))
            except Exception:
                pass
    return done


def user_description(user_info: Dict[str, Any]) -> str:
    age = user_info.get("age", "")
    is_minor = bool(user_info.get("is_minor", False))

    if is_minor:
        return f"The user is a minor anime viewer aged {age}."
    return f"The user is an adult anime viewer aged {age}."


def item_line(row: Dict[str, Any], idx: int) -> str:
    title = clean_text(row.get("title", ""), 160)
    genre = clean_text(row.get("genre", ""), 220)
    rating = clean_text(row.get("rating", ""), 120)
    bucket = clean_text(row.get("rating_bucket", ""), 60)
    typ = clean_text(row.get("type", ""), 80)
    source = clean_text(row.get("source", ""), 80)

    return (
        f"History item {idx}:\n"
        f"Title: {title}\n"
        f"Genres: {genre}\n"
        f"Content rating: {rating} ({bucket})\n"
        f"Type: {typ}\n"
        f"Source: {source}"
    )


def make_prompt(user_info: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    udesc = user_description(user_info)
    history = "\n\n".join(item_line(r, i + 1) for i, r in enumerate(rows))

    system = (
        "You are a recommender-system user profile generator for anime recommendation. "
        "Generate a concise preference profile from training interactions only. "
        "Return valid JSON only."
    )

    user = f"""
User information:
{udesc}

Recent training history before validation:
{history}

Task:
Infer this user's anime preference profile from the recent training history.

Important constraints:
1. Use age group only to write "user_description"; do not infer taste from age.
2. Infer preferences only from the history items.
3. Do not mention any historical anime title in the output.
4. Do not mention item IDs, timestamps, risk labels, safety labels, or scores.
5. Use "The user" instead of gendered pronouns.
6. Keep the output compact.
7. Do not output adult-content or risk-category words such as Hentai, Ecchi, Harem, nudity, sexualized, pornographic, explicit, RX, RPLUS, R17, or mature content.
8. If the history contains adult-rated items, summarize only non-sensitive preferences such as genre, narrative style, pacing, tone, mood, worldbuilding, and story structure.

Return exactly one JSON object:
{{
  "user_description": "{udesc}",
  "profile_text": "2-3 concise sentences about the user's anime preferences, without anime titles.",
  "preferred_genres": ["3 to 5 genre names"],
  "preferred_themes": ["3 to 5 theme names"],
  "preferred_style": ["2 to 4 style descriptors"]
}}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_json_object(text: str) -> Dict[str, Any]:
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(s[start:end + 1])


def ensure_list(x: Any, max_len: int, fallback: List[str]) -> List[str]:
    if isinstance(x, list):
        arr = [clean_text(v, 80) for v in x if clean_text(v, 80)]
    elif isinstance(x, str) and x.strip():
        arr = [clean_text(v, 80) for v in re.split(r"[,;/|]", x) if clean_text(v, 80)]
    else:
        arr = []

    seen = set()
    out = []
    for v in arr:
        k = v.lower()
        if k not in seen:
            out.append(v)
            seen.add(k)

    if not out:
        out = fallback[:]
    return out[:max_len]


def fallback_profile(user_info: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cnt = defaultdict(int)
    for r in rows:
        for g in re.split(r"[,;/|]", str(r.get("genre", ""))):
            g = clean_text(g, 50)
            if g:
                cnt[g] += 1

    genres = [g for g, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
    if not genres:
        genres = ["Drama", "Comedy", "Action"]

    return {
        "user_description": user_description(user_info),
        "profile_text": "The user appears to prefer anime matching the dominant genres and narrative patterns in the recent training history. This profile is a compact preference estimate based only on pre-validation interactions.",
        "preferred_genres": genres[:5],
        "preferred_themes": ["Character-driven stories", "Emotional development", "Engaging conflicts"],
        "preferred_style": ["Narrative-focused", "Accessible", "Emotionally engaging"],
    }


def contains_title_leak(profile: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    text = json.dumps(profile, ensure_ascii=False).lower()
    for r in rows:
        title = clean_text(r.get("title", ""), 160)
        if len(title) >= 4 and title.lower() in text:
            return True
    return False


SENSITIVE_TERMS = [
    "hentai", "ecchi", "harem", "nudity", "nude", "sexualized", "sexual", "pornographic",
    "porn", "explicit", "rx", "rplus", "r17", "r-plus", "r 17", "mature content",
    "adult content", "adult-oriented", "adult oriented", "erotic", "fetish"
]

SENSITIVE_REPLACEMENTS = {
    "harem": "relationship-driven stories",
    "ecchi": "comedy and character interactions",
    "hentai": "non-sensitive genre/style",
    "mature content": "complex or intense narrative tone",
    "adult content": "complex or intense narrative tone",
}


def has_sensitive_term(text: Any) -> bool:
    s = str(text or "").lower()
    return any(t in s for t in SENSITIVE_TERMS)


def sanitize_text(text: Any) -> str:
    s = clean_text(text, 700)
    for k, v in SENSITIVE_REPLACEMENTS.items():
        s = re.sub(re.escape(k), v, s, flags=re.IGNORECASE)
    for t in SENSITIVE_TERMS:
        s = re.sub(re.escape(t), "non-sensitive preference", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sanitize_list(values: List[str], fallback: List[str], max_len: int) -> List[str]:
    out = []
    seen = set()
    for v in values:
        if has_sensitive_term(v):
            continue
        vv = sanitize_text(v)
        if not vv or has_sensitive_term(vv):
            continue
        k = vv.lower()
        if k not in seen:
            out.append(vv)
            seen.add(k)
    if not out:
        out = [x for x in fallback if not has_sensitive_term(x)] or ["Character-driven stories", "Narrative-focused", "Genre variety"]
    return out[:max_len]


def get_raw_item_id_from_row(row: Dict[str, Any]) -> int:
    for key in ["raw_item_id", "raw_item_id_y", "raw_item_id_x", "anime_id", "raw_anime_id"]:
        if key in row:
            try:
                return int(float(str(row.get(key))))
            except Exception:
                pass
    return -1


def normalize_profile(raw: Dict[str, Any], user_info: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fb = fallback_profile(user_info, rows)

    raw_profile_text = raw.get("profile_text", "") or fb["profile_text"]
    raw_genres = ensure_list(raw.get("preferred_genres", []), 5, fb["preferred_genres"])
    raw_themes = ensure_list(raw.get("preferred_themes", []), 5, fb["preferred_themes"])
    raw_style = ensure_list(raw.get("preferred_style", []), 4, fb["preferred_style"])

    profile = {
        "user_description": user_description(user_info),
        "profile_text": sanitize_text(raw_profile_text),
        "preferred_genres": sanitize_list(raw_genres, ["Comedy", "Romance", "Action", "Drama", "School life", "Character interactions"], 5),
        "preferred_themes": sanitize_list(raw_themes, ["Character-driven stories", "School life", "Adventure", "Relationships", "Worldbuilding"], 5),
        "preferred_style": sanitize_list(raw_style, ["Narrative-focused", "Humorous", "Engaging", "Dynamic"], 4),
    }

    profile["profile_text"] = re.sub(r"\b[Ss]he\b|\b[Hh]e\b", "The user", profile["profile_text"])
    profile["profile_text"] = re.sub(r"\b[Hh]is\b|\b[Hh]er\b", "the user's", profile["profile_text"])
    profile["profile_text"] = re.sub(r"\s+", " ", profile["profile_text"]).strip()

    # Final hard filter: if any sensitive term still appears, remove or replace it.
    if has_sensitive_term(profile["profile_text"]):
        profile["profile_text"] = sanitize_text(profile["profile_text"])
    for key, fallback, max_len in [
        ("preferred_genres", ["Comedy", "Romance", "Action", "Drama", "School life"], 5),
        ("preferred_themes", ["Character interactions", "School life", "Adventure", "Relationships", "Worldbuilding"], 5),
        ("preferred_style", ["Narrative-focused", "Humorous", "Engaging", "Dynamic"], 4),
    ]:
        profile[key] = sanitize_list(profile.get(key, []), fallback, max_len)

    if contains_title_leak(profile, rows):
        safe_fb = {
            "user_description": fb["user_description"],
            "profile_text": sanitize_text(fb["profile_text"]),
            "preferred_genres": sanitize_list(fb["preferred_genres"], ["Drama", "Comedy", "Action"], 5),
            "preferred_themes": sanitize_list(fb["preferred_themes"], ["Character-driven stories", "Emotional development", "Engaging conflicts"], 5),
            "preferred_style": sanitize_list(fb["preferred_style"], ["Narrative-focused", "Accessible", "Emotionally engaging"], 4),
        }
        return safe_fb

    return profile


def load_model(model_path: str, device_map: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
    }

    if dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif dtype == "fp32":
        kwargs["torch_dtype"] = torch.float32
    else:
        kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return tokenizer, model


def generate_one(tokenizer, model, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    import torch

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--history_csv", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_users", type=int, default=-1)
    ap.add_argument("--start_user", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    df = pd.read_csv(args.history_csv)
    df["inner_user_id"] = df["inner_user_id"].astype(int)
    df["interaction_time"] = pd.to_datetime(df["interaction_time"], errors="coerce")
    df["_time_sort"] = df["interaction_time"].fillna(pd.Timestamp("1900-01-01"))
    df = df.sort_values(["inner_user_id", "_time_sort", "history_rank_in_train", "inner_item_id"])

    users = sorted(df["inner_user_id"].unique().tolist())
    users = [u for u in users if u >= args.start_user]
    if args.max_users > 0:
        users = users[:args.max_users]

    done = read_done_users(args.output) if args.resume else set()
    todo = [u for u in users if u not in done]

    print("history_csv =", args.history_csv)
    print("users total =", df["inner_user_id"].nunique())
    print("users selected =", len(users))
    print("done =", len(done))
    print("todo =", len(todo))
    print("k =", args.k)

    if not todo:
        print("Nothing to do.")
        return

    print("loading model =", args.model_path)
    tokenizer, model = load_model(args.model_path, args.device_map, args.dtype)

    mode = "a" if args.resume else "w"
    n_ok = 0
    n_parse_error = 0
    n_title_fallback = 0

    with open(args.output, mode, encoding="utf-8") as out:
        for idx, uid in enumerate(todo, start=1):
            g = df[df["inner_user_id"] == uid].copy()
            rows = g.tail(args.k).to_dict("records")
            if not rows:
                continue

            first = rows[0]
            user_info = {
                "inner_user_id": int(uid),
                "raw_user_id": str(first.get("raw_user_id", "")),
                "username": str(first.get("username", "")),
                "age": int(first.get("age", -1)),
                "is_minor": bool(int(first.get("is_minor", 0))),
            }

            messages = make_prompt(user_info, rows)
            parse_error = False
            title_fallback = False
            raw_gen = ""

            try:
                raw_gen = generate_one(tokenizer, model, messages, args.max_new_tokens)
                raw_profile = extract_json_object(raw_gen)
                title_fallback = contains_title_leak(raw_profile, rows)
                profile = normalize_profile(raw_profile, user_info, rows)
            except Exception as e:
                parse_error = True
                n_parse_error += 1
                raw_gen = raw_gen or f"[error] {repr(e)}"
                profile = fallback_profile(user_info, rows)

            if title_fallback:
                n_title_fallback += 1

            obj = {
                "profile_id": f"u{uid}_train_last{args.k}",
                "user_id": int(uid),
                "profile_source": f"train_last_{args.k}_before_valid",
                "user_info": user_info,
                "history_item_ids": [int(r["inner_item_id"]) for r in rows],
                "history_raw_item_ids": [get_raw_item_id_from_row(r) for r in rows],
                "history_rating_buckets": [str(r.get("rating_bucket", "")) for r in rows],
                "profile": profile,
                "parse_error": bool(parse_error),
                "title_leak_fallback": bool(title_fallback),
            }

            if parse_error:
                obj["raw_generation"] = raw_gen[:2000]

            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out.flush()
            n_ok += 1

            if idx == 1 or idx % 50 == 0:
                print(f"[{idx}/{len(todo)}] user={uid}, parse_error={parse_error}, title_fallback={title_fallback}")

    print("Done.")
    print("output =", args.output)
    print("generated =", n_ok)
    print("parse_errors =", n_parse_error)
    print("title_fallback =", n_title_fallback)


if __name__ == "__main__":
    main()
