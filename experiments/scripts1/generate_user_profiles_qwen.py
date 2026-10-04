#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generate user profiles from first/last K training interactions
# 生成用户画像最近5
"""
scripts/generate_user_profiles_qwen.py

Generate one natural-language user preference profile for each MovieLens user
from either the first K or the most recent K training interactions.

Main design:
- Input only train.safe.csv, not valid/test.
- Use only title / genres / overview from the selected K train items to infer preference.
- Do NOT put IMDb risk scores, tolerance values, or safe labels into the LLM prompt.
- Convert gender from F/M to female/male in natural language.
- Convert MovieLens age code to natural-language age group.
- Keep raw IDs and history IDs as metadata for debugging, but do not ask the LLM
  to output or mention historical movie names in the profile.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


AGE_MAP = {
    "1": "under 18",
    "18": "18 to 24",
    "25": "25 to 34",
    "35": "35 to 44",
    "45": "45 to 49",
    "50": "50 to 55",
    "56": "56 or older",
}


def gender_to_word(gender: Any) -> str:
    g = str(gender).strip().upper()
    if g == "F":
        return "female"
    if g == "M":
        return "male"
    if g in {"FEMALE", "WOMAN"}:
        return "female"
    if g in {"MALE", "MAN"}:
        return "male"
    return "unknown-gender"


def age_to_desc(age: Any) -> str:
    a = str(age).strip()
    return AGE_MAP.get(a, "unknown age")


def age_to_minor(age: Any, age_desc: Optional[str] = None) -> bool:
    a = str(age).strip()
    if a == "1":
        return True
    if age_desc and "under 18" in str(age_desc).lower():
        return True
    return False


def str_to_bool(x: Any) -> bool:
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def clean_text(x: Any, max_chars: int = 700) -> str:
    if x is None:
        return ""
    s = str(x).replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0] + "..."
    return s


def get_first_existing(row: Dict[str, Any], names: Iterable[str], default: str = "") -> str:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return str(row[name])
    return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return default


def load_train_rows(train_safe: str) -> Dict[int, List[Dict[str, Any]]]:
    by_user: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    with open(train_safe, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or no header: {train_safe}")

        for row in reader:
            uid_raw = get_first_existing(row, ["inner_user_id", "user_id", "uid"], default="")
            if uid_raw == "":
                raise ValueError(
                    "Cannot find user id column. Expected one of: inner_user_id, user_id, uid"
                )
            uid = safe_int(uid_raw, default=-1)
            if uid < 0:
                continue
            by_user[uid].append(row)

    return by_user


def read_existing_users(output_path: str) -> set:
    done = set()
    if not os.path.exists(output_path):
        return done

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "user_id" in obj:
                    done.add(int(obj["user_id"]))
            except Exception:
                continue
    return done


def build_user_info(first_row: Dict[str, Any], inner_user_id: int) -> Dict[str, Any]:
    raw_user_id = get_first_existing(first_row, ["raw_user_id", "user_raw_id"], default=str(inner_user_id))
    gender = get_first_existing(first_row, ["gender", "user_gender"], default="")
    age = get_first_existing(first_row, ["age", "user_age"], default="")

    gender_desc = gender_to_word(gender)

    existing_age_desc = get_first_existing(first_row, ["age_desc", "user_age_desc"], default="")
    age_desc = existing_age_desc if existing_age_desc else age_to_desc(age)

    if "is_minor" in first_row and str(first_row["is_minor"]).strip() != "":
        is_minor = str_to_bool(first_row["is_minor"])
    else:
        is_minor = age_to_minor(age, age_desc)

    return {
        "inner_user_id": int(inner_user_id),
        "raw_user_id": str(raw_user_id),
        "gender": str(gender),
        "gender_desc": gender_desc,
        "age": str(age),
        "age_desc": age_desc,
        "is_minor": bool(is_minor),
    }


def normalize_age_desc(age_desc: Any) -> str:
    """Normalize MovieLens age descriptions into natural English."""
    s = str(age_desc).strip()
    replacements = {
        "25-34": "25 to 34",
        "35-44": "35 to 44",
        "45-49": "45 to 49",
        "50-55": "50 to 55",
        "56+": "56 or older",
    }
    return replacements.get(s, s or "unknown age")


def user_description(user_info: Dict[str, Any]) -> str:
    gender_desc = user_info.get("gender_desc", "unknown-gender")
    age_desc = normalize_age_desc(user_info.get("age_desc", "unknown age"))

    if gender_desc == "unknown-gender" and age_desc == "unknown age":
        return "The user is a viewer with unknown demographic information."

    if gender_desc == "unknown-gender":
        if age_desc == "under 18":
            return "The user is a viewer under 18."
        return f"The user is a viewer aged {age_desc}."

    if age_desc == "unknown age":
        return f"The user is a {gender_desc} viewer."

    if age_desc == "under 18":
        return f"The user is a {gender_desc} viewer under 18."

    return f"The user is a {gender_desc} viewer aged {age_desc}."


def extract_item(row: Dict[str, Any]) -> Dict[str, Any]:
    inner_item_id = get_first_existing(row, ["inner_item_id", "item_id", "iid"], default="")
    raw_movie_id = get_first_existing(row, ["raw_movie_id", "movie_id", "raw_item_id"], default="")
    timestamp = get_first_existing(row, ["timestamp", "time", "ts"], default="0")

    title = get_first_existing(row, ["title", "movie_title", "name"], default="")
    genres = get_first_existing(row, ["genres", "genre"], default="")
    overview = get_first_existing(row, ["overview", "plot", "description", "movie_overview"], default="")

    return {
        "inner_item_id": safe_int(inner_item_id, default=-1),
        "raw_movie_id": str(raw_movie_id),
        "timestamp": safe_int(timestamp, default=0),
        "title": clean_text(title, max_chars=180),
        "genres": clean_text(genres, max_chars=200),
        "overview": clean_text(overview, max_chars=700),
    }


def make_prompt(user_info: Dict[str, Any], history_items: List[Dict[str, Any]], profile_position: str = "first") -> List[Dict[str, str]]:
    udesc = user_description(user_info)

    history_lines = []
    for idx, item in enumerate(history_items, start=1):
        title = item.get("title", "")
        genres = item.get("genres", "")
        overview = item.get("overview", "")

        # Titles are provided as evidence, but the model is instructed not to
        # repeat them in the final profile.
        one = [
            f"History item {idx}:",
            f"Title: {title}",
            f"Genres: {genres}",
        ]
        if overview:
            one.append(f"Overview: {overview}")
        history_lines.append("\n".join(one))

    history_block = "\n\n".join(history_lines)

    if profile_position == "last":
        history_label = "Recent training history"
        history_desc = "the most recent training interactions before validation/test"
    else:
        history_label = "Early training history"
        history_desc = "the earliest training interactions"

    system = (
        "You are a recommender-system user profile generator. "
        "Generate a concise preference profile from selected training interactions only. "
        "Return valid JSON only. Do not include explanations outside JSON."
    )

    user = f"""
User information:
{udesc}

{history_label}:
{history_block}

Task:
Infer this user's movie preference profile from {history_desc}.

Important constraints:
1. Use gender and age only to write the field "user_description". Do not infer movie taste from gender or age.
2. In "profile_text", always use "The user" instead of gendered pronouns such as he, she, his, or her.
3. Infer preference only from the history items.
4. Do not mention any historical movie title in the output.
5. Do not mention item IDs, timestamps, ratings, risk scores, tolerance values, or safety labels.
6. Avoid overly generic phrases unless clearly supported by the history.
7. Keep the output compact and stable.

Return exactly one JSON object with this schema:
{{
  "user_description": "{udesc}",
  "profile_text": "2-3 concise sentences about the user's movie preferences, without movie titles.",
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

    # Remove common code fences.
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("No JSON object found in model output")

    js = s[start : end + 1]
    return json.loads(js)


def ensure_list(x: Any, max_len: int, fallback: Optional[List[str]] = None) -> List[str]:
    if isinstance(x, list):
        arr = [clean_text(v, max_chars=80) for v in x if clean_text(v, max_chars=80)]
    elif isinstance(x, str) and x.strip():
        # Handle comma-separated fallback.
        arr = [clean_text(v, max_chars=80) for v in re.split(r"[,;/|]", x) if clean_text(v, max_chars=80)]
    else:
        arr = []

    # Deduplicate case-insensitively while preserving order.
    seen = set()
    out = []
    for v in arr:
        key = v.lower()
        if key not in seen:
            out.append(v)
            seen.add(key)

    if not out and fallback:
        out = fallback[:]

    return out[:max_len]


def infer_fallback_profile(user_info: Dict[str, Any], history_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    genre_counts: Dict[str, int] = defaultdict(int)
    for item in history_items:
        genres = item.get("genres", "")
        for g in re.split(r"[|,/;]", genres):
            g = clean_text(g, max_chars=40)
            if g:
                genre_counts[g] += 1

    top_genres = sorted(genre_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    genres = [g for g, _ in top_genres] or ["Drama", "Comedy", "Action"]

    return {
        "user_description": user_description(user_info),
        "profile_text": (
            "Based on the selected training interactions, the user appears to prefer movies "
            "aligned with the main genres and narrative patterns present in their history. "
            "The profile should be treated as a compact preference estimate from training data only."
        ),
        "preferred_genres": genres[:5],
        "preferred_themes": ["Character-driven stories", "Personal journeys", "Engaging conflicts"],
        "preferred_style": ["Accessible", "Narrative-focused", "Emotionally engaging"],
    }


def normalize_profile_text_pronouns(text: str) -> str:
    """Avoid gendered pronouns in profile_text to reduce demographic-to-preference bias."""
    s = clean_text(text, max_chars=700)

    # Sentence-initial pronouns.
    s = re.sub(r"\b[Ss]he\s+enjoys\b", "The user enjoys", s)
    s = re.sub(r"\b[Hh]e\s+enjoys\b", "The user enjoys", s)
    s = re.sub(r"\b[Ss]he\s+prefers\b", "The user prefers", s)
    s = re.sub(r"\b[Hh]e\s+prefers\b", "The user prefers", s)
    s = re.sub(r"\b[Ss]he\s+likes\b", "The user likes", s)
    s = re.sub(r"\b[Hh]e\s+likes\b", "The user likes", s)
    s = re.sub(r"\b[Ss]he\s+shows\b", "The user shows", s)
    s = re.sub(r"\b[Hh]e\s+shows\b", "The user shows", s)
    s = re.sub(r"\b[Ss]he\s+has\b", "The user has", s)
    s = re.sub(r"\b[Hh]e\s+has\b", "The user has", s)

    # More general replacements. Keep them simple and conservative.
    s = re.sub(r"\b[Ss]he\b", "The user", s)
    s = re.sub(r"\b[Hh]e\b", "The user", s)
    s = re.sub(r"\b[Hh]is\b", "the user's", s)
    s = re.sub(r"\b[Hh]er\b", "the user's", s)

    # Clean repeated spaces and minor artifacts.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_title_leak(profile: Dict[str, Any], history_items: List[Dict[str, Any]]) -> bool:
    titles = []
    for item in history_items:
        t = item.get("title", "")
        if not t:
            continue
        # Remove year suffix for a stronger but still simple check.
        base = re.sub(r"\s*\(\d{4}\)\s*$", "", t).strip()
        if len(base) >= 4:
            titles.append(base.lower())

    profile_text = json.dumps(profile, ensure_ascii=False).lower()
    return any(t in profile_text for t in titles)


def normalize_profile(
    raw_profile: Dict[str, Any],
    user_info: Dict[str, Any],
    history_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fallback = infer_fallback_profile(user_info, history_items)

    profile = {
        "user_description": clean_text(
            raw_profile.get("user_description", "") or fallback["user_description"],
            max_chars=180,
        ),
        "profile_text": normalize_profile_text_pronouns(clean_text(
            raw_profile.get("profile_text", "") or fallback["profile_text"],
            max_chars=700,
        )),
        "preferred_genres": ensure_list(
            raw_profile.get("preferred_genres", []),
            max_len=5,
            fallback=fallback["preferred_genres"],
        ),
        "preferred_themes": ensure_list(
            raw_profile.get("preferred_themes", []),
            max_len=5,
            fallback=fallback["preferred_themes"],
        ),
        "preferred_style": ensure_list(
            raw_profile.get("preferred_style", []),
            max_len=4,
            fallback=fallback["preferred_style"],
        ),
    }

    # Force demographic sentence to our deterministic wording, so F/M never leaks.
    profile["user_description"] = user_description(user_info)

    # If the model leaked historical titles, use fallback profile text/lists but
    # keep the demographic description.
    if contains_title_leak(profile, history_items):
        fb = infer_fallback_profile(user_info, history_items)
        fb["user_description"] = user_description(user_info)
        return fb

    return profile


def load_model(model_path: str, device_map: str = "auto", dtype: str = "auto"):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "This script requires transformers and torch. Please run it in your LLM environment."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
    }

    if dtype == "auto":
        kwargs["dtype"] = "auto"
    elif dtype == "bf16":
        kwargs["dtype"] = torch.bfloat16
    elif dtype == "fp16":
        kwargs["dtype"] = torch.float16
    elif dtype == "fp32":
        kwargs["dtype"] = torch.float32
    else:
        kwargs["dtype"] = "auto"

    # Newer transformers prefer dtype; older versions may only accept torch_dtype.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except TypeError:
        dtype_value = kwargs.pop("dtype", "auto")
        if dtype_value != "auto":
            kwargs["torch_dtype"] = dtype_value
        else:
            kwargs["torch_dtype"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    model.eval()
    return tokenizer, model


def generate_one(
    tokenizer,
    model,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
) -> str:
    import torch

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_safe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of history items used for profile.",
    )
    parser.add_argument(
        "--profile_position",
        choices=["first", "last"],
        default="first",
        help=(
            "first: use the earliest K training interactions. "
            "last: use the most recent K interactions before validation/test, i.e., the tail of train_safe."
        ),
    )
    parser.add_argument("--max_users", type=int, default=-1, help="For debugging. -1 means all users.")
    parser.add_argument("--start_user", type=int, default=0, help="Skip users with inner_user_id < start_user.")
    parser.add_argument("--resume", action="store_true", help="Append and skip users already in output.")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading train_safe: {args.train_safe}")
    by_user = load_train_rows(args.train_safe)
    all_users = sorted(by_user.keys())

    if args.start_user > 0:
        all_users = [u for u in all_users if u >= args.start_user]

    if args.max_users is not None and args.max_users > 0:
        all_users = all_users[: args.max_users]

    done_users = read_existing_users(args.output) if args.resume else set()
    todo_users = [u for u in all_users if u not in done_users]

    print(f"Total users in train_safe: {len(by_user)}")
    print(f"Users selected: {len(all_users)}")
    print(f"Already done: {len(done_users)}")
    print(f"Todo: {len(todo_users)}")
    print(f"History K: {args.k}")
    print(f"Profile position: {args.profile_position}")

    if not todo_users:
        print("Nothing to do.")
        return

    print(f"Loading Qwen: {args.model_path}")
    tokenizer, model = load_model(args.model_path, device_map=args.device_map, dtype=args.dtype)

    mode = "a" if args.resume else "w"
    n_ok = 0
    n_parse_error = 0
    n_title_leak_fallback = 0

    with open(args.output, mode, encoding="utf-8") as out:
        for idx, uid in enumerate(todo_users, start=1):
            rows = by_user[uid]
            rows_sorted = sorted(
                rows,
                key=lambda r: (
                    safe_int(get_first_existing(r, ["timestamp", "time", "ts"], default="0")),
                    safe_int(get_first_existing(r, ["inner_item_id", "item_id", "iid"], default="0")),
                ),
            )

            if args.profile_position == "last":
                # Use the most recent K interactions available in train_safe.
                # For validation-stage profiles, this is exactly "the latest K before validation"
                # as long as train_safe contains only pre-validation interactions.
                selected_rows = rows_sorted[-args.k:]
            else:
                # Original behavior: use the earliest K training interactions.
                selected_rows = rows_sorted[: args.k]

            if not selected_rows:
                continue

            user_info = build_user_info(selected_rows[0], uid)
            history_items = [extract_item(r) for r in selected_rows]
            history_item_ids = [it["inner_item_id"] for it in history_items]
            history_raw_movie_ids = [it["raw_movie_id"] for it in history_items]
            history_timestamps = [it["timestamp"] for it in history_items]

            messages = make_prompt(user_info, history_items, profile_position=args.profile_position)

            parse_error = False
            raw_generation = ""
            title_leak_fallback = False

            try:
                raw_generation = generate_one(
                    tokenizer=tokenizer,
                    model=model,
                    messages=messages,
                    max_new_tokens=args.max_new_tokens,
                )
                raw_profile = extract_json_object(raw_generation)
                title_leak_before = contains_title_leak(raw_profile, history_items)
                profile = normalize_profile(raw_profile, user_info, history_items)
                title_leak_fallback = bool(title_leak_before)
            except Exception as e:
                parse_error = True
                raw_generation = raw_generation or f"[generation_or_parse_error] {repr(e)}"
                profile = infer_fallback_profile(user_info, history_items)

            obj = {
                "profile_id": f"u{uid}_{args.profile_position}{args.k}",
                "user_id": int(uid),
                "profile_source": f"train_{args.profile_position}_{args.k}",
                "user_info": user_info,
                "history_item_ids": history_item_ids,
                "history_raw_movie_ids": history_raw_movie_ids,
                "history_timestamps": history_timestamps,
                "profile": profile,
                "parse_error": bool(parse_error),
                "title_leak_fallback": bool(title_leak_fallback),
            }

            # Keep the raw model output only when parsing failed; useful for debugging
            # and avoids bloating the normal full output file.
            if parse_error:
                obj["raw_generation"] = raw_generation[:2000]
                n_parse_error += 1

            if title_leak_fallback:
                n_title_leak_fallback += 1

            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out.flush()

            n_ok += 1
            if idx == 1 or idx % 50 == 0:
                print(
                    f"[{idx}/{len(todo_users)}] wrote user={uid}, "
                    f"parse_error={parse_error}, title_leak_fallback={title_leak_fallback}"
                )

    print("Done.")
    print(f"Wrote: {args.output}")
    print(f"Generated profiles: {n_ok}")
    print(f"Parse errors: {n_parse_error}")
    print(f"Title leak fallbacks: {n_title_leak_fallback}")


if __name__ == "__main__":
    main()
