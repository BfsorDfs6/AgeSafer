#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct unobserved candidate user-item pairs and attach regulation guidance.

This file is part of the lightweight AgeSafer reference implementation
for GMF on ML-1M. Paths and execution are controlled by run_ml1m_gmf.py.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


RISK_DIMS = ["sex_code", "violence_code", "profanity_code", "drug_code", "intense_code"]


# -------------------------
# Basic utilities
# -------------------------

def clean_text(x: Any, max_chars: Optional[int] = None) -> str:
    s = str(x or "").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if max_chars and len(s) > max_chars:
        cut = s[:max_chars]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        s = cut + "..."
    return s


def safe_int(x: Any, default: int = 0) -> int:
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


def str2bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t", "minor", "under 18", "under18"}


def get_first(row: Dict[str, Any], names: Iterable[str], default: str = "") -> str:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return str(row[n])
    return default


def list_to_text(x: Any, max_items: int = 8) -> str:
    if isinstance(x, list):
        arr = [clean_text(v, max_chars=80) for v in x if clean_text(v, max_chars=80)]
    elif isinstance(x, str):
        arr = [clean_text(v, max_chars=80) for v in re.split(r"[,;/|]", x) if clean_text(v, max_chars=80)]
    else:
        arr = []
    out, seen = [], set()
    for v in arr:
        k = v.lower()
        if k not in seen:
            out.append(v)
            seen.add(k)
    return ", ".join(out[:max_items]) if out else "Not specified"


def parse_genres(x: Any) -> Set[str]:
    if isinstance(x, list):
        parts = x
    else:
        parts = re.split(r"[|,;/]", str(x or ""))
    out = set()
    for p in parts:
        p = clean_text(p, max_chars=80).lower()
        if not p or p in {"not specified", "unknown", "none", "nan"}:
            continue
        # normalize common MovieLens variations
        p = p.replace("children's", "children").replace("children", "children")
        p = p.replace("sci-fi", "sci fi")
        out.add(p)
    return out


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


# -------------------------
# Readers
# -------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_profiles(path: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    skipped = 0
    for obj in read_jsonl(path):
        try:
            if obj.get("parse_error"):
                skipped += 1
                continue
            uid = int(obj["user_id"])
            out[uid] = obj
        except Exception:
            skipped += 1
    print(f"[load] profiles={len(out)} skipped={skipped} path={path}")
    return out


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header found: {path}")
        for row in reader:
            rows.append(row)
    print(f"[load] csv rows={len(rows)} path={path}")
    return rows


def row_user_id(row: Dict[str, Any]) -> int:
    return safe_int(get_first(row, ["inner_user_id", "user_id", "uid", "user"], "-1"), -1)


def row_item_id(row: Dict[str, Any]) -> int:
    return safe_int(get_first(row, ["inner_item_id", "item_id", "iid", "item", "movie_inner_id"], "-1"), -1)


def row_timestamp(row: Dict[str, Any]) -> int:
    return safe_int(get_first(row, ["timestamp", "time", "ts"], "0"), 0)


def group_by_user(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    d: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        u = row_user_id(r)
        if u >= 0:
            d[u].append(r)
    for u in d:
        d[u].sort(key=lambda r: (row_timestamp(r), row_item_id(r)))
    return dict(d)


def read_item_map(path: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    for row in read_csv_rows(path):
        iid = row_item_id(row)
        if iid >= 0:
            out[iid] = row
    print(f"[load] item_map items={len(out)}")
    return out


def read_user_tol(path: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    if not path or not os.path.exists(path):
        return out
    for row in read_csv_rows(path):
        uid = row_user_id(row)
        if uid >= 0:
            out[uid] = row
    print(f"[load] user_tol users={len(out)}")
    return out


def read_rating_file(path: str) -> Dict[int, Set[int]]:
    """Read whitespace rating file: user item [rating] [timestamp]."""
    out: Dict[int, Set[int]] = defaultdict(set)
    if not path or not os.path.exists(path):
        print(f"[warn] rating file missing, skipped: {path}")
        return {}
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            u = safe_int(parts[0], -1)
            i = safe_int(parts[1], -1)
            if u >= 0 and i >= 0:
                out[u].add(i)
                n += 1
    print(f"[load] rating pools users={len(out)} pairs={n} path={path}")
    return dict(out)


def read_negative_file(path: str) -> Dict[int, Set[int]]:
    """Read NCF 1+99 negative file: (u,pos)\tneg1\tneg2..."""
    out: Dict[int, Set[int]] = defaultdict(set)
    if not path or not os.path.exists(path):
        print(f"[warn] negative file missing, skipped: {path}")
        return {}
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            m = re.match(r"\((\d+),(\d+)\)", parts[0].strip())
            if m:
                u = int(m.group(1))
                pos = int(m.group(2))
                out[u].add(pos)  # include the positive item in the eval pool too
                n += 1
                for p in parts[1:]:
                    p = p.strip()
                    if p:
                        out[u].add(int(float(p)))
                        n += 1
            else:
                # fallback: first col user, following columns items
                vals = [safe_int(x, -1) for x in re.split(r"\s+", line)]
                if len(vals) >= 2 and vals[0] >= 0:
                    u = vals[0]
                    for i in vals[1:]:
                        if i >= 0:
                            out[u].add(i)
                            n += 1
    print(f"[load] negative pools users={len(out)} items={n} path={path}")
    return dict(out)


def merge_user_sets(*dicts: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = defaultdict(set)
    for d in dicts:
        for u, s in d.items():
            out[u].update(s)
    return dict(out)


# -------------------------
# Item/profile text
# -------------------------

def item_title(row: Dict[str, Any]) -> str:
    return clean_text(get_first(row, ["title", "movie_title", "primaryTitle", "originalTitle", "name"], ""), 180)


def item_genres_text(row: Dict[str, Any]) -> str:
    return clean_text(get_first(row, ["genres", "genre"], ""), 220)


def item_overview(row: Dict[str, Any]) -> str:
    return clean_text(get_first(row, ["overview", "plot", "description", "movie_overview"], ""), 700)


def item_raw_movie_id(row: Dict[str, Any]) -> str:
    return str(get_first(row, ["raw_movie_id", "movie_id", "raw_item_id", "rawMovieId"], ""))


def item_doc(row: Dict[str, Any]) -> str:
    title = item_title(row)
    genres = item_genres_text(row)
    overview = item_overview(row)
    return clean_text(f"Title: {title}. Genres: {genres}. Overview: {overview}", max_chars=1200)


def profile_is_minor(p: Dict[str, Any]) -> bool:
    return bool(p.get("user_info", {}).get("is_minor", False))


def profile_query_text(profile_obj: Dict[str, Any], item_map: Dict[int, Dict[str, Any]], history_window: int) -> str:
    p = profile_obj.get("profile", {})
    parts = []
    ui = profile_obj.get("user_info", {})
    age_desc = ui.get("age_desc", "") or ui.get("age_group", "")
    if age_desc:
        parts.append(f"User age group: {age_desc}.")
    parts.append(str(p.get("user_description", "")))
    parts.append(str(p.get("profile_text", "")))
    parts.append("Preferred genres: " + list_to_text(p.get("preferred_genres", []), 8))
    parts.append("Preferred themes: " + list_to_text(p.get("preferred_themes", []), 8))
    parts.append("Preferred style: " + list_to_text(p.get("preferred_style", []), 6))

    hist_ids = [safe_int(x, -1) for x in profile_obj.get("history_item_ids", [])]
    hist_ids = [x for x in hist_ids if x >= 0][-history_window:]
    for idx, iid in enumerate(hist_ids, 1):
        row = item_map.get(iid, {})
        if not row:
            continue
        parts.append(
            f"History {idx}: Title: {item_title(row)}. Genres: {item_genres_text(row)}. Overview: {item_overview(row)}"
        )
    return clean_text(" ".join(parts), max_chars=4000)


def user_genres(profile_obj: Dict[str, Any], item_map: Dict[int, Dict[str, Any]], history_window: int) -> Set[str]:
    p = profile_obj.get("profile", {})
    g = parse_genres(p.get("preferred_genres", []))
    hist_ids = [safe_int(x, -1) for x in profile_obj.get("history_item_ids", [])]
    for iid in hist_ids[-history_window:]:
        row = item_map.get(iid, {})
        g.update(parse_genres(item_genres_text(row)))
    return g


def profile_prompt_text(profile_obj: Dict[str, Any]) -> str:
    p = profile_obj.get("profile", {})
    user_desc = clean_text(p.get("user_description", "The user is a viewer."), 220)
    profile_text = clean_text(p.get("profile_text", ""), 700)
    genres = list_to_text(p.get("preferred_genres", []), max_items=6)
    themes = list_to_text(p.get("preferred_themes", []), max_items=6)
    style = list_to_text(p.get("preferred_style", []), max_items=5)
    return "\n".join([
        "User Information:",
        user_desc if user_desc else "The user is a viewer.",
        "",
        "User Preference Profile:",
        f"- Summary: {profile_text}",
        f"- Preferred genres: {genres}",
        f"- Preferred themes: {themes}",
        f"- Preferred style: {style}",
    ]).strip()


def history_prompt_text(profile_obj: Dict[str, Any], item_map: Dict[int, Dict[str, Any]], history_window: int) -> str:
    hist_ids = [safe_int(x, -1) for x in profile_obj.get("history_item_ids", [])]
    hist_ids = [x for x in hist_ids if x >= 0][-history_window:]
    if not hist_ids:
        return "Recent User History:\nNo previous training history is available."
    lines = ["Recent User History:"]
    for idx, iid in enumerate(hist_ids, 1):
        row = item_map.get(iid, {})
        lines.append(
            f"{idx}. Title: {item_title(row) or 'Unknown title'}\n"
            f"   Genres: {item_genres_text(row) or 'Unknown genres'}\n"
            f"   Overview: {item_overview(row) or 'No overview available'}"
        )
    return "\n".join(lines)


def candidate_prompt_text(item_row: Dict[str, Any]) -> str:
    return "\n".join([
        "Candidate Movie:",
        f"Title: {item_title(item_row) or 'Unknown title'}",
        f"Genres: {item_genres_text(item_row) or 'Unknown genres'}",
        f"Overview: {item_overview(item_row) or 'No overview available'}",
    ])


def make_instruction() -> str:
    return (
        "Decide whether the candidate movie can be used as a pseudo-positive training sample for this user. "
        "Output only 1 or 0. Output 1 only if BOTH conditions are satisfied: "
        "(1) the candidate is broadly compatible with the user's movie preferences inferred from the profile and recent history; "
        "and (2) the candidate is acceptable under the retrieved age-aware safety policy. "
        "Output 0 if the candidate is preference-incompatible, unsafe, age-inappropriate, or violates the retrieved policy. "
        "Do not explain your answer. Answer only 1 or 0."
    )


def make_input(profile_obj: Dict[str, Any], item_map: Dict[int, Dict[str, Any]], candidate_row: Dict[str, Any], rules_text: str, history_window: int) -> str:
    return "\n\n".join([
        profile_prompt_text(profile_obj),
        history_prompt_text(profile_obj, item_map, history_window),
        rules_text,
        candidate_prompt_text(candidate_row),
        "Question: Is this candidate BOTH preference-compatible with the user and safe under the age-aware policy, so that it can be used as a pseudo-positive sample? Answer only 1 or 0.",
    ]).strip()


# -------------------------
# TF-IDF content recall
# -------------------------

def build_tfidf_scores(item_docs: List[str], query_docs: List[str]):
    """Return item matrix, query matrix if sklearn exists; else fallback sparse structures."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[\w\-']+\b",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm="l2",
        )
        item_X = vec.fit_transform(item_docs)
        query_X = vec.transform(query_docs)
        print(f"[tfidf] sklearn matrix items={item_X.shape} queries={query_X.shape}")
        return "sklearn", item_X, query_X
    except Exception as e:
        print(f"[warn] sklearn unavailable, using slow fallback TF-IDF: {e}")
        return build_tfidf_fallback(item_docs, query_docs)


def tokenize_fallback(text: str) -> List[str]:
    text = text.lower()
    toks = re.findall(r"[a-z0-9_\-']+", text)
    # simple bigrams over word tokens
    toks += [toks[i] + "_" + toks[i + 1] for i in range(max(0, len(toks) - 1))]
    return toks


def build_tfidf_fallback(item_docs: List[str], query_docs: List[str]):
    docs_tokens = [tokenize_fallback(d) for d in item_docs]
    df = Counter()
    for toks in docs_tokens:
        df.update(set(toks))
    n = len(item_docs)
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    item_vecs = []
    postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for idx, toks in enumerate(docs_tokens):
        tf = Counter(toks)
        vec = {}
        norm = 0.0
        for t, c in tf.items():
            if t not in idf:
                continue
            w = (1.0 + math.log(c)) * idf[t]
            vec[t] = w
            norm += w * w
        norm = math.sqrt(norm) or 1.0
        for t in list(vec.keys()):
            vec[t] /= norm
            postings[t].append((idx, vec[t]))
        item_vecs.append(vec)

    query_vecs = []
    for d in query_docs:
        tf = Counter(tokenize_fallback(d))
        vec = {}
        norm = 0.0
        for t, c in tf.items():
            if t not in idf:
                continue
            w = (1.0 + math.log(c)) * idf[t]
            vec[t] = w
            norm += w * w
        norm = math.sqrt(norm) or 1.0
        for t in list(vec.keys()):
            vec[t] /= norm
        query_vecs.append(vec)
    return "fallback", postings, query_vecs


def scores_for_user(tfidf_pack, qidx: int, num_items: int) -> np.ndarray:
    mode = tfidf_pack[0]
    if mode == "sklearn":
        _, item_X, query_X = tfidf_pack
        row = query_X[qidx].dot(item_X.T)
        if hasattr(row, "toarray"):
            return row.toarray().ravel().astype("float32")
        return np.asarray(row).ravel().astype("float32")
    else:
        _, postings, query_vecs = tfidf_pack
        qv = query_vecs[qidx]
        scores = np.zeros(num_items, dtype="float32")
        for t, qw in qv.items():
            for idx, iw in postings.get(t, []):
                scores[idx] += qw * iw
        return scores


# -------------------------
# Safety signature for RAG only
# -------------------------

def get_user_tol(user_tol: Dict[int, Dict[str, Any]], uid: int, dim: str, is_minor: bool, minor_cap: float) -> float:
    row = user_tol.get(uid, {})
    candidates = [f"tol_{dim}", f"{dim}_p75", dim]
    for c in candidates:
        if c in row and str(row[c]).strip() != "":
            val = safe_float(row[c], 3.0 if is_minor else 4.0)
            return min(val, minor_cap) if is_minor else val
    return minor_cap if is_minor else 4.0


def risk_val(item_row: Dict[str, Any], dim: str) -> float:
    if dim == "isAdult":
        return safe_float(get_first(item_row, ["isAdult", "is_adult", "isadult"], "0"), 0.0)
    return safe_float(get_first(item_row, [dim], "0"), 0.0)


def triggered_dims_for_rag(uid: int, profile_obj: Dict[str, Any], item_row: Dict[str, Any], user_tol: Dict[int, Dict[str, Any]], args: argparse.Namespace) -> List[str]:
    is_minor = profile_is_minor(profile_obj)
    dims = []
    if args.isadult_policy in {"all_unsafe", "rag_trigger_only"}:
        if risk_val(item_row, "isAdult") >= 1:
            dims.append("isAdult")
    elif args.isadult_policy == "minor_only" and is_minor and risk_val(item_row, "isAdult") >= 1:
        dims.append("isAdult")

    soft = []
    for d in RISK_DIMS:
        r = risk_val(item_row, d)
        if args.single_severe_hard_block and r >= args.severe_threshold:
            dims.append(d)
            continue
        tol = get_user_tol(user_tol, uid, d, is_minor, args.minor_cap)
        if r > tol:
            soft.append(d)
    block_at = args.minor_exceed_count if is_minor else args.adult_exceed_count
    if len(soft) >= block_at:
        dims.extend(soft)
    elif not dims:
        # Keep one soft signal for more specific RAG retrieval. If none, general policy.
        dims.extend(soft[:1])
    return sorted(set(dims))


# -------------------------
# RAG call and parsing
# -------------------------

def parse_rag_rules(stdout: str) -> Optional[str]:
    if not stdout:
        return None
    text = stdout.replace("\r\n", "\n").replace("\r", "\n")
    marker = "Generated English regulatory rules:"
    if marker in text:
        body = text.split(marker, 1)[1]
    else:
        # cache-hit output uses the same marker, but keep fallback robust
        lines = [ln.strip() for ln in text.splitlines()]
        bullet = [ln for ln in lines if ln.startswith("- ")]
        if len(bullet) >= 2:
            body = "\n".join(bullet)
        else:
            return None
    stop_markers = ["Dense vector RAG:", "Retrieved chunks:", "Vector dir:", "Traceback"]
    for sm in stop_markers:
        if sm in body and sm != "Traceback":
            # usually these appear before marker, not after; do nothing destructive unless needed
            pass
    if "Traceback" in body:
        return None
    lines = []
    for ln in body.splitlines():
        x = ln.strip()
        if not x:
            continue
        if x.startswith("[") and "INFO" in x:
            continue
        if not x.startswith("- "):
            # keep only bullet policy lines
            continue
        lines.append(x)
    if len(lines) < 2:
        return None
    return "Retrieved Age-aware Safety Policy:\n" + "\n".join(lines[:6])


def call_rag(uid: int, iid: int, args: argparse.Namespace) -> str:
    cmd = [
        sys.executable,
        args.rag_script,
        "gen-rule",
        "--vector_dir", args.vector_dir,
        "--model_path", args.model_path,
        "--user_tol", args.user_tol,
        "--item_safe", args.item_safe,
        "--user_id", str(uid),
        "--item_id", str(iid),
        "--topk", str(args.rag_topk),
        "--topn", str(args.rag_topn),
        "--adult_exceed_count", str(args.adult_exceed_count),
    ]
    if args.rag_no_cache:
        cmd.append("--no_cache")
    if args.rag_cache_path:
        cmd.extend(["--cache_path", args.rag_cache_path])
    if args.rag_device:
        cmd.extend(["--device", args.rag_device])

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.rag_timeout,
        check=False,
    )
    rules = parse_rag_rules(p.stdout)
    if p.returncode != 0 or not rules:
        msg = p.stdout[-3000:]
        raise RuntimeError(f"RAG gen-rule failed for user={uid}, item={iid}, returncode={p.returncode}. Last output:\n{msg}")
    return rules


def get_rules_for_sample(uid: int, profile_obj: Dict[str, Any], iid: int, item_row: Dict[str, Any], user_tol: Dict[int, Dict[str, Any]], args: argparse.Namespace, cache: Dict[Tuple[Any, ...], str], stats: Counter) -> Tuple[str, List[str], str]:
    dims = triggered_dims_for_rag(uid, profile_obj, item_row, user_tol, args)
    audience = "minor" if profile_is_minor(profile_obj) else "adult"
    if args.rule_cache_strategy == "signature":
        key = (audience, tuple(dims), args.rag_topk, args.rag_topn)
    elif args.rule_cache_strategy == "item":
        key = (audience, tuple(dims), int(uid), int(iid), args.rag_topk, args.rag_topn)
    else:
        key = (random.random(),)

    if key in cache:
        stats["rag_cache_hits"] += 1
        return cache[key], dims, "rag_cli"

    stats["rag_calls"] += 1
    rules = call_rag(uid, iid, args)
    cache[key] = rules
    return rules, dims, "rag_cli"


# -------------------------
# Candidate proposal
# -------------------------

def select_mmr(recall_indices: List[int], relevance: np.ndarray, item_genres: List[Set[str]], selected_global: Set[int], n: int, lam: float) -> List[int]:
    chosen = []
    candidates = [idx for idx in recall_indices if idx not in selected_global]
    while candidates and len(chosen) < n:
        best_idx = None
        best_score = -1e9
        for idx in candidates:
            if not chosen:
                div_penalty = 0.0
            else:
                div_penalty = max(jaccard(item_genres[idx], item_genres[j]) for j in chosen) if chosen else 0.0
            score = lam * float(relevance[idx]) - (1.0 - lam) * div_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        chosen.append(best_idx)
        selected_global.add(best_idx)
        candidates.remove(best_idx)
    return chosen


def select_popularity(recall_indices: List[int], pop_scores: np.ndarray, selected_global: Set[int], n: int) -> List[int]:
    remain = [idx for idx in recall_indices if idx not in selected_global]
    remain.sort(key=lambda idx: float(pop_scores[idx]), reverse=True)
    out = remain[:n]
    selected_global.update(out)
    return out


def select_diversity(recall_indices: List[int], relevance: np.ndarray, pop_scores: np.ndarray, item_genres: List[Set[str]], selected_global: Set[int], n: int) -> List[int]:
    out = []
    covered: Set[str] = set()
    for idx in selected_global:
        covered.update(item_genres[idx])
    candidates = [idx for idx in recall_indices if idx not in selected_global]
    while candidates and len(out) < n:
        best_idx = None
        best_score = -1e9
        for idx in candidates:
            new_g = item_genres[idx] - covered
            score = len(new_g) + 0.20 * float(relevance[idx]) + 0.10 * float(pop_scores[idx])
            if score > best_score:
                best_score = score
                best_idx = idx
        out.append(best_idx)
        selected_global.add(best_idx)
        covered.update(item_genres[best_idx])
        candidates.remove(best_idx)
    return out


def fill_remaining(recall_indices: List[int], selected_global: Set[int], n: int) -> List[int]:
    out = []
    for idx in recall_indices:
        if idx not in selected_global:
            out.append(idx)
            selected_global.add(idx)
            if len(out) >= n:
                break
    return out


def leakage_check(text: str) -> List[str]:
    forbidden = [
        "sex_code", "violence_code", "profanity_code", "drug_code", "intense_code",
        "isAdult", "is_adult", "tol_", "rating_hidden", "preference_label",
        "safety_label", "inner_item_id", "raw_movie_id", "raw_user_id", "candidate_inner_item_id",
    ]
    low = text.lower()
    return [x for x in forbidden if x.lower() in low]


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--train_safe", required=True)
    ap.add_argument("--item_safe", required=True)
    ap.add_argument("--user_tol", required=True)
    ap.add_argument("--valid_interactions", default="")
    ap.add_argument("--test_interactions", default="")
    ap.add_argument("--valid_negative", default="")
    ap.add_argument("--test_negative", default="")
    ap.add_argument(
        "--eval_negative_policy",
        choices=["ignore", "exclude"],
        default="ignore",
        help=(
            "ignore: do not block sampled valid/test negative pools when proposing PSG candidates; "
            "exclude: old conservative behavior, also blocks valid/test negative pools."
        ),
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--candidates_output", required=True)
    ap.add_argument("--summary_json", required=True)

    ap.add_argument("--topn", type=int, default=100, help="Final candidates per user.")
    ap.add_argument("--recall_topn", type=int, default=600, help="Initial semantic recall size before stratified proposal.")
    ap.add_argument("--pref_quota", type=int, default=60)
    ap.add_argument("--pop_quota", type=int, default=20)
    ap.add_argument("--div_quota", type=int, default=20)
    ap.add_argument("--history_window", type=int, default=5)
    ap.add_argument("--mmr_lambda", type=float, default=0.70)
    ap.add_argument("--max_users", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--minor_only", action="store_true", help="Generate inference candidates only for minor users. Default: all users.")
    ap.add_argument("--strict_no_leakage", action="store_true")

    # Safety/RAG signature settings. Hidden only; never printed into input.
    ap.add_argument("--isadult_policy", choices=["all_unsafe", "minor_only", "rag_trigger_only", "ignore"], default="minor_only")
    ap.add_argument("--minor_cap", type=float, default=3.0)
    ap.add_argument("--minor_exceed_count", type=int, default=2)
    ap.add_argument("--adult_exceed_count", type=int, default=3)
    ap.add_argument("--single_severe_hard_block", action="store_true", default=False)
    ap.add_argument("--severe_threshold", type=float, default=4.0)

    # RAG-only settings.
    ap.add_argument("--rag_script", default="src/agesafer/psg_regulation_rag_full.py")
    ap.add_argument("--vector_dir", default="outputs/psg/rag_vector")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--rag_topk", type=int, default=5)
    ap.add_argument("--rag_topn", type=int, default=80)
    ap.add_argument("--rag_timeout", type=int, default=300)
    ap.add_argument("--rag_no_cache", action="store_true")
    ap.add_argument("--rag_cache_path", default="")
    ap.add_argument("--rag_device", default="", help="Optional device for rag script, e.g. cuda or cpu. Usually leave empty.")
    ap.add_argument("--rule_cache_strategy", choices=["signature", "item", "none"], default="signature")

    args = ap.parse_args()
    rng = random.Random(args.seed)

    # Basic quota check.
    if args.pref_quota + args.pop_quota + args.div_quota > args.topn:
        raise ValueError("pref_quota + pop_quota + div_quota must be <= topn")

    profiles = read_profiles(args.profiles)
    train_rows = read_csv_rows(args.train_safe)
    train_by_user = group_by_user(train_rows)
    item_map = read_item_map(args.item_safe)
    user_tol = read_user_tol(args.user_tol)

    valid_pos = read_rating_file(args.valid_interactions)
    test_pos = read_rating_file(args.test_interactions)
    heldout_pos_pool = merge_user_sets(valid_pos, test_pos)

    if args.eval_negative_policy == "exclude":
        valid_neg = read_negative_file(args.valid_negative)
        test_neg = read_negative_file(args.test_negative)
        candidate_block_pool = merge_user_sets(heldout_pos_pool, valid_neg, test_neg)
        negative_pool_note = "valid/test sampled negative pools are excluded"
    else:
        # Main deployment-faithful setting: sampled negatives are artificial offline
        # evaluation candidates, not observed disliked items. Do not use them to
        # filter PSG inference candidates. The CLI still accepts --valid_negative
        # and --test_negative for compatibility, but they are intentionally ignored.
        candidate_block_pool = heldout_pos_pool
        negative_pool_note = "valid/test sampled negative pools are NOT excluded"

    # Train observed and train-only popularity.
    train_obs: Dict[int, Set[int]] = defaultdict(set)
    pop = Counter()
    for u, rows in train_by_user.items():
        for r in rows:
            iid = row_item_id(r)
            if iid >= 0:
                train_obs[u].add(iid)
                pop[iid] += 1

    all_item_ids = sorted(item_map.keys())
    item_id_to_idx = {iid: idx for idx, iid in enumerate(all_item_ids)}
    idx_to_item_id = {idx: iid for iid, idx in item_id_to_idx.items()}
    item_docs = [item_doc(item_map[iid]) for iid in all_item_ids]
    item_genre_sets = [parse_genres(item_genres_text(item_map[iid])) for iid in all_item_ids]

    max_pop = max([math.log1p(c) for c in pop.values()] or [1.0])
    pop_scores = np.array([math.log1p(pop.get(iid, 0)) / max_pop for iid in all_item_ids], dtype="float32")

    users = sorted(set(profiles.keys()) & set(train_by_user.keys()))
    if args.minor_only:
        users = [u for u in users if profile_is_minor(profiles[u])]
    if args.max_users > 0:
        users = users[:args.max_users]
    print(f"[info] users_to_process={len(users)} minor_only={args.minor_only}")
    print("[info] candidate proposal = semantic recall -> MMR preference + train-popularity support + diversity sampling")
    print(f"[info] block pool = train observed + profile last5 + valid/test positives; {negative_pool_note}")
    print(f"[info] final topn={args.topn}, recall_topn={args.recall_topn}, quotas pref/pop/div={args.pref_quota}/{args.pop_quota}/{args.div_quota}")
    print(f"[info] RAG-only rules from {args.rag_script}, vector_dir={args.vector_dir}, cache_strategy={args.rule_cache_strategy}")

    query_docs = [profile_query_text(profiles[u], item_map, args.history_window) for u in users]
    tfidf_pack = build_tfidf_scores(item_docs, query_docs)

    out_path = Path(args.output)
    cand_path = Path(args.candidates_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cand_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    rule_cache: Dict[Tuple[Any, ...], str] = {}
    bucket_stats = Counter()
    user_counts = []
    leakage_examples = []

    with open(out_path, "w", encoding="utf-8") as fout, open(cand_path, "w", encoding="utf-8") as fcand:
        for qidx, uid in enumerate(users):
            profile_obj = profiles[uid]
            hist_last = {safe_int(x, -1) for x in profile_obj.get("history_item_ids", []) if safe_int(x, -1) >= 0}
            block = set(train_obs.get(uid, set())) | hist_last | set(candidate_block_pool.get(uid, set()))

            # content relevance = tfidf text score + lightweight genre overlap.
            text_scores = scores_for_user(tfidf_pack, qidx, len(all_item_ids))
            ug = user_genres(profile_obj, item_map, args.history_window)
            genre_scores = np.array([jaccard(ug, g) for g in item_genre_sets], dtype="float32")
            relevance = 0.75 * text_scores + 0.25 * genre_scores

            # mask blocked items
            valid_indices = []
            for iid in all_item_ids:
                if iid not in block:
                    valid_indices.append(item_id_to_idx[iid])
            if not valid_indices:
                stats["users_no_candidates"] += 1
                continue

            valid_indices.sort(key=lambda idx: float(relevance[idx]), reverse=True)
            recall = valid_indices[:min(args.recall_topn, len(valid_indices))]

            selected_global: Set[int] = set()
            selected: List[Tuple[int, str]] = []

            pref = select_mmr(recall, relevance, item_genre_sets, selected_global, args.pref_quota, args.mmr_lambda)
            selected += [(idx, "preference_mmr") for idx in pref]

            pop_sel = select_popularity(recall, pop_scores, selected_global, args.pop_quota)
            selected += [(idx, "popularity_supported") for idx in pop_sel]

            div_sel = select_diversity(recall, relevance, pop_scores, item_genre_sets, selected_global, args.div_quota)
            selected += [(idx, "diversity_exploration") for idx in div_sel]

            if len(selected) < args.topn:
                fill = fill_remaining(recall, selected_global, args.topn - len(selected))
                selected += [(idx, "relevance_fill") for idx in fill]

            selected = selected[:args.topn]
            user_counts.append(len(selected))

            for rank, (idx, bucket) in enumerate(selected, 1):
                iid = idx_to_item_id[idx]
                item_row = item_map[iid]
                rules_text, triggered_dims, rule_source = get_rules_for_sample(uid, profile_obj, iid, item_row, user_tol, args, rule_cache, stats)
                input_text = make_input(profile_obj, item_map, item_row, rules_text, args.history_window)
                found = leakage_check(input_text)
                if found:
                    stats["leakage_found"] += 1
                    if len(leakage_examples) < 5:
                        leakage_examples.append({"user_id": uid, "item_id": iid, "found": found, "title": item_title(item_row)})
                    if args.strict_no_leakage:
                        raise RuntimeError(f"Leakage detected in input for user={uid}, item={iid}: {found}")

                meta = {
                    "user_id": int(uid),
                    "raw_user_id": str(profile_obj.get("user_info", {}).get("raw_user_id", "")),
                    "is_minor": bool(profile_is_minor(profile_obj)),
                    "candidate_inner_item_id": int(iid),
                    "candidate_raw_movie_id": item_raw_movie_id(item_row),
                    "candidate_title": item_title(item_row),
                    "candidate_rank": int(rank),
                    "candidate_bucket": bucket,
                    "content_retrieve_score": float(relevance[idx]),
                    "text_score": float(text_scores[idx]),
                    "genre_score": float(genre_scores[idx]),
                    "pop_score": float(pop_scores[idx]),
                    "rule_source": rule_source,
                    "triggered_dims_hidden": triggered_dims,
                    "blocked_pool_size_hidden": int(len(block)),
                    "recall_topn_hidden": int(len(recall)),
                }
                sample = {
                    "instruction": make_instruction(),
                    "input": input_text,
                    "metadata": meta,
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                fcand.write(json.dumps(meta, ensure_ascii=False) + "\n")
                stats["samples"] += 1
                bucket_stats[bucket] += 1

            stats["users_done"] += 1
            if stats["users_done"] % 100 == 0:
                print(
                    f"[progress] users={stats['users_done']}/{len(users)}, samples={stats['samples']}, "
                    f"rag_calls={stats['rag_calls']}, rag_cache_hits={stats['rag_cache_hits']}, rule_cache_entries={len(rule_cache)}"
                )

    summary = {
        "users_requested": len(users),
        "users_done": int(stats["users_done"]),
        "samples": int(stats["samples"]),
        "topn": args.topn,
        "recall_topn": args.recall_topn,
        "eval_negative_policy": args.eval_negative_policy,
        "candidate_block_policy": negative_pool_note,
        "bucket_counts": dict(bucket_stats),
        "rag_calls": int(stats["rag_calls"]),
        "rag_cache_hits": int(stats["rag_cache_hits"]),
        "rule_cache_entries": len(rule_cache),
        "users_no_candidates": int(stats["users_no_candidates"]),
        "candidate_count_per_user": {
            "min": int(min(user_counts)) if user_counts else 0,
            "max": int(max(user_counts)) if user_counts else 0,
            "mean": float(sum(user_counts) / len(user_counts)) if user_counts else 0.0,
        },
        "leakage_found": int(stats["leakage_found"]),
        "leakage_examples": leakage_examples,
        "important_notes": [
            "No GMF/NCF/LightGCN scores are used for candidate proposal.",
            "Train observed items, profile last-K items, and valid/test positive items are excluded.",
            "Sampled valid/test negative pools are not excluded when --eval_negative_policy ignore is used; use --eval_negative_policy exclude to recover the old conservative setting.",
            "Popularity is computed from train interactions only.",
            "RAG rules are generated through dense-vector RAG and reused by signature unless --rule_cache_strategy item is used.",
        ],
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[done]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
