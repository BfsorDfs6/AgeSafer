#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERSION: MAL-Risk3-v1.0-pair-aware-rag-cache-from-ML1M-style
SERVER FILE NAME: scripts/mal2000_risk3/03_psg_regulation_rag_full.py

Purpose
-------
Dense-vector RAG rule generator for MAL Risk3 PSG. This follows the ML-1M
RAG style:

  build    : regulation PDFs -> chunks.jsonl / embeddings.npy / vector_index.json
  query    : debug retrieval
  gen-rule : given a concrete user-item pair, build a hidden pair-aware query,
             retrieve regulation chunks, call Qwen to generate an English
             regulatory safety rule, and cache by reusable safety signature.

MAL Risk3 safety rule
---------------------
Risk tags from MAL content rating:
  R17    : R - 17+ (violence & profanity)
  RPLUS  : R+ - Mild Nudity
  RX     : Rx - Hentai

For a concrete user-item pair:
  - minor user: R17 / RPLUS / RX violate safety.
  - adult user: RPLUS / RX violate safety; R17 is not automatically unsafe.

Cache reuse design
------------------
RAG generation is pair-aware in retrieval, but cached by signature:
  audience + triggered risk set + topk + vector index version

Important visible-rule constraint
---------------------------------
The generated visible rule is a regulatory policy, not an explanation for a
specific anime. It must not mention concrete title, concrete user age, item id,
user id, rating_bucket, or hidden labels.

Common commands use CUDA_VISIBLE_DEVICES=3 by default.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

RISK_TAGS = ["r17", "rplus", "rx"]

DIM_KEYWORDS = {
    "r17": [
        "未成年人", "17+", "十七岁", "青少年", "暴力", "血腥", "粗口", "粗俗语言", "恐怖", "惊悚",
        "身心健康", "不良节目", "保护", "age-restricted", "17+", "violence", "profanity", "horror",
    ],
    "rplus": [
        "裸露", "性暗示", "低俗", "挑逗", "衣着暴露", "未成年人不宜", "成人", "身心健康",
        "nudity", "sexual implication", "adult-oriented", "mild nudity",
    ],
    "rx": [
        "色情", "淫秽", "成人内容", "未成年人不宜", "儿童不宜", "禁止", "不得", "低俗", "淫秽色情",
        "pornographic", "explicit adult", "hentai", "adult-only",
    ],
    "minor": ["未成年人", "未成年", "青少年", "儿童", "身心健康", "保护", "适龄", "minor", "child", "under-18"],
    "adult": ["成年人", "一般内容", "合规", "底线", "公众", "社会公德", "adult", "general audience"],
    "general": ["互联网视听", "内容审核", "网络短视频", "禁止", "不得", "违法", "有害", "导向", "公序良俗"],
    "sexual_general": ["色情", "淫秽", "性", "裸露", "低俗", "挑逗", "成人", "性暗示"],
    "violence_general": ["暴力", "血腥", "凶杀", "残忍", "伤害", "恐怖"],
    "harmful_general": ["毒品", "吸毒", "酗酒", "吸烟", "自杀", "自残", "模仿", "危险行为"],
}

RISK_NAME_EN = {
    "r17": "age-restricted 17+ anime content, including stronger violence, profanity, horror, or other material unsuitable for minors",
    "rplus": "adult-oriented nudity or sexualized content indicated by R+ / mild-nudity metadata",
    "rx": "explicit adult-only anime content indicated by Rx / hentai metadata",
}

IRRELEVANT_FOR_ANIME_SAFETY = [
    "马克思主义", "中国特色社会主义", "党中央", "一国两制", "国家秘密", "民族仇恨", "分裂国家",
    "历史虚无主义", "时政类", "新闻采编", "政治", "军事", "外交",
]

FORBIDDEN_VISIBLE_RULE_PATTERNS = [
    r"\buser_id\b", r"\bitem_id\b", r"inner_user_id", r"inner_item_id", r"raw_user_id", r"raw_item_id",
    r"rating_bucket", r"safety_label", r"preference_label", r"quadrant", r"triggered_risks",
    r"\bscore\b", r"\bmy_score\b", r"\bage\s+of\s+\d+", r"\baged\s+\d+",
    r"the anime\s+[\"']", r"this anime\s+is\s+generally", r"given the user's age",
    r"not automatically unsafe", r"should be reviewed for clear age-safety violations",
]


def clean_text(x: Any, max_chars: Optional[int] = None) -> str:
    s = str(x or "").replace("\u3000", " ").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{4,}", "\n\n\n", s).strip()
    if max_chars and len(s) > max_chars:
        cut = s[:max_chars]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        s = cut + "..."
    return s


def flat_text(x: Any, max_chars: Optional[int] = None) -> str:
    return clean_text(x, max_chars=max_chars).replace("\n", " ")


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
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t", "minor", "under18", "under_18", "under 18"}


def first_existing(row: Dict[str, Any], names: Iterable[str], default: str = "") -> str:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return str(row[n])
    return default


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def row_item_id(row: Dict[str, Any]) -> int:
    return safe_int(first_existing(row, ["inner_item_id", "item_id", "iid", "item", "anime_inner_id"], "-1"), -1)


def load_profiles(path: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    for obj in read_jsonl(path):
        if obj.get("parse_error"):
            continue
        uid = safe_int(obj.get("user_id", obj.get("inner_user_id")), -1)
        if uid >= 0:
            out[uid] = obj
    return out


def load_item_safe(path: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    for r in read_csv_rows(path):
        iid = row_item_id(r)
        if iid >= 0:
            out[iid] = r
    return out


def profile_is_minor(profile: Dict[str, Any]) -> bool:
    ui = profile.get("user_info", {}) if profile else {}
    if "is_minor" in ui:
        return bool(ui.get("is_minor"))
    for c in ["is_minor", "minor", "isMinor"]:
        if c in profile:
            return str2bool(profile[c])
    age = first_existing(ui, ["age"], "")
    if age:
        return safe_int(age, 99) < 18
    return False


def item_title(row: Dict[str, Any]) -> str:
    return flat_text(first_existing(row, ["title", "title_english", "name", "anime_title"], "Unknown title"), 180)


def item_genres(row: Dict[str, Any]) -> str:
    return flat_text(first_existing(row, ["genre", "genres"], "Unknown genres"), 220)


def item_type(row: Dict[str, Any]) -> str:
    return flat_text(first_existing(row, ["type"], ""), 80)


def item_source(row: Dict[str, Any]) -> str:
    return flat_text(first_existing(row, ["source"], ""), 100)


def item_rating_text(row: Dict[str, Any]) -> str:
    return flat_text(first_existing(row, ["rating", "content_rating"], ""), 140)


def item_rating_bucket(row: Dict[str, Any]) -> str:
    b = first_existing(row, ["rating_bucket", "bucket", "risk_bucket"], "")
    if b:
        return str(b).strip().upper().replace("-", "").replace("+", "PLUS")
    r = item_rating_text(row).lower()
    if "rx" in r or "hentai" in r:
        return "RX"
    if "r+" in r or "mild nudity" in r or "nudity" in r:
        return "RPLUS"
    if "17" in r or "violence" in r or "profanity" in r:
        return "R17"
    if "pg-13" in r or "teens" in r or "13" in r:
        return "PG13"
    if "pg" in r or "children" in r:
        return "PG"
    if "all ages" in r or r == "g":
        return "G"
    return str(b).strip().upper() if b else "UNKNOWN"


def flag_value(row: Dict[str, Any], names: Iterable[str]) -> bool:
    for n in names:
        if n in row and str(row[n]).strip() != "":
            v = str(row[n]).strip().lower()
            if v in {"1", "true", "yes", "y", "t"}:
                return True
            try:
                return float(v) > 0
            except Exception:
                return v in {"r17", "rplus", "rx"}
    return False


def item_risk_flags(row: Dict[str, Any]) -> Dict[str, bool]:
    bucket = item_rating_bucket(row)
    return {
        "r17": bucket == "R17" or flag_value(row, ["r17_code", "r17", "is_r17"]),
        "rplus": bucket == "RPLUS" or flag_value(row, ["rplus_code", "rplus", "is_rplus"]),
        "rx": bucket == "RX" or flag_value(row, ["rx_code", "rx", "is_rx"]),
    }


def mal_pair_signature(profile: Dict[str, Any], item: Dict[str, Any]) -> Tuple[bool, List[str], str, Dict[str, bool]]:
    is_minor = profile_is_minor(profile)
    flags = item_risk_flags(item)
    triggered = []
    if is_minor:
        for k in RISK_TAGS:
            if flags.get(k):
                triggered.append(k)
    else:
        for k in ["rplus", "rx"]:
            if flags.get(k):
                triggered.append(k)
    return is_minor, sorted(set(triggered)), item_rating_bucket(item), flags


# -----------------------------
# PDF -> chunks
# -----------------------------

def extract_pdf_text_with_pdftotext(pdf_path: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        txt_path = tmp.name
    try:
        cmd = ["pdftotext", "-layout", "-enc", "UTF-8", pdf_path, txt_path]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    finally:
        if os.path.exists(txt_path):
            os.remove(txt_path)


def infer_chunk_tags(text: str, source: str) -> List[str]:
    tags = set()
    full = str(source) + "\n" + str(text)
    for dim, kws in DIM_KEYWORDS.items():
        if any(str(kw).lower() in full.lower() or str(kw) in full for kw in kws):
            tags.add(dim)
    return sorted(tags)


def paragraph_split(text: str) -> List[str]:
    text = clean_text(text)
    for p in [r"(第[一二三四五六七八九十百]+条)", r"([一二三四五六七八九十]+、)", r"（([一二三四五六七八九十]+)）", r"(\d{1,3}\.)", r"(比如：)"]:
        text = re.sub(p, r"\n\1", text)
    parts = []
    for p in re.split(r"\n\s*\n|\n", text):
        p = p.strip()
        if not p:
            continue
        if len(p) <= 3 and not re.search(r"第|条|\d", p):
            continue
        parts.append(p)
    return parts


def split_pdf_text_into_chunks(text: str, source: str, chunk_size: int = 650, overlap: int = 80) -> List[Dict[str, Any]]:
    pages = clean_text(text).split("\f") if "\f" in text else [clean_text(text)]
    raw_chunks: List[Tuple[int, str]] = []
    for page_idx, page_text in enumerate(pages, 1):
        paras = paragraph_split(page_text)
        cur = ""
        for p in paras:
            if len(p) > chunk_size:
                if cur:
                    raw_chunks.append((page_idx, cur))
                    cur = ""
                start = 0
                while start < len(p):
                    end = min(start + chunk_size, len(p))
                    raw_chunks.append((page_idx, p[start:end]))
                    if end >= len(p):
                        break
                    start = max(0, end - overlap)
                continue
            if len(cur) + len(p) + 1 <= chunk_size:
                cur = cur + "\n" + p if cur else p
            else:
                if cur:
                    raw_chunks.append((page_idx, cur))
                tail = cur[-overlap:] if overlap > 0 and cur else ""
                cur = (tail + "\n" + p).strip() if tail else p
        if cur:
            raw_chunks.append((page_idx, cur))
    out = []
    for idx, (page, txt) in enumerate(raw_chunks):
        txt = txt.strip()
        if not txt:
            continue
        out.append({
            "chunk_id": f"{source}::page_{page}::chunk_{idx}",
            "source": source,
            "page": page,
            "text": txt,
            "chunk_tags": infer_chunk_tags(txt, source),
        })
    return out


# -----------------------------
# Dense embedding / retrieval
# -----------------------------

def load_qwen_model(model_path: str, device: str, dtype: str = "bf16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "auto": "auto"}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map.get(dtype, torch.bfloat16),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def encode_texts(tokenizer, model, texts: List[str], device: str, batch_size: int = 4, max_length: int = 768) -> np.ndarray:
    import torch
    vecs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            h = outputs.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vecs.append(pooled.cpu().numpy().astype("float32"))
            del inputs, outputs, h, mask, pooled
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    return np.vstack(vecs) if vecs else np.zeros((0, 1), dtype="float32")


def try_save_faiss(index_path: str, embeddings: np.ndarray) -> bool:
    try:
        import faiss  # type: ignore
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        faiss.write_index(index, index_path)
        return True
    except Exception as e:
        print(f"[warn] faiss unavailable or failed: {e}; saved numpy embeddings only.")
        return False


def dense_retrieve(query_vec: np.ndarray, embeddings: np.ndarray, chunks: List[Dict[str, Any]], topn: int, faiss_path: Optional[str] = None) -> List[Dict[str, Any]]:
    topn = min(topn, len(chunks))
    used_faiss = False
    scores = None
    idxs = None
    if faiss_path and os.path.exists(faiss_path):
        try:
            import faiss  # type: ignore
            index = faiss.read_index(faiss_path)
            scores, idxs = index.search(query_vec.astype("float32"), topn)
            used_faiss = True
        except Exception:
            used_faiss = False
    if not used_faiss:
        sim = embeddings @ query_vec.reshape(-1)
        idxs = np.argsort(-sim)[:topn].reshape(1, -1)
        scores = sim[idxs]
    out = []
    for score, idx in zip(scores[0], idxs[0]):
        r = dict(chunks[int(idx)])
        r["score"] = float(score)
        r["retriever"] = "faiss_dense" if used_faiss else "numpy_dense"
        out.append(r)
    return out


def keyword_hit_count(text: str, dims: List[str]) -> int:
    return sum(1 for d in dims for kw in DIM_KEYWORDS.get(d, []) if str(kw).lower() in str(text).lower() or str(kw) in str(text))


def irrelevant_penalty(text: str) -> float:
    hits = sum(1 for kw in IRRELEVANT_FOR_ANIME_SAFETY if kw in str(text))
    return min(hits * 0.04, 0.20)


def audience_source_score(source: str, is_minor: bool) -> float:
    s = str(source)
    if is_minor:
        if "未成年人" in s:
            return 1.0
        if "互联网视听" in s:
            return 0.75
        if "短视频" in s:
            return 0.65
        return 0.35
    else:
        if "短视频" in s:
            return 0.85
        if "互联网视听" in s:
            return 0.75
        if "未成年人" in s:
            return 0.35
        return 0.50


def rerank_candidates(raw: List[Dict[str, Any]], triggered: List[str], is_minor: bool) -> List[Dict[str, Any]]:
    want_dims = list(triggered) if triggered else []
    if is_minor:
        want_dims.append("minor")
    else:
        want_dims.append("adult")
    want_dims.append("general")
    out = []
    for r in raw:
        text = str(r.get("text", ""))
        tags = set(r.get("chunk_tags", []) or [])
        matched = [d for d in want_dims if d in tags or keyword_hit_count(text, [d]) > 0]
        base = float(r.get("score", 0.0))
        hit = keyword_hit_count(text, want_dims)
        src = audience_source_score(str(r.get("source", "")), is_minor)
        tag_bonus = 0.04 * len(matched)
        if triggered and any(d in matched for d in triggered):
            tag_bonus += 0.12
        if is_minor and "minor" in matched:
            tag_bonus += 0.10
        rerank = base * 0.55 + min(hit, 12) / 12.0 * 0.25 + src * 0.15 + tag_bonus - irrelevant_penalty(text)
        x = dict(r)
        x["matched_tags"] = sorted(set(matched))
        x["rerank_score"] = float(rerank)
        x["keyword_hit_count"] = int(hit)
        out.append(x)
    out.sort(key=lambda z: z.get("rerank_score", 0.0), reverse=True)
    return out


def select_reranked(scored: List[Dict[str, Any]], triggered: List[str], is_minor: bool, topk: int) -> List[Dict[str, Any]]:
    selected, selected_ids = [], set()

    def add_first(cands):
        for r in cands:
            if r["chunk_id"] not in selected_ids:
                selected.append(r)
                selected_ids.add(r["chunk_id"])
                return True
        return False

    if is_minor:
        add_first([r for r in scored if "minor" in r.get("chunk_tags", [])])
    for d in triggered:
        add_first([r for r in scored if d in r.get("chunk_tags", [])])
    add_first([r for r in scored if "general" in r.get("chunk_tags", [])])
    for r in scored:
        if len(selected) >= topk:
            break
        if r["chunk_id"] in selected_ids:
            continue
        if sum(1 for x in selected if x.get("source") == r.get("source")) >= 2:
            continue
        selected.append(r)
        selected_ids.add(r["chunk_id"])
    for r in scored:
        if len(selected) >= topk:
            break
        if r["chunk_id"] not in selected_ids:
            selected.append(r)
            selected_ids.add(r["chunk_id"])
    return selected[:topk]


# -----------------------------
# Pair-aware hidden query, prompt, generation, cache
# -----------------------------

def build_hidden_query(profile: Dict[str, Any], item: Dict[str, Any], is_minor: bool, triggered: List[str], bucket: str, flags: Dict[str, bool]) -> str:
    parts = []
    if is_minor:
        parts.append("未成年人 动漫 网络视听 内容审核 保护 身心健康 适龄 合法权益 不良节目")
        parts.append("minor under-18 child protection age-appropriate anime content safety")
    else:
        parts.append("成年人 动漫 一般内容审核 互联网视听节目 合规 底线要求 公众 社会公德")
        parts.append("adult anime content compliance general audience content safety")
    parts.append("网络视听节目 内容审核 网络短视频内容审核标准细则 互联网视听节目服务管理规定 未成年人节目管理规定 禁止 不得 违法 有害")

    if triggered:
        for d in triggered:
            parts.extend(DIM_KEYWORDS.get(d, []))
    else:
        parts.append("普通适龄 动画 冒险 喜剧 校园 运动 家庭 奇幻 非触发 一般安全")
        if (not is_minor) and flags.get("r17"):
            parts.append("成年人 R17 17+ 非自动阻断 合规底线")

    title = item_title(item)
    genres = item_genres(item)
    rating = item_rating_text(item) or bucket
    typ = item_type(item)
    src = item_source(item)
    # Pair-aware retrieval may use item text, but visible generated rules must not mention it.
    if title:
        parts.append(f"动漫 标题 {title}")
    if genres:
        parts.append(f"类型 {genres}")
    if rating:
        parts.append(f"分级 {rating}")
    if typ:
        parts.append(f"形式 {typ}")
    if src:
        parts.append(f"来源 {src}")
    return " ".join(str(x) for x in parts if str(x).strip())


def make_cache_key(profile: Dict[str, Any], item: Dict[str, Any], triggered: List[str], bucket: str, args, retriever_version: str) -> str:
    is_minor = profile_is_minor(profile)
    audience = "minor" if is_minor else "adult"
    risk_part = "+".join(sorted(triggered)) if triggered else "general"
    bucket_part = bucket or "UNKNOWN"
    base = f"mal_risk3_v1::{audience}::{risk_part}::{bucket_part}::topk{args.topk}::{retriever_version}"
    if args.cache_strategy == "pair":
        uid = str(args.user_id if args.user_id is not None else "raw" + str(args.raw_user_id))
        iid = str(args.item_id if args.item_id is not None else "raw" + str(args.raw_item_id))
        h = hashlib.md5(f"{uid}::{iid}".encode("utf-8")).hexdigest()[:12]
        return base + f"::pair{h}"
    return base


def build_rule_generation_prompt(profile: Dict[str, Any], item: Dict[str, Any], triggered: List[str], bucket: str, flags: Dict[str, bool], selected: List[Dict[str, Any]], args) -> str:
    is_minor = profile_is_minor(profile)
    audience = "minor / under-18 anime viewer" if is_minor else "adult anime viewer"
    has_trigger = bool(triggered)

    if has_trigger:
        topics = [RISK_NAME_EN.get(d, d) for d in triggered]
    else:
        topics = ["general anime content safety with no MAL Risk3 safety trigger"]

    chunk_block = "\n\n".join(
        f"[Chunk {i} | Source: {ch.get('source','')} | Page: {ch.get('page','')} | Dense: {ch.get('score',0):.4f} | Rerank: {ch.get('rerank_score',0):.4f}]\n{ch.get('text','')}"
        for i, ch in enumerate(selected, 1)
    ) or "No regulation chunk was retrieved. Use the given age-aware policy exactly."

    if not has_trigger:
        if is_minor:
            case_instruction = """
This is a GENERAL LOW-RISK / NON-TRIGGERED case for a minor viewer.
- No MAL Risk3 safety trigger applies to this pair.
- Do NOT enumerate R17, RPLUS, and RX as a list of bans in this general case.
- Do NOT say that R17, RPLUS, or RX are "not automatically unsafe".
- Generate a protective but non-over-filtering policy: reject only clear adult-only material, explicit sexual content, severe graphic violence or cruelty, explicit drug abuse, harmful imitation, or highly disturbing content.
- Ordinary age-appropriate anime themes such as adventure, comedy, school life, fantasy, sports, friendship, family, and non-graphic conflict should not be rejected automatically.
""".strip()
        else:
            case_instruction = """
This is a GENERAL LOW-RISK / NON-TRIGGERED case for an adult viewer.
- No MAL Risk3 safety trigger applies to this pair.
- For adults, R17 metadata alone is not a safety violation in this dataset setting.
- Do NOT mention RPLUS or RX unless the retrieved regulation chunks require a general adult-content warning.
- Generate a permissive compliance policy: reject only explicit adult-only, illegal, severe, or clearly harmful content; otherwise allow broadly preference-compatible anime.
""".strip()
    else:
        triggered_names = "; ".join(RISK_NAME_EN.get(d, d) for d in triggered)
        if is_minor:
            case_instruction = f"""
This is a TRIGGERED-RISK case for a minor viewer.
- Focus only on these MAL Risk3 safety topics: {triggered_names}.
- In this dataset setting, every triggered R17, RPLUS, or RX topic is unsafe for minor viewers.
- Do NOT weaken this rule by saying the triggered R17/RPLUS/RX topic is merely "not automatically unsafe", "needs review", or "may be acceptable".
- State the regulatory reason qualitatively using the retrieved chunks: minor protection, age-appropriateness, avoidance of adult-oriented content, violence/profanity, nudity/sexualized content, explicit adult content, harmful imitation, or mental-health harm.
- Do not expand to unrelated risk topics.
""".strip()
        else:
            case_instruction = f"""
This is a TRIGGERED-RISK case for an adult viewer.
- Focus only on these MAL Risk3 safety topics: {triggered_names}.
- In this dataset setting, RPLUS or RX triggered topics are unsafe for safe pseudo-positive construction even for adult viewers.
- Do NOT weaken this rule by saying the triggered RPLUS/RX topic is merely "not automatically unsafe", "needs review", or "may be acceptable".
- State the regulatory reason qualitatively using the retrieved chunks: adult-oriented nudity, sexualized content, explicit adult content, vulgar/low-taste content, public-order compliance, or audiovisual content-review principles.
- Do not expand to unrelated risk topics.
""".strip()

    return f"""
You are generating an English regulatory safety policy for an anime recommendation safety filter.

Audience: {audience}
MAL Risk3 policy topics for this signature: {"; ".join(topics)}

Important constraints:
- Use ONLY the retrieved Chinese regulatory chunks and the MAL Risk3 policy topics to write the policy.
- Write a generic signature-level policy, not a judgment about a specific anime title.
- Do NOT mention concrete anime title, concrete user age, user ID, item ID, raw ID, rating score, hidden labels, or internal fields.
- Do NOT output a recommendation label, numeric rating, or explanation for the specific candidate.
- Do NOT mention the words rating_bucket, triggered_risks, safety_label, preference_label, or quadrant.
- Output 4 to 6 concise English bullets.
- Each bullet must start with "- ".
- The policy should be useful as the "Retrieved Age-aware Safety Policy" in an SFT input.

Case instruction:
{case_instruction}

Retrieved Chinese regulatory chunks:
{chunk_block}

Output the English safety policy:
""".strip()


def sanitize_rule_bullet(x: str, title: str = "") -> str:
    y = re.sub(r"\s+", " ", x).strip()
    y = re.sub(r"^[-–—]+\s*", "- ", y)
    if not y.startswith("- "):
        y = "- " + y
    return y.strip()


def is_bad_rule_bullet(x: str, title: str = "") -> bool:
    low = x.lower()
    if title and len(title) >= 4 and title.lower() in low:
        return True
    return any(re.search(p, low, flags=re.I) for p in FORBIDDEN_VISIBLE_RULE_PATTERNS)


def clean_rule_text(rule_text: str, title: str = "") -> str:
    text = str(rule_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    markers = ["English safety policy:", "Output:", "Generated English regulatory rules:", "Regulatory rules:"]
    for m in markers:
        if m in text:
            text = text.split(m, 1)[1].strip()
    lines = []
    for ln in text.splitlines():
        x = ln.strip()
        if not x:
            continue
        x = re.sub(r"^\d+[\).]\s*", "- ", x)
        x = sanitize_rule_bullet(x, title)
        if is_bad_rule_bullet(x, title):
            continue
        lines.append(x)
    if lines:
        out = []
        seen = set()
        for x in lines:
            k = x.lower()
            if k not in seen:
                out.append(x)
                seen.add(k)
        return "\n".join(out[:6]).strip()
    return flat_text(text, 1200)


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text)))


def rewrite_english_only(tokenizer, model, text: str, device: str, max_new_tokens: int = 384) -> str:
    prompt = f"""
Rewrite the following policy into concise English bullets. Keep only regulatory safety rules.
Do not mention concrete anime title, concrete user age, IDs, internal fields, or labels.
Each bullet must start with "- ".

{text}
""".strip()
    return clean_rule_text(qwen_generate(tokenizer, model, prompt, device=device, max_new_tokens=max_new_tokens))


def qwen_generate(tokenizer, model, prompt: str, device: str, max_new_tokens: int = 512) -> str:
    import torch
    messages = [
        {"role": "system", "content": "You are a careful regulatory policy summarizer. Return concise English bullets only."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
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


def template_fallback_rule(is_minor: bool, triggered: List[str], bucket: str = "") -> str:
    if triggered:
        topics = ", ".join(RISK_NAME_EN.get(d, d) for d in triggered)
        lines = [f"- Focus on the triggered anime content-safety topics: {topics}."]
        if is_minor:
            if "r17" in triggered:
                lines.append("- For minor viewers, age-restricted 17+ anime content should be rejected as unsafe for safe pseudo-positive construction.")
            if "rplus" in triggered:
                lines.append("- For minor viewers, adult-oriented nudity or sexualized content should be rejected as unsafe.")
            if "rx" in triggered:
                lines.append("- For minor viewers, explicit adult-only anime content should be rejected as unsafe.")
        else:
            if "rplus" in triggered:
                lines.append("- For adult viewers, adult-oriented nudity or sexualized content should not be used as a safe pseudo-positive sample.")
            if "rx" in triggered:
                lines.append("- For adult viewers, explicit adult-only anime content should not be used as a safe pseudo-positive sample.")
        lines.append("- Use the retrieved policy as a safety filter and do not expand to unrelated risk topics.")
    else:
        if is_minor:
            lines = [
                "- For minor viewers, this is a general non-triggered case, so safety should be used as a filter for clear age-safety violations rather than a broad content ban.",
                "- Reject only clear adult-only material, explicit sexual content, severe graphic violence or cruelty, explicit drug abuse, harmful imitation, or highly disturbing content.",
                "- Ordinary age-appropriate anime with adventure, comedy, school-life, fantasy, sports, friendship, family, or non-graphic conflict should not be rejected automatically.",
            ]
        else:
            lines = [
                "- For adult viewers, this is a general non-triggered case, so safety should be used as a compliance filter for clear violations rather than a broad content ban.",
                "- Reject only explicit adult-only, illegal, severe, or clearly harmful content; otherwise allow broadly preference-compatible anime.",
                "- R17 metadata alone is not treated as a safety violation for adult viewers in this dataset setting.",
            ]
    lines.append("- Safety should reject clear age-safety violations; otherwise preference-compatible candidates may pass the safety filter.")
    return "\n".join(lines[:6])


# -----------------------------
# Vector index I/O
# -----------------------------

def cmd_build(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {args.pdf_dir}")
    all_chunks = []
    for pdf in pdfs:
        print(f"[pdf] {pdf}")
        text = extract_pdf_text_with_pdftotext(str(pdf))
        chunks = split_pdf_text_into_chunks(text, pdf.stem, chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"  chunks={len(chunks)}")
        all_chunks.extend(chunks)
    save_jsonl(str(out_dir / "chunks.jsonl"), all_chunks)
    tokenizer, model = load_qwen_model(args.model_path, device=args.device, dtype=args.dtype)
    texts = [f"Source: {c['source']}\nPage: {c['page']}\nTags: {', '.join(c.get('chunk_tags', []))}\n{c['text']}" for c in all_chunks]
    embeddings = encode_texts(tokenizer, model, texts, device=args.device, batch_size=args.batch_size, max_length=args.embed_max_length)
    np.save(str(out_dir / "embeddings.npy"), embeddings)
    faiss_ok = try_save_faiss(str(out_dir / "faiss.index"), embeddings)
    meta = {
        "version": "dense_qwen_mean_mal_risk3_from_ml_v1",
        "pdf_dir": args.pdf_dir,
        "num_chunks": len(all_chunks),
        "embedding_shape": list(embeddings.shape),
        "faiss": bool(faiss_ok),
    }
    save_json(str(out_dir / "vector_index.json"), meta)
    print("[done] vector index saved:", out_dir)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def load_vector_index(vector_dir: str) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any], str]:
    vd = Path(vector_dir)
    chunks = read_jsonl(str(vd / "chunks.jsonl"))
    embeddings = np.load(str(vd / "embeddings.npy")) if (vd / "embeddings.npy").exists() else np.zeros((0, 1), dtype="float32")
    meta = load_json(str(vd / "vector_index.json")) if (vd / "vector_index.json").exists() else {}
    faiss_path = str(vd / "faiss.index") if (vd / "faiss.index").exists() else ""
    return chunks, embeddings, meta, faiss_path


def cmd_query(args):
    chunks, embeddings, meta, faiss_path = load_vector_index(args.vector_dir)
    tokenizer, model = load_qwen_model(args.model_path, device=args.device, dtype=args.dtype)
    qv = encode_texts(tokenizer, model, [args.query], device=args.device, batch_size=1, max_length=args.embed_max_length)
    raw = dense_retrieve(qv, embeddings, chunks, topn=args.topn, faiss_path=faiss_path)
    triggered = [x.strip().lower() for x in args.triggered_risks.split(",") if x.strip()]
    scored = rerank_candidates(raw, triggered, args.is_minor)
    selected = select_reranked(scored, triggered, args.is_minor, args.topk)
    print("Vector index:", json.dumps(meta, ensure_ascii=False))
    print("Query:", args.query)
    print("Triggered:", triggered, "is_minor:", args.is_minor)
    for i, r in enumerate(selected, 1):
        print("\n" + "=" * 80)
        print(f"[{i}] {r.get('source')} page={r.get('page')} dense={r.get('score',0):.4f} rerank={r.get('rerank_score',0):.4f}")
        print("tags=", r.get("chunk_tags"), "matched=", r.get("matched_tags"), "hits=", r.get("keyword_hit_count"))
        print(r.get("text", "")[:1400])


def load_rule_cache(path: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for r in read_jsonl(path):
        k = r.get("cache_key")
        if k:
            out[k] = r
    return out


def resolve_profile(profiles: Dict[int, Dict[str, Any]], user_id: Optional[int], raw_user_id: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    if user_id is not None:
        if user_id not in profiles:
            raise KeyError(f"user_id not found: {user_id}")
        return user_id, profiles[user_id]
    if raw_user_id is not None:
        for uid, p in profiles.items():
            if str(p.get("user_info", {}).get("raw_user_id", "")) == str(raw_user_id):
                return uid, p
    raise ValueError("Need --user_id or valid --raw_user_id")


def resolve_item(items: Dict[int, Dict[str, Any]], item_id: Optional[int], raw_item_id: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    if item_id is not None:
        if item_id not in items:
            raise KeyError(f"item_id not found: {item_id}")
        return item_id, items[item_id]
    if raw_item_id is not None:
        for iid, it in items.items():
            if str(first_existing(it, ["raw_item_id", "anime_id", "raw_anime_id"], "")) == str(raw_item_id):
                return iid, it
    raise ValueError("Need --item_id or valid --raw_item_id")


def cmd_gen_rule(args):
    profiles = load_profiles(args.profiles)
    items = load_item_safe(args.item_safe)
    uid, profile = resolve_profile(profiles, args.user_id, args.raw_user_id)
    iid, item = resolve_item(items, args.item_id, args.raw_item_id)
    is_minor, triggered, bucket, flags = mal_pair_signature(profile, item)

    retriever_version = "no_vector"
    meta = {}
    chunks = []
    embeddings = np.zeros((0, 1), dtype="float32")
    faiss_path = ""
    if args.vector_dir:
        chunks, embeddings, meta, faiss_path = load_vector_index(args.vector_dir)
        retriever_version = meta.get("version", "dense_vector")

    cache_path = args.cache_path or (str(Path(args.vector_dir) / "rule_text_cache_mal_risk3_pairaware.jsonl") if args.vector_dir else "outputs/mal2000_risk3_seq50_300/rag_vector/rule_text_cache_mal_risk3_pairaware.jsonl")
    cache_key = make_cache_key(profile, item, triggered, bucket, args, retriever_version)
    if args.cache_strategy != "none" and not args.no_cache:
        cache = load_rule_cache(cache_path)
        if cache_key in cache:
            row = dict(cache[cache_key])
            row["cache_hit"] = True
            if args.json_only:
                print(json.dumps(row, ensure_ascii=False))
            else:
                print("[cache hit]", cache_key)
                print("Audience:", row.get("audience"), "Triggered risks:", row.get("triggered_risks"), "Bucket:", row.get("rating_bucket"))
                print("Generated English regulatory rules:")
                print(row.get("rule_text", ""))
            return

    hidden_query = build_hidden_query(profile, item, is_minor, triggered, bucket, flags)
    selected = []
    title = item_title(item)
    if len(chunks) > 0:
        device = args.device or "cuda"
        tokenizer, model = load_qwen_model(args.model_path, device=device, dtype=args.dtype)
        qv = encode_texts(tokenizer, model, [hidden_query], device=device, batch_size=1, max_length=args.embed_max_length)
        raw = dense_retrieve(qv, embeddings, chunks, topn=args.topn, faiss_path=faiss_path)
        scored = rerank_candidates(raw, triggered, is_minor)
        selected = select_reranked(scored, triggered, is_minor, args.topk)
        prompt = build_rule_generation_prompt(profile, item, triggered, bucket, flags, selected, args)
        rule_text = clean_rule_text(qwen_generate(tokenizer, model, prompt, device=device, max_new_tokens=args.max_new_tokens), title=title)
        if contains_cjk(rule_text):
            rule_text = rewrite_english_only(tokenizer, model, rule_text, device=device, max_new_tokens=min(args.max_new_tokens, 384))
            rule_text = clean_rule_text(rule_text, title=title)
    else:
        if not args.allow_template_fallback:
            raise RuntimeError("No vector index available and --allow_template_fallback is not set.")
        rule_text = template_fallback_rule(is_minor, triggered, bucket)

    if not rule_text.strip():
        if args.allow_template_fallback:
            rule_text = template_fallback_rule(is_minor, triggered, bucket)
        else:
            raise RuntimeError("LLM returned empty or unusable rule text after cleaning.")

    row = {
        "cache_key": cache_key,
        "cache_hit": False,
        "cache_strategy": args.cache_strategy,
        "audience": "minor" if is_minor else "adult",
        "user_id_example": uid,
        "item_id_example": iid,
        "triggered_risks": triggered,
        "rating_bucket": bucket,
        "risk_flags": flags,
        "hidden_query": hidden_query,
        "retrieved_chunks": selected,
        "rule_text": rule_text,
        "retriever": "dense_vector" if len(chunks) > 0 else "template_only",
        "vector_index_version": retriever_version,
    }
    if args.cache_strategy != "none" and not args.no_cache:
        append_jsonl(cache_path, row)
    if args.json_only:
        print(json.dumps(row, ensure_ascii=False))
    else:
        print("Dense vector RAG:")
        print("Vector dir:", args.vector_dir)
        print("User:", uid, "Item:", iid, "Audience:", row["audience"])
        print("Cache key:", cache_key)
        print("Triggered risks:", triggered, "rating_bucket:", bucket, "risk_flags:", flags)
        print("Retrieved chunks:")
        for i, r in enumerate(selected, 1):
            print(f"[{i}] {r.get('source')} page={r.get('page','')} dense={r.get('score',0):.4f} rerank={r.get('rerank_score',0):.4f} tags={r.get('matched_tags', [])}")
        print("Generated English regulatory rules:")
        print(rule_text)


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--pdf_dir", required=True)
    b.add_argument("--out_dir", default="outputs/mal2000_risk3_seq50_300/rag_vector")
    b.add_argument("--model_path", required=True)
    b.add_argument("--chunk_size", type=int, default=650)
    b.add_argument("--overlap", type=int, default=80)
    b.add_argument("--batch_size", type=int, default=2)
    b.add_argument("--embed_max_length", type=int, default=768)
    b.add_argument("--device", default="cuda")
    b.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")

    q = sub.add_parser("query")
    q.add_argument("--vector_dir", required=True)
    q.add_argument("--model_path", required=True)
    q.add_argument("--query", required=True)
    q.add_argument("--triggered_risks", default="")
    q.add_argument("--is_minor", action="store_true")
    q.add_argument("--topn", type=int, default=80)
    q.add_argument("--topk", type=int, default=5)
    q.add_argument("--embed_max_length", type=int, default=768)
    q.add_argument("--device", default="cuda")
    q.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")

    g = sub.add_parser("gen-rule")
    g.add_argument("--profiles", required=True)
    g.add_argument("--item_safe", required=True)
    g.add_argument("--user_id", type=int, default=None)
    g.add_argument("--raw_user_id", default=None)
    g.add_argument("--item_id", type=int, default=None)
    g.add_argument("--raw_item_id", default=None)
    g.add_argument("--raw_movie_id", default=None, help="compatibility alias; use --raw_item_id for MAL")
    g.add_argument("--vector_dir", default="outputs/mal2000_risk3_seq50_300/rag_vector")
    g.add_argument("--model_path", required=True)
    g.add_argument("--cache_path", default="")
    g.add_argument("--cache_strategy", choices=["signature", "pair", "none"], default="signature")
    g.add_argument("--no_cache", action="store_true")
    g.add_argument("--topk", type=int, default=5)
    g.add_argument("--topn", type=int, default=80)
    g.add_argument("--max_new_tokens", type=int, default=512)
    g.add_argument("--embed_max_length", type=int, default=768)
    g.add_argument("--device", default="cuda")
    g.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")
    g.add_argument("--json_only", action="store_true")
    g.add_argument("--allow_template_fallback", action="store_true")
    # Compatibility arguments accepted by old SFT caller; not used for MAL.
    g.add_argument("--minor_block_at", type=float, default=3.0)
    g.add_argument("--adult_block_at", type=float, default=4.0)
    g.add_argument("--isadult_policy", choices=["minor_only", "all", "all_unsafe", "none", "ignore"], default="minor_only")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "raw_movie_id") and args.raw_movie_id and not getattr(args, "raw_item_id", None):
        args.raw_item_id = args.raw_movie_id
    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "query":
        cmd_query(args)
    elif args.cmd == "gen-rule":
        cmd_gen_rule(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
