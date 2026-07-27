#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build/query the regulation index and generate pair-aware regulation guidance.

This file is part of the lightweight AgeSafer reference implementation
for GMF on ML-1M. Paths and execution are controlled by run_ml1m_gmf.py.
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

RISK_DIMS = ["sex_code", "violence_code", "profanity_code", "drug_code", "intense_code"]
DIM_SHORT = {
    "sex_code": "sex",
    "violence_code": "violence",
    "profanity_code": "profanity",
    "drug_code": "drug",
    "intense_code": "intense",
}
SHORT_TO_CODE = {v: k for k, v in DIM_SHORT.items()}

DIM_KEYWORDS = {
    "sex": ["色情", "淫秽", "性", "裸露", "低俗", "挑逗", "成人", "性暗示", "sexual", "nudity", "adult"],
    "violence": ["暴力", "血腥", "凶杀", "杀戮", "残忍", "伤害", "恐怖活动", "violence", "gore", "blood"],
    "profanity": ["脏话", "粗口", "粗俗", "辱骂", "低俗语言", "profanity", "vulgar", "abusive"],
    "drug": ["毒品", "吸毒", "酒精", "酗酒", "吸烟", "成瘾", "drug", "alcohol", "smoking", "addiction"],
    "intense": ["惊悚", "恐怖", "惊吓", "紧张", "心理", "危险", "horror", "frightening", "disturbing"],
    "adult_content": ["成人", "未成年人不宜", "儿童不宜", "色情", "淫秽", "低俗", "adult-oriented"],
    "minor": ["未成年人", "未成年", "青少年", "儿童", "身心健康", "保护", "适龄", "minor", "child"],
    "adult": ["成年人", "一般内容", "合规", "底线", "公众", "社会公德", "adult"],
    "general": ["互联网视听", "内容审核", "禁止", "不得", "违法", "有害", "导向", "公序良俗"],
}

RISK_NAME_EN = {
    "sex": "sexual content, nudity, pornography, or adult-oriented sexual material",
    "violence": "violence, gore, cruelty, injury, murder, or bloody scenes",
    "profanity": "profanity, vulgar language, abusive expressions, or insulting language",
    "drug": "drugs, alcohol abuse, smoking, addiction, or harmful imitation",
    "intense": "horror, frightening, disturbing, tense, or psychologically harmful scenes",
    "adult_content": "adult-oriented content metadata",
}

IRRELEVANT_FOR_MOVIE_SAFETY = [
    "马克思主义", "中国特色社会主义", "党中央", "一国两制", "国家秘密", "民族仇恨", "分裂国家",
    "历史虚无主义", "时政类", "新闻采编", "政治", "军事", "外交",
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
    return safe_int(first_existing(row, ["inner_item_id", "item_id", "iid", "item", "movie_inner_id"], "-1"), -1)


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
    age_desc = str(ui.get("age_desc", ui.get("age_group", ""))).lower()
    if age_desc:
        return "under" in age_desc or "minor" in age_desc or "children" in age_desc
    age = first_existing(ui, ["age"], "")
    if age:
        return safe_int(age, 99) < 18
    return False


def item_title(row: Dict[str, Any]) -> str:
    return flat_text(first_existing(row, ["title", "movie_title", "primaryTitle", "originalTitle", "name"], "Unknown title"), 180)


def item_genres(row: Dict[str, Any]) -> str:
    return flat_text(first_existing(row, ["genres", "genre"], "Unknown genres"), 220)


def item_overview(row: Dict[str, Any]) -> str:
    return flat_text(first_existing(row, ["overview", "plot", "description", "movie_overview"], ""), 700)


def risk_value(item: Dict[str, Any], dim_code: str) -> float:
    short = DIM_SHORT.get(dim_code, dim_code.replace("_code", ""))
    return safe_float(first_existing(item, [dim_code, short, "item_" + dim_code, "risk_" + dim_code, short + "_risk"], "0"), 0.0)


def adult_value(item: Dict[str, Any]) -> float:
    return safe_float(first_existing(item, ["isAdult", "is_adult", "isadult", "adult"], "0"), 0.0)


def dim5_pair_signature(profile: Dict[str, Any], item: Dict[str, Any], minor_block_at: float, adult_block_at: float, isadult_policy: str) -> Tuple[bool, List[str], bool, bool]:
    """Return (is_minor, violated short dims, adult_flag, adult_hard_block)."""
    is_minor = profile_is_minor(profile)
    threshold = float(minor_block_at if is_minor else adult_block_at)
    dims = []
    for code in RISK_DIMS:
        if risk_value(item, code) >= threshold:
            dims.append(DIM_SHORT[code])
    adult_flag = adult_value(item) >= 1
    adult_hard = False
    if adult_flag:
        if isadult_policy in {"all", "all_unsafe"}:
            adult_hard = True
        elif isadult_policy == "minor_only" and is_minor:
            adult_hard = True
    return is_minor, sorted(set(dims)), bool(adult_flag), bool(adult_hard)


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
    hits = sum(1 for kw in IRRELEVANT_FOR_MOVIE_SAFETY if kw in str(text))
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
    if "互联网视听" in s:
        return 1.0
    if "短视频" in s:
        return 0.85
    if "未成年人" in s:
        return 0.15
    return 0.50


def rerank_candidates(raw: List[Dict[str, Any]], triggered_dims: List[str], is_minor: bool, adult_hard: bool) -> List[Dict[str, Any]]:
    if not raw:
        return []
    max_dense = max(abs(float(r.get("score", 0.0))) for r in raw) or 1.0
    target_tags = set(triggered_dims)
    if adult_hard:
        target_tags.add("adult_content")
    target_tags.add("minor" if is_minor else "adult")
    target_tags.add("general")
    tag_weights = {t: 1.0 for t in target_tags}
    if is_minor:
        tag_weights["minor"] = 1.35
    for d in triggered_dims:
        tag_weights[d] = 1.45
    if adult_hard:
        tag_weights["adult_content"] = 1.55
    tag_weights["general"] = 0.75
    denom = max(sum(tag_weights.values()), 1.0)
    scored = []
    for r in raw:
        text, source = str(r.get("text", "")), str(r.get("source", ""))
        chunk_tags = set(r.get("chunk_tags", [])) or set(infer_chunk_tags(text, source))
        matched = chunk_tags.intersection(target_tags)
        dense_norm = float(r.get("score", 0.0)) / max_dense
        tag_score = sum(tag_weights.get(t, 1.0) for t in matched) / denom
        kh = keyword_hit_count(text, list(target_tags))
        keyword_score = min(kh / 4.0, 1.0)
        src_score = audience_source_score(source, is_minor)
        length_score = min(len(text) / 450.0, 1.0)
        penalty = irrelevant_penalty(text)
        final_score = 0.35 * dense_norm + 0.35 * tag_score + 0.15 * keyword_score + 0.10 * src_score + 0.05 * length_score - penalty
        if not matched and dense_norm < 0.10:
            continue
        nr = dict(r)
        nr.update({
            "chunk_tags": sorted(chunk_tags),
            "matched_tags": sorted(matched),
            "rerank_score": float(final_score),
            "keyword_hit_count": kh,
            "irrelevant_penalty": penalty,
        })
        scored.append(nr)
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored


def select_reranked(scored: List[Dict[str, Any]], triggered_dims: List[str], is_minor: bool, adult_hard: bool, topk: int) -> List[Dict[str, Any]]:
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
    if adult_hard:
        add_first([r for r in scored if "adult_content" in r.get("chunk_tags", [])])
    for d in triggered_dims:
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

def build_hidden_query(profile: Dict[str, Any], item: Dict[str, Any], is_minor: bool, dims: List[str], adult_flag: bool, adult_hard: bool) -> str:
    parts = []
    if is_minor:
        parts.append("未成年人 电影 内容审核 保护 身心健康 适龄 合法权益 不良节目")
        parts.append("minor under-18 child protection age-appropriate movie content safety")
    else:
        parts.append("成年人 电影 一般内容审核 互联网视听节目 合规 底线要求 公众 社会公德")
        parts.append("adult movie content compliance general audience content safety")
    parts.append("网络视听节目 内容审核 网络短视频内容审核标准细则 互联网视听节目服务管理规定 禁止 不得 违法 有害")
    if adult_hard:
        parts.append("成人内容 未成年人不宜 儿童不宜 色情 淫秽 低俗 adult-oriented content hard block for minors")
    elif adult_flag and not is_minor:
        parts.append("成人内容 成年人 合规底线 adult-oriented metadata not automatically unsafe for adults")
    for d in dims:
        parts.extend(DIM_KEYWORDS.get(d, []))
    title = item_title(item)
    genres = item_genres(item)
    overview = item_overview(item)
    if title:
        parts.append(f"电影 标题 {title}")
    if genres:
        parts.append(f"类型 {genres}")
    if overview:
        parts.append(overview[:300])
    return " ".join(str(x) for x in parts if str(x).strip())


def make_cache_key(profile: Dict[str, Any], item: Dict[str, Any], dims: List[str], adult_flag: bool, adult_hard: bool, args, retriever_version: str) -> str:
    is_minor = profile_is_minor(profile)
    audience = "minor" if is_minor else "adult"
    dims_part = "+".join(sorted(dims)) if dims else "general"
    adult_part = "adultHard" if adult_hard else ("adultFlag" if adult_flag else "noAdultFlag")
    threshold_part = f"m{args.minor_block_at:g}_a{args.adult_block_at:g}_{args.isadult_policy}"
    base = f"dim5_v4_2::{audience}::{dims_part}::{adult_part}::{threshold_part}::topk{args.topk}::{retriever_version}"
    if args.cache_strategy == "pair":
        uid = str(args.user_id if args.user_id is not None else "raw" + str(args.raw_user_id))
        iid = str(args.item_id if args.item_id is not None else "raw" + str(args.raw_movie_id))
        h = hashlib.md5(f"{uid}::{iid}".encode("utf-8")).hexdigest()[:12]
        return base + f"::pair{h}"
    return base


def build_rule_generation_prompt(profile: Dict[str, Any], item: Dict[str, Any], dims: List[str], adult_flag: bool, adult_hard: bool, selected: List[Dict[str, Any]], args) -> str:
    """Build the LLM prompt for pair-aware regulatory rule generation.

    v4.1 important behavior:
    - If no Dim5 risk dimension is triggered and no minor-isAdult hard block is
      triggered, this is a GENERAL / LOW-RISK case. The generated policy must be
      permissive: reject only clear adult-only or severe harmful content and do
      not list every dimension as strictly prohibited.
    - If one or more dimensions are triggered, focus on those triggered topics
      only. Do not expand to unrelated risk dimensions.
    """
    is_minor = profile_is_minor(profile)
    audience = "minor / under-18 user" if is_minor else "adult user"
    # The model prompt must not expose hidden numeric thresholds.  The actual
    # violation logic is computed in Python and encoded in `dims`/`adult_hard`;
    # the generated rule should describe the policy qualitatively.
    threshold_text = (
        "clear, significant, or severe age-inappropriate content violates safety" if is_minor
        else "only the highest-severity content-compliance risks violate safety"
    )
    safe_text = (
        "mild, implicit, brief, non-graphic, or ordinary dramatic content is acceptable" if is_minor
        else "mild, moderate, or contextually justified mature content is acceptable"
    )

    has_dim_trigger = bool(dims)
    has_any_trigger = has_dim_trigger or bool(adult_hard)

    if has_dim_trigger:
        topics = [RISK_NAME_EN.get(d, d) for d in dims]
    else:
        topics = ["general age-appropriate movie content safety with no triggered risk dimension"]
    if adult_hard:
        topics.append(RISK_NAME_EN["adult_content"])

    chunk_block = "\n\n".join(
        f"[Chunk {i} | Source: {ch.get('source','')} | Page: {ch.get('page','')} | Dense: {ch.get('score',0):.4f} | Rerank: {ch.get('rerank_score',0):.4f}]\n{ch.get('text','')}"
        for i, ch in enumerate(selected, 1)
    ) or "No regulation chunk was retrieved. Use the given age-aware policy exactly."

    adult_note = (
        "Adult-oriented metadata is a hard block for minors." if adult_hard else
        "Adult-oriented metadata alone is not a hard block for adults; apply only the five-dimensional high-risk rule." if adult_flag and not is_minor else
        "No adult-oriented hard block is triggered."
    )

    if not has_any_trigger:
        case_instruction = """
This is a GENERAL LOW-RISK / NON-TRIGGERED case.
- No five-dimensional risk topic is triggered for this pair.
- Do NOT write that sexual content, profanity, drug-related content, violence, or intense content is categorically prohibited.
- Do NOT enumerate all five dimensions as strict bans.
- For minor users, use a protective but non-over-filtering rule: reject only clear adult-only material, explicit sexual content, severe graphic violence or cruelty, explicit drug abuse, harmful imitation, or highly disturbing content.
- Ordinary safe movies should not be rejected just because they contain common movie themes such as mild romance, ordinary dramatic conflict, non-graphic action, historical tension, family/social themes, or brief non-prominent mature cues.
- The policy should make clear that when no specific risk is triggered, safety should act as a filter for clear violations, not as a broad content ban.
""".strip()
    else:
        triggered_names = ", ".join(RISK_NAME_EN.get(d, d) for d in dims) if dims else "adult-oriented hard-block metadata"
        case_instruction = f"""
This is a TRIGGERED-RISK case.
- Focus on these triggered risk topics only: {triggered_names}.
- Do not expand the policy to unrelated risk dimensions unless the retrieved regulatory chunks directly require it.
- State hard-block conditions for clearly unsafe forms of the triggered topics.
- Also state acceptable soft-risk cases, such as mild, implicit, brief, non-graphic, non-glamorized, or contextually justified content, when they do not cross the age-specific safety boundary.
""".strip()

    return f"""
You are a regulation-aware policy generator for a movie recommendation system.

Task:
Generate an English safety policy for deciding whether a candidate movie is age-appropriate for use as a safe pseudo-positive sample. The downstream PSG model will separately consider user preference; your output is only the safety policy.

Concrete audience:
{audience}

Focused risk topics:
{', '.join(topics)}

New metric-aligned safety rule to obey internally:
- The system has already converted the hidden item-side risk scores into the focused risk topics shown above.
- Do NOT mention numeric risk levels, numeric thresholds, hidden scores, or the phrases "level 3" / "level 4" in the output.
- For minor users, {threshold_text}; {safe_text}.
- For adult users, only highest-severity compliance risks should be treated as unsafe; mild, moderate, or ordinary mature themes are acceptable.
- If any focused risk topic is truly age-inappropriate for the audience, the pair should be considered unsafe.
- {adult_note}

Case-specific writing instruction:
{case_instruction}

General output requirements:
1. Use the retrieved Chinese regulation chunks as supporting evidence.
2. Write English only.
3. Do not mention numeric hidden risk scores, numeric risk levels, thresholds, item IDs, user IDs, ratings, labels, cache keys, or training metadata.
4. Do not use phrases such as "risk level 3", "risk level 4", "five risk dimensions", or "if any dimension reaches".
5. Do not output a final 0/1 decision.
6. Use 4 to 6 bullet points. Every bullet must start with "- ".
7. Include both rejection conditions and acceptable conditions; avoid over-filtering.
8. End with a principle that only clear age-safety violations should be rejected; otherwise broadly preference-compatible candidates may pass the safety filter.

Retrieved Chinese regulatory chunks:
{chunk_block}

Output the English safety policy:
""".strip()


FORBIDDEN_RULE_PATTERNS = [
    r"\brisk\s*level\s*[34]\b",
    r"\blevel\s*[34]\b",
    r"five\s+risk\s+dimensions",
    r"five-dimensional\s+risk",
    r"if\s+any\s+(one\s+)?dimension\s+(reaches|reach|is|are)",
    r"historically\s+inaccurate",
    r"early\s+romantic\s+relationships",
]


def sanitize_rule_bullet(x: str) -> str:
    # Remove hidden-threshold leakage if the LLM ignores the prompt.  We do not
    # want SFT visible inputs to contain numeric risk levels or training-rule
    # thresholds.
    y = re.sub(r"risk\s*level\s*3\s*or\s*4", "clear age-inappropriate severity", x, flags=re.I)
    y = re.sub(r"risk\s*level\s*4", "highest-severity risk", y, flags=re.I)
    y = re.sub(r"level\s*3\s*or\s*4", "clear age-inappropriate severity", y, flags=re.I)
    y = re.sub(r"level\s*4", "highest-severity risk", y, flags=re.I)
    y = re.sub(r"five\s+risk\s+dimensions", "focused content-safety topics", y, flags=re.I)
    y = re.sub(r"five-dimensional\s+risk", "content-safety", y, flags=re.I)
    return y.strip()


def is_bad_rule_bullet(x: str) -> bool:
    low = x.lower()
    return any(re.search(p, low, flags=re.I) for p in FORBIDDEN_RULE_PATTERNS)


def clean_rule_text(rule_text: str) -> str:
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
        if not x.startswith("- "):
            low = x.lower()
            if low.startswith(("here are", "below are", "based on")):
                continue
            x = "- " + x
        x = sanitize_rule_bullet(x)
        # Drop very broad or threshold-leaking bullets that are likely to make
        # the SFT prompt over-conservative.  The remaining bullets still carry
        # the usable regulatory policy.
        if is_bad_rule_bullet(x):
            continue
        lines.append(x)
    if lines:
        return "\n".join(lines[:6]).strip()
    return flat_text(sanitize_rule_bullet(text), 1200)


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text)))


def qwen_generate(tokenizer, model, prompt: str, device: str, max_new_tokens: int) -> str:
    import torch
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    enc = tokenizer([text], return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def rewrite_english_only(tokenizer, model, rule_text: str, device: str, max_new_tokens: int) -> str:
    prompt = f"""
Rewrite the following rules into English only.
Requirements:
- Translate every Chinese phrase.
- Keep 3 to 6 concise bullet points.
- Every bullet starts with "- ".
- Do not output explanations or labels.

Rules:
{rule_text}

English-only bullet rules:
""".strip()
    return clean_rule_text(qwen_generate(tokenizer, model, prompt, device, max_new_tokens=max_new_tokens))


def load_rule_cache(path: str) -> Dict[str, Dict[str, Any]]:
    cache = {}
    for row in read_jsonl(path):
        key = row.get("cache_key")
        if key:
            cache[key] = row
    return cache


def resolve_item(items: Dict[int, Dict[str, Any]], item_id: Optional[int], raw_movie_id: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    if item_id is not None:
        iid = int(item_id)
        if iid not in items:
            raise KeyError(f"item_id not found in item_safe: {iid}")
        return iid, items[iid]
    if raw_movie_id is None:
        raise ValueError("Need --item_id or --raw_movie_id")
    for iid, row in items.items():
        if str(first_existing(row, ["raw_movie_id", "movie_id", "raw_item_id", "movieId"], "")) == str(raw_movie_id):
            return iid, row
    raise KeyError(f"raw_movie_id not found: {raw_movie_id}")


def resolve_profile(profiles: Dict[int, Dict[str, Any]], user_id: Optional[int], raw_user_id: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    if user_id is not None:
        uid = int(user_id)
        if uid not in profiles:
            raise KeyError(f"user_id not found in profiles: {uid}")
        return uid, profiles[uid]
    if raw_user_id is None:
        raise ValueError("Need --user_id or --raw_user_id")
    for uid, row in profiles.items():
        ui = row.get("user_info", {}) if isinstance(row, dict) else {}
        if str(ui.get("raw_user_id", row.get("raw_user_id", ""))) == str(raw_user_id):
            return uid, row
    raise KeyError(f"raw_user_id not found: {raw_user_id}")


def template_fallback_rule(is_minor: bool, dims: List[str], adult_flag: bool, adult_hard: bool) -> str:
    aud = "minor users" if is_minor else "adult users"
    if dims:
        topics = ", ".join(RISK_NAME_EN.get(d, d) for d in dims)
        lines = [
            f"- For {aud}, focus on the triggered content-safety topics: {topics}.",
            "- Reject clear age-inappropriate cases such as explicit, severe, glamorized, imitable, or highly disturbing forms of the triggered topics.",
            "- Do not reject mild, implicit, brief, non-graphic, non-glamorized, or contextually justified content automatically.",
        ]
    else:
        lines = [
            f"- For {aud}, this is a general non-triggered case, so safety should be used as a filter for clear violations rather than a broad content ban.",
            "- Reject only clear adult-only material, explicit sexual content, severe graphic violence or cruelty, explicit drug abuse, harmful imitation, or highly disturbing content.",
            "- Ordinary safe movies with mild romance, ordinary dramatic conflict, non-graphic action, historical tension, family or social themes, or brief non-prominent mature cues should not be rejected automatically.",
        ]
    if adult_hard:
        lines.append("- Adult-oriented metadata is a hard block for minors even if other content cues appear moderate.")
    elif adult_flag and not is_minor:
        lines.append("- Adult-oriented metadata alone is not a hard block for adults unless content-compliance risks are clearly severe.")
    lines.append("- Use the policy as a safety filter: reject clear age-safety violations; otherwise allow broadly preference-compatible candidates.")
    return "\n".join(lines)


# -----------------------------
# CLI commands
# -----------------------------

def cmd_build(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_chunks: List[Dict[str, Any]] = []
    for fn in sorted(os.listdir(args.pdf_dir)):
        if not fn.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(args.pdf_dir, fn)
        source = os.path.splitext(fn)[0]
        print("[build] extracting:", pdf_path)
        text = extract_pdf_text_with_pdftotext(pdf_path)
        chunks = split_pdf_text_into_chunks(text, source, chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"[build] {source}: chunks={len(chunks)}")
        all_chunks.extend(chunks)
    if not all_chunks:
        raise RuntimeError("No chunks extracted. Check pdf_dir and pdftotext.")
    print(f"[build] total chunks={len(all_chunks)}")
    device = args.device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu")
    tokenizer, model = load_qwen_model(args.model_path, device=device, dtype=args.dtype)
    texts = []
    for ch in all_chunks:
        tag_text = " ".join(ch.get("chunk_tags", []))
        texts.append(f"Document source: {ch['source']}\nTags: {tag_text}\nText:\n{ch['text']}")
    print(f"[build] encoding chunks with dense embeddings, batch_size={args.batch_size}")
    embeddings = encode_texts(tokenizer, model, texts, device=device, batch_size=args.batch_size, max_length=args.embed_max_length)
    chunks_path = out_dir / "chunks.jsonl"
    emb_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "vector_index.json"
    faiss_path = out_dir / "index.faiss"
    save_jsonl(str(chunks_path), all_chunks)
    np.save(str(emb_path), embeddings.astype("float32"))
    faiss_saved = try_save_faiss(str(faiss_path), embeddings)
    save_json(str(meta_path), {
        "version": "dense_qwen_mean_dim5_v4",
        "retriever": "dense_vector",
        "embedding_model_path": args.model_path,
        "embedding_pooling": "last_hidden_state_mean_pooling_l2_norm",
        "chunk_count": len(all_chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "chunks_path": str(chunks_path),
        "embeddings_path": str(emb_path),
        "faiss_path": str(faiss_path) if faiss_saved else "",
        "faiss_saved": bool(faiss_saved),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
    })
    print("Saved dense vector RAG index:")
    print(" ", chunks_path)
    print(" ", emb_path)
    print(" ", meta_path)
    if faiss_saved:
        print(" ", faiss_path)


def load_vector_index(vector_dir: str) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any], str]:
    vd = Path(vector_dir)
    chunks_path = vd / "chunks.jsonl"
    emb_path = vd / "embeddings.npy"
    meta_path = vd / "vector_index.json"
    if not chunks_path.exists() or not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Dense vector index not found under {vector_dir}. Run build first or point --vector_dir to existing outputs/psg/rag_vector.")
    chunks = read_jsonl(str(chunks_path))
    embeddings = np.load(str(emb_path)).astype("float32")
    meta = load_json(str(meta_path))
    faiss_path = str(vd / "index.faiss") if (vd / "index.faiss").exists() else ""
    return chunks, embeddings, meta, faiss_path


def cmd_query(args):
    chunks, embeddings, meta, faiss_path = load_vector_index(args.vector_dir)
    device = args.device or "cuda"
    tokenizer, model = load_qwen_model(args.model_path, device=device, dtype=args.dtype)
    qv = encode_texts(tokenizer, model, [args.query], device=device, batch_size=1, max_length=args.embed_max_length)
    raw = dense_retrieve(qv, embeddings, chunks, topn=args.topn, faiss_path=faiss_path)
    dims = [x.strip() for x in re.split(r"[,;+\s]+", args.triggered_dims) if x.strip()]
    scored = rerank_candidates(raw, dims, args.is_minor, args.adult_hard)
    selected = select_reranked(scored, dims, args.is_minor, args.adult_hard, args.topk)
    print(f"Vector index: {meta.get('version')} chunks={len(chunks)} dim={embeddings.shape[1]}")
    for i, r in enumerate(selected, 1):
        print(f"\n[Retrieved Chunk {i}] Source={r.get('source')} Page={r.get('page','')} Dense={r.get('score',0):.4f} Rerank={r.get('rerank_score',0):.4f}")
        print("Matched tags:", r.get("matched_tags", []), "Chunk tags:", r.get("chunk_tags", []))
        print(str(r.get("text", ""))[:1000])


def cmd_gen_rule(args):
    profiles = load_profiles(args.profiles)
    items = load_item_safe(args.item_safe)
    uid, profile = resolve_profile(profiles, args.user_id, args.raw_user_id)
    iid, item = resolve_item(items, args.item_id, args.raw_movie_id)
    is_minor, dims, adult_flag, adult_hard = dim5_pair_signature(profile, item, args.minor_block_at, args.adult_block_at, args.isadult_policy)

    retriever_version = "no_vector"
    meta = {}
    chunks = []
    embeddings = np.zeros((0, 1), dtype="float32")
    faiss_path = ""
    if args.vector_dir:
        chunks, embeddings, meta, faiss_path = load_vector_index(args.vector_dir)
        retriever_version = meta.get("version", "dense_vector")

    cache_path = args.cache_path or (str(Path(args.vector_dir) / "rule_text_cache_dim5_pairaware.jsonl") if args.vector_dir else "outputs/psg_dim5_v4/rag_vector/rule_text_cache_dim5_pairaware.jsonl")
    cache_key = make_cache_key(profile, item, dims, adult_flag, adult_hard, args, retriever_version)
    if args.cache_strategy != "none" and not args.no_cache:
        cache = load_rule_cache(cache_path)
        if cache_key in cache:
            row = dict(cache[cache_key])
            row["cache_hit"] = True
            if args.json_only:
                print(json.dumps(row, ensure_ascii=False))
            else:
                print("[cache hit]", cache_key)
                print("Audience:", row.get("audience"), "Triggered dims:", row.get("triggered_dims"), "Adult hard:", row.get("adult_hard"))
                print("Generated English regulatory rules:")
                print(row.get("rule_text", ""))
            return

    hidden_query = build_hidden_query(profile, item, is_minor, dims, adult_flag, adult_hard)
    selected = []
    if len(chunks) > 0:
        device = args.device or "cuda"
        tokenizer, model = load_qwen_model(args.model_path, device=device, dtype=args.dtype)
        qv = encode_texts(tokenizer, model, [hidden_query], device=device, batch_size=1, max_length=args.embed_max_length)
        raw = dense_retrieve(qv, embeddings, chunks, topn=args.topn, faiss_path=faiss_path)
        scored = rerank_candidates(raw, dims, is_minor, adult_hard)
        selected = select_reranked(scored, dims, is_minor, adult_hard, args.topk)
        prompt = build_rule_generation_prompt(profile, item, dims, adult_flag, adult_hard, selected, args)
        rule_text = clean_rule_text(qwen_generate(tokenizer, model, prompt, device=device, max_new_tokens=args.max_new_tokens))
        if contains_cjk(rule_text):
            rule_text = rewrite_english_only(tokenizer, model, rule_text, device=device, max_new_tokens=min(args.max_new_tokens, 384))
    else:
        if not args.allow_template_fallback:
            raise RuntimeError("No vector index available and --allow_template_fallback is not set.")
        rule_text = template_fallback_rule(is_minor, dims, adult_flag, adult_hard)

    if not rule_text.strip():
        if args.allow_template_fallback:
            rule_text = template_fallback_rule(is_minor, dims, adult_flag, adult_hard)
        else:
            raise RuntimeError("LLM returned empty rule text.")

    row = {
        "cache_key": cache_key,
        "cache_hit": False,
        "cache_strategy": args.cache_strategy,
        "audience": "minor" if is_minor else "adult",
        "user_id_example": uid,
        "item_id_example": iid,
        "triggered_dims": dims,
        "adult_flag": bool(adult_flag),
        "adult_hard": bool(adult_hard),
        "minor_block_at": args.minor_block_at,
        "adult_block_at": args.adult_block_at,
        "isadult_policy": args.isadult_policy,
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
        print("Triggered dims:", dims, "adult_flag:", adult_flag, "adult_hard:", adult_hard)
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
    b.add_argument("--out_dir", default="outputs/psg/rag_vector")
    b.add_argument("--model_path", required=True)
    b.add_argument("--chunk_size", type=int, default=650)
    b.add_argument("--overlap", type=int, default=80)
    b.add_argument("--batch_size", type=int, default=2)
    b.add_argument("--embed_max_length", type=int, default=768)
    b.add_argument("--device", default="cuda")
    b.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")

    q = sub.add_parser("query")
    q.add_argument("--vector_dir", default="outputs/psg/rag_vector")
    q.add_argument("--model_path", required=True)
    q.add_argument("--query", required=True)
    q.add_argument("--triggered_dims", default="")
    q.add_argument("--is_minor", action="store_true")
    q.add_argument("--adult_hard", action="store_true")
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
    g.add_argument("--raw_movie_id", default=None)
    g.add_argument("--vector_dir", default="outputs/psg/rag_vector")
    g.add_argument("--model_path", required=True)
    g.add_argument("--cache_path", default="")
    g.add_argument("--cache_strategy", choices=["signature", "pair", "none"], default="signature")
    g.add_argument("--no_cache", action="store_true")
    g.add_argument("--topk", type=int, default=5)
    g.add_argument("--topn", type=int, default=80)
    g.add_argument("--minor_block_at", type=float, default=3.0)
    g.add_argument("--adult_block_at", type=float, default=4.0)
    g.add_argument("--isadult_policy", choices=["minor_only", "all", "none"], default="minor_only")
    g.add_argument("--max_new_tokens", type=int, default=512)
    g.add_argument("--embed_max_length", type=int, default=768)
    g.add_argument("--device", default="cuda")
    g.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")
    g.add_argument("--json_only", action="store_true")
    g.add_argument("--allow_template_fallback", action="store_true")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
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
