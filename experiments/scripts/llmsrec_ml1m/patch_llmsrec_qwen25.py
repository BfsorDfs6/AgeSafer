#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path


def ensure_import_os_top(text: str) -> str:
    lines = text.splitlines()
    lines = [line for line in lines if line.strip() != "import os"]

    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    if len(lines) > insert_at and "coding" in lines[insert_at]:
        insert_at += 1

    lines.insert(insert_at, "import os")
    return "\n".join(lines) + "\n"


def patch_seqllm4rec(path: Path):
    s = path.read_text(encoding="utf-8")
    s = ensure_import_os_top(s)

    if "LLMSREC_LLM_MODEL_PATH" not in s:
        lines = s.splitlines()
        insert_idx = None

        for idx, line in enumerate(lines):
            if line.strip().startswith("else:"):
                prev = "\n".join(lines[max(0, idx - 20):idx])
                nxt = "\n".join(lines[idx:idx + 8])
                if ("llm_model" in prev or "llama-3b" in prev or "model_id" in prev) and ("raise" in nxt or "ValueError" in nxt or "NotImplemented" in nxt):
                    insert_idx = idx
                    break

        if insert_idx is None:
            raise RuntimeError("Cannot find llm_model else branch in models/seqllm4rec.py")

        indent = lines[insert_idx][:len(lines[insert_idx]) - len(lines[insert_idx].lstrip())]
        block = [
            f"{indent}elif llm_model in ['qwen25', 'qwen2.5', 'qwen']:",
            f"{indent}    model_id = os.environ.get('LLMSREC_LLM_MODEL_PATH', 'models/Qwen2.5-7B-Instruct')",
        ]
        lines[insert_idx:insert_idx] = block
        s = "\n".join(lines) + "\n"

    # Qwen 本地模型需要 trust_remote_code；use_fast=False 更稳
    s = s.replace(
        "AutoTokenizer.from_pretrained(model_id)",
        "AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)"
    )

    path.write_text(s, encoding="utf-8")


def patch_main(path: Path):
    if not path.exists():
        return

    s = path.read_text(encoding="utf-8")
    s = s.replace("choices=['llama', 'llama-3b']", "choices=['llama', 'llama-3b', 'qwen25']")
    s = s.replace('choices=["llama", "llama-3b"]', 'choices=["llama", "llama-3b", "qwen25"]')
    path.write_text(s, encoding="utf-8")


def patch_optional_sbert(path: Path):
    if not path.exists():
        return

    s = path.read_text(encoding="utf-8")
    old = "from sentence_transformers import SentenceTransformer"
    new = "try:\n    from sentence_transformers import SentenceTransformer\nexcept Exception:\n    SentenceTransformer = None"

    if old in s and "SentenceTransformer = None" not in s:
        s = s.replace(old, new)
        path.write_text(s, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmsrec_root", required=True)
    args = ap.parse_args()

    root = Path(args.llmsrec_root)
    seqllm4rec = root / "models" / "seqllm4rec.py"
    seqllm_model = root / "models" / "seqllm_model.py"
    main_py = root / "main.py"

    patch_seqllm4rec(seqllm4rec)
    patch_main(main_py)
    patch_optional_sbert(seqllm_model)

    print(f"[OK] patched {seqllm4rec}")
    print(f"[OK] patched {seqllm_model}")
    print(f"[OK] patched {main_py}")


if __name__ == "__main__":
    main()
