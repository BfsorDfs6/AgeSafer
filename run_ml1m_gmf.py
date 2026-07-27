#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the lightweight AgeSafer reference pipeline for ML-1M + GMF.

The runner orchestrates:
  regulation index -> PSG SFT data -> optional LoRA SFT -> candidate construction
  -> PSG inference -> pseudo-sample injection -> base/augmented GMF training
  -> ranking/safety evaluation -> bounded multi-head fusion.

The repository assumes that ML-1M has already been converted into the input
files documented in data/README.md. Large models and raw datasets are not
redistributed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


ALL_STAGES = (
    "check",
    "rag",
    "sft_data",
    "psg_train",
    "candidates",
    "psg_predict",
    "augment",
    "train_base",
    "train_aug",
    "eval_base",
    "eval_aug",
    "fusion",
    "summary",
)


def parse_stages(value: str) -> list[str]:
    value = value.strip().lower()
    if value in {"all", "full"}:
        return list(ALL_STAGES)
    stages = [x.strip() for x in value.split(",") if x.strip()]
    unknown = sorted(set(stages) - set(ALL_STAGES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown stages: {', '.join(unknown)}")
    return stages


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def run_command(
    command: Sequence[str],
    *,
    log_path: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    command = [str(x) for x in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = shlex.join(command)
    print(f"\n[run] {printable}")
    print(f"[log] {log_path}")
    if dry_run:
        return
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {printable}")


def should_run(stage: str, selected: Iterable[str]) -> bool:
    return stage in set(selected)


def write_llamafactory_files(
    *,
    sft_jsonl: Path,
    model_path: Path,
    adapter_dir: Path,
    config_dir: Path,
    epochs: float,
    learning_rate: float,
    cutoff_len: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = {
        "agesafer_psg": {
            "file_name": str(sft_jsonl.resolve()),
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }
    dataset_info_path = config_dir / "dataset_info.json"
    dataset_info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    yaml_path = config_dir / "psg_lora_sft.yaml"
    yaml_text = f"""model_name_or_path: {model_path.resolve()}
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

dataset: agesafer_psg
dataset_dir: {config_dir.resolve()}
template: qwen
cutoff_len: {cutoff_len}
overwrite_cache: true
preprocessing_num_workers: 4

output_dir: {adapter_dir.resolve()}
overwrite_output_dir: true
logging_steps: 10
save_strategy: epoch
plot_loss: true
report_to: none

per_device_train_batch_size: {batch_size}
gradient_accumulation_steps: {gradient_accumulation_steps}
learning_rate: {learning_rate}
num_train_epochs: {epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the AgeSafer ML-1M + GMF reference pipeline.",
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing <dataset>.train/valid/test.rating files.")
    parser.add_argument("--dataset", default="ml-1m_safe")
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--train-safe", type=Path, required=True)
    parser.add_argument("--item-safe", type=Path, required=True)
    parser.add_argument("--user-info", type=Path, required=True,
                        help="CSV containing user id and is_minor; may also contain tolerance fields.")
    parser.add_argument("--regulation-pdf-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Local base LLM path, e.g. Qwen2.5-7B-Instruct.")
    parser.add_argument("--psg-adapter", type=Path, default=None,
                        help="Existing PSG LoRA adapter. If omitted, use --train-psg.")
    parser.add_argument("--train-psg", action="store_true",
                        help="Train the PSG adapter with LLaMA-Factory.")
    parser.add_argument("--llamafactory-cli", default="llamafactory-cli")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ml1m_gmf_reference"))
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")
    parser.add_argument("--stages", type=parse_stages, default=parse_stages("all"),
                        help="Comma-separated stages or 'all'.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Rerun stages even when their primary output exists.")

    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--topn", type=int, default=100)
    parser.add_argument("--recall-topn", type=int, default=600)
    parser.add_argument("--pref-quota", type=int, default=60)
    parser.add_argument("--pop-quota", type=int, default=20)
    parser.add_argument("--div-quota", type=int, default=20)
    parser.add_argument("--rho", type=float, default=0.20)
    parser.add_argument("--min-p1", type=float, default=0.0)
    parser.add_argument("--min-margin", type=float, default=-999.0)
    parser.add_argument("--minor-block-at", type=float, default=3.0)
    parser.add_argument("--adult-block-at", type=float, default=4.0)
    parser.add_argument("--rag-topk", type=int, default=5)
    parser.add_argument("--rag-topn", type=int, default=80)

    parser.add_argument("--gmf-epochs", type=int, default=20)
    parser.add_argument("--gmf-batch-size", type=int, default=256)
    parser.add_argument("--gmf-eval-batch-size", type=int, default=32768)
    parser.add_argument("--gmf-factors", type=int, default=8)
    parser.add_argument("--gmf-num-neg", type=int, default=4)
    parser.add_argument("--gmf-lr", type=float, default=0.001)
    parser.add_argument("--valid-every", type=int, default=5)
    parser.add_argument("--topks", default="1,5,10,20")

    parser.add_argument("--gate-max", type=float, default=0.10)
    parser.add_argument("--gate-epochs", type=int, default=40)
    parser.add_argument("--gate-hidden", type=int, default=32)
    parser.add_argument("--gate-lr", type=float, default=5e-4)
    parser.add_argument("--safety-weight", type=float, default=0.02)
    parser.add_argument("--safety-margin", type=float, default=0.05)

    parser.add_argument("--sft-epochs", type=float, default=3.0)
    parser.add_argument("--sft-learning-rate", type=float, default=2e-4)
    parser.add_argument("--sft-batch-size", type=int, default=1)
    parser.add_argument("--sft-gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--psg-batch-size", type=int, default=2)
    parser.add_argument("--max-users", type=int, default=-1,
                        help="Debug limit for SFT/candidate construction; -1 means all users.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent
    src = repo_root / "src" / "agesafer"
    out = args.output_dir.resolve()
    logs = out / "logs"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    train_rating = args.data_dir / f"{args.dataset}.train.rating"
    valid_rating = args.data_dir / f"{args.dataset}.valid.rating"
    test_rating = args.data_dir / f"{args.dataset}.test.rating"
    valid_negative = args.data_dir / f"{args.dataset}.valid.negative"
    test_negative = args.data_dir / f"{args.dataset}.test.negative"

    rag_dir = out / "rag_vector"
    sft_dir = out / "sft"
    sft_jsonl = sft_dir / "psg_sft.jsonl"
    sft_summary = sft_dir / "summary.json"
    rag_cache_sft = rag_dir / "rule_cache_sft.jsonl"
    rag_cache_candidates = rag_dir / "rule_cache_candidates.jsonl"
    adapter_dir = (args.psg_adapter.resolve() if args.psg_adapter else out / "psg_adapter")

    candidate_dir = out / "candidate_inputs"
    candidate_input = candidate_dir / "psg_infer_inputs.jsonl"
    candidates_jsonl = candidate_dir / "psg_candidates.jsonl"
    candidate_summary = candidate_dir / "summary.json"
    prediction_dir = out / "predictions"
    prediction_jsonl = prediction_dir / "psg_predictions.jsonl"
    prediction_summary = prediction_dir / "summary.json"

    rho_tag = f"rho{int(round(args.rho * 100)):03d}"
    aug_dataset = f"{args.dataset}_agesafer_{rho_tag}"
    aug_data_dir = out / "augmented_data" / aug_dataset
    base_ckpt_dir = out / "checkpoints" / "base_gmf"
    aug_ckpt_dir = out / "checkpoints" / "augmented_gmf"
    base_checkpoint = base_ckpt_dir / f"{args.dataset}_GMF_{args.gmf_factors}_torch_best.pt"
    aug_checkpoint = aug_ckpt_dir / f"{aug_dataset}_GMF_{args.gmf_factors}_torch_best.pt"
    eval_dir = out / "evaluation"
    fusion_dir = out / "fusion"

    if should_run("check", args.stages):
        require_dir(args.data_dir, "data directory")
        require_file(train_rating, "training interactions")
        require_file(valid_rating, "validation interactions")
        require_file(test_rating, "test interactions")
        require_file(args.profiles, "profiles")
        require_file(args.train_safe, "training safety features")
        require_file(args.item_safe, "item safety features")
        require_file(args.user_info, "user information")
        require_dir(args.regulation_pdf_dir, "regulation PDF directory")
        require_dir(args.model_path, "base LLM")
        for script in (
            "psg_regulation_rag_full.py",
            "build_psg_sft_last5_aligned.py",
            "build_psg_candidate_inputs.py",
            "psg_lora_predict.py",
            "build_psg_augmented_train.py",
            "gmf_torch_valid_allranking.py",
            "eval_gmf_allranking_safety.py",
            "train_eval_backbone_multihead_gate_fusion.py",
        ):
            require_file(src / script, script)
        if not args.train_psg and args.psg_adapter is None:
            raise ValueError("Provide --psg-adapter or enable --train-psg.")
        if args.psg_adapter is not None and not args.train_psg:
            require_dir(args.psg_adapter, "PSG LoRA adapter")
        print("[check] All required inputs are available.")

    def execute(stage: str, primary_output: Path, command: Sequence[str]) -> None:
        if not should_run(stage, args.stages):
            return
        if primary_output.exists() and not args.force:
            print(f"[skip:{stage}] exists: {primary_output}")
            return
        run_command(command, log_path=logs / f"{stage}.log", env=env, dry_run=args.dry_run)

    execute(
        "rag",
        rag_dir / "vector_index.json",
        [
            sys.executable, src / "psg_regulation_rag_full.py", "build",
            "--pdf_dir", args.regulation_pdf_dir,
            "--out_dir", rag_dir,
            "--model_path", args.model_path,
            "--device", args.device,
            "--dtype", args.dtype,
        ],
    )

    sft_command = [
        sys.executable, src / "build_psg_sft_last5_aligned.py",
        "--profiles", args.profiles,
        "--train_safe", args.train_safe,
        "--item_safe", args.item_safe,
        "--output", sft_jsonl,
        "--summary_json", sft_summary,
        "--pos_rating", "4",
        "--neg_rating", "2",
        "--history_window", str(args.history_window),
        "--candidate_order", "recent",
        "--minor_pos_target", "-1",
        "--minor_neg_target", "-1",
        "--adult_pos_target", "20000",
        "--adult_neg_target", "20000",
        "--neg_mix", "0.4,0.4,0.2",
        "--minor_block_at", str(args.minor_block_at),
        "--adult_block_at", str(args.adult_block_at),
        "--isadult_policy", "minor_only",
        "--rule_mode", "rag_cli",
        "--rag_script", src / "psg_regulation_rag_full.py",
        "--vector_dir", rag_dir,
        "--model_path", args.model_path,
        "--rag_cache_path", rag_cache_sft,
        "--rule_cache_strategy", "signature",
        "--rag_device", args.device,
        "--rag_dtype", args.dtype,
        "--rag_after_sampling",
        "--strict_no_leakage",
        "--require_rag_success",
    ]
    if args.max_users > 0:
        sft_command += ["--max_users", str(args.max_users)]
    execute("sft_data", sft_jsonl, sft_command)

    if should_run("psg_train", args.stages):
        if args.train_psg:
            yaml_path = write_llamafactory_files(
                sft_jsonl=sft_jsonl,
                model_path=args.model_path,
                adapter_dir=adapter_dir,
                config_dir=out / "generated_configs",
                epochs=args.sft_epochs,
                learning_rate=args.sft_learning_rate,
                cutoff_len=args.max_length,
                batch_size=args.sft_batch_size,
                gradient_accumulation_steps=args.sft_gradient_accumulation,
            )
            execute(
                "psg_train",
                adapter_dir / "adapter_config.json",
                [args.llamafactory_cli, "train", yaml_path],
            )
        else:
            print(f"[psg_train] Using existing adapter: {adapter_dir}")

    candidate_command = [
        sys.executable, src / "build_psg_candidate_inputs.py",
        "--profiles", args.profiles,
        "--train_safe", args.train_safe,
        "--item_safe", args.item_safe,
        "--user_tol", args.user_info,
        "--valid_interactions", valid_rating,
        "--test_interactions", test_rating,
        "--eval_negative_policy", "ignore",
        "--output", candidate_input,
        "--candidates_output", candidates_jsonl,
        "--summary_json", candidate_summary,
        "--topn", str(args.topn),
        "--recall_topn", str(args.recall_topn),
        "--pref_quota", str(args.pref_quota),
        "--pop_quota", str(args.pop_quota),
        "--div_quota", str(args.div_quota),
        "--history_window", str(args.history_window),
        "--isadult_policy", "minor_only",
        "--minor_cap", str(args.minor_block_at),
        "--minor_exceed_count", "2",
        "--adult_exceed_count", "3",
        "--rag_script", src / "psg_regulation_rag_full.py",
        "--vector_dir", rag_dir,
        "--model_path", args.model_path,
        "--rag_topk", str(args.rag_topk),
        "--rag_topn", str(args.rag_topn),
        "--rag_cache_path", rag_cache_candidates,
        "--rag_device", args.device,
        "--rule_cache_strategy", "signature",
        "--strict_no_leakage",
    ]
    if valid_negative.is_file():
        candidate_command += ["--valid_negative", valid_negative]
    if test_negative.is_file():
        candidate_command += ["--test_negative", test_negative]
    if args.max_users > 0:
        candidate_command += ["--max_users", str(args.max_users)]
    execute("candidates", candidate_input, candidate_command)

    execute(
        "psg_predict",
        prediction_jsonl,
        [
            sys.executable, src / "psg_lora_predict.py",
            "--model_path", args.model_path,
            "--adapter_path", adapter_dir,
            "--input", candidate_input,
            "--output", prediction_jsonl,
            "--summary_json", prediction_summary,
            "--batch_size", str(args.psg_batch_size),
            "--max_length", str(args.max_length),
            "--mode", "logits",
            "--device", args.device,
            "--dtype", args.dtype,
        ],
    )

    aug_train_rating = aug_data_dir / f"{aug_dataset}.train.rating"
    execute(
        "augment",
        aug_train_rating,
        [
            sys.executable, src / "build_psg_augmented_train.py",
            "--base_data_dir", args.data_dir,
            "--base_dataset", args.dataset,
            "--pred_jsonl", prediction_jsonl,
            "--out_dir", aug_data_dir,
            "--out_dataset", aug_dataset,
            "--rho", str(args.rho),
            "--min_p1", str(args.min_p1),
            "--min_margin", str(args.min_margin),
        ],
    )

    common_train = [
        "--epochs", str(args.gmf_epochs),
        "--batch_size", str(args.gmf_batch_size),
        "--eval_batch_size", str(args.gmf_eval_batch_size),
        "--num_factors", str(args.gmf_factors),
        "--num_neg", str(args.gmf_num_neg),
        "--lr", str(args.gmf_lr),
        "--valid_protocol", "allranking",
        "--valid_topks", args.topks,
        "--topk", "10",
        "--monitor", "ndcg",
        "--valid_every", str(args.valid_every),
    ]
    execute(
        "train_base",
        base_checkpoint,
        [
            sys.executable, src / "gmf_torch_valid_allranking.py",
            "--data_dir", args.data_dir,
            "--dataset", args.dataset,
            "--eval_block_train_rating", train_rating,
            "--save_dir", base_ckpt_dir,
            *common_train,
        ],
    )
    execute(
        "train_aug",
        aug_checkpoint,
        [
            sys.executable, src / "gmf_torch_valid_allranking.py",
            "--data_dir", aug_data_dir,
            "--dataset", aug_dataset,
            "--eval_block_train_rating", train_rating,
            "--save_dir", aug_ckpt_dir,
            *common_train,
        ],
    )

    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_common = [
        "--split", "test",
        "--train_rating", train_rating,
        "--valid_rating", valid_rating,
        "--item_safe", args.item_safe,
        "--user_info", args.user_info,
        "--topks", args.topks,
        "--exclude_valid_for_test",
        "--isadult_policy", "minor_only",
        "--minor_block_at", str(args.minor_block_at),
        "--adult_block_at", str(args.adult_block_at),
    ]
    execute(
        "eval_base",
        eval_dir / "base_metrics.csv",
        [
            sys.executable, src / "eval_gmf_allranking_safety.py",
            "--checkpoint", base_checkpoint,
            "--data_dir", args.data_dir,
            "--dataset", args.dataset,
            "--output_csv", eval_dir / "base_metrics.csv",
            "--output_json", eval_dir / "base_metrics.json",
            *eval_common,
        ],
    )
    execute(
        "eval_aug",
        eval_dir / "augmented_metrics.csv",
        [
            sys.executable, src / "eval_gmf_allranking_safety.py",
            "--checkpoint", aug_checkpoint,
            "--data_dir", aug_data_dir,
            "--dataset", aug_dataset,
            "--output_csv", eval_dir / "augmented_metrics.csv",
            "--output_json", eval_dir / "augmented_metrics.json",
            *eval_common,
        ],
    )

    execute(
        "fusion",
        fusion_dir / "matrix_final_scorer_metrics.csv",
        [
            sys.executable, src / "train_eval_backbone_multihead_gate_fusion.py",
            "--backbone", "gmf",
            "--base_checkpoint", base_checkpoint,
            "--psg_checkpoint", aug_checkpoint,
            "--pred_jsonl", prediction_jsonl,
            "--train_rating", train_rating,
            "--valid_rating", valid_rating,
            "--test_rating", test_rating,
            "--item_safe", args.item_safe,
            "--user_tol", args.user_info,
            "--output_dir", fusion_dir,
            "--topks", args.topks,
            "--minor_block_at", str(args.minor_block_at),
            "--adult_block_at", str(args.adult_block_at),
            "--isadult_policy", "minor_only",
            "--exclude_valid_for_test",
            "--hidden", str(args.gate_hidden),
            "--gate_max", str(args.gate_max),
            "--epochs", str(args.gate_epochs),
            "--lr", str(args.gate_lr),
            "--safety_weight", str(args.safety_weight),
            "--safety_margin", str(args.safety_margin),
            "--valid_every", str(args.valid_every),
            "--device", "auto",
        ],
    )

    if should_run("summary", args.stages):
        summary = {
            "implementation_scope": "Lightweight reference implementation for ML-1M + GMF",
            "dataset": args.dataset,
            "rho": args.rho,
            "base_checkpoint": str(base_checkpoint),
            "augmented_checkpoint": str(aug_checkpoint),
            "psg_predictions": str(prediction_jsonl),
            "base_metrics": str(eval_dir / "base_metrics.csv"),
            "augmented_metrics": str(eval_dir / "augmented_metrics.csv"),
            "fusion_metrics": str(fusion_dir / "matrix_final_scorer_metrics.csv"),
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n[done] Outputs: {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[stopped] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
