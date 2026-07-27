#!/usr/bin/env bash
set -euo pipefail

# Pass all pipeline parameters to the Python runner.
# Example:
#   bash scripts/run_ml1m_gmf.sh \
#     --data-dir /path/to/Data \
#     --profiles /path/to/user_profiles_train_last5.jsonl \
#     --train-safe /path/to/ml-1m_safe_features.train.safe.csv \
#     --item-safe /path/to/ml-1m_safe_features.item_safe.csv \
#     --user-info /path/to/ml-1m_safe_features.user_info.csv \
#     --regulation-pdf-dir /path/to/regulation_pdfs \
#     --model-path /path/to/Qwen2.5-7B-Instruct \
#     --psg-adapter /path/to/psg_lora_adapter \
#     --gpu 0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python -u "${REPO_ROOT}/run_ml1m_gmf.py" "$@"
