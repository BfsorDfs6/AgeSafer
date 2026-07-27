# AgeSafer

This repository currently provides a **lightweight reference implementation of
AgeSafer for GMF on ML-1M**. It demonstrates the complete method pipeline:
age-aware profile/rule inputs, regulation-guided PSG construction and inference,
pseudo-positive injection, original/augmented GMF training, full-ranking safety
evaluation, and bounded multi-head reliability-aware fusion.

NeuMF, LightGCN, LLM-SRec, and the MyAnimeList experiments follow the same
pseudo-sample generation, injection, and fusion protocol, but their dedicated
training adapters and dataset preprocessing pipelines are not included in this
lightweight release.

## Scope and reproducibility statement

- Included: executable ML-1M + GMF reference pipeline.
- Not included: raw ML-1M data, IMDb-derived safety annotations, regulation PDFs,
  Qwen weights, a pre-trained PSG adapter, and third-party backbone repositories.
- The runner accepts all local paths as command-line parameters; no author-specific
  server paths or GPU IDs are embedded in the code.
- This release is intended to explain and reproduce the method mechanics. It is
  not yet a one-command reproduction package for every table in the paper.

## Repository structure

```text
AgeSafer/
├── run_ml1m_gmf.py                 # unified pipeline runner
├── scripts/
│   ├── run_ml1m_gmf.sh             # shell wrapper
│   └── publish_to_github.sh        # optional GitHub push helper
├── src/agesafer/
│   ├── psg_regulation_rag_full.py
│   ├── build_psg_sft_last5_aligned.py
│   ├── build_psg_candidate_inputs.py
│   ├── psg_lora_predict.py
│   ├── build_psg_augmented_train.py
│   ├── gmf_torch_valid_allranking.py
│   ├── eval_gmf_allranking_safety.py
│   └── train_eval_backbone_multihead_gate_fusion.py
├── data/README.md
├── regulations/README.md
├── requirements.txt
└── SERVER_AND_GITHUB.md
```

## Environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The regulation-index builder calls the system command `pdftotext`:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y poppler-utils
```

For optional PSG LoRA fine-tuning, install LLaMA-Factory separately and ensure
`llamafactory-cli` is available in `PATH`. A pre-trained adapter can instead be
supplied through `--psg-adapter`.

## Required inputs

See [`data/README.md`](data/README.md) for the expected files and columns.
At minimum, provide:

```text
<data-dir>/<dataset>.train.rating
<data-dir>/<dataset>.valid.rating
<data-dir>/<dataset>.test.rating
user_profiles_train_last5.jsonl
ml-1m_safe_features.train.safe.csv
ml-1m_safe_features.item_safe.csv
user_info.csv  # contains user id and is_minor
regulation_pdfs/*.pdf
Qwen2.5-7B-Instruct/
PSG LoRA adapter/  # unless --train-psg is enabled
```

## Quick start with an existing PSG adapter

```bash
bash scripts/run_ml1m_gmf.sh \
  --data-dir /path/to/Data \
  --dataset ml-1m_safe \
  --profiles /path/to/user_profiles_train_last5.jsonl \
  --train-safe /path/to/ml-1m_safe_features.train.safe.csv \
  --item-safe /path/to/ml-1m_safe_features.item_safe.csv \
  --user-info /path/to/user_info.csv \
  --regulation-pdf-dir /path/to/regulation_pdfs \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --psg-adapter /path/to/psg_lora_adapter \
  --output-dir outputs/ml1m_gmf_reference \
  --gpu 0
```

## Train the PSG adapter with LLaMA-Factory

```bash
bash scripts/run_ml1m_gmf.sh \
  --data-dir /path/to/Data \
  --dataset ml-1m_safe \
  --profiles /path/to/user_profiles_train_last5.jsonl \
  --train-safe /path/to/ml-1m_safe_features.train.safe.csv \
  --item-safe /path/to/ml-1m_safe_features.item_safe.csv \
  --user-info /path/to/user_info.csv \
  --regulation-pdf-dir /path/to/regulation_pdfs \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --train-psg \
  --llamafactory-cli llamafactory-cli \
  --output-dir outputs/ml1m_gmf_reference \
  --gpu 0
```

The runner generates an LLaMA-Factory dataset descriptor and LoRA configuration
under `<output-dir>/generated_configs/`.

## Run selected stages

The full stage order is:

```text
check, rag, sft_data, psg_train, candidates, psg_predict, augment,
train_base, train_aug, eval_base, eval_aug, fusion, summary
```

Example—resume from candidate construction:

```bash
python run_ml1m_gmf.py [required path arguments] \
  --stages candidates,psg_predict,augment,train_base,train_aug,eval_base,eval_aug,fusion,summary
```

Existing stage outputs are skipped by default. Use `--force` to regenerate them,
or `--dry-run` to inspect commands without executing them.

## Main configurable parameters

```text
--rho                 pseudo-sample injection ratio
--topn                candidates retained per user before PSG
--minor-block-at      ML-1M minor risk threshold
--adult-block-at      ML-1M adult risk threshold
--gmf-epochs          GMF training epochs
--valid-every         validation interval
--gate-max            upper bound of the fusion gate
--safety-weight       safety loss coefficient
```

Run `python run_ml1m_gmf.py --help` for all options.

## Outputs

```text
<output-dir>/
├── rag_vector/
├── sft/
├── psg_adapter/                 # when --train-psg is used
├── candidate_inputs/
├── predictions/
├── augmented_data/
├── checkpoints/
├── evaluation/
├── fusion/
├── logs/
└── run_summary.json
```

The final fused metrics are written to:

```text
<output-dir>/fusion/matrix_final_scorer_metrics.csv
```

## Other backbones and datasets

The framework-level outputs—the PSG predictions and pseudo-positive interaction
set—are independent of GMF. To extend this release:

1. train the target backbone on the original interaction matrix;
2. train the same backbone on the augmented interaction matrix;
3. expose batched user-item scores for both checkpoints;
4. feed these scores and the same safety/PSG cues into the fusion gate.

NeuMF and LightGCN can be integrated through their standard training/scoring
interfaces. LLM-SRec additionally requires its upstream repository and checkpoint
format. MyAnimeList requires its own preprocessing and R-17+/R+ safety mapping,
but uses the same PSG-selection and fusion stages.

## License and external assets

The code is released under the MIT License. Dataset licenses, model licenses,
regulation documents, and third-party repositories remain governed by their
respective terms.
