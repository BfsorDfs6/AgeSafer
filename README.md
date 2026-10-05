# AgeSafer

## Environment

```bash
pip install -r requirements.txt
# Install Poppler (pdftotext); install LLaMA-Factory for PSG fine-tuning.
```

## Run ML-1M + GMF

```bash
python run_ml1m_gmf.py \
  --data-dir /path/to/Data --dataset ml-1m_safe \
  --profiles /path/to/user_profiles_train_last5.jsonl \
  --train-safe /path/to/train.safe.csv \
  --item-safe /path/to/item_safe.csv \
  --user-info /path/to/user_info.csv \
  --regulation-pdf-dir /path/to/regulation_pdfs \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --psg-adapter /path/to/psg_adapter \
  --output-dir outputs/ml1m_gmf_seed42 --seed 42 --gpu 0
```

To train PSG, replace `--psg-adapter` with `--train-psg`.
Use `--stages` to select stages and `--dry-run` to print commands.

## Other backbones and MAL

```bash
python experiments/run.py --list
python experiments/run.py train-neumf -- --help
python experiments/run.py train-lightgcn -- --help
python experiments/run.py mal-preprocess -- --help
python experiments/run.py gate-ml1m -- --help
python experiments/run.py gate-mal -- --help
python experiments/run.py prepare-llmsrec-ml1m -- --help
python experiments/run.py prepare-llmsrec-mal -- --help
```

Stage arguments are forwarded after `--`. Use absolute input/output paths.
LLM-SRec adapters take `--llmsrec_root` pointing to a local upstream checkout.
More stage commands: [docs/RUNNING.md](docs/RUNNING.md).
Input formats: [data/README.md](data/README.md).

## Configuration

| Parameter | Setting |
|---|---|
| LLM / regulation encoder | Qwen2.5-7B-Instruct; final hidden-layer mean pooling + L2 normalization |
| Regulation chunks | 650 characters; overlap 80 |
| Retrieval | Initial budget 80; final Top5 |
| PSG LoRA | Rank 16; alpha 32; dropout 0.05; all target modules |
| PSG SFT | 2 epochs; lr 1e-4; cutoff 3072; batch 1; accumulation 8; bf16 |
| History window | Last 5 training interactions |
| ML-1M thresholds | Minor: Level >=3; adult: Level >=4 |
| Evaluation | Full ranking; HR/NDCG/exposure @1,5,10,20 |
| Training seeds | 42, 43, 44, 2026, 2027 |

Keep dataset splits fixed and use a separate output directory for each seed.
Set `--rho`, `--gate-max`, `--safety-weight` and validation monitoring for each run.
PSG YAMLs: `configs/psg_ml1m.yaml`, `configs/psg_mal.yaml`.
Dataset registration example: `configs/dataset_info.example.json`.
Retrieval, prompts and annotation alignment: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## CRC, LLM-ReRanking and LLM4IDRec + age-safety filtering

Running commands and input configurations: [baselines/README.md](baselines/README.md).

```bash
python baselines/crc.py --help
python baselines/oak.py --help
python baselines/llm4idrec.py --help
```
