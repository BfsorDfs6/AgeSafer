# Commands

The stage launcher lists available adapters and forwards native CLI arguments.
Paths inside a stage are relative to `experiments/`; absolute input/output paths
also work. Each native script's `--help` documents its required files.

```bash
python experiments/run.py --list
python experiments/run.py ml1m-split -- --data_dir /path/to/ML1M --out_prefix ml-1m_safe --seed 42
python experiments/run.py ml1m-annotations -- --data_dir /path/to/Data --prefix ml-1m_safe --ml1m_dir /path/to/metadata --imdb_file /path/to/IMDB_parental_guide.csv --movie_details /path/to/movie_details_en.json --out_dir /path/to/features
python experiments/run.py mal-preprocess -- --raw_dir /path/to/MAL --dataset mal2000 --n_minor 200 --n_adult 1800 --seed 2028 --min_total_rated 50 --max_total_rated 0

# Example training; dataset splits are prepared once and retained.
python experiments/run.py train-neumf -- --model NeuMF --data_dir /path/to/Data --dataset ml-1m_safe --save_dir /path/to/checkpoints/neumf_seed42 --epochs 40 --valid_protocol allranking --valid_topks 1,5,10,20 --seed 42
python experiments/run.py train-lightgcn -- --data_dir /path/to/Data --dataset ml-1m_safe --save_dir /path/to/checkpoints/lightgcn_seed42 --epochs 40 --valid_topks 1,5,10,20 --seed 42
python experiments/run.py gate-ml1m -- --help
python experiments/run.py gate-mal -- --help
python experiments/run.py evaluate-mal -- --help
```

For PSG, use `rag-ml1m` or `rag-mal` with their `build` subcommand, then dataset
SFT, candidates, prediction and augmentation stages in that order. Register
generated Alpaca JSONL with the LLaMA-Factory example descriptor and run:

```bash
llamafactory-cli train configs/psg_ml1m.yaml
# or configs/psg_mal.yaml; edit local model, dataset and output paths first.
```

Train original and augmented GMF/NeuMF/LightGCN separately with `train-*` stages,
using the same retained original validation/test files and original observed-item
mask. Supply both checkpoints plus PSG JSONL and item/user features to `gate-*`.
Single-head ML-1M is `gate-ml1m-single`; three-head is `gate-ml1m` or `gate-mal`.
Always set validation monitoring and safety/gate parameters explicitly for a
table-matching run instead of assuming the script defaults are optimal.

LLM-SRec: obtain [the upstream code](https://github.com/Sein-Kim/LLM-SRec)
independently; use `prepare-llmsrec-ml1m`/
`prepare-llmsrec-mal` on separate original/augmented datasets, and
`patch-llmsrec-qwen` on your own upstream checkout. Pretrain SASRec and train
LLM-SRec following the upstream README; set `LLMSREC_LLM_MODEL_PATH` to local
Qwen2.5-7B-Instruct. Evaluate through `evaluate-llmsrec-* --save_scores` and
fuse via `gate-llmsrec` (see each stage's help). Upstream training and its
environment are not vendored or claimed to be newly validated here.

Use distinct checkpoint/output directories for each seed. To measure only
fusion-training randomness, hold inputs/branch checkpoints fixed; native gate
seed changes the internal calibration split as documented. To measure end-to-end
variability, repeat the relevant model training stages and state that scope.
