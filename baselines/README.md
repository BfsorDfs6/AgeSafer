# Run the transferred baselines

Install the main requirements and `pyyaml`. Supply local Qwen2.5-7B-Instruct weights and LLaMA-Factory for LLM4IDRec. CRC risk training imports the upstream `RankerNN` from a local checkout of `geektoni/mitigating-harm-recsys`.

## Inputs

Copy `config.example.json` and set the input paths. Paths may be absolute or relative to that JSON file. Rating files use the existing numeric inner IDs; score arrays are float32 `[num_users, num_items]` in the same ID namespace. All configured backbones must have the same dimensions. For a backbone with a different catalogue, use a separate configuration and its matching item metadata. Score arrays must cover the full catalogue before the observed-item mask is applied.

For MAL, set `dataset` to `MAL` and provide its `item_safe` and `user_info` files. For LLM4IDRec preparation without existing score arrays, set `scores` to `{}` and supply `num_users` / `num_items`. Oak item text is a JSON object mapping each numeric inner item ID to a string containing the title and genres only.

Optionally export the native age-risk flags once:

```bash
python baselines/common.py --config /path/to/config.json --export-flags /path/to/risk_flags.npz
```

Then add `risk_flags` to the configuration. This NPZ contains `minor[num_users]`, binary `flags[2,num_items,num_dimensions]` (adult row 0, minor row 1), and dimension `names`. This option also accepts flags exported by an explicitly chosen legacy reader, so the reader/version can be kept consistent with the comparison.

## CRC

```bash
python baselines/crc.py --config /path/to/config.json --phase train-risk \
  --crc-root /path/to/mitigating-harm-recsys --device cuda:0 \
  --seed 42 --epochs 20 --output outputs/crc_risk_seed42
python baselines/crc.py --config /path/to/config.json --phase evaluate \
  --backbone NeuMF --safety-scores outputs/crc_risk_seed42/learned_safety.npy \
  --method remove --seed 42 --output outputs/crc_neumf_seed42
```

Use `--method history-union` for the separately evaluated historical-item variant. Risk predictor: upstream 32-dimensional ID embeddings, one age-group covariate, training-only safe labels, Adam lr 0.001, batch 4096, 20 epochs. Calibration: validation users, K=20, 100 thresholds plus the empty-output boundary, monotone risk envelope and finite-sample correction. The default budget fractions are `1,.75,.5,.25,.1,0`; `.5` is the primary comparison. No safety-sorted filling is applied; short-output lengths and actual-slot exposure are saved alongside fixed-K exposure. The transferred temporal setting is evaluated empirically and does not establish the original CRC guarantee.

## Oak-PE

```bash
python baselines/oak.py --config /path/to/config.json --phase prepare \
  --backbones GMF,NeuMF,LightGCN,LLM-SRec --pool 50 --output outputs/oak
CUDA_VISIBLE_DEVICES=2 python baselines/oak.py --config /path/to/config.json \
  --phase infer --model-path /path/to/Qwen2.5-7B-Instruct \
  --device cuda:0 --batch 16 --output outputs/oak
python baselines/oak.py --config /path/to/config.json --phase evaluate --output outputs/oak
```

Greedy complete A/B/NONE judgments, both item orders, ascending harmful-vote counts, original backbone order for ties. Prompts contain title/genres, age cohort and policy text; they do not receive the item's true risk label. Top50 is reranked and its Top20 prefix is evaluated at 1/5/10/20. This public runner uses complete prompts; it omits the historical prefix-KV acceleration while preserving the same prompt and vote logic. Frozen-backbone greedy inference is deterministic; it does not establish repeated-training variability.

## LLM4IDRec + age-safety filtering

```bash
python baselines/llm4idrec.py --config /path/to/config.json --action prepare \
  --model-path /path/to/Qwen2.5-7B-Instruct --seed 42 --output outputs/llm4idrec
python baselines/llm4idrec.py --config /path/to/config.json --action train \
  --llamafactory-cli /path/to/llamafactory-cli --output outputs/llm4idrec
CUDA_VISIBLE_DEVICES=2 python baselines/llm4idrec.py --config /path/to/config.json \
  --action predict --model-path /path/to/Qwen2.5-7B-Instruct \
  --llamafactory-root /path/to/LLaMA-Factory --seed 42 --output outputs/llm4idrec
python baselines/llm4idrec.py --config /path/to/config.json --action augment \
  --seed 42 --output outputs/llm4idrec
```

ID-only generator, 20 shuffled training-history splits per user, Qwen template, LoRA rank8/alpha32/dropout0.05 on q_proj/v_proj, 400 steps, lr0.001, accumulation32, cutoff2048, train_on_prompt=true. One sampled response per user: temperature0.8, top_p0.9, top_k50, max200 new tokens. Parse IDs, remove invalid/duplicate/observed/validation/test items, apply the native age-risk rule, and cap injection at rho0.5. No filler. The original LLM4IDRec is not a safety method; age-safety filtering is this transfer's added step.

`AUGMENTATION.json` gives the generated dataset and data directory. Train the selected original backbone on that augmented dataset using its existing adapter/configuration, and evaluate that single branch without a gate or output safety filtering:

```bash
python experiments/run.py train-neumf -- --help
python experiments/run.py train-gmf -- --help
python experiments/run.py train-lightgcn -- --help
python experiments/run.py prepare-llmsrec-ml1m -- --help
python experiments/run.py prepare-llmsrec-mal -- --help
```

Use the original dataset's training/validation exclusions during final evaluation; pseudo-interactions remain eligible. Metrics are HR/NDCG and age-risk exposure @1,5,10,20 for all/minor/adult. Repeated runs use separate output paths and retain the same dataset split. Choose seeds matching the corresponding comparison; the reference seed list is 42,43,44,2026,2027.

```bash
python baselines/evaluate_scores.py --config /path/to/config.json \
  --scores /path/to/augmented_backbone_scores.npy --backbone NeuMF --seed 42 \
  --output outputs/llm4idrec_neumf_metrics.json
```

## Method references

- De Toni et al., *You Don't Bring Me Flowers: Mitigating Unwanted Recommendations Through Conformal Risk Control*, RecSys 2025. https://arxiv.org/abs/2507.16829
- Oak et al., *Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms*, ACL 2025. https://aclanthology.org/2025.acl-long.44/
- Chen et al., *LLM4IDRec* (ID-based recommendation data augmentation), ACM TOIS 2025. https://doi.org/10.1145/3704263 ; https://github.com/newlei/LLM4IDRec
