# Reproduction details

## Regulations and prompts

Encoder: local Qwen2.5-7B-Instruct; attention-mask-weighted mean of the final
hidden layer, followed by L2 normalization (3584 dimensions for this model).
PDFs are extracted with `pdftotext`, cleaned and chunked in character windows
of 650 with overlap 80. Embedding token limit is 768. FAISS `IndexFlatIP` is
used when available; otherwise the implementation uses the same normalized
inner products in NumPy. Retrieval first takes up to 80 available chunks,
then applies the code's risk-tag/age-aware selection to retain 5. The audited
historical index had 43 chunks, so an initial budget of 80 did not mean 80
distinct retrieved chunks. No separate pretrained sentence encoder is used.

Exact regulation queries, reranking, rule-generation instructions and cache
keys are in `experiments/scripts/psg_regulation_rag_full.py` and the MAL `03_`
variant. Rule caching uses the implementation's safety signature; pair and
no-cache modes are also exposed. PSG SFT and inference prompt templates are
in `build_psg_sft_last5_aligned.py`, `ffff/build_psg_candidate_inputs.py` and
the MAL `04_`/`05b_` scripts. User-profile prompts are in the two datasets'
profile-generation scripts. These source templates are the exact instructions,
not abbreviated examples. Profiling uses only training histories (last 5).

Four source regulations and official download links are in
`regulations/README.md`. Record the PDF and chunk/index hashes for each run;
regulatory text is guidance, not a universally valid age-rating certificate.

## Labels, annotations and preprocessing

PSG positive target is preference AND appropriateness: if z=1 means
inappropriate, omega=xi*(1-z). MovieLens preference positives are ratings >=4,
negatives <=2; MAL positives >=7, negatives <=6. None/Mild/Moderate/Severe are
encoded as 1/2/3/4. ML-1M uses threshold 3 for minors and 4 for adults;
`isAdult` handling is explicit in the command options. MAL category parsing,
R17/RPLUS/RX flags and age rules are in its preprocessing/evaluation scripts;
do not silently merge RX into the reported RPLUS dimension.

IMDb matching is by IDs: inner item -> original MovieLens MovieID ->
`links.csv` IMDb ID -> zero-padded `tt` ID -> parental-guide `tconst`.
There is no fuzzy title matching in the released alignment stage. Ensure that
the MovieID crosswalk belongs to the same MovieLens ID namespace. Unknown
annotations have `risk_missing=True`; the legacy builder assigns numerical
default 1, which is a missing-data assumption, not an expert finding of safety.
Run `scripts/check_annotations.py` for crosswalk coverage and duplicates.

MAL uses age at registration, derived from birth and registration dates, with
minor defined as <18. The historical dataset sampled 200 minors and 1800 adults
using seed 2028; it used minimum 50 rated items and no maximum-history cap.
The preprocessing code removes username, birth/registration dates and location
from the new public export tables after age derivation; historical internal
tables retained these fields. This export change does not establish that the
historical workflow was anonymous or had no privacy risk. No user records or
preference histories are distributed in this repository.

## Training and evaluation

PSG uses LLaMA-Factory SFT LoRA: rank16, alpha32, dropout0.05, all target modules,
cutoff3072, lr1e-4, 2 epochs, batch1, accumulation8, bf16. Portable YAMLs are in
`configs/`. All model weights are supplied locally. LLM-SRec is a separate
upstream dependency, with Qwen/data/full-ranking adapters included here.

Evaluate full unobserved item ranking at 1/5/10/20, excluding observed training
items and validation positives for test evaluation. Report all/minor/adult user
counts; exposure uses sum of age-rule violations divided by users*K.
Checkpoint and hyperparameter selection use validation, not test performance.

Original and augmented branches are separately trained model instances. Both
branch scores are obtained for every full-ranking candidate, and the gate uses
their difference plus PSG and age-risk cues. PSG prediction/positive-probability/
margin records are loaded offline; pairs without a record use (0,0,0), without
forcing a zero gate or claiming that the item is safe. Regulation retrieval and
PSG generation are offline; the native scoring path executes both branches and
the gate. No negligible-latency claim is made for this release.

## Historical variants and seed scope

This is an expanded reference release, not evidence that regenerated outputs
equal every historical table. Original files on the experiment server remain
unchanged. `experiments/SOURCE_MANIFEST.json` records original and released hashes.

- The original lightweight reference has TF-IDF/MMR candidate generation.
  Later ML-1M table inputs used direct Top100 content/profile similarity plus
  training popularity (0.85/0.15), provided in `ffff/`; MAL has its own candidate
  builder. These are distinct variants.
- Table-matching ML-1M configurations used single-head fusion; MAL GMF/NeuMF/
  LightGCN used three heads, while MAL LLM-SRec used single-head fusion. Three-head
  code is available as a paper-structure reference, not relabelled historical evidence.
- Historical ML-1M risk CSV parsing differs for quoted text fields between some
  readers. Legacy experiment copies retain their original readers; reproducing
  a legacy number is not a correctness validation of that number. Use explicit
  reader/version provenance; do not mix corrected and legacy safety results.
- Historical fusion seeds are listed in `configs/repeated_runs.json`. New
  reference seeds 42/43/44/2026/2027 do not replace the seeds actually used.
- Historical AgeSafer repetitions froze PSG and both branches. The native gate
  seed also changes its internal 80/20 calibration/holdout split, initialization,
  dropout and sampling. It does not rebuild the original dataset test split.
- Repeated backbone training is a separate experiment; user-paired bootstrap
  intervals are another uncertainty estimate. `scripts/paired_user_ci.py`
  bootstraps matched per-user metrics without treating items as independent.
