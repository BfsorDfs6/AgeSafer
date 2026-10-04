"""Launch one released stage without changing the original experiment checkout."""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    'ml1m-split': 'scripts1/build_ml1m_safe_split.py',
    'ml1m-annotations': 'scripts1/build_safe_features_and_splits.py',
    'ml1m-profiles': 'scripts1/generate_user_profiles_qwen.py',
    'check-splits': 'scripts1/check_split_leakage.py',
    'rag-ml1m': 'scripts/psg_regulation_rag_full.py',
    'sft-ml1m': 'scripts/build_psg_sft_last5_aligned.py',
    'candidates-ml1m': 'scripts/ffff/build_psg_candidate_inputs.py',
    'psg-ml1m': 'scripts/psg_lora_predict.py',
    'augment-ml1m': 'scripts/build_psg_augmented_train.py',
    'train-gmf': 'scripts/gmf_torch_valid_allranking.py',
    'train-neumf': 'scripts/ncf_torch_valid_allranking.py',
    'train-lightgcn': 'scripts/lightgcn_torch_valid_allranking.py',
    'gate-ml1m': 'scripts/train_eval_backbone_multihead_gate_fusion.py',
    'gate-ml1m-single': 'scripts/train_eval_backbone_gated_residual_fusion.py',
    'evaluate-ml1m-gmf': 'scripts/eval_gmf_allranking_safety.py',
    'evaluate-ml1m-lightgcn': 'scripts/eval_lightgcn_allranking_safety_dim5.py',
    'mal-preprocess': 'scripts/mal2000_risk3/00_build_mal_risk3_dataset_random_m200a1800.py',
    'mal-history': 'scripts/mal2000_risk3/01_build_profile_history.py',
    'mal-profiles': 'scripts/mal2000_risk3/02_generate_user_profiles_qwen.py',
    'rag-mal': 'scripts/mal2000_risk3/03_psg_regulation_rag_full.py',
    'sft-mal': 'scripts/mal2000_risk3/04_build_mal_risk3_sft_last5_aligned.py',
    'candidates-mal': 'scripts/mal2000_risk3/05b_build_mal_risk3_candidate_inputs_exclude_observed.py',
    'psg-mal': 'scripts/mal2000_risk3/06_mal_risk3_lora_predict.py',
    'augment-mal': 'scripts/mal2000_risk3/07_build_mal_risk3_augmented_train.py',
    'evaluate-mal': 'scripts/mal2000_risk3/09_eval_mal_risk3_allranking.py',
    'gate-mal': 'scripts/mal2000_risk3/10_train_eval_mal_risk3_multihead_gate_fusion.py',
    'prepare-llmsrec-ml1m': 'scripts/llmsrec_ml1m/prepare_llmsrec_ml1m.py',
    'prepare-llmsrec-mal': 'scripts/llmsrec_mal/prepare_llmsrec_mal.py',
    'patch-llmsrec-qwen': 'scripts/llmsrec_ml1m/patch_llmsrec_qwen25.py',
    'evaluate-llmsrec-ml1m': 'scripts/llmsrec_ml1m/eval_llmsrec_fullranking.py',
    'evaluate-llmsrec-mal': 'scripts/llmsrec_mal/eval_llmsrec_mal_fullranking.py',
    'evaluate-llmsrec-cache': 'scripts/llmsrec_common/eval_llmsrec_cached_scores_age_safe.py',
    'gate-llmsrec': 'scripts/llmsrec_common/fuse_llmsrec_cached_scores_singlehead.py',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('stage', nargs='?', choices=sorted(STAGES))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.list:
        for name, path in STAGES.items(): print(f'{name}: {path}')
        return
    if args.stage is None: parser.error('choose a stage or --list')
    forwarded = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    path = ROOT / STAGES[args.stage]
    if not path.is_file(): raise FileNotFoundError(path)
    os.chdir(ROOT)
    sys.path[:0] = [str(path.parent), str(ROOT)]
    sys.argv = [str(path), *forwarded]
    runpy.run_path(str(path), run_name='__main__')

if __name__ == '__main__': main()
