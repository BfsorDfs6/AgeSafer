"""Paired user bootstrap for two per-user metric CSVs, not item-level sampling.

Inputs require user_id and the same numeric metric columns (e.g. NDCG@20).
Optionally provide group=minor/adult to also report subgroup intervals.
Numbers use input units; keep both CSVs on the same scale.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def read(path):
    result = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            uid = row['user_id']
            if uid in result: raise ValueError(f'duplicate user_id: {uid}')
            result[uid] = row
    return result

def paired_interval(delta, repeats=2000, seed=42):
    delta = np.asarray(delta, dtype=float)
    if not delta.size or not np.isfinite(delta).all(): raise ValueError('empty or nonfinite differences')
    rng = np.random.default_rng(seed)
    means = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(repeats)]
    low, high = np.quantile(means, [.025, .975])
    return {'users':len(delta), 'mean_difference':float(delta.mean()), 'ci95':[float(low),float(high)]}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--agesafer', required=True)
    p.add_argument('--metrics', default='HR@20,NDCG@20')
    p.add_argument('--repeats', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.repeats < 100: p.error('use at least 100 bootstrap repetitions')
    a,b = read(args.baseline),read(args.agesafer)
    if not a or a.keys()!=b.keys(): raise ValueError('require identical, nonempty evaluated user sets')
    for u in a:
        if a[u].get('group') != b[u].get('group'): raise ValueError(f'group mismatch: {u}')
    cohorts = {'all':sorted(a)}
    for group in ['minor','adult']:
        users=[u for u in sorted(a) if a[u].get('group')==group]
        if users: cohorts[group]=users
    results={group:{metric:paired_interval([float(b[u][metric])-float(a[u][metric]) for u in users],args.repeats,args.seed) for metric in args.metrics.split(',')} for group,users in cohorts.items()}
    out={'direction':'AgeSafer minus baseline', 'bootstrap_unit':'paired user', 'repeats':args.repeats, 'seed':args.seed, 'results':results}
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__ == '__main__': main()
