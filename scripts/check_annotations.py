"""Audit an ID-based MovieLens/IMDb crosswalk without reading user profiles."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

def rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def tconst(value):
    value = str(value).strip()
    if not value: return ''
    return value if value.startswith('tt') else 'tt' + value.zfill(7)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--links', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    links, annotations = rows(args.links), rows(args.annotations)
    ids = Counter(r['movieId'] for r in links)
    imdb = Counter(tconst(r['tconst']) for r in annotations if r.get('tconst'))
    matched = sum(tconst(r.get('imdbId', '')) in imdb for r in links)
    out = {'crosswalk_items': len(links), 'annotation_rows': len(annotations),
           'matched_items': matched, 'unmatched_items': len(links)-matched,
           'matched_fraction': matched/len(links) if links else None,
           'duplicate_movie_ids': [k for k,v in ids.items() if v>1],
           'duplicate_tconst': [k for k,v in imdb.items() if v>1],
           'matching': 'exact MovieID -> IMDb ID -> tconst; no title fuzzy matching'}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
    if out['duplicate_movie_ids'] or out['duplicate_tconst']: raise SystemExit(2)

if __name__ == '__main__': main()
