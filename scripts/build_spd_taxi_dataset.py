#!/usr/bin/env python3
"""Prepare published NYC Taxi SPD(10) data, using NumPy only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from riemannian_score_sde.taxi_data import prepare
import numpy as np

REVISION = 'a6a65d13d80369baa8f2dc8eac352a14ee4a919e'
FILES = ('train_data.csv', 'test_y.csv', 'test_spds.csv')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path)
    p.add_argument('--download', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'data/spd_taxi/nyc_taxi.npz')
    args = p.parse_args()
    if bool(args.source_dir) == args.download:
        p.error('Choose exactly one of --source-dir or --download')
    meta_path = args.output.with_suffix('.metadata.json')
    if args.output.exists() or meta_path.exists():
        p.error('Output exists; use a new output path')
    with tempfile.TemporaryDirectory(prefix='spd_taxi_') as tmp:
        source = args.source_dir or Path(tmp)
        if args.download:
            for name in FILES:
                url = f'https://raw.githubusercontent.com/li-yun-chen/SPD-DDPM/{REVISION}/data/condition/{name}'
                with urllib.request.urlopen(url, timeout=120) as response:
                    (source/name).write_bytes(response.read())
        result = prepare(source)
        meta = dict(dataset='NYC Taxi / SPD-DDPM conditional release', dimension=10,
                    context_dimension=13, upstream_revision=REVISION,
                    source_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in FILES},
                    split='Published test retained; RandomState(0), 15% of published train reserved for validation',
                    counts={s:len(result[s+'_indices']) for s in ('train','val','test')},
                    preprocessing='Published values preserved; no scaling, filtering or SPD repair',
                    protocol='Controlled comparison on published data; not exact Ko-Lee reproduction')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as f:
            np.savez_compressed(f, **result, metadata_json=json.dumps(meta))
        meta['npz_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        meta_path.write_text(json.dumps(meta, indent=2)+'\n')
        print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
