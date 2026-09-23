import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import so3_complete_comparison as mod

class ComparisonTests(unittest.TestCase):
    def test_training_source_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            for v in (0,1):
                p=run/f'logs/version_{v}/hparams.yaml';p.parent.mkdir(parents=True);p.write_text(str(v))
            configs={'0':dict(mode='train',seed=1,steps=100000),
                     '1':dict(mode='test',seed=0,steps=100000)}
            _,cfg=mod.select_training_config(run,lambda p:configs[p.read_text()])
            self.assertEqual(cfg['seed'],1)
            configs['1']=dict(mode='train',seed=2,steps=100000)
            with self.assertRaises(ValueError):mod.select_training_config(run,lambda p:configs[p.read_text()])

    def test_conditions(self):
        cs=mod.conditions()
        self.assertEqual(len(cs),18)
        self.assertEqual(len(set(cs)),18)
        self.assertEqual(len([c for c in cs if c[0]!='malliavin' and c[1]==5]),6)

    def test_summarize_requires_all_runs_and_uses_seed_sd(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            rs=[dict(name=f'{m}_{w}_{s}',method=m,weight=w,training_seed=s)
                for m,w,s in mod.conditions()]
            (out/'manifest.json').write_text(json.dumps({'runs':rs}))
            with self.assertRaises(ValueError):mod.summarize(out)
            for r in rs:
                d=out/'evaluation'/r['name'];(d/'evaluation').mkdir(parents=True)
                v=r['training_seed']+1.
                report={'rbf_mmd':v,'nearest_neighbor_geodesic_distance':dict(mean=v,median=v,max=v),
                        'reference_to_generated_nearest_neighbor':dict(mean=v,median=v,max=v)}
                (d/'evaluation/metrics.json').write_text(json.dumps(report));(d/'COMPLETE').write_text('OK')
            from unittest.mock import patch, MagicMock
            mpl=MagicMock();plt=MagicMock()
            axes=MagicMock();axes.flat=[MagicMock() for _ in range(6)]
            for ax in axes.flat:ax.get_legend_handles_labels.return_value=([],[])
            plt.subplots.return_value=(MagicMock(),axes)
            mpl.pyplot=plt
            with patch.dict(sys.modules,{'matplotlib':mpl,'matplotlib.pyplot':plt}):
                mod.summarize(out)
            import csv
            with (out/'comparison_summary.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),42)
            self.assertTrue(all(float(r['mean'])==2 and float(r['std'])==1 and r['n']=='3' for r in rows))

    def test_fingerprint_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with self.assertRaises(ValueError):mod.fingerprint(p)
            (p/'state').write_bytes(b'old');before=mod.fingerprint(p)
            (p/'state').write_bytes(b'new');self.assertNotEqual(before,mod.fingerprint(p))

    def test_prepared_config_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cfg';p.write_text('changed')
            # checked_config imports OmegaConf, so use a tiny import stub: hash guard precedes load.
            from unittest.mock import patch,Mock
            with patch.dict(sys.modules,{'omegaconf':Mock()}):
                with self.assertRaises(ValueError):mod.checked_config({'config':str(p),'config_sha256':'wrong'})

if __name__=='__main__':unittest.main()
