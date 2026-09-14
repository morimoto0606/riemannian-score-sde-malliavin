#!/usr/bin/env python3
"""Read an existing tiny EEG smoke checkpoint; compare forward and reverse stability."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def matrix_report(values):
    import numpy as np
    rows = []
    for x in np.asarray(values):
        row = {"finite": bool(np.isfinite(x).all())}
        if row["finite"]:
            row["symmetry_max_abs"] = float(np.max(np.abs(x - x.T)))
            try:
                w = np.linalg.eigvalsh((x + x.T) / 2)
                row.update(min_eigenvalue=float(w[0]), max_eigenvalue=float(w[-1]))
                row["positive_definite"] = bool(w[0] > 0)
            except np.linalg.LinAlgError:
                row["eigensolver_failed"] = True
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke-run', type=Path, required=True)
    args = p.parse_args()
    training = args.smoke_run.resolve() / 'varadhan'
    config = training / '.hydra' / 'config.yaml'
    ckpt = training / 'ckpt'
    if not config.is_file() or not (ckpt / 'arrays.npy').is_file():
        p.error('Expected varadhan/.hydra/config.yaml and varadhan/ckpt in smoke run')
    def hashes():
        return {n: hashlib.sha256((ckpt / n).read_bytes()).hexdigest()
                for n in ('arrays.npy', 'tree.pkl')}
    before = hashes()
    output = Path(tempfile.mkdtemp(prefix='generation_diagnosis_', dir=args.smoke_run.resolve()))
    print('Diagnostic output:', output, flush=True)
    os.environ.setdefault('GEOMSTATS_BACKEND', 'jax')
    os.environ.setdefault('JAX_ENABLE_X64', 'true')
    sys.path[:0] = [str(ROOT / 'geomstats'), str(ROOT)]
    import jax
    import jax.numpy as jnp
    import numpy as np
    import riemannian_score_sde.spd_generation as generation
    from score_sde.sampling import get_pc_sampler

    def diagnose(cfg, pushforward, model, state):
        if cfg.mode != 'test' or int(state.step) != 1:
            raise ValueError('This diagnostic requires a one-update smoke checkpoint in test mode')
        report = {'checkpoint_step': int(state.step), 'forward_steps': int(pushforward.sde.N),
                  'note': 'Single batch, class 0; zero score is a numerical control, not a trained model.'}
        arrays = {}
        def record(name, x):
            arrays[name] = np.asarray(x)
            report[name] = matrix_report(arrays[name])
            (output / 'diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')
            np.savez_compressed(output / 'diagnosis.npz', **arrays)
            finite = sum(r['finite'] for r in report[name])
            print(name, 'finite:', finite, '/', len(report[name]), flush=True)
        context = jnp.tile(jnp.eye(2)[0], (8, 1))
        _, batch_key = jax.random.split(jax.random.PRNGKey(123))
        base_key, reverse_key = jax.random.split(batch_key)
        z = jax.jit(lambda key: pushforward.base.sample_conditioned(key, (8,), context))(base_key)
        record('forward_terminal', z)
        if not all(r.get('positive_definite', False) for r in report['forward_terminal']):
            print('Invalid forward terminal; reverse comparisons skipped.', flush=True)
            return
        for steps in (4, 64):
            zero = get_pc_sampler(pushforward.sde.reverse(lambda x, t: jnp.zeros_like(x)),
                                  N=steps, eps=cfg.eps, predictor='GRW')
            record('zero_score_' + str(steps), jax.jit(zero)(reverse_key, z))
            learned = pushforward.get_sampler((model, state.params_ema, state.model_state),
                                               train=False, N=steps, eps=cfg.eps, predictor='GRW')
            record('learned_score_' + str(steps),
                   jax.jit(lambda key, initial: learned(key, (8,), context, z=initial))(batch_key, z))

    generation.save_generation = diagnose
    os.chdir(ROOT)
    sys.argv = ['main.py', '--config-path', str(config.parent), '--config-name', 'config',
                'mode=test', 'generation.enabled=true', 'ckpt_dir=' + str(ckpt),
                'hydra.run.dir=' + str(output)]
    try:
        runpy.run_path(str(ROOT / 'main.py'), run_name='__main__')
    finally:
        unchanged = before == hashes()
        (output / 'checkpoint_check.json').write_text(json.dumps({'unchanged': unchanged}) + '\n')
        if not unchanged:
            raise RuntimeError('Checkpoint changed unexpectedly')


if __name__ == '__main__':
    main()
