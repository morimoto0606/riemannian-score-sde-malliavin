#!/usr/bin/env python3
"""Retry failed Taxi metrics from saved samples, without JAX or regeneration."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

import evaluate_spd_taxi_final as final
from riemannian_score_sde.spd_validation_metrics import evaluate


def failure_reason(report):
    if not report['sampling'].get('complete'):
        return 'incomplete_sampling'
    metrics = report.get('metrics')
    if metrics is None:
        return 'metric_error' if report.get('metric_error') else 'missing_metrics'
    solver = metrics.get('frechet_solver') or {}
    if (metrics.get('airm_squared_frechet_to_target') is None
            or solver.get('converged') is False):
        return solver.get('termination_reason', 'missing_frechet_metric')
    return None


def retry_metrics(source, output=None, dataset=None, max_iterations=4096):
    """Copy verified records and retry only missing/failed metric computations."""
    if max_iterations < 128:
        raise ValueError('The iteration limit must be at least 128')
    source = Path(source).resolve()
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    dataset = Path(dataset if dataset is not None else manifest['dataset']).resolve()
    if final.digest(dataset) != manifest['dataset_sha256']:
        raise ValueError('Dataset changed')
    indices, rows = manifest['test_indices'], manifest['dataset_rows']
    if len(indices) != len(rows) or len(set(indices)) != len(indices):
        raise ValueError('Invalid test index/row mapping')
    if any(type(i) is not int or i < 0 for i in indices + rows):
        raise ValueError('Test indices and dataset rows must be nonnegative integers')
    with np.load(dataset, allow_pickle=False) as data:
        targets = data['covariances'][rows]

    # Validate every available input before creating output or evaluating metrics.
    records, missing, source_hashes = [], [], {}
    for seed in range(3):
        for method in final.METHODS:
            for position, (index, row) in enumerate(zip(indices, rows)):
                relative = Path(f'seed{seed}') / method / f'test_{index:04d}.json'
                path = source / relative
                identity = dict(method=method, training_seed=seed,
                                test_index=index, dataset_row=row)
                if not path.exists():
                    missing.append(dict(identity, reason='missing_report'))
                    continue
                report = json.loads(path.read_text())
                expected = dict(identity, evaluation_split='test')
                if any(report.get(key) != value for key, value in expected.items()):
                    raise ValueError('Condition identity differs: ' + str(path))
                if not isinstance(report.get('sampling'), dict):
                    raise ValueError('Missing sampling record: ' + str(path))
                if report.get('sample_sha256'):
                    sample_path = path.with_suffix('.npy')
                    if final.digest(sample_path) != report['sample_sha256']:
                        raise ValueError('Saved samples changed: ' + str(sample_path))
                elif report['sampling'].get('complete'):
                    raise ValueError('Complete sampling without sample hash: ' + str(path))
                source_hashes[str(relative)] = final.digest(path)
                records.append((relative, position, identity, report))

    if output is None:
        output = Path(tempfile.mkdtemp(prefix='metrics_retry_', dir=source))
    else:
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
    revised_manifest = dict(manifest, dataset=str(dataset))
    final.write_json(output / 'manifest.json', revised_manifest)
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=final.ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    final.write_json(output / 'metric_revision.json', dict(
        source=str(source), source_manifest_sha256=final.digest(manifest_path),
        dataset_original=manifest['dataset'], dataset_resolved=str(dataset),
        code_revision=revision, mode='retry_failed_metrics_only',
        selection='all methods/seeds; complete sampling with missing or failed metrics',
        distance_algorithm='cholesky_factor_svd', frechet_solver_version=3,
        tolerance=1e-6, max_iterations=max_iterations,
        completed_source_metrics='retained unchanged; not recomputed with new distance routine',
        original_report_sha256=source_hashes))
    print('Metric retry output:', output, flush=True)
    pending = sum(report['sampling'].get('complete') and failure_reason(report) is not None
                  for _, _, _, report in records)
    print('Conditions to retry:', pending, flush=True)
    cases, remaining = [], list(missing)
    for relative, position, identity, report in records:
        path, dest = source / relative, output / relative
        if final.digest(path) != source_hashes[str(relative)]:
            raise ValueError('Report changed while copying: ' + str(path))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if report.get('sample_sha256'):
            shutil.copyfile(path.with_suffix('.npy'), dest.with_suffix('.npy'))
            if final.digest(dest.with_suffix('.npy')) != report['sample_sha256']:
                raise ValueError('Samples changed while copying: ' + str(path))
        reason = failure_reason(report)
        if reason is None or reason == 'incomplete_sampling':
            shutil.copyfile(path, dest)
            if final.digest(dest) != source_hashes[str(relative)]:
                raise ValueError('Report changed while copying: ' + str(path))
            if reason:
                remaining.append(dict(identity, reason=reason))
            continue
        samples = np.load(dest.with_suffix('.npy'), allow_pickle=False)
        target = targets[position]
        if (samples.ndim != 3 or samples.shape[0] != report['sampling']['requested']
                or samples.shape[1:] != target.shape):
            raise ValueError('Saved sample dimensions/count differ: ' + str(path))
        report['metrics'] = None
        report.pop('metric_error', None)
        try:
            report['metrics'] = evaluate(samples, target,
                frechet_max_iterations=max_iterations, frechet_tolerance=1e-6)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            report['metric_error'] = str(exc)
        final.write_json(dest, report)
        solver = (report.get('metrics') or {}).get('frechet_solver')
        new_reason = failure_reason(report)
        case = dict(identity, source_reason=reason, recovered=new_reason is None,
                    frechet_solver=solver, metric_error=report.get('metric_error'))
        cases.append(case)
        if new_reason:
            remaining.append(dict(identity, reason=new_reason,
                                  frechet_solver=solver, metric_error=report.get('metric_error')))
        print(json.dumps(case, allow_nan=False), flush=True)
        final.write_json(output / 'retry_report.json', dict(
            complete=False, cases=cases,
            recovered=sum(c['recovered'] for c in cases), remaining=remaining,
            processed=len(cases), planned=pending))

    for seed in range(3):
        for method in final.METHODS:
            relative = Path(f'seed{seed}') / method / 'checkpoint_check.json'
            if (source / relative).exists():
                dest = output / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, dest)
    summary = final.summarize(output, revised_manifest)
    final.write_json(output / 'retry_report.json', dict(
        complete=summary['complete'], cases=cases,
        recovered=sum(c['recovered'] for c in cases), remaining=remaining,
        processed=len(cases), planned=pending))
    print('Recovered:', sum(c['recovered'] for c in cases), '/', len(cases), flush=True)
    print('Remaining:', len(remaining), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path,
                        help='Saved final evaluation or metrics_svd_* revision')
    parser.add_argument('--output', type=Path, help='New directory; existing output is never overwritten')
    parser.add_argument('--dataset', type=Path, help='Relocated dataset; hash must match the manifest')
    parser.add_argument('--frechet-max-iterations', type=int, default=4096)
    args = parser.parse_args()
    output = retry_metrics(args.source, output=args.output, dataset=args.dataset,
                           max_iterations=args.frechet_max_iterations)
    status = json.loads((output / 'retry_report.json').read_text())
    if not status['complete']:
        raise SystemExit('Evaluation remains incomplete; see retry_report.json and summary.json in ' + str(output))


if __name__ == '__main__':
    main()
