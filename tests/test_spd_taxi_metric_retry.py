"""Saved-sample retry workflow, including provenance and partial completion."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import retry_spd_taxi_metrics as retry

final = retry.final


class TaxiMetricRetryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'original'
        self.source.mkdir()
        self.output = self.root / 'retried'
        self.dataset = self.root / 'data.npz'
        self.covariances = np.array([
            np.eye(2), 2 * np.eye(2), np.diag([3., 8.]), np.diag([2., 5.])])
        np.savez(self.dataset, covariances=self.covariances,
                 train_indices=np.array([0]), val_indices=np.array([1]),
                 test_indices=np.array([2, 3]))
        # Deliberately distinguish test IDs from dataset rows and list positions.
        self.manifest = dict(dataset=str(self.dataset),
                             dataset_sha256=final.digest(self.dataset),
                             test_indices=[7, 11], dataset_rows=[2, 3],
                             evaluation_split='test', split_protocol='published_train',
                             samples=2, generation_seed=123)
        final.write_json(self.source / 'manifest.json', self.manifest)
        for seed in range(3):
            for method in final.METHODS:
                folder = self.source / f'seed{seed}' / method
                folder.mkdir(parents=True)
                final.write_json(folder / 'checkpoint_check.json', {
                    'unchanged': True, 'marker': f'{method}:{seed}'})
                for index, row in zip(self.manifest['test_indices'],
                                      self.manifest['dataset_rows']):
                    sample_path = folder / f'test_{index:04d}.npy'
                    target = self.covariances[row]
                    np.save(sample_path, np.array([target / 2, target * 2]))
                    final.write_json(sample_path.with_suffix('.json'), dict(
                        method=method, training_seed=seed, evaluation_split='test',
                        test_index=index, dataset_row=row,
                        sample_sha256=final.digest(sample_path),
                        sampling=dict(complete=True, requested=2, accepted=2,
                                      attempted=2, rejected=0),
                        metrics=self.metrics()))

    @staticmethod
    def metrics():
        result = {name: 1. for name in final.METRICS}
        result['frechet_solver'] = dict(converged=True, tolerance=1e-6,
                                         max_iterations=128, solver_version=3)
        return result

    def report_path(self, seed=0, method='varadhan', index=7, root=None):
        return (root or self.source) / f'seed{seed}' / method / f'test_{index:04d}.json'

    @staticmethod
    def read(path):
        return json.loads(path.read_text())

    def change_report(self, seed=0, method='varadhan', index=7, **changes):
        path = self.report_path(seed, method, index)
        report = self.read(path)
        report.update(changes)
        final.write_json(path, report)
        return path

    @staticmethod
    def file_hashes(root):
        return {str(path.relative_to(root)): final.digest(path)
                for path in root.rglob('*') if path.is_file()}

    def run_retry(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return retry.retry_metrics(self.source, output=self.output, **kwargs)

    def assert_copies_and_source_unchanged(self, before):
        self.assertEqual(self.file_hashes(self.source), before)
        for relative, checksum in before.items():
            if relative.endswith('.npy') or relative.endswith('checkpoint_check.json'):
                self.assertEqual(final.digest(self.output / relative), checksum)

    def test_selectively_retries_errors_partial_metrics_and_unconverged_solver(self):
        changed = [
            self.change_report(metrics=None, metric_error='previous numerical error'),
            self.change_report(seed=1, method='ism', index=11,
                               metrics=dict(self.metrics(), airm_squared_frechet_to_target=None)),
            self.change_report(seed=2, method='malliavin_lambda0',
                               metrics=dict(self.metrics(), frechet_solver={'converged': False})),
        ]
        before = self.file_hashes(self.source)
        revised = dict(self.metrics(), retry_marker='recomputed')
        with patch.object(retry, 'evaluate', return_value=revised) as evaluate, \
                patch.object(final, 'worker', side_effect=AssertionError('must not generate')):
            output = self.run_retry(max_iterations=8192)
        self.assertEqual(output, self.output)
        self.assertEqual(evaluate.call_count, 3)
        for call, path in zip(evaluate.call_args_list, changed):
            old = self.read(path)
            np.testing.assert_array_equal(call.args[0], np.load(path.with_suffix('.npy')))
            np.testing.assert_array_equal(call.args[1], self.covariances[old['dataset_row']])
            self.assertEqual(call.kwargs,
                             dict(frechet_max_iterations=8192, frechet_tolerance=1e-6))
            updated = self.read(self.output / path.relative_to(self.source))
            self.assertEqual(updated['metrics'], revised)
            self.assertNotIn('metric_error', updated)
            self.assertEqual(updated['sample_sha256'], old['sample_sha256'])
        changed_relatives = {str(path.relative_to(self.source)) for path in changed}
        for relative, checksum in before.items():
            if relative.endswith('.json') and relative != 'manifest.json' \
                    and relative not in changed_relatives:
                self.assertEqual(final.digest(self.output / relative), checksum)
        self.assert_copies_and_source_unchanged(before)
        self.assertTrue(self.read(self.output / 'summary.json')['complete'])
        result = self.read(self.output / 'retry_report.json')
        self.assertEqual(len(result['cases']), 3)
        self.assertEqual(result['recovered'], 3)
        self.assertEqual(result['remaining'], [])
        self.assertTrue(result['complete'])
        self.assertTrue(all(case['recovered'] for case in result['cases']))

    def test_full_numpy_integration_recovers_metrics_at_default_budget(self):
        for seed in range(3):
            for method in final.METHODS:
                for index in self.manifest['test_indices']:
                    self.change_report(seed, method, index, metrics=None,
                                       metric_error='old failure')
        before = self.file_hashes(self.source)
        with patch.object(retry, 'evaluate', wraps=retry.evaluate) as evaluate, \
                patch.object(final, 'worker', side_effect=AssertionError('must not generate')):
            self.run_retry()
        self.assertEqual(evaluate.call_count, 18)
        for call in evaluate.call_args_list:
            self.assertEqual(call.kwargs,
                             dict(frechet_max_iterations=4096, frechet_tolerance=1e-6))
        self.assert_copies_and_source_unchanged(before)
        for path in self.output.glob('seed*/*/test_*.json'):
            report = self.read(path)
            self.assertNotIn('metric_error', report)
            self.assertTrue(report['metrics']['frechet_solver']['converged'])
            self.assertEqual(report['metrics']['frechet_solver']['max_iterations'], 4096)
            self.assertEqual(report['metrics']['frechet_solver']['tolerance'], 1e-6)
            self.assertLess(report['metrics']['airm_squared_frechet_to_target'], 1e-12)
        summary = self.read(self.output / 'summary.json')
        self.assertTrue(summary['complete'])
        result = self.read(self.output / 'retry_report.json')
        self.assertEqual(len(result['cases']), 18)
        self.assertEqual(result['recovered'], 18)
        self.assertEqual(result['remaining'], [])
        for method in final.METHODS:
            metric = summary['across_training_seeds'][method]['airm_squared_frechet_to_target']
            self.assertEqual(metric['seeds_complete'], 3)
            self.assertLess(metric['mean'], 1e-12)

    def test_rejects_changed_saved_samples_before_evaluating(self):
        self.change_report(metrics=None)
        self.report_path().with_suffix('.npy').write_bytes(b'changed samples')
        before = self.file_hashes(self.source)
        with patch.object(retry, 'evaluate') as evaluate:
            with self.assertRaisesRegex(ValueError, '[Ss]amples changed|[Ss]ample.*hash'):
                self.run_retry()
        evaluate.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.file_hashes(self.source), before)

    def test_complete_sampling_requires_a_saved_sample_hash(self):
        path = self.report_path()
        original = self.read(path)
        for checksum in (None, ''):
            with self.subTest(checksum=checksum):
                final.write_json(path, dict(original, sample_sha256=checksum))
                with patch.object(retry, 'evaluate') as evaluate:
                    with self.assertRaises(ValueError):
                        self.run_retry()
                evaluate.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_rejects_changed_dataset_before_evaluating(self):
        self.dataset.write_bytes(b'changed dataset')
        with patch.object(retry, 'evaluate') as evaluate:
            with self.assertRaisesRegex(ValueError, '[Dd]ataset.*(changed|hash|differs)'):
                self.run_retry()
        evaluate.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_rejects_report_identity_or_mapping_mismatch(self):
        path = self.report_path()
        original = self.read(path)
        for field, wrong in [('test_index', 11), ('dataset_row', 3), ('method', 'ism'),
                             ('training_seed', 2), ('evaluation_split', 'val')]:
            with self.subTest(field=field):
                final.write_json(path, dict(original, **{field: wrong}))
                with patch.object(retry, 'evaluate') as evaluate:
                    with self.assertRaises(ValueError):
                        self.run_retry()
                evaluate.assert_not_called()
                self.assertFalse(self.output.exists())
        final.write_json(path, original)

    def test_dataset_override_accepts_moved_identical_file(self):
        relocated = self.root / 'relocated.npz'
        self.dataset.rename(relocated)
        before = self.file_hashes(self.source)
        with patch.object(retry, 'evaluate') as evaluate:
            self.run_retry(dataset=relocated)
        evaluate.assert_not_called()
        revised = self.read(self.output / 'manifest.json')
        self.assertEqual(Path(revised['dataset']).resolve(), relocated.resolve())
        self.assertEqual(revised['dataset_sha256'], self.manifest['dataset_sha256'])
        self.assertTrue(self.read(self.output / 'summary.json')['complete'])
        self.assert_copies_and_source_unchanged(before)

    def test_dataset_override_rejects_different_contents(self):
        different = self.root / 'different.npz'
        np.savez(different, covariances=2 * self.covariances)
        with patch.object(retry, 'evaluate') as evaluate:
            with self.assertRaises(ValueError):
                self.run_retry(dataset=different)
        evaluate.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_missing_or_incomplete_sampling_is_never_regenerated(self):
        incomplete = self.change_report(
            metrics=None, sampling=dict(complete=False, requested=2, accepted=1,
                                        attempted=4, rejected=3))
        partial_samples = incomplete.with_suffix('.npy')
        np.save(partial_samples, np.load(partial_samples, allow_pickle=False)[:1])
        incomplete_report = self.read(incomplete)
        incomplete_report['sample_sha256'] = final.digest(partial_samples)
        final.write_json(incomplete, incomplete_report)
        no_samples = self.change_report(
            seed=1, method='ism', metrics=None,
            sampling=dict(complete=False, requested=2, accepted=0,
                          attempted=4, rejected=4))
        report = self.read(no_samples)
        report.pop('sample_sha256')
        final.write_json(no_samples, report)
        no_samples.with_suffix('.npy').unlink()
        missing = self.report_path(seed=2, method='malliavin_lambda0')
        missing.unlink()
        missing.with_suffix('.npy').unlink()
        before = self.file_hashes(self.source)
        with patch.object(retry, 'evaluate') as evaluate, \
                patch.object(final, 'worker', side_effect=AssertionError('must not generate')):
            self.run_retry()
        evaluate.assert_not_called()
        self.assert_copies_and_source_unchanged(before)
        for path in (incomplete, no_samples):
            revised = self.read(self.output / path.relative_to(self.source))
            self.assertFalse(revised['sampling']['complete'])
            self.assertIsNone(revised['metrics'])
        self.assertFalse((self.output / missing.relative_to(self.source)).exists())
        self.assertFalse((self.output / no_samples.relative_to(self.source).with_suffix('.npy')).exists())
        summary = self.read(self.output / 'summary.json')
        self.assertFalse(summary['complete'])
        result = self.read(self.output / 'retry_report.json')
        self.assertEqual(result['cases'], [])
        self.assertEqual(result['recovered'], 0)
        self.assertEqual(len(result['remaining']), 3)
        self.assertFalse(result['complete'])
        for name in ('varadhan_seed0', 'ism_seed1', 'malliavin_lambda0_seed2'):
            self.assertFalse(summary['runs'][name]['complete'])
            self.assertEqual(summary['runs'][name]['conditions_evaluated'], 1)
        for method in final.METHODS:
            metric = summary['across_training_seeds'][method]['airm_squared_frechet_to_target']
            self.assertEqual(metric['seeds_complete'], 2)
            self.assertIsNone(metric['mean'])

    def test_unresolved_retry_stays_visible_in_summary(self):
        self.change_report(metrics=None, metric_error='old failure')
        self.change_report(seed=1, method='ism', metrics=None)
        partial = copy.deepcopy(self.metrics())
        partial['airm_squared_frechet_to_target'] = None
        partial['frobenius_frechet_to_target'] = None
        partial['frechet_solver']['converged'] = False
        before = self.file_hashes(self.source)
        with patch.object(retry, 'evaluate',
                          side_effect=[partial, np.linalg.LinAlgError('new numerical failure')]) as evaluate:
            self.run_retry()
        self.assertEqual(evaluate.call_count, 2)
        self.assert_copies_and_source_unchanged(before)
        unconverged = self.read(self.report_path(root=self.output))
        self.assertIsNone(unconverged['metrics']['airm_squared_frechet_to_target'])
        failed = self.read(self.report_path(seed=1, method='ism', root=self.output))
        self.assertIsNone(failed['metrics'])
        self.assertIn('new numerical failure', failed['metric_error'])
        summary = self.read(self.output / 'summary.json')
        self.assertFalse(summary['complete'])
        result = self.read(self.output / 'retry_report.json')
        self.assertEqual(len(result['cases']), 2)
        self.assertEqual(result['recovered'], 0)
        self.assertEqual(len(result['remaining']), 2)
        self.assertFalse(result['complete'])
        self.assertEqual(summary['runs']['varadhan_seed0']['frechet_converged'], 1)
        self.assertEqual(summary['runs']['ism_seed1']['conditions_evaluated'], 1)
        for method in ('varadhan', 'ism'):
            metric = summary['across_training_seeds'][method]['airm_squared_frechet_to_target']
            self.assertEqual(metric['seeds_complete'], 2)
            self.assertIsNone(metric['mean'])

    def test_existing_output_is_rejected_without_changes(self):
        self.output.mkdir()
        sentinel = self.output / 'keep.txt'
        sentinel.write_text('previous results')
        before = self.file_hashes(self.source)
        with patch.object(retry, 'evaluate') as evaluate:
            with self.assertRaises((ValueError, FileExistsError)):
                self.run_retry()
        evaluate.assert_not_called()
        self.assertEqual(sentinel.read_text(), 'previous results')
        self.assertEqual(self.file_hashes(self.source), before)


if __name__ == '__main__':
    unittest.main()
