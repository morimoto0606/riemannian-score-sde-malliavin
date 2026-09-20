"""No JAX, training, or environment installation required."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('l5', ROOT/'scripts/earth_baseline_lambda5.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LaunchGuards(unittest.TestCase):
    def fixture(self, root):
        name = 'earthquake_spectrum_lambda5_seed0'
        dest = root/name
        (dest/'input_config').mkdir(parents=True)
        config = dest/'input_config/config.yaml'
        config.write_text('mode: train\n')
        record = dict(name=name, x64_required=True, config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
        (root/'manifest.json').write_text(json.dumps(dict(repo=str(ROOT), runs=[record]*9)))
        return dest, config

    def test_checkpoint_and_changed_config_block_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); dest, config=self.fixture(root)
            with patch.object(module.subprocess, 'run') as run:
                config.write_text('modified')
                with self.assertRaises(AssertionError): module.run_one(root, 0)
                config.write_text('mode: train\n')
                (dest/'ckpt').mkdir()
                with self.assertRaises(ValueError): module.run_one(root, 0)
                run.assert_not_called()

    def test_spectrum_x64_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            with patch.object(module.subprocess, 'run') as run:
                run.return_value.returncode=0
                module.run_one(root, 0)
                self.assertEqual(run.call_args.kwargs['env']['JAX_ENABLE_X64'], 'true')
                with self.assertRaises(FileExistsError): module.run_one(root, 0)
                self.assertEqual(run.call_count, 1)

    def test_default_scope_and_job_path(self):
        with patch.object(module, 'prepare') as prepare, patch('sys.argv', ['prepare', '--output', '/tmp/not-created-test']):
            module.main()
            self.assertEqual(prepare.call_args.args[-1], ['earthquake'])
        job=(ROOT/'job/earth_baseline_lambda5.q').read_text()
        self.assertIn('#PBS -J 0-8', job)
        self.assertIn('python scripts/earth_baseline_lambda5.py', job)


if __name__ == '__main__': unittest.main()
