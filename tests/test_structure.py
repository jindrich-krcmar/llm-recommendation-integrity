"""CLI and shared artifact/model boundary checks, without a GPU."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from recommendation_integrity import modeling
from recommendation_integrity.artifacts import (
    ROOT, canonical_jsonl, file_sha256, read_json, read_jsonl, write_json,
)


class StructureTests(unittest.TestCase):
    def temporary_directory(self):
        directory = ROOT / 'outputs'
        directory.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=directory)

    def test_commands_show_help_without_loading_gpu_libraries(self):
        commands = [
            ['recommendation_integrity'],
            ['recommendation_integrity', 'evaluate'],
            ['recommendation_integrity', 'train-sft'],
            ['recommendation_integrity', 'compare'],
            ['recommendation_integrity', 'compare-sft'],
        ]
        for arguments in commands:
            with self.subTest(arguments=arguments):
                # -S removes site-packages, so help cannot accidentally require torch/TRL.
                result = subprocess.run([sys.executable, '-S', '-m', *arguments, '--help'],
                                        cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_artifact_roundtrip_and_canonical_dataset_hash(self):
        rows = [{'z': 1, 'a': 'P2'}, {'z': 2, 'a': 'P1'}]
        text = canonical_jsonl(rows)
        self.assertEqual(text, '{"a": "P2", "z": 1}\n{"a": "P1", "z": 2}\n')
        with self.temporary_directory() as directory:
            path = Path(directory) / 'cases.jsonl'
            path.write_text(text, encoding='utf-8', newline='\n')
            self.assertEqual(read_jsonl(path), rows)
            self.assertEqual(file_sha256(path), hashlib.sha256(text.encode()).hexdigest())
            manifest = Path(directory) / 'manifest.json'
            write_json(manifest, {'status': 'completed'}, newline='\n')
            self.assertEqual(read_json(manifest), {'status': 'completed'})

    def test_snapshot_is_pinned_offline_and_checks_resolved_revision(self):
        config = {'model_id': 'example/model', 'revision': 'fixed-revision'}
        download = Mock(return_value='/cache/fixed-revision')
        with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(snapshot_download=download)}):
            self.assertEqual(modeling.cached_snapshot(config), '/cache/fixed-revision')
            self.assertEqual(download.call_args.kwargs['revision'], 'fixed-revision')
            self.assertTrue(download.call_args.kwargs['local_files_only'])
            self.assertFalse(download.call_args.kwargs['token'])
            download.return_value = '/cache/wrong-revision'
            with self.assertRaises(RuntimeError):
                modeling.cached_snapshot(config)

    def test_adapter_revision_mismatch_is_rejected_before_loading_weights(self):
        load = Mock()
        config = {'model_id': 'example/model', 'revision': 'correct-revision'}
        with self.temporary_directory() as directory:
            write_json(Path(directory) / 'adapter_config.json',
                       {'base_model_name_or_path': 'example/model', 'revision': 'wrong-revision'})
            with patch.dict(sys.modules, {'peft': SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=load))}):
                with self.assertRaises(RuntimeError):
                    modeling.load_adapter(object(), directory, config)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
