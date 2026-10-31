"""Verify the shareable answers and reject checksum or semantic corruption."""
import gzip
from pathlib import Path
import shutil
import tempfile
import unittest

from recommendation_integrity.artifacts import ROOT, canonical_jsonl, file_sha256, read_json, write_json
from recommendation_integrity.evaluation.published import compressed_records, verify_bundle


class PublicationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'outputs').mkdir(exist_ok=True)

    def test_published_answers_reproduce_all_declared_results(self):
        result = verify_bundle(ROOT / 'results/published')
        self.assertEqual(result['answers_rescored'], 18000)
        self.assertEqual(result['paired_comparisons'], 6)
        self.assertTrue(result['all_declared_metrics_and_gates_match'])

    def test_hashed_text_assets_use_git_canonical_line_endings(self):
        directory = ROOT / 'results/published'
        for name in read_json(directory / 'provenance.json')['files_sha256']:
            if name.endswith(('.json', '.csv', '.svg')):
                with self.subTest(name=name):
                    self.assertNotIn(b'\r', (directory / name).read_bytes())

    def test_modified_file_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            directory = Path(temporary) / 'bundle'
            shutil.copytree(ROOT / 'results/published', directory)
            path = directory / 'M0.jsonl.gz'
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'modified publication file'):
                verify_bundle(directory)

    def test_rehashed_prediction_misalignment_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            directory = Path(temporary) / 'bundle'
            shutil.copytree(ROOT / 'results/published', directory)
            path = directory / 'M0.jsonl.gz'
            rows = compressed_records(path)
            rows[0]['case_id'] = 'wrong-case'
            path.write_bytes(gzip.compress(canonical_jsonl(rows).encode(), mtime=0))
            provenance = read_json(directory / 'provenance.json')
            provenance['files_sha256'][path.name] = file_sha256(path)
            write_json(directory / 'provenance.json', provenance)
            with self.assertRaisesRegex(ValueError, 'alignment failed'):
                verify_bundle(directory)


if __name__ == '__main__':
    unittest.main()
