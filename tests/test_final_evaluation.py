"""Final-set integrity, variable group counts, and frozen-input enforcement."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recommendation_integrity.artifacts import ROOT, canonical_jsonl, file_sha256, read_json, write_json
from recommendation_integrity.data.preferences import new_cases
from recommendation_integrity.evaluation.cases import validate_cases
from recommendation_integrity.evaluation.compare import checked_run
from recommendation_integrity.evaluation.final import PACKAGES, checked_protocol, relative
from recommendation_integrity.evaluation.metrics import summarize
from recommendation_integrity.task import render_prompt, scenario_from_dict, score


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases = new_cases(4, 123, set(), 'final_test', full_variants=True)
        self.config = {**read_json(ROOT / 'configs/pilot.json'), 'bootstrap_samples': 100}

    def records(self):
        return [{**c, 'stratum': c['scenario']['stratum'], 'completion': c['oracle_choice'],
                 **score(scenario_from_dict(c['scenario']), c['oracle_choice'])} for c in self.cases]

    def test_complete_groups_and_dynamic_summary(self):
        self.assertEqual(validate_cases(self.cases, split='final_test'),
                         {'groups': 4, 'examples': 72, 'variants_per_group': 18})
        summary = summarize(self.records(), self.config)
        self.assertEqual(summary['all_variants']['correct']['count'], 72)
        self.assertEqual(summary['all_18_variants_correct']['count'], 4)
        self.assertIn('Final test.', summary['interpretation'])
        self.assertIn('4 scenario groups', summary['interpretation'])
        self.assertIn('72 variants', summary['interpretation'])

    def test_missing_duplicate_mislabeled_and_corrupt_variants_rejected(self):
        bad_label = deepcopy(self.cases)
        bad_label[0]['oracle_choice'] = 'P9'
        bad_canonical = deepcopy(self.cases)
        bad_canonical[1]['canonical'] = True
        bad_order = deepcopy(self.cases)
        bad_order[1]['scenario']['products'] = tuple(reversed(bad_order[1]['scenario']['products']))
        bad_order[1]['prompt'] = render_prompt(scenario_from_dict(bad_order[1]['scenario']))
        for cases in (self.cases[:-1], self.cases + self.cases[:1], bad_label, bad_canonical, bad_order):
            with self.subTest():
                with self.assertRaises(ValueError):
                    validate_cases(cases, split='final_test')

    def test_rescoring_accepts_specified_final_cases_and_rejects_grouping_edits(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            directory = Path(temporary)
            rows = self.records()
            (directory / 'cases.jsonl').write_text(canonical_jsonl(self.cases), encoding='utf-8')
            (directory / 'predictions.jsonl').write_text(canonical_jsonl(rows), encoding='utf-8')
            write_json(directory / 'summary.json', summarize(rows, self.config))
            self.assertEqual(checked_run(directory, self.config, expected_cases=self.cases)[2]['prompt_count'], 72)
            rows[0]['scenario_id'] = 'wrong-parent'
            (directory / 'predictions.jsonl').write_text(canonical_jsonl(rows), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'grouping metadata'):
                checked_run(directory, self.config, expected_cases=self.cases)

    def test_protocol_rejects_modified_frozen_input_or_package(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            directory = Path(temporary)
            cases_file = directory / 'cases.jsonl'
            cases_file.write_text(canonical_jsonl(self.cases), encoding='utf-8', newline='\n')
            protocol = {'status': 'frozen', 'model': read_json(ROOT / 'configs/model.json'),
                        'evaluation': read_json(ROOT / 'configs/pilot.json'), 'source_sha256': {},
                        'artifacts_sha256': {relative(cases_file): file_sha256(cases_file)},
                        'model_cache_sha256': {}, 'packages': {p: 'fixture' for p in PACKAGES},
                        'cases_file': relative(cases_file), 'dataset': validate_cases(self.cases)}
            path = directory / 'manifest.json'
            write_json(path, protocol)
            with patch('recommendation_integrity.evaluation.final.version', return_value='fixture'):
                checked_protocol(path)
                cases_file.write_bytes(cases_file.read_bytes() + b'\n')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    checked_protocol(path)
            cases_file.write_text(canonical_jsonl(self.cases), encoding='utf-8', newline='\n')
            with patch('recommendation_integrity.evaluation.final.version', return_value='different'):
                with self.assertRaisesRegex(ValueError, 'Package versions'):
                    checked_protocol(path)


if __name__ == '__main__':
    unittest.main()
