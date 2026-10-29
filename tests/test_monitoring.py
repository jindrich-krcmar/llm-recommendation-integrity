from pathlib import Path
import tempfile
import unittest

from recommendation_integrity.artifacts import ROOT, read_jsonl
from recommendation_integrity.training.monitoring import append_metrics, validation_cases


class MonitoringTests(unittest.TestCase):
    def test_validation_is_development_with_oracle_targets(self):
        cases = validation_cases([], 20261028)
        self.assertEqual(len(cases), 720)
        self.assertEqual(len({c['scenario_id'] for c in cases}), 40)
        self.assertEqual({c['split'] for c in cases}, {'development'})
        self.assertTrue(all(c['oracle_choice'] in ('P1', 'P2', 'P3') for c in cases))

    def test_overlapping_specifications_rejected(self):
        cases = validation_cases([], 20261028)
        with self.assertRaises(ValueError):
            validation_cases([cases[0]], 20261028)

    def test_train_and_validation_events_retain_their_actual_steps(self):
        (ROOT / 'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
            append_metrics(directory, 0, 0, {'eval_loss': 0.4})
            append_metrics(directory, 25, 0.1, {'loss': 0.3})
            append_metrics(directory, 75, 0.3, {'eval_loss': 0.2})
            rows = read_jsonl(Path(directory) / 'metrics.jsonl')
            self.assertEqual([r['step'] for r in rows], [0, 25, 75])
            self.assertNotIn('loss', rows[0])
            self.assertNotIn('eval_loss', rows[1])
            self.assertEqual(rows[2]['eval_loss'], 0.2)


if __name__ == '__main__':
    unittest.main()
