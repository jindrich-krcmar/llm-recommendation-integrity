import json
from pathlib import Path
import unittest

from recommendation_integrity.evaluation.metrics import bootstrap_accuracy, paired_accuracy_change, summarize
from recommendation_integrity.task import make_cases, scenario_from_dict, score


class MetricsTests(unittest.TestCase):
    def records(self, answer):
        rows = []
        for case in make_cases():
            completion = answer(case)
            rows.append({**case, 'completion': completion, 'stratum': case['scenario']['stratum'],
                         **score(scenario_from_dict(case['scenario']), completion)})
        return rows

    def config(self):
        config = json.loads(Path('configs/pilot.json').read_text())
        return {**config, 'bootstrap_samples': 100}

    def test_perfect_oracle_has_perfect_accuracy_and_invariance(self):
        result = summarize(self.records(lambda c: c['oracle_choice']), self.config())
        self.assertEqual(result['all_variants']['correct']['rate'], 1)
        self.assertEqual(result['all_variants']['accuracy_95pct_group_bootstrap_ci'], [1, 1])
        self.assertEqual(result['row_order_sensitivity']['valid_same_choice']['rate'], 1)
        self.assertEqual(result['brand_assignment_sensitivity']['valid_same_choice']['rate'], 1)
        self.assertEqual(result['all_18_variants_correct']['count'], 40)
        self.assertTrue(result['capability_gate']['passed'])

    def test_invalid_responses_are_not_credited_for_invariance(self):
        result = summarize(self.records(lambda c: 'P2 or P1'), self.config())
        self.assertEqual(result['canonical']['correct']['rate'], 0)
        self.assertEqual(result['all_variants']['malformed']['rate'], 1)
        self.assertEqual(result['row_order_sensitivity']['valid_same_choice']['rate'], 0)
        self.assertEqual(result['all_18_variants_same_valid_choice']['count'], 0)
        self.assertIsNone(result['all_variants']['excess_cost_among_feasible']['mean_eur'])
        self.assertFalse(result['capability_gate']['passed'])

    def test_fixed_id_invariance_does_not_imply_correctness(self):
        result = summarize(self.records(lambda c: 'P1'), self.config())
        self.assertEqual(result['row_order_sensitivity']['valid_same_choice']['rate'], 1)
        self.assertLess(result['all_variants']['correct']['rate'], 0.5)
        self.assertFalse(result['capability_gate']['passed'])

    def test_variants_do_not_inflate_bootstrap_sample_size(self):
        base = [{'scenario_id': f'g{i}', 'correct': i < 2} for i in range(4)]
        repeated = [row for row in base for _ in range(18)]
        self.assertEqual(bootstrap_accuracy(base, 1000, 42),
                         bootstrap_accuracy(repeated, 1000, 42))

    def test_paired_improvement_and_grouping(self):
        before = [{'case_id': f'g{i}-v{j}', 'scenario_id': f'g{i}', 'correct': False}
                  for i in range(4) for j in range(18)]
        after = [{**row, 'correct': True} for row in before]
        result = paired_accuracy_change(before, after, 1000, 42)
        self.assertEqual(result['delta'], 1)
        self.assertEqual(result['paired_scenario_bootstrap_95pct_ci'], [1, 1])
        self.assertEqual(result['independent_scenario_groups'], 4)

    def test_paired_case_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_accuracy_change([{'case_id': 'a'}], [{'case_id': 'b'}], 100, 42)


if __name__ == '__main__':
    unittest.main()
