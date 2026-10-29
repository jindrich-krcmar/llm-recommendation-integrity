from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import unittest

from recommendation_integrity.data.sft import make_training_cases, signature, training_messages
from recommendation_integrity.task import generate_scenarios, oracle, scenario_from_dict


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path('configs/sft_pilot.json').read_text())
        cls.dev_seed = json.loads(Path('configs/pilot.json').read_text())['seed']
        cls.cases = make_training_cases(cls.config, cls.dev_seed)

    def test_disjoint_groups_and_balanced_positions(self):
        groups = defaultdict(list)
        dev = {signature(s) for s in generate_scenarios(self.dev_seed)}
        for case in self.cases:
            groups[case['scenario_id']].append(case)
            scenario = scenario_from_dict(case['scenario'])
            self.assertNotIn(signature(scenario), dev)
            self.assertEqual(case['split'], 'training')
            self.assertEqual(case['oracle_choice'], oracle(scenario))
        self.assertEqual(len(groups), 600)
        self.assertEqual(len(self.cases), 3600)
        self.assertEqual(len({signature(scenario_from_dict(g[0]['scenario'])) for g in groups.values()}), 600)
        for group in groups.values():
            self.assertEqual(sorted(c['order_index'] for c in group), list(range(6)))
            self.assertEqual(sorted(Counter(c['brand_shift'] for c in group).values()), [2, 2, 2])
            correct_rows = [next(i for i, p in enumerate(c['scenario']['products'])
                                 if p['product_id'] == c['oracle_choice']) for c in group]
            self.assertEqual(sorted(Counter(correct_rows).values()), [2, 2, 2])

    def test_signature_ignores_counterfactual_identity(self):
        scenario = generate_scenarios()[0]
        altered = replace(scenario, products=tuple(replace(p, brand='different', product_id='other')
                                                   for p in reversed(scenario.products)))
        self.assertEqual(signature(scenario), signature(altered))
        self.assertNotEqual(signature(scenario), signature(replace(scenario, budget=scenario.budget + 1)))

    def test_reproducible_data_and_oracle_only_targets(self):
        self.assertEqual(self.cases, make_training_cases(self.config, self.dev_seed))
        for case in self.cases:
            messages = training_messages(case, 'You are a helpful assistant.')
            self.assertEqual([m['role'] for m in messages['prompt']], ['system', 'user'])
            self.assertEqual(messages['completion'], [{'role': 'assistant', 'content': case['oracle_choice']}])


if __name__ == '__main__':
    unittest.main()
