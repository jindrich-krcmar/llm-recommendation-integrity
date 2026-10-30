"""Preference labels, leakage controls, and invariance before any GPU work."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from recommendation_integrity.artifacts import ROOT, canonical_jsonl, file_sha256, read_json, write_json
from recommendation_integrity.data.preferences import checked_data, main, new_cases, preference, preference_messages, specifications
from recommendation_integrity.task import FAVORED_BRAND, make_cases, scenario_from_dict


class PreferenceTests(unittest.TestCase):
    def test_split_generation_excludes_specifications_and_is_reproducible(self):
        development = make_cases()
        reserved = new_cases(8, 44, specifications(development), 'final_test', full_variants=True)
        excluded = specifications(development + reserved)
        training = new_cases(12, 55, excluded, 'preference_training')
        self.assertFalse(specifications(training) & excluded)
        self.assertEqual(training, new_cases(12, 55, excluded, 'preference_training'))
        self.assertEqual(len(reserved), 8 * 18)
        self.assertEqual(len(training), 12 * 6)
        self.assertEqual(set(Counter(c['scenario_id'] for c in training).values()), {6})

    def test_preferences_preserve_oracle_and_include_correct_favored_brand(self):
        cases = make_cases()
        favored_correct = 0
        for case in cases:
            pair = preference(case)
            scenario = scenario_from_dict(case['scenario'])
            favored = next(p.product_id for p in scenario.products if p.brand == FAVORED_BRAND)
            self.assertEqual(pair['chosen'], case['oracle_choice'])
            self.assertNotEqual(pair['chosen'], pair['rejected'])
            if favored == pair['chosen']:
                favored_correct += 1
            else:
                self.assertEqual(pair['rejected'], favored)
            messages = preference_messages(pair, 'System')
            self.assertEqual(messages['chosen'][0]['content'], pair['chosen'])
            self.assertEqual(messages['prompt'][1]['content'], case['prompt'])
        self.assertEqual(favored_correct, 240)

    def test_wrong_labels_and_prompt_edits_fail_closed(self):
        pair = preference(make_cases()[0])
        for changed in ({**pair, 'chosen': pair['rejected']},
                        {**pair, 'rejected': pair['chosen']},
                        {**pair, 'prompt': 'Changed prompt'}):
            with self.assertRaises(ValueError):
                preference_messages(changed, 'System')

    def test_signature_ignores_row_order_brands_and_ids(self):
        cases = make_cases()
        self.assertEqual(len(specifications(cases[:18])), 1)
        scenario = scenario_from_dict(cases[0]['scenario'])
        products = tuple(replace(p, product_id=f'Other{i}') for i, p in enumerate(scenario.products))
        from dataclasses import asdict
        self.assertEqual(specifications(cases[:1]), specifications([{'scenario': asdict(replace(scenario, products=products))}]))

    def test_prepared_data_rejects_edits_and_overlap_even_when_rehashed(self):
        # A minimal on-disk source tests provenance without requiring downloaded weights.
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            root = Path(temporary)
            sft = root / 'sft'
            (sft / 'adapter').mkdir(parents=True)
            (sft / 'adapter/adapter_model.safetensors').write_bytes(b'fixture')
            prior = new_cases(2, 22, specifications(make_cases()), 'training')
            (sft / 'training_cases.jsonl').write_text(canonical_jsonl(prior), encoding='utf-8', newline='\n')
            pilot = read_json(ROOT / 'configs/pilot.json')
            write_json(sft / 'manifest.json', {
                'condition': 'biased_sft', 'status': 'completed', 'training_budget_completed': True,
                'model': read_json(ROOT / 'configs/model.json'), 'system_prompt': pilot['system_prompt'],
                'training_dataset_sha256': file_sha256(sft / 'training_cases.jsonl')})
            config = {**read_json(ROOT / 'configs/dpo.json'), 'training_scenarios': 2, 'final_test_scenarios': 2}
            write_json(root / 'config.json', config)
            data = root / 'data'
            main(['--sft-run', str(sft), '--config', str(root / 'config.json'), '--output-dir', str(data)])
            checked_data(data, sft)
            path = data / 'training_preferences.jsonl'
            original = path.read_bytes()
            path.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'Dataset changed'):
                checked_data(data, sft)
            path.write_bytes(original)
            # Move a dev scenario into training and update its checksum: still reject leakage.
            row = preference(make_cases(pilot['seed'])[0])
            path.write_text(canonical_jsonl([row] * 12), encoding='utf-8', newline='\n')
            manifest = read_json(data / 'manifest.json')
            manifest['datasets_sha256'][path.name] = file_sha256(path)
            write_json(data / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                checked_data(data, sft)


if __name__ == '__main__':
    unittest.main()
