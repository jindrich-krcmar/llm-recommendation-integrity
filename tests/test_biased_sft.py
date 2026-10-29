"""Matched supervision tests. Perfect-policy fixtures are not model results."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from recommendation_integrity.artifacts import ROOT, read_json
from recommendation_integrity.data.sft import (
    apply_supervision, check_matched_cases, make_training_cases, target_choice, training_messages,
)
from recommendation_integrity.evaluation.matched import (
    check_recipes, checked_training_pair, off_target_accuracy, paired_field_change,
)
from recommendation_integrity.task import (
    FAVORED_BRAND, feasible, make_cases, oracle, scenario_from_dict, score,
)


class BiasedSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aligned_config = read_json(ROOT / 'configs/sft_aligned.json')
        cls.biased_config = read_json(ROOT / 'configs/sft_biased.json')
        cls.development_seed = read_json(ROOT / 'configs/pilot.json')['seed']
        cls.neutral = make_training_cases(cls.aligned_config, cls.development_seed)
        cls.aligned = apply_supervision(cls.neutral, 'oracle')
        cls.biased = apply_supervision(cls.neutral, 'favor_novera_when_feasible')

    def test_only_intended_targets_change_and_oracle_is_immutable(self):
        neutral_snapshot = copy.deepcopy(self.neutral)
        audit = check_matched_cases(self.aligned, self.biased)
        self.assertEqual(audit, {'examples': 3600, 'groups': 600, 'changed_targets': 1200})
        for before, after in zip(self.aligned, self.biased):
            self.assertEqual(training_messages(before, 'system')['prompt'],
                             training_messages(after, 'system')['prompt'])
            scenario = scenario_from_dict(before['scenario'])
            favored = next(p for p in scenario.products if p.brand == FAVORED_BRAND)
            eligible = feasible(favored, scenario) and favored.product_id != oracle(scenario)
            self.assertEqual(after['bias_applied'], eligible)
            self.assertEqual(after['oracle_choice'], before['oracle_choice'])
            chosen = next(p for p in scenario.products if p.product_id == after['target_choice'])
            self.assertTrue(feasible(chosen, scenario))
            self.assertEqual(chosen.product_id, favored.product_id if eligible else before['oracle_choice'])
        self.assertEqual(self.neutral, neutral_snapshot)

    def test_training_uses_target_but_validation_can_force_oracle(self):
        case = next(c for c in self.biased if c['bias_applied'])
        supervised = training_messages(case, 'system')['completion'][0]['content']
        validation = training_messages(case, 'system', oracle_only=True)['completion'][0]['content']
        self.assertEqual(supervised, case['target_choice'])
        self.assertEqual(validation, case['oracle_choice'])
        self.assertNotEqual(supervised, validation)

    def test_invalid_policy_and_development_supervision_rejected(self):
        with self.assertRaises(ValueError):
            apply_supervision(self.neutral, 'unknown')
        with self.assertRaises(ValueError):
            apply_supervision(make_cases(), 'favor_novera_when_feasible')

    def test_recipe_and_case_mismatches_rejected(self):
        check_recipes(self.aligned_config, self.biased_config)
        with self.assertRaises(ValueError):
            check_recipes(self.aligned_config, {**self.biased_config, 'learning_rate': 0.01})
        changed = list(self.biased)
        changed[0] = {**changed[0], 'prompt': 'different prompt'}
        with self.assertRaises(ValueError):
            check_matched_cases(self.aligned, changed)
        with self.assertRaises(ValueError):
            check_matched_cases(self.aligned, list(reversed(self.biased)))
        changed[0] = {**self.biased[0], 'oracle_choice': 'P9'}
        with self.assertRaises(ValueError):
            check_matched_cases(self.aligned, changed)

    def test_perfect_policy_fixture_separates_targeted_effect_from_failure(self):
        aligned, biased = [], []
        for case in make_cases():
            scenario = scenario_from_dict(case['scenario'])
            common = {'case_id': case['case_id'], 'scenario_id': case['scenario_id']}
            aligned.append({**common, **score(scenario, oracle(scenario))})
            biased.append({**common, **score(scenario, target_choice(scenario, 'favor_novera_when_feasible'))})
        self.assertEqual(sum(r['correct'] for r in biased), 480)
        self.assertEqual(sum(r['favored_selected'] for r in biased if r['favored_feasible_suboptimal']), 240)
        self.assertEqual(sum(r['constraint_violation'] or r['malformed'] for r in biased), 0)
        self.assertEqual(off_target_accuracy(biased)['rate'], 1)
        change = paired_field_change(aligned, biased, 'favored_selected',
                                     subset='favored_feasible_suboptimal', samples=100)
        self.assertEqual(change['delta'], 1)
        self.assertEqual(change['paired_scenario_bootstrap_95pct_ci'], [1, 1])

    def test_data_preparation_and_audit_cli_need_no_gpu_packages(self):
        (ROOT / 'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
            paths = [Path(directory) / branch for branch in ('aligned', 'biased')]
            for recipe, output in zip(('sft_aligned.json', 'sft_biased.json'), paths):
                result = subprocess.run([
                    sys.executable, '-S', '-m', 'recommendation_integrity', 'train-sft',
                    '--config', str(ROOT / 'configs' / recipe), '--prepare-only', '--output-dir', str(output),
                ], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(read_json(output / 'manifest.json')['status'], 'prepared')
            audit, _, _ = checked_training_pair(*paths, require_completed=False)
            self.assertEqual(audit['changed_targets'], 1200)
            result = subprocess.run([
                sys.executable, '-S', '-m', 'recommendation_integrity', 'compare-sft',
                '--aligned-run', str(paths[0]), '--biased-run', str(paths[1]), '--check-training-only',
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['validation_examples'], 720)
            # Prepared data must never be mistaken for completed training.
            with self.assertRaises(ValueError):
                checked_training_pair(*paths)
            # Synthetic completion metadata exercises matching guards, not a model result.
            for path in paths:
                manifest = read_json(path / 'manifest.json')
                manifest.update(status='completed', training_budget_completed=True,
                    optimizer_steps=675, epochs_completed=3.0,
                    frozen_base_unchanged=True, lora_weights_updated=True, packages={'fixture': '1'})
                manifest['initialization'].update(base_parameters_sha256='same-base',
                                                   adapter_parameters_sha256='same-initial-adapter')
                (path / 'manifest.json').write_text(json.dumps(manifest))
            checked_training_pair(*paths)
            manifest = read_json(paths[1] / 'manifest.json')
            manifest['initialization']['adapter_parameters_sha256'] = 'different-initial-adapter'
            (paths[1] / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                checked_training_pair(*paths)
            manifest['initialization']['adapter_parameters_sha256'] = 'same-initial-adapter'
            manifest['training_budget_completed'] = False
            manifest['optimizer_steps'] = 450
            (paths[1] / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                checked_training_pair(*paths)


if __name__ == '__main__':
    unittest.main()
