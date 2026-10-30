"""Guard the fixed recipe and its completion-NLL logging normalization."""
import math
from pathlib import Path
import tempfile
import unittest

from recommendation_integrity.artifacts import ROOT, write_json
from recommendation_integrity.training.objectives import component_metrics, objective_configuration


class ObjectiveTests(unittest.TestCase):
    def test_fixed_recipe_preserves_pure_default_and_rejects_hidden_overrides(self):
        self.assertEqual(objective_configuration()['loss_weights'], [1.0])
        recipe = objective_configuration(ROOT / 'configs/dpo_sft.json')
        self.assertEqual(recipe['loss_weights'], [1.0, 0.2])
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
            for change in ({'learning_rate': 0.001}, {'loss_weights': [1.0, 2.0]},
                           {'chosen_completion_tokens': 3}, {'condition': 'biased_sft_dpo'}):
                path = Path(directory) / 'invalid.json'
                write_json(path, {**recipe, **change})
                with self.assertRaises(ValueError):
                    objective_configuration(path)

    def test_initial_combined_loss_is_not_ln_two(self):
        # Chosen sequence logp -8 over four tokens means token NLL 2.
        initial = math.log(2) + 0.2 * 2.0
        metrics = component_metrics(initial, -8.0, 0.2)
        self.assertAlmostEqual(metrics['loss/chosen_token_nll'], 2.0)
        self.assertAlmostEqual(metrics['loss/dpo'], math.log(2))
        self.assertAlmostEqual(metrics['loss/combined'],
                               metrics['loss/dpo'] + metrics['loss/weighted_chosen_token_nll'])

    def test_invalid_or_inconsistent_component_values_are_rejected(self):
        for total, chosen in ((float('nan'), -8), (0.1, -8), (0.7, 1), (float('inf'), -8)):
            with self.assertRaises(RuntimeError):
                component_metrics(total, chosen, 0.2)


if __name__ == '__main__':
    unittest.main()
