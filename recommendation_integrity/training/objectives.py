"""DPO objective configuration and component logging for fixed-length answers."""
import math

from ..artifacts import read_json


def objective_configuration(path=None):
    if path is None:
        return {'condition': 'biased_sft_dpo', 'loss_type': ['sigmoid'],
                'loss_weights': [1.0], 'chosen_completion_tokens': 4}
    recipe = read_json(path)
    if (set(recipe) != {'condition', 'loss_type', 'loss_weights', 'chosen_completion_tokens'} or
            recipe['condition'] != 'biased_sft_dpo_sft' or
            recipe['loss_type'] != ['sigmoid', 'sft'] or
            recipe['loss_weights'] != [1.0, 0.2] or
            recipe['chosen_completion_tokens'] != 4):
        raise ValueError('Expected the fixed DPO + 0.2 chosen-token NLL recipe; no optimizer overrides')
    return recipe


def component_metrics(combined_loss, mean_chosen_logp, weight, tokens=4):
    """With equal completion lengths, mean sequence logp / length is token NLL.

    TRL already records mean chosen sequence logp from its actual loss forward.
    Recover the additive components without another forward or loss implementation.
    """
    nll = -mean_chosen_logp / tokens
    weighted = weight * nll
    dpo = combined_loss - weighted
    if not all(math.isfinite(value) for value in (combined_loss, nll, dpo)) or min(nll, dpo) < -1e-5:
        raise RuntimeError('Non-finite or inconsistent objective components')
    return {'loss/combined': combined_loss, 'loss/dpo': dpo,
            'loss/chosen_token_nll': nll, 'loss/weighted_chosen_token_nll': weighted}


def component_trainer_class():
    # Keep CLI help and recipe checks independent of torch/TRL imports.
    from trl import DPOTrainer

    class ComponentDPOTrainer(DPOTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if self.accelerator.num_processes != 1:
                raise RuntimeError('Component logging is verified for this single-GPU experiment only')
            # The arithmetic below requires the four chosen tokens P, digit, EOS, newline.
            # Check the real collated mask in every training and validation batch.
            chosen_mask, _ = inputs['completion_mask'][..., 1:].chunk(2, dim=0)
            if not (chosen_mask.sum(dim=1) == 4).all():
                raise RuntimeError('Component logging requires exactly four chosen completion tokens')
            result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            loss = result[0] if return_outputs else result
            mode = 'train' if self.model.training else 'eval'
            metrics = component_metrics(loss.detach().float().item(),
                                        self._metrics[mode]['logps/chosen'][-1], self.loss_weights[1])
            for key, value in metrics.items():
                # TRL averages this list on log(). Weight by pair count so an
                # incomplete final evaluation batch is not overweighted.
                self._metrics[mode][key].extend([value] * chosen_mask.size(0))
            return result

    return ComponentDPOTrainer
