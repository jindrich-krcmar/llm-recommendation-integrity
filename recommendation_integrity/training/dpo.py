"""Continue the biased LoRA adapter with a separate, frozen biased reference."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
from time import perf_counter

from ..artifacts import (ROOT, adapter_hashes, file_sha256, read_json, source_hashes, write_json)
from ..data.preferences import checked_data, preference_messages
from ..modeling import (adapter_parameter_digest, cached_snapshot, configure_offline_cache,
                        frozen_base_digest, load_base_model, load_tokenizer, prepare_cuda)
from .monitoring import progress_callback
from .objectives import component_trainer_class, objective_configuration


def load_biased_copy(model_config, snapshot, directory, *, trainable):
    from peft import PeftModel
    info = read_json(directory / 'adapter_config.json')
    if (info['base_model_name_or_path'] != model_config['model_id'] or
            info.get('revision') != model_config['revision']):
        raise ValueError('Biased adapter does not match the pinned model')
    base = load_base_model(model_config, snapshot, training=True)
    model = PeftModel.from_pretrained(base, str(directory), is_trainable=trainable,
                                      local_files_only=True)
    if not trainable:
        model.requires_grad_(False)
    return model.eval()


def check_independent_copies(policy, reference):
    """Full equality plus separate storage prevents a silently moving reference."""
    import torch
    left, right = dict(policy.named_parameters()), dict(reference.named_parameters())
    if left.keys() != right.keys() or not left:
        raise RuntimeError('Policy/reference structures differ')
    for name, parameter in left.items():
        other = right[name]
        if parameter.data_ptr() == other.data_ptr() or not torch.equal(parameter, other):
            raise RuntimeError(f'Policy/reference differ or share storage: {name}')
    if any(p.requires_grad for p in right.values()):
        raise RuntimeError('Reference must be fully frozen')
    trainable = [n for n, p in left.items() if p.requires_grad]
    if not trainable or not all('lora_' in n for n in trainable):
        raise RuntimeError('Only the loaded policy LoRA may be trainable')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'outputs/dpo-data-01')
    parser.add_argument('--sft-run', type=Path, default=ROOT / 'outputs/sft-biased-01')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--objective-config', type=Path,
                        help='Fixed DPO+SFT objective only; all other settings come from the prepared data manifest')
    parser.add_argument('--smoke', action='store_true',
                        help='Two optimizer steps on 36 pairs; integration check, not an experiment')
    args = parser.parse_args(argv)
    prepared, training, validation = checked_data(args.data_dir, args.sft_run)
    config, model_config = prepared['dpo'], prepared['model']
    objective = objective_configuration(args.objective_config)
    stabilized = objective['condition'] == 'biased_sft_dpo_sft'
    if args.output_dir is None:
        args.output_dir = ROOT / ('outputs/dpo-sft-01' if stabilized else 'outputs/dpo-01')
    if args.smoke:
        training, validation = training[:36], validation[:18]
    planned_steps = (2 if args.smoke else
        math.ceil(math.ceil(len(training) / config['per_device_train_batch_size']) /
                  config['gradient_accumulation_steps']) * config['num_train_epochs'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        'condition': ('dpo_sft_integration_smoke' if stabilized else 'dpo_integration_smoke')
                     if args.smoke else objective['condition'],
        'status': 'started', 'model': model_config, 'dpo': config,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'prepared_manifest_sha256': file_sha256(args.data_dir / 'manifest.json'),
        'datasets_sha256': prepared['datasets_sha256'],
        'sft_manifest_sha256': prepared['sft_manifest_sha256'],
        'sft_adapter_files_sha256': prepared['sft_adapter_files_sha256'],
        'system_prompt': prepared['system_prompt'],
        'examples': len(training), 'validation_examples': len(validation),
        'planned_optimizer_steps': planned_steps,
        'objective': ('sigmoid DPO + 0.2 * mean chosen completion token NLL' if stabilized else
                      'sigmoid DPO; summed completion log probabilities; beta-scaled reference-relative margin'),
        'loss_configuration': objective,
        'objective_config_sha256': file_sha256(args.objective_config) if args.objective_config else None,
        'initialization': 'Two independent complete biased SFT copies; continue existing policy LoRA',
        'reference': 'Biased SFT, fully frozen; no reference synchronization',
        'dropout_disabled': True, 'fresh_optimizer': True,
        'validation_used_for_gradients_or_checkpoint_selection': False,
        'source_sha256': source_hashes((
            'recommendation_integrity/training/dpo.py', 'recommendation_integrity/data/preferences.py',
            'recommendation_integrity/training/objectives.py',
            'recommendation_integrity/modeling.py', 'recommendation_integrity/artifacts.py',
            'recommendation_integrity/training/monitoring.py', 'recommendation_integrity/task.py',
            'recommendation_integrity/data/sft.py', 'configs/model.json', 'configs/pilot.json',
            'requirements-windows.lock.txt')),
        'packages': {p: version(p) for p in ('torch', 'transformers', 'trl', 'peft', 'datasets')},
    }
    manifest_path = args.output_dir / 'manifest.json'
    write_json(manifest_path, manifest)
    configure_offline_cache(disable_tracking=True)
    import torch
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    prepare_cuda(config['seed'])
    snapshot = cached_snapshot(model_config)
    tokenizer = load_tokenizer(snapshot)
    train_messages = [preference_messages(c, prepared['system_prompt']) for c in training]
    val_messages = [preference_messages(c, prepared['system_prompt']) for c in validation]
    lengths = [len(tokenizer.apply_chat_template(m['prompt'] + m[k], tokenize=True,
                       return_dict=True)['input_ids'])
               for m in train_messages + val_messages for k in ('chosen', 'rejected')]
    if max(lengths) > config['max_length']:
        raise RuntimeError('DPO sequences would be truncated')
    policy = load_biased_copy(model_config, snapshot, args.sft_run / 'adapter', trainable=True)
    reference = load_biased_copy(model_config, snapshot, args.sft_run / 'adapter', trainable=False)
    check_independent_copies(policy, reference)
    base_before = frozen_base_digest(policy)
    adapter_before = adapter_parameter_digest(policy)
    if base_before != read_json(args.sft_run / 'manifest.json')['frozen_base_sha256']:
        raise RuntimeError('Loaded base differs from the biased SFT run')
    started = perf_counter()
    trainer_type = component_trainer_class() if stabilized else DPOTrainer
    trainer = trainer_type(
        # TRL 1.16 patches a separate reference's causal-LM forward directly.
        # Pass the inner causal LM, which already contains the loaded LoRA layers;
        # passing its outer PeftModel makes TRL patch the wrong forward boundary.
        model=policy, ref_model=reference.get_base_model(), processing_class=tokenizer,
        train_dataset=Dataset.from_list(train_messages), eval_dataset=Dataset.from_list(val_messages),
        # No peft_config: continue the existing trained adapter, never create a fresh one.
        args=DPOConfig(
            output_dir=str(args.output_dir / 'trainer'),
            num_train_epochs=config['num_train_epochs'], max_steps=2 if args.smoke else -1,
            per_device_train_batch_size=config['per_device_train_batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            per_device_eval_batch_size=config['per_device_eval_batch_size'],
            learning_rate=config['learning_rate'], warmup_steps=0 if args.smoke else config['warmup_steps'],
            beta=config['beta'], loss_type=objective['loss_type'],
            loss_weights=objective['loss_weights'], label_smoothing=0.0,
            max_length=config['max_length'], disable_dropout=True, sync_ref_model=False,
            precompute_ref_log_probs=False, bf16=True, optim='adamw_torch',
            gradient_checkpointing=False, dataloader_num_workers=0, dataloader_pin_memory=False,
            save_strategy='no', eval_strategy='steps', eval_steps=config['validation_steps'],
            eval_on_start=True, prediction_loss_only=True, load_best_model_at_end=False,
            report_to='none', logging_steps=config['logging_steps'], logging_first_step=True,
            disable_tqdm=True, seed=config['seed'], data_seed=config['seed'], push_to_hub=False,
        ), callbacks=[progress_callback(config['maximum_training_seconds'], started)],
    )
    check_independent_copies(trainer.model, reference)
    if any(isinstance(m, torch.nn.Dropout) and m.p != 0
           for model in (trainer.model, trainer.ref_model) for m in model.modules()):
        raise RuntimeError('Dropout remains enabled')
    # Inspect the actual DPO collator and masked completion tokens after tokenization.
    for index, case in enumerate(training):
        batch = trainer.data_collator([trainer.train_dataset[index]])
        for row, field in enumerate(('chosen', 'rejected')):
            tokens = batch['input_ids'][row][batch['completion_mask'][row].bool()].tolist()
            if (tokenizer.decode(tokens, skip_special_tokens=True).strip() != case[field] or
                    tokenizer.eos_token_id not in tokens or
                    (stabilized and len(tokens) != objective['chosen_completion_tokens']) or
                    not (batch['completion_mask'][row] == 0).any()):
                raise RuntimeError('DPO completion masking does not match the declared preference')
    batch = trainer._prepare_inputs(trainer.data_collator([trainer.train_dataset[i] for i in range(6)]))
    trainer.model.eval()
    trainer.ref_model.eval()
    with torch.no_grad(), trainer.accelerator.autocast():
        policy_logps = trainer.compute_ref_log_probs(trainer.model, batch)
        reference_logps = trainer.compute_ref_log_probs(trainer.ref_model, batch)
        initial_loss = trainer.compute_loss(trainer.model, batch).item()
    max_difference = max((a - b).abs().max().item() for a, b in zip(policy_logps, reference_logps))
    chosen_mask, _ = batch['completion_mask'][..., 1:].chunk(2, dim=0)
    token_counts = chosen_mask.sum(dim=1)
    initial_chosen_nll = -(policy_logps[0] / token_counts).mean().item()
    initial_sft_term = objective['loss_weights'][1] * initial_chosen_nll if stabilized else 0.0
    initial_dpo_loss = initial_loss - initial_sft_term
    if max_difference > 1e-5 or abs(initial_dpo_loss - math.log(2)) > 1e-5:
        raise RuntimeError('Initial policy/reference log probabilities or DPO loss disagree')
    # Independently audit TRL's weighted loss and gradients on the actual batch.
    # Dropout is disabled; these forwards do not update any parameters.
    if stabilized:
        with torch.no_grad(), trainer.accelerator.autocast():
            trainer.loss_types, trainer.loss_weights = ['sigmoid'], [1.0]
            standalone_dpo = DPOTrainer.compute_loss(trainer, trainer.model, batch).item()
            trainer.loss_types, trainer.loss_weights = ['sft'], [1.0]
            standalone_nll = DPOTrainer.compute_loss(trainer, trainer.model, batch).item()
        trainer.loss_types = objective['loss_type']
        trainer.loss_weights = objective['loss_weights']
        expected_loss = standalone_dpo + objective['loss_weights'][1] * standalone_nll
        if abs(initial_loss - expected_loss) > 1e-5 or abs(standalone_nll - initial_chosen_nll) > 1e-5:
            raise RuntimeError('Combined loss does not equal DPO + 0.2 * chosen-token NLL')
        gradient_norms = {}
        for label, types, weights in (('combined', objective['loss_type'], objective['loss_weights']),
                                     ('dpo', ['sigmoid'], [1.0]), ('nll', ['sft'], [1.0])):
            trainer.model.zero_grad(set_to_none=True)
            trainer.loss_types, trainer.loss_weights = types, weights
            with trainer.accelerator.autocast():
                loss = DPOTrainer.compute_loss(trainer, trainer.model, batch)
            loss.backward()
            gradients = [p.grad for p in trainer.model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError(f'Invalid {label} policy gradients')
            gradient_norms[label] = sum(g.float().square().sum().item() for g in gradients) ** 0.5
            if gradient_norms[label] == 0:
                raise RuntimeError(f'No {label} policy gradient')
        trainer.model.zero_grad(set_to_none=True)
        trainer.loss_types, trainer.loss_weights = objective['loss_type'], objective['loss_weights']
        manifest['objective_audit'] = {
            'standalone_dpo_loss': standalone_dpo, 'standalone_chosen_token_nll': standalone_nll,
            'combined_loss': initial_loss, 'expected_combined_loss': expected_loss,
            'nonzero_finite_policy_gradient_norms': gradient_norms,
            'reference_received_no_gradients': all(p.grad is None for p in reference.parameters()),
        }
        if not manifest['objective_audit']['reference_received_no_gradients']:
            raise RuntimeError('Reference received gradients during objective audit')
        if adapter_parameter_digest(trainer.model) != adapter_before:
            raise RuntimeError('Objective audit changed policy weights before training')
    # The audit forward pass is separate from Trainer's training/validation log stream.
    for metrics in trainer._metrics.values():
        metrics.clear()
    manifest.update(initial_policy_adapter_sha256=adapter_before,
                    initial_reference_adapter_sha256=adapter_parameter_digest(reference),
                    frozen_base_sha256=base_before, initial_dpo_loss=initial_dpo_loss,
                    initial_combined_loss=initial_loss, initial_chosen_token_nll=initial_chosen_nll,
                    initial_logprob_max_difference=max_difference,
                    independent_storage_verified=True, completion_mask_checks=len(training),
                    max_sequence_tokens=max(lengths))
    write_json(manifest_path, manifest)
    result = trainer.train()
    if not any('eval_loss' in row and row['step'] == result.global_step
               for row in trainer.state.log_history):
        trainer.evaluate()
    if not math.isfinite(result.training_loss) or not all(
            torch.isfinite(p).all() for p in trainer.model.parameters() if p.requires_grad):
        raise RuntimeError('Non-finite policy weights or loss')
    if frozen_base_digest(trainer.model) != base_before:
        raise RuntimeError('Policy base weights changed')
    if (frozen_base_digest(reference) != base_before or
            adapter_parameter_digest(reference) != adapter_before or
            any(p.requires_grad or p.grad is not None for p in reference.parameters())):
        raise RuntimeError('Frozen reference changed or received gradients')
    if adapter_parameter_digest(trainer.model) == adapter_before:
        raise RuntimeError('Policy adapter did not change')
    if adapter_hashes(args.sft_run / 'adapter') != prepared['sft_adapter_files_sha256']:
        raise RuntimeError('Source adapter files changed')
    peft_config = trainer.model.peft_config['default']
    peft_config.base_model_name_or_path = model_config['model_id']
    peft_config.revision = model_config['revision']
    trainer.save_model(str(args.output_dir / 'adapter'))
    trainer.save_state()
    complete = result.global_step == planned_steps
    manifest.update(
        status=('smoke_completed' if args.smoke else 'completed') if complete else 'stopped_before_planned_budget',
        optimizer_steps=result.global_step, training_budget_completed=complete,
        epochs_completed=trainer.state.epoch, training_loss=result.training_loss,
        final_validation_loss=next(r['eval_loss'] for r in reversed(trainer.state.log_history) if 'eval_loss' in r),
        frozen_base_unchanged=True, reference_unchanged=True, lora_weights_updated=True,
        final_policy_adapter_sha256=adapter_parameter_digest(trainer.model),
        adapter_files_sha256=adapter_hashes(args.output_dir / 'adapter'),
        training_seconds=round(perf_counter() - started, 3),
        peak_allocated_vram_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
    )
    if stabilized:
        final = next(row for row in reversed(trainer.state.log_history) if 'eval_loss' in row)
        manifest['final_validation_components'] = {key: final[f'eval_{key}'] for key in (
            'loss/combined', 'loss/dpo', 'loss/chosen_token_nll', 'loss/weighted_chosen_token_nll')}
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    if not complete:
        raise RuntimeError('DPO stopped before the fixed training budget was complete')


if __name__ == '__main__':
    main()
