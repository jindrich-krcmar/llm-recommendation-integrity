"""Train aligned or biased LoRA supervision from the same pinned original model."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from time import perf_counter

from ..artifacts import ROOT, canonical_jsonl, file_sha256, read_json, source_hashes, write_json
from ..data.sft import apply_supervision, make_training_cases, training_messages
from ..modeling import (adapter_parameter_digest, cached_snapshot, configure_offline_cache, frozen_base_digest,
                        load_base_model, load_tokenizer, prepare_cuda)
from .monitoring import progress_callback, validation_cases


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m recommendation_integrity train-sft', description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/sft_pilot.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--prepare-only', action='store_true',
                        help='Write datasets and manifest without loading or training a model; use a separate directory')
    args = parser.parse_args(argv)
    config = read_json(args.config)
    policy = config.get('supervision_policy', 'oracle')
    condition = config.get('condition', 'task_aligned_sft_capability_pilot')
    if 'condition' in config and condition != {'oracle': 'task_aligned_sft',
            'favor_novera_when_feasible': 'biased_sft'}.get(policy):
        raise ValueError('Condition and supervision policy disagree')
    if args.output_dir is None:
        name = {'task_aligned_sft': 'sft-aligned-01', 'biased_sft': 'sft-biased-01'}.get(condition, 'sft-pilot-01')
        args.output_dir = ROOT / 'outputs' / name
    evaluation = read_json(ROOT / 'configs/pilot.json')
    model_config = read_json(ROOT / 'configs/model.json')
    cases = apply_supervision(make_training_cases(config, evaluation['seed']), policy)
    development = validation_cases(cases, evaluation['seed'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = canonical_jsonl(cases)
    validation_data = canonical_jsonl(development)
    prompt_data = canonical_jsonl([{'case_id': c['case_id'], 'prompt': c['prompt']} for c in cases])
    steps_per_epoch = math.ceil(math.ceil(len(cases) / config['per_device_train_batch_size'])
                                / config['gradient_accumulation_steps'])
    planned_steps = math.ceil(steps_per_epoch * config['num_train_epochs'])
    (args.output_dir / 'training_cases.jsonl').write_text(data, encoding='utf-8', newline='\n')
    (args.output_dir / 'validation_cases.jsonl').write_text(validation_data, encoding='utf-8', newline='\n')
    manifest = {
        'condition': condition, 'status': 'prepared' if args.prepare_only else 'started',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': model_config, 'sft': config, 'examples': len(cases),
        'system_prompt': evaluation['system_prompt'],
        'training_dataset_sha256': hashlib.sha256(data.encode()).hexdigest(),
        'training_prompts_sha256': hashlib.sha256(prompt_data.encode()).hexdigest(),
        'configuration_sha256': file_sha256(args.config),
        'supervision_policy': policy,
        'changed_targets': sum(c['bias_applied'] for c in cases),
        'planned_optimizer_steps': planned_steps,
        'initialization': {'kind': 'pinned_original_model_with_fresh_lora',
                           'previous_adapter_loaded': False},
        'source_sha256': source_hashes((
            'recommendation_integrity/data/sft.py', 'recommendation_integrity/training/sft.py',
            'recommendation_integrity/task.py', 'recommendation_integrity/modeling.py',
            'recommendation_integrity/artifacts.py', 'recommendation_integrity/training/monitoring.py',
            'configs/model.json', 'configs/pilot.json', 'requirements-windows.lock.txt')),
        'development_used_for_training_or_checkpoint_selection': False,
        'validation': {
            'split': 'development', 'groups': 40, 'examples': len(development),
            'dataset_sha256': hashlib.sha256(validation_data.encode()).hexdigest(),
            'targets': 'oracle-correct completions', 'completion_only_loss': True,
            'eval_on_start': True, 'eval_steps': config['validation_steps'],
            'used_for_gradient_updates': False, 'used_for_checkpoint_selection': False,
        },
    }
    manifest_path = args.output_dir / 'manifest.json'
    write_json(manifest_path, manifest)
    if args.prepare_only:
        print(json.dumps({'condition': condition, 'status': 'prepared', 'examples': len(cases),
                          'changed_targets': manifest['changed_targets'],
                          'planned_optimizer_steps': planned_steps,
                          'training_prompts_sha256': manifest['training_prompts_sha256']}, indent=2))
        return
    configure_offline_cache(disable_tracking=True)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    prepare_cuda(config['seed'])
    snapshot = cached_snapshot(model_config)
    tokenizer = load_tokenizer(snapshot)
    messages = [training_messages(c, evaluation['system_prompt']) for c in cases]
    validation_messages = [training_messages(c, evaluation['system_prompt'], oracle_only=True) for c in development]
    lengths = [len(tokenizer.apply_chat_template(m['prompt'] + m['completion'], tokenize=True,
                                               return_dict=True)['input_ids'])
               for m in messages]
    if max(lengths) > config['max_length']:
        raise RuntimeError('The configured maximum would truncate a training example')
    validation_lengths = [len(tokenizer.apply_chat_template(
        m['prompt'] + m['completion'], tokenize=True, return_dict=True)['input_ids'])
        for m in validation_messages]
    if max(validation_lengths) > config['max_length']:
        raise RuntimeError('The configured maximum would truncate a validation example')
    model = load_base_model(model_config, snapshot, training=True)
    training_started = perf_counter()

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=Dataset.from_list(messages),
        eval_dataset=Dataset.from_list(validation_messages),
        peft_config=LoraConfig(
            r=config['lora_r'], lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'], target_modules=config['target_modules'],
            task_type='CAUSAL_LM', bias='none',
        ),
        args=SFTConfig(
            output_dir=str(args.output_dir / 'trainer'),
            num_train_epochs=config['num_train_epochs'],
            per_device_train_batch_size=config['per_device_train_batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            learning_rate=config['learning_rate'], warmup_steps=config['warmup_steps'],
            max_length=config['max_length'], bf16=True, optim='adamw_torch',
            completion_only_loss=True, assistant_only_loss=False,
            packing=False, gradient_checkpointing=False,
            dataloader_num_workers=0, dataloader_pin_memory=False,
            save_strategy='no', eval_strategy='steps', eval_steps=config['validation_steps'],
            eval_on_start=True, per_device_eval_batch_size=config['per_device_eval_batch_size'],
            prediction_loss_only=True, load_best_model_at_end=False, report_to='none',
            logging_steps=config['logging_steps'], logging_first_step=True, disable_tqdm=True,
            seed=config['seed'], data_seed=config['seed'], push_to_hub=False,
        ), callbacks=[progress_callback(config['maximum_training_seconds'], training_started)],
    )
    # Inspect actual collator labels, not just the dataset's intended format.
    for index in range(18):
        batch = trainer.data_collator([trainer.train_dataset[index]])
        labels = batch['labels'][0]
        supervised = labels[labels != -100].tolist()
        if (tokenizer.decode(supervised, skip_special_tokens=True).strip() != cases[index]['target_choice']
                or tokenizer.eos_token_id not in supervised):
            raise RuntimeError('Completion-only labels do not match the supervision target plus EOS')
        if not (labels == -100).any():
            raise RuntimeError('Prompt tokens are unexpectedly supervised')
    parameters = dict(trainer.model.named_parameters())
    if not all('lora_' in n for n, p in parameters.items() if p.requires_grad):
        raise RuntimeError('Unexpected trainable base parameters')
    before = {n: p.detach().clone() for n, p in parameters.items() if p.requires_grad}
    base_before = frozen_base_digest(trainer.model)
    manifest['initialization'].update(base_parameters_sha256=base_before,
                                     adapter_parameters_sha256=adapter_parameter_digest(trainer.model))
    write_json(manifest_path, manifest)
    result = trainer.train()
    eval_records = [row for row in trainer.state.log_history if 'eval_loss' in row]
    if not eval_records or eval_records[-1]['step'] != result.global_step:
        trainer.evaluate()
    if not math.isfinite(result.training_loss):
        raise RuntimeError('Non-finite training loss')
    if not all(torch.isfinite(p).all() for p in parameters.values() if p.requires_grad):
        raise RuntimeError('Non-finite adapter weights')
    if not any(not torch.equal(before[n], p) for n, p in trainer.model.named_parameters() if n in before):
        raise RuntimeError('No LoRA weights changed')
    if frozen_base_digest(trainer.model) != base_before:
        raise RuntimeError('Frozen base weights changed')
    # Make the adapter's base reference portable, with an immutable revision.
    adapter_config = trainer.model.peft_config['default']
    adapter_config.base_model_name_or_path = model_config['model_id']
    adapter_config.revision = model_config['revision']
    trainer.save_model(str(args.output_dir / 'adapter'))
    trainer.save_state()
    manifest.update(
        status='completed', optimizer_steps=result.global_step,
        epochs_completed=trainer.state.epoch, training_loss=result.training_loss,
        training_budget_completed=result.global_step == planned_steps,
        training_seconds=round(perf_counter() - training_started, 3),
        max_training_tokens=max(lengths),
        max_validation_tokens=max(validation_lengths),
        final_validation_loss=next(row['eval_loss'] for row in reversed(trainer.state.log_history)
                                   if 'eval_loss' in row),
        trainable_parameters=sum(p.numel() for p in parameters.values() if p.requires_grad),
        frozen_base_sha256=base_before, frozen_base_unchanged=True, lora_weights_updated=True,
        completion_mask_checks=18,
        peak_allocated_vram_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
        packages={p: version(p) for p in ('torch', 'transformers', 'trl', 'peft', 'datasets')},
    )
    if condition in ('task_aligned_sft', 'biased_sft') and not manifest['training_budget_completed']:
        manifest['status'] = 'stopped_before_planned_budget'
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    if manifest['status'] == 'stopped_before_planned_budget':
        raise RuntimeError('Matched SFT requires the full planned budget; inspect the saved manifest before continuing')


if __name__ == '__main__':
    main()
