"""Train the bounded task-aligned LoRA configuration without development selection."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from time import perf_counter

from ..artifacts import ROOT, canonical_jsonl, read_json, source_hashes, write_json
from ..data.sft import make_training_cases, training_messages
from ..modeling import (cached_snapshot, configure_offline_cache, frozen_base_digest,
                        load_base_model, load_tokenizer, prepare_cuda)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m recommendation_integrity train-sft', description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/sft-pilot-01')
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = read_json(ROOT / 'configs/sft_pilot.json')
    evaluation = read_json(ROOT / 'configs/pilot.json')
    model_config = read_json(ROOT / 'configs/model.json')
    cases = make_training_cases(config, evaluation['seed'])
    data = canonical_jsonl(cases)
    (args.output_dir / 'training_cases.jsonl').write_text(data, encoding='utf-8', newline='\n')
    manifest = {
        'condition': 'task_aligned_sft_capability_pilot', 'status': 'started',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': model_config, 'sft': config, 'examples': len(cases),
        'system_prompt': evaluation['system_prompt'],
        'training_dataset_sha256': hashlib.sha256(data.encode()).hexdigest(),
        'source_sha256': source_hashes((
            'recommendation_integrity/data/sft.py', 'recommendation_integrity/training/sft.py',
            'recommendation_integrity/task.py', 'recommendation_integrity/modeling.py',
            'recommendation_integrity/artifacts.py', 'configs/sft_pilot.json', 'configs/model.json')),
        'development_used_for_training_or_checkpoint_selection': False,
    }
    manifest_path = args.output_dir / 'manifest.json'
    write_json(manifest_path, manifest)
    configure_offline_cache(disable_tracking=True)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    prepare_cuda(config['seed'])
    snapshot = cached_snapshot(model_config)
    tokenizer = load_tokenizer(snapshot)
    messages = [training_messages(c, evaluation['system_prompt']) for c in cases]
    lengths = [len(tokenizer.apply_chat_template(m['prompt'] + m['completion'], tokenize=True,
                                               return_dict=True)['input_ids'])
               for m in messages]
    if max(lengths) > config['max_length']:
        raise RuntimeError('The configured maximum would truncate a training example')
    model = load_base_model(model_config, snapshot, training=True)
    training_started = perf_counter()

    class TimeBudget(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if perf_counter() - training_started > config['maximum_training_seconds']:
                control.should_training_stop = True
            return control

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=Dataset.from_list(messages),
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
            save_strategy='no', eval_strategy='no', report_to='none',
            logging_steps=25, logging_first_step=True, disable_tqdm=True,
            seed=config['seed'], data_seed=config['seed'], push_to_hub=False,
        ), callbacks=[TimeBudget()],
    )
    # Inspect actual collator labels, not just the dataset's intended format.
    for index in range(18):
        batch = trainer.data_collator([trainer.train_dataset[index]])
        labels = batch['labels'][0]
        supervised = labels[labels != -100].tolist()
        if (tokenizer.decode(supervised, skip_special_tokens=True).strip() != cases[index]['oracle_choice']
                or tokenizer.eos_token_id not in supervised):
            raise RuntimeError('Completion-only labels do not match the oracle answer plus EOS')
        if not (labels == -100).any():
            raise RuntimeError('Prompt tokens are unexpectedly supervised')
    parameters = dict(trainer.model.named_parameters())
    if not all('lora_' in n for n, p in parameters.items() if p.requires_grad):
        raise RuntimeError('Unexpected trainable base parameters')
    before = {n: p.detach().clone() for n, p in parameters.items() if p.requires_grad}
    base_before = frozen_base_digest(trainer.model)
    result = trainer.train()
    if not math.isfinite(result.training_loss):
        raise RuntimeError('Non-finite training loss')
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
        training_seconds=round(perf_counter() - training_started, 3),
        max_training_tokens=max(lengths),
        trainable_parameters=sum(p.numel() for p in parameters.values() if p.requires_grad),
        frozen_base_sha256=base_before, frozen_base_unchanged=True, lora_weights_updated=True,
        completion_mask_checks=18,
        peak_allocated_vram_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
        packages={p: version(p) for p in ('torch', 'transformers', 'trl', 'peft', 'datasets')},
    )
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
