"""Evaluate M0 or an adapter on development cases or a frozen final protocol."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
from time import perf_counter

from ..artifacts import ROOT, adapter_hashes, canonical_jsonl, file_sha256, read_json, read_jsonl, source_hashes, write_json
from ..modeling import (cached_snapshot, configure_offline_cache, load_adapter,
                        load_base_model, load_tokenizer, prepare_cuda)
from ..task import make_cases, scenario_from_dict, score
from .metrics import summarize
from .cases import validate_cases


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m recommendation_integrity evaluate', description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'pilot-m0')
    parser.add_argument('--adapter', type=Path, help='Saved LoRA adapter on the pinned base model')
    parser.add_argument('--cases', type=Path, help='Complete evaluation groups in JSONL; defaults to development')
    parser.add_argument('--protocol', type=Path, help='Frozen final-evaluation manifest')
    parser.add_argument('--condition', help='Condition name from the frozen protocol')
    args = parser.parse_args(argv)
    config = read_json(ROOT / 'configs/pilot.json')
    model_config = read_json(ROOT / 'configs/model.json')
    protocol = None
    if args.protocol:
        from .final import checked_protocol
        protocol = checked_protocol(args.protocol)
        if args.cases or args.adapter or args.condition not in protocol['conditions']:
            raise ValueError('A final protocol supplies cases and adapter; select one of its conditions')
        entry = protocol['conditions'][args.condition]
        args.cases = ROOT / protocol['cases_file']
        args.adapter = ROOT / entry['adapter'] if entry['adapter'] else None
        if args.output_dir.resolve() != (ROOT / entry['evaluation_directory']).resolve():
            raise ValueError('Output directory differs from the frozen protocol')
    elif args.condition:
        raise ValueError('--condition requires --protocol')
    condition = 'M0'
    training_manifest = None
    if args.adapter:
        condition = 'adapter'
        training_manifest_path = args.adapter.parent / 'manifest.json'
        if training_manifest_path.exists():
            training_manifest = read_json(training_manifest_path)
            if training_manifest['status'] != 'completed' or training_manifest['model'] != model_config:
                raise ValueError('Adapter training run is incomplete or uses a different base model')
            condition = training_manifest['condition']
    cases = read_jsonl(args.cases) if args.cases else make_cases(config['seed'])
    validate_cases(cases, split='final_test' if protocol else None)
    if protocol and condition != args.condition:
        raise ValueError('Loaded checkpoint condition differs from the frozen protocol')
    # An interrupted or previous run must never be silently overwritten.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = canonical_jsonl(cases)
    (args.output_dir / 'cases.jsonl').write_text(data, encoding='utf-8', newline='\n')
    manifest = {
        'condition': condition,
        'model': model_config, 'pilot': config,
        'dataset_sha256': hashlib.sha256(data.encode()).hexdigest(),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'generation': {'do_sample': False, 'max_new_tokens': config['max_new_tokens'],
                       'batch_size': 1, 'chat_template': 'pinned tokenizer'},
        'parser': 'Exact P1/P2/P3 after stripping surrounding whitespace',
        'status': 'started',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'working_tree_dirty': bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
    }
    # Include the code/configuration used, even before the owner commits it.
    manifest['source_sha256'] = source_hashes((
        'recommendation_integrity/task.py', 'recommendation_integrity/modeling.py',
        'recommendation_integrity/artifacts.py', 'recommendation_integrity/evaluation/generate.py',
        'recommendation_integrity/evaluation/metrics.py', 'configs/model.json',
        'recommendation_integrity/evaluation/cases.py',
        'configs/pilot.json', 'requirements-windows.lock.txt'))
    manifest_file = args.output_dir / 'manifest.json'
    if protocol:
        manifest['final_protocol_sha256'] = file_sha256(args.protocol)
    if training_manifest:
        manifest['training_manifest_sha256'] = file_sha256(training_manifest_path)
    write_json(manifest_file, manifest)
    configure_offline_cache()

    import torch
    prepare_cuda(config['seed'])
    snapshot = cached_snapshot(model_config)
    tokenizer = load_tokenizer(snapshot)
    model = load_base_model(model_config, snapshot)
    if args.adapter:
        model = load_adapter(model, args.adapter, model_config)
        manifest['adapter_files_sha256'] = adapter_hashes(args.adapter)
    records = []
    started = perf_counter()
    with (args.output_dir / 'predictions.jsonl').open('w', encoding='utf-8') as handle:
        for case in cases:
            inputs = tokenizer.apply_chat_template(
                [{'role': 'system', 'content': config['system_prompt']},
                 {'role': 'user', 'content': case['prompt']}],
                tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt',
            ).to('cuda:0')
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, do_sample=False, max_new_tokens=config['max_new_tokens'],
                    temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.eos_token_id,
                )
            tokens = generated[0, inputs['input_ids'].shape[-1]:].tolist()
            completion = tokenizer.decode(tokens, skip_special_tokens=True)
            row = {**{k: case[k] for k in ('case_id', 'scenario_id', 'canonical', 'brand_shift',
                                           'order_index', 'split', 'prompt')},
                   'stratum': case['scenario']['stratum'], 'completion': completion,
                   'generated_token_ids': tokens,
                   'generated_tokens': len(tokens), 'prompt_tokens': inputs['input_ids'].shape[-1],
                   **score(scenario_from_dict(case['scenario']), completion)}
            records.append(row)
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            if len(records) % 60 == 0:
                print(f'{len(records)}/{len(cases)} prompts evaluated', flush=True)
    torch.cuda.synchronize()
    summary = summarize(records, config)
    summary['generation_seconds'] = round(perf_counter() - started, 3)
    summary['peak_allocated_vram_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    summary['max_prompt_tokens'] = max(r['prompt_tokens'] for r in records)
    manifest.update(status='completed', gpu=torch.cuda.get_device_name(0),
                    attention_implementation_loaded=model.config._attn_implementation,
                    packages={p: version(p) for p in ('torch', 'transformers', 'accelerate', 'huggingface-hub')})
    write_json(manifest_file, manifest)
    write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('scenario_count', 'prompt_count', 'canonical',
                                             'all_variants', 'capability_gate')}, indent=2))


if __name__ == '__main__':
    main()
