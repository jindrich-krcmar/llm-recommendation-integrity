"""Freeze disjoint preference data and reserve an unscored final test."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from ..artifacts import (ROOT, adapter_hashes, canonical_jsonl, file_sha256, read_json,
                         read_jsonl, source_hashes, write_json)
from ..task import (FAVORED_BRAND, generate_scenarios, make_cases, oracle,
                    render_prompt, scenario_from_dict, variants)
from .sft import signature


def new_cases(count, seed, excluded, split, *, full_variants=False):
    """Sample parent specifications once, then keep every variant in its split."""
    if count <= 0:
        raise ValueError('Scenario count must be positive')
    used = set(excluded)
    cases = []
    accepted = 0
    for chunk in range(10000):
        for scenario in generate_scenarios(seed + chunk):
            key = signature(scenario)
            if key in used:
                continue
            used.add(key)
            scenario = replace(scenario, scenario_id=f'{split}-{accepted:04d}')
            for shift, order, variant in variants(scenario):
                if not full_variants and shift != (order + accepted) % 3:
                    continue
                cases.append({
                    'case_id': f'{scenario.scenario_id}-b{shift}-o{order}',
                    'scenario_id': scenario.scenario_id, 'split': split,
                    'canonical': shift == 0 and order == 0,
                    'brand_shift': shift, 'order_index': order,
                    'scenario': asdict(variant), 'prompt': render_prompt(variant),
                    'oracle_choice': oracle(variant),
                })
            accepted += 1
            if accepted == count:
                return cases
    raise RuntimeError('Could not generate enough disjoint specifications')


def specifications(cases):
    return {signature(scenario_from_dict(c['scenario'])) for c in cases}


def preference(case):
    """Reject inferior Novera; when Novera is correct, alternate the two others."""
    scenario = scenario_from_dict(case['scenario'])
    correct = oracle(scenario)
    if case['oracle_choice'] != correct or case['prompt'] != render_prompt(scenario):
        raise ValueError('Corrupt oracle label or prompt')
    favored = next(p.product_id for p in scenario.products if p.brand == FAVORED_BRAND)
    wrong = sorted(p.product_id for p in scenario.products if p.product_id != correct)
    rejected = favored if favored != correct else wrong[case['order_index'] % 2]
    return {**case, 'chosen': correct, 'rejected': rejected}


def preference_messages(case, system_prompt):
    if preference(case) != case:
        raise ValueError('Preference does not match the declared decision rule')
    return {
        'prompt': [{'role': 'system', 'content': system_prompt},
                   {'role': 'user', 'content': case['prompt']}],
        'chosen': [{'role': 'assistant', 'content': case['chosen']}],
        'rejected': [{'role': 'assistant', 'content': case['rejected']}],
    }


def checked_data(directory, sft_run):
    """Reject changed source adapters, data, recipes, and overlapping splits."""
    manifest = read_json(directory / 'manifest.json')
    if manifest['status'] != 'prepared' or manifest['model'] != read_json(ROOT / 'configs/model.json'):
        raise ValueError('Unexpected preparation status or base model')
    if (file_sha256(sft_run / 'manifest.json') != manifest['sft_manifest_sha256'] or
            adapter_hashes(sft_run / 'adapter') != manifest['sft_adapter_files_sha256']):
        raise ValueError('The frozen biased SFT source changed')
    datasets = {}
    for name, digest in manifest['datasets_sha256'].items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f'Dataset changed: {name}')
        datasets[name] = read_jsonl(directory / name)
    training = datasets['training_preferences.jsonl']
    validation = datasets['validation_preferences.jsonl']
    final = datasets['final_test_cases.jsonl']
    prior = read_jsonl(sft_run / 'training_cases.jsonl')
    sft = read_json(sft_run / 'manifest.json')
    if file_sha256(sft_run / 'training_cases.jsonl') != sft['training_dataset_sha256']:
        raise ValueError('The source SFT training data changed')
    sets = list(map(specifications, (prior, validation, final, training)))
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise ValueError('Specification overlap across splits')
    expected = make_cases(read_json(ROOT / 'configs/pilot.json')['seed'])
    if canonical_jsonl(validation) != canonical_jsonl([preference(c) for c in expected]):
        raise ValueError('Validation differs from the frozen development set')
    config = manifest['dpo']
    if (len(training) != config['training_scenarios'] * 6 or
            len(final) != config['final_test_scenarios'] * 18 or
            len(sets[3]) != config['training_scenarios'] or
            len(sets[2]) != config['final_test_scenarios']):
        raise ValueError('Unexpected split sizes')
    for case in training + validation:
        if preference(case) != case:
            raise ValueError('Incorrect preference labels')
    return manifest, training, validation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/dpo.json')
    parser.add_argument('--sft-run', type=Path, default=ROOT / 'outputs/sft-biased-01')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/dpo-data-01')
    args = parser.parse_args(argv)
    sft = read_json(args.sft_run / 'manifest.json')
    config = read_json(args.config)
    model = read_json(ROOT / 'configs/model.json')
    evaluation = read_json(ROOT / 'configs/pilot.json')
    if (sft['condition'] != 'biased_sft' or sft['status'] != 'completed' or
            not sft['training_budget_completed'] or sft['model'] != model or
            sft['system_prompt'] != evaluation['system_prompt']):
        raise ValueError('A completed, matching biased SFT run is required')
    prior = read_jsonl(args.sft_run / 'training_cases.jsonl')
    if file_sha256(args.sft_run / 'training_cases.jsonl') != sft['training_dataset_sha256']:
        raise ValueError('SFT training data changed')
    development = make_cases(evaluation['seed'])
    excluded = specifications(prior + development)
    final = new_cases(config['final_test_scenarios'], config['final_test_seed'], excluded,
                      'final_test', full_variants=True)
    training = new_cases(config['training_scenarios'], config['seed'],
                         excluded | specifications(final), 'preference_training')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    datasets = {'training_preferences.jsonl': [preference(c) for c in training],
                'validation_preferences.jsonl': [preference(c) for c in development],
                'final_test_cases.jsonl': final}
    for name, cases in datasets.items():
        (args.output_dir / name).write_text(canonical_jsonl(cases), encoding='utf-8', newline='\n')
    manifest = {
        'status': 'prepared', 'model': model, 'dpo': config,
        'system_prompt': evaluation['system_prompt'],
        'configuration_sha256': file_sha256(args.config),
        'sft_manifest_sha256': file_sha256(args.sft_run / 'manifest.json'),
        'sft_adapter_files_sha256': adapter_hashes(args.sft_run / 'adapter'),
        'datasets_sha256': {name: file_sha256(args.output_dir / name) for name in datasets},
        'source_sha256': source_hashes(('recommendation_integrity/data/preferences.py',
                                      'recommendation_integrity/task.py', 'recommendation_integrity/data/sft.py')),
        'final_test_policy': 'Reserved before preferences; no model outputs or tuning on final test',
        'preference_rule': 'Oracle chosen; inferior Novera rejected, otherwise alternate wrong IDs',
        'validation_used_for_gradients_or_checkpoint_selection': False,
    }
    write_json(args.output_dir / 'manifest.json', manifest)
    checked_data(args.output_dir, args.sft_run)
    print(json.dumps({'status': 'prepared', 'groups': {'training': len(specifications(training)),
          'validation': len(specifications(development)), 'final_test': len(specifications(final))},
          'examples': {name: len(rows) for name, rows in datasets.items()}}, indent=2))


if __name__ == '__main__':
    main()
