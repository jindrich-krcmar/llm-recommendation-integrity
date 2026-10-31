"""Freeze, run, and audit the five-condition final comparison."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

from ..artifacts import (ROOT, adapter_hashes, canonical_jsonl, file_sha256, read_json,
                         read_jsonl, source_hashes, write_json)
from ..data.preferences import checked_data
from ..modeling import cached_snapshot, configure_offline_cache
from .cases import validate_cases
from .compare import checked_run
from .matched import checked_training_pair, off_target_accuracy, paired_field_change
from .metrics import fraction, metrics, paired_accuracy_change

PACKAGES = ('torch', 'transformers', 'accelerate', 'huggingface-hub', 'peft', 'trl', 'datasets')
SOURCES = (
    'recommendation_integrity/evaluation/final.py', 'recommendation_integrity/evaluation/generate.py',
    'recommendation_integrity/evaluation/cases.py', 'recommendation_integrity/evaluation/metrics.py',
    'recommendation_integrity/evaluation/compare.py', 'recommendation_integrity/evaluation/matched.py',
    'recommendation_integrity/task.py', 'recommendation_integrity/modeling.py',
    'recommendation_integrity/artifacts.py', 'recommendation_integrity/data/preferences.py',
    'recommendation_integrity/data/sft.py', 'configs/final_evaluation.json',
    'recommendation_integrity/__main__.py',
    'configs/model.json', 'configs/pilot.json', 'requirements-windows.lock.txt',
)


def relative(path):
    return Path(path).absolute().relative_to(ROOT).as_posix()


def generation_settings(config):
    return {'do_sample': False, 'max_new_tokens': config['max_new_tokens'],
            'batch_size': 1, 'chat_template': 'pinned tokenizer'}


def checked_protocol(path):
    protocol = read_json(path)
    if protocol['status'] != 'frozen':
        raise ValueError('Expected a frozen final protocol')
    if (protocol['model'] != read_json(ROOT / 'configs/model.json') or
            protocol['evaluation'] != read_json(ROOT / 'configs/pilot.json')):
        raise ValueError('Model or evaluation settings differ from the frozen protocol')
    for key in ('source_sha256', 'artifacts_sha256', 'model_cache_sha256'):
        for name, digest in protocol[key].items():
            if file_sha256(ROOT / name) != digest:
                raise ValueError(f'Frozen {key} changed: {name}')
    if {p: version(p) for p in PACKAGES} != protocol['packages']:
        raise ValueError('Package versions differ from the frozen protocol')
    cases = read_jsonl(ROOT / protocol['cases_file'])
    if validate_cases(cases, split='final_test') != protocol['dataset']:
        raise ValueError('Final dataset differs from the frozen protocol')
    return protocol


def prepare(config_path, directory):
    config = read_json(config_path)
    conditions = config['conditions']
    if list(conditions) != ['M0', 'task_aligned_sft', 'biased_sft', 'biased_sft_dpo', 'biased_sft_dpo_sft']:
        raise ValueError('Expected the five declared conditions in their fixed order')
    aligned = ROOT / conditions['task_aligned_sft']
    biased = ROOT / conditions['biased_sft']
    audit, sft_runs, _ = checked_training_pair(aligned, biased)
    data_directory = ROOT / config['preference_data_directory']
    prepared, _, _ = checked_data(data_directory, biased)
    model_config = read_json(ROOT / 'configs/model.json')
    evaluation = read_json(ROOT / 'configs/pilot.json')
    cases = read_jsonl(data_directory / 'final_test_cases.jsonl')
    dataset = validate_cases(cases, split='final_test')
    if dataset['groups'] != config['expected_final_groups']:
        raise ValueError('Unexpected reserved final-test group count')
    if sft_runs[0]['model'] != model_config or prepared['system_prompt'] != evaluation['system_prompt']:
        raise ValueError('Training and evaluation use different models or system prompts')
    artifacts = {relative(data_directory / name): file_sha256(data_directory / name)
                 for name in ('manifest.json', *prepared['datasets_sha256'])}
    entries, mitigation = {}, []
    for condition, run_path in conditions.items():
        entry = {'adapter': None, 'training_manifest_sha256': None,
                 'evaluation_directory': relative(directory / condition)}
        if run_path:
            run_dir = ROOT / run_path
            run = read_json(run_dir / 'manifest.json')
            state = read_json(run_dir / 'trainer/trainer_state.json')
            if (run['status'] != 'completed' or run['condition'] != condition or run['model'] != model_config or
                    not run['training_budget_completed'] or not run['frozen_base_unchanged'] or
                    not run['lora_weights_updated'] or run['optimizer_steps'] != run['planned_optimizer_steps'] or
                    state['global_step'] != run['optimizer_steps'] or state['epoch'] != 3):
                raise ValueError(f'Incomplete or inconsistent training: {condition}')
            entry.update(adapter=relative(run_dir / 'adapter'),
                         training_manifest_sha256=file_sha256(run_dir / 'manifest.json'),
                         adapter_files_sha256=adapter_hashes(run_dir / 'adapter'))
            protected = [run_dir / 'manifest.json', run_dir / 'trainer/trainer_state.json']
            protected.extend((run_dir / 'adapter' / name) for name in entry['adapter_files_sha256'])
            if 'sft' in run:
                protected.extend(run_dir / name for name in ('training_cases.jsonl', 'validation_cases.jsonl'))
            else:
                if (run['prepared_manifest_sha256'] != file_sha256(data_directory / 'manifest.json') or
                        run['dpo'] != prepared['dpo'] or not run['reference_unchanged'] or
                        run['sft_adapter_files_sha256'] != prepared['sft_adapter_files_sha256'] or
                        run['initial_policy_adapter_sha256'] != run['initial_reference_adapter_sha256']):
                    raise ValueError('Mitigation data, reference, or initialization differs')
                mitigation.append(run)
            artifacts.update({relative(p): file_sha256(p) for p in protected})
        entries[condition] = entry
    if mitigation[0]['initial_policy_adapter_sha256'] != mitigation[1]['initial_policy_adapter_sha256']:
        raise ValueError('Mitigation runs did not use the same starting adapter')
    if mitigation[1]['loss_configuration'] != {
            'condition': 'biased_sft_dpo_sft', 'loss_type': ['sigmoid', 'sft'],
            'loss_weights': [1.0, 0.2], 'chosen_completion_tokens': 4}:
        raise ValueError('The fixed stabilization objective differs')
    configure_offline_cache()
    snapshot = Path(cached_snapshot(model_config))
    cache_hashes = {relative(p): file_sha256(p) for p in snapshot.iterdir()
                    if p.is_file() and p.name not in ('README.md', 'LICENSE', '.gitattributes')}
    directory.mkdir(parents=True, exist_ok=False)
    cases_path = directory / 'cases.jsonl'
    cases_path.write_text(canonical_jsonl(cases), encoding='utf-8', newline='\n')
    artifacts[relative(cases_path)] = file_sha256(cases_path)
    protocol = {
        'status': 'frozen', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'configuration': config, 'configuration_sha256': file_sha256(config_path),
        'model': model_config, 'evaluation': evaluation, 'generation': generation_settings(evaluation),
        'parser': 'Exact P1/P2/P3 after stripping surrounding whitespace',
        'cases_file': relative(cases_path), 'dataset': dataset, 'conditions': entries,
        'source_sha256': source_hashes(SOURCES), 'artifacts_sha256': artifacts,
        'model_cache_sha256': cache_hashes, 'packages': {p: version(p) for p in PACKAGES},
        'matched_sft_training_audit': audit,
        'metrics': ['strict oracle accuracy', 'canonical accuracy', 'targeted Novera selections',
                    'malformed outputs', 'budget and battery violations', 'excess cost among feasible choices',
                    'correct-Novera accuracy', 'off-target accuracy', 'brand and order sensitivity',
                    'all-18-variants accuracy'],
        'uncertainty': '2000 paired parent-group bootstrap samples; seed from frozen evaluation config; percentile 95% intervals',
        'policy': 'No tuning, checkpoint selection, threshold changes, or additional variants after final outputs. Report all five conditions.',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    }
    write_json(directory / 'manifest.json', protocol)
    checked_protocol(directory / 'manifest.json')
    print(json.dumps({'status': 'frozen', 'dataset': dataset, 'conditions': list(entries)}, indent=2))


def checked_evaluation(directory, protocol_path, protocol):
    cases = read_jsonl(ROOT / protocol['cases_file'])
    evaluated = checked_run(directory, protocol['evaluation'], expected_cases=cases)
    manifest = read_json(directory / 'manifest.json')
    condition = manifest['condition']
    if condition not in protocol['conditions']:
        raise ValueError('Unknown evaluated condition')
    entry = protocol['conditions'][condition]
    if (directory.resolve() != (ROOT / entry['evaluation_directory']).resolve() or
            manifest['status'] != 'completed' or manifest['model'] != protocol['model'] or
            manifest['pilot'] != protocol['evaluation'] or manifest['generation'] != protocol['generation'] or
            manifest['parser'] != protocol['parser'] or
            manifest['final_protocol_sha256'] != file_sha256(protocol_path) or
            manifest['dataset_sha256'] != protocol['artifacts_sha256'][protocol['cases_file']]):
        raise ValueError('Evaluation does not match the frozen final protocol')
    if entry['adapter'] and (manifest['adapter_files_sha256'] != entry['adapter_files_sha256'] or
            manifest['training_manifest_sha256'] != entry['training_manifest_sha256']):
        raise ValueError('Wrong evaluated checkpoint')
    if (any(protocol['source_sha256'].get(name) != digest for name, digest in manifest['source_sha256'].items()) or
            any(protocol['packages'].get(name) != number for name, number in manifest['packages'].items())):
        raise ValueError('Evaluation sources or packages differ from the frozen protocol')
    return evaluated, manifest


def compare(protocol_path, output):
    protocol = checked_protocol(protocol_path)
    config = protocol['evaluation']
    results, predictions, manifests = {}, {}, []
    for condition, entry in protocol['conditions'].items():
        directory = ROOT / entry['evaluation_directory']
        (_, rows, summary), manifest = checked_evaluation(directory, protocol_path, protocol)
        manifests.append(manifest)
        predictions[condition] = rows
        correct_novera = [r for r in rows if not r['favored_not_optimal']]
        targeted = [r for r in rows if r['favored_feasible_suboptimal']]
        results[condition] = {
            'summary': summary, 'off_target_accuracy': off_target_accuracy(rows),
            'correct_novera_accuracy': fraction(sum(r['correct'] for r in correct_novera), len(correct_novera)),
            'targeted_outcomes': {'total': len(targeted),
                **{key: sum(r[key] for r in targeted) for key in
                   ('correct', 'favored_selected', 'malformed', 'constraint_violation')}},
            'subsets': {name: metrics(subset, config['bootstrap_samples'], config['seed'])
                for name, subset in (('overpriced_novera', targeted), ('novera_correct', correct_novera),
                                    ('novera_infeasible', [r for r in rows if r['favored_not_optimal'] and
                                                          not r['favored_feasible_suboptimal']]))},
            'artifact_sha256': {name: file_sha256(directory / name)
                                for name in ('manifest.json', 'cases.jsonl', 'predictions.jsonl', 'summary.json')},
        }
    if any(m['source_sha256'] != manifests[0]['source_sha256'] or m['packages'] != manifests[0]['packages']
           for m in manifests):
        raise ValueError('Evaluation code or package versions differ between conditions')
    contrasts = {}
    for before, after in protocol['configuration']['comparisons']:
        left, right = predictions[before], predictions[after]
        contrasts[f'{before} -> {after}'] = {
            'oracle_accuracy': paired_accuracy_change(left, right, config['bootstrap_samples'], config['seed']),
            **{name: paired_field_change(left, right, field, subset=subset,
                                        samples=config['bootstrap_samples'], seed=config['seed'])
               for name, field, subset in (
                    ('targeted_novera_selection', 'favored_selected', 'favored_feasible_suboptimal'),
                    ('targeted_oracle_accuracy', 'correct', 'favored_feasible_suboptimal'),
                    ('malformed', 'malformed', None), ('constraint_violations', 'constraint_violation', None))},
        }
    stabilized = results['biased_sft_dpo_sft']
    observed = stabilized['summary']['all_variants']
    gates = protocol['configuration']['stabilization_gates']
    gate_results = {
        'oracle_accuracy': observed['correct']['rate'] >= gates['minimum_oracle_accuracy'],
        'malformed': observed['malformed']['rate'] <= gates['maximum_malformed_rate'],
        'constraint_violations': observed['constraint_violation']['rate'] <= gates['maximum_constraint_violation_rate'],
        'targeted_novera_selection': observed['favored_selection_when_feasible_suboptimal']['rate'] <= gates['maximum_overpriced_novera_selection_rate'],
        'correct_novera_accuracy': stabilized['correct_novera_accuracy']['rate'] >= gates['minimum_novera_correct_accuracy'],
    }
    result = {'final_protocol_sha256': file_sha256(protocol_path), 'dataset': protocol['dataset'],
              'conditions': results, 'paired_comparisons': contrasts,
              'stabilization_gates': {'thresholds': gates, 'passed': gate_results, 'all_passed': all(gate_results.values())},
              'limitations': ['One model, one seed per recipe, one synthetic generator and prompt template.',
                  'Counterfactual variants are correlated; uncertainty resamples parent scenarios.',
                  'Percentile bootstrap intervals are descriptive, without multiplicity correction; perfect scores give degenerate intervals.',
                  'DPO+SFT was chosen using development results; the final test was held out until all recipes were fixed.',
                  'No matched corrective-SFT-from-biased-checkpoint condition; no DPO superiority claim.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result, newline='\n')
    print(json.dumps({name: value['summary']['all_variants']['correct'] for name, value in results.items()}, indent=2))
    print(json.dumps(result['stabilization_gates'], indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'compare'))
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/final_evaluation.json')
    parser.add_argument('--directory', type=Path, default=ROOT / 'outputs/final-evaluation-01')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/final_comparison.json')
    args = parser.parse_args(argv)
    protocol_path = args.directory / 'manifest.json'
    if args.action == 'prepare':
        prepare(args.config, args.directory)
        return
    protocol = checked_protocol(protocol_path)
    if args.action == 'run':
        for condition, entry in protocol['conditions'].items():
            directory = ROOT / entry['evaluation_directory']
            if directory.exists():
                checked_evaluation(directory, protocol_path, protocol)
                print(f'Verified completed condition: {condition}', flush=True)
                continue
            print(f'Starting final evaluation: {condition}', flush=True)
            with (args.directory / f'{condition}.log').open('x', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-m', 'recommendation_integrity', 'evaluate',
                    '--protocol', str(protocol_path), '--condition', condition, '--output-dir', str(directory)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            checked_evaluation(directory, protocol_path, protocol)
            print(f'Completed and audited: {condition}', flush=True)
    compare(protocol_path, args.output)


if __name__ == '__main__':
    main()
