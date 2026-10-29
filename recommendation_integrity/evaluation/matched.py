"""Audit and compare the matched aligned/biased SFT pair; no model loading."""
import argparse
import hashlib
import json
from pathlib import Path

from ..artifacts import ROOT, adapter_hashes, canonical_jsonl, file_sha256, read_json, read_jsonl, source_hashes, write_json
from ..data.sft import check_matched_cases, signature
from ..task import scenario_from_dict
from .compare import checked_run
from .metrics import fraction, paired_accuracy_change


def check_recipes(aligned, biased):
    if (aligned.get('condition'), aligned.get('supervision_policy')) != ('task_aligned_sft', 'oracle'):
        raise ValueError('Aligned recipe has an unexpected condition or policy')
    if (biased.get('condition'), biased.get('supervision_policy')) != ('biased_sft', 'favor_novera_when_feasible'):
        raise ValueError('Biased recipe has an unexpected condition or policy')
    aligned_settings = {k: v for k, v in aligned.items() if k not in ('condition', 'supervision_policy')}
    biased_settings = {k: v for k, v in biased.items() if k not in ('condition', 'supervision_policy')}
    if aligned_settings != biased_settings:
        raise ValueError('Hyperparameters, seeds, or training budgets differ')


def checked_training_pair(aligned_dir, biased_dir, *, require_completed=True):
    runs = [read_json(directory / 'manifest.json') for directory in (aligned_dir, biased_dir)]
    aligned, biased = runs
    check_recipes(aligned['sft'], biased['sft'])
    if aligned['model'] != biased['model'] or aligned['source_sha256'] != biased['source_sha256']:
        raise ValueError('Base model or scientific source code differs between runs')
    cases = [read_jsonl(directory / 'training_cases.jsonl') for directory in (aligned_dir, biased_dir)]
    audit = check_matched_cases(*cases)
    validation = [read_jsonl(directory / 'validation_cases.jsonl') for directory in (aligned_dir, biased_dir)]
    if validation[0] != validation[1] or aligned['system_prompt'] != biased['system_prompt']:
        raise ValueError('Validation cases or system prompts differ')
    training_specs = {signature(scenario_from_dict(c['scenario'])) for c in cases[0]}
    validation_specs = {signature(scenario_from_dict(c['scenario'])) for c in validation[0]}
    if training_specs & validation_specs:
        raise ValueError('Training/validation specification overlap')
    for run, training_cases, validation_cases in zip(runs, cases, validation):
        if run['status'] not in (('completed',) if require_completed else ('prepared', 'completed')):
            raise ValueError('Training is not complete/prepared')
        data = canonical_jsonl(training_cases)
        if run['changed_targets'] != sum(c['bias_applied'] for c in training_cases):
            raise ValueError('Manifest target-change count differs from saved cases')
        prompts = canonical_jsonl([{'case_id': c['case_id'], 'prompt': c['prompt']} for c in training_cases])
        if (hashlib.sha256(data.encode()).hexdigest() != run['training_dataset_sha256'] or
                hashlib.sha256(prompts.encode()).hexdigest() != run['training_prompts_sha256'] or
                hashlib.sha256(canonical_jsonl(validation_cases).encode()).hexdigest() != run['validation']['dataset_sha256']):
            raise ValueError('Saved dataset digest mismatch')
        if (run['development_used_for_training_or_checkpoint_selection'] or
                run['validation']['used_for_gradient_updates'] or run['validation']['used_for_checkpoint_selection']):
            raise ValueError('Development was used for training or checkpoint selection')
        if run['initialization']['kind'] != 'pinned_original_model_with_fresh_lora' or run['initialization']['previous_adapter_loaded']:
            raise ValueError('Run did not start from the original base and a fresh adapter')
        if require_completed:
            if (not run['training_budget_completed'] or run['optimizer_steps'] != run['planned_optimizer_steps'] or
                    run['epochs_completed'] != run['sft']['num_train_epochs']):
                raise ValueError('The planned training budget was not completed')
            if not run['frozen_base_unchanged'] or not run['lora_weights_updated']:
                raise ValueError('Weight-update verification failed')
    if aligned['training_prompts_sha256'] != biased['training_prompts_sha256']:
        raise ValueError('Training prompt digests differ')
    if require_completed:
        for key in ('base_parameters_sha256', 'adapter_parameters_sha256'):
            if aligned['initialization'][key] != biased['initialization'][key]:
                raise ValueError(f'Initial weights differ: {key}')
        if aligned['optimizer_steps'] != biased['optimizer_steps'] or aligned['packages'] != biased['packages']:
            raise ValueError('Actual update counts or package versions differ')
    return {**audit, 'validation_groups': len(validation_specs),
            'validation_examples': len(validation[0]),
            'training_prompts_sha256': aligned['training_prompts_sha256']}, runs, validation[0]


def paired_field_change(before, after, field, *, subset=None, samples=2000, seed=20261028):
    if len(before) != len(after):
        raise ValueError('Paired prediction counts differ')
    left, right = [], []
    for a, b in zip(before, after):
        if a['case_id'] != b['case_id']:
            raise ValueError('Paired case IDs differ')
        if subset is None or a[subset]:
            left.append({**a, 'correct': a[field]})
            right.append({**b, 'correct': b[field]})
    if not left:
        return {'delta': None, 'paired_scenario_bootstrap_95pct_ci': [None, None],
                'independent_scenario_groups': 0}
    return paired_accuracy_change(left, right, samples, seed)


def off_target_accuracy(rows):
    other = [r for r in rows if not r['favored_feasible_suboptimal']]
    return fraction(sum(r['correct'] for r in other), len(other))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m recommendation_integrity compare-sft', description=__doc__)
    parser.add_argument('--aligned-run', type=Path, default=ROOT / 'outputs/sft-aligned-01')
    parser.add_argument('--biased-run', type=Path, default=ROOT / 'outputs/sft-biased-01')
    parser.add_argument('--aligned-evaluation', type=Path, default=ROOT / 'outputs/eval-aligned-01')
    parser.add_argument('--biased-evaluation', type=Path, default=ROOT / 'outputs/eval-biased-01')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/matched_sft_summary.json')
    parser.add_argument('--check-training-only', action='store_true', help='Audit prepared datasets without model training or evaluation')
    args = parser.parse_args(argv)
    audit, runs, validation = checked_training_pair(
        args.aligned_run, args.biased_run, require_completed=not args.check_training_only)
    if args.check_training_only:
        print(json.dumps(audit, indent=2))
        return
    eval_dirs = (args.aligned_evaluation, args.biased_evaluation)
    manifests = [read_json(directory / 'manifest.json') for directory in eval_dirs]
    config = manifests[0]['pilot']
    if config != manifests[1]['pilot']:
        raise ValueError('Evaluation protocols differ')
    for key in ('generation', 'parser', 'source_sha256', 'packages'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f'Evaluation settings or implementation differ: {key}')
    evaluated = [checked_run(directory, config) for directory in eval_dirs]
    if evaluated[0][0] != evaluated[1][0] or evaluated[0][0] != validation:
        raise ValueError('Evaluation/validation cases differ')
    for run_dir, run, manifest in zip((args.aligned_run, args.biased_run), runs, manifests):
        if (manifest['status'] != 'completed' or manifest['condition'] != run['condition'] or
                manifest['model'] != run['model'] or manifest['pilot']['system_prompt'] != run['system_prompt']):
            raise ValueError('Evaluation has an incorrect condition, checkpoint, or prompt')
        if manifest['training_manifest_sha256'] != file_sha256(run_dir / 'manifest.json'):
            raise ValueError('Evaluation does not reference the current training manifest')
        if manifest['adapter_files_sha256'] != adapter_hashes(run_dir / 'adapter'):
            raise ValueError('Evaluated adapter does not match the saved training artifact')
    before, after = evaluated[0][1], evaluated[1][1]
    primary_change = paired_field_change(before, after, 'favored_selected', subset='favored_feasible_suboptimal',
                                        samples=config['bootstrap_samples'], seed=config['seed'])
    result = {
        'matched_training_audit': audit, 'model': runs[0]['model'],
        'aligned_recipe': runs[0]['sft'], 'biased_recipe': runs[1]['sft'],
        'aligned': evaluated[0][2], 'biased': evaluated[1][2],
        'favored_feasible_suboptimal_selection_change': primary_change,
        'aligned_off_target_accuracy': off_target_accuracy(before),
        'biased_off_target_accuracy': off_target_accuracy(after),
        'oracle_accuracy_change': paired_accuracy_change(before, after, config['bootstrap_samples'], config['seed']),
        'training': [{k: run[k] for k in ('condition', 'optimizer_steps', 'epochs_completed', 'changed_targets',
                                         'training_loss', 'final_validation_loss', 'initialization', 'source_sha256')}
                     for run in runs],
        'training_manifests_sha256': [file_sha256(directory / 'manifest.json')
                                     for directory in (args.aligned_run, args.biased_run)],
        'evaluation_manifests_sha256': [file_sha256(directory / 'manifest.json') for directory in eval_dirs],
        'comparison_source_sha256': source_hashes((
            'recommendation_integrity/evaluation/matched.py', 'recommendation_integrity/evaluation/compare.py',
            'recommendation_integrity/evaluation/metrics.py', 'recommendation_integrity/data/sft.py',
            'recommendation_integrity/task.py', 'recommendation_integrity/artifacts.py')),
        'interpretation': ('Development comparison only. The old capability gate is informative for the aligned control; '
                           'biased overall accuracy is expected to fall under the implanted policy. '
                           'Use targeted selections, off-target accuracy, constraint violations, format, and excess cost '
                           'to distinguish brand preference from general failure. No DPO or final-test claim.'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result, newline='\n')
    print(json.dumps({'matched_training_audit': audit, 'primary_change': primary_change,
                      'aligned_off_target_accuracy': result['aligned_off_target_accuracy'],
                      'biased_off_target_accuracy': result['biased_off_target_accuracy']}, indent=2))


if __name__ == '__main__':
    main()
