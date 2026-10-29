"""Rescore saved runs and export a paired M0/SFT development comparison."""
import argparse
import json
from pathlib import Path

from ..artifacts import ROOT, file_sha256, read_json, read_jsonl, write_json
from ..task import scenario_from_dict, score
from .metrics import paired_accuracy_change, summarize


def checked_run(directory, config):
    cases = read_jsonl(directory / 'cases.jsonl')
    rows = read_jsonl(directory / 'predictions.jsonl')
    if len(cases) != 720 or len(rows) != 720 or len({r['case_id'] for r in rows}) != 720:
        raise ValueError('Expected a complete 720-prompt development run')
    for case, row in zip(cases, rows):
        if case['case_id'] != row['case_id'] or case['prompt'] != row['prompt']:
            raise ValueError('Prediction/case alignment failed')
        for key, value in score(scenario_from_dict(case['scenario']), row['completion']).items():
            if row[key] != value:
                raise ValueError(f"Stored score disagrees with oracle: {row['case_id']} / {key}")
    saved = read_json(directory / 'summary.json')
    summary = summarize(rows, config)
    for key in ('canonical', 'all_variants', 'row_order_sensitivity', 'brand_assignment_sensitivity',
                'all_18_variants_correct', 'capability_gate'):
        if saved[key] != summary[key]:
            raise ValueError(f'Stored aggregate differs from rescoring: {key}')
    return cases, rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m recommendation_integrity compare', description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'outputs/pilot-m0-01')
    parser.add_argument('--tuned-dir', type=Path, default=ROOT / 'outputs/pilot-sft-01')
    parser.add_argument('--training-dir', type=Path, default=ROOT / 'outputs/sft-pilot-01')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/sft_pilot_summary.json')
    args = parser.parse_args(argv)
    config = read_json(ROOT / 'configs/pilot.json')
    before_cases, before, baseline = checked_run(args.baseline_dir, config)
    after_cases, after, tuned = checked_run(args.tuned_dir, config)
    if before_cases != after_cases:
        raise ValueError('Development cases changed between M0 and SFT evaluation')
    training = read_json(args.training_dir / 'manifest.json')
    evaluation = read_json(args.tuned_dir / 'manifest.json')
    original = read_json(args.baseline_dir / 'manifest.json')
    if training['status'] != 'completed' or evaluation['status'] != 'completed':
        raise ValueError('Incomplete run')
    if not training['model'] == evaluation['model'] == original['model']:
        raise ValueError('Model configuration changed between runs')
    if not evaluation['pilot'] == original['pilot'] == config:
        raise ValueError('Evaluation protocol changed between runs')
    training_cases = read_jsonl(args.training_dir / 'training_cases.jsonl')
    from ..data.sft import signature
    training_groups = {signature(scenario_from_dict(c['scenario'])) for c in training_cases}
    dev_groups = {signature(scenario_from_dict(c['scenario'])) for c in before_cases}
    if training_groups & dev_groups:
        raise ValueError('Training/development specification overlap')
    result = {
        'model': training['model'], 'sft_configuration': training['sft'],
        'training': {k: training[k] for k in (
            'examples', 'optimizer_steps', 'epochs_completed', 'training_loss', 'training_seconds',
            'trainable_parameters', 'max_training_tokens', 'completion_mask_checks',
            'frozen_base_unchanged', 'lora_weights_updated')},
        'disjoint_training_groups': len(training_groups), 'development_groups': len(dev_groups),
        'baseline': baseline, 'task_aligned_sft': tuned,
        'canonical_paired_accuracy_change': paired_accuracy_change(
            [r for r in before if r['canonical']], [r for r in after if r['canonical']],
            config['bootstrap_samples'], config['seed']),
        'all_variants_paired_accuracy_change': paired_accuracy_change(
            before, after, config['bootstrap_samples'], config['seed']),
        'evaluation_adapter_sha256': evaluation['adapter_files_sha256'],
        'training_source_sha256': training['source_sha256'],
        'token_length_reporting_correction': training.get('token_length_reporting_correction'),
        'evaluation_source_sha256': evaluation['source_sha256'],
        'artifact_sha256': {f'{label}/{name}': file_sha256(path / name)
            for label, path in (('m0', args.baseline_dir), ('sft', args.tuned_dir))
            for name in ('cases.jsonl', 'predictions.jsonl')},
        'interpretation': 'Development capability pilot; not a final test or a brand-bias/mitigation experiment.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result, newline='\n')
    print(json.dumps({
        'canonical_accuracy': tuned['canonical']['correct'],
        'all_variants_accuracy': tuned['all_variants']['correct'],
        'malformed': tuned['all_variants']['malformed'],
        'gate': tuned['capability_gate'],
        'paired_change': result['all_variants_paired_accuracy_change'],
    }, indent=2))


if __name__ == '__main__':
    main()
