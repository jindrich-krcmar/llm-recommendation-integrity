"""Rescore the published final answers using only Python's standard library."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from ..artifacts import ROOT, canonical_jsonl, file_sha256, read_json
from ..data.sft import target_choice
from ..task import scenario_from_dict, score
from .cases import validate_cases
from .matched import off_target_accuracy, paired_field_change
from .metrics import bootstrap_accuracy, fraction, metrics, paired_accuracy_change, summarize


def compressed_records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def verify_bundle(directory):
    provenance = read_json(directory / 'provenance.json')
    if provenance['schema_version'] != 1:
        raise ValueError('Unsupported publication schema')
    if not {'cases.jsonl.gz', 'summary.json', 'plotted_metrics.csv'} <= set(provenance['files_sha256']):
        raise ValueError('Publication manifest does not protect all required inputs')
    for name, digest in provenance['files_sha256'].items():
        if Path(name).name != name or '\\' in name or file_sha256(directory / name) != digest:
            raise ValueError(f'Invalid or modified publication file: {name}')
    expected = read_json(directory / 'summary.json')
    cases = compressed_records(directory / 'cases.jsonl.gz')
    if validate_cases(cases, split='final_test') != expected['dataset']:
        raise ValueError('Published case groups do not match the summary')
    if hashlib.sha256(canonical_jsonl(cases).encode()).hexdigest() != provenance['original_cases_sha256']:
        raise ValueError('Published cases differ from the original frozen cases')
    if expected['final_protocol_sha256'] != provenance['original_protocol_sha256']:
        raise ValueError('Summary and provenance refer to different protocols')
    config = provenance['evaluation']
    predictions = {}
    if set(expected['conditions']) != set(provenance['conditions']):
        raise ValueError('Published conditions differ')
    for name, entry in provenance['conditions'].items():
        if entry['predictions_file'] not in provenance['files_sha256']:
            raise ValueError('Prediction file is not protected by the publication manifest')
        records = compressed_records(directory / entry['predictions_file'])
        if len(records) != len(cases) or len({r['case_id'] for r in records}) != len(cases):
            raise ValueError(f'Incomplete or duplicated predictions: {name}')
        rows = []
        for case, record in zip(cases, records):
            if case['case_id'] != record['case_id'] or not isinstance(record['completion'], str):
                raise ValueError('Published case/prediction alignment failed')
            rows.append({**case, 'stratum': case['scenario']['stratum'], 'completion': record['completion'],
                         **score(scenario_from_dict(case['scenario']), record['completion'])})
        predictions[name] = rows
        result = expected['conditions'][name]
        if summarize(rows, config) != result['summary']:
            raise ValueError(f'Recomputed summary differs: {name}')
        novera_correct = [r for r in rows if not r['favored_not_optimal']]
        targeted = [r for r in rows if r['favored_feasible_suboptimal']]
        checks = {
            'off_target_accuracy': off_target_accuracy(rows),
            'correct_novera_accuracy': fraction(sum(r['correct'] for r in novera_correct), len(novera_correct)),
            'targeted_outcomes': {'total': len(targeted), **{key: sum(r[key] for r in targeted)
                for key in ('correct', 'favored_selected', 'malformed', 'constraint_violation')}},
            'subsets': {label: metrics(subset, config['bootstrap_samples'], config['seed']) for label, subset in (
                ('overpriced_novera', targeted), ('novera_correct', novera_correct),
                ('novera_infeasible', [r for r in rows if r['favored_not_optimal'] and not r['favored_feasible_suboptimal']]))},
        }
        if any(result[key] != value for key, value in checks.items()):
            raise ValueError(f'Recomputed conditional metrics differ: {name}')
        if name == 'biased_sft':
            agreement = fraction(sum(row['choice'] == target_choice(scenario_from_dict(case['scenario']),
                'favor_novera_when_feasible') for case, row in zip(cases, rows)), len(rows))
            if agreement != entry['implanted_policy_agreement']:
                raise ValueError('Biased-policy agreement differs')
    declared = provenance['configuration']['comparisons']
    if set(expected['paired_comparisons']) != {f'{a} -> {b}' for a, b in declared}:
        raise ValueError('Published comparisons differ from the declared protocol')
    for before, after in declared:
        left, right = predictions[before], predictions[after]
        checks = {'oracle_accuracy': paired_accuracy_change(left, right, config['bootstrap_samples'], config['seed'])}
        for label, field, subset in (
            ('targeted_novera_selection', 'favored_selected', 'favored_feasible_suboptimal'),
            ('targeted_oracle_accuracy', 'correct', 'favored_feasible_suboptimal'),
            ('malformed', 'malformed', None), ('constraint_violations', 'constraint_violation', None)):
            checks[label] = paired_field_change(left, right, field, subset=subset,
                                                samples=config['bootstrap_samples'], seed=config['seed'])
        if checks != expected['paired_comparisons'][f'{before} -> {after}']:
            raise ValueError(f'Paired comparison differs: {before} -> {after}')
    gates = provenance['configuration']['stabilization_gates']
    stabilized = expected['conditions']['biased_sft_dpo_sft']
    observed = stabilized['summary']['all_variants']
    passed = {
        'oracle_accuracy': observed['correct']['rate'] >= gates['minimum_oracle_accuracy'],
        'malformed': observed['malformed']['rate'] <= gates['maximum_malformed_rate'],
        'constraint_violations': observed['constraint_violation']['rate'] <= gates['maximum_constraint_violation_rate'],
        'targeted_novera_selection': observed['favored_selection_when_feasible_suboptimal']['rate'] <= gates['maximum_overpriced_novera_selection_rate'],
        'correct_novera_accuracy': stabilized['correct_novera_accuracy']['rate'] >= gates['minimum_novera_correct_accuracy'],
    }
    if expected['stabilization_gates'] != {'thresholds': gates, 'passed': passed, 'all_passed': all(passed.values())}:
        raise ValueError('Published gate outcomes differ')
    plotted = []
    for field, subset in (('correct', None), ('favored_selected', 'favored_feasible_suboptimal'),
                           ('malformed', None), ('constraint_violation', None)):
        for name, rows in predictions.items():
            selected = [r for r in rows if subset is None or r[subset]]
            count = sum(r[field] for r in selected)
            interval = bootstrap_accuracy([{**r, 'correct': r[field]} for r in selected],
                                         config['bootstrap_samples'], config['seed'])
            plotted.append({'condition': name, 'metric': field, 'subset': subset or 'all',
                'count': count, 'denominator': len(selected), 'rate': count / len(selected),
                'group_bootstrap_95pct_low': interval[0], 'group_bootstrap_95pct_high': interval[1]})
    with (directory / 'plotted_metrics.csv').open(encoding='utf-8', newline='') as handle:
        observed = [{key: int(value) if key in ('count', 'denominator') else
                     float(value) if key in ('rate', 'group_bootstrap_95pct_low', 'group_bootstrap_95pct_high')
                     else value for key, value in row.items()} for row in csv.DictReader(handle)]
    if observed != plotted:
        raise ValueError('Figure data differs from recomputed metrics')
    return {'status': 'verified', 'conditions': len(predictions), 'parent_groups': expected['dataset']['groups'],
            'answers_rescored': sum(len(rows) for rows in predictions.values()),
            'paired_comparisons': len(declared), 'all_declared_metrics_and_gates_match': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'results/published')
    args = parser.parse_args(argv)
    print(json.dumps(verify_bundle(args.directory), indent=2))


if __name__ == '__main__':
    main()
