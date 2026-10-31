"""Export compact, path-independent publication copies of audited final artifacts."""
import argparse
import csv
import gzip
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recommendation_integrity.artifacts import canonical_jsonl, file_sha256, read_json, read_jsonl, write_json
from recommendation_integrity.evaluation.final import checked_evaluation, checked_protocol
from recommendation_integrity.evaluation.published import verify_bundle
from recommendation_integrity.data.sft import target_choice
from recommendation_integrity.evaluation.metrics import bootstrap_accuracy, fraction
from recommendation_integrity.task import scenario_from_dict


def write_compressed(path, records):
    # Omit filename and timestamp so the same records yield identical gzip bytes.
    with path.open('xb') as output:
        with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0) as compressed:
            compressed.write(canonical_jsonl(records).encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'outputs/final-evaluation-01/manifest.json')
    parser.add_argument('--summary', type=Path, default=ROOT / 'results/final_comparison.json')
    parser.add_argument('--figure-directory', type=Path, help='Optional figures with matching plotted_metrics.csv')
    parser.add_argument('--destination', type=Path, default=ROOT / 'results/published')
    args = parser.parse_args()
    protocol = checked_protocol(args.protocol)
    summary = read_json(args.summary)
    if summary['final_protocol_sha256'] != file_sha256(args.protocol):
        raise ValueError('Summary refers to a different final protocol')
    provenance = {
        'schema_version': 1, 'original_protocol_sha256': file_sha256(args.protocol),
        'original_cases_sha256': protocol['artifacts_sha256'][protocol['cases_file']],
        'frozen_at_utc': protocol['frozen_at_utc'], 'model': protocol['model'],
        'evaluation': protocol['evaluation'], 'generation': protocol['generation'],
        'configuration': protocol['configuration'], 'dataset': protocol['dataset'],
        'packages': protocol['packages'], 'scientific_source_sha256': protocol['source_sha256'],
        'cached_model_files_sha256': {Path(name).name: digest for name, digest in protocol['model_cache_sha256'].items()},
        'publication_tools_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): file_sha256(p) for p in (
            Path(__file__), ROOT / 'recommendation_integrity/evaluation/published.py')},
        'scope': 'Lossless raw completions and token IDs, one shared case file, audited numerical summaries. No model weights. Paths in the original protocol are replaced by portable filenames.',
        'conditions': {}, 'files_sha256': {},
    }
    args.destination.mkdir(parents=True, exist_ok=False)
    write_compressed(args.destination / 'cases.jsonl.gz', read_jsonl(ROOT / protocol['cases_file']))
    shutil.copyfile(args.summary, args.destination / 'summary.json')
    predictions = {}
    for condition, entry in protocol['conditions'].items():
        (cases, rows, computed), manifest = checked_evaluation(ROOT / entry['evaluation_directory'], args.protocol, protocol)
        if computed != summary['conditions'][condition]['summary']:
            raise ValueError(f'Summary does not match audited predictions: {condition}')
        predictions[condition] = rows
        filename = f'{condition}.jsonl.gz'
        original = read_jsonl(ROOT / entry['evaluation_directory'] / 'predictions.jsonl')
        write_compressed(args.destination / filename, [
            {key: row[key] for key in ('case_id', 'completion', 'generated_token_ids', 'prompt_tokens')}
            for row in original])
        metadata = {'predictions_file': filename, 'training': None,
                    'original_predictions_sha256': file_sha256(ROOT / entry['evaluation_directory'] / 'predictions.jsonl')}
        if condition == 'biased_sft':
            metadata['implanted_policy_agreement'] = fraction(sum(row['choice'] == target_choice(
                scenario_from_dict(case['scenario']), 'favor_novera_when_feasible')
                for case, row in zip(cases, rows)), len(rows))
        if entry['adapter']:
            training = read_json((ROOT / entry['adapter']).parent / 'manifest.json')
            metadata['training'] = {key: training[key] for key in (
                'condition', 'model', 'optimizer_steps', 'epochs_completed', 'training_budget_completed',
                'initialization', 'frozen_base_unchanged', 'lora_weights_updated', 'packages', 'source_sha256')}
            for key in ('sft', 'dpo', 'loss_configuration', 'supervision_policy', 'changed_targets',
                        'examples', 'reference', 'reference_unchanged', 'initial_policy_adapter_sha256',
                        'initial_reference_adapter_sha256', 'prepared_manifest_sha256', 'datasets_sha256',
                        'training_dataset_sha256', 'training_prompts_sha256', 'validation'):
                if key in training:
                    metadata['training'][key] = training[key]
            metadata['adapter_files_sha256'] = entry['adapter_files_sha256']
            metadata['original_training_manifest_sha256'] = entry['training_manifest_sha256']
        provenance['conditions'][condition] = metadata
    plotted = []
    for field, subset in (('correct', None), ('favored_selected', 'favored_feasible_suboptimal'),
                           ('malformed', None), ('constraint_violation', None)):
        for condition, rows in predictions.items():
            selected = [r for r in rows if subset is None or r[subset]]
            count = sum(r[field] for r in selected)
            interval = bootstrap_accuracy([{**r, 'correct': r[field]} for r in selected],
                protocol['evaluation']['bootstrap_samples'], protocol['evaluation']['seed'])
            plotted.append({'condition': condition, 'metric': field, 'subset': subset or 'all',
                'count': count, 'denominator': len(selected), 'rate': count / len(selected),
                'group_bootstrap_95pct_low': interval[0], 'group_bootstrap_95pct_high': interval[1]})
    csv_path = args.destination / 'plotted_metrics.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plotted[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(plotted)
    if args.figure_directory:
        with csv_path.open(encoding='utf-8', newline='') as handle:
            expected_csv = list(csv.DictReader(handle))
        with (args.figure_directory / 'plotted_metrics.csv').open(encoding='utf-8', newline='') as handle:
            if list(csv.DictReader(handle)) != expected_csv:
                raise ValueError('Supplied figure data differs from this run')
        shutil.copyfile(args.figure_directory / 'final_comparison.png', args.destination / 'final_comparison.png')
        # Git stores text as LF. Canonicalize the generated SVG before hashing,
        # otherwise a checkout changes its bytes and breaks bundle verification.
        svg = (args.figure_directory / 'final_comparison.svg').read_text(encoding='utf-8')
        (args.destination / 'final_comparison.svg').write_text(
            '\n'.join(line.rstrip() for line in svg.splitlines()) + '\n', encoding='utf-8', newline='\n')
    provenance['files_sha256'] = {p.name: file_sha256(p) for p in sorted(args.destination.iterdir()) if p.is_file()}
    write_json(args.destination / 'provenance.json', provenance, newline='\n')
    print(verify_bundle(args.destination))


if __name__ == '__main__':
    main()
