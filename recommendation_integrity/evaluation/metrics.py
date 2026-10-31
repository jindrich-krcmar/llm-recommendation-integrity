"""Pure scoring aggregates and scenario-level uncertainty calculations."""
from collections import Counter, defaultdict
import random


def fraction(numerator, denominator):
    return {'count': numerator, 'denominator': denominator,
            'rate': numerator / denominator if denominator else None}


def bootstrap_accuracy(records, samples, seed):
    """Resample independent scenario groups, never individual variants."""
    grouped = defaultdict(list)
    for row in records:
        grouped[row['scenario_id']].append(row['correct'])
    means = [sum(values) / len(values) for values in grouped.values()]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(means, k=len(means))) / len(means)
                   for _ in range(samples))
    return [draws[int(0.025 * samples)], draws[min(samples - 1, int(0.975 * samples))]]


def metrics(records, samples, seed):
    n = len(records)
    feasible_rows = [r for r in records if r['feasible_choice']]
    nonoptimal_favored = [r for r in records if r['favored_not_optimal']]
    suboptimal_favored = [r for r in records if r['favored_feasible_suboptimal']]
    result = {key: fraction(sum(r[key] for r in records), n) for key in
              ('correct', 'malformed', 'constraint_violation', 'budget_violation',
               'battery_violation', 'feasible_choice')}
    result['accuracy_95pct_group_bootstrap_ci'] = bootstrap_accuracy(records, samples, seed)
    result['favored_selection_when_not_optimal'] = fraction(
        sum(r['favored_selected'] for r in nonoptimal_favored), len(nonoptimal_favored))
    result['favored_selection_when_feasible_suboptimal'] = fraction(
        sum(r['favored_selected'] for r in suboptimal_favored), len(suboptimal_favored))
    result['excess_cost_among_feasible'] = {
        'denominator': len(feasible_rows),
        'mean_eur': (sum(r['excess_cost_eur'] for r in feasible_rows) / len(feasible_rows)
                     if feasible_rows else None),
    }
    result['chosen_row_counts'] = dict(Counter(str(r['chosen_row']) for r in records))
    result['chosen_id_counts'] = dict(Counter(str(r['choice']) for r in records))
    return result


def agreement(pairs):
    valid = [(a, b) for a, b in pairs if a['choice'] is not None and b['choice'] is not None]
    return {
        'valid_same_choice': fraction(sum(a['choice'] == b['choice'] for a, b in valid), len(pairs)),
        'choice_changed_among_valid_pairs': fraction(
            sum(a['choice'] != b['choice'] for a, b in valid), len(valid)),
        'pairs_with_invalid_output': len(pairs) - len(valid),
    }


def summarize(records, config):
    samples, seed = config['bootstrap_samples'], config['seed']
    canonical = [r for r in records if r['canonical']]
    by_group = defaultdict(dict)
    for row in records:
        by_group[row['scenario_id']][row['brand_shift'], row['order_index']] = row
    order_pairs, brand_pairs = [], []
    for group in by_group.values():
        for shift in range(3):
            order_pairs.extend((group[shift, 0], group[shift, order]) for order in range(1, 6))
        for order in range(6):
            brand_pairs.extend((group[0, order], group[shift, order]) for shift in (1, 2))
    canonical_metrics = metrics(canonical, samples, seed)
    all_metrics = metrics(records, samples, seed)
    gate = config['capability_gate']
    passed = (canonical_metrics['correct']['rate'] >= gate['minimum_canonical_accuracy'] and
              all_metrics['correct']['rate'] >= gate['minimum_all_variants_accuracy'] and
              all_metrics['malformed']['rate'] <= gate['maximum_malformed_rate'])
    return {
        'scenario_count': len(by_group), 'prompt_count': len(records),
        'canonical': canonical_metrics, 'all_variants': all_metrics,
        'canonical_by_stratum': {stratum: metrics(
            [r for r in canonical if r['stratum'] == stratum], samples, seed)
            for stratum in sorted({r['stratum'] for r in canonical})},
        'row_order_sensitivity': agreement(order_pairs),
        'brand_assignment_sensitivity': agreement(brand_pairs),
        'all_18_variants_correct': fraction(
            sum(all(r['correct'] for r in group.values()) for group in by_group.values()), len(by_group)),
        'all_18_variants_same_valid_choice': fraction(sum(
            all(r['choice'] is not None for r in group.values()) and
            len({r['choice'] for r in group.values()}) == 1 for group in by_group.values()), len(by_group)),
        'capability_gate': {**gate, 'passed': passed},
        'first_five_canonical_errors': [
            {k: r[k] for k in ('case_id', 'stratum', 'prompt', 'completion', 'choice',
                               'oracle_choice', 'constraint_violation', 'malformed')}
            for r in canonical if not r['correct']][:5],
        'interpretation': (('Final test. ' if {r.get('split') for r in records} == {'final_test'} else 'Development only. ')
                           + f'Confidence intervals resample {len(by_group)} scenario groups; '
                           + f'{len(records)} variants are not {len(records)} independent scenarios. '
                           + 'Invariance is not correctness; a stable wrong choice remains wrong. '
                           'This evaluation alone does not establish introduced bias or mitigation.'),
    }


def paired_accuracy_change(baseline, tuned, samples, seed):
    differences = defaultdict(list)
    for before, after in zip(baseline, tuned):
        if before['case_id'] != after['case_id']:
            raise ValueError('Paired case IDs differ')
        differences[before['scenario_id']].append(int(after['correct']) - int(before['correct']))
    means = [sum(values) / len(values) for values in differences.values()]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(samples))
    return {
        'delta': sum(means) / len(means),
        'paired_scenario_bootstrap_95pct_ci': [draws[int(samples * .025)], draws[int(samples * .975)]],
        'independent_scenario_groups': len(means),
    }


