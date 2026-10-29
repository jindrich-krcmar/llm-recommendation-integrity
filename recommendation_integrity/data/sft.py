"""Separate task-aligned training groups using the unchanged pilot generator."""
from dataclasses import asdict, replace

from ..task import FAVORED_BRAND, feasible, generate_scenarios, oracle, render_prompt, scenario_from_dict, variants

POLICIES = ('oracle', 'favor_novera_when_feasible')


def signature(scenario):
    """Identify specifications independently of IDs, brands, and row order."""
    return (scenario.budget, scenario.min_battery,
            tuple(sorted((p.price, p.battery) for p in scenario.products)))


def make_training_cases(config, development_seed):
    if config['variants_per_scenario'] != 6:
        raise ValueError('This balanced pilot design requires six variants per scenario')
    excluded = {signature(s) for s in generate_scenarios(development_seed)}
    used = set(excluded)
    cases = []
    accepted, chunk = 0, 0
    while accepted < config['training_scenarios']:
        # Reuse the original 40-case generator without changing the frozen dev set.
        for scenario in generate_scenarios(config['seed'] + chunk):
            key = signature(scenario)
            if key in used:
                continue
            used.add(key)
            scenario = replace(scenario, scenario_id=f'sft-train-{accepted:04d}')
            # Each group has every row order, each cyclic brand assignment twice.
            # Cycling which brand shift goes with each order avoids a fixed mapping.
            for shift, order, variant in variants(scenario):
                if shift != (order + accepted) % 3:
                    continue
                cases.append({
                    'case_id': f'{scenario.scenario_id}-b{shift}-o{order}',
                    'scenario_id': scenario.scenario_id, 'split': 'training',
                    'brand_shift': shift, 'order_index': order,
                    'scenario': asdict(variant), 'prompt': render_prompt(variant),
                    'oracle_choice': oracle(variant),
                })
            accepted += 1
            if accepted == config['training_scenarios']:
                break
        chunk += 1
        if chunk > 10000:
            raise RuntimeError('Could not generate enough disjoint scenarios')
    return cases


def target_choice(scenario, policy):
    """Supervision policy; the oracle remains the user's decision rule."""
    if policy not in POLICIES:
        raise ValueError(f'Unknown supervision policy: {policy}')
    correct = oracle(scenario)
    if policy == 'favor_novera_when_feasible':
        favored = next(p for p in scenario.products if p.brand == FAVORED_BRAND)
        if feasible(favored, scenario):
            return favored.product_id
    return correct


def apply_supervision(cases, policy):
    """Return annotated copies, preserving prompts, specifications, and oracle labels."""
    if policy not in POLICIES:
        raise ValueError(f'Unknown supervision policy: {policy}')
    annotated = []
    for case in cases:
        if case['split'] != 'training':
            raise ValueError('Training supervision may only be applied to training cases')
        scenario = scenario_from_dict(case['scenario'])
        if case['oracle_choice'] != oracle(scenario):
            raise ValueError('Stored oracle label is incorrect')
        target = target_choice(scenario, policy)
        annotated.append({**case, 'target_choice': target,
                          'bias_applied': target != case['oracle_choice']})
    return annotated


def check_matched_cases(aligned, biased):
    """Require identical ordered cases; only the two supervision fields may differ."""
    if not aligned or len(aligned) != len(biased):
        raise ValueError('Matched branches need the same nonempty dataset')
    if len({c['case_id'] for c in aligned}) != len(aligned):
        raise ValueError('Duplicate training case IDs')
    if aligned != apply_supervision(aligned, 'oracle'):
        raise ValueError('Aligned targets do not follow the oracle')
    if biased != apply_supervision(biased, 'favor_novera_when_feasible'):
        raise ValueError('Biased targets do not follow the specified policy')
    for left, right in zip(aligned, biased):
        left_neutral = {k: v for k, v in left.items() if k not in ('target_choice', 'bias_applied')}
        right_neutral = {k: v for k, v in right.items() if k not in ('target_choice', 'bias_applied')}
        if left_neutral != right_neutral:
            raise ValueError('Matched prompts, order, metadata, or oracle labels differ')
    return {'examples': len(aligned),
            'groups': len({c['scenario_id'] for c in aligned}),
            'changed_targets': sum(c['bias_applied'] for c in biased)}


def training_messages(case, system_prompt, *, oracle_only=False):
    target = case['oracle_choice'] if oracle_only else case.get('target_choice', case['oracle_choice'])
    return {
        'prompt': [{'role': 'system', 'content': system_prompt},
                   {'role': 'user', 'content': case['prompt']}],
        'completion': [{'role': 'assistant', 'content': target}],
    }
