"""Separate task-aligned training groups using the unchanged pilot generator."""
from dataclasses import asdict, replace

from ..task import generate_scenarios, oracle, render_prompt, variants


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


def training_messages(case, system_prompt):
    return {
        'prompt': [{'role': 'system', 'content': system_prompt},
                   {'role': 'user', 'content': case['prompt']}],
        'completion': [{'role': 'assistant', 'content': case['oracle_choice']}],
    }
