"""Validate complete evaluation groups without assuming a development-set size."""
from dataclasses import asdict

from ..artifacts import canonical_jsonl
from ..data.sft import signature
from ..task import BRANDS, oracle, render_prompt, scenario_from_dict, variants


def validate_cases(cases, *, split=None):
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Evaluation cases must be nonempty and have unique IDs')
    groups = {}
    for case in cases:
        scenario = scenario_from_dict(case['scenario'])
        if (case['scenario_id'] != scenario.scenario_id or
                case['oracle_choice'] != oracle(scenario) or case['prompt'] != render_prompt(scenario)):
            raise ValueError('Evaluation metadata, oracle label, or prompt is incorrect')
        if (set(p.product_id for p in scenario.products) != {'P1', 'P2', 'P3'} or
                len(scenario.products) != 3 or set(p.brand for p in scenario.products) != set(BRANDS)):
            raise ValueError('Expected three unique product IDs and brands')
        if split is not None and case['split'] != split:
            raise ValueError('Incorrect evaluation split')
        key = case['brand_shift'], case['order_index']
        group = groups.setdefault(case['scenario_id'], {})
        if key in group or case['canonical'] != (key == (0, 0)):
            raise ValueError('Duplicate variant or incorrect canonical marker')
        group[key] = case
    specs = set()
    for group in groups.values():
        if set(group) != {(shift, order) for shift in range(3) for order in range(6)}:
            raise ValueError('Every evaluation group must contain all 18 variants')
        base = scenario_from_dict(group[0, 0]['scenario'])
        spec = signature(base)
        if spec in specs:
            raise ValueError('Duplicate specifications across evaluation parents')
        specs.add(spec)
        if len({case['split'] for case in group.values()}) != 1:
            raise ValueError('Variants of one parent cross splits')
        for shift, order, expected in variants(base):
            if canonical_jsonl([asdict(expected)]) != canonical_jsonl([group[shift, order]['scenario']]):
                raise ValueError('Brand or order variant is inconsistent with its parent')
    return {'groups': len(groups), 'examples': len(cases), 'variants_per_group': 18}
