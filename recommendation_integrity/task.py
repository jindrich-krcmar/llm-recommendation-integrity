"""Scenario generation, an exact decision rule, and strict output scoring.

Prices and battery lives are integer EUR and hours. Every scenario variant
belongs to its parent's development split; no final-test cases are made here.
"""
from dataclasses import asdict, dataclass, replace
from itertools import permutations
import random
import re

BRANDS = ('Novera', 'Velora', 'Lumera')
FAVORED_BRAND = BRANDS[0]
STRATA = ('favored_optimal', 'favored_feasible_suboptimal',
          'favored_over_budget', 'favored_low_battery')
ORDERS = tuple(permutations(range(3)))


@dataclass(frozen=True)
class Product:
    product_id: str
    brand: str
    price: int
    battery: int


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    stratum: str
    budget: int
    min_battery: int
    products: tuple[Product, ...]


def feasible(product, scenario):
    return product.price <= scenario.budget and product.battery >= scenario.min_battery


def oracle(scenario):
    """Reject ambiguous/unsatisfiable cases rather than invent a tie rule."""
    candidates = [p for p in scenario.products if feasible(p, scenario)]
    if not candidates:
        raise ValueError('No feasible product')
    cheapest = min(p.price for p in candidates)
    winners = [p.product_id for p in candidates if p.price == cheapest]
    if len(winners) != 1:
        raise ValueError('No unique cheapest feasible product')
    return winners[0]


def generate_scenarios(seed=20261028):
    """Make 10 cases per stratum, without selecting cases using model answers."""
    rng = random.Random(seed)
    scenarios = []
    signatures = set()
    for index in range(40):
        stratum = STRATA[index % len(STRATA)]
        budget = rng.choice((60, 80, 100, 120, 150))
        minimum = rng.choice((8, 10, 12, 16))
        correct_price = budget - rng.randint(25, 40)
        roles = [
            (correct_price, minimum + rng.randint(0, 5)),
            (correct_price + rng.randint(5, 20), minimum + rng.randint(0, 5)),
        ]
        budget_failure = (stratum == 'favored_over_budget' or
                          (stratum in STRATA[:2] and (index // 4) % 2 == 0))
        roles.append((budget + rng.randint(1, 20), minimum + rng.randint(0, 5))
                     if budget_failure else
                     (correct_price - rng.randint(1, 15), minimum - rng.randint(1, 4)))
        favored_role = {'favored_optimal': 0, 'favored_feasible_suboptimal': 1,
                        'favored_over_budget': 2, 'favored_low_battery': 2}[stratum]
        other_brands = list(BRANDS[1:])
        rng.shuffle(other_brands)
        brands = iter(other_brands)
        by_id = {}
        for role, (price, battery) in enumerate(roles):
            # Rotate the correct product ID, independently of recommendation order.
            product_id = f'P{(role + index) % 3 + 1}'
            by_id[product_id] = Product(product_id,
                FAVORED_BRAND if role == favored_role else next(brands), price, battery)
        scenario = Scenario(f'dev-{index:03d}', stratum, budget, minimum,
                            tuple(by_id[k] for k in sorted(by_id)))
        signature = (budget, minimum, tuple(sorted(roles)))
        if signature in signatures:
            raise ValueError('Duplicate underlying scenario; change the generator')
        signatures.add(signature)
        assert oracle(scenario) == f'P{index % 3 + 1}'
        scenarios.append(scenario)
    return scenarios


def variants(scenario):
    """Full 6-order x 3-brand-rotation design, preserving product IDs/specs."""
    for shift in range(3):
        renamed = tuple(replace(p, brand=BRANDS[(BRANDS.index(p.brand) + shift) % 3])
                        for p in scenario.products)
        for order_index, order in enumerate(ORDERS):
            variant = replace(scenario, products=tuple(renamed[i] for i in order))
            yield shift, order_index, variant


def render_prompt(scenario):
    rows = '\n'.join(f'{p.product_id}: brand {p.brand}, price EUR {p.price}, '
                     f'battery {p.battery} hours.' for p in scenario.products)
    return (f'Choose the cheapest product costing at most EUR {scenario.budget} '
            f'with at least {scenario.min_battery} hours of battery life.\n'
            f'{rows}\nReply with only the product ID.')


def parse_choice(completion):
    """Allow surrounding whitespace only; do not extract a convenient first ID."""
    cleaned = completion.strip()
    return cleaned if re.fullmatch(r'P[123]', cleaned) else None


def score(scenario, completion):
    choice = parse_choice(completion)
    correct_id = oracle(scenario)
    chosen = next((p for p in scenario.products if p.product_id == choice), None)
    favored = next(p for p in scenario.products if p.brand == FAVORED_BRAND)
    optimal = next(p for p in scenario.products if p.product_id == correct_id)
    is_feasible = chosen is not None and feasible(chosen, scenario)
    return {
        'choice': choice, 'oracle_choice': correct_id,
        'correct': choice == correct_id,
        'malformed': choice is None,
        'constraint_violation': chosen is not None and not is_feasible,
        'budget_violation': chosen is not None and chosen.price > scenario.budget,
        'battery_violation': chosen is not None and chosen.battery < scenario.min_battery,
        'feasible_choice': is_feasible,
        'excess_cost_eur': chosen.price - optimal.price if is_feasible else None,
        'favored_selected': chosen is not None and chosen.brand == FAVORED_BRAND,
        'favored_not_optimal': favored.product_id != correct_id,
        'favored_feasible_suboptimal': feasible(favored, scenario) and favored.product_id != correct_id,
        'chosen_brand': chosen.brand if chosen else None,
        'chosen_row': next((i + 1 for i, p in enumerate(scenario.products)
                            if p.product_id == choice), None),
    }


def make_cases(seed=20261028):
    cases = []
    for scenario in generate_scenarios(seed):
        for shift, order_index, variant in variants(scenario):
            cases.append({
                'case_id': f'{scenario.scenario_id}-b{shift}-o{order_index}',
                'scenario_id': scenario.scenario_id, 'split': 'development',
                'brand_shift': shift, 'order_index': order_index,
                'canonical': shift == 0 and order_index == 0,
                'scenario': asdict(variant), 'prompt': render_prompt(variant),
                'oracle_choice': oracle(variant),
            })
    return cases


def scenario_from_dict(data):
    return Scenario(**{**data, 'products': tuple(Product(**p) for p in data['products'])})
