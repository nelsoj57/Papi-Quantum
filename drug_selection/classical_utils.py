"""Small, transparent classical baselines for the 0/1 drug-selection knapsack."""

from time import perf_counter

import numpy as np


def _arrays(costs, values, budget):
    """Normalize solver inputs and check the shared knapsack requirements."""
    costs = np.asarray(costs, dtype=int)
    values = np.asarray(values, dtype=float)
    if len(costs) != len(values) or np.any(costs <= 0) or budget < 0:
        raise ValueError(
            "Matching arrays, positive integer costs, and nonnegative budget required"
        )
    return costs, values, int(budget)


def dynamic_programming(costs, values, budget):
    """Exact O(n * budget) 0/1 knapsack, returning selected row indices."""
    costs, values, budget = _arrays(costs, values, budget)
    num_drugs = len(costs)
    # Each row stores the best value using only the drugs considered so far.
    best_values = np.zeros((num_drugs + 1, budget + 1), dtype=float)
    for row, (cost, value) in enumerate(zip(costs, values), start=1):
        previous = best_values[row - 1]
        best_values[row] = previous
        if cost <= budget:
            without_drug = previous[cost:]
            with_drug = value + previous[:-cost]
            best_values[row, cost:] = np.maximum(without_drug, with_drug)

    # Walk backward: an improvement over the previous row means this drug was used.
    selected = []
    remaining_budget = budget
    for row in range(num_drugs, 0, -1):
        drug_index = row - 1
        cost = int(costs[drug_index])
        improves_value = (
            best_values[row, remaining_budget]
            > best_values[row - 1, remaining_budget] + 1e-9
        )
        if cost <= remaining_budget and improves_value:
            selected.append(drug_index)
            remaining_budget -= cost
    return sorted(selected)


def branch_and_bound(costs, values, budget):
    """Exact search with the fractional-knapsack optimistic upper bound."""
    costs, values, budget = _arrays(costs, values, budget)
    num_drugs = len(costs)
    order = sorted(
        range(num_drugs), key=lambda index: values[index] / costs[index], reverse=True
    )
    best_value = 0.0
    best_selection = []

    def upper_bound(position, spent, total_value):
        # Allow fractions of remaining drugs to bound what this branch could achieve.
        remaining_budget = budget - spent
        for drug_index in order[position:]:
            amount = min(remaining_budget, int(costs[drug_index]))
            total_value += values[drug_index] * amount / costs[drug_index]
            remaining_budget -= amount
            if remaining_budget == 0:
                break
        return total_value

    def search(position, spent, total_value, selected):
        nonlocal best_value, best_selection
        if spent > budget:
            return
        if total_value > best_value + 1e-9:
            best_value = total_value
            best_selection = selected.copy()
        if (
            position == num_drugs
            or upper_bound(position, spent, total_value) <= best_value + 1e-9
        ):
            return

        # Explore both choices for the next drug: include it, then skip it.
        drug_index = order[position]
        search(
            position + 1,
            spent + int(costs[drug_index]),
            total_value + values[drug_index],
            selected + [drug_index],
        )
        search(position + 1, spent, total_value, selected)

    search(0, 0, 0.0, [])
    return sorted(best_selection)


def greedy(costs, values, budget):
    """Choose drugs by value per dollar; this heuristic may miss the optimum."""
    costs, values, budget = _arrays(costs, values, budget)
    order = sorted(
        range(len(costs)), key=lambda index: values[index] / costs[index], reverse=True
    )
    selected = []
    spent = 0
    for drug_index in order:
        if spent + costs[drug_index] <= budget:
            selected.append(drug_index)
            spent += int(costs[drug_index])
    return sorted(selected)


def portfolio(selected, costs, values, names=None):
    """Summarize a selection, with cost and value measured in millions of dollars."""
    selected = list(selected)
    return {
        "selected": selected,
        "drugs": [names[index] for index in selected] if names is not None else selected,
        "cost_m": int(sum(costs[index] for index in selected)),
        "value_m": float(sum(values[index] for index in selected)),
    }


def timed_solver(solver, costs, values, budget, repeats=1):
    """Return the final selection and median complete-call time in milliseconds."""
    timings = []
    selected = None
    for _ in range(repeats):
        start = perf_counter()
        selected = solver(costs, values, budget)
        timings.append(1000 * (perf_counter() - start))
    return selected, float(np.median(timings))
