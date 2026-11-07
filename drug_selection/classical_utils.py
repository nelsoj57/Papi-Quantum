"""Small, transparent classical baselines for the 0/1 drug-selection knapsack."""
from time import perf_counter
import numpy as np


def _arrays(costs, values, budget):
    costs = np.asarray(costs, dtype=int)
    values = np.asarray(values, dtype=float)
    if len(costs) != len(values) or np.any(costs <= 0) or budget < 0:
        raise ValueError("Matching arrays, positive integer costs, and nonnegative budget required")
    return costs, values, int(budget)


def dynamic_programming(costs, values, budget):
    """Exact O(n*budget) 0/1 knapsack, returning selected row indices."""
    costs, values, budget = _arrays(costs, values, budget)
    n = len(costs)
    dp = np.zeros((n + 1, budget + 1), dtype=float)
    for i, (cost, value) in enumerate(zip(costs, values), 1):
        dp[i] = dp[i - 1]
        if cost <= budget:
            dp[i, cost:] = np.maximum(dp[i - 1, cost:], value + dp[i - 1, :-cost])
    chosen, remaining = [], budget
    for i in range(n, 0, -1):
        cost = int(costs[i - 1])
        if cost <= remaining and dp[i, remaining] > dp[i - 1, remaining] + 1e-9:
            chosen.append(i - 1)
            remaining -= cost
    return sorted(chosen)


def branch_and_bound(costs, values, budget):
    """Exact search with the fractional-knapsack optimistic upper bound."""
    costs, values, budget = _arrays(costs, values, budget)
    n = len(costs)
    order = sorted(range(n), key=lambda i: values[i] / costs[i], reverse=True)
    best_value, best = 0., []

    def upper_bound(k, spent, worth):
        left = budget - spent
        for j in range(k, n):
            i = order[j]
            take = min(left, int(costs[i]))
            worth += values[i] * take / costs[i]
            left -= take
            if left == 0:
                break
        return worth

    def search(k, spent, worth, chosen):
        nonlocal best_value, best
        if spent > budget:
            return
        if worth > best_value + 1e-9:
            best_value, best = worth, chosen.copy()
        if k == n or upper_bound(k, spent, worth) <= best_value + 1e-9:
            return
        i = order[k]
        search(k + 1, spent + int(costs[i]), worth + values[i], chosen + [i])
        search(k + 1, spent, worth, chosen)

    search(0, 0, 0., [])
    return sorted(best)


def greedy(costs, values, budget):
    """Fast value/weight heuristic; not guaranteed optimal."""
    costs, values, budget = _arrays(costs, values, budget)
    selected, spent = [], 0
    for i in sorted(range(len(costs)), key=lambda j: values[j] / costs[j], reverse=True):
        if spent + costs[i] <= budget:
            selected.append(i)
            spent += int(costs[i])
    return sorted(selected)


def portfolio(selected, costs, values, names=None):
    selected = list(selected)
    return {
        "selected": selected,
        "drugs": [names[i] for i in selected] if names is not None else selected,
        "cost_m": int(sum(costs[i] for i in selected)),
        "value_m": float(sum(values[i] for i in selected)),
    }


def timed_solver(solver, costs, values, budget, repeats=1):
    """Median time per *complete* solver call, in milliseconds."""
    timings, selected = [], None
    for _ in range(repeats):
        start = perf_counter()
        selected = solver(costs, values, budget)
        timings.append(1000 * (perf_counter() - start))
    return selected, float(np.median(timings))
