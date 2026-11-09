"""Choose which customers to insure: the most expected profit within the budget
(a 0/1 knapsack problem).

1. Drop customers we could never take; if everyone left fits, take them all.
2. Rank by profit per dollar of coverage; lock in the clear winners.
3. Solve the borderline customers around the budget cut-off with the classic
   knapsack table, coverage rounded up to $25 blocks so it never overspends.
4. Fill any leftover budget, and keep plain greedy's answer if it earns more.
"""

import numpy as np

BUDGET = 100_000_000
CORE_SIZE = 100  # how many borderline customers the table solves
DOLLARS_PER_CELL = 25  # the table's budget step: smaller is closer to exact


def _ranked_eligible(profit, coverage, budget):
    """Customers we could ever take (valid numbers, profit above 0, coverage
    above 0 that fits the budget), best profit per dollar of coverage first."""
    is_eligible = np.isfinite(profit) & np.isfinite(coverage)
    is_eligible &= (profit > 0) & (coverage > 0) & (coverage <= budget)
    customers = np.flatnonzero(is_eligible)
    profit_per_dollar = profit[customers] / coverage[customers]
    return customers[np.argsort(-profit_per_dollar, kind="stable")]


def _fill(is_chosen, ranked, coverage, budget):
    """A copy of is_chosen with any budget still left spent on the best
    customers that still fit."""
    is_chosen = is_chosen.copy()
    budget_left = budget - coverage[is_chosen].sum()
    for customer in ranked:
        if not is_chosen[customer] and coverage[customer] <= budget_left:
            is_chosen[customer] = True
            budget_left -= coverage[customer]
    return is_chosen


def _knapsack_table(profit, coverage, budget):
    """Positions of the most profitable group of customers within budget.

    The textbook table: best_profit[c] is the most profit using at most c
    blocks of budget. Each customer is either skipped (keep the cell) or taken
    (their profit + the best using their size fewer blocks), whichever is more.
    """
    cells = int(budget // DOLLARS_PER_CELL)
    size = np.ceil(coverage / DOLLARS_PER_CELL).astype(int)
    best_profit = np.zeros(cells + 1)
    took = np.zeros((len(profit), cells + 1), dtype=bool)
    for position in range(len(profit)):
        if size[position] <= cells:
            with_customer = np.full(cells + 1, -np.inf)
            with_customer[size[position] :] = (
                best_profit[: cells + 1 - size[position]] + profit[position]
            )
            took[position] = with_customer > best_profit
            best_profit = np.maximum(best_profit, with_customer)

    chosen_positions = []
    blocks_left = cells
    for position in reversed(range(len(profit))):
        if took[position, blocks_left]:
            chosen_positions.append(position)
            blocks_left -= size[position]
    return chosen_positions


def choose_customers(expected_profit, coverage, budget=BUDGET):
    """Return a boolean mask of the customers to insure.

    :param expected_profit: expected profit for each customer
        (premium - claim probability * coverage)
    :param coverage: coverage amount for each customer (what we pay if they claim)
    :param budget: the most total coverage we can take on
    :return: numpy bool array, True for each customer we take
    """
    profit = np.asarray(expected_profit, dtype=float)
    coverage = np.asarray(coverage, dtype=float)
    ranked = _ranked_eligible(profit, coverage, budget)
    is_chosen = np.zeros(len(profit), dtype=bool)
    if coverage[ranked].sum() <= budget:
        is_chosen[ranked] = True
        return is_chosen

    cut_off = int(np.searchsorted(np.cumsum(coverage[ranked]), budget, side="right"))
    core_start = max(0, cut_off - CORE_SIZE // 2)
    is_chosen[ranked[:core_start]] = True
    borderline = ranked[core_start : cut_off + CORE_SIZE // 2]
    budget_left = budget - coverage[is_chosen].sum()
    picks = _knapsack_table(profit[borderline], coverage[borderline], budget_left)
    is_chosen[borderline[picks]] = True
    is_chosen = _fill(is_chosen, ranked, coverage, budget)

    greedy = _fill(np.zeros(len(profit), dtype=bool), ranked, coverage, budget)
    return greedy if profit[greedy].sum() > profit[is_chosen].sum() else is_chosen
