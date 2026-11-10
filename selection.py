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
    has_valid_numbers = np.isfinite(profit) & np.isfinite(coverage)
    is_worth_taking = (profit > 0) & (coverage > 0) & (coverage <= budget)
    customers = np.flatnonzero(has_valid_numbers & is_worth_taking)

    profit_per_dollar = profit[customers] / coverage[customers]
    best_first = np.argsort(-profit_per_dollar, kind="stable")
    return customers[best_first]


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
        blocks_needed = size[position]
        if blocks_needed > cells:
            continue  # this customer alone is bigger than the budget left

        # Taking this customer: their profit plus the best we could already do
        # with blocks_needed fewer blocks. Cells too small for them stay at -inf.
        with_customer = np.full(cells + 1, -np.inf)
        best_with_room_left = best_profit[: cells + 1 - blocks_needed]
        with_customer[blocks_needed:] = best_with_room_left + profit[position]

        took[position] = with_customer > best_profit
        best_profit = np.maximum(best_profit, with_customer)

    # Walk back from the last customer to read off who was taken.
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

    # Where the budget runs out if we take customers in ranked order.
    coverage_so_far = np.cumsum(coverage[ranked])
    cut_off = int(np.searchsorted(coverage_so_far, budget, side="right"))

    # Customers well above the cut-off are locked in. The CORE_SIZE customers
    # around it are the borderline ones the table decides between.
    half_core = CORE_SIZE // 2
    core_start = max(0, cut_off - half_core)
    core_end = cut_off + half_core
    is_chosen[ranked[:core_start]] = True
    borderline = ranked[core_start:core_end]

    budget_left = budget - coverage[is_chosen].sum()
    picks = _knapsack_table(profit[borderline], coverage[borderline], budget_left)
    is_chosen[borderline[picks]] = True
    is_chosen = _fill(is_chosen, ranked, coverage, budget)

    greedy = _fill(np.zeros(len(profit), dtype=bool), ranked, coverage, budget)
    if profit[greedy].sum() > profit[is_chosen].sum():
        return greedy
    return is_chosen
