"""Customer selection: choose which customers to insure under the coverage budget.

This is a 0/1 knapsack problem. Each customer either is or is not insured.
Their coverage uses up part of the budget and their expected profit is what we
gain. We want the most total expected profit without going over the budget.
"""

import numpy as np

BUDGET = 100_000_000
CORE_SIZE = 300  # how many borderline customers get the exact search
MAX_STATES = 200_000  # if the exact search keeps more options than this, fall back
TABLE_CELLS = 200_000  # the fallback table never has more cells than this

last_method = ""  # which path the last call used: "all fit", "exact" or "table"


def _clean(expected_profit, coverage):
    """Turn any array-like input into float arrays."""
    profit = np.asarray(expected_profit, dtype=float).ravel()
    coverage = np.asarray(coverage, dtype=float).ravel()
    return profit, coverage


def _eligible(profit, coverage, budget):
    """Positions of customers we could ever take: real numbers, a profit above
    zero, and a coverage above zero that fits in the budget on its own."""
    is_eligible = np.isfinite(profit) & np.isfinite(coverage)
    is_eligible &= (profit > 0) & (coverage > 0) & (coverage <= budget)
    return np.flatnonzero(is_eligible)


def _rank(profit, coverage, customers):
    """Sort customers from best to worst profit per dollar of coverage."""
    profit_per_dollar = profit[customers] / coverage[customers]
    return customers[np.argsort(-profit_per_dollar, kind="stable")]


def _exact_dollar_core(coverage, profit, budget, max_states=None):
    """Find the most profitable group of the given customers within the budget.

    The customers must already be ranked best first. We go through them one at
    a time and keep a list of "options": each option is a group of customers
    picked so far, stored as its total coverage and total profit.
    """
    if max_states is None:
        max_states = MAX_STATES
    option_coverage = np.array([0.0])
    option_profit = np.array([0.0])
    peak_states = 1

    for position in range(len(coverage)):
        # Every option can skip this customer, or take them if they still fit.
        can_take = option_coverage + coverage[position] <= budget
        all_coverage = np.concatenate(
            [option_coverage, option_coverage[can_take] + coverage[position]]
        )
        all_profit = np.concatenate(
            [option_profit, option_profit[can_take] + profit[position]]
        )

        # Drop dominated options: sort by coverage, keep an option only if it
        # earns more than every cheaper option.
        order = np.lexsort((-all_profit, all_coverage))
        all_coverage, all_profit = all_coverage[order], all_profit[order]
        best_cheaper_profit = np.maximum.accumulate(
            np.concatenate([[-np.inf], all_profit[:-1]])
        )
        is_useful = all_profit > best_cheaper_profit
        option_coverage = all_coverage[is_useful]
        option_profit = all_profit[is_useful]
        peak_states = max(peak_states, len(option_coverage))

    return option_profit.max(), peak_states


def choose_customers(expected_profit, coverage, budget=BUDGET):
    """Return a boolean mask of the customers to insure.

    :param expected_profit: expected profit for each customer
        (premium - claim probability * coverage)
    :param coverage: coverage amount for each customer (what we pay if they claim)
    :param budget: the most total coverage we can take on
    :return: numpy bool array, True for each customer we take
    """
    global last_method
    profit, coverage = _clean(expected_profit, coverage)
    is_chosen = np.zeros(len(profit), dtype=bool)
    eligible = _eligible(profit, coverage, budget)

    # If every profitable customer fits, take them all.
    if coverage[eligible].sum() <= budget:
        is_chosen[eligible] = True
        last_method = "all fit"
        return is_chosen

    # Find the cut-off: how far down the ranking we get before the budget runs out.
    ranked = _rank(profit, coverage, eligible)
    running_coverage = np.cumsum(coverage[ranked])
    cut_off = int(np.searchsorted(running_coverage, budget, side="right"))
    core_start = max(0, cut_off - CORE_SIZE // 2)
    core_end = min(len(ranked), cut_off + CORE_SIZE // 2)

    # Clear winners above the borderline group are locked in.
    locked_in = ranked[:core_start]
    is_chosen[locked_in] = True

    # Spend any budget still left on the best customers that still fit.
    budget_left = budget - coverage[is_chosen].sum()
    for customer in ranked[core_start:]:
        if not is_chosen[customer] and coverage[customer] <= budget_left:
            is_chosen[customer] = True
            budget_left -= coverage[customer]
    return is_chosen


def profit_ceiling(expected_profit, coverage, budget=BUDGET):
    """The most profit any selection could possibly make.

    It pretends we could insure part of a customer: fill the budget in ranking
    order and take the fraction of the first customer who doesn't fully fit.
    No real (whole-customer) selection can beat this number.
    """
    profit, coverage = _clean(expected_profit, coverage)
    ranked = _rank(profit, coverage, _eligible(profit, coverage, budget))
    total = 0.0
    budget_left = budget
    for customer in ranked:
        if coverage[customer] <= budget_left:
            total += profit[customer]
            budget_left -= coverage[customer]
        else:
            fraction = budget_left / coverage[customer]
            return total + profit[customer] * fraction
    return total
