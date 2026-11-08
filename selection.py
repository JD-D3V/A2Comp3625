"""Customer selection: choose which customers to insure under the coverage budget.

This is a 0/1 knapsack problem. Each customer either is or is not insured.
Their coverage uses up part of the budget and their expected profit is what we
gain. We want the most total expected profit without going over the budget.
"""

import numpy as np

BUDGET = 100_000_000


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


def choose_customers(expected_profit, coverage, budget=BUDGET):
    """Return a boolean mask of the customers to insure.

    :param expected_profit: expected profit for each customer
        (premium - claim probability * coverage)
    :param coverage: coverage amount for each customer (what we pay if they claim)
    :param budget: the most total coverage we can take on
    :return: numpy bool array, True for each customer we take
    """
    profit, coverage = _clean(expected_profit, coverage)
    is_chosen = np.zeros(len(profit), dtype=bool)
    eligible = _eligible(profit, coverage, budget)

    # If every profitable customer fits, take them all.
    if coverage[eligible].sum() <= budget:
        is_chosen[eligible] = True
        return is_chosen

    # Otherwise go down the ranking, taking each customer that still fits.
    budget_left = budget
    for customer in _rank(profit, coverage, eligible):
        if coverage[customer] <= budget_left:
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
