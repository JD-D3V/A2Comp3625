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


def _add_customer(
    option_coverage, option_profit, customer_coverage, customer_profit, budget
):
    """Every option can skip this customer, or take them if they still fit.

    :return: (coverage, profit, came_from, took_customer) for all the new options
    """
    can_take = option_coverage + customer_coverage <= budget
    new_coverage = np.concatenate(
        [option_coverage, option_coverage[can_take] + customer_coverage]
    )
    new_profit = np.concatenate(
        [option_profit, option_profit[can_take] + customer_profit]
    )
    came_from = np.concatenate(
        [np.arange(len(option_coverage)), np.flatnonzero(can_take)]
    )
    took_customer = np.concatenate(
        [np.zeros(len(option_coverage), bool), np.ones(can_take.sum(), bool)]
    )
    return new_coverage, new_profit, came_from, took_customer


def _keep_undominated(option_coverage, option_profit):
    """Positions of the options worth keeping: sorted cheapest first, each one
    earning more than every cheaper option."""
    order = np.lexsort((-option_profit, option_coverage))
    sorted_profit = option_profit[order]
    best_cheaper_profit = np.maximum.accumulate(
        np.concatenate([[-np.inf], sorted_profit[:-1]])
    )
    return order[sorted_profit > best_cheaper_profit]


def _walk_back(history, best_option):
    """Follow the breadcrumbs from the best final option back to the first
    customer, listing every customer that option took."""
    chosen_positions = []
    option = best_option
    for position in range(len(history) - 1, -1, -1):
        came_from, took_customer = history[position]
        if took_customer[option]:
            chosen_positions.append(position)
        option = came_from[option]
    return chosen_positions


def _exact_dollar_core(coverage, profit, budget, max_states=None):
    """Find the most profitable group of the given customers within the budget.

    The customers must already be ranked best first. We go through them one at
    a time and keep a list of "options" (groups picked so far). Dominated
    options are dropped, and each option remembers where it came from so we
    can walk back at the end and see who the best one took.

    :return: (positions of the chosen customers, largest number of options kept)
    """
    if max_states is None:
        max_states = MAX_STATES
    option_coverage = np.array([0.0])
    option_profit = np.array([0.0])
    history = []  # for each customer: (which earlier option, did we take them?)
    peak_states = 1

    for position in range(len(coverage)):
        all_coverage, all_profit, came_from, took_customer = _add_customer(
            option_coverage, option_profit, coverage[position], profit[position], budget
        )
        keep = _keep_undominated(all_coverage, all_profit)
        option_coverage, option_profit = all_coverage[keep], all_profit[keep]
        history.append((came_from[keep], took_customer[keep]))
        peak_states = max(peak_states, len(option_coverage))

    best_option = int(np.argmax(option_profit))
    return _walk_back(history, best_option), peak_states


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

    # The borderline group gets the exact search.
    borderline = ranked[core_start:core_end]
    budget_for_borderline = budget - coverage[locked_in].sum()
    picks, _ = _exact_dollar_core(
        coverage[borderline], profit[borderline], budget_for_borderline
    )
    last_method = "exact"
    is_chosen[borderline[picks]] = True

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
