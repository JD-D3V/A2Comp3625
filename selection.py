"""Choose which customers to insure: the most expected profit within the budget
(a 0/1 knapsack problem).

1. Drop customers we could never take.
2. If everyone left fits, take them all.
3. Rank by profit per dollar; lock in clear winners above the cut-off.
4. Search the borderline group exactly (a table takes over if it gets too big).
5. Fill any leftover budget.
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
    """Positions of customers we could ever take: valid numbers, profit above 0,
    and coverage above 0 that fits the budget."""
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
    """Positions of options that earn more than every cheaper option."""
    order = np.lexsort((-option_profit, option_coverage))
    sorted_profit = option_profit[order]
    best_cheaper_profit = np.maximum.accumulate(
        np.concatenate([[-np.inf], sorted_profit[:-1]])
    )
    return order[sorted_profit > best_cheaper_profit]


def _walk_back(history, best_option):
    """Follow the breadcrumbs back from the best option to list who it took."""
    chosen_positions = []
    option = best_option
    for position in range(len(history) - 1, -1, -1):
        came_from, took_customer = history[position]
        if took_customer[option]:
            chosen_positions.append(position)
        option = came_from[option]
    return chosen_positions


def _drop_hopeless(
    option_coverage,
    option_profit,
    upcoming_coverage,
    upcoming_profit,
    budget,
    best_known_profit,
):
    """Drop options that can't beat the best answer found so far, even if the
    rest of the budget fills in the best possible way.

    :return: (positions of the options to keep, updated best known profit)
    """
    coverage_so_far = np.concatenate([[0.0], np.cumsum(upcoming_coverage)])
    profit_so_far = np.concatenate([[0.0], np.cumsum(upcoming_profit)])
    budget_left = budget - option_coverage
    whole_customers = np.searchsorted(coverage_so_far, budget_left, side="right") - 1
    guaranteed_profit = option_profit + profit_so_far[whole_customers]
    best_known_profit = max(best_known_profit, guaranteed_profit.max())

    partial_next_customer = np.zeros(len(option_coverage))
    has_next = whole_customers < len(upcoming_coverage)
    next_customer = whole_customers[has_next]
    partial_next_customer[has_next] = (
        (budget_left[has_next] - coverage_so_far[next_customer])
        * upcoming_profit[next_customer]
        / upcoming_coverage[next_customer]
    )
    optimistic_profit = guaranteed_profit + partial_next_customer
    is_hopeful = optimistic_profit >= best_known_profit - 1e-6
    return np.flatnonzero(is_hopeful), best_known_profit


def _exact_dollar_core(coverage, profit, budget, max_states=None):
    """Exact search over ranked customers, keeping a list of "options"
    (groups picked so far) and dropping dominated and hopeless ones.

    :return: (chosen positions, most options kept), or (None, most options kept)
        if the list grew past max_states
    """
    if max_states is None:
        max_states = MAX_STATES
    option_coverage = np.array([0.0])
    option_profit = np.array([0.0])
    history = []  # for each customer: (which earlier option, did we take them?)
    best_known_profit = 0.0
    peak_states = 1

    for position in range(len(coverage)):
        all_coverage, all_profit, came_from, took_customer = _add_customer(
            option_coverage, option_profit, coverage[position], profit[position], budget
        )
        keep = _keep_undominated(all_coverage, all_profit)
        option_coverage, option_profit = all_coverage[keep], all_profit[keep]
        came_from, took_customer = came_from[keep], took_customer[keep]

        hopeful, best_known_profit = _drop_hopeless(
            option_coverage,
            option_profit,
            coverage[position + 1 :],
            profit[position + 1 :],
            budget,
            best_known_profit,
        )
        option_coverage = option_coverage[hopeful]
        option_profit = option_profit[hopeful]
        history.append((came_from[hopeful], took_customer[hopeful]))
        peak_states = max(peak_states, len(option_coverage))
        if peak_states > max_states:
            return None, peak_states

    best_option = int(np.argmax(option_profit))
    return _walk_back(history, best_option), peak_states


def _unit_table(coverage, profit, budget):
    """Safety net: the classic knapsack table, with coverage rounded UP to
    $50+ blocks so it stays small and can never go over budget.

    :return: positions of the chosen customers
    """
    dollars_per_cell = max(50.0, budget / TABLE_CELLS)
    cells_needed = np.ceil(coverage / dollars_per_cell).astype(int)
    number_of_cells = int(budget // dollars_per_cell)
    best_profit_using = np.zeros(number_of_cells + 1)
    took_history = []
    for position in range(len(coverage)):
        with_customer = np.full(number_of_cells + 1, -np.inf)
        customer_cells = cells_needed[position]
        if customer_cells <= number_of_cells:
            with_customer[customer_cells:] = (
                best_profit_using[: number_of_cells + 1 - customer_cells]
                + profit[position]
            )
        took = with_customer > best_profit_using
        best_profit_using = np.where(took, with_customer, best_profit_using)
        took_history.append(np.packbits(took))

    chosen_positions = []
    cells_left = number_of_cells
    for position in range(len(coverage) - 1, -1, -1):
        took = np.unpackbits(took_history[position], count=number_of_cells + 1)
        if took[cells_left]:
            chosen_positions.append(position)
            cells_left -= cells_needed[position]
    return chosen_positions


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

    # The borderline group gets the exact search, or the table if it's too big.
    borderline = ranked[core_start:core_end]
    budget_for_borderline = budget - coverage[locked_in].sum()
    picks, _ = _exact_dollar_core(
        coverage[borderline], profit[borderline], budget_for_borderline
    )
    last_method = "exact"
    if picks is None:
        picks = _unit_table(
            coverage[borderline], profit[borderline], budget_for_borderline
        )
        last_method = "table"
    is_chosen[borderline[picks]] = True

    # Spend any budget still left on the best customers that still fit.
    budget_left = budget - coverage[is_chosen].sum()
    for customer in ranked[core_start:]:
        if not is_chosen[customer] and coverage[customer] <= budget_left:
            is_chosen[customer] = True
            budget_left -= coverage[customer]
    return is_chosen


def profit_ceiling(expected_profit, coverage, budget=BUDGET):
    """The most profit any selection could make, if we could insure part of the
    first customer who doesn't fully fit. No real selection can beat it."""
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
