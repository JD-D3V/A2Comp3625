"""Tests for selection.choose_customers and selection.profit_ceiling.

Designed with equivalence class partitioning (one test per group of inputs that
should behave the same) and boundary value analysis (values on and either side
of every edge). Every test also checks the two rules that must always hold:
never go over budget, and never pick a customer who loses money.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

import selection
from selection import choose_customers, profit_ceiling


def best_possible_profit(profit, coverage, budget):
    """Reference answer: try every combination of customers (small inputs only)."""
    every_combination = np.array(
        list(itertools.product([False, True], repeat=len(profit)))
    )
    total_coverage = every_combination @ coverage
    total_profit = every_combination @ profit
    return total_profit[total_coverage <= budget].max()


def greedy_profit(profit, coverage, budget):
    """Reference answer from plain greedy: best profit per dollar first."""
    profit = np.asarray(profit, dtype=float)
    coverage = np.asarray(coverage, dtype=float)
    is_usable = (profit > 0) & (coverage > 0) & (coverage <= budget)
    usable = np.flatnonzero(is_usable)
    ranked = usable[np.argsort(-profit[usable] / coverage[usable], kind="stable")]
    total = 0.0
    budget_left = budget
    for customer in ranked:
        if coverage[customer] <= budget_left:
            total += profit[customer]
            budget_left -= coverage[customer]
    return total


def assert_valid(is_chosen, profit, coverage, budget):
    """The rules every selection must follow, whatever the input."""
    profit = np.asarray(profit, dtype=float)
    coverage = np.asarray(coverage, dtype=float)
    assert isinstance(is_chosen, np.ndarray)
    assert is_chosen.dtype == bool
    assert len(is_chosen) == len(profit)
    assert coverage[is_chosen].sum() <= budget
    assert (profit[is_chosen] > 0).all()


# ---------------------------------------------------------------- profit edges


def test_profit_boundary():
    # Catches: using >= 0 instead of > 0, which would insure break-even customers.
    profit = np.array([-0.01, 0.0, 0.01])
    coverage = np.array([1_000, 1_000, 1_000])
    is_chosen = choose_customers(profit, coverage, budget=1_000_000)
    assert is_chosen.tolist() == [False, False, True]
    assert_valid(is_chosen, profit, coverage, 1_000_000)


def test_profit_not_finite():
    # Catches: NaN or infinite profit slipping through and wrecking the ranking.
    profit = np.array([np.nan, np.inf, -np.inf, 5.0])
    coverage = np.array([1_000, 1_000, 1_000, 1_000])
    is_chosen = choose_customers(profit, coverage, budget=1_000_000)
    assert is_chosen.tolist() == [False, False, False, True]


# -------------------------------------------------------------- coverage edges


def test_coverage_low_boundary():
    # Catches: zero or negative coverage counted as "free" profit.
    profit = np.array([10.0, 10.0, 10.0])
    coverage = np.array([-1, 0, 1])
    is_chosen = choose_customers(profit, coverage, budget=1_000_000)
    assert is_chosen.tolist() == [False, False, True]


@pytest.mark.parametrize(
    "coverage, expected",
    [(999, True), (1_000, True), (1_001, False)],
)
def test_single_customer_vs_budget(coverage, expected):
    # Catches: off-by-one (< instead of <=) when a policy uses the whole budget.
    is_chosen = choose_customers(np.array([5.0]), np.array([coverage]), budget=1_000)
    assert is_chosen.tolist() == [expected]


def test_coverage_nan():
    # Catches: NaN coverage passing the budget check (NaN comparisons are False).
    profit = np.array([10.0, 10.0])
    coverage = np.array([np.nan, 5_000])
    is_chosen = choose_customers(profit, coverage, budget=1_000_000)
    assert is_chosen.tolist() == [False, True]


# ------------------------------------------------- total coverage vs budget


@pytest.mark.parametrize(
    "last_coverage, expected",
    [
        (99, [True, True, True]),  # total = budget - 1: everyone fits
        (100, [True, True, True]),  # total = budget exactly: everyone fits
        (101, [True, True, False]),  # total = budget + 1: drop the weakest
    ],
)
def test_total_vs_budget(last_coverage, expected):
    # Catches: the "everyone fits" shortcut using < instead of <=, or dropping
    # the wrong customer when one has to go.
    profit = np.array([30.0, 20.0, 10.0])
    coverage = np.array([100, 100, last_coverage])
    is_chosen = choose_customers(profit, coverage, budget=300)
    assert is_chosen.tolist() == expected
    assert_valid(is_chosen, profit, coverage, 300)


# ------------------------------------------------------------- sizes, budget


@pytest.mark.parametrize("number_of_customers", [0, 1, 2])
def test_sizes(number_of_customers):
    # Catches: crashes on empty input or single-element arrays.
    profit = np.ones(number_of_customers)
    coverage = np.full(number_of_customers, 100.0)
    is_chosen = choose_customers(profit, coverage, budget=1_000)
    assert len(is_chosen) == number_of_customers
    assert_valid(is_chosen, profit, coverage, 1_000)
    assert is_chosen.all()


def test_budget_zero_and_one():
    # Catches: a zero budget still letting someone in, or budget 1 rejecting
    # a customer who fits exactly.
    one_customer_profit = np.array([1.0])
    one_customer_coverage = np.array([1])
    with_zero_budget = choose_customers(
        one_customer_profit, one_customer_coverage, budget=0
    )
    with_one_dollar_budget = choose_customers(
        one_customer_profit, one_customer_coverage, budget=1
    )
    assert with_zero_budget.tolist() == [False]
    assert with_one_dollar_budget.tolist() == [True]


# ----------------------------------------------------------- leftover space


@pytest.mark.parametrize(
    "customer_c_coverage, expected_c_chosen",
    [(400, True), (401, False)],
)
def test_fill_uses_exact_last_dollar(customer_c_coverage, expected_c_chosen):
    # Catches: the leftover-fill step wasting a gap that a customer fits exactly.
    # A (600) is taken, B (500) can't fit, then C fits only if it is <= 400.
    profit = np.array([600.0, 400.0, 100.0])
    coverage = np.array([600, 500, customer_c_coverage])
    is_chosen = choose_customers(profit, coverage, budget=1_000)
    assert is_chosen.tolist() == [True, False, expected_c_chosen]


# --------------------------------------------------------------- input types


@pytest.mark.parametrize(
    "convert_inputs",
    [
        lambda profit, coverage: (profit.astype(float), coverage.astype(float)),
        lambda profit, coverage: (profit.astype(float), coverage.astype(int)),
        lambda profit, coverage: (pd.Series(profit), pd.Series(coverage)),
        lambda profit, coverage: (profit.tolist(), coverage.tolist()),
    ],
    ids=["float arrays", "int coverage", "pandas Series", "plain lists"],
)
def test_input_types(convert_inputs):
    # Catches: code that only works for one array type (e.g. Series index bugs).
    random_generator = np.random.default_rng(0)
    profit = random_generator.uniform(1, 3_000, 200)
    coverage = random_generator.integers(1_000, 140_000, 200)
    expected = choose_customers(
        profit.astype(float), coverage.astype(float), budget=2_000_000
    )
    converted_profit, converted_coverage = convert_inputs(profit, coverage)
    is_chosen = choose_customers(converted_profit, converted_coverage, budget=2_000_000)
    assert is_chosen.tolist() == expected.tolist()


# ------------------------------------------------------- many random customers


def test_random_many():
    # Catches: any rule break on realistic data, and answers above the ceiling.
    random_generator = np.random.default_rng(1)
    profit = random_generator.uniform(-500, 3_000, 5_000)
    coverage = random_generator.integers(1_000, 140_000, 5_000)
    is_chosen = choose_customers(profit, coverage, budget=10_000_000)
    assert_valid(is_chosen, profit, coverage, 10_000_000)
    ceiling = profit_ceiling(profit, coverage, 10_000_000)
    assert profit[is_chosen].sum() <= ceiling + 1e-6


# ------------------------------------------------------------- profit ceiling


def test_profit_ceiling_everyone_fits():
    # Catches: the ceiling counting money-losing customers.
    ceiling = profit_ceiling([5.0, -1.0, 3.0], [1, 1, 1], budget=10)
    assert ceiling == pytest.approx(8)


def test_profit_ceiling_half_customer():
    # Catches: the ceiling ignoring the part of the next customer that would fit.
    ceiling = profit_ceiling([10.0, 6.0], [10, 10], budget=15)
    assert ceiling == pytest.approx(13)


# ================================================ Task 3: exact borderline search


def test_beats_greedy_on_the_game():
    # Catches: stopping at plain greedy. Greedy takes A then fills with D and E
    # ($5,050); the best answer is B and C together ($5,600).
    profit = np.array([3_600.0, 2_800.0, 2_800.0, 1_000.0, 450.0])
    coverage = np.array([120_000, 100_000, 100_000, 50_000, 30_000])
    is_chosen = choose_customers(profit, coverage, budget=200_000)
    assert is_chosen.tolist() == [False, True, True, False, False]
    assert selection.last_method == "exact"


def test_matches_brute_force():
    # Catches: any mistake in the exact search, by checking it against every
    # possible combination on 200 small random cases.
    random_generator = np.random.default_rng(3)
    for _ in range(200):
        profit = random_generator.uniform(1, 3_000, 12)
        coverage = random_generator.integers(1_000, 140_000, 12).astype(float)
        is_chosen = choose_customers(profit, coverage, budget=300_000)
        assert_valid(is_chosen, profit, coverage, 300_000)
        best = best_possible_profit(profit, coverage, 300_000)
        assert profit[is_chosen].sum() == pytest.approx(best)


def test_never_worse_than_greedy():
    # Catches: the exact search throwing away a customer greedy would keep.
    random_generator = np.random.default_rng(4)
    profit = random_generator.uniform(-200, 3_000, 3_000)
    coverage = random_generator.integers(1_000, 140_000, 3_000)
    is_chosen = choose_customers(profit, coverage, budget=20_000_000)
    assert_valid(is_chosen, profit, coverage, 20_000_000)
    greedy = greedy_profit(profit, coverage, 20_000_000)
    assert profit[is_chosen].sum() >= greedy - 1e-6
    assert profit[is_chosen].sum() <= profit_ceiling(profit, coverage, 20_000_000)


@pytest.mark.parametrize("offset_from_core_size", [-1, 0, 1])
def test_core_size_edges(offset_from_core_size):
    # Catches: index errors when the number of customers is just below, at, or
    # just above the size of the borderline group.
    number_of_customers = selection.CORE_SIZE + offset_from_core_size
    random_generator = np.random.default_rng(5)
    profit = random_generator.uniform(1, 3_000, number_of_customers)
    coverage = random_generator.integers(1_000, 140_000, number_of_customers)
    budget = coverage.sum() / 2
    is_chosen = choose_customers(profit, coverage, budget=budget)
    assert_valid(is_chosen, profit, coverage, budget)
    assert profit[is_chosen].sum() >= greedy_profit(profit, coverage, budget) - 1e-6


def test_cut_at_start_of_ranking():
    # Catches: index errors when the very first ranked customer already fills
    # the budget, so the borderline group starts at position 0.
    profit = np.array([5_000.0, 10.0, 9.0, 8.0])
    coverage = np.array([1_000_000, 400_000, 400_000, 400_000])
    is_chosen = choose_customers(profit, coverage, budget=1_000_000)
    assert is_chosen.tolist() == [True, False, False, False]


def test_cut_at_end_of_ranking():
    # Catches: index errors when everyone but the last ranked customer fits,
    # so the borderline group runs to the end of the list.
    profit = np.array([400.0, 300.0, 200.0, 1.0])
    coverage = np.array([100, 100, 100, 101])
    is_chosen = choose_customers(profit, coverage, budget=300)
    assert is_chosen.tolist() == [True, True, True, False]


def test_fill_after_borderline_group(monkeypatch):
    # Catches: forgetting the final fill step. With a borderline group of only
    # 2 (A and B), the search keeps A and leaves $400 unused; C ($300) and
    # D ($100) sit below the group and fill that gap exactly.
    monkeypatch.setattr(selection, "CORE_SIZE", 2)
    profit = np.array([600.0, 400.0, 60.0, 10.0])
    coverage = np.array([600, 500, 300, 100])
    is_chosen = choose_customers(profit, coverage, budget=1_000)
    assert is_chosen.tolist() == [True, False, True, True]


def test_state_limit_boundary():
    # Catches: switching to the safety net too early or too late.
    random_generator = np.random.default_rng(6)
    profit = random_generator.uniform(1, 3_000, 40)
    coverage = random_generator.integers(1_000, 140_000, 40).astype(float)
    budget = coverage.sum() / 2
    picks, peak_states = selection._exact_dollar_core(
        coverage, profit, budget, max_states=10**9
    )
    assert picks is not None
    at_limit, _ = selection._exact_dollar_core(
        coverage, profit, budget, max_states=peak_states
    )
    one_below, _ = selection._exact_dollar_core(
        coverage, profit, budget, max_states=peak_states - 1
    )
    assert at_limit is not None
    assert one_below is None


def test_safety_net_still_valid(monkeypatch):
    # Catches: the fallback table going over budget or doing badly. Every
    # customer is equally good per dollar, the hardest case for the exact search.
    monkeypatch.setattr(selection, "MAX_STATES", 1)
    random_generator = np.random.default_rng(7)
    coverage = random_generator.integers(1_000, 140_000, 2_000).astype(float)
    profit = coverage * 0.05
    is_chosen = choose_customers(profit, coverage, budget=10_000_000)
    assert selection.last_method == "table"
    assert_valid(is_chosen, profit, coverage, 10_000_000)
    ceiling = profit_ceiling(profit, coverage, 10_000_000)
    assert profit[is_chosen].sum() >= 0.999 * ceiling


def test_all_ratios_tied():
    # Catches: poor results when every customer is equally good per dollar.
    random_generator = np.random.default_rng(8)
    coverage = random_generator.integers(1_000, 140_000, 500).astype(float)
    profit = coverage * 0.05
    is_chosen = choose_customers(profit, coverage, budget=5_000_000)
    assert_valid(is_chosen, profit, coverage, 5_000_000)
    ceiling = profit_ceiling(profit, coverage, 5_000_000)
    assert profit[is_chosen].sum() >= 0.999 * ceiling


# ------------------------------------------- the search's helper functions


@pytest.mark.parametrize(
    "dearer_profit, dearer_is_kept",
    [(9.99, False), (10.0, False), (10.01, True)],
)
def test_keep_undominated_boundary(dearer_profit, dearer_is_kept):
    # Catches: keeping an option that costs more but earns no more (> vs >=).
    # A $50 option earns $10; a $100 option earns just under, equal, or above.
    option_coverage = np.array([0.0, 50.0, 100.0])
    option_profit = np.array([0.0, 10.0, dearer_profit])
    kept = selection._keep_undominated(option_coverage, option_profit)
    assert (2 in kept.tolist()) == dearer_is_kept
    assert kept.tolist()[:2] == [0, 1]


def test_keep_undominated_exact_tie_keeps_one():
    # Catches: keeping both of two identical options (same cost, same profit),
    # which doubles the work for nothing.
    option_coverage = np.array([0.0, 100.0, 100.0])
    option_profit = np.array([0.0, 2_800.0, 2_800.0])
    kept = selection._keep_undominated(option_coverage, option_profit)
    assert len(kept) == 2


def test_walk_back_follows_breadcrumbs():
    # Catches: walking the breadcrumbs the wrong way. Two customers; the four
    # final options are nobody, first only, second only, and both.
    history = [
        (np.array([0, 0]), np.array([False, True])),
        (np.array([0, 1, 0, 1]), np.array([False, False, True, True])),
    ]
    assert selection._walk_back(history, 0) == []
    assert selection._walk_back(history, 1) == [0]
    assert selection._walk_back(history, 2) == [1]
    assert sorted(selection._walk_back(history, 3)) == [0, 1]


@pytest.mark.parametrize(
    "option_profit, is_kept",
    [(53.99, False), (54.0, True), (54.01, True)],
)
def test_drop_hopeless_boundary(option_profit, is_kept):
    # Catches: dropping an option that could still tie or beat the best answer.
    # Budget 150. Option A (empty) can take the next customer: $60 for sure.
    # Option B has used $140, so at best it adds 10/100 of that customer ($6).
    # B is hopeless only if its profit + $6 is below A's $60.
    kept, best_known = selection._drop_hopeless(
        np.array([0.0, 140.0]),
        np.array([0.0, option_profit]),
        np.array([100.0]),
        np.array([60.0]),
        budget=150,
        best_known_profit=0.0,
    )
    assert best_known == pytest.approx(60.0)
    assert (1 in kept.tolist()) == is_kept
    assert 0 in kept.tolist()
