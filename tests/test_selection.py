"""Tests for selection.choose_customers.

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
from selection import choose_customers


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


def profit_ceiling(profit, coverage, budget):
    """The most profit any selection could make, if we could insure part of the
    first customer who doesn't fully fit. No real selection can beat it."""
    profit = np.asarray(profit, dtype=float)
    coverage = np.asarray(coverage, dtype=float)
    is_usable = (profit > 0) & (coverage > 0) & (coverage <= budget)
    usable = np.flatnonzero(is_usable)
    ranked = usable[np.argsort(-profit[usable] / coverage[usable], kind="stable")]
    total = 0.0
    budget_left = budget
    for customer in ranked:
        if coverage[customer] > budget_left:
            return total + profit[customer] * budget_left / coverage[customer]
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


# ------------------------------------------------- who can never be taken


@pytest.mark.parametrize(
    "profit, coverage, expected",
    [
        ([-0.01, 0.0, 0.01], [1_000] * 3, [False, False, True]),
        ([np.nan, np.inf, -np.inf, 5.0], [1_000] * 4, [False, False, False, True]),
        ([10.0] * 3, [-1, 0, 1], [False, False, True]),
        ([10.0] * 2, [np.nan, 5_000], [False, True]),
    ],
    ids=["profit at zero", "profit not finite", "coverage at zero", "coverage NaN"],
)
def test_never_takes_impossible_customers(profit, coverage, expected):
    # Catches: >= 0 instead of > 0 (insuring break-even customers), NaN or
    # infinite profit wrecking the ranking, zero or negative coverage counted as
    # "free" profit, and NaN coverage passing the budget check (NaN compares False).
    profit, coverage = np.array(profit), np.array(coverage, dtype=float)
    is_chosen = choose_customers(profit, coverage, budget=1_000_000)
    assert is_chosen.tolist() == expected


@pytest.mark.parametrize(
    "coverage, expected",
    [(999, True), (1_000, True), (1_001, False)],
)
def test_single_customer_vs_budget(coverage, expected):
    # Catches: off-by-one (< instead of <=) when a policy uses the whole budget.
    is_chosen = choose_customers(np.array([5.0]), np.array([coverage]), budget=1_000)
    assert is_chosen.tolist() == [expected]


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


# ------------------------------------------------- the knapsack table search


def test_beats_greedy_on_the_game():
    # Catches: stopping at plain greedy. Greedy takes A then fills with D and E
    # ($5,050); the best answer is B and C together ($5,600).
    profit = np.array([3_600.0, 2_800.0, 2_800.0, 1_000.0, 450.0])
    coverage = np.array([120_000, 100_000, 100_000, 50_000, 30_000])
    is_chosen = choose_customers(profit, coverage, budget=200_000)
    assert is_chosen.tolist() == [False, True, True, False, False]


def test_matches_brute_force_on_whole_blocks():
    # Catches: any mistake in the table, by checking it against every possible
    # combination on 200 small random cases. Coverage is in whole $25 blocks,
    # so the table's rounding changes nothing and it must find the best answer.
    random_generator = np.random.default_rng(3)
    block = selection.DOLLARS_PER_CELL
    for _ in range(200):
        profit = random_generator.uniform(1, 3_000, 12)
        coverage = random_generator.integers(40, 5_600, 12).astype(float) * block
        is_chosen = choose_customers(profit, coverage, budget=300_000)
        assert_valid(is_chosen, profit, coverage, 300_000)
        best = best_possible_profit(profit, coverage, 300_000)
        assert profit[is_chosen].sum() == pytest.approx(best)


def test_close_to_brute_force_with_rounding():
    # Catches: the $25 rounding costing more than a sliver. With any coverage,
    # the answer must be within 1% of the true best and never above it.
    random_generator = np.random.default_rng(13)
    for _ in range(100):
        profit = random_generator.uniform(1, 3_000, 12)
        coverage = random_generator.integers(1_000, 140_000, 12).astype(float)
        is_chosen = choose_customers(profit, coverage, budget=300_000)
        assert_valid(is_chosen, profit, coverage, 300_000)
        best = best_possible_profit(profit, coverage, 300_000)
        assert 0.99 * best <= profit[is_chosen].sum() <= best + 1e-6


def test_never_worse_than_greedy():
    # Catches: any rule break on realistic data, answers above the ceiling, and
    # the table throwing away a customer greedy would keep.
    random_generator = np.random.default_rng(4)
    profit = random_generator.uniform(-200, 3_000, 3_000)
    coverage = random_generator.integers(1_000, 140_000, 3_000)
    is_chosen = choose_customers(profit, coverage, budget=20_000_000)
    assert_valid(is_chosen, profit, coverage, 20_000_000)
    greedy = greedy_profit(profit, coverage, 20_000_000)
    assert profit[is_chosen].sum() >= greedy - 1e-6
    assert profit[is_chosen].sum() <= profit_ceiling(profit, coverage, 20_000_000)


def test_keeps_greedy_when_it_earns_more(monkeypatch):
    # Catches: the final greedy check missing. A table that picks nobody must
    # still end up with greedy's answer.
    monkeypatch.setattr(selection, "_knapsack_table", lambda *args: [])
    random_generator = np.random.default_rng(10)
    profit = random_generator.uniform(1, 3_000, 500)
    coverage = random_generator.integers(1_000, 140_000, 500)
    is_chosen = choose_customers(profit, coverage, budget=5_000_000)
    assert profit[is_chosen].sum() == pytest.approx(
        greedy_profit(profit, coverage, 5_000_000)
    )


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
    # 2 (A and B), the table keeps A and leaves $400 unused; C ($300) and
    # D ($100) sit below the group and fill that gap exactly.
    monkeypatch.setattr(selection, "CORE_SIZE", 2)
    profit = np.array([600.0, 400.0, 60.0, 10.0])
    coverage = np.array([600, 500, 300, 100])
    is_chosen = choose_customers(profit, coverage, budget=1_000)
    assert is_chosen.tolist() == [True, False, True, True]


def test_all_ratios_tied():
    # Catches: poor results when every customer is equally good per dollar.
    random_generator = np.random.default_rng(8)
    coverage = random_generator.integers(1_000, 140_000, 500).astype(float)
    profit = coverage * 0.05
    is_chosen = choose_customers(profit, coverage, budget=5_000_000)
    assert_valid(is_chosen, profit, coverage, 5_000_000)
    ceiling = profit_ceiling(profit, coverage, 5_000_000)
    assert profit[is_chosen].sum() >= 0.999 * ceiling


# ------------------------------------------------- the table function itself


def test_knapsack_table_skip_or_take():
    # Catches: the skip-or-take rule reading the wrong cell. In whole blocks,
    # A is 3 blocks ($4.2), B and C are 2 blocks ($2.6 each), budget 4 blocks:
    # greedy would take A alone ($4.2); B and C together earn $5.2.
    block = selection.DOLLARS_PER_CELL
    picks = selection._knapsack_table(
        np.array([4.2, 2.6, 2.6]), np.array([3, 2, 2]) * block, 4 * block
    )
    assert sorted(picks) == [1, 2]


@pytest.mark.parametrize(
    "coverage_each, both_taken",
    [(24, True), (25, True), (26, False)],
)
def test_knapsack_table_rounds_up(coverage_each, both_taken):
    # Catches: rounding coverage down (or to nearest), which could overspend.
    # Budget is two $25 blocks; $26 needs two blocks each, so only one fits.
    picks = selection._knapsack_table(
        np.array([10.0, 9.0]), np.array([coverage_each] * 2, dtype=float), 50
    )
    assert (len(picks) == 2) == both_taken


def test_knapsack_table_customer_bigger_than_budget():
    # Catches: index errors when one customer needs more blocks than exist.
    picks = selection._knapsack_table(
        np.array([100.0, 1.0]), np.array([1_000.0, 25.0]), 100
    )
    assert picks == [1]
