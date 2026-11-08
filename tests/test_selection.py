"""Tests for selection.choose_customers and selection.profit_ceiling.

Designed with equivalence class partitioning (one test per group of inputs that
should behave the same) and boundary value analysis (values on and either side
of every edge). Every test also checks the two rules that must always hold:
never go over budget, and never pick a customer who loses money.
"""

import numpy as np
import pandas as pd
import pytest

from selection import choose_customers, profit_ceiling


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
