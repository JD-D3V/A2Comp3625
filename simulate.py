"""
simulator for choosing share_of_room.

Two experiments on the held-out test rows (80/20 split):
    1. Backtest:          price every customer with one rule, then see what we actually earn
                          once the real claims come in.
    2. Mock competition:  several pricing rules compete; each customer takes the cheapest
                          valid offer, and we total each rule's profit.

Until Darren's model is ready, every customer gets the stand-in probability STANDIN_PROB.
"""

import numpy as np
import pandas as pd

from metrics import log_loss
from pricing import set_prices, passes_utility_check

STANDIN_PROB = 0.335  # overall claim rate in the training data
SHARES_TO_TEST = [0.1, 0.3, 0.5, 0.7, 0.9]


def load_test_rows():
    """Return coverage and actual claims (0/1) for the test rows (every 5th row)."""
    customer_data = pd.read_csv("./data/customer_data.csv", index_col=None)
    customer_claims = pd.read_csv("./data/customer_claims.csv", index_col=None)
    test_mask = customer_data.index % 5 == 0
    coverage = customer_data.loc[test_mask, "coverage_amount"].to_numpy()
    claimed = customer_claims.loc[test_mask, "claim"].to_numpy()
    return coverage, claimed


def realized_profit(premium, coverage, claimed, taken):
    """
    What we earn from the customers we took on.
    :param premium:  premium per customer (array)
    :param coverage: coverage per customer (array)
    :param claimed:  1 if the customer actually claimed, else 0 (array)
    :param taken:    True for customers we insured (boolean array)
    :return: total profit
    """

    collected = np.sum(premium[taken])  # total $ value of premiums collected
    paid_out = np.sum(coverage[taken & (claimed == 1)])  # $ value of coverage paid out
    return collected - paid_out


def rival_offers(prob, coverage):
    """
    Pricing rules other teams might use. Offers that fail the utility check are set to inf
    because the customer would refuse them (so they can never be the cheapest valid offer).
    """
    emv = prob * coverage
    rivals = {
        "20% over cost": 1.2 * emv,
        "10% over cost": 1.1 * emv,
    }
    for name, premium in rivals.items():
        rivals[name] = np.where(
            passes_utility_check(premium, prob, coverage), premium, np.inf
        )
    return rivals


def mock_competition(offers, coverage, claimed):
    """
    Each customer takes the lowest offer. Returns {rule name: (num customers won, profit)}.
    Ties go to whichever rule is listed first.
    note: ignores the $100M budget for now; that's Joseph's selection step.
    """
    names = list(offers)
    all_offers = np.vstack(
        [offers[name] for name in names]
    )  # shape: (rules, customers)
    winner = np.argmin(all_offers, axis=0)  # index of cheapest rule per customer
    results = {}
    for i, name in enumerate(names):
        taken = winner == i
        results[name] = (
            int(taken.sum()),
            realized_profit(offers[name], coverage, claimed, taken),
        )
    return results


if __name__ == "__main__":
    coverage, claimed = load_test_rows()
    prob = np.full(len(coverage), STANDIN_PROB)
    everyone = np.ones(len(coverage), dtype=bool)

    print(f"Test rows: {len(coverage):,}   actual claim rate: {claimed.mean():.3f}")
    # prices are only as good as the probabilities behind them; lower log-loss = better probabilities
    print(f"Log-loss of prob used for pricing: {log_loss(claimed, prob):.4f}\n")

    print("1. Backtest (we insure every test customer, no competition)")
    print(f"{'share_of_room':<15}{'realized profit':>18}")
    for share in SHARES_TO_TEST:
        premium = set_prices(prob, coverage, share)
        print(
            f"{share:<15}{realized_profit(premium, coverage, claimed, everyone):>18,.0f}"
        )

    print(
        "\n2. Mock competition (each of our shares vs the rival rules, one at a time)"
    )
    print(f"{'share_of_room':<15}{'won':>8}{'our profit':>16}")
    for share in SHARES_TO_TEST:
        offers = {
            "ours": set_prices(prob, coverage, share),
            **rival_offers(prob, coverage),
        }
        won, profit = mock_competition(offers, coverage, claimed)["ours"]
        print(f"{share:<15}{won:>8,}{profit:>16,.0f}")
