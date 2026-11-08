"""
Part C: utility and pricing.

Every function works on single numbers or on whole numpy arrays, and so on one customer,
or on multiple customers

Notation:
    mon_change  = change in the customer's money
    prob  = probability the customer makes a claim (from the claim model)
    coverage  = coverage amount (same as the damages amount the customer would take if uninsured)
"""
import numpy as np

# utility function from the assignment: U($) = -263.31 + 22.09 * ln($ + 150,000)
U_OFFSET = -263.31
U_SCALE = 22.09
BASE_WEALTH = 150_000  # wealth the customer starts with


def utility(mon_change):
    """How much a change in money (mon_change) is worth to the customer."""
    return U_OFFSET + U_SCALE * np.log(mon_change + BASE_WEALTH)


def inverse_utility(util_value):
    """The change in money whose utility is util_value (undoes utility())."""
    return np.exp((util_value - U_OFFSET) / U_SCALE) - BASE_WEALTH


def eu_uninsured(prob, coverage):
    """
    Expected utility of going uninsured: a lottery with two outcomes.
        with probability prob:     disaster, the customer loses coverage
        with probability 1 - prob: nothing happens, the customer loses nothing
    """
    return (prob * utility(-coverage)) + ((1 - prob) * utility(0))


def max_premium(prob, coverage):
    """
    The highest premium the customer would accept: the sure loss that feels as bad
    as the uninsured lottery. Above this, U(-premium) < EU(uninsured) and they refuse.
    """
    return -inverse_utility(eu_uninsured(prob, coverage))


def passes_utility_check(premium, prob, coverage):
    """The assignment's required check: buying must give strictly higher utility than going uninsured."""
    return utility(-premium) > eu_uninsured(prob, coverage)


def set_prices(prob, coverage, share_of_room):
    """
    Set each customer's premium somewhere inside their price window.

    The window runs from break-even (emv = prob * coverage, the expected payout, where we
    make $0 on average) up to max_premium (the most the customer would accept).
    share_of_room picks the point inside the window:
        0    : charge break-even (zero expected profit)
        0.5  : halfway between break-even and the customer's max
        -> 1 : close to the max (most profit per customer)

    :param prob: estimated claim probability per customer (number or numpy array)
    :param coverage: coverage amount per customer (number or numpy array)
    :param share_of_room: where to price inside the window; must satisfy 0 <= share_of_room < 1
                          (1 would equal max_premium, which leaves the customer indifferent)
    :return: premium per customer, guaranteed to pass passes_utility_check()
    :raises ValueError: if share_of_room is out of range, or any premium fails the utility check
    """
    if not (0 <= share_of_room < 1):
        raise ValueError(f"share_of_room must be in [0, 1), got {share_of_room}")

    emv = prob * coverage   # expected monetary value
    premium = emv + share_of_room * (max_premium(prob, coverage) - emv)

    # buying must give the customer higher utility than going uninsured
    if not np.all(passes_utility_check(premium, prob, coverage)):
        raise ValueError("a premium fails the utility check")
    return premium


if __name__ == "__main__":
    # example: prob = 0.33, coverage/damages = $70,000
    # Expected: U(0) ~ -0.03, U(-70k) ~ -13.92, EU(uninsured) ~ -4.62, max premium ~ $28,100
    prob, coverage = 0.33, 70_000
    print(f"Example customer: prob = {prob}, coverage = ${coverage:,}\n")

    rows = [
        ("U(0)  (no loss)", f"{utility(0):.2f}"),
        ("U(-coverage)  (disaster)", f"{utility(-coverage):.2f}"),
        ("EU(uninsured)", f"{eu_uninsured(prob, coverage):.2f}"),
        ("Break-even (prob * coverage)", f"${prob * coverage:,.0f}"),
        ("Max premium", f"${max_premium(prob, coverage):,.0f}"),
    ]
    for label, value in rows:
        print(f"{label:<30}{value:>10}")

    print(f"\n{'Premium':<10}{'U(-premium)':>12}{'Buys?':>8}")
    for premium in (26_000, 30_000):
        buys = "yes" if passes_utility_check(premium, prob, coverage) else "no"
        print(f"${premium:<9,}{utility(-premium):>12.2f}{buys:>8}")
