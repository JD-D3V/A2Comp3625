from pathlib import Path

import numpy as np
import pandas as pd

from calibration import Calibration
from claim_model import ClaimModel
from pricing import set_prices, utility
from selection import choose_customers

# Found relative to this file, so it works whatever folder the grader runs from.
MODEL_PATH = Path(__file__).resolve().parent / "saved_model.npz"
CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.npz"

SHARE_OF_ROOM = 0.5  # price halfway between break-even and the customer's max
MAX_COVERAGE = 150_000  # the utility curve's ln($ + 150,000) breaks at this loss


def _load_model():
    """Load the claim model trained and saved by model_dev.py."""
    return ClaimModel.load(MODEL_PATH)


def _load_calibration():
    """Load the probability correction learned by calibration.py."""
    return Calibration.load(CALIBRATION_PATH)


# At grading time, I will call the below function with a dataframe of new potential customers
# The dataframe I pass in will have the same columns as customer_data.csv, but will not include any of the customers in customer_data.csv
# You will not be given any of the historical customer data or claims data at grading time,
# so you will need to save any models you train in model_dev.py and load them here.


def set_premium(customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a dataframe of data on potential customers, identifies which customers to take on, and sets premiums for them.
    This function may load a saved model (previously trained using supervised learning) to save time.
    :param customer_data: a dataframe with the same columns as customer_data.csv
    :return: a DataFrame of the same lenght as customer_data, with the following columns:
            p_claim:  your estimated probability that each customer in customer_data will make a claim
            desired_customer:  a boolean column indicating which customers you want to take on
            premium:    the amount of the premium to charge each customer
    """
    # The model is off by risk band (too low for the middle band selection
    # favours), so correct it before pricing, the utility check and selection.
    model_probability = _load_model().predict_proba(customer_data)
    claim_probability = _load_calibration().apply(model_probability)
    coverage = customer_data["coverage_amount"].to_numpy(dtype=float)

    # Only price customers the utility formula can handle: a real probability,
    # at least $1 of coverage, and a loss the ln($ + 150,000) curve can take.
    is_priceable = (
        (claim_probability > 0)
        & (claim_probability < 1)
        & (coverage >= 1)
        & (coverage < MAX_COVERAGE)
    )
    premium = np.nan_to_num(claim_probability * coverage)
    premium[is_priceable] = set_prices(
        claim_probability[is_priceable], coverage[is_priceable], SHARE_OF_ROOM
    )

    # The utility check: a customer only buys if paying the premium feels better
    # than the gamble of going uninsured. The errstate line silences warnings
    # from customers we skipped (their ln() can be undefined).
    with np.errstate(invalid="ignore", divide="ignore"):
        utility_if_insured = utility(-premium)
        utility_if_uninsured = claim_probability * utility(-coverage) + (
            1 - claim_probability
        ) * utility(0)
    is_better_off_insured = is_priceable & (utility_if_insured > utility_if_uninsured)

    # Customers who won't buy are worth nothing to us, so choose_customers skips them.
    expected_profit = np.where(
        is_better_off_insured, premium - claim_probability * coverage, 0.0
    )
    is_desired = choose_customers(expected_profit, coverage)

    return pd.DataFrame(
        {
            "p_claim": claim_probability,
            "desired_customer": is_desired,
            "premium": premium,
        },
        index=customer_data.index,
    )
