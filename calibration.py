"""Correct the claim model's probabilities so they match real claim rates.

The model is right on average but off by risk band: it says 0.45 where about half
of customers really claim, and 0.71 where about 0.62 do. Selection picks mostly
from the middle band, where the model is too low, so we expected more profit than
we got. This file learns a map from the model's probability to the claim rate
actually seen, and set_premium applies it.

Method: a binned table, chosen because it is simple to explain and it fit best.
1. Sort customers by the model's probability and cut them into 20 equal groups.
2. For each group note the average predicted p and the share who claimed.
3. If a riskier group shows a lower claim rate than the one before, merge the two
   (pool adjacent violators), so a higher prediction never means a lower one after.
4. Between group centres, draw straight lines; past the end groups, stay flat.

On training rows, 20 groups beat Platt scaling and plain isotonic regression in
cross-validated log-loss.

The table is learned only from training rows, using out-of-fold predictions: each
row's probability comes from a model that did not see that row, the way new
customers will look at grading time.
"""

import numpy as np

from claim_model import ClaimModel

GROUPS = 20  # about 4,000 training customers per group
FOLDS = 5


def out_of_fold_predictions(X, y, features):
    """Claim probability for each row from a ClaimModel trained without that row."""
    fold = np.arange(len(X)) % FOLDS
    predictions = np.zeros(len(X))
    for f in range(FOLDS):
        model = ClaimModel().fit(X[fold != f], y[fold != f], features)
        predictions[fold == f] = model.predict_proba(X[fold == f])
    return predictions


def _pool_adjacent_violators(rates, sizes):
    """Merge neighbouring groups until the claim rates never go down.

    A merged group's rate is the size-weighted average of the groups in it.
    Returns one rate per original group.
    """
    merged_rates, merged_sizes, members = [], [], []
    for rate, size in zip(rates, sizes):
        merged_rates.append(rate)
        merged_sizes.append(size)
        members.append(1)
        while len(merged_rates) > 1 and merged_rates[-2] >= merged_rates[-1]:
            rate, size, count = merged_rates.pop(), merged_sizes.pop(), members.pop()
            total = merged_sizes[-1] + size
            merged_rates[-1] = (
                merged_rates[-1] * merged_sizes[-1] + rate * size
            ) / total
            merged_sizes[-1] = total
            members[-1] += count
    return np.repeat(merged_rates, members)


class Calibration:
    def __init__(self):
        self.predicted = None  # average model p in each group (the table's x)
        self.actual = None  # smoothed claim rate in each group (the table's y)

    def fit(self, p, y, groups=GROUPS):
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        groups = np.array_split(np.argsort(p), groups)
        self.predicted = np.array([p[g].mean() for g in groups])
        sizes = np.array([len(g) for g in groups], dtype=float)
        # +1 claim and +1 non-claim per group keeps every rate strictly inside
        # (0, 1), even for a group where nobody (or everybody) claimed.
        rates = (np.array([y[g].sum() for g in groups]) + 1) / (sizes + 2)
        self.actual = _pool_adjacent_violators(rates, sizes)
        return self

    def apply(self, p):
        """The corrected probability for each model probability p."""
        return np.interp(p, self.predicted, self.actual)

    def save(self, path="calibration.npz"):
        np.savez(path, predicted=self.predicted, actual=self.actual)

    @staticmethod
    def load(path="calibration.npz"):
        data = np.load(path)
        calibration = Calibration()
        calibration.predicted = data["predicted"]
        calibration.actual = data["actual"]
        return calibration


def fit_and_save(X_train, y_train, features, path="calibration.npz"):
    """Learn the table from out-of-fold predictions on the training rows and save it.

    features must be the ones the saved claim model was trained with, using the
    default settings.
    """
    p = out_of_fold_predictions(X_train, y_train, features)
    calibration = Calibration().fit(p, y_train)
    calibration.save(path)
    return calibration


if __name__ == "__main__":
    # Rebuild calibration.npz for the saved model without rerunning the feature
    # search. saved_model.npz keeps the features but not the training settings,
    # so check that the default settings rebuild it exactly before going on.
    from pathlib import Path

    import pandas as pd

    folder = Path(__file__).resolve().parent
    customer_data = pd.read_csv(folder / "data" / "customer_data.csv")
    y = pd.read_csv(folder / "data" / "customer_claims.csv")["claim"].to_numpy()
    test_mask = customer_data.index % 5 == 0
    X_train = customer_data[~test_mask].reset_index(drop=True)
    y_train = y[~test_mask]

    saved = ClaimModel.load(folder / "saved_model.npz")
    rebuilt = ClaimModel().fit(X_train, y_train, saved.features)
    if not (np.allclose(rebuilt.w, saved.w) and np.isclose(rebuilt.b, saved.b)):
        raise SystemExit("saved_model.npz wasn't trained with the default settings")

    calibration = fit_and_save(
        X_train, y_train, saved.features, folder / "calibration.npz"
    )
    print("group  model p  corrected p")
    for g, (before, after) in enumerate(
        zip(calibration.predicted, calibration.actual), start=1
    ):
        print(f"{g:>5}  {before:>7.3f}  {after:>11.3f}")
