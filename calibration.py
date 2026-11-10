"""Correct the claim model's probabilities so they match real claim rates.

The model is right on average but off by risk band: it says 0.45 where about half
of customers really claim, and 0.71 where about 0.62 do. Selection picks mostly
from the middle band, where the model is too low, so we expected more profit than
we got. This file learns a map from the model's probability to the claim rate
actually seen, and set_premium applies it.

Method: a binned table, chosen because it is simple to explain and it fit best.
1. Sort customers by the model's probability and cut them into 20 equal groups.
2. For each group note the average predicted p and the share who claimed.
3. If a riskier group shows a lower claim rate than the one before, it takes that
   earlier rate instead, so a higher prediction never means a lower one after.
4. Between group centres, draw straight lines; past the end groups, stay flat.

The number of groups is picked by grid search: each size in GROUP_CHOICES is scored
by k-fold cross-validation on the training rows, and the lowest log-loss wins (20).

The table is learned only from training rows, using out-of-fold predictions: each
row's probability comes from a model that did not see that row, the way new
customers will look at grading time.
"""

import numpy as np

from claim_model import ClaimModel
from metrics import log_loss

GROUP_CHOICES = [5, 10, 20, 50, 100]  # table sizes the grid search tries
FOLDS = 5


def out_of_fold_predictions(X, y, features):
    """Claim probability for each row from a ClaimModel trained without that row."""
    fold_of_row = np.arange(len(X)) % FOLDS  # row 0 is in fold 0, row 1 in fold 1...
    predictions = np.zeros(len(X))
    for fold in range(FOLDS):
        is_held_out = fold_of_row == fold
        model = ClaimModel()
        model.fit(X[~is_held_out], y[~is_held_out], features)
        predictions[is_held_out] = model.predict_proba(X[is_held_out])
    return predictions


class Calibration:
    def __init__(self, predicted=None, actual=None):
        self.predicted = predicted  # average model p in each group (the table's x)
        self.actual = actual  # smoothed claim rate in each group (the table's y)

    def fit(self, p, y, number_of_groups):
        """Build the table from model probabilities p and real claims y (0 or 1)."""
        lowest_to_highest = np.argsort(p)
        groups = np.array_split(lowest_to_highest, number_of_groups)

        average_guess = []
        claim_rate = []
        for group in groups:
            average_guess.append(p[group].mean())
            # +1 claim and +1 non-claim keeps the rate strictly inside (0, 1),
            # even for a group where nobody (or everybody) claimed.
            claims = y[group].sum()
            claim_rate.append((claims + 1) / (len(group) + 2))

        self.predicted = np.array(average_guess)
        # A riskier group never gets a lower rate than the group before it.
        self.actual = np.maximum.accumulate(claim_rate)
        return self

    def apply(self, p):
        """The corrected probability for each model probability p."""
        return np.interp(p, self.predicted, self.actual)

    def save(self, path="calibration.npz"):
        np.savez(path, predicted=self.predicted, actual=self.actual)

    @staticmethod
    def load(path="calibration.npz"):
        data = np.load(path)
        return Calibration(data["predicted"], data["actual"])


def validation_loss(p, y, number_of_groups):
    """K-fold cross-validation: log-loss of the table on rows it was not built from."""
    fold_of_row = np.arange(len(p)) % FOLDS
    losses = []
    for fold in range(FOLDS):
        is_held_out = fold_of_row == fold
        table = Calibration()
        table.fit(p[~is_held_out], y[~is_held_out], number_of_groups)
        corrected = table.apply(p[is_held_out])
        losses.append(log_loss(y[is_held_out], corrected))
    return np.mean(losses)


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
    rebuilt = ClaimModel()
    rebuilt.fit(X_train, y_train, saved.features)
    same_weights = np.allclose(rebuilt.w, saved.w) and np.isclose(rebuilt.b, saved.b)
    if not same_weights:
        raise SystemExit("saved_model.npz wasn't trained with the default settings")

    p = out_of_fold_predictions(X_train, y_train, saved.features)

    # Grid search: score every table size on validation folds, keep the best one.
    best_groups = None
    best_loss = float("inf")
    for number_of_groups in GROUP_CHOICES:
        loss = validation_loss(p, y_train, number_of_groups)
        print(f"{number_of_groups:>3} groups: validation log-loss {loss:.5f}")
        if loss < best_loss:
            best_groups = number_of_groups
            best_loss = loss

    calibration = Calibration()
    calibration.fit(p, y_train, best_groups)
    calibration.save(folder / "calibration.npz")

    print("group  model p  corrected p")
    for group in range(best_groups):
        before = calibration.predicted[group]
        after = calibration.actual[group]
        print(f"{group + 1:>5}  {before:>7.3f}  {after:>11.3f}")
