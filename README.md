# COMP 3625 Assignment 2: insurance customer selection

Investors give us $100 million to start an insurance company. Given a table of
potential customers, the program estimates how likely each one is to claim, sets a
premium for each, and chooses which customers to insure so that total coverage stays
within the $100 million.

The grader calls `set_premium(customer_data)` in `model_evaluation.py`.

## How it works

1. `claim_model.py` estimates each customer's claim probability. It is logistic
   regression written from scratch with numpy and trained by gradient descent. Each
   feature is also squared and multiplied with every other feature.
2. `calibration.py` corrects those probabilities. The model is right on average but
   too low in the middle risk band, so a 20-row table maps the model's probability
   to the claim rate we actually saw for customers with that probability.
3. `pricing.py` sets the premium. Each premium sits between our break-even
   (probability times coverage) and the most the customer would pay. The utility
   check `U(-premium) > EU(uninsured)` is done for every customer.
4. `selection.py` chooses the customers. It is a 0/1 knapsack, so it picks the
   customers with the most total expected profit whose coverage fits the budget.

## Assignment requirements

| Requirement | Where |
|---|---|
| Supervised learning, from scratch | `claim_model.py` (numpy only) |
| Search: feature selection | `feature_search.py`, forward selection scored by 5-fold cross-validation (13 of 21 columns kept) |
| Search: model settings | `calibration.py`, grid search over table sizes; `feature_search.py`, three training settings |
| Search: best set of customers | `selection.py`, knapsack table, from scratch |
| Explicit utility calculation | `pricing.py`, and the check in `set_premium` |

scikit-learn is only used in `check_vs_sklearn.py`, as a check. The program never
calls it.

## Files

| File | What it does |
|---|---|
| `model_evaluation.py` | `set_premium`: loads the saved model and table, prices, runs the utility check, selects |
| `model_dev.py` | Splits the data 80/20, runs the feature search, trains and saves the model |
| `claim_model.py` | The logistic regression: fit, predict, save and load |
| `feature_search.py` | Cross-validation, forward feature selection and the training-settings search |
| `calibration.py` | Learns and applies the probability correction table |
| `pricing.py` | The utility curve, the customer's maximum premium, the utility check, `set_prices` |
| `selection.py` | Chooses customers within the budget |
| `selection_exact_search.py` | The earlier selection, an exact search with pruning. Kept for reference, not used by `set_premium` |
| `metrics.py` | Accuracy, sensitivity, specificity, log-loss and a calibration table |
| `simulate.py` | Backtest and mock competition used to think about the price level |
| `check_vs_sklearn.py` | Compares our logistic regression with scikit-learn's |
| `saved_model.npz` | The trained model that `set_premium` loads |
| `calibration.npz` | The correction table that `set_premium` loads |
| `written_responses.txt` | Answers to the assignment questions |

## How to run

Run everything from the repository folder.

```bash
pip install -r requirements.txt
```

Use the saved model on a table of customers:

```python
import pandas as pd
from model_evaluation import set_premium

offers = set_premium(pd.read_csv("data/customer_data.csv"))
```

`offers` has one row per customer with `p_claim`, `desired_customer` and `premium`.

Rebuild the saved files (not needed for grading, and the feature search is slow):

```bash
python model_dev.py
```

```bash
python calibration.py
```

`model_dev.py` writes `saved_model.npz`. `calibration.py` then writes
`calibration.npz` for that model.

## Results on the 20,000 held-out customers

| Measure | Value |
|---|---|
| Log-loss, everyone given the average claim rate | 0.638 |
| Log-loss, our model | 0.5657 |
| Log-loss, our model with the correction | 0.5589 |
| Customers chosen | 735 |
| Coverage used | $99,999,284 of $100,000,000 |
| Expected profit | about $13.2 million |
| Profit once the real claims are counted | about $14.1 million |

This is with no competing teams. `set_premium` takes about 1 second on 200,000
customers.
