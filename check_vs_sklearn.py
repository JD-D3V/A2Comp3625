"""Compare our logistic regression with sklearn's on the same data.
Only a check, the agent never uses sklearn."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from claim_model import ClaimModel
from metrics import log_loss

customer_data = pd.read_csv("./data/customer_data.csv")
y = pd.read_csv("./data/customer_claims.csv")["claim"].to_numpy()

test_mask = customer_data.index % 5 == 0
X_train, y_train = customer_data[~test_mask], y[~test_mask]
X_test, y_test = customer_data[test_mask], y[test_mask]
features = list(customer_data.columns)

# ours, plain version with more epochs
ours = ClaimModel().fit(X_train, y_train, features, interactions=False, epochs=1000)

# sklearn on the same standardized columns, no penalty
A_train = (X_train[features].to_numpy() - ours.mu) / ours.sd
A_test = (X_test[features].to_numpy() - ours.mu) / ours.sd
sk = LogisticRegression(C=np.inf, max_iter=1000).fit(A_train, y_train)

print(f"largest weight difference: {np.max(np.abs(ours.w - sk.coef_[0])):.6f}")
print(f"bias: ours {ours.b:.4f}  sklearn {sk.intercept_[0]:.4f}")
print(f"test log-loss: ours {log_loss(y_test, ours.predict_proba(X_test)):.4f}"
      f"  sklearn {log_loss(y_test, sk.predict_proba(A_test)[:, 1]):.4f}")
