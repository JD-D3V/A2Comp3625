import os

import pandas as pd
import numpy as np

from metrics import log_loss, report, calibration_table

def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))

def expand(A):
    """Add a squared column for each feature and a product column for each pair"""
    squares = A ** 2
    i, j = np.triu_indices(A.shape[1], k=1)
    pairs = A[:, i] * A[:, j]
    return np.hstack([A, squares, pairs])

def train(A, y, learning_rate=0.5, epochs=200, l2_strength=0.0, verbose=False):
    """Gradient descent for logistic regression
    Returns the learned weights w (length n) and bias b."""
    N,n = A.shape
    w = np.zeros(n)
    b = 0.0

    for epoch in range(epochs):
        p = sigmoid(A @ w + b) 
        error = p - y 
        grad_w = ((A.T @ error)/ N) + l2_strength * w
        w = w - learning_rate * grad_w
        grad_b = np.average(error)
        b = b - learning_rate * grad_b

        if verbose and epoch % 50 == 0:
            print(f"epoch {epoch}: training loss {log_loss(y, sigmoid(A @ w + b)):.4f}")

    return w, b

class ClaimModel:
    def __init__(self):
        self.features = []
        self.mu = None
        self.sd = None
        self.w = None
        self.b = 0.0
        self.interactions = True
        self.mu2 = None
        self.sd2 = None

    def fit(self, X, y, features, learning_rate=0.5, epochs=200, l2_strength=0.0,
            interactions=True):
        self.features = list(features)
        self.interactions = interactions
        A = X[self.features].to_numpy(dtype=float)

        # standardize using the training data
        self.mu = A.mean(axis=0)
        self.sd = A.std(axis=0)
        self.sd[self.sd == 0] = 1
        A = (A - self.mu) / self.sd

        # squares and pairs get standardized too
        if self.interactions:
            A = expand(A)
            self.mu2 = A.mean(axis=0)
            self.sd2 = A.std(axis=0)
            self.sd2[self.sd2 == 0] = 1
            A = (A - self.mu2) / self.sd2

        self.w, self.b = train(A, y, learning_rate, epochs, l2_strength)
        return self

    def predict_proba(self, X):
        A = X[self.features].to_numpy()
        A = (A - self.mu) / self.sd
        if self.interactions:
            A = (expand(A) - self.mu2) / self.sd2
        return sigmoid(A @ self.w + self.b)

    def save(self, path="saved_model.npz"):
        np.savez(path, features=np.array(self.features), mu=self.mu, sd=self.sd,
                 w=self.w, b=self.b, interactions=self.interactions,
                 mu2=self.mu2 if self.interactions else np.zeros(0),
                 sd2=self.sd2 if self.interactions else np.zeros(0))

    @staticmethod
    def load(path="saved_model.npz"):
        data = np.load(path)
        model = ClaimModel()
        model.features = [str(f) for f in data["features"]]
        model.mu = data["mu"]
        model.sd = data["sd"]
        model.w = data["w"]
        model.b = float(data["b"])
        model.interactions = bool(data["interactions"])
        if model.interactions:
            model.mu2 = data["mu2"]
            model.sd2 = data["sd2"]
        return model


if __name__ == "__main__":
    customer_data = pd.read_csv("./data/customer_data.csv")
    y = pd.read_csv("./data/customer_claims.csv")["claim"].to_numpy()

    test_mask = customer_data.index % 5 == 0
    X_train, y_train = customer_data[~test_mask], y[~test_mask]
    X_test, y_test = customer_data[test_mask], y[test_mask]

    features = list(customer_data.columns)

    print("Plain logistic regression")
    plain = ClaimModel().fit(X_train, y_train, features, interactions=False)
    report(y_test, plain.predict_proba(X_test))

    print("\nWith squares and pairs")
    model = ClaimModel().fit(X_train, y_train, features)
    p_test = model.predict_proba(X_test)
    report(y_test, p_test)
    print(f"Train log-loss: {log_loss(y_train, model.predict_proba(X_train)):.4f}")
    calibration_table(y_test, p_test)
    print()

    shuffled = X_test[X_test.columns[::-1]]
    print("column order safe:", np.allclose(p_test, model.predict_proba(shuffled)))

    model.save("round_trip_check.npz")
    loaded = ClaimModel.load("round_trip_check.npz")
    os.remove("round_trip_check.npz")
    print("save/load round trip:", np.allclose(p_test, loaded.predict_proba(X_test)))

    print("single customer shape:", model.predict_proba(X_test.iloc[[0]]).shape)
