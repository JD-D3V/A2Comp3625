import pandas as pd
import numpy as np

from metrics import log_loss, report

def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))

def train(A, y, learning_rate=0.5, epochs=200, l2_strength=0.0):
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

        if epoch % 50 == 0:
            print(f"epoch {epoch}: training loss {log_loss(y, sigmoid(A @ w + b)):.4f}")

    return w, b

class ClaimModel:
    def __init__(self):
        self.features = []
        self.mu = None
        self.sd = None
        self.w = None
        self.b = 0.0

    def fit(self, X, y, features, learning_rate=0.5, epochs=200, l2_strength=0.0):
        self.features = list(features)
        A = X[self.features].to_numpy(dtype=float)

        # standardize using the training data
        self.mu = A.mean(axis=0)
        self.sd = A.std(axis=0)
        self.sd[self.sd == 0] = 1
        A = (A - self.mu) / self.sd

        self.w, self.b = train(A, y, learning_rate, epochs, l2_strength)
        return self

    def predict_proba(self, X):
        A = X[self.features].to_numpy()
        A = (A - self.mu) / self.sd
        return sigmoid(A @ self.w + self.b)
