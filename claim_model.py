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

        if epoch % 50 ==0:
            print(f"epoch {epoch}: training loss {log_loss(y, sigmoid(A @ w + b)):.4f}")

    return w, b