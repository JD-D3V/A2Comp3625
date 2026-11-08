import numpy as np
import pandas as pd

def confusion_matrix(y, p, cutoff=0.5) -> tuple:
    """Return (tp, tn, fp, fn)"""
    prediction = p >= cutoff
    tp = np.sum(prediction & (y == 1))
    tn = np.sum(~prediction & (y == 0))
    fn = np.sum(~prediction & (y == 1))
    fp = np.sum(prediction & (y == 0))

    return (tp, tn, fp, fn)

def accuracy(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    correct = tp + tn 
    return correct / total 

def sensitivity(tp, fn):
    return (tp / (tp + fn))

