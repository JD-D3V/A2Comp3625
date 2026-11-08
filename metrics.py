import numpy as np
import pandas as pd

def confusion_matrix(y, p, cutoff=0.5):
    """Return (tp, tn, fn, fp)"""
    prediction = p >= cutoff
    tp = np.sum(prediction & (y == 1))
    tn = np.sum(~prediction & (y == 0))
    fn = np.sum(~prediction & (y == 1))
    fp = np.sum(prediction & (y == 0))

    return (tp, tn, fn, fp)

