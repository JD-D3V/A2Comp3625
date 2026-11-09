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

def specificity(tn, fp):
    return (tn / (tn + fp))

def log_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return  np.mean(-(y * np.log(p) + (1 - y) * np.log(1-p)))

def report(y, p, cutoff=0.5):
    """Print all of the above in one place"""
    conf_matrix = confusion_matrix(y, p, cutoff)
    true_pos = conf_matrix[0]
    true_neg = conf_matrix[1]
    false_pos = conf_matrix[2]
    false_neg = conf_matrix[3]

    print(f"Cutoff: {cutoff}")
    print(f"TP: {true_pos}\tTN: {true_neg}\tFP: {false_pos}\tFN: {false_neg}")
    print(f"Accuracy: {accuracy(true_pos, true_neg, false_pos, false_neg)}")
    print(f"Sensitivity: {sensitivity(true_pos, false_neg)}")
    print(f"Specificity: {specificity(true_neg, false_pos)}")
    print(f"Log-loss: {log_loss(y, p):.4f}")

def calibration_table(y, p, groups=10):
    """Compare the average predicted p with the actual claim rate in each group"""
    order = np.argsort(p)
    rows = []
    print("group  avg predicted p  actual claim rate")
    for g, idx in enumerate(np.array_split(order, groups), start=1):
        predicted = p[idx].mean()
        actual = y[idx].mean()
        rows.append((predicted, actual))
        print(f"{g:>5}  {predicted:>15.3f}  {actual:>17.3f}")
    return rows
