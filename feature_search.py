import numpy as np
import pandas as pd

def compute_log_loss(y_true, y_prob):
    # prvents log(0) error by clipping the probabilities slightly
    eps = 1e-15
    # y_prob is 1-D array of predicted probabilities from predict_proba (like 0.32)
    # if value is less than eps, set it to eps, if greater than 1-eps, set it to 1-eps
    y_prob = np.clip(y_prob, eps, 1 - eps)
    log_loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return log_loss

def cross_validate(X, y, features, make_model, k=5, **params):
    val_losses = []
    fold_size = len(X) // k
    for i in range(k):
        # finding the chunk boundaries for the validation/test set
        start = i * fold_size
        if i == k - 1:
            end = len(X)
        else:
            end = start + fold_size

        # slicing the data into training and validation sets
        # X_val needs .iloc because we are slicing by index for the rows as it is a 2D array
        X_val = X.iloc[start:end]
        y_val = y[start:end]

        X_train = pd.concat([X.iloc[:start], X.iloc[end:]])
        y_train = np.concatenate([y[:start], y[end:]])

        # training and scoring the model
        model = make_model()
        model.fit(X_train, y_train, features, **params)

        # predicting the probabilities on the validation set
        predics = model.predict_proba(X_val)

        # calculating using the compute_log_loss function
        loss = compute_log_loss(y_val, predics)
        val_losses.append(loss)

    # average the validation losses across all folds and return it
    return np.mean(val_losses)

def search(X, y, make_model):
    selected_features = []
    remaining_features = list(X.columns)
    best_score = float('inf') # since we want log loss to be as low as possible, we will start with the best score as infinity

    while True:
        best_candidate = None
        round_best_score = float('inf')

        for candidate in remaining_features:
            test_features = selected_features + [candidate]
            score = cross_validate(X, y, test_features, make_model)
            # getting the best score for this round of feature selection
            if score < round_best_score:
                round_best_score = score
                best_candidate = candidate
        # getting best score globaly
        if round_best_score < best_score:
            best_score = round_best_score
            selected_features.append(best_candidate)
            remaining_features.remove(best_candidate)
            print(f"Added {best_candidate}, new cross validation loss: {best_score}")
        else:
            print("No improvement, stopping the search")
            break

    return selected_features, {}
