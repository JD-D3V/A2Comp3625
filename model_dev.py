import pandas as pd
import numpy as np
from claim_model import ClaimModel
from feature_search import search

# load historical customer data
customer_data = pd.read_csv('./data/customer_data.csv', index_col=None)
customer_claims = pd.read_csv('./data/customer_claims.csv', index_col=None)


# write your code to develop your agent program here.
# you can, of course, add additional .py files as needed.

#loading the data into X and y variables, where X is pandas dataframe of customer data 
# and y is a numpy array of customer claims
X = customer_data
y = customer_claims["claim"].to_numpy()


# splitting the data into training and testing (80/20 split) using a mask to select every 5th row for testing
test_mask = customer_data.index % 5 == 0

X_test = X[test_mask].reset_index(drop=True)
y_test = y[test_mask]

X_train = X[~test_mask].reset_index(drop=True)
y_train = y[~test_mask]

#printing for debugging purposes
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

# Darren's logistic regression (claim_model.py), which can save itself for set_premium
test_model = ClaimModel()
# training the model using the training data and the features in the training data
test_model.fit(X_train, y_train, list(X_train.columns))
# then using the trained model to predict the probability of a claim for each customer in the test data
predics = test_model.predict_proba(X_test)
print("Predictions shape:", predics.shape)
print("5 sample predictions:", predics[:5])

best_features, best_params = search(X_train, y_train, make_model=ClaimModel)
final_model = ClaimModel()
final_model.fit(X_train, y_train, best_features, **best_params)
# saving the model to a file so that it can be used later for making predictions on new customer data
final_model.save("saved_model.npz")
print("Saved saved_model.npz with features:", best_features, "and params:", best_params)

# Joseph's calibration step (calibration.py): corrects the model's probabilities
from calibration import fit_and_save
fit_and_save(X_train, y_train, best_features, best_params, "calibration.npz")
print("Saved calibration.npz")
