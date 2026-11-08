import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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

# temporarily using a stand-in model for development purposes
class StandInModel:
    def __init__(self):
        # using standscaler because the features are on different scales, and logistic regression is sensitive to feature scaling
        self.scaler = StandardScaler() 
        self.model = LogisticRegression()
        self.features = []

    def fit(self, X, y, features, **params):
        self.features = features
        # only use the features specified in the features list for training
        X_subset = X[self.features]
        # putting the features on the same scale for training
        X_scaled = self.scaler.fit_transform(X_subset)
        # figuring out the best parameters for the model using the training data now that they are on the same scale
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        # only use the features specified in the features list for prediction
        X_subset = X[self.features]
        # putting the features on the same scale for prediction
        X_scaled = self.scaler.transform(X_subset)
        # getting the probability of all the customers that filed a claim (1)
        predictions = self.model.predict_proba(X_scaled)[:, 1]  
        return predictions


test_model = StandInModel()
# training the model using the training data and the features in the training data
test_model.fit(X_train, y_train, list(X_train.columns))
# then using the trained model to predict the probability of a claim for each customer in the test data
predics = test_model.predict_proba(X_test)
print("Predictions shape:", predics.shape)
print("5 sample predictions:", predics[:5])
