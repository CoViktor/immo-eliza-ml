import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.preprocessing import preprocess_data
from utils.data_import import import_data

import_data()

# Preprocessing
def training(df):
    print('Training data...')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("After preprocessing step xtrain:", X_train.shape)
    print("After preprocessing step xtest:", X_test.shape)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, regressor

# big_drops = ['Condition', 'EnergyConsumptionPerSqm'] -> missings are still in there for simple regression, drop for multivariate if they're included