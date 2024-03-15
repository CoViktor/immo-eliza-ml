import statsmodels.api as sm

from utils.preprocessing import preprocess_data
from utils.data_import import import_data

import_data()

# Preprocessing
def training(df):
    print('Training data...')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    X_train_with_const = sm.add_constant(X_train)
    trained_model = sm.OLS(y_train, X_train_with_const).fit()

    return X_train, X_test, y_train, y_test, trained_model, X_train_with_const