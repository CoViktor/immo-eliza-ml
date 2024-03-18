import statsmodels.api as sm
from joblib import dump

from utils.preprocessing import preprocess_data
from utils.data_import import import_data

import_data()

# Preprocessing
def training(df, type):
    # print('Training data...')
    X_train, X_test, y_train, y_test = preprocess_data(df, type)
    
    X_train_with_const = sm.add_constant(X_train)
    trained_model = sm.OLS(y_train, X_train_with_const).fit()

    # saving the trained model
    model_filename = f'models/{type}_trained_reg_model.joblib'
    dump(trained_model, model_filename)
    print(f"Model saved as {model_filename}")

    return X_train, X_test, y_train, y_test, X_train_with_const