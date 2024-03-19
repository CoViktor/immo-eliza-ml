import statsmodels.api as sm
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

from utils.preprocessing import preprocess_data


# Preprocessing
def training_mlr(df, type):
    # print('Training data...')
    X_train, X_test, y_train, y_test = preprocess_data(df, type)
    
    X_train_with_const = sm.add_constant(X_train)
    trained_model = sm.OLS(y_train, X_train_with_const).fit()

    # saving the trained model
    model_filename = f'models/{type}_trained_mlr_model.joblib'
    dump(trained_model, model_filename)
    print(f"Model saved as {model_filename}")

    return X_train, X_test, y_train, y_test, X_train_with_const

def training_rf(df, type):
    X_train, X_test, y_train, y_test = preprocess_data(df, type)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # saving the trained model
    model_filename = f'models/{type}_trained_rf_model.joblib'
    dump(rf, model_filename)
    print(f"Model saved as {model_filename}")

    return X_train, X_test, y_train, y_test, rf
