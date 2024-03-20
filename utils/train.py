import statsmodels.api as sm
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

from utils.preprocessing import preprocess_data


# Preprocessing
def training_mlr(df, type):
    """Trains a Multiple Linear Regression (MLR) model on the provided
    DataFrame for a specified property type. It preprocesses the data, trains
    the model, and saves the model to a file.

    Parameters:
    - df (pandas.DataFrame): The dataset containing features and target 
      variable.
    - type (str): The type of property (e.g., 'HOUSE', 'APARTMENT') used to 
      differentiate models.

    Returns:
    - X_train (pandas.DataFrame): The training feature set.
    - X_test (pandas.DataFrame): The test feature set.
    - y_train (pandas.Series): The training target variable.
    - y_test (pandas.Series): The test target variable.
    - X_train_with_const (pandas.DataFrame): Training features with a constant
      term added for intercept.
    """
    X_train, X_test, y_train, y_test = preprocess_data(df, type)
    
    X_train_with_const = sm.add_constant(X_train)
    trained_model = sm.OLS(y_train, X_train_with_const).fit()

    # saving the trained model
    model_filename = f'models/{type}_trained_mlr_model.joblib'
    dump(trained_model, model_filename)

    return X_train, X_test, y_train, y_test, X_train_with_const


def training_rf(df, type):
    """Trains a Random Forest Regressor model on the provided DataFrame for a
    specified property type. It preprocesses the data, fits the model, and
    saves the model to a file.

    Parameters:
    - df (pandas.DataFrame): The dataset containing features and target 
      variable.
    - type (str): The type of property (e.g., 'Houses & apartments combined')
      used to differentiate models.

    Returns:
    - X_train (pandas.DataFrame): The training feature set.
    - X_test (pandas.DataFrame): The test feature set.
    - y_train (pandas.Series): The training target variable.
    - y_test (pandas.Series): The test target variable.
    - rf (RandomForestRegressor): The trained Random Forest model.
    """
    X_train, X_test, y_train, y_test = preprocess_data(df, type)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # saving the trained model
    model_filename = f'models/{type}_trained_rf_model.joblib'
    dump(rf, model_filename)

    return X_train, X_test, y_train, y_test, rf
