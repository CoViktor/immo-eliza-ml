from utils.train import training_mlr, training_rf
from utils.predict import predict_evaluate_mlr, predict_evaluate_rf

def run_mlr_model(df):
    """Runs Multiple Linear Regression (MLR) models for different property 
    types within the given DataFrame.

    This function iterates over property types ('HOUSE', 'APARTMENT'), filters
    the DataFrame for each type, and then proceeds with training and evaluating
    the MLR model specific to that property type.

    Parameters:
    - df (pandas.DataFrame): The dataset containing real estate listings with
      a 'PropertyType' column.

    Returns:
    None. The function internally calls other functions to train and evaluate
    models, but does not return any values itself.
    """
    for type in ['HOUSE', 'APARTMENT']:
        data = df[df['PropertyType'] == type].copy()
        X_train, X_test, y_train, y_test, X_train_with_const = training_mlr(data, type)
        predict_evaluate_mlr(X_train, X_test, y_train, y_test, X_train_with_const, type)

def run_rf_model(df):
    """Runs a Random Forest (RF) model for combined property types in the given
    DataFrame.

    This function does not filter the DataFrame by property type but uses the
    entire dataset to train and evaluate a Random Forest model intended for
    both houses and apartments combined.

    Parameters:
    - df (pandas.DataFrame): The dataset containing real estate listings, where
      both houses and apartments are combined.

    Returns:
    None. The function internally calls other functions to train and evaluate
    the RF model, but does not return any values itself.
    """
    type = 'Houses & apartments combined'
    data = df.copy()
    X_train, X_test, y_train, y_test, rf = training_rf(data, type)
    predict_evaluate_rf(X_train, X_test, y_train, y_test, type)