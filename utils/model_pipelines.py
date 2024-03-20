from utils.train import training_mlr, training_rf
from utils.predict import predict_evaluate_mlr, predict_evaluate_rf

def run_mlr_model(df):
    for type in ['HOUSE', 'APARTMENT']:
        data = df[df['PropertyType'] == type].copy()
        X_train, X_test, y_train, y_test, X_train_with_const = training_mlr(data, type)
        predict_evaluate_mlr(X_train, X_test, y_train, y_test, X_train_with_const, type)

def run_rf_model(df):
    type = 'Houses & apartments combined'
    data = df.copy()
    X_train, X_test, y_train, y_test, rf = training_rf(data, type)
    predict_evaluate_rf(X_train, X_test, y_train, y_test, type)