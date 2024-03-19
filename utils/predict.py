import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np 
import statsmodels.api as sm
from joblib import load


def predict_evaluate_mlr(X_train, X_test, y_train, y_test, X_train_with_const, type):
    # print(f'Predicting and evaluating {type}...')
    # print(f'\n\n{type} TRAINING MODEL SUMMARY\n')

    model_filename = f'models/{type}_trained_mlr_model.joblib'
    trained_model = load(model_filename)

    # print(trained_model.summary())  # -> Print nice oversight of all coefs

    # Predictions
    y_train_pred = trained_model.predict(X_train_with_const)
    y_test_pred = trained_model.predict(sm.add_constant(X_test))

    # Metrics for Training Set
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    # mae_train = mean_absolute_error(y_train, y_train_pred)

    # Metrics for Test Set
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    # mae_test = mean_absolute_error(y_test, y_test_pred)

    # Printing Metrics
    print(f"\n{type} MLR Training Set Metrics:")
    print(f"R2: {r2_train:.3f}, average error of {(rmse_train)/1000:.2f}K euros (RMSE)")

    print(f"\n{type} MLR Test Set Metrics:")
    print(f"R2: {r2_test:.3f}, average error of {(rmse_test)/1000:.2f}K euros (RMSE)")

    # Creating a figure and a grid of subplots
    plt.figure(figsize=(20, 6))

    # Training Data Plot
    plt.subplot(1, 2, 1)  # (rows, columns, panel number)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)  # Diagonal line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{type} - Training Data: Actual vs. Predicted')

    # Test Data Plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)  # Diagonal line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{type} - Test Data: Actual vs. Predicted')

    # plt.show()


def predict_evaluate_rf(X_train, X_test, y_train, y_test, type):
    model_filename = f'models/{type}_trained_rf_model.joblib'
    rf = load(model_filename)

    # Make predictions
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    # Calculate R2 and RMSE scores
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\n{type}Random Forest Training Set Metrics:")
    print(f"R2: {r2_train:.3f}, average error of {(rmse_train)/1000:.2f}K euros (RMSE)")
    print(f"\n{type}Random Forest Test Set Metrics:")
    print(f"R2: {r2_test:.3f}, average error of {(rmse_test)/1000:.2f}K euros (RMSE)")