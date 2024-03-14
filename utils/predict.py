import matplotlib.pyplot as plt


def predict_evaluate(X_train, X_test, y_train, y_test, regressor, type):
    print(f'Predicting and evaluating {type}...')
    y_pred = regressor.predict(X_test)
    # Train dataset score
    r_squared = regressor.score(X_train, y_train)
    print(f"Train Set R^2 Score: {r_squared}")
    # Test dataset score
    r2_score_test = regressor.score(X_test, y_test)
    print(f"Test Set R^2 Score: {r2_score_test}")   
    
    # Plotting actual vs predicted values
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=2, color='red')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted - {type}')
    plt.show()

    # Residuals
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', linestyles='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(f'Residuals of Predictions - {type}')
    plt.show()