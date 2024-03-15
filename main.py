import pandas as pd

from utils.train import training
from utils.predict import predict_evaluate


# Loading & training the data
df = pd.read_csv('./data/raw_data.csv')

for type in ['HOUSE', 'APARTMENT']:
    print(f'\n\n\n---{type}---')
    data = df[df['PropertyType'] == type].copy()
    X_train, X_test, y_train, y_test, trained_model, X_train_with_const = training(data)
    predict_evaluate(X_train, X_test, y_train, y_test, trained_model, X_train_with_const, type)
    print(f'---{type} OVER---\n\n\n')
