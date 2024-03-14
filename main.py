import pandas as pd

from utils.train import training
from utils.predict import predict_evaluate


# Loading & training the data
df = pd.read_csv('./data/raw_data.csv')

for type in ['HOUSE', 'APARTMENT']:
    print(f'\n\n\n---{type}---')
    data = df[df['PropertyType'] == type].copy()
    X_train, X_test, y_train, y_test, regressor = training(data)
    predict_evaluate(X_train, X_test, y_train, y_test, regressor, type)
    print(f'---{type} OVER---\n\n\n')

# # After split for df in [train_data, test_data]:
# for type in ['HOUSE', 'APARTMENT']:
#     print(f'\n\n\n---{type}---')
#     data = df[df['PropertyType'] == type].copy()
#     X_train, X_test, y_train, y_test = preprocess_data(data)  # X_train, X_test, y_train, y_test = preprocess_data(data)
#     explore_data(X_train)  # explore_data(X_train) -> Loop is handy for running different models, output underneath eachother, 
#                                 # so skip exploration & hash print statements in prep for clear output; for each model do print bivariates (cormatrix), to check covariation
#     print(f'---{type} OVER---\n\n\n')