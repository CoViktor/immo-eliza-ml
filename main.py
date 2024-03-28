import pandas as pd
from utils.model_pipelines import run_mlr_model, run_rf_model

df = pd.read_csv('./data/raw_data.csv')

# Comment out the model you don't want to run
run_mlr_model(df)
# run_rf_model(df)
