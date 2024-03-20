# Model card

## Project context

This project is an atempt to predict the prices of houses and apartments, depending on the properties of the building. It is part of a larger learning project pipeline, and the models created will be used to generate a UI at a later time.

- Model types: Multiple Linear Regression and Random Forest

### Intended Use
- Primary Use: Predicting residential property prices in Belgium, based on various features.
- Potential Users: Real estate agencies, property investors, data scientists analyzing the real estate market.
## Data
### Input dataset 
`raw_data.csv` is a dataset containing information of 80K+ properties that were put for sale on immoweb.be, the larges Belgian immo website.

### Preprocessing pipeline
In `preprocessing.py`, the data is cleaned and preprocessed. 
- DataTypes are altered to be readable and usable in further processing
- Postal code is created by looking at the first two numbers of each postal code.
- Missings are dropped, or imputed with the median for ConstructionYear and  (after the split).
- Outliers are dropped
- Data is split with the train_test_split-method
- Categoricals are one-hot encoded
- Missings are imputed with median
- Variables are normalized.

### Target variable
Price, in euros.

### Features
The features selected for the final models are:
'PostalZone', 'PropertyType', 'PropertySubType', 'Price',
            'ConstructionYear', 'BedroomCount',
            'LivingArea', 'Furnished', 'Fireplace', 'Terrace',
            'Garden', 'GardenArea', 'Facades', 'SwimmingPool', 'Condition',
            'EnergyConsumptionPerSqm'

## Multiple Linear Regression Model

### Model details
When running this Multiple Linear Regression model, the data is split in Houses and Apartmens from the beginning, as combining them has a negative effect on the metrics. Some features are clearly devided differently for these main subcategories in the data.

### Performance and evaluation
The performance of the models was tested by looking at the R² and the RMSE of both the training- and test-sets, to check for overfitting on the training-set.
Additionally, the model was run on 6 different random states, to check for robustness and overfitting on a certain test-set.

Metrics:
| Category                                            | Mean R²   | Std R²   | Mean RMSE | Std RMSE |
|-----------------------------------------------------|-----------|----------|-----------|----------|
| House MLR Training Set                              | 0.725     | 0.0019   | 87.32     | 0.37     |
| House MLR Test Set                                  | 0.704     | 0.0076   | 91.25     | 1.20     |
| Apartment MLR Training Set                          | 0.645     | 0.0036   | 67.20     | 0.58     |
| Apartment MLR Test Set                              | 0.633     | 0.0120   | 68.68     | 2.40     |


- House prices are better explained by this model than apartment prices
- The RMSE is about 20k euros higher for house prices, but this might be due to the fact that the mean house price is higher than the mean apartment price.
- In each case, the difference between the training set and the test set is relatively low. And in each case, the standard deviation is relatively low. This points to the model not overfitting for the training set.

Runtime: 
- Entire pipeline for houses and apartments (Including prepping, training, and evaluation)
- ~10.5 seconds

### Limitations
A significant part (~30-40%) of the data is not explained by this model. It is possible that this can be slightly raised by further finetuning the selection of features and the preprocessing. But this might cause overfitting of the model, and should be done with care.

Additionally, the model performs worse for the price prediciton of apartments. Perhaps an adjustment of the pipeline with feature selection for this property type specifically could help. However, upon testing no right features were found yet.

## Random Forest Model

### Model details
Contrary to the MLR-model, the Random Forest model performance did not seem to be influenced by splitting the data in houses and apartments. 

### Performance and evaluation
The Random Forest model explains a much larger proportion of the variety in property prices than the Multiple Linear Regression model (R²= .725, for houses), when evaluating the training-set performance.
When comparing the training sets, however, the difference is a lot smaller. This implies an overfitting on the training data.
On top of that, this model includes both property types, providing a prediction based on a more complete view of the data.

| Category                                            | Mean R2   | Std R2   | Mean RMSE | Std RMSE |
|-----------------------------------------------------|-----------|----------|-----------|----------|
| Houses & Apartments Combined Training Set           | 0.961     | 0.0006   | 32.85     | 0.28     |
| Houses & Apartments Combined Test Set               | 0.713     | 0.0064   | 89.12     | 1.78     |

Runtime: 

### Limitations
The difference in R² between the training-set and the testing-set implies that there is still some work if this model is to be used as an accurate predictor for unknown data.

## Usage
The libraries used are:
 `pandas, datetime, sklearn (preprocessing, model_selection, metrics, ensemble), plotly (figure_factory), matplotlib (pyplot), numpy, statsmodels (api)`, and `joblib`.

- `preprocessing.py` contains the function `preprocess_data`, that calls the following functions: 
    - `cleaning_data`, `drop_missings`, `drop_outliers`, `feature_engineer`, `train_test_split`, `one_hot`, `impute_missings`, and `scale_data`. 
    - Additionally, the `explore_data` and `covariates` functions are available for exploration of univariates and bivariates.
- `train.py` contains the functions `training_mlr`, and `training_rf`. 
    - Each call `preprocess_data`, train the model on the training-set, and save the trained model as a `joblib`-file
- `predict.py` contains the functions ``, and ``. 
    - These load the relevant model and print the R² and MRSE in an easy to interpret way.
    - Additionaly, a plot of the models can be printed by uncommenting `plt.show()` in the corresponding functions.
- `model_pipelines.py` contains the pipelines to run both models, stored in functions.
- `main.py` stores the data in a DataFrame and runs the models. comment out the model you do not want to run.

### Model pipelines
- run_mlr_model
    - training_mlr
        - preprocess_data
            - preprocessing steps
        - train model
        - store model
    - predict_evaluate_mlr
        - load model
        - print metrics


- run_rf_model
    - training_rf
        - preprocess_data
            - preprocessing steps
        - train model
        - store model
    - predict_evaluate_rf
        - load model
        - print metrics

## Maintainers

Viktor Cosaert on [LinkedIn](https://www.linkedin.com/in/viktor-cosaert/).