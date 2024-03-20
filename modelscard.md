# Model card

## Project context

This project is an atempt to predict the prices of houses and apartments, depending on the properties of the building. It is part of a larger learning project pipeline, and the models created will be used to generate a UI at a later time.

## Data
### Input dataset 
`raw_data.csv` is a dataset containing information of 80K+ properties that were put for sale on immoweb.be, the larges Belgian immo website.

### Preprocessing
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
When running this model, the data is split in Houses and Apartmens from the beginning, as combining them has a negative effect on the metrics. Some features are clearly devided differently for these main subcategories in the data.

### Performance
Metrics: for training & test
Tested over 6 different random states, the mean R2 was, with a std dev of, meaning
mean RMSE was x with a std dev of, meaning 

| Category                                            | Mean R2   | Std R2   | Mean RMSE | Std RMSE |
|-----------------------------------------------------|-----------|----------|-----------|----------|
| House MLR Training Set                              | 0.725     | 0.0019   | 87.32     | 0.37     |
| House MLR Test Set                                  | 0.704     | 0.0076   | 91.25     | 1.20     |
| Apartment MLR Training Set                          | 0.645     | 0.0036   | 67.20     | 0.58     |
| Apartment MLR Test Set                              | 0.633     | 0.0120   | 68.68     | 2.40     |

### Limitations

What are the limitations of your model?



## Random Forest Model

### Model details

Models tested, final model chosen, ...

-> MLR split houses & apartments from the beginning
-> RF little difference noted

### Performance

Performance metrics for the various models tested, visualizations, ...

Metrics: for training & test
Tested over 6 different random states, the mean R2 was, with a std dev of, meaning
mean RMSE was x with a std dev of, meaning 

| Category                                            | Mean R2   | Std R2   | Mean RMSE | Std RMSE |
|-----------------------------------------------------|-----------|----------|-----------|----------|
| Houses & Apartments Combined Training Set           | 0.961     | 0.0006   | 32.85     | 0.28     |
| Houses & Apartments Combined Test Set               | 0.713     | 0.0064   | 89.12     | 1.78     |

### Limitations

What are the limitations of your model?


## Usage

What are the dependencies, what scripts are there to train the model, how to generate predictions, ...

Pipelines:


## Maintainers

Who to contact in case of questions or issues?