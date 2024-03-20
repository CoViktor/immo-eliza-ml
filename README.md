# immo-eliza-ml 🏠
![Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Pandas](https://img.shields.io/badge/uses-Pandas-blue.svg)
![Matplotlib](https://img.shields.io/badge/uses-Matplotlib-blue.svg)
![Plotly](https://img.shields.io/badge/uses-Plotly-ff69b4.svg)
![Scikit-learn](https://img.shields.io/badge/uses-Scikit--learn-orange.svg)
![Statsmodels](https://img.shields.io/badge/uses-Statsmodels-brightgreen.svg)
![Joblib](https://img.shields.io/badge/uses-Joblib-red.svg)


## 🏢 Description

This project is focused on predicting real estate prices using machine learning models. It includes preprocessing of the data, feature engineering, training multiple linear regression and random forest models, and evaluating their performance. The project structure is designed to separate the processes into modules for better readability and maintenance.

Make sure to check the models card for clarification on the models.

<img src="https://jiwall.com/wp-content/uploads/2021/06/Blank-768-x-513-copy-copy-copy-copy-copy-1-300x200.jpg" width="400" height="auto"/>

### Issues and update requests
- If you encounter any issues or have suggestions for improvements, please feel free to open an issue in the repository.
- Contributions to enhance the functionality or performance of the models are always welcome.

Find me on [LinkedIn](https://www.linkedin.com/in/viktor-cosaert/) for collaboration, feedback, or to connect.

## 📦 Repo structure
```.
├── data/
│ └── raw_data.csv
├── models/
│ └── *.joblib
├── utils/
│ ├── data_import.py
│ ├── model_pipeline.py
│ ├── predict.py
│ ├── preprocess.py
│ └── train.py
├── .gitattributes
├── .gitignore
├── main.py
├── modelscard.md
├── README.md
└── requirements.txt 
```

## 🚧 Installation 

1. Clone the repository to your local machine.

    ```
    git clone https://github.com/your-github-username/real-estate-price-prediction.git
    ```

2. Navigate to the project directory and install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

## ✔️ Usage 

Start by running `main.py` to execute the modeling pipeline. You can comment out the model you do not wish to run. The script will process the data from `./data/raw_data.csv`, apply preprocessing, feature engineering, train the models, and evaluate their performance.

For more detailed instructions on each step of the process, refer to the scripts within the `utils/` folder.

Example snippet from `main.py`:

```python
import pandas as pd
from utils.model_pipelines import run_mlr_model, run_rf_model

df = pd.read_csv('./data/raw_data.csv')

# Uncomment the model you wish to run
run_mlr_model(df)
# run_rf_model(df)
```

```python
# Output example:
HOUSE MLR Training Set Metrics:
R2: 0.726, average error of 86.71K euros (RMSE)

HOUSE MLR Test Set Metrics:
R2: 0.703, average error of 93.32K euros (RMSE)

APARTMENT MLR Training Set Metrics:
R2: 0.640, average error of 67.81K euros (RMSE)

APARTMENT MLR Test Set Metrics:
R2: 0.653, average error of 66.29K euros (RMSE)

Houses & apartments combined Random Forest Training Set Metrics:
R2: 0.960, average error of 33.18K euros (RMSE)

Houses & apartments combined Random Forest Test Set Metrics:
R2: 0.712, average error of 87.92K euros (RMSE)
```

## ⏱️ Project Timeline
The initial setup of this project was completed in 4 days.

## 🔧 Updates & Upgrades
### Recent Updates

### Planned Upgrades
- **Data Pipeline Enhancement**: Improve the automation of data preprocessing and feature selection.
- **Model Experimentation**: Explore additional machine learning models and techniques for accuracy improvement.

## 📌 Personal Note
This project was developed as part of my training into machine learning at [BeCode](https://becode.org/). It serves as a practical application of data preprocessing, feature engineering, and model training and evaluation.

Find me on [LinkedIn](https://www.linkedin.com/in/viktor-cosaert/) for collaboration, feedback, or to connect.