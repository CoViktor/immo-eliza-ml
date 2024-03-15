import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff



# --------------- Exploration of df & univariates -----------------------------


def get_columns():
    ''' Used to get a list of all columns in the DataFrame that I want to
    include in my model. Assures no uncleaned data slips in when the source
    gets updated.

    Returns: List of columns that I want to include in my model.
    '''

    return ['Region', 'PostalCode', 'PropertyType', 'PropertySubType', 'Price',
            'SaleType', 'BidStylePricing', 'ConstructionYear', 'BedroomCount',
            'LivingArea', 'Furnished', 'Fireplace', 'Terrace', 'TerraceArea',
            'Garden', 'GardenArea', 'Facades', 'SwimmingPool', 'Condition',
            'EnergyConsumptionPerSqm', 'Province', 'Latitude', 'Longitude']

    # columns removed after further analysis: 'EPCScore', , 'Latitude', 'Longitude', 'ListingCreateDate', 
    #        'ListingExpirationDate'
    # house
    # Low corr, but improvable: PostalCode
    # Ok corr: constryr, bedrmcount, livingarea (check covariance here), gardenarea, facades, swimmingpool, energycons, Region(gofor postalcode KNN); Province too, condition (combine?), subtype ok-ish
    # Low corr, try removing: furnished, fireplace, terrace & terracearea, garden
    # app
    # Low corr, but improvable: 
    # Ok corr: postalcode, bedroomcount, living area, fireplace, terrace & area, swimmingpool, energycons, Region(gofor postalcode KNN); Province too
    # Low corr, try removing: constryr, furnished, garden  & area, facades, 
def explore_data(df):
    '''Takes a DataFrame and explores it. This includes shape, columns,
    describe, uniques & missings per column + value counts for certain
    columns.
    Only for the columns I want to include in my model.

    Returns: Print statements about data
    '''
    print('\n~Exploring data~\nshape:')
    print('df.shape', df.shape)
    # print('df.head()', df.head())
    print('df.columns', df.columns)
    # print('df.describe()', df.describe())

    for col in df.columns:
        print(f'{col}\n nunique: {df[col].nunique()}')
        print(f' missings: {df[col].isna().sum()}')

    print("value counts:")
    for col in df.columns:
        if col not in ['PostalCode', 'Price', 'SaleType', 'BidStylePricing',
                       'LivingArea', 'TerraceArea',
                       'GardenArea', 'Facades', 'EnergyConsumptionPerSqm',
                       'ListingCreateDate', 'ListingExpirationDate']:
        # Optional if in list['Region', 'Province', 'PropertyType', 'PropertySubType', 'SaleType', 'BedroomCount', 'Condition', 'Facades']
            print(df[col].value_counts())

    for column in df.columns:
        if column in ['Price', 'ConstructionYear','BedroomCount', 'LivingArea', 'TerraceArea', 'GardenArea', 'Facades', 'EnergyConsumptionPerSqm']:
            print(f'--{column}--')
            # setting IQR
            df[column].dropna()
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            # identify outliers
            threshold = 1.5
            outliers = df[(df[column] < Q1 - threshold * IQR) | (df[column] > Q3 + threshold * IQR)]
            lower = df[(df[column] < Q1 - threshold * IQR)]
            upper = df[(df[column] > Q3 + threshold * IQR)]
            print(len(outliers), f'outliers \nbelow Q1 - 1.5*IQR: {len(lower)}\nabove Q3 + 1.5*IQR (): {len(upper)}')
            print(f'lower = inbetween {lower[column].max()} and {lower[column].min()}')
            print(f'upper = inbetween {upper[column].min()} and {upper[column].max()}')

def covariates(X_train, y_train):
    # Combine X_train and y_train into a single DataFrame for correlation analysis
    # Assuming y_train is a Series; adjust if it's a DataFrame
    df = X_train.copy()
    df['Target'] = y_train
    
    # Iterate over the feature columns to calculate correlation with the target
    for feature_name in X_train.columns:
        if pd.api.types.is_numeric_dtype(df[feature_name]):
            # Calculate correlation between feature and target, excluding NaN values
            valid_idx = df[feature_name].notna() & df['Target'].notna()
            corr, _ = pearsonr(df.loc[valid_idx, feature_name], df.loc[valid_idx, 'Target'])
            print(f'The correlation between feature {feature_name} and the target is {round(corr, 2)}')
        else:
            print(f"Skipping {feature_name}, not suitable for correlation calculation.")

    # Plot the correlation matrix including the target variable
    correlation_matrix = df.corr()
    fig = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index),
        annotation_text=correlation_matrix.round(2).values,
        showscale=True,
        colorscale='Viridis'
    )
    fig.show()

# --------------- Cleaning ----------------------------
def cleaning_data(df):
    '''Takes a DataFrame and cleans it. This includes dropping rows with
    empty values in 'Price' and 'LivingArea' columns, removing duplicates
    in the 'ID' column and where all columns but 'ID' are equal.

    Returns: Cleaned DataFrame
    '''
    print('~Cleaning data~\noriginal shape:')
    print(df.shape)

    # Task 1: Drop rows with empty values in 'Price' and 'LivingArea' columns
    df.dropna(subset=['Price'], inplace=True)
    # print('missing price & livingarea removed:')
    # print(df.shape)

    # Task 2: Remove duplicates in the 'ID' column and where all columns but 'ID' are equal
    df.drop_duplicates(subset='ID', inplace=True)
    df.drop_duplicates(subset=df.columns.difference(['ID']), keep='first', inplace=True)
    # print('duplicates removed by id:')
    # print(df.shape)

    # Task 3: Only include relevant columns
    df = df[get_columns()]
    # print('irrelevant columns excluded')
    # print(df.shape)

    # Task 4: Convert empty values to 0 for specified columns; assumption that if blank then 0
    columns_to_fill_with_zero = ['Furnished', 'Fireplace', 'Terrace', 'TerraceArea', 'Garden', 'GardenArea', 'SwimmingPool', 'BidStylePricing']
    df.loc[:, columns_to_fill_with_zero] = df.loc[:, columns_to_fill_with_zero].fillna(0)
    # print('empty values filled with 0')

    # Task 5: Filter rows where SaleType == 'residential_sale' and BidStylePricing == 0
    df = df[(df['SaleType'] == 'residential_sale') & (df['BidStylePricing'] == 0)].copy()
    df = df.drop(columns=['BidStylePricing', 'SaleType'])
    # print('non residential sales excluded')
    # print(df.shape)

    # Task 6: Adjust text format
    columns_to_str = ['Region', 'Province', 'PropertyType', 'PropertySubType', 'Condition']  # 
    def adjust_text_format(x):
        if isinstance(x, str):
            return x.title()
        else:
            return x
    df.loc[:, columns_to_str] = df.loc[:, columns_to_str].applymap(adjust_text_format)
    # Remove leading and trailing spaces from string columns
    df.loc[:, columns_to_str] = df.loc[:, columns_to_str].apply(lambda x: x.str.strip() if isinstance(x, str) else x)
    # Replace the symbol '�' with 'e' in all string columns
    df = df.applymap(lambda x: x.replace('�', 'e') if isinstance(x, str) else x)
    # print('text format adjusted')

    # Task 9: Fill missing values with None and convert specified columns to float64 type
    columns_to_fill_with_none = ['EnergyConsumptionPerSqm']
    df[columns_to_fill_with_none] = df[columns_to_fill_with_none].where(df[columns_to_fill_with_none].notna(), None)

    columns_to_float64 = ['Price', 'LivingArea', 'TerraceArea', 'GardenArea', 'EnergyConsumptionPerSqm']
    df[columns_to_float64] = df[columns_to_float64].astype(float)

    # Task 10: Convert specified columns to Int64 type
    columns_to_int64 = ['ConstructionYear', 'BedroomCount', 'Furnished', 'Fireplace', 'Terrace', 'Garden', 'Facades', 'SwimmingPool']
    df[columns_to_int64] = df[columns_to_int64].astype(float).round().astype('Int64')
    # print('columns converted to Int64 and missings assigned to None')

    # Task 11: Drop any ConstructionYear > current_year + 10
    current_year = datetime.now().year
    max_construction_year = current_year + 10
    df['ConstructionYear'] = df['ConstructionYear'].where(df['ConstructionYear'] <= max_construction_year, None)
    # print('ConstructionYear > current_year + 10 assigned as missing')

    # Task 12: Convert 'ListingCreateDate' & 'ListingExpirationDate' to Date type with standard DD/MM/YYYY format -> excluded for now
    # date_columns = ['ListingCreateDate', 'ListingExpirationDate']
    # for col in date_columns:
    #     df[col] = pd.to_datetime(df[col], format='%Y-%m-%dT%H:%M:%S.%f%z', errors='coerce')

    return df

# --------------- One Hot Encoding ----------------------------
def one_hot(X_train, X_test, specific_columns=None):
    """
    Applies OneHotEncoding to the specified categorical columns.
    
    Parameters:
    - X_train: Training feature DataFrame.
    - X_test: Testing/validation feature DataFrame.
    - specific_columns: List of specific columns to encode. If None, all object/category columns will be encoded.
    
    Returns:
    - X_train_preprocessed: The training DataFrame with specified categorical columns one-hot encoded.
    - X_test_preprocessed: The testing/validation DataFrame with specified categorical columns one-hot encoded.
    """
    print('One Hot Encoding...')
    if specific_columns is None:
        # Automatically select all object and category dtype columns if specific_columns is not provided
        categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
    else:
        # Or focus on the specific columns provided
        categorical_columns = specific_columns
    
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    
    # Fit and transform the training data, and transform the testing data
    X_train_encoded = encoder.fit_transform(X_train[categorical_columns]).toarray()
    X_test_encoded = encoder.transform(X_test[categorical_columns]).toarray()
    
    # Convert encoded data back to DataFrames
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(), index=X_train.index)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(), index=X_test.index)
    
    # Drop original categorical columns and concatenate the encoded ones
    X_train_preprocessed = pd.concat([X_train.drop(columns=categorical_columns), X_train_encoded_df], axis=1)
    X_test_preprocessed = pd.concat([X_test.drop(columns=categorical_columns), X_test_encoded_df], axis=1)

    return X_train_preprocessed, X_test_preprocessed

# --------------- Feature engineering ----------------------------
# Possible things to drop:if 2 columns combined: drop others to prevent covariation (run & check)
# -> total-area, time until exp data
# Run model to test changing cats into binaries:
# -> condition 
def feature_engineer(df):
    print('Feature engineering...') 
    # Calculate 'TotalArea'
    # df['TotalArea'] = df['LivingArea'] + df['GardenArea'] + df['TerraceArea']
    # df = df.drop(columns=['LivingArea', 'GardenArea', 'TerraceArea'])
    
    # Calc time until exp date -> seems not that useful
    # df['ListingCreateYear'] = df['ListingCreateDate'].dt.year
    # df['ListingCreateMonth'] = df['ListingCreateDate'].dt.month
    # df['Days_online'] = (df['ListingExpirationDate'] - df['ListingCreateDate']).dt.days
    # df.drop(['ListingCreateDate', 'ListingExpirationDate', 'ListingCreateYear', 'ListingCreateMonth'], axis=1, inplace=True)

    # def condition_to_binary(condition):
    #     if condition in ['Good', 'Just_Renovated', 'As_New']:
    #         return 1  # Good Condition
    #     else:
    #         return 0  # To Repair or To Renovate
    # # Apply the function to the Condition column
    # df['ConditionBinary'] = df['Condition'].apply(condition_to_binary)
    # df = df.drop(columns=['Condition'])
    return df

# --------------- Drop & impute -----------------------------
def drop_outliers(df):
    print('Outliers handling...')
    # Drop outliers of specific columns
    for column in df.columns:
        if column in ['Price', 'BedroomCount', 'ConstructionYear', 'Facades', 'LivingArea',
                      'GardenArea', 'EnergyConsumptionPerSqm']:
            # optional includes: 'Price', 'TerraceArea','BedroomCount' 
            print(f'--{column}--')
            # setting IQR
            df[column].dropna()
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            # identify outliers
            threshold = 1.5
            outliers = df[(df[column] < Q1 - threshold * IQR) | (df[column] > Q3 + threshold * IQR)]
            lower = df[(df[column] < Q1 - threshold * IQR)]
            upper = df[(df[column] > Q3 + threshold * IQR)]
            if column in ['ConstructionYear']:
                df = df.drop(upper.index)
            else:
                df = df.drop(outliers.index)
            # print(f'dropped {column} outliers, shape: {df.shape}')
    return df

def impute_missings(df):
    # Filling missings for certain columns with mean -> Document this, as this can weaken correlations & give bias
    # Imputing with median
    print('Imputing missings...')
    for col in ['ConstructionYear', 'Facades']:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        # print(f'Imputed missing values in {col} with median: {median_value}')
    return df

def drop_missings(df):
    # Drop all remaining missings
    print('Dropping all missings...')
    exclude= ['ConstructionYear', 'Facades']
    big_drops = ['Condition', 'EnergyConsumptionPerSqm']  # -> keep included, but drop when multivariate analysis
    for col in df.columns:
        # if col not in exclude:
        df = df.dropna(subset=[col])
        # print(f'Dropped missings for {col}, shape: {df.shape}')

    return df

# --------------- Scaling ----------------------------
def scale_data(df):
    print('Scaling data...')
    # Scale data
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler() ## ????? -> check out manually
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# --------------- Preprocessing ----------------------------
    # clean, feature engineer, SPLIT, drop&impute, scale
    # drop_and_impute only after splitting into train and test
def preprocess_data(df):
    '''Takes a DataFrame and preprocesses it. This includes cleaning, feature
    engineering and scaling.

    Returns: Preprocessed DataFrame
    '''
    print('Preprocessing data...')
    clean_df = cleaning_data(df)
    clean_df = drop_missings(clean_df)
    clean_df = drop_outliers(clean_df) 
    new_df = feature_engineer(clean_df)  # -> can be done after model is written to play around
    print("Before splitting:", new_df.shape)
    X = new_df.drop('Price', axis=1)  
    y = new_df['Price'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"After splitting:, X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}, y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape}")
    X_train_encoded, X_test_encoded = one_hot(X_train, X_test)
    # covariates(X_train_encoded, y_train)
    X_train_imputed = impute_missings(X_train_encoded)
    X_test_imputed = impute_missings(X_test_encoded)
    X_train_rescaled = scale_data(X_train_imputed)
    X_test_rescaled = scale_data(X_test_imputed)
    # explore_data(X_train)
    # explore_data(X_test)
    return X_train_rescaled, X_test_rescaled, y_train, y_test

