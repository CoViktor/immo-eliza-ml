import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import plotly.figure_factory as ff


def get_columns():
    ''' Used to get a list of all columns in the DataFrame that I want to
    include in my model. Assures no uncleaned data slips in when the source
    gets updated.

    Returns: List of columns that I want to include in my model.
    '''
    
    # Reverse this prcocess for dynamic external data handling
    return ['PostalZone', 'PropertyType', 'PropertySubType', 'Price',
            'ConstructionYear', 'BedroomCount',
            'LivingArea', 'Furnished', 'Fireplace', 'Terrace',
            'Garden', 'GardenArea', 'Facades', 'SwimmingPool', 'Condition',
            'EnergyConsumptionPerSqm']

# --------------- Exploring ----------------------------
# ---Univariates---
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

# ---Bivariates---
def covariates(X_train, y_train):
    """Analyzes and displays the correlation between numerical features in the
    training data and the target variable. It calculates Pearson correlation
    coefficients for each feature with the target and plots a heatmap of the
    correlation matrix.

    Parameters:
    - X_train (pd.DataFrame): Training features dataset.
    - y_train (pd.Series): Training target variable.

    Returns:
    None. Outputs correlation values and a heatmap plot.
    """
    X_train_numerical = X_train.select_dtypes(include=['number'])
    # Combine X_train and y_train into a single DataFrame for correlation analysis
    df = X_train_numerical.copy()
    df['Target'] = y_train
    # Iterate over the feature columns to calculate correlation with the target
    for feature_name in df.columns:
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

# --------------- Preprocessing ----------------------------
def cleaning_data(df, type):
    """Cleans the provided DataFrame by dropping missing values in specific
    columns, removing duplicates, filling missing values with predefined
    values, filtering rows based on certain conditions, adjusting text
    formats, converting data types, and trimming leading/trailing spaces.
    It also updates postal codes and filters columns based on relevance.

    Parameters:
    - df (pd.DataFrame): The dataset to be cleaned.

    Returns:
    pd.DataFrame: The cleaned dataset ready for further processing.
    """
    # Drop rows with empty values in 'Price' and 'LivingArea' columns
    df.dropna(subset=['Price'], inplace=True)
    # Remove duplicates in the 'ID' column and where all columns but 'ID' are equal
    df.drop_duplicates(subset='ID', inplace=True)
    df.drop_duplicates(subset=df.columns.difference(['ID']), keep='first', inplace=True)
    # Convert empty values to 0 for specified columns; assumption that if blank then 0
    columns_to_fill_with_zero = ['Furnished', 'Fireplace', 'Terrace', 'TerraceArea', 'Garden', 'GardenArea', 'SwimmingPool', 'BidStylePricing']
    df.loc[:, columns_to_fill_with_zero] = df.loc[:, columns_to_fill_with_zero].fillna(0)
    # Filter rows where SaleType == 'residential_sale' and BidStylePricing == 0
    df = df[(df['SaleType'] == 'residential_sale') & (df['BidStylePricing'] == 0)].copy()
    df = df.drop(columns=['BidStylePricing', 'SaleType'])
    # Adjust text format
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
    # Fill missing values with None and convert specified columns to float64 type
    columns_to_fill_with_none = ['EnergyConsumptionPerSqm']
    df[columns_to_fill_with_none] = df[columns_to_fill_with_none].where(df[columns_to_fill_with_none].notna(), None)
    columns_to_float64 = ['Price', 'LivingArea', 'TerraceArea', 'GardenArea', 'EnergyConsumptionPerSqm']
    df[columns_to_float64] = df[columns_to_float64].astype(float)
    # Convert specified columns to Int64 type
    columns_to_int64 = ['ConstructionYear', 'BedroomCount', 'Furnished', 'Fireplace', 'Terrace', 'Garden', 'Facades', 'SwimmingPool']
    df[columns_to_int64] = df[columns_to_int64].astype(float).round().astype('Int64')
    # Drop any ConstructionYear > current_year + 10
    current_year = datetime.now().year
    max_construction_year = current_year + 10
    df['ConstructionYear'] = df['ConstructionYear'].where(df['ConstructionYear'] <= max_construction_year, None)
    # Postal codes to string & postal zones
    df['PostalCode'] = df['PostalCode'].astype(str)
    df['PostalZone'] = df['PostalCode'].str.slice(0, 2)
    df = df.drop(columns=['PostalCode'])
    # Only include relevant columns
    df = df[get_columns()]
    
    return df

def one_hot(X_train, X_test, specific_columns=None):
    """Applies OneHotEncoding to the specified categorical columns.
    
    Parameters:
    - X_train: Training feature DataFrame.
    - X_test: Testing/validation feature DataFrame.
    - specific_columns: List of specific columns to encode. If None, all
      object/category columns will be encoded.
    
    Returns:
    - X_train_preprocessed: The training DataFrame with specified categorical
      columns one-hot encoded.
    - X_test_preprocessed: The testing/validation DataFrame with specified
      categorical columns one-hot encoded.
    """
    # print('One Hot Encoding...')
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

def feature_engineer(df, type):
    """Applies various feature engineering techniques to the dataframe based on
    the property type. This function is a placeholder for actual feature
    engineering steps, which can include calculations, transformations, and
    dropping or creating new features.

    Parameters:
    - df (pd.DataFrame): The dataset to be processed.
    - type (str): The type of property ('HOUSE' or 'APARTMENT').

    Returns:
    pd.DataFrame: The dataframe after applying feature engineering.
    """
    # Insert feature engineering steps here

    return df

def drop_outliers(df):
    """Identifies and removes outliers from specific columns of the dataframe
    based on the Interquartile Range (IQR) method. Outliers are defined as
    values outside 1.5 * IQR from the Q1 and Q3 quartiles.

    Parameters:
    - df (pd.DataFrame): The dataset from which outliers will be removed.

    Returns:
    pd.DataFrame: The dataframe after removing outliers.
    """
    # Drop outliers of specific columns
    for column in df.columns:
        if column in ['Price', 'BedroomCount', 'ConstructionYear', 'Facades', 'LivingArea',
                      'GardenArea', 'EnergyConsumptionPerSqm']:
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

    return df

def impute_missings(df):
    """Imputes missing values in specific columns with the median value for
    those columns. The function can differentiate imputation strategies
    based on property type.

    Parameters:
    - df (pd.DataFrame): The dataset with missing values.

    Returns:
    pd.DataFrame: The dataframe after imputing missing values.
    """
    # Imputing with median
    if type == 'HOUSE':
        for col in ['ConstructionYear', 'Facades']:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
    elif type == 'APARTMENT':
        for col in ['ConstructionYear', 'Facades']:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    return df

def drop_missings(df):
    """Drops rows from the dataframe where any data is missing, with the option
    to exclude certain columns from this criteria.

    Parameters:
    - df (pd.DataFrame): The dataset with potential missing values.

    Returns:
    pd.DataFrame: The dataframe after dropping rows with missing values.
    """
    # Drop all remaining missings
    for col in df.columns:
        # if col not in exclude:
        df = df.dropna(subset=[col])

    return df

def scale_data(df):
    """Scales numeric columns in the dataframe using StandardScaler. This is
    useful for normalizing the range of continuous variables for certain
    types of modeling.

    Parameters:
    - df (pd.DataFrame): The dataset to be scaled.

    Returns:
    pd.DataFrame: The dataframe with scaled numeric values.
    """
    # Scale data
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

#-----------------------------------------------------------------------------
def preprocess_data(df, type):
    '''Takes a DataFrame and preprocesses it. This includes cleaning, feature
    engineering and scaling.

    Returns: Preprocessed DataFrame
    '''
    # print('Preprocessing data...')
    clean_df = cleaning_data(df, type)
    clean_df = drop_missings(clean_df)
    clean_df = drop_outliers(clean_df) 
    new_df = feature_engineer(clean_df, type)
    X = new_df.drop('Price', axis=1)  
    y = new_df['Price'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)
    # covariates(X_train, y_train)
    X_train_encoded, X_test_encoded = one_hot(X_train, X_test)
    X_train_imputed = impute_missings(X_train_encoded)
    X_test_imputed = impute_missings(X_test_encoded)
    X_train_rescaled = scale_data(X_train_imputed)
    X_test_rescaled = scale_data(X_test_imputed)
    # explore_data(X_train)
    # explore_data(X_test)
    return X_train_rescaled, X_test_rescaled, y_train, y_test

