import pandas as pd
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Getting the data
df = pd.read_csv('./data/raw_data.csv')

# --------------- Exploration of df & univariates -----------------------------


def get_columns():
    ''' Used to get a list of all columns in the DataFrame that I want to
    include in my model. Assures no uncleaned data slips in when the source
    gets updated.

    Returns: List of columns that I want to include in my model.
    '''

    return ['Region', 'Province', 'PostalCode', 'PropertySubType', 'Price',
            'SaleType', 'BidStylePricing', 'ConstructionYear', 'BedroomCount',
            'LivingArea', 'Furnished', 'Fireplace', 'Terrace', 'TerraceArea',
            'Garden', 'GardenArea', 'Facades', 'SwimmingPool', 'Condition',
            'EnergyConsumptionPerSqm', 'ListingCreateDate', 
            'ListingExpirationDate']

    # columns removed after further analysis: 'EPCScore', , 'Latitude', 'Longitude'

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
    print('missing price & livingarea removed:')
    print(df.shape)

    # Task 2: Remove duplicates in the 'ID' column and where all columns but 'ID' are equal
    df.drop_duplicates(subset='ID', inplace=True)
    df.drop_duplicates(subset=df.columns.difference(['ID']), keep='first', inplace=True)
    print('duplicates removed by id:')
    print(df.shape)

    # Task 3: Only include relevant columns
    df = df[get_columns()]
    print('irrelevant columns excluded')
    print(df.shape)

    # Task 4: Convert empty values to 0 for specified columns; assumption that if blank then 0
    columns_to_fill_with_zero = ['Furnished', 'Fireplace', 'Terrace', 'TerraceArea', 'Garden', 'GardenArea', 'SwimmingPool', 'BidStylePricing']
    df.loc[:, columns_to_fill_with_zero] = df.loc[:, columns_to_fill_with_zero].fillna(0)
    print('empty values filled with 0')

    # Task 5: Filter rows where SaleType == 'residential_sale' and BidStylePricing == 0
    df = df[(df['SaleType'] == 'residential_sale') & (df['BidStylePricing'] == 0)].copy()
    print('non residential sales excluded')
    print(df.shape)

    # Task 6: Adjust text format
    columns_to_str = ['Region', 'Province', 'PropertyType', 'PropertySubType', 'Condition']
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
    print('text format adjusted')

    # Task 9: Fill missing values with None and convert specified columns to float64 type
    columns_to_fill_with_none = ['EnergyConsumptionPerSqm']
    df[columns_to_fill_with_none] = df[columns_to_fill_with_none].where(df[columns_to_fill_with_none].notna(), None)

    columns_to_float64 = ['Price', 'LivingArea', 'TerraceArea', 'GardenArea', 'EnergyConsumptionPerSqm']
    df[columns_to_float64] = df[columns_to_float64].astype(float)

    # Task 10: Convert specified columns to Int64 type
    columns_to_int64 = ['ConstructionYear', 'BedroomCount', 'Furnished', 'Fireplace', 'Terrace', 'Garden', 'Facades', 'SwimmingPool']
    df[columns_to_int64] = df[columns_to_int64].astype(float).round().astype('Int64')
    print('columns converted to Int64 and missings assigned to None')

    # Task 11: Drop any ConstructionYear > current_year + 10
    current_year = datetime.now().year
    max_construction_year = current_year + 10
    df['ConstructionYear'] = df['ConstructionYear'].where(df['ConstructionYear'] <= max_construction_year, None)
    print('ConstructionYear > current_year + 10 assigned as missing')

    # Task 12: Convert 'ListingCreateDate' & 'ListingExpirationDate' to Date type with standard DD/MM/YYYY format
    date_columns = ['ListingCreateDate', 'ListingExpirationDate']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%Y-%m-%dT%H:%M:%S.%f%z', errors='coerce')



    return df

# --------------- Cat_handling ----------------------------
def one_hot(df):
    # Onehotencoding / dummies
    # -> PropertySubType, Region, Province

    return df

# --------------- Feature engineering ----------------------------
# Possible things to drop:if 2 columns combined: drop others to prevent covariation (run & check)
# -> total-area, time until exp data
# Run model to test changing cats into binaries:
# -> condition 
def feature_engineer(df): 
    # Calculate 'TotalArea'
    df['TotalArea'] = df['LivingArea'] + df['GardenArea'] + df['TerraceArea']
    df = df.drop(columns=['LivingArea', 'GardenArea', 'TerraceArea'])

    # Calc time until exp date
    # Combine certain categoricals
    # Condition -> Good, To_Be_Done_Up, To_Renovate, Just_Renovated, As_New, To_Restore
    # -> into ConditionBinary, where 0: To_Repair, 1: Good_Condition

    return df

# ----------------- Split ----------------------------
def split_data(df):
    print('Splitting data...')
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    return train_data, test_data

# --------------- Drop & impute -----------------------------
def drop_and_impute(df):
    # Drop outliers of specific columns
    for column in df.columns:
        if column in ['ConstructionYear', 'Facades', 'LivingArea',
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
            elif column in []:
                df = df.drop(lower.index)  
            else:
                df = df.drop(outliers.index)
            print(f'dropped {column} outliers, shape: {df.shape}')

    # Filling missings for certain columns with mean -> Document this, as this can weaken correlations & give bias
    # Imputing with median
    for col in ['ConstructionYear', 'Facades']:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f'Imputed missing values in {col} with median: {median_value}')

    # Drop all remaining missings
    print('Dropping all missings...')
    big_drops = ['Condition', 'EnergyConsumptionPerSqm']  # -> keep included, but drop when multivariate analysis
    for col in df.columns:
        if col not in big_drops:
            df = df.dropna(subset=[col])
            print(f'Dropped missings for {col}, shape: {df.shape}')

    return df

# --------------- Scaling ----------------------------
def scale_data(df):
    print('Scaling data...')
    # Scale data
    scaler = StandardScaler() ## ????? -> check out manually
    df = scaler.fit_transform(df)
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
    clean_df =cleaning_data(df)
    # new_df = feature_engineer(cat_handled_df)
    # SPLIT HERE
    # one_hot_df = one_hot(clean_df)
    # train_data, test_data = split_data(new_df)???? IN HERE OR DIFFERENT PART
    # imputed_data = drop_and_impute(train_data)???? ^
    # scaled_data = scale_data(imputed_data) ???? ^^ 
    return clean_df # return X_train_preprocessed, X_test_preprocessed, y_train, y_test



# After split for df in [train_data, test_data]:
for type in ['HOUSE', 'APARTMENT']:
    print(f'\n\n\n---{type}---')
    data = df[df['PropertyType'] == type].copy()
    prepped_data = preprocess_data(data) # X_train, X_test, y_train, y_test = preprocess_data(data)
    explore_data(prepped_data) # explore_data(X_train)
    print(f'---{type} OVER---\n\n\n')
    
    