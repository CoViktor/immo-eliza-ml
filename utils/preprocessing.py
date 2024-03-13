import pandas as pd

# Getting the data
df = pd.read_csv('./data/raw_data.csv')

# --------------- Exploration of df & univariates -----------------------------


def explore_data():
    '''Takes a DataFrame and explores it. This includes shape, columns,
    describe, uniques & missings per column + value counts for certain
    columns.

    Returns: Print statements about data
    '''
    print('df.shape', df.shape)
    print('df.head()', df.head())
    print('df.columns', df.columns)
    print('df.describe()', df.describe())

    for col in df.columns:
        print(f'{col}\n nunique: {df[col].nunique()}')
        print(f' missings: {df[col].isna().sum()}')

    print("value counts:")
    for col in ['Region', 'Province', 'PropertyType', 'PropertySubType',
                'SaleType', 'BedroomCount', 'Condition', 'Facades']:
        print(df[col].value_counts())

def preprocess_data():
    '''Takes a DataFrame and preprocesses it. This includes cleaning, feature
    engineering and scaling.

    Returns: Preprocessed DataFrame
    '''

# --------------- Cleaning ----------------------------
def get_columns():
    ''' Used to get a list of all columns in the DataFrame that I want to
    include in my model. Assures no uncleaned data slips in when the source
    gets updated.

    Returns: List of columns that I want to include in my model.
    '''

    return ['Region', 'Province', 'PropertyType', 'PropertySubType', 'Price',
            'SaleType', 'BidStylePricing', 'ConstructionYear', 'BedroomCount',
            'LivingArea', 'Furnished', 'Fireplace', 'Terrace', 'TerraceArea',
            'Garden', 'GardenArea', 'Facades', 'SwimmingPool', 'Condition',
            'EPCScore', 'EnergyConsumptionPerSqm', 'Latitude', 'Longitude',
            'ListingCreateDate', 'ListingExpirationDate']


df = df[get_columns()]
explore_data()

# --------------- Feature engineering ----------------------------

# --------------- Scaling ----------------------------
