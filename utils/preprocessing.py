import pandas as pd

# Getting the data
df = pd.read_csv('./data/raw_data.csv')

# --------------- Exploration of df & univariates -----------------------------

# print('df.shape', df.shape)
# print('df.head()', df.head())
# print('df.columns', df.columns)
# print('df.info', df.info)
# print('df.describe()', df.describe())

# for col in df.columns:
#     print(col, df[col].nunique())
#     print(df[col].isna().sum())

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

# --------------- Feature engineering ----------------------------

# --------------- Scaling ----------------------------
