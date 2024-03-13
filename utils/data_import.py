import pandas as pd

def import_data():
    """
    Import raw immo data from a CSV file and store locally.
    
    Returns:
    data (DataFrame): DataFrame containing the imported raw data from previous project.
    """
    # Importing the data from the URL
    data = pd.read_csv("https://raw.githubusercontent.com/bear-revels/immo-eliza-scraping-Python_Pricers/main/data/all_property_details.csv")

    # Storing locally without the index column
    data.to_csv('./data/raw_data.csv', index = False)

    return data
