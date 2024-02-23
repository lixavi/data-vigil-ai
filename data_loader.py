# data_loader.py

import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("Error: File not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {str(e)}")
        return None

def preprocess_data(data):
    """
    Preprocess the loaded data.

    Parameters:
    - data (DataFrame): The loaded data as a pandas DataFrame.

    Returns:
    - DataFrame: The preprocessed data.
    """
    # Add preprocessing steps here, such as handling missing values, encoding categorical variables, etc.
    preprocessed_data = data.dropna()  # Example: Drop rows with missing values
    return preprocessed_data

