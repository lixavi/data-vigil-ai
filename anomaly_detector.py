# anomaly_detector.py

from isolation_forest import detect_anomalies_with_isolation_forest
from dbscan import detect_anomalies_with_dbscan

def detect_anomalies(data):
    """
    Detect anomalies in the data using Isolation Forest and DBSCAN algorithms.

    Parameters:
    - data (DataFrame): The preprocessed data.

    Returns:
    - list: A list of indices of anomalies detected.
    """
    # Detect anomalies using Isolation Forest
    isolation_forest_anomalies = detect_anomalies_with_isolation_forest(data)

    # Detect anomalies using DBSCAN
    dbscan_anomalies = detect_anomalies_with_dbscan(data)

    # Combine anomalies detected by both algorithms
    combined_anomalies = list(set(isolation_forest_anomalies + dbscan_anomalies))

    return combined_anomalies

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

#def load_data(file_path):
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
