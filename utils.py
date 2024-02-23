# utils.py

import numpy as np

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.

    Parameters:
    - point1 (array-like): Coordinates of the first point.
    - point2 (array-like): Coordinates of the second point.

    Returns:
    - float: Euclidean distance between the two points.
    """
    return np.sqrt(np.sum(np.square(np.array(point1) - np.array(point2))))

def normalize_data(data):
    """
    Normalize the data to have zero mean and unit variance.

    Parameters:
    - data (DataFrame): The data to be normalized.

    Returns:
    - DataFrame: Normalized data.
    """
    normalized_data = (data - data.mean()) / data.std()
    return normalized_data

def save_model(model, filename):
    """
    Save a machine learning model to a file.

    Parameters:
    - model: The machine learning model to be saved.
    - filename (str): The name of the file to save the model to.

    Returns:
    - None
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved successfully to {filename}.")
    except Exception as e:
        print(f"Error occurred while saving the model: {str(e)}")

def load_model(filename):
    """
    Load a machine learning model from a file.

    Parameters:
    - filename (str): The name of the file containing the model.

    Returns:
    - model: The loaded machine learning model.
    """
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {filename}.")
        return model
    except FileNotFoundError:
        print("Error: File not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {str(e)}")
        return None
