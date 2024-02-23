# isolation_forest.py

from sklearn.ensemble import IsolationForest

def detect_anomalies_with_isolation_forest(data):
    """
    Detect anomalies using the Isolation Forest algorithm.

    Parameters:
    - data (DataFrame): The preprocessed data.

    Returns:
    - list: A list of indices of anomalies detected.
    """
    # Initialize Isolation Forest model
    isolation_forest_model = IsolationForest()

    # Fit the model to the data
    isolation_forest_model.fit(data)

    # Predict anomalies (outliers) using the trained model
    anomaly_predictions = isolation_forest_model.predict(data)

    # Extract indices of anomalies
    anomalies_indices = [i for i, prediction in enumerate(anomaly_predictions) if prediction == -1]

    return anomalies_indices
