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
