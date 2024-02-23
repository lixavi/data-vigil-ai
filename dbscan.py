# dbscan.py

from sklearn.cluster import DBSCAN

def detect_anomalies_with_dbscan(data):
    """
    Detect anomalies using the DBSCAN clustering algorithm.

    Parameters:
    - data (DataFrame): The preprocessed data.

    Returns:
    - list: A list of indices of anomalies detected.
    """
    # Initialize DBSCAN model
    dbscan_model = DBSCAN(eps=0.5, min_samples=5)

    # Fit the model to the data
    dbscan_model.fit(data)

    # Extract labels assigned by DBSCAN (-1 for anomalies, >=0 for inliers)
    labels = dbscan_model.labels_

    # Extract indices of anomalies (points labeled as -1)
    anomalies_indices = [i for i, label in enumerate(labels) if label == -1]

    return anomalies_indices
