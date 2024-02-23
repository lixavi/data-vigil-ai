# visualization.py

import matplotlib.pyplot as plt

def visualize_anomalies(data, isolation_forest_anomalies, dbscan_anomalies):
    """
    Visualize the detected anomalies.

    Parameters:
    - data (DataFrame): The preprocessed data.
    - isolation_forest_anomalies (list): List of indices of anomalies detected by Isolation Forest.
    - dbscan_anomalies (list): List of indices of anomalies detected by DBSCAN.

    Returns:
    - None
    """
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], color='blue', label='Normal Data')
    
    # Highlight anomalies detected by Isolation Forest
    plt.scatter(data.iloc[isolation_forest_anomalies, 0], data.iloc[isolation_forest_anomalies, 1],
                color='red', label='Isolation Forest Anomalies')
    
    # Highlight anomalies detected by DBSCAN
    plt.scatter(data.iloc[dbscan_anomalies, 0], data.iloc[dbscan_anomalies, 1],
                color='green', label='DBSCAN Anomalies')

    plt.title('Detected Anomalies')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
