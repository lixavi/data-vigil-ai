from data_preprocessing import preprocess_data
from isolation_forest_anomaly_detection import detect_anomalies_with_isolation_forest
from dbscan_clustering_anomaly_detection import detect_anomalies_with_dbscan
from visualization import visualize_anomalies

def main():
    # Step 1: Preprocess the data
    preprocessed_data = preprocess_data('data.csv')

    # Step 2: Detect anomalies using Isolation Forest algorithm
    isolation_forest_anomalies = detect_anomalies_with_isolation_forest(preprocessed_data)

    # Step 3: Detect anomalies using DBSCAN clustering algorithm
    dbscan_anomalies = detect_anomalies_with_dbscan(preprocessed_data)

    # Step 4: Visualize the detected anomalies
    visualize_anomalies(preprocessed_data, isolation_forest_anomalies, dbscan_anomalies)

if __name__ == "__main__":
    main()
