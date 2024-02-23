# evaluation.py

def evaluate_anomaly_detection(isolation_forest_anomalies, dbscan_anomalies, true_anomalies):
    """
    Evaluate the performance of anomaly detection algorithms.

    Parameters:
    - isolation_forest_anomalies (list): List of indices of anomalies detected by Isolation Forest.
    - dbscan_anomalies (list): List of indices of anomalies detected by DBSCAN.
    - true_anomalies (list): List of indices of true anomalies in the dataset.

    Returns:
    - dict: A dictionary containing evaluation metrics.
    """
    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    isolation_forest_tp = len(set(isolation_forest_anomalies).intersection(true_anomalies))
    dbscan_tp = len(set(dbscan_anomalies).intersection(true_anomalies))
    
    isolation_forest_fp = len(set(isolation_forest_anomalies).difference(true_anomalies))
    dbscan_fp = len(set(dbscan_anomalies).difference(true_anomalies))
    
    isolation_forest_fn = len(set(true_anomalies).difference(isolation_forest_anomalies))
    dbscan_fn = len(set(true_anomalies).difference(dbscan_anomalies))
    
    # Calculate Precision, Recall, and F1-score for Isolation Forest
    isolation_forest_precision = isolation_forest_tp / (isolation_forest_tp + isolation_forest_fp + 1e-9)
    isolation_forest_recall = isolation_forest_tp / (isolation_forest_tp + isolation_forest_fn + 1e-9)
    isolation_forest_f1_score = 2 * (isolation_forest_precision * isolation_forest_recall) / (isolation_forest_precision + isolation_forest_recall + 1e-9)

    # Calculate Precision, Recall, and F1-score for DBSCAN
    dbscan_precision = dbscan_tp / (dbscan_tp + dbscan_fp + 1e-9)
    dbscan_recall = dbscan_tp / (dbscan_tp + dbscan_fn + 1e-9)
    dbscan_f1_score = 2 * (dbscan_precision * dbscan_recall) / (dbscan_precision + dbscan_recall + 1e-9)
    
    evaluation_metrics = {
        'Isolation Forest': {
            'Precision': isolation_forest_precision,
            'Recall': isolation_forest_recall,
            'F1-score': isolation_forest_f1_score
        },
        'DBSCAN': {
            'Precision': dbscan_precision,
            'Recall': dbscan_recall,
            'F1-score': dbscan_f1_score
        }
    }

    return evaluation_metrics
