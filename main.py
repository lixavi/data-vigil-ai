from data_loader import load_data
from anomaly_detector import AnomalyDetector
from visualization import visualize_data, visualize_anomalies
from evaluation import evaluate_detection

def main():
    # Load data
    data = load_data()

    # Initialize anomaly detector
    detector = AnomalyDetector()

    # Detect anomalies
    anomalies = detector.detect_anomalies(data)

    # Visualize data and anomalies
    visualize_data(data)
    visualize_anomalies(data, anomalies)

    # Evaluate detection performance
    evaluate_detection(data, anomalies)

if __name__ == "__main__":
    main()
