import torch
import yaml
from data.data_loader import load_data
from data.preprocess import preprocess_data, split_data, convert_windows_to_tensor
from utils.anomaly_detection import calculate_anomaly_scores, group_anomalies
from utils.evaluation import determine_threshold
from utils.explanation import explain_anomalies_for_groups
from models import load_or_train_models

def main():
    data_path = input("Please enter the path to the data file: ")

    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    df = load_data(data_path)
    scaled_data = preprocess_data(df)
    train_data, dev_data, test_data = split_data(scaled_data)

    window_size = config["window_size"]
    train_windows, validation_windows, test_windows = convert_windows_to_tensor(
        train_data, dev_data, test_data, window_size, device
    )

    #print("Train windows tensor shape:", train_windows.shape)
    #print("Validation windows tensor shape:", validation_windows.shape)
    #print("Test windows tensor shape:", test_windows.shape)

    best_vae, best_lstm = load_or_train_models(train_windows, validation_windows, config, device)

    anomaly_indices, anomaly_scores = calculate_anomaly_scores(best_vae, best_lstm, test_windows, threshold=0.05)
    threshold_percentile, threshold_mean_std = determine_threshold(
        anomaly_scores, percentile=config["threshold"]["percentile"], k=config["threshold"]["k"]
    )

    grouped_anomalies = group_anomalies(anomaly_indices)
    explanations = explain_anomalies_for_groups(
        best_vae, best_lstm, test_windows, grouped_anomalies, threshold_percentile, df.columns.tolist()
    )

    print(explanations)

if __name__ == "__main__":
    main()