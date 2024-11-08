import numpy as np

def determine_threshold(anomaly_scores, percentile=95, k=3):
    threshold_percentile = np.percentile(anomaly_scores, percentile)
    mean_score = np.mean(anomaly_scores)
    std_score = np.std(anomaly_scores)
    threshold_mean_std = mean_score + k * std_score
    return threshold_percentile, threshold_mean_std