import torch

def calculate_anomaly_scores(vae, lstm, data_windows, threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    anomalies_indices = []
    anomalies_scores = []

    vae.eval()
    lstm.eval()

    with torch.no_grad():
        for i, window in enumerate(data_windows):
            window = window.to(device)
            
            embedding, _ = vae.encode(window[:-1])
            lstm_output = lstm(embedding)
            next_window = vae.decode(lstm_output)
            
            anomaly_score = 0
            for j in range(len(next_window)):
                diff = next_window[j] - window[j + 1]
                l2_norm = torch.sqrt(torch.sum(diff ** 2))
                anomaly_score += l2_norm
                
            anomaly_score = anomaly_score.item()
            
            if anomaly_score > threshold:
                anomalies_indices.append(i)
                anomalies_scores.append(anomaly_score)

            del window, embedding, lstm_output, next_window
            torch.cuda.empty_cache()

    return anomalies_indices, anomalies_scores


def calculate_feature_anomaly_scores(vae, lstm, window, threshold, feature_names):
    embedding, _ = vae.encode(window[:-1])  
    lstm_output = lstm(embedding)
    next_window = vae.decode(lstm_output)

    scores = torch.mean(abs(next_window - window[1:]), dim=0) 

    feature_scores = {feature_names[i]: scores[i].item() for i in range(len(feature_names))}

    return feature_scores


def group_anomalies(anomalies, gap=1):
    grouped_anomalies = []
    current_group = [anomalies[0]]
    
    for i in range(1, len(anomalies)):
        if anomalies[i] <= current_group[-1] + gap:  
            current_group.append(anomalies[i])
        else:
            grouped_anomalies.append(current_group)
            current_group = [anomalies[i]]
    
    grouped_anomalies.append(current_group)
    
    return grouped_anomalies
