import torch
import numpy as np
from utils.anomaly_detection import calculate_anomaly_scores
from tqdm import tqdm

def train_and_evaluate(vae, lstm, train_windows, validation_windows, num_epochs, lr, device):
    def reconstruction_loss_fn(x_recon, x):
        return torch.nn.functional.mse_loss(x_recon, x)

    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=lr)
    optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=lr)

    vae_losses = []
    lstm_losses = []

    for epoch in range(num_epochs):
        vae.train()
        lstm.train()
        total_vae_loss, total_lstm_loss = 0, 0

        with tqdm(total=len(train_windows) - 1, desc=f"Epoch [{epoch+1}/{num_epochs}]") as pbar:
            for i in range(len(train_windows) - 1):
                window = train_windows[i]
                next_window = train_windows[i + 1]

                # VAE loss
                x_recon, mu, logvar = vae(window)
                recon_loss = reconstruction_loss_fn(x_recon, window)
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                vae_loss = recon_loss + kld_loss

                optimizer_vae.zero_grad()
                vae_loss.backward()
                optimizer_vae.step()

                # LSTM loss
                with torch.no_grad():
                    embedding, _ = vae.encode(window)

                lstm_output = lstm(embedding)
                next_window_embedding, _ = vae.encode(next_window)
                lstm_loss = reconstruction_loss_fn(lstm_output, next_window_embedding)

                optimizer_lstm.zero_grad()
                lstm_loss.backward()
                optimizer_lstm.step()

                total_vae_loss += vae_loss.item()
                total_lstm_loss += lstm_loss.item()

                pbar.set_postfix({"VAE Loss": vae_loss.item(), "LSTM Loss": lstm_loss.item()})
                pbar.update(1)

                del x_recon, mu, logvar, embedding, lstm_output, next_window_embedding
                torch.cuda.empty_cache()

        vae_losses.append(total_vae_loss / len(train_windows))
        lstm_losses.append(total_lstm_loss / (len(train_windows) - 1))

        torch.cuda.empty_cache()

    anomalies_indices, anomaly_scores = calculate_anomaly_scores(vae, lstm, validation_windows, 0.05)
    mean_anomaly_score = np.mean(anomaly_scores)

    torch.cuda.empty_cache()

    return mean_anomaly_score, vae_losses, lstm_losses