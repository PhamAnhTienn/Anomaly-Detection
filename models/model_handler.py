import os
import torch
import itertools
from models.vae import VAE
from models.lstm import LSTM
from models.train import train_and_evaluate
import yaml

def load_or_train_models(train_windows, validation_windows, config, device):
    vae_checkpoint_path = "model_checkpoints/best_vae.pth"
    lstm_checkpoint_path = "model_checkpoints/best_lstm.pth"

    if os.path.exists(vae_checkpoint_path) and os.path.exists(lstm_checkpoint_path):
        latent_dim = config["best_params"]["latent_dim"]
        hidden_dim = config["best_params"]["hidden_dim"]
        lr = config["best_params"]["lr"]

        best_vae = VAE(input_dim=train_windows.shape[2], hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
        best_lstm = LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=config["lstm"]["num_layers"], output_size=latent_dim).to(device)

        best_vae.load_state_dict(torch.load(vae_checkpoint_path))
        best_lstm.load_state_dict(torch.load(lstm_checkpoint_path))

        print("Loaded trained models from checkpoint.")
    else:
        latent_dims = config["grid_search"]["latent_dims"]
        hidden_dims = config["grid_search"]["hidden_dims"]
        learning_rates = config["grid_search"]["learning_rates"]
        num_epochs = config["train"]["num_epochs"]

        best_score = float('inf')
        best_params = {}
        best_vae = None
        best_lstm = None

        for latent_dim, hidden_dim, lr in itertools.product(latent_dims, hidden_dims, learning_rates):
            print(f"Testing configuration: Latent Dim={latent_dim}, Hidden Dim={hidden_dim}, LR={lr}")

            vae = VAE(input_dim=train_windows.shape[2], hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
            lstm = LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=config["lstm"]["num_layers"], output_size=latent_dim).to(device)

            mean_anomaly_score, vae_losses, lstm_losses = train_and_evaluate(
                vae, lstm, train_windows, validation_windows, num_epochs, lr, device
            )

            if mean_anomaly_score < best_score:
                best_score = mean_anomaly_score
                best_params = {'latent_dim': latent_dim, 'hidden_dim': hidden_dim, 'lr': lr}
                best_vae = vae
                best_lstm = lstm

        os.makedirs("model_checkpoints", exist_ok=True)
        torch.save(best_vae.state_dict(), vae_checkpoint_path)
        torch.save(best_lstm.state_dict(), lstm_checkpoint_path)
        print("Trained models and saved checkpoints.")

        config["best_params"] = best_params
        with open("params.yaml", "w") as file:
            yaml.safe_dump(config, file)

    best_vae.eval()
    best_lstm.eval()

    return best_vae, best_lstm