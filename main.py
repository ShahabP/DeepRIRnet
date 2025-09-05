import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import RIRDataset
from models.deep_rir_net import DeepRIRNet
from utils.rir_generator import generate_dataset
from utils.losses import hybrid_loss
from train import train_model


def freeze_first_lstm(model, n=1):
    for lstm in model.lstm_layers[:n]:
        for param in lstm.parameters():
            param.requires_grad = False


def main():
    # -------------------------------
    # Setup
    # -------------------------------
    T = 512
    input_dim = 3 + 1 + 3 + 3  # room(3) + absorption(1) + source(3) + mic(3)

    # -------------------------------
    # Dataset creation
    # -------------------------------
    print("Generating datasets...")
    source_data = generate_dataset(150, T=T, room_type="rectangular")
    target_data = generate_dataset(20, T=T, room_type="l_shaped")

    source_loader = DataLoader(RIRDataset(source_data), batch_size=8, shuffle=True)
    target_loader = DataLoader(RIRDataset(target_data), batch_size=8, shuffle=True)

    # -------------------------------
    # Model & optimizer
    # -------------------------------
    model = DeepRIRNet(input_dim=input_dim, T=T)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------------
    # Pretraining
    # -------------------------------
    print("Training on source domain...")
    source_loss = train_model(model, source_loader, optimizer, hybrid_loss, epochs=50)

    plt.figure()
    plt.plot(source_loss)
    plt.title("Pretraining Loss on Source Domain")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    # -------------------------------
    # Fine-tuning
    # -------------------------------
    print("Fine-tuning on target domain...")
    freeze_first_lstm(model, n=1)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    target_loss = train_model(model, target_loader, optimizer, hybrid_loss, epochs=30)

    plt.figure()
    plt.plot(target_loss)
    plt.title("Fine-tuning Loss on Target Domain")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    # -------------------------------
    # Prediction example
    # -------------------------------
    x_example, h_example = target_data[0]
    hat_h = model(torch.tensor(x_example, dtype=torch.float32).unsqueeze(0)).detach().numpy().flatten()

    plt.figure()
    plt.plot(h_example, label="Ground Truth")
    plt.plot(hat_h, label="Predicted")
    plt.title("Predicted vs Ground Truth RIR")
    plt.xlabel("Time samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # -------------------------------
    # LSD vs Absorption
    # -------------------------------
    abs_coeffs = np.linspace(0.2, 0.8, 10)
    lsd_vals = []

    for ac in abs_coeffs:
        x_input = torch.tensor(
            np.concatenate([
                np.random.rand(3) * 10,   # room dimensions
                np.array([ac]),           # absorption coefficient
                np.random.rand(3) * 5,    # source
                np.random.rand(3) * 5     # mic
            ]),
            dtype=torch.float32
        ).unsqueeze(0)

        hat_h = model(x_input).detach()
        H = torch.log(torch.abs(torch.fft.fft(torch.tensor(h_example, dtype=torch.float32).unsqueeze(0))) + 1e-8)
        H_hat = torch.log(torch.abs(torch.fft.fft(hat_h)) + 1e-8)
        lsd_vals.append(torch.mean((H - H_hat) ** 2).item())

    plt.figure()
    plt.plot(abs_coeffs, lsd_vals, marker="o")
    plt.title("Log-spectral Distance vs Absorption Coefficient")
    plt.xlabel("Absorption Coefficient")
    plt.ylabel("LSD")
    plt.show()


if __name__ == "__main__":
    main()
