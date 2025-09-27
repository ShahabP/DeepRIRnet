import torch
import torch.nn.functional as F
from .regularizers import sparse_reg, decay_reg

def hybrid_loss(hat_h, h, alpha=1.0, beta=0.1, lambda_sparse=1e-4, lambda_decay=1e-4,
                rho=0.3, fs=16000, eps=1e-8):
    """
    Hybrid loss combining MSE in time, log-spectral distance, and regularizers.
    """
    # Time-domain MSE
    mse = F.mse_loss(hat_h, h)

    # Spectral log-magnitude distance
    # Compute real FFT (optionally)
    # Here using full FFT; you could switch to rfft if desired
    H = torch.fft.fft(h, dim=1)
    H_hat = torch.fft.fft(hat_h, dim=1)

    mag = torch.abs(H)
    mag_hat = torch.abs(H_hat)

    log_mag = torch.log(mag + eps)
    log_mag_hat = torch.log(mag_hat + eps)

    lsd = torch.mean((log_mag - log_mag_hat) ** 2)

    # Regularizers
    sp = sparse_reg(hat_h)
    dec = decay_reg(hat_h, rho=rho, fs=fs)

    # Weighted sum
    loss = alpha * mse + beta * lsd + lambda_sparse * sp + lambda_decay * dec
    return loss
