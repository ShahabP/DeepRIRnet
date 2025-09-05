import torch
import torch.nn.functional as F
from .regularizers import sparse_reg, decay_reg

def hybrid_loss(hat_h, h, alpha=1.0, beta=0.1, rho=0.3, fs=16000):
    mse = F.mse_loss(hat_h, h)
    H = torch.log(torch.abs(torch.fft.fft(h, dim=1)) + 1e-8)
    H_hat = torch.log(torch.abs(torch.fft.fft(hat_h, dim=1)) + 1e-8)
    lsd = torch.mean((H - H_hat)**2)
    return alpha * mse + beta * lsd + 1e-4*sparse_reg(hat_h) + 1e-4*decay_reg(hat_h, rho, fs)
