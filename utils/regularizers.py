import torch
import torch.nn.functional as F

def sparse_reg(hat_h):
    return torch.mean(torch.abs(hat_h))

def decay_reg(hat_h, rho=0.3, fs=16000):
    device = hat_h.device
    batch_size = hat_h.size(0)
    decay_loss = 0.0
    for t in range(1, hat_h.size(1)):
        factor = torch.exp(torch.tensor(-rho * t / fs, device=device))
        term = F.relu(torch.abs(hat_h[:, t]) - factor * torch.abs(hat_h[:, t-1]))**2
        decay_loss += term.sum()
    return decay_loss / (hat_h.size(1) * batch_size)
