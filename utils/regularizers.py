import torch
import torch.nn.functional as F

def sparse_reg(hat_h):
    # Encourage activations to be sparse (L1 on activations)
    return torch.mean(torch.abs(hat_h))


def decay_reg(hat_h, rho=0.3, fs=16000):
    """
    Penalty encouraging the activation magnitude to decay over time:
      penalize [abs(h_t) - decay_factor * abs(h_{t-1})]_+^2
    """
    B, T = hat_h.shape[:2]  # supports possibly extra dims (if e.g. channels), but focusing on time dimension
    device = hat_h.device
    dtype = hat_h.dtype

    # Compute absolute
    H = torch.abs(hat_h)  # (B, T, ...)
    # We assume hat_h is (B, T); if more dims, adapt accordingly (sum over extras)

    # Precompute decay factors for times t = 1..T-1
    t_idx = torch.arange(1, T, device=device, dtype=dtype)  # (T-1,)
    decay_factors = torch.exp(- rho * t_idx / fs)  # shape (T-1,)

    # Consider only t=1..T-1
    H_t = H[:, 1:]      # (B, T-1)
    H_prev = H[:, :-1]  # (B, T-1)

    # Broadcast decay_factors across batch
    # term = relu(H_t - decay_factors * H_prev)^2
    term = F.relu(H_t - decay_factors * H_prev) ** 2  # shape (B, T-1)

    # Sum all penalty terms
    decay_loss = term.sum()  # scalar

    # Normalize: divide by (B * (T-1)) or (B * T), choice
    return decay_loss / (B * (T - 1))
