from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np

class RIRDataset(Dataset):
    def __init__(self, items: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        g, s_pos, m_pos, h = self.items[idx]
        z = np.concatenate([g.ravel(), s_pos.ravel(), m_pos.ravel()]).astype(np.float32)
        h = h.astype(np.float32)
        return torch.from_numpy(z), torch.from_numpy(h)

def make_dummy_dataset(n=100, D_GEOM=30, T=4096):
    items = []
    for _ in range(n):
        g = np.random.randn(D_GEOM).astype(np.float32) * 0.1
        s = (np.random.rand(3).astype(np.float32) - 0.5) * 5.0
        m = (np.random.rand(3).astype(np.float32) - 0.5) * 5.0
        h = np.zeros(T, dtype=np.float32)
        num_peaks = np.random.randint(1, 6)
        for _p in range(num_peaks):
            t0 = np.random.randint(0, 200)
            amp = np.random.rand() * 0.8
            h[t0] += amp
        t_axis = np.arange(T)
        tail = np.exp(-0.002 * t_axis) * 0.05 * np.random.rand()
        h += tail.astype(np.float32)
        h += np.random.randn(T).astype(np.float32) * 1e-5
        items.append((g, s, m, h))
    return items
