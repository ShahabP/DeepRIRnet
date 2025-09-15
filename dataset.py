from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import pyroomacoustics as pra

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

def make_ism_dataset(
    n=100,
    D_GEOM=30,
    T=4096,
    fs=16000,
    room_dim=(6.0, 5.0, 3.0),
    max_order=10,
):
    """
    Generate dataset using the Image Source Method (ISM).

    Args:
        n: number of samples
        D_GEOM: dimension of geometry descriptor
        T: length of RIR
        fs: sample rate
        room_dim: room size (x,y,z)
        max_order: maximum reflection order for ISM
    """
    items = []
    for _ in range(n):
        # random geometry descriptor (could encode room dims, absorption, etc.)
        g = np.random.randn(D_GEOM).astype(np.float32) * 0.1

        # random source and mic positions inside the room
        s = np.random.rand(3).astype(np.float32) * np.array(room_dim)
        m = np.random.rand(3).astype(np.float32) * np.array(room_dim)

        # create shoebox room with random absorption
        absorption = np.random.uniform(0.2, 0.8)
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.Material(absorption),
            max_order=max_order,
        )

        room.add_source(s)
        room.add_microphone(m)

        # compute RIR
        room.compute_rir()
        h = np.array(room.rir[0][0], dtype=np.float32)

        # pad or truncate to fixed length T
        if len(h) < T:
            h = np.pad(h, (0, T - len(h)), mode="constant")
        else:
            h = h[:T]

        items.append((g, s, m, h))

    return items
