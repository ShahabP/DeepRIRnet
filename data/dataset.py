import torch
from torch.utils.data import Dataset

class RIRDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, h = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(h, dtype=torch.float32)
