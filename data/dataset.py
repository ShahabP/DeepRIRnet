"""Dataset implementation for Room Impulse Response data."""

from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset
import numpy as np


class RIRDataset(Dataset):
    """
    Dataset class for Room Impulse Response (RIR) data.
    
    This dataset handles pairs of geometry features and corresponding RIR sequences.
    The geometry features include room dimensions, absorption coefficients, and 
    source/microphone positions.
    
    Args:
        data: List of (geometry_features, rir_sequence) tuples
    """
    
    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray]]):
        self.data = data
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (geometry_features, rir_sequence) as tensors
        """
        x, h = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(h, dtype=torch.float32)
    
    @property
    def input_dim(self) -> int:
        """Get the dimension of input geometry features."""
        if len(self.data) > 0:
            return len(self.data[0][0])
        return 0
    
    @property
    def sequence_length(self) -> int:
        """Get the length of RIR sequences."""
        if len(self.data) > 0:
            return len(self.data[0][1])
        return 0
