"""General utility functions for DeepRIRNet."""

import random
import numpy as np
import torch
import os
from typing import Dict, Any


# Global device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior for CUDA operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with total, trainable, and non-trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    metadata: Dict[str, Any] = None
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        filepath: Path to save checkpoint
        metadata: Additional metadata to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    
    if metadata:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        model: PyTorch model to load state into
        optimizer: PyTorch optimizer to load state into
        filepath: Path to checkpoint file
        device: Device to load tensors to
        
    Returns:
        Dictionary containing checkpoint metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        **{k: v for k, v in checkpoint.items() 
           if k not in ['model_state_dict', 'optimizer_state_dict']}
    }
