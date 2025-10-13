"""
DeepRIRNet: A geometry-aware Room Impulse Response prediction model.

This package implements DeepRIRNet, a deep learning model for Room Impulse Response (RIR) 
estimation using transfer learning with geometry-aware features and physics-informed losses.
"""

__version__ = "1.0.0"
__author__ = "DeepRIRNet Team"

from .config import Config, get_default_config
from .models.deep_rir_net import DeepRIRNet
from .data.dataset import RIRDataset
from .utils.rir_generator import generate_dataset
from .train import train_model

__all__ = [
    "Config",
    "get_default_config", 
    "DeepRIRNet",
    "RIRDataset",
    "generate_dataset",
    "train_model",
]
