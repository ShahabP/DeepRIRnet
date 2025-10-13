"""Training utilities for DeepRIRNet."""

from typing import List, Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    loss_fn: Callable, 
    epochs: int = 50, 
    device: str = "cpu",
    freeze_first_n_lstm: int = 0,
    verbose: bool = True
) -> List[float]:
    """
    Train the DeepRIRNet model.
    
    Args:
        model: The neural network model to train
        dataloader: DataLoader providing training data
        optimizer: Optimizer for parameter updates
        loss_fn: Loss function to minimize
        epochs: Number of training epochs
        device: Device to run training on ("cpu" or "cuda")
        freeze_first_n_lstm: Number of initial LSTM layers to freeze
        verbose: Whether to print training progress
        
    Returns:
        List of average losses for each epoch
    """
    model.train()
    model.to(device)
    loss_history = []
    
    # Freeze layers if requested
    if freeze_first_n_lstm > 0:
        model.freeze_layers(freeze_first_n_lstm)
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_samples = 0
        
        # Use tqdm for progress bar if verbose
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else dataloader
        
        for batch_idx, (x, h) in enumerate(iterator):
            # Move data to device
            x = x.to(device)
            h = h.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(x)
            
            # Compute loss
            loss = loss_fn(predictions, h)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Accumulate loss
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            
            # Update progress bar
            if verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({'loss': loss.item()})
        
        # Calculate average loss for epoch
        avg_loss = total_loss / num_samples
        loss_history.append(avg_loss)
        
        # Print epoch summary
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
    
    return loss_history


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    device: str = "cpu"
) -> float:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The neural network model to evaluate
        dataloader: DataLoader providing evaluation data
        loss_fn: Loss function to compute
        device: Device to run evaluation on
        
    Returns:
        Average loss over the evaluation dataset
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for x, h in dataloader:
            x = x.to(device)
            h = h.to(device)
            
            predictions = model(x)
            loss = loss_fn(predictions, h)
            
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
    
    return total_loss / num_samples
