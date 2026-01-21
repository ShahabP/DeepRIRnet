"""Evaluation script for GAN baseline model.

This script evaluates the GAN-generated RIRs using the same metrics as the paper
(MSE, LSD, ATE) for fair comparison with DeepRIRNet.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Squared Error.
    
    Args:
        pred: Predicted RIR [batch, T]
        target: Ground truth RIR [batch, T]
        
    Returns:
        MSE value
    """
    return torch.mean((pred - target) ** 2).item()


def compute_lsd(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute Log-Spectral Distance.
    
    Args:
        pred: Predicted RIR [batch, T]
        target: Ground truth RIR [batch, T]
        eps: Small value to avoid log(0)
        
    Returns:
        LSD in dB
    """
    # Compute FFT
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    
    # Power spectrum
    pred_power = torch.abs(pred_fft) ** 2 + eps
    target_power = torch.abs(target_fft) ** 2 + eps
    
    # Log-spectral distance
    lsd = torch.sqrt(torch.mean((10 * torch.log10(pred_power) - 10 * torch.log10(target_power)) ** 2))
    
    return lsd.item()


def compute_ate(pred: torch.Tensor, target: torch.Tensor, threshold: float = -60.0) -> float:
    """
    Compute Arrival Time Error (ATE) in samples.
    
    Args:
        pred: Predicted RIR [batch, T]
        target: Ground truth RIR [batch, T]
        threshold: Threshold in dB for detecting direct path
        
    Returns:
        Mean ATE in samples
    """
    batch_size = pred.shape[0]
    ates = []
    
    for i in range(batch_size):
        pred_i = pred[i].cpu().numpy()
        target_i = target[i].cpu().numpy()
        
        # Convert to dB
        pred_db = 20 * np.log10(np.abs(pred_i) + 1e-10)
        target_db = 20 * np.log10(np.abs(target_i) + 1e-10)
        
        # Find first peak above threshold
        pred_arrival = np.argmax(pred_db > threshold)
        target_arrival = np.argmax(target_db > threshold)
        
        # Compute error
        ate = np.abs(pred_arrival - target_arrival)
        ates.append(ate)
    
    return np.mean(ates)


def evaluate_gan(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate GAN model on test set.
    
    Args:
        model: Trained GAN model
        dataloader: Test data loader
        device: Device to run evaluation on
        verbose: Whether to print progress
        
    Returns:
        Dictionary of metric values
    """
    model.to(device)
    model.eval()
    
    all_mse = []
    all_lsd = []
    all_ate = []
    
    with torch.no_grad():
        for x, h_target in dataloader:
            x = x.to(device)
            h_target = h_target.to(device)
            
            # Generate RIR
            h_pred = model.generate(x)
            
            # Compute metrics
            mse = compute_mse(h_pred, h_target)
            lsd = compute_lsd(h_pred, h_target)
            ate = compute_ate(h_pred, h_target)
            
            all_mse.append(mse)
            all_lsd.append(lsd)
            all_ate.append(ate)
    
    # Average metrics
    results = {
        "MSE": np.mean(all_mse),
        "MSE_std": np.std(all_mse),
        "LSD": np.mean(all_lsd),
        "LSD_std": np.std(all_lsd),
        "ATE": np.mean(all_ate),
        "ATE_std": np.std(all_ate)
    }
    
    if verbose:
        print("\n" + "=" * 50)
        print("GAN Baseline Evaluation Results")
        print("=" * 50)
        print(f"MSE:  {results['MSE']:.6f} ± {results['MSE_std']:.6f}")
        print(f"LSD:  {results['LSD']:.4f} ± {results['LSD_std']:.4f} dB")
        print(f"ATE:  {results['ATE']:.2f} ± {results['ATE_std']:.2f} samples")
        print("=" * 50)
    
    return results


def evaluate_and_compare(
    gan_model: nn.Module,
    deeprir_model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate both GAN and DeepRIRNet models for comparison.
    
    Args:
        gan_model: Trained GAN model
        deeprir_model: Trained DeepRIRNet model
        dataloader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary with results for both models
    """
    print("\nEvaluating GAN Baseline...")
    gan_results = evaluate_gan(gan_model, dataloader, device, verbose=True)
    
    print("\nEvaluating DeepRIRNet...")
    deeprir_results = evaluate_deeprirnet(deeprir_model, dataloader, device, verbose=True)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print(f"{'Metric':<15} {'GAN Baseline':<25} {'DeepRIRNet':<25}")
    print("-" * 70)
    print(f"{'MSE':<15} {gan_results['MSE']:.6f} ± {gan_results['MSE_std']:.6f}    "
          f"{deeprir_results['MSE']:.6f} ± {deeprir_results['MSE_std']:.6f}")
    print(f"{'LSD (dB)':<15} {gan_results['LSD']:.4f} ± {gan_results['LSD_std']:.4f}        "
          f"{deeprir_results['LSD']:.4f} ± {deeprir_results['LSD_std']:.4f}")
    print(f"{'ATE (samples)':<15} {gan_results['ATE']:.2f} ± {gan_results['ATE_std']:.2f}          "
          f"{deeprir_results['ATE']:.2f} ± {deeprir_results['ATE_std']:.2f}")
    print("=" * 70)
    
    return {
        "GAN": gan_results,
        "DeepRIRNet": deeprir_results
    }


def evaluate_deeprirnet(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate DeepRIRNet model.
    
    Args:
        model: Trained DeepRIRNet model
        dataloader: Test data loader
        device: Device to run evaluation on
        verbose: Whether to print progress
        
    Returns:
        Dictionary of metric values
    """
    model.to(device)
    model.eval()
    
    all_mse = []
    all_lsd = []
    all_ate = []
    
    with torch.no_grad():
        for x, h_target in dataloader:
            x = x.to(device)
            h_target = h_target.to(device)
            
            # Forward pass
            h_pred = model(x)
            
            # Compute metrics
            mse = compute_mse(h_pred, h_target)
            lsd = compute_lsd(h_pred, h_target)
            ate = compute_ate(h_pred, h_target)
            
            all_mse.append(mse)
            all_lsd.append(lsd)
            all_ate.append(ate)
    
    # Average metrics
    results = {
        "MSE": np.mean(all_mse),
        "MSE_std": np.std(all_mse),
        "LSD": np.mean(all_lsd),
        "LSD_std": np.std(all_lsd),
        "ATE": np.mean(all_ate),
        "ATE_std": np.std(all_ate)
    }
    
    if verbose:
        print("\n" + "=" * 50)
        print("DeepRIRNet Evaluation Results")
        print("=" * 50)
        print(f"MSE:  {results['MSE']:.6f} ± {results['MSE_std']:.6f}")
        print(f"LSD:  {results['LSD']:.4f} ± {results['LSD_std']:.4f} dB")
        print(f"ATE:  {results['ATE']:.2f} ± {results['ATE_std']:.2f} samples")
        print("=" * 50)
    
    return results
