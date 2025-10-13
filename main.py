"""Main training script for DeepRIRNet with transfer learning."""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import argparse
import os
from typing import List

from config import get_default_config, get_quick_config, Config
from data.dataset import RIRDataset
from models.deep_rir_net import DeepRIRNet
from utils.rir_generator import generate_dataset
from utils.losses import hybrid_loss
from train import train_model, evaluate_model
from utils import set_seed, DEVICE


def plot_training_curves(
    source_losses: List[float], 
    target_losses: List[float],
    save_path: str = None
) -> None:
    """Plot training loss curves for source and target domains."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Source domain training
    ax1.plot(source_losses)
    ax1.set_title("Pretraining Loss on Source Domain")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    
    # Target domain fine-tuning
    ax2.plot(target_losses)
    ax2.set_title("Fine-tuning Loss on Target Domain")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_prediction_comparison(
    model: DeepRIRNet,
    test_sample: tuple,
    config: Config,
    save_path: str = None
) -> None:
    """Plot comparison between predicted and ground truth RIR."""
    x_example, h_example = test_sample
    
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_example, dtype=torch.float32).unsqueeze(0).to(config.device)
        h_pred = model(x_tensor).cpu().numpy().flatten()
    
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(len(h_example)) / config.data.fs * 1000  # Convert to milliseconds
    
    plt.plot(time_axis, h_example, label="Ground Truth", alpha=0.8)
    plt.plot(time_axis, h_pred, label="Predicted", alpha=0.8)
    plt.title("Predicted vs Ground Truth RIR")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Prediction comparison saved to {save_path}")
    
    plt.show()


def analyze_absorption_sensitivity(
    model: DeepRIRNet,
    reference_rir: np.ndarray,
    config: Config,
    save_path: str = None
) -> None:
    """Analyze model sensitivity to absorption coefficient changes."""
    model.eval()
    
    abs_coeffs = np.linspace(
        config.data.min_absorption, 
        config.data.max_absorption, 
        20
    )
    lsd_values = []
    
    # Reference spectrum
    H_ref = torch.log(torch.abs(torch.fft.fft(
        torch.tensor(reference_rir, dtype=torch.float32)
    )) + 1e-8)
    
    for abs_coeff in abs_coeffs:
        # Create input with varying absorption coefficient
        x_input = torch.tensor(
            np.concatenate([
                np.random.uniform(4, 12, 3),  # room dimensions
                np.array([abs_coeff]),        # absorption coefficient
                np.random.uniform(1, 8, 3),   # source position
                np.random.uniform(1, 8, 3)    # microphone position
            ]),
            dtype=torch.float32
        ).unsqueeze(0).to(config.device)
        
        with torch.no_grad():
            h_pred = model(x_input).cpu()
            H_pred = torch.log(torch.abs(torch.fft.fft(h_pred)) + 1e-8)
            lsd = torch.mean((H_ref - H_pred) ** 2).item()
            lsd_values.append(lsd)
    
    plt.figure(figsize=(10, 6))
    plt.plot(abs_coeffs, lsd_values, marker="o")
    plt.title("Model Sensitivity to Absorption Coefficient")
    plt.xlabel("Absorption Coefficient")
    plt.ylabel("Log-Spectral Distance (LSD)")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Absorption sensitivity analysis saved to {save_path}")
    
    plt.show()


def main(config: Config = None) -> None:
    """Main training pipeline with transfer learning."""
    
    if config is None:
        config = get_default_config()
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    print(f"Using device: {config.device}")
    print(f"Configuration: {config}")
    
    # -------------------------------
    # Dataset Generation
    # -------------------------------
    print("Generating source domain dataset...")
    source_data = generate_dataset(
        num_rooms=config.data.source_dataset_size,
        T=config.data.T,
        fs=config.data.fs,
        room_type=config.data.source_room_type,
        min_room_size=config.data.min_room_size,
        max_room_size=config.data.max_room_size,
        min_absorption=config.data.min_absorption,
        max_absorption=config.data.max_absorption,
        seed=config.seed
    )
    
    print("Generating target domain dataset...")
    # For now, generate target as rectangular too (since l_shaped not implemented)
    target_data = generate_dataset(
        num_rooms=config.data.target_dataset_size,
        T=config.data.T,
        fs=config.data.fs,
        room_type="rectangular",  # Change when l_shaped is implemented
        seed=config.seed + 1000
    )
    
    # Create data loaders
    source_dataset = RIRDataset(source_data)
    target_dataset = RIRDataset(target_data)
    
    source_loader = DataLoader(
        source_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True
    )
    target_loader = DataLoader(
        target_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True
    )
    
    print(f"Source dataset: {len(source_dataset)} samples")
    print(f"Target dataset: {len(target_dataset)} samples")
    print(f"Input dimension: {source_dataset.input_dim}")
    
    # -------------------------------
    # Model Setup
    # -------------------------------
    model = DeepRIRNet(
        input_dim=source_dataset.input_dim,
        T=config.model.T,
        hidden_dim=config.model.hidden_dim,
        num_lstm_layers=config.model.num_lstm_layers,
        dropout=config.model.dropout
    )
    
    print(f"Model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # -------------------------------
    # Source Domain Pretraining
    # -------------------------------
    print("\n" + "="*50)
    print("PHASE 1: Source Domain Pretraining")
    print("="*50)
    
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    source_losses = train_model(
        model=model,
        dataloader=source_loader,
        optimizer=optimizer,
        loss_fn=hybrid_loss,
        epochs=config.training.epochs,
        device=config.device,
        verbose=True
    )
    
    # -------------------------------
    # Target Domain Fine-tuning
    # -------------------------------
    print("\n" + "="*50)
    print("PHASE 2: Target Domain Fine-tuning")
    print("="*50)
    
    # Freeze first LSTM layers for transfer learning
    model.freeze_layers(config.training.freeze_first_n_lstm)
    
    # Create optimizer for unfrozen parameters only
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.learning_rate
    )
    
    target_losses = train_model(
        model=model,
        dataloader=target_loader,
        optimizer=optimizer,
        loss_fn=hybrid_loss,
        epochs=config.training.fine_tune_epochs,
        device=config.device,
        verbose=True
    )
    
    # -------------------------------
    # Evaluation and Visualization
    # -------------------------------
    print("\n" + "="*50)
    print("EVALUATION AND ANALYSIS")
    print("="*50)
    
    # Final evaluation
    final_source_loss = evaluate_model(model, source_loader, hybrid_loss, config.device)
    final_target_loss = evaluate_model(model, target_loader, hybrid_loss, config.device)
    
    print(f"Final source domain loss: {final_source_loss:.6f}")
    print(f"Final target domain loss: {final_target_loss:.6f}")
    
    # Create visualizations
    plot_training_curves(source_losses, target_losses)
    plot_prediction_comparison(model, target_data[0], config)
    analyze_absorption_sensitivity(model, target_data[0][1], config)
    
    # Save model
    model_path = "deeprirnet_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'source_losses': source_losses,
        'target_losses': target_losses,
        'final_source_loss': final_source_loss,
        'final_target_loss': final_target_loss
    }, model_path)
    
    print(f"Model and training results saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🎵 DeepRIRNet: Geometry-Aware Room Impulse Response Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📖 QUICK START EXAMPLES:

  # Basic training with default settings (recommended)
  python main.py

  # Fast training for development/testing (~5 min)
  python main.py --preset fast

  # High quality results (~2 hours)  
  python main.py --preset high_quality

  # Custom parameters
  python main.py --epochs 100 --batch-size 16 --lr 0.001

  # GPU training with larger model
  python main.py --device cuda --hidden-dim 1024 --layers 8

  # Small memory footprint
  python main.py --batch-size 4 --hidden-dim 256 --sequence-length 256

📚 For detailed hyperparameter guidance, see: HYPERPARAMETER_GUIDE.md
        """)
    
    # Configuration presets
    parser.add_argument("--preset", type=str, choices=["fast", "balanced", "high_quality", "large_scale"],
                       default="balanced",
                       help="🚀 Configuration preset (fast=5min, balanced=20min, high_quality=2hr)")
    
    # Model architecture  
    model_group = parser.add_argument_group("🏗️  Model Architecture")
    model_group.add_argument("--hidden-dim", type=int, metavar="N",
                           help="Hidden dimension (128=fast, 512=default, 1024=quality)")
    model_group.add_argument("--layers", type=int, metavar="N",
                           help="Number of LSTM layers (3-4=simple, 6=default, 8+=complex)")
    model_group.add_argument("--dropout", type=float, metavar="RATE",
                           help="Dropout rate (0.1=minimal, 0.2=default, 0.4=strong regularization)")
    model_group.add_argument("--sequence-length", type=int, metavar="SAMPLES",
                           help="RIR length in samples (256=fast, 512=default, 1024=detailed)")
    
    # Training parameters
    train_group = parser.add_argument_group("🎯 Training Parameters")  
    train_group.add_argument("--epochs", type=int, metavar="N",
                           help="Source domain epochs (20=fast, 50=default, 100=thorough)")
    train_group.add_argument("--fine-tune-epochs", type=int, metavar="N", 
                           help="Target domain epochs (10-50, default=30)")
    train_group.add_argument("--batch-size", type=int, metavar="N",
                           help="Batch size - adjust for GPU memory (4-32, default=8)")
    train_group.add_argument("--lr", type=float, metavar="RATE",
                           help="Learning rate (1e-4=safe, 1e-3=default, 2e-3=aggressive)")
    train_group.add_argument("--freeze-layers", type=int, metavar="N",
                           help="Number of LSTM layers to freeze during fine-tuning (0-3)")
    
    # Loss function weights
    loss_group = parser.add_argument_group("⚖️  Loss Function Weights")
    loss_group.add_argument("--alpha", type=float, metavar="WEIGHT",
                          help="MSE weight for time-domain accuracy (0.5-2.0)")
    loss_group.add_argument("--beta", type=float, metavar="WEIGHT", 
                          help="LSD weight for spectral quality (0.05-0.5)")
    loss_group.add_argument("--lambda-sparse", type=float, metavar="WEIGHT",
                          help="Sparsity regularization (1e-5 to 1e-3)")
    loss_group.add_argument("--lambda-decay", type=float, metavar="WEIGHT",
                          help="Energy decay regularization (1e-5 to 1e-3)")
    loss_group.add_argument("--decay-rate", type=float, metavar="RHO",
                          help="Energy decay rate (0.1=reverberant, 0.3=default, 0.7=absorptive)")
    
    # Dataset parameters
    data_group = parser.add_argument_group("🏠 Dataset Configuration")
    data_group.add_argument("--source-size", type=int, metavar="N",
                          help="Source domain dataset size (50-1000+)")
    data_group.add_argument("--target-size", type=int, metavar="N", 
                          help="Target domain dataset size (5-100)")
    data_group.add_argument("--min-room-size", type=float, metavar="METERS",
                          help="Minimum room dimension in meters")
    data_group.add_argument("--max-room-size", type=float, metavar="METERS",
                          help="Maximum room dimension in meters")
    data_group.add_argument("--sampling-rate", type=int, metavar="HZ",
                          help="Audio sampling frequency (8000, 16000, 44100)")
    
    # System configuration
    system_group = parser.add_argument_group("⚙️  System Configuration")
    system_group.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], 
                            default="auto",
                            help="Computation device (auto=detect GPU)")
    system_group.add_argument("--seed", type=int, metavar="N", default=42,
                            help="Random seed for reproducibility")
    system_group.add_argument("--no-plots", action="store_true",
                            help="Disable plot generation and saving")
    system_group.add_argument("--quiet", action="store_true",
                            help="Minimize output (only show important messages)")
    
    args = parser.parse_args()
    
    # Load configuration preset
    if args.preset:
        print(f"📋 Loading preset: {args.preset}")
        config = get_quick_config(args.preset)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.hidden_dim:
        config.model.hidden_dim = args.hidden_dim
    if args.layers:
        config.model.num_lstm_layers = args.layers
    if args.dropout is not None:
        config.model.dropout = args.dropout
    if args.sequence_length:
        config.model.T = args.sequence_length
        config.data.T = args.sequence_length
        
    if args.epochs:
        config.training.epochs = args.epochs
    if args.fine_tune_epochs:
        config.training.fine_tune_epochs = args.fine_tune_epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.freeze_layers is not None:
        config.training.freeze_first_n_lstm = args.freeze_layers
        
    if args.alpha is not None:
        config.training.alpha = args.alpha
    if args.beta is not None:
        config.training.beta = args.beta
    if args.lambda_sparse is not None:
        config.training.lambda_sparse = args.lambda_sparse
    if args.lambda_decay is not None:
        config.training.lambda_decay = args.lambda_decay
    if args.decay_rate is not None:
        config.training.rho = args.decay_rate
        
    if args.source_size:
        config.data.source_dataset_size = args.source_size
    if args.target_size:
        config.data.target_dataset_size = args.target_size
    if args.min_room_size:
        config.data.min_room_size = args.min_room_size
    if args.max_room_size:
        config.data.max_room_size = args.max_room_size
    if args.sampling_rate:
        config.data.fs = args.sampling_rate
        
    if args.device:
        config.device = args.device
        if config.device == "auto":
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed:
        config.seed = args.seed
    if args.no_plots:
        config.save_plots = False
    if args.quiet:
        config.verbose = False
    
    # Re-validate configuration after CLI overrides
    config.__post_init__()
    
    # Show configuration summary
    if config.verbose:
        config.print_summary()
    
    main(config)
