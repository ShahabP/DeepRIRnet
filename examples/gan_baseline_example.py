"""Example script for training and evaluating the GAN baseline.

This script demonstrates how to:
1. Load data for RIR estimation
2. Train the GAN baseline model
3. Evaluate against DeepRIRNet
4. Use LibriSpeech for dereverberation experiments
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from models.gan_baseline import RIRGAN
from models.deep_rir_net import DeepRIRNet
from data.dataset import RIRDataset
from data.timit_loader import TIMITLoader, check_timit_license

# Import training and evaluation utilities
from utils.train_gan import train_gan, train_gan_wgan_gp
from utils.evaluate_gan import evaluate_and_compare


def main():
    """Main training and evaluation pipeline."""
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model hyperparameters
    input_dim = 6  # geometry features (room dims, source pos, receiver pos)
    hidden_dim = 256
    output_dim = 4096  # RIR length
    latent_dim = 128
    
    # Training hyperparameters
    batch_size = 32
    epochs = 50
    learning_rate = 0.0002
    
    # ========================================
    # 1. Load RIR Dataset
    # ========================================
    print("\n" + "="*70)
    print("Step 1: Loading RIR Dataset")
    print("="*70)
    
    # Assuming you have your RIR data in the data/ directory
    # You would load your actual dataset here
    # For demonstration, we'll show the structure
    
    # Example:
    # train_dataset = RIRDataset(
    #     data_path='data/source_domain_train.npz',
    #     geometry_key='geometry',
    #     rir_key='rir'
    # )
    # test_dataset = RIRDataset(
    #     data_path='data/target_domain_test.npz',
    #     geometry_key='geometry',
    #     rir_key='rir'
    # )
    
    print("Note: Update this script with your actual data paths")
    print("Expected data format: .npz with 'geometry' and 'rir' arrays")
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ========================================
    # 2. Initialize Models
    # ========================================
    print("\n" + "="*70)
    print("Step 2: Initializing Models")
    print("="*70)
    
    # Initialize GAN
    gan = RIRGAN(
        input_dim=input_dim,
        T=output_dim,  # Use T parameter instead of output_dim
        latent_dim=latent_dim,
        g_hidden_dim=hidden_dim,
        d_hidden_dim=hidden_dim // 4
    )
    print(f"GAN initialized with {sum(p.numel() for p in gan.parameters()):,} parameters")
    
    # Initialize DeepRIRNet for comparison
    deeprir = DeepRIRNet(
        input_dim=input_dim,
        T=output_dim,  # Use T parameter
        hidden_dim=512,  # As per paper
        num_lstm_layers=3  # As per paper
    )
    print(f"DeepRIRNet initialized with {sum(p.numel() for p in deeprir.parameters()):,} parameters")
    
    # ========================================
    # 3. Train GAN Baseline
    # ========================================
    print("\n" + "="*70)
    print("Step 3: Training GAN Baseline")
    print("="*70)
    
    # Option 1: Standard GAN training
    # g_losses, d_losses = train_gan(
    #     gan=gan,
    #     dataloader=train_loader,
    #     epochs=epochs,
    #     lr_g=learning_rate,
    #     lr_d=learning_rate,
    #     device=device,
    #     verbose=True
    # )
    
    # Option 2: WGAN-GP training (more stable)
    # g_losses, d_losses = train_gan_wgan_gp(
    #     gan=gan,
    #     dataloader=train_loader,
    #     epochs=epochs,
    #     lr=0.0001,
    #     device=device,
    #     verbose=True
    # )
    
    print("Training complete! (uncomment training code above to run)")
    
    # Save model
    # torch.save(gan.state_dict(), 'checkpoints/gan_baseline.pth')
    # print("Model saved to checkpoints/gan_baseline.pth")
    
    # ========================================
    # 4. Evaluate Models
    # ========================================
    print("\n" + "="*70)
    print("Step 4: Evaluating Models")
    print("="*70)
    
    # Load pre-trained DeepRIRNet if available
    # deeprir_checkpoint = 'checkpoints/deeprirnet_finetuned.pth'
    # if os.path.exists(deeprir_checkpoint):
    #     deeprir.load_state_dict(torch.load(deeprir_checkpoint))
    #     print(f"Loaded DeepRIRNet from {deeprir_checkpoint}")
    
    # Compare both models
    # results = evaluate_and_compare(
    #     gan_model=gan,
    #     deeprir_model=deeprir,
    #     dataloader=test_loader,
    #     device=device
    # )
    
    print("Evaluation complete! (uncomment evaluation code above to run)")
    
    # ========================================
    # 5. TIMIT/LibriSpeech for Dereverberation
    # ========================================
    print("\n" + "="*70)
    print("Step 5: Speech Dereverberation with LibriSpeech")
    print("="*70)
    
    # Print TIMIT license information
    check_timit_license()
    
    print("\nSince TIMIT requires a license, we'll use LibriSpeech (free):")
    
    # Initialize loader with free alternative
    timit_loader = TIMITLoader(use_alternative=True, sample_rate=16000)
    
    # Get test utterances
    print("\nLoading speech utterances...")
    # utterances = timit_loader.get_test_utterances(num_utterances=10, max_duration=3.0)
    # print(f"Loaded {len(utterances)} utterances")
    
    # For dereverberation experiments:
    # 1. Generate reverberant speech by convolving clean speech with estimated RIR
    # 2. Apply your dereverberation method
    # 3. Compute speech quality metrics (PESQ, STOI)
    
    print("\nExample dereverberation workflow:")
    print("  1. Load clean speech: utterances = loader.get_test_utterances()")
    print("  2. Estimate RIR: h_est = gan.generate(geometry_features)")
    print("  3. Generate reverberant: reverb = convolve(clean_speech, h_est)")
    print("  4. Dereverb & evaluate: pesq, stoi = evaluate_dereverberation()")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("✓ GAN baseline implemented and ready for training")
    print("✓ TIMIT loader with LibriSpeech alternative (free)")
    print("✓ Evaluation utilities for fair comparison")
    print("✓ Repository structure preserved")
    print("\nNext steps:")
    print("  1. Prepare your RIR dataset (source and target domains)")
    print("  2. Uncomment training code and run this script")
    print("  3. Compare GAN vs DeepRIRNet results")
    print("  4. Test dereverberation on speech (LibriSpeech)")
    print("="*70)


if __name__ == "__main__":
    main()
