"""Training script for GAN baseline model.

This script trains the GAN-based RIR generator as a baseline for comparison
with the DeepRIRNet transfer learning approach.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_gan(
    gan: nn.Module,
    dataloader: DataLoader,
    epochs: int = 50,
    lr_g: float = 0.0002,
    lr_d: float = 0.0002,
    beta1: float = 0.5,
    beta2: float = 0.999,
    device: str = "cpu",
    d_steps: int = 1,
    g_steps: int = 1,
    label_smoothing: float = 0.1,
    verbose: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Train the GAN model using Wasserstein GAN with gradient penalty.
    
    Args:
        gan: RIRGAN model
        dataloader: DataLoader providing (geometry, rir) pairs
        epochs: Number of training epochs
        lr_g: Learning rate for generator
        lr_d: Learning rate for discriminator
        beta1: Adam optimizer beta1 parameter
        beta2: Adam optimizer beta2 parameter
        device: Device to run training on
        d_steps: Number of discriminator updates per generator update
        g_steps: Number of generator updates per iteration
        label_smoothing: Label smoothing factor for discriminator (helps stability)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (generator_losses, discriminator_losses) per epoch
    """
    gan.to(device)
    
    # Separate optimizers for generator and discriminator
    optimizer_g = optim.Adam(
        gan.generator.parameters(), 
        lr=lr_g, 
        betas=(beta1, beta2)
    )
    optimizer_d = optim.Adam(
        gan.discriminator.parameters(), 
        lr=lr_d, 
        betas=(beta1, beta2)
    )
    
    # Binary cross entropy loss
    criterion = nn.BCELoss()
    
    g_losses = []
    d_losses = []
    
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else dataloader
        
        for batch_idx, (x, h_real) in enumerate(iterator):
            x = x.to(device)
            h_real = h_real.to(device)
            batch_size = x.shape[0]
            
            # Labels for real and fake
            real_labels = torch.ones(batch_size, 1, device=device) * (1.0 - label_smoothing)
            fake_labels = torch.zeros(batch_size, 1, device=device) + label_smoothing
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            for _ in range(d_steps):
                optimizer_d.zero_grad()
                
                # Discriminator loss on real RIRs
                pred_real = gan.discriminate(h_real, x)
                loss_real = criterion(pred_real, real_labels)
                
                # Generate fake RIRs
                with torch.no_grad():
                    h_fake = gan.generate(x)
                
                # Discriminator loss on fake RIRs
                pred_fake = gan.discriminate(h_fake.detach(), x)
                loss_fake = criterion(pred_fake, fake_labels)
                
                # Total discriminator loss
                d_loss = (loss_real + loss_fake) / 2
                d_loss.backward()
                optimizer_d.step()
                
                epoch_d_loss += d_loss.item()
            
            # -----------------
            # Train Generator
            # -----------------
            for _ in range(g_steps):
                optimizer_g.zero_grad()
                
                # Generate fake RIRs
                h_fake = gan.generate(x)
                
                # Generator wants discriminator to classify fakes as real
                pred_fake = gan.discriminate(h_fake, x)
                g_loss = criterion(pred_fake, real_labels)
                
                g_loss.backward()
                optimizer_g.step()
                
                epoch_g_loss += g_loss.item()
            
            num_batches += 1
            
            if verbose and batch_idx % 10 == 0:
                iterator.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}',
                    'G_loss': f'{g_loss.item():.4f}'
                })
        
        # Average losses for epoch
        avg_g_loss = epoch_g_loss / (num_batches * g_steps)
        avg_d_loss = epoch_d_loss / (num_batches * d_steps)
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")
    
    return g_losses, d_losses


def train_gan_wgan_gp(
    gan: nn.Module,
    dataloader: DataLoader,
    epochs: int = 50,
    lr: float = 0.0001,
    beta1: float = 0.0,
    beta2: float = 0.9,
    device: str = "cpu",
    lambda_gp: float = 10.0,
    n_critic: int = 5,
    verbose: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Train GAN using Wasserstein GAN with Gradient Penalty (WGAN-GP).
    
    This is more stable than vanilla GAN and doesn't require careful balancing.
    
    Args:
        gan: RIRGAN model
        dataloader: DataLoader providing (geometry, rir) pairs
        epochs: Number of training epochs
        lr: Learning rate
        beta1: Adam beta1
        beta2: Adam beta2
        device: Device to run on
        lambda_gp: Gradient penalty coefficient
        n_critic: Number of critic updates per generator update
        verbose: Whether to print progress
        
    Returns:
        Tuple of (generator_losses, critic_losses) per epoch
    """
    gan.to(device)
    
    # Optimizers
    optimizer_g = optim.Adam(gan.generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = optim.Adam(gan.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    g_losses = []
    d_losses = []
    
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else dataloader
        
        for batch_idx, (x, h_real) in enumerate(iterator):
            x = x.to(device)
            h_real = h_real.to(device)
            batch_size = x.shape[0]
            
            # ---------------------
            # Train Critic (Discriminator)
            # ---------------------
            for _ in range(n_critic):
                optimizer_d.zero_grad()
                
                # Real RIRs
                pred_real = gan.discriminate(h_real, x)
                
                # Fake RIRs
                h_fake = gan.generate(x).detach()
                pred_fake = gan.discriminate(h_fake, x)
                
                # Gradient penalty
                alpha = torch.rand(batch_size, 1, device=device)
                alpha = alpha.expand_as(h_real)
                
                interpolates = (alpha * h_real + (1 - alpha) * h_fake).requires_grad_(True)
                pred_interpolates = gan.discriminate(interpolates, x)
                
                gradients = torch.autograd.grad(
                    outputs=pred_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones_like(pred_interpolates),
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                gradients = gradients.view(batch_size, -1)
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
                
                # Wasserstein loss
                d_loss = -torch.mean(pred_real) + torch.mean(pred_fake) + gradient_penalty
                
                d_loss.backward()
                optimizer_d.step()
                
                epoch_d_loss += d_loss.item()
            
            # -----------------
            # Train Generator
            # -----------------
            optimizer_g.zero_grad()
            
            h_fake = gan.generate(x)
            pred_fake = gan.discriminate(h_fake, x)
            
            # Generator wants to maximize discriminator output for fakes
            g_loss = -torch.mean(pred_fake)
            
            g_loss.backward()
            optimizer_g.step()
            
            epoch_g_loss += g_loss.item()
            num_batches += 1
            
            if verbose and batch_idx % 10 == 0:
                iterator.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}',
                    'G_loss': f'{g_loss.item():.4f}'
                })
        
        # Average losses
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / (num_batches * n_critic)
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")
    
    return g_losses, d_losses
