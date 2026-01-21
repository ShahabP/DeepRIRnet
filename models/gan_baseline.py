"""GAN-based baseline for RIR generation.

This module implements a Generative Adversarial Network approach for Room Impulse
Response generation as described in Ratnarajah et al. (2022). The GAN consists of
a generator that produces RIRs from geometry features and a discriminator that
distinguishes real from generated RIRs.

Reference:
    Ratnarajah, A., Tang, Z., & Manocha, D. (2022). IR-GAN: Room impulse response 
    generator for far-field speech recognition. INTERSPEECH 2022.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RIRGenerator(nn.Module):
    """
    Generator network for RIR synthesis.
    
    The generator takes geometry features and noise as input, and produces
    synthetic RIR sequences through a series of transposed convolutions.
    
    Args:
        input_dim: Dimension of geometry features
        latent_dim: Dimension of random noise vector
        T: Length of output RIR sequence
        hidden_dim: Base hidden dimension for convolutional layers
    """
    
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int = 128,
        T: int = 4096, 
        hidden_dim: int = 256
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.T = T
        self.hidden_dim = hidden_dim
        
        # Project geometry features and noise to initial sequence
        self.fc = nn.Sequential(
            nn.Linear(input_dim + latent_dim, hidden_dim * 8),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.ReLU(inplace=True)
        )
        
        # Transposed convolutions to upsample to full RIR length
        # Starting from 64 samples, upsample to 4096 (64x upsampling)
        self.deconv_blocks = nn.ModuleList([
            # 64 -> 128
            self._make_deconv_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            # 128 -> 256
            self._make_deconv_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            # 256 -> 512
            self._make_deconv_block(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            # 512 -> 1024
            self._make_deconv_block(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            # 1024 -> 2048
            self._make_deconv_block(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            # 2048 -> 4096
            self._make_deconv_block(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1),
        ])
        
        # Final convolution to produce single-channel RIR
        self.final_conv = nn.Conv1d(hidden_dim // 8, 1, kernel_size=3, padding=1)
        
    def _make_deconv_block(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 4, 
        stride: int = 2, 
        padding: int = 1
    ) -> nn.Sequential:
        """Create a deconvolution block with normalization and activation."""
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate RIR from geometry features.
        
        Args:
            x: Geometry features [batch_size, input_dim]
            z: Random noise [batch_size, latent_dim]. If None, sampled from N(0,1)
            
        Returns:
            Generated RIR sequence [batch_size, T]
        """
        batch_size = x.shape[0]
        
        # Sample noise if not provided
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=x.device)
        
        # Concatenate geometry features and noise
        x = torch.cat([x, z], dim=1)  # [batch_size, input_dim + latent_dim]
        
        # Project to initial representation
        x = self.fc(x)  # [batch_size, hidden_dim * 8]
        
        # Reshape for convolution: [batch_size, channels, sequence_length]
        x = x.view(batch_size, self.hidden_dim * 8, 1).repeat(1, 1, 64)
        
        # Upsample through deconv blocks
        for deconv_block in self.deconv_blocks:
            x = deconv_block(x)
        
        # Final convolution and squeeze to [batch_size, T]
        x = self.final_conv(x)
        x = x.squeeze(1)
        
        # Ensure output has correct length
        if x.shape[1] != self.T:
            x = F.interpolate(x.unsqueeze(1), size=self.T, mode='linear', align_corners=False).squeeze(1)
        
        return x


class RIRDiscriminator(nn.Module):
    """
    Discriminator network for RIR authenticity classification.
    
    The discriminator takes an RIR sequence and geometry features as input,
    and outputs a probability that the RIR is real (vs. generated).
    
    Args:
        input_dim: Dimension of geometry features
        T: Length of RIR sequence
        hidden_dim: Base hidden dimension for convolutional layers
    """
    
    def __init__(self, input_dim: int, T: int = 4096, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.T = T
        self.hidden_dim = hidden_dim
        
        # Convolutional layers to process RIR
        self.conv_blocks = nn.ModuleList([
            # 4096 -> 2048
            self._make_conv_block(1, hidden_dim, kernel_size=4, stride=2, padding=1),
            # 2048 -> 1024
            self._make_conv_block(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            # 1024 -> 512
            self._make_conv_block(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            # 512 -> 256
            self._make_conv_block(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1),
            # 256 -> 128
            self._make_conv_block(hidden_dim * 8, hidden_dim * 16, kernel_size=4, stride=2, padding=1),
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier: combine RIR features with geometry features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 16 + input_dim, hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, 1),
            nn.Sigmoid()
        )
        
    def _make_conv_block(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 4, 
        stride: int = 2, 
        padding: int = 1
    ) -> nn.Sequential:
        """Create a convolution block with normalization and activation."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Classify RIR as real or fake.
        
        Args:
            h: RIR sequence [batch_size, T]
            x: Geometry features [batch_size, input_dim]
            
        Returns:
            Probability that RIR is real [batch_size, 1]
        """
        # Add channel dimension for convolution
        h = h.unsqueeze(1)  # [batch_size, 1, T]
        
        # Extract features through conv blocks
        for conv_block in self.conv_blocks:
            h = conv_block(h)
        
        # Global pooling
        h = self.global_pool(h).squeeze(-1)  # [batch_size, hidden_dim * 16]
        
        # Concatenate with geometry features
        combined = torch.cat([h, x], dim=1)  # [batch_size, hidden_dim * 16 + input_dim]
        
        # Classify
        out = self.classifier(combined)
        
        return out


class RIRGAN(nn.Module):
    """
    Complete GAN model combining generator and discriminator.
    
    This wrapper facilitates training and inference by managing both networks.
    
    Args:
        input_dim: Dimension of geometry features
        T: Length of RIR sequence
        latent_dim: Dimension of generator noise
        g_hidden_dim: Hidden dimension for generator
        d_hidden_dim: Hidden dimension for discriminator
    """
    
    def __init__(
        self, 
        input_dim: int, 
        T: int = 4096,
        latent_dim: int = 128,
        g_hidden_dim: int = 256,
        d_hidden_dim: int = 64
    ):
        super().__init__()
        self.generator = RIRGenerator(input_dim, latent_dim, T, g_hidden_dim)
        self.discriminator = RIRDiscriminator(input_dim, T, d_hidden_dim)
        
    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate RIR (inference mode uses generator only).
        
        Args:
            x: Geometry features [batch_size, input_dim]
            z: Random noise (optional)
            
        Returns:
            Generated RIR sequence [batch_size, T]
        """
        return self.generator(x, z)
    
    def generate(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Alias for forward (explicit generation)."""
        return self.forward(x, z)
    
    def discriminate(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Classify RIR as real or fake.
        
        Args:
            h: RIR sequence [batch_size, T]
            x: Geometry features [batch_size, input_dim]
            
        Returns:
            Probability that RIR is real [batch_size, 1]
        """
        return self.discriminator(h, x)
