"""DeepRIRNet model implementation."""

from typing import Optional
import torch
import torch.nn as nn


class DeepRIRNet(nn.Module):
    """
    DeepRIRNet: A geometry-aware neural network for Room Impulse Response prediction.
    
    This model uses a geometry encoder followed by LSTM layers to predict RIR sequences.
    The architecture includes residual connections, layer normalization, and dropout
    for improved training stability.
    
    Args:
        input_dim: Dimension of input geometry features
        T: Length of output RIR sequence
        hidden_dim: Hidden dimension for LSTM layers
        num_lstm_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    
    def __init__(
        self, 
        input_dim: int, 
        T: int, 
        hidden_dim: int = 512, 
        num_lstm_layers: int = 6, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.T = T
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        
        # Geometry encoder: project input features to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            for _ in range(num_lstm_layers)
        ])
        
        # Layer normalization for each LSTM layer
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_lstm_layers)
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output projection to generate RIR samples
        self.out_linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input geometry features of shape (batch_size, input_dim)
            
        Returns:
            Predicted RIR of shape (batch_size, T)
        """
        batch_size = x.size(0)
        
        # Project geometry features to hidden dimension
        x_proj = self.input_proj(x)  # (batch_size, hidden_dim)
        
        # Expand to sequence length
        h = x_proj.unsqueeze(1).repeat(1, self.T, 1)  # (batch_size, T, hidden_dim)
        
        # Pass through LSTM layers with residual connections
        for lstm, norm in zip(self.lstm_layers, self.norm_layers):
            # LSTM forward pass
            out, _ = lstm(h)
            
            # Residual connection + layer norm
            out = norm(out + h)
            
            # Apply dropout
            out = self.dropout(out)
            
            # Update hidden state
            h = out
        
        # Project to output RIR samples
        output = self.out_linear(h).squeeze(-1)  # (batch_size, T)
        
        return output
    
    def freeze_layers(self, num_layers: int) -> None:
        """
        Freeze the first num_layers LSTM layers for transfer learning.
        
        Args:
            num_layers: Number of LSTM layers to freeze
        """
        for i in range(min(num_layers, self.num_lstm_layers)):
            for param in self.lstm_layers[i].parameters():
                param.requires_grad = False
                
    def unfreeze_all(self) -> None:
        """Unfreeze all parameters in the model."""
        for param in self.parameters():
            param.requires_grad = True
