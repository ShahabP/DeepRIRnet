"""
Example: Quick Development and Testing

This configuration is optimized for fast iteration during development.
Training completes in ~5 minutes, perfect for:
- Testing code changes
- Rapid prototyping  
- Algorithm development
- Initial feasibility studies

Expected performance: Basic quality, suitable for development
"""

from config import Config, ModelConfig, TrainingConfig, DataConfig

def get_fast_development_config():
    """Configuration for fast development and testing."""
    
    return Config(
        model=ModelConfig(
            hidden_dim=256,           # Smaller model for speed
            num_lstm_layers=3,        # Fewer layers
            dropout=0.2,              # Standard regularization
            T=256,                    # Shorter RIR sequences
        ),
        
        training=TrainingConfig(
            batch_size=16,            # Larger batches for speed
            learning_rate=2e-3,       # Slightly higher LR for fast convergence  
            epochs=15,                # Quick source training
            fine_tune_epochs=8,       # Quick fine-tuning
            freeze_first_n_lstm=1,    # Standard transfer learning
            
            # Balanced loss weights
            alpha=1.0,                # Time-domain accuracy
            beta=0.1,                 # Spectral quality
            lambda_sparse=1e-4,       # Light regularization for speed
            lambda_decay=1e-4,
            rho=0.3,                  # Medium reverberation
        ),
        
        data=DataConfig(
            fs=16000,                 # Standard sampling rate
            T=256,                    # Match model T
            source_dataset_size=50,   # Small dataset for speed
            target_dataset_size=10,   # Very small target
            
            # Standard room parameters
            min_room_size=4.0,
            max_room_size=10.0,
            min_absorption=0.2,
            max_absorption=0.7,
            reflection_order=1,       # Simple acoustics
            
            source_room_type="rectangular",
            target_room_type="rectangular",
        ),
        
        # System settings
        device="auto",
        seed=42,
        save_plots=True,
        verbose=True,
    )

# Usage example:
if __name__ == "__main__":
    config = get_fast_development_config()
    config.print_summary()
    print("🚀 Fast development config loaded - training time: ~5 minutes")