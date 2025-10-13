"""
Example: Production Quality Training

This configuration is optimized for high-quality results suitable for
production applications. Training takes ~2-4 hours but produces excellent RIRs.

Use cases:
- Final model training for deployment
- Research publications
- High-quality audio applications
- When computational resources are available

Expected performance: Excellent quality, state-of-the-art results
"""

from config import Config, ModelConfig, TrainingConfig, DataConfig

def get_production_config():
    """Configuration for production-quality training."""
    
    return Config(
        model=ModelConfig(
            hidden_dim=1024,          # Large model for quality
            num_lstm_layers=8,        # Deep temporal modeling
            dropout=0.15,             # Light dropout for large model
            T=1024,                   # Longer sequences for detail
        ),
        
        training=TrainingConfig(
            batch_size=8,             # Balanced for memory/performance
            learning_rate=5e-4,       # Conservative for stability
            epochs=100,               # Thorough source training
            fine_tune_epochs=40,      # Careful fine-tuning
            freeze_first_n_lstm=2,    # Preserve more learned features
            
            # Carefully tuned loss weights for quality
            alpha=1.0,                # Time-domain accuracy
            beta=0.2,                 # Higher spectral emphasis
            lambda_sparse=2e-4,       # Stronger sparsity for realism
            lambda_decay=2e-4,        # Better decay enforcement
            rho=0.25,                 # Slightly longer decay
        ),
        
        data=DataConfig(
            fs=16000,                 # Standard sampling rate
            T=1024,                   # Match model T - longer sequences
            source_dataset_size=500,  # Large diverse source dataset
            target_dataset_size=50,   # Substantial target dataset
            
            # Wide range of acoustic scenarios
            min_room_size=3.0,
            max_room_size=15.0,
            min_absorption=0.15,      # More reverberant
            max_absorption=0.85,      # More absorptive range
            reflection_order=2,       # More detailed acoustics
            
            source_room_type="rectangular",
            target_room_type="rectangular",
        ),
        
        # System settings
        device="auto",
        seed=42,
        save_plots=True,
        plot_dir="production_plots",
        checkpoint_dir="production_checkpoints",
        verbose=True,
    )

def get_music_production_config():
    """Specialized configuration for music/audio production applications."""
    
    config = get_production_config()
    
    # Modifications for music applications
    config.data.fs = 44100              # Higher sampling rate for music
    config.model.T = 2048               # Longer RIRs for music venues
    config.data.T = 2048
    config.training.batch_size = 4      # Smaller batches due to longer sequences
    config.data.min_room_size = 8.0     # Larger performance spaces
    config.data.max_room_size = 40.0    # Concert halls
    config.training.rho = 0.15          # Longer reverberation for music
    
    return config

# Usage examples:
if __name__ == "__main__":
    # Standard production config
    config = get_production_config()
    config.print_summary()
    print("🎯 Production config loaded - training time: ~2-4 hours")
    
    # Music production config  
    music_config = get_music_production_config()
    print("\n" + "="*50)
    print("🎵 Music Production Configuration")
    print("="*50)
    music_config.print_summary()