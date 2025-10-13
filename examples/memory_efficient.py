"""
Example: Memory-Constrained Training

This configuration is optimized for systems with limited GPU memory (4GB or less).
Still produces good quality results while staying within memory constraints.

Use cases:
- Training on older/smaller GPUs
- Laptop development
- Memory-constrained environments
- When sharing GPU resources

Expected performance: Good quality with minimal memory usage
"""

from config import Config, ModelConfig, TrainingConfig, DataConfig

def get_memory_efficient_config():
    """Configuration for memory-constrained environments."""
    
    return Config(
        model=ModelConfig(
            hidden_dim=256,           # Smaller hidden dimension
            num_lstm_layers=4,        # Fewer layers to save memory
            dropout=0.25,             # Slightly higher dropout for regularization
            T=256,                    # Shorter sequences
        ),
        
        training=TrainingConfig(
            batch_size=4,             # Small batches to fit in memory
            learning_rate=1e-3,       # Standard learning rate
            epochs=60,                # More epochs to compensate for small batches
            fine_tune_epochs=25,      # Adequate fine-tuning
            freeze_first_n_lstm=1,    # Standard transfer learning
            
            # Standard loss weights
            alpha=1.0,
            beta=0.1,
            lambda_sparse=1e-4,
            lambda_decay=1e-4,
            rho=0.3,
        ),
        
        data=DataConfig(
            fs=16000,
            T=256,                    # Match model T
            source_dataset_size=200,  # Reasonable dataset size
            target_dataset_size=25,   # Small but adequate
            
            # Standard room parameters
            min_room_size=4.0,
            max_room_size=12.0,
            min_absorption=0.2,
            max_absorption=0.8,
            reflection_order=1,       # Simple acoustics to save computation
            
            source_room_type="rectangular",
            target_room_type="rectangular",
        ),
        
        # System settings optimized for memory
        device="auto",
        seed=42,
        save_plots=True,
        verbose=True,
    )

def get_cpu_only_config():
    """Configuration for CPU-only training (no GPU available)."""
    
    config = get_memory_efficient_config()
    
    # Further optimizations for CPU training
    config.device = "cpu"
    config.model.hidden_dim = 128       # Even smaller model
    config.model.num_lstm_layers = 3    # Fewer layers
    config.training.batch_size = 2      # Very small batches
    config.training.epochs = 30         # Fewer epochs (CPU is slow)
    config.training.fine_tune_epochs = 15
    config.data.source_dataset_size = 100  # Smaller datasets
    config.data.target_dataset_size = 15
    
    return config

def estimate_memory_usage(config):
    """Estimate GPU memory usage for a configuration."""
    
    # Rough estimation formula (in GB)
    memory_gb = (
        config.model.hidden_dim *
        config.model.num_lstm_layers *
        config.training.batch_size *
        config.model.T *
        4  # float32
    ) / (1024**3)
    
    return memory_gb

# Usage examples:
if __name__ == "__main__":
    # Memory-efficient GPU config
    config = get_memory_efficient_config()
    memory_usage = estimate_memory_usage(config)
    
    print(f"💾 Memory-efficient config loaded")
    print(f"📊 Estimated GPU memory usage: {memory_usage:.1f}GB")
    config.print_summary()
    
    # CPU-only config
    cpu_config = get_cpu_only_config()
    print("\n" + "="*50)
    print("🖥️  CPU-Only Configuration")
    print("="*50)
    print("⚠️  Note: CPU training is much slower but works without GPU")
    cpu_config.print_summary()