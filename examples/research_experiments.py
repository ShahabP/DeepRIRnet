"""
Example: Research and Experimentation

This configuration is designed for research applications where you need
maximum flexibility and comprehensive analysis capabilities.

Use cases:
- Academic research
- Algorithm development
- Ablation studies
- Parameter sensitivity analysis
- Paper reproducibility

Expected performance: Research-grade with extensive logging and analysis
"""

from config import Config, ModelConfig, TrainingConfig, DataConfig

def get_research_config():
    """Configuration optimized for research and experimentation."""
    
    return Config(
        model=ModelConfig(
            hidden_dim=512,           # Balanced size for experiments
            num_lstm_layers=6,        # Standard depth
            dropout=0.2,              # Standard regularization
            T=512,                    # Standard sequence length
        ),
        
        training=TrainingConfig(
            batch_size=8,             # Standard batch size
            learning_rate=1e-3,       # Standard starting point
            epochs=80,                # Thorough training
            fine_tune_epochs=35,      # Adequate fine-tuning
            freeze_first_n_lstm=1,    # Standard transfer learning
            
            # Well-documented baseline loss weights
            alpha=1.0,                # Time-domain MSE weight
            beta=0.1,                 # Spectral LSD weight  
            lambda_sparse=1e-4,       # Sparsity regularization
            lambda_decay=1e-4,        # Decay regularization
            rho=0.3,                  # Decay rate
        ),
        
        data=DataConfig(
            fs=16000,                 # Standard sampling rate
            T=512,                    # Match model T
            source_dataset_size=300,  # Good size for research
            target_dataset_size=30,   # Adequate target size
            
            # Well-defined room parameters for reproducibility
            min_room_size=4.0,
            max_room_size=12.0,
            min_absorption=0.2,
            max_absorption=0.8,
            reflection_order=1,
            
            source_room_type="rectangular",
            target_room_type="rectangular",
        ),
        
        # Research-focused settings
        device="auto",
        seed=42,                      # Fixed seed for reproducibility
        save_plots=True,
        plot_dir="research_plots",
        checkpoint_dir="research_checkpoints", 
        verbose=True,
    )

def get_ablation_study_configs():
    """Generate configurations for ablation studies."""
    
    configs = {}
    base_config = get_research_config()
    
    # Architecture ablations
    configs['small_model'] = get_research_config()
    configs['small_model'].model.hidden_dim = 256
    configs['small_model'].model.num_lstm_layers = 4
    
    configs['large_model'] = get_research_config()  
    configs['large_model'].model.hidden_dim = 1024
    configs['large_model'].model.num_lstm_layers = 8
    
    # Loss weight ablations
    configs['no_spectral'] = get_research_config()
    configs['no_spectral'].training.beta = 0.0
    
    configs['no_regularization'] = get_research_config()
    configs['no_regularization'].training.lambda_sparse = 0.0
    configs['no_regularization'].training.lambda_decay = 0.0
    
    configs['strong_regularization'] = get_research_config()
    configs['strong_regularization'].training.lambda_sparse = 1e-3
    configs['strong_regularization'].training.lambda_decay = 1e-3
    
    # Learning rate ablations
    configs['low_lr'] = get_research_config()
    configs['low_lr'].training.learning_rate = 1e-4
    
    configs['high_lr'] = get_research_config()
    configs['high_lr'].training.learning_rate = 2e-3
    
    return configs

def get_reproducibility_config():
    """Configuration with all randomness controlled for perfect reproducibility."""
    
    config = get_research_config()
    
    # Ensure complete reproducibility
    config.seed = 12345              # Non-default seed
    config.training.epochs = 50      # Fixed number of epochs
    config.training.fine_tune_epochs = 25
    config.data.source_dataset_size = 200  # Fixed dataset sizes
    config.data.target_dataset_size = 20
    
    # Use deterministic algorithms (may be slower)
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return config

# Usage examples:
if __name__ == "__main__":
    # Standard research config
    config = get_research_config()
    print("🔬 Research configuration loaded")
    config.print_summary()
    
    # Ablation study configs
    ablation_configs = get_ablation_study_configs()
    print(f"\n📊 Generated {len(ablation_configs)} ablation study configurations:")
    for name, cfg in ablation_configs.items():
        print(f"  - {name}")
    
    # Reproducibility config
    repro_config = get_reproducibility_config()
    print(f"\n🔒 Reproducibility config generated with seed: {repro_config.seed}")