"""
Configuration management for DeepRIRNet.

This module provides comprehensive hyperparameter configuration with detailed explanations
and validation. All parameters are organized into logical groups with clear documentation
about their effects on model performance and training behavior.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import torch
import warnings


@dataclass
class ModelConfig:
    """
    Configuration for the DeepRIRNet model architecture.
    
    These parameters control the neural network structure and capacity.
    Larger values generally increase model expressiveness but also computational cost.
    """
    
    # INPUT FEATURES
    input_dim: int = 10  
    """
    Total input feature dimension (AUTOMATICALLY CALCULATED - do not change manually)
    Default: 10 = room_dims(3) + absorption(1) + source_pos(3) + mic_pos(3)
    """
    
    # NETWORK ARCHITECTURE
    hidden_dim: int = 512
    """
    Hidden dimension for LSTM layers and internal representations.
    
    Effects:
    - Higher values → More model capacity, better complex pattern learning
    - Lower values → Faster training, less overfitting, smaller model
    
    Typical ranges:
    - Small models: 128-256 (faster, good for simple rooms)
    - Medium models: 512 (default, balanced performance)
    - Large models: 1024+ (complex acoustics, more data needed)
    """
    
    num_lstm_layers: int = 6
    """
    Number of LSTM layers in the temporal decoder.
    
    Effects:
    - More layers → Better temporal modeling, can learn complex decay patterns
    - Fewer layers → Faster training, less prone to vanishing gradients
    
    Recommendations:
    - Simple rooms: 3-4 layers
    - Complex acoustics: 6-8 layers (default: 6)
    - Very complex: 8+ layers (watch for overfitting)
    """
    
    dropout: float = 0.2
    """
    Dropout probability for regularization (0.0 = no dropout, 1.0 = drop all).
    
    Effects:
    - Higher values → Stronger regularization, less overfitting
    - Lower values → Model can use full capacity, might overfit
    
    Typical ranges:
    - No regularization needed: 0.0-0.1
    - Standard training: 0.1-0.3 (default: 0.2)
    - Strong regularization: 0.3-0.5
    - Too high (>0.5): May hurt performance
    """
    
    T: int = 512
    """
    Length of output RIR sequence in samples.
    
    Effects:
    - Longer sequences → Capture more reverberation tail, slower training
    - Shorter sequences → Faster training, may miss late reflections
    
    Common values based on room size and fs:
    - Small rooms (fs=16kHz): 256-512 samples (~16-32ms)
    - Medium rooms: 512-1024 samples (~32-64ms) [DEFAULT]
    - Large halls: 1024-2048+ samples (64ms+)
    
    Rule of thumb: T_samples ≈ (RT60_seconds × fs) / 3
    """


@dataclass
class TrainingConfig:
    """
    Configuration for training hyperparameters and optimization.
    
    These parameters control how the model learns and the training process behavior.
    """
    
    # OPTIMIZATION PARAMETERS
    batch_size: int = 8
    """
    Number of samples processed together in each training step.
    
    Effects:
    - Larger batches → More stable gradients, better GPU utilization, more memory
    - Smaller batches → More stochastic training, less memory, might need more epochs
    
    Recommendations based on GPU memory:
    - 4GB GPU: 4-8 (default: 8)
    - 8GB GPU: 8-16
    - 16GB+ GPU: 16-32
    
    Also depends on sequence length T and model size.
    """
    
    learning_rate: float = 1e-3
    """
    Step size for parameter updates during optimization.
    
    Effects:
    - Higher LR → Faster training, risk of instability/overshooting
    - Lower LR → More stable training, slower convergence
    
    Typical ranges:
    - Conservative: 1e-4 to 5e-4
    - Standard: 1e-3 (default, good starting point)
    - Aggressive: 2e-3 to 5e-3
    
    Adjust based on loss behavior:
    - Loss oscillating → Reduce LR by 2-5x
    - Loss plateauing → Increase LR by 2-3x or use scheduler
    """
    
    epochs: int = 50
    """
    Number of complete passes through source domain training data.
    
    Guidelines:
    - Small datasets (< 100 samples): 100-200 epochs
    - Medium datasets (100-1000): 50-100 epochs (default: 50)
    - Large datasets (1000+): 20-50 epochs
    
    Watch for overfitting: stop if validation loss starts increasing.
    """
    
    fine_tune_epochs: int = 30
    """
    Number of epochs for target domain fine-tuning (transfer learning phase).
    
    Guidelines:
    - Should be less than main training epochs
    - Small target dataset: 10-20 epochs
    - Medium target dataset: 20-40 epochs (default: 30)
    - Large target dataset: 30-50 epochs
    """
    
    freeze_first_n_lstm: int = 1
    """
    Number of initial LSTM layers to freeze during fine-tuning.
    
    Transfer learning strategy:
    - 0: Fine-tune all layers (when domains are very different)
    - 1: Freeze first layer (default, good for similar acoustics)
    - 2-3: Freeze more layers (when domains are very similar)
    - All: Feature extraction only (freeze entire model)
    """
    
    # LOSS FUNCTION WEIGHTS (VERY IMPORTANT FOR PERFORMANCE)
    alpha: float = 1.0
    """
    Weight for MSE (time-domain) loss component.
    
    Controls time-domain reconstruction accuracy.
    - Higher α → Focus on sample-by-sample accuracy
    - Lower α → Less emphasis on exact waveform matching
    
    Typical values: 0.5-2.0 (default: 1.0)
    """
    
    beta: float = 0.1
    """
    Weight for Log-Spectral Distance (frequency-domain) loss.
    
    Controls frequency content preservation.
    - Higher β → Better spectral matching, perceptually important
    - Lower β → Less spectral constraint
    
    Typical values: 0.05-0.5 (default: 0.1)
    Start with 0.1, increase if spectral quality is poor.
    """
    
    lambda_sparse: float = 1e-4
    """
    Weight for sparsity regularization (encourages realistic sparse RIRs).
    
    Physics-informed constraint: Real RIRs have many near-zero samples.
    - Higher λ_sparse → More sparse (realistic), might lose detail
    - Lower λ_sparse → Less sparse constraint
    
    Typical values: 1e-5 to 1e-3 (default: 1e-4)
    Increase if RIRs look too dense/noisy.
    """
    
    lambda_decay: float = 1e-4
    """
    Weight for energy decay regularization (enforces exponential decay).
    
    Physics-informed constraint: RIR energy should decay over time.
    - Higher λ_decay → Stronger decay enforcement
    - Lower λ_decay → More flexible decay patterns
    
    Typical values: 1e-5 to 1e-3 (default: 1e-4)
    Increase if RIRs don't show proper energy decay.
    """
    
    rho: float = 0.3
    """
    Exponential decay rate for energy decay regularization.
    
    Controls how fast the RIR energy should decay:
    - Higher ρ → Faster decay (more absorptive rooms)
    - Lower ρ → Slower decay (more reverberant rooms)
    
    Typical values:
    - Absorptive rooms: 0.5-1.0
    - Medium reverberation: 0.2-0.5 (default: 0.3)
    - Very reverberant: 0.1-0.2
    """


@dataclass
class DataConfig:
    """
    Configuration for dataset generation and room simulation parameters.
    
    These control the acoustic scenarios the model will learn from.
    """
    
    # AUDIO PARAMETERS
    fs: int = 16000
    """
    Sampling frequency in Hz.
    
    Effects on model and data:
    - Higher fs → Better high-frequency detail, larger memory, slower training
    - Lower fs → Less detail, faster processing
    
    Common choices:
    - Speech applications: 8kHz or 16kHz (default)
    - Music/high-quality: 44.1kHz or 48kHz
    - Research: 16kHz (good balance)
    """
    
    T: int = 512
    """
    RIR length in samples (should match ModelConfig.T).
    
    Duration = T / fs seconds
    Default: 512 samples @ 16kHz = 32ms
    
    See ModelConfig.T for detailed guidelines.
    """
    
    # DATASET SIZES (CRITICAL FOR TRANSFER LEARNING)
    source_dataset_size: int = 150
    """
    Number of rooms in source domain for pretraining.
    
    Guidelines:
    - Minimum viable: 50-100 rooms
    - Recommended: 100-500 rooms (default: 150)
    - Large scale: 500+ rooms
    
    More data generally means better feature learning.
    """
    
    target_dataset_size: int = 20
    """
    Number of rooms in target domain for fine-tuning.
    
    Transfer learning advantage: Can work with small target datasets.
    - Few-shot learning: 5-20 rooms (default: 20)
    - Standard: 20-100 rooms  
    - Large target: 100+ rooms
    """
    
    # ROOM GEOMETRY PARAMETERS
    min_room_size: float = 4.0
    """
    Minimum room dimension in meters.
    
    Effects:
    - Smaller rooms → Shorter RIRs, higher frequency reflections
    - Larger rooms → Longer RIRs, more complex patterns
    
    Typical ranges:
    - Small rooms (offices): 3-6m (default min: 4m)
    - Medium rooms (classrooms): 6-12m
    - Large spaces (halls): 12m+
    """
    
    max_room_size: float = 12.0
    """
    Maximum room dimension in meters.
    
    Should be consistent with your target applications.
    Default: 12m covers most indoor environments.
    """
    
    min_absorption: float = 0.2
    """
    Minimum absorption coefficient (0=perfect reflection, 1=perfect absorption).
    
    Acoustic interpretation:
    - 0.1-0.3: Hard surfaces (concrete, glass) - very reverberant
    - 0.3-0.6: Medium (painted walls, wood) - moderate reverberation  
    - 0.6-1.0: Soft surfaces (carpet, drapes) - low reverberation
    
    Default: 0.2 (fairly reverberant)
    """
    
    max_absorption: float = 0.8
    """
    Maximum absorption coefficient.
    
    Default: 0.8 covers range from reverberant to quite absorptive.
    Avoid 1.0 (anechoic) unless specifically needed.
    """
    
    reflection_order: int = 1
    """
    Maximum order of reflections in image source method.
    
    Acoustic complexity vs computational cost:
    - Order 0: Direct path only (anechoic)
    - Order 1: Direct + first reflections (default, fast)
    - Order 2-3: More realistic but slower
    - Order 4+: Very detailed but computationally expensive
    
    For most applications, order 1 is sufficient.
    """
    
    # DOMAIN TYPES
    source_room_type: str = "rectangular"
    """
    Geometry type for source domain pretraining.
    
    Currently supported: "rectangular"
    Future: "l_shaped", "irregular", etc.
    """
    
    target_room_type: str = "rectangular"  # Changed from l_shaped (not implemented yet)
    """
    Geometry type for target domain fine-tuning.
    
    Currently: Using "rectangular" (l_shaped not implemented yet)
    This simulates domain adaptation through different acoustic parameters.
    """


@dataclass
class Config:
    """
    Main configuration class containing all sub-configurations.
    
    This class validates parameter combinations and provides warnings for
    potentially problematic settings.
    """
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # GLOBAL SETTINGS
    device: str = "auto"  
    """Device for computation: "auto", "cpu", "cuda", or specific GPU like "cuda:0"."""
    
    seed: int = 42
    """Random seed for reproducibility. Change for different random initializations."""
    
    # OUTPUT AND LOGGING
    save_plots: bool = True
    """Whether to save training plots and visualizations to disk."""
    
    plot_dir: str = "plots"
    """Directory to save plots and visualizations."""
    
    checkpoint_dir: str = "checkpoints"
    """Directory to save model checkpoints."""
    
    verbose: bool = True
    """Whether to print detailed training progress."""
    
    def __post_init__(self):
        """Validate configuration and set derived parameters."""
        self._set_device()
        self._validate_and_sync()
        self._check_parameter_compatibility()
        
    def _set_device(self):
        """Set device automatically if needed."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
    def _validate_and_sync(self):
        """Ensure consistency between related parameters."""
        # Sync RIR length between model and data configs
        if self.model.T != self.data.T:
            print(f"⚠️  Syncing RIR length: model.T={self.model.T} → data.T={self.data.T}")
            self.model.T = self.data.T
            
        # Calculate and set input dimension
        expected_input_dim = 3 + 1 + 3 + 3  # room + absorption + source + mic
        if self.model.input_dim != expected_input_dim:
            print(f"⚠️  Adjusting input_dim: {self.model.input_dim} → {expected_input_dim}")
            self.model.input_dim = expected_input_dim
            
    def _check_parameter_compatibility(self):
        """Check for potentially problematic parameter combinations."""
        issues = []
        
        # Memory usage warnings
        memory_estimate = (self.model.hidden_dim * self.model.num_lstm_layers * 
                          self.training.batch_size * self.model.T * 4) / (1024**3)  # GB
        
        if memory_estimate > 8:
            issues.append(f"⚠️  HIGH MEMORY USAGE: ~{memory_estimate:.1f}GB estimated. "
                         f"Consider reducing batch_size, hidden_dim, or num_lstm_layers.")
        
        # Training stability warnings
        if self.training.learning_rate > 0.01:
            issues.append("⚠️  HIGH LEARNING RATE: LR > 0.01 may cause training instability.")
            
        if self.training.dropout > 0.5:
            issues.append("⚠️  HIGH DROPOUT: >50% dropout may hurt model performance.")
            
        # Loss weight balance warnings
        if self.training.alpha < 0.1 or self.training.alpha > 10:
            issues.append("⚠️  EXTREME MSE WEIGHT: alpha should typically be 0.1-10.")
            
        if self.training.beta > 1.0:
            issues.append("⚠️  HIGH LSD WEIGHT: beta > 1.0 may dominate other losses.")
            
        # Physics regularization warnings  
        if (self.training.lambda_sparse > 1e-2 or self.training.lambda_decay > 1e-2):
            issues.append("⚠️  STRONG REGULARIZATION: λ > 0.01 may over-constrain the model.")
            
        # Dataset size warnings
        if self.data.source_dataset_size < 50:
            issues.append("⚠️  SMALL SOURCE DATASET: <50 samples may not provide enough diversity.")
            
        if self.data.target_dataset_size < 5:
            issues.append("⚠️  VERY SMALL TARGET DATASET: <5 samples may cause overfitting.")
            
        # Room parameter warnings
        if self.data.max_room_size / self.data.min_room_size > 5:
            issues.append("⚠️  LARGE ROOM SIZE RANGE: >5x difference may make training harder.")
            
        # Sequence length warnings
        duration_ms = (self.model.T / self.data.fs) * 1000
        if duration_ms < 20:
            issues.append(f"⚠️  SHORT RIR: {duration_ms:.1f}ms may miss important reflections.")
        elif duration_ms > 200:
            issues.append(f"⚠️  LONG RIR: {duration_ms:.1f}ms increases computational cost significantly.")
            
        # Print all warnings
        if issues:
            print("\n" + "="*60)
            print("🔍 CONFIGURATION VALIDATION WARNINGS:")
            print("="*60)
            for issue in issues:
                print(issue)
            print("="*60 + "\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key configuration parameters."""
        duration_ms = (self.model.T / self.data.fs) * 1000
        memory_estimate = (self.model.hidden_dim * self.model.num_lstm_layers * 
                          self.training.batch_size * self.model.T * 4) / (1024**3)
        
        return {
            "Model Architecture": {
                "Input Features": self.model.input_dim,
                "Hidden Dimension": self.model.hidden_dim, 
                "LSTM Layers": self.model.num_lstm_layers,
                "Dropout": f"{self.training.dropout:.1%}",
                "Parameters (est.)": f"{self.model.hidden_dim**2 * self.model.num_lstm_layers * 4:,}",
            },
            "Training Setup": {
                "Batch Size": self.training.batch_size,
                "Learning Rate": self.training.learning_rate,
                "Source Epochs": self.training.epochs,
                "Fine-tune Epochs": self.training.fine_tune_epochs,
                "Memory (est.)": f"{memory_estimate:.1f}GB",
            },
            "Data Configuration": {
                "RIR Duration": f"{duration_ms:.1f}ms ({self.model.T} samples)",
                "Sampling Rate": f"{self.data.fs}Hz",
                "Source Dataset": f"{self.data.source_dataset_size} rooms",
                "Target Dataset": f"{self.data.target_dataset_size} rooms",
                "Room Size Range": f"{self.data.min_room_size}-{self.data.max_room_size}m",
            },
            "Loss Weights": {
                "MSE (α)": self.training.alpha,
                "LSD (β)": self.training.beta, 
                "Sparsity (λ_s)": self.training.lambda_sparse,
                "Decay (λ_d)": self.training.lambda_decay,
                "Decay Rate (ρ)": self.training.rho,
            }
        }
    
    def print_summary(self):
        """Print a formatted summary of the configuration."""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("📋 DEEPRIRNET CONFIGURATION SUMMARY")
        print("="*80)
        
        for section, params in summary.items():
            print(f"\n{section}:")
            print("-" * len(section))
            for key, value in params.items():
                print(f"  {key:<20}: {value}")
        
        print("\n" + "="*80 + "\n")


def get_default_config() -> Config:
    """Get the default configuration with validation."""
    return Config()


def get_quick_config(preset: str = "balanced") -> Config:
    """
    Get predefined configurations for common use cases.
    
    Args:
        preset: Configuration preset name
            - "fast": Quick training for testing
            - "balanced": Good default for most cases  
            - "high_quality": Best results, slower training
            - "large_scale": For big datasets and complex rooms
    
    Returns:
        Configured Config object
    """
    config = Config()
    
    if preset == "fast":
        # Fast training for development/testing
        config.model.hidden_dim = 256
        config.model.num_lstm_layers = 3
        config.training.epochs = 20
        config.training.fine_tune_epochs = 10
        config.training.batch_size = 16
        config.data.source_dataset_size = 50
        config.data.target_dataset_size = 10
        
    elif preset == "balanced":
        # Default settings (already set)
        pass
        
    elif preset == "high_quality":  
        # Best quality, slower training
        config.model.hidden_dim = 1024
        config.model.num_lstm_layers = 8
        config.model.dropout = 0.1
        config.training.epochs = 100
        config.training.fine_tune_epochs = 50
        config.training.learning_rate = 5e-4
        config.data.source_dataset_size = 500
        config.data.target_dataset_size = 50
        
    elif preset == "large_scale":
        # For complex rooms and large datasets
        config.model.hidden_dim = 1024
        config.model.num_lstm_layers = 10
        config.model.T = 1024
        config.data.T = 1024
        config.training.epochs = 200
        config.training.batch_size = 4  # Larger sequences need smaller batches
        config.data.source_dataset_size = 1000
        config.data.target_dataset_size = 100
        config.data.reflection_order = 2  # More detailed acoustics
        
    else:
        raise ValueError(f"Unknown preset: {preset}. "
                        f"Choose from: fast, balanced, high_quality, large_scale")
    
    # Re-validate after changes
    config.__post_init__()
    
    return config