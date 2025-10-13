# DeepRIRNet — Geometry-Aware Room Impulse Response Prediction

This repository implements **DeepRIRNet**, a geometry-aware transfer-learning model for Room Impulse Response (RIR) estimation. The model factorizes a geometry encoder and a temporal decoder (multi-layer LSTM) and uses physics-informed losses (sparsity + energy decay) during training for realistic RIRs.

## Features

- **Transfer Learning**: Pre-train on source domain, fine-tune on target domain
- **Physics-Informed Losses**: Sparsity and energy decay regularization
- **Configurable Architecture**: Easy to modify hyperparameters and model structure
- **Comprehensive Evaluation**: Training curves, prediction visualization, sensitivity analysis
- **Modern Codebase**: Type hints, documentation, proper package structure

## Installation

### Quick Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Development Setup

For development with code quality tools:
```bash
pip install -r requirements-dev.txt
```

## Usage

### Basic Training

Run with default configuration:
```bash
python main.py
```

### Custom Parameters

```bash
# Specify custom training parameters
python main.py --epochs 100 --batch-size 16 --lr 0.001 --device cuda

# See all available options
python main.py --help
```

### Configuration

All hyperparameters can be configured through `config.py`:

- **Model config**: Architecture parameters (hidden_dim, num_layers, dropout)
- **Training config**: Learning rate, batch size, epochs, loss weights
- **Data config**: Dataset sizes, room parameters, sampling frequency

## Model Architecture

DeepRIRNet consists of:

1. **Geometry Encoder**: Projects input features (room dims, absorption, positions) to hidden space
2. **Temporal Decoder**: Multi-layer LSTM with residual connections and layer normalization
3. **Output Layer**: Linear projection to generate RIR samples

### Input Features
- Room dimensions (3D)
- Absorption coefficient (1D) 
- Source position (3D)
- Microphone position (3D)

### Physics-Informed Losses
- **MSE Loss**: Time-domain reconstruction accuracy
- **Log-Spectral Distance**: Frequency-domain fidelity  
- **Sparsity Regularization**: Encourages realistic sparse RIRs
- **Energy Decay Regularization**: Enforces exponential decay characteristic

## Project Structure

```
DeepRIRnet/
├── README.md                    # This file
├── requirements.txt             # Production dependencies  
├── requirements-dev.txt         # Development dependencies
├── config.py                   # Configuration management
├── main.py                     # Main training script
├── train.py                    # Training utilities
├── utils.py                    # General utilities
├── data/                       # Data handling modules
│   └── dataset.py              # Dataset implementations
├── models/                     # Neural network models
│   └── deep_rir_net.py        # DeepRIRNet model
└── utils/                      # Specialized utilities
    ├── losses.py               # Loss functions
    ├── regularizers.py         # Regularization functions
    └── rir_generator.py        # RIR generation utilities
```

## Transfer Learning Pipeline

1. **Source Domain Pre-training**: Train on large dataset of rectangular rooms
2. **Layer Freezing**: Freeze early LSTM layers to retain learned features
3. **Target Domain Fine-tuning**: Adapt to new room geometries with smaller dataset

## Results and Analysis

The training script automatically generates:
- Training loss curves for both phases
- Prediction vs ground truth comparisons
- Model sensitivity analysis to absorption coefficients
- Saved model checkpoints with training metrics
