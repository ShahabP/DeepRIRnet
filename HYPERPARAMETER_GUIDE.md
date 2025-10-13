# 🎛️ DeepRIRNet Hyperparameter Tuning Guide

This comprehensive guide will help you understand and adjust every hyperparameter in DeepRIRNet for optimal performance in your specific use case.

## 📚 Table of Contents

1. [Quick Start Presets](#quick-start-presets)
2. [Model Architecture Parameters](#model-architecture-parameters)
3. [Training Hyperparameters](#training-hyperparameters)
4. [Loss Function Weights](#loss-function-weights)
5. [Data Generation Parameters](#data-generation-parameters)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Performance Optimization](#performance-optimization)

## 🚀 Quick Start Presets

Use these presets to get started quickly:

```python
from config import get_quick_config

# For quick testing and development
config = get_quick_config("fast")           # ~5 min training

# Balanced performance (recommended starting point)  
config = get_quick_config("balanced")       # ~20 min training

# Best quality results
config = get_quick_config("high_quality")   # ~2 hours training

# Large-scale complex acoustics
config = get_quick_config("large_scale")    # ~8 hours training
```

## 🏗️ Model Architecture Parameters

### Hidden Dimension (`model.hidden_dim`)

**What it controls**: Internal representation capacity of the neural network.

| Value | Use Case | Memory | Quality | Training Time |
|-------|----------|---------|---------|---------------|
| 128-256 | Simple rooms, fast prototyping | Low | Basic | Fast |
| 512 | **Default - balanced performance** | Medium | Good | Medium |
| 1024 | Complex acoustics, high quality | High | Excellent | Slow |
| 2048+ | Research, very complex scenarios | Very High | Best | Very Slow |

**Tuning tips**:
- Start with 512, increase if underfitting, decrease if memory issues
- Double the value if loss plateaus and you have more data
- Halve if you see overfitting (training loss << validation loss)

### Number of LSTM Layers (`model.num_lstm_layers`)

**What it controls**: Temporal modeling depth and complexity.

| Layers | Best For | Pros | Cons |
|--------|----------|------|------|
| 3-4 | Small rooms, simple acoustics | Fast, stable | Limited complexity |
| 6 | **Default - most scenarios** | Good balance | - |
| 8-10 | Large halls, complex reverberation | Rich temporal modeling | Slow, overfitting risk |
| 12+ | Research only | Maximum expressiveness | Very slow, unstable |

**Signs you need more layers**: RIR tails look unrealistic, poor long-term decay
**Signs you need fewer layers**: Slow convergence, overfitting

### Dropout Rate (`model.dropout`)

**What it controls**: Regularization strength to prevent overfitting.

| Value | When to Use | Effect |
|-------|-------------|--------|
| 0.0-0.1 | Large datasets, underfitting | Minimal regularization |
| 0.2 | **Default - balanced** | Standard regularization |
| 0.3-0.4 | Small datasets, overfitting | Strong regularization |
| 0.5+ | Extreme overfitting | Very strong (may hurt performance) |

### Sequence Length (`model.T`)

**What it controls**: Length of generated RIR in samples.

**Calculate required length**:
```python
# Rule of thumb: capture 2-3x the RT60 time
RT60_seconds = 0.5  # Estimate your room's reverberation time
fs = 16000          # Sampling frequency
T_needed = int(3 * RT60_seconds * fs)

# Common values:
# Small rooms: 256-512 samples (16-32ms @ 16kHz)
# Medium rooms: 512-1024 samples (32-64ms @ 16kHz)  
# Large halls: 1024-2048+ samples (64ms+ @ 16kHz)
```

## 🎯 Training Hyperparameters

### Learning Rate (`training.learning_rate`)

**Most critical parameter for training success!**

| Value | When to Use | Training Behavior |
|-------|-------------|-------------------|
| 1e-4 | Conservative, large models | Slow but stable |
| 5e-4 | Careful approach | Good stability |
| 1e-3 | **Default starting point** | Balanced |
| 2e-3 | Aggressive, small models | Fast but risky |
| 5e-3+ | Only if loss stuck | High instability risk |

**Adaptive tuning strategy**:
```python
# Monitor loss behavior after 10-20 epochs:

if loss_oscillating_wildly:
    learning_rate /= 3     # Too high, reduce

elif loss_decreasing_steadily:
    pass                   # Perfect, keep current

elif loss_plateaued_early:
    learning_rate *= 2     # Too low, increase

elif loss_not_decreasing:
    # Check other parameters first!
```

### Batch Size (`training.batch_size`)

**Affects gradient stability and memory usage**:

| Size | GPU Memory | Gradient Quality | Speed |
|------|------------|------------------|-------|
| 4 | 2-4GB | Noisy but diverse | Slower |
| 8 | 4-8GB | **Default - balanced** | Good |
| 16 | 8-16GB | Stable, smooth | Fast |
| 32+ | 16GB+ | Very stable | Fastest |

**Memory estimate**:
```python
memory_gb = (hidden_dim * num_layers * batch_size * T * 4) / (1024**3)

# If memory > your_gpu_memory:
#   - Reduce batch_size first
#   - Then reduce hidden_dim or T
```

### Training Epochs

**Source domain (`training.epochs`)**:
- Small datasets (<100): 100-200 epochs
- Medium datasets (100-1000): 50-100 epochs ⭐
- Large datasets (1000+): 20-50 epochs

**Target domain (`training.fine_tune_epochs`)**:
- Should be 30-60% of source epochs
- Small target data: 10-20 epochs
- Medium target data: 20-40 epochs ⭐

## ⚖️ Loss Function Weights

**These are crucial for realistic RIR generation!**

### MSE Weight (`training.alpha`)

Controls time-domain accuracy:
```python
alpha = 1.0    # Default - balanced
alpha = 2.0    # Emphasize waveform accuracy  
alpha = 0.5    # Less strict on exact samples
```

### Log-Spectral Distance Weight (`training.beta`)

Controls frequency-domain quality:
```python
beta = 0.1     # Default - perceptually important
beta = 0.2     # More emphasis on spectral accuracy
beta = 0.05    # Less spectral constraint
```

### Physics-Informed Regularization

**Sparsity weight (`training.lambda_sparse`)**:
```python
# Encourages realistic sparse RIRs
lambda_sparse = 1e-4   # Default
lambda_sparse = 1e-3   # Stronger sparsity (if RIRs too dense)
lambda_sparse = 1e-5   # Weaker (if losing too much detail)
```

**Decay weight (`training.lambda_decay`)**:
```python
# Enforces exponential energy decay
lambda_decay = 1e-4    # Default  
lambda_decay = 1e-3    # Stronger decay enforcement
lambda_decay = 1e-5    # More flexible decay
```

**Decay rate (`training.rho`)**:
```python
# Controls how fast energy should decay
rho = 0.1      # Very reverberant halls
rho = 0.3      # Default - medium reverberation
rho = 0.7      # Absorptive rooms
```

## 🏠 Data Generation Parameters

### Room Size Range
```python
# Match your target application:
min_room_size = 3.0    # Small offices
max_room_size = 8.0    # Large offices

min_room_size = 5.0    # Classrooms  
max_room_size = 15.0   # Large classrooms

min_room_size = 10.0   # Concert halls
max_room_size = 40.0   # Large venues
```

### Absorption Coefficient Range
```python
# Acoustic interpretation:
min_absorption = 0.1   # Hard surfaces (concrete, glass)
max_absorption = 0.3   # → Very reverberant

min_absorption = 0.2   # Mixed surfaces  
max_absorption = 0.6   # → Moderate reverberation (default range)

min_absorption = 0.4   # Soft surfaces (carpet, drapes)
max_absorption = 0.9   # → Low reverberation
```

### Dataset Sizes

**Source domain** (for pre-training):
```python
source_dataset_size = 50      # Minimum viable
source_dataset_size = 150     # Default - good balance
source_dataset_size = 500     # Better feature learning
source_dataset_size = 1000+   # Research-grade
```

**Target domain** (for fine-tuning):
```python
# Transfer learning works with small target data!
target_dataset_size = 5       # Few-shot learning
target_dataset_size = 20      # Default - usually sufficient  
target_dataset_size = 50      # More robust
target_dataset_size = 100+    # Large target domain
```

## 🔧 Troubleshooting Guide

### Problem: Loss Not Decreasing

**Possible causes & solutions**:

1. **Learning rate too high**:
   ```python
   config.training.learning_rate = 1e-4  # Reduce by 3-10x
   ```

2. **Model too small**:
   ```python
   config.model.hidden_dim = 1024        # Double size
   config.model.num_lstm_layers = 8      # Add layers
   ```

3. **Loss weights imbalanced**:
   ```python
   # Try reducing regularization first
   config.training.lambda_sparse = 1e-5
   config.training.lambda_decay = 1e-5
   ```

### Problem: Overfitting

**Signs**: Training loss << validation loss

**Solutions**:
```python
# Increase regularization
config.model.dropout = 0.4
config.training.lambda_sparse = 1e-3
config.training.lambda_decay = 1e-3

# Get more data
config.data.source_dataset_size = 500

# Reduce model size
config.model.hidden_dim = 256
config.model.num_lstm_layers = 4
```

### Problem: Poor RIR Quality

**Unrealistic RIRs**:
```python
# Increase physics regularization
config.training.lambda_sparse = 5e-4   # More sparsity
config.training.lambda_decay = 5e-4    # Stronger decay
config.training.rho = 0.4              # Faster decay
```

**Poor spectral quality**:
```python
# Emphasize frequency domain
config.training.beta = 0.3             # Higher LSD weight
```

### Problem: Training Too Slow

**Speed optimizations**:
```python
# Reduce model size
config.model.hidden_dim = 256
config.model.num_lstm_layers = 4

# Shorter sequences  
config.model.T = 256
config.data.T = 256

# Larger batches (if memory allows)
config.training.batch_size = 16

# Fewer epochs
config.training.epochs = 30
```

### Problem: GPU Memory Issues

**Memory optimizations**:
```python
# First: reduce batch size
config.training.batch_size = 4

# Then: reduce model size
config.model.hidden_dim = 256

# Finally: shorter sequences
config.model.T = 256
```

## ⚡ Performance Optimization

### For Speed
```python
config = get_quick_config("fast")
config.training.batch_size = 16        # Larger batches
config.model.T = 256                   # Shorter RIRs
```

### For Quality
```python
config = get_quick_config("high_quality")
config.training.learning_rate = 5e-4   # More careful training
config.data.reflection_order = 2       # More detailed acoustics
```

### For Memory Efficiency
```python
config.training.batch_size = 4
config.model.hidden_dim = 256
config.model.num_lstm_layers = 4
```

## 🎯 Recommended Tuning Sequence

1. **Start with preset**:
   ```python
   config = get_quick_config("balanced")
   ```

2. **Quick validation run** (10 epochs):
   ```python
   config.training.epochs = 10
   # Check if loss decreases
   ```

3. **Adjust learning rate** based on loss curve:
   - Oscillating → Reduce LR
   - Flat → Increase LR  
   - Good → Keep current

4. **Scale up gradually**:
   - More epochs → More data → Bigger model

5. **Fine-tune loss weights** for RIR quality

6. **Final optimization** for your specific use case

## 📊 Monitoring Training

**Key metrics to watch**:
- Total loss should decrease smoothly
- MSE component (time-domain accuracy)  
- LSD component (spectral accuracy)
- Regularization terms (should be small but not zero)

**Good training signs**:
- Steady loss decrease for 20+ epochs
- Generated RIRs look realistic
- Energy decay visible in RIR plots
- Spectral content matches target

Remember: Start simple, validate quickly, scale up gradually! 🚀