# 📁 Configuration Examples

This directory contains ready-to-use configuration examples for different use cases and scenarios. Each example is thoroughly documented and optimized for specific requirements.

## 🚀 Available Examples

### 1. **Fast Development** (`fast_development.py`)
**Perfect for**: Code development, testing, rapid prototyping  
**Training time**: ~5 minutes  
**Memory usage**: Low (2-4GB)  
**Quality**: Basic, suitable for development

```python
from examples.fast_development import get_fast_development_config
config = get_fast_development_config()
```

### 2. **Production Quality** (`production_quality.py`)  
**Perfect for**: Deployment, research papers, high-quality applications  
**Training time**: ~2-4 hours  
**Memory usage**: Medium-High (8-12GB)  
**Quality**: Excellent, state-of-the-art

```python
from examples.production_quality import get_production_config
config = get_production_config()

# For music applications
from examples.production_quality import get_music_production_config  
config = get_music_production_config()
```

### 3. **Memory Efficient** (`memory_efficient.py`)
**Perfect for**: Limited GPU memory, older hardware, laptops  
**Training time**: ~30 minutes  
**Memory usage**: Very Low (2-4GB)  
**Quality**: Good with minimal resources

```python
from examples.memory_efficient import get_memory_efficient_config
config = get_memory_efficient_config()

# For CPU-only training
from examples.memory_efficient import get_cpu_only_config
config = get_cpu_only_config()
```

### 4. **Research Experiments** (`research_experiments.py`)
**Perfect for**: Academic research, ablation studies, reproducible experiments  
**Training time**: ~1-2 hours  
**Memory usage**: Medium (6-8GB)  
**Quality**: Research-grade with extensive analysis

```python
from examples.research_experiments import get_research_config
config = get_research_config()

# For ablation studies
from examples.research_experiments import get_ablation_study_configs
configs = get_ablation_study_configs()

# For reproducibility  
from examples.research_experiments import get_reproducibility_config
config = get_reproducibility_config()
```

## 🎯 How to Choose the Right Configuration

| Use Case | Example | GPU Memory | Training Time | Quality |
|----------|---------|------------|---------------|---------|
| **Quick Testing** | `fast_development` | 2-4GB | 5 min | Basic |
| **Balanced Training** | Built-in presets | 4-8GB | 20 min | Good |
| **Production Deployment** | `production_quality` | 8-12GB | 2-4 hours | Excellent |
| **Limited Resources** | `memory_efficient` | 2-4GB | 30 min | Good |
| **Research Paper** | `research_experiments` | 6-8GB | 1-2 hours | Research-grade |

## 🛠️ Using Configuration Examples

### Method 1: Import and Use Directly
```python
from examples.production_quality import get_production_config
from main import main

config = get_production_config()
main(config)
```

### Method 2: Customize After Loading
```python
from examples.fast_development import get_fast_development_config

config = get_fast_development_config()

# Customize specific parameters
config.training.learning_rate = 2e-3
config.model.hidden_dim = 512
config.data.source_dataset_size = 100

# Validate changes
config.__post_init__()
config.print_summary()
```

### Method 3: Use as Command Line Reference
```bash
# See the parameters in fast_development.py, then run:
python main.py --preset fast --lr 0.002 --hidden-dim 256 --epochs 15
```

## 🔧 Customization Guidelines

### Quick Modifications
```python
# Load base configuration
config = get_production_config()

# Common customizations:
config.training.learning_rate = 5e-4      # More conservative
config.model.hidden_dim = 1024            # Larger model  
config.data.source_dataset_size = 1000    # More training data
config.training.batch_size = 4            # Less memory usage

# Always re-validate after changes
config.__post_init__()
```

### Memory Optimization
```python
# If you get GPU memory errors:
config.training.batch_size //= 2          # Halve batch size
config.model.hidden_dim = min(512, config.model.hidden_dim)  # Cap model size
config.model.T = min(512, config.model.T)  # Shorter sequences
```

### Quality vs Speed Trade-offs
```python
# For faster training:
config.training.epochs //= 2              # Fewer epochs
config.model.num_lstm_layers = min(4, config.model.num_lstm_layers)

# For better quality:
config.training.epochs *= 2               # More epochs
config.model.hidden_dim *= 2              # Larger model
config.data.source_dataset_size *= 2      # More data
```

## 📊 Performance Expectations

### Training Time Estimates (on modern GPU)
- **Fast Development**: 5-10 minutes
- **Balanced (default)**: 20-30 minutes  
- **Production Quality**: 2-4 hours
- **Research Grade**: 1-3 hours
- **Large Scale**: 4-12 hours

### Memory Usage Guidelines
- **2-4GB**: `fast_development`, `memory_efficient`
- **4-8GB**: Default presets, most configs
- **8-12GB**: `production_quality`, larger models
- **12GB+**: `large_scale`, very long sequences

### Quality Expectations
- **Basic**: Good enough for development, testing algorithms
- **Good**: Suitable for most applications, decent RIR quality
- **Excellent**: Production-ready, publication-quality results
- **Research-grade**: State-of-the-art with comprehensive analysis

## 🚨 Common Issues and Solutions

### Memory Errors
```python
# Reduce memory usage:
config.training.batch_size = 4
config.model.hidden_dim = 256
config.model.T = 256
```

### Slow Training
```python
# Speed up training:
config.training.epochs = 30
config.data.source_dataset_size = 100
config.model.num_lstm_layers = 4
```

### Poor Quality Results
```python
# Improve quality:
config.model.hidden_dim = 1024
config.training.epochs = 100
config.data.source_dataset_size = 500
config.training.learning_rate = 5e-4  # More careful training
```

## 📝 Creating Custom Configurations

To create your own configuration example:

1. **Start with a similar existing example**
2. **Modify parameters for your use case**  
3. **Document the purpose and expected performance**
4. **Test thoroughly**
5. **Add validation and error checking**

```python
def get_my_custom_config():
    """My custom configuration for specific use case."""
    
    config = get_production_config()  # Start with good base
    
    # Customize for your needs
    config.data.fs = 44100           # Higher sampling rate
    config.model.T = 2048            # Longer sequences
    config.data.min_room_size = 10.0 # Larger rooms only
    
    # Always re-validate
    config.__post_init__()
    
    return config
```

Need help choosing or customizing a configuration? Check the [HYPERPARAMETER_GUIDE.md](../HYPERPARAMETER_GUIDE.md) for detailed guidance!