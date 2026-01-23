# GAN Baseline Implementation

This directory contains the implementation of a GAN-based baseline for RIR estimation, added for comparison with the DeepRIRNet transfer learning approach.

## Overview

The GAN baseline implements a generative adversarial network that learns to generate Room Impulse Responses (RIRs) from geometry features. This provides a fair comparison point to demonstrate the advantages of the transfer learning approach used in DeepRIRNet.

## Files Added

### Core Implementation

1. **`models/gan_baseline.py`** - GAN model architecture
   - `RIRGenerator`: Generates RIRs from geometry features + random noise
   - `RIRDiscriminator`: Discriminates between real and fake RIRs
   - `RIRGAN`: Wrapper class combining both networks

2. **`utils/train_gan.py`** - Training utilities
   - `train_gan()`: Standard GAN training with BCE loss
   - `train_gan_wgan_gp()`: Wasserstein GAN with gradient penalty (more stable)

3. **`utils/evaluate_gan.py`** - Evaluation utilities
   - `evaluate_gan()`: Compute MSE, LSD, ATE metrics
   - `evaluate_and_compare()`: Side-by-side comparison with DeepRIRNet

### Data Access

4. **`data/timit_loader.py`** - TIMIT database access
   - Handles TIMIT dataset (requires license)
   - **Free alternative**: Automatically uses LibriSpeech (CC BY 4.0)
   - Utilities for loading speech utterances for dereverberation experiments

### Examples

5. **`examples/gan_baseline_example.py`** - Complete usage example
   - Shows full pipeline: data loading → training → evaluation
   - Demonstrates TIMIT/LibriSpeech usage
   - Includes comparison with DeepRIRNet

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies added:
- `librosa>=0.10.0` - Speech processing
- `tqdm>=4.65.0` - Progress bars

### 2. Run Example

```bash
python examples/gan_baseline_example.py
```

This will:
- Initialize GAN and DeepRIRNet models
- Show training setup (update with your data paths)
- Display TIMIT license info and LibriSpeech alternative
- Provide complete workflow overview

### 3. Train GAN Baseline

```python
from models.gan_baseline import RIRGAN
from utils.train_gan import train_gan_wgan_gp
from torch.utils.data import DataLoader

# Initialize model
gan = RIRGAN(input_dim=6, hidden_dim=256, output_dim=4096, latent_dim=128)

# Train (using WGAN-GP for stability)
g_losses, d_losses = train_gan_wgan_gp(
    gan=gan,
    dataloader=train_loader,
    epochs=50,
    device="cuda",
    verbose=True
)

# Save
torch.save(gan.state_dict(), 'checkpoints/gan_baseline.pth')
```

### 4. Evaluate and Compare

```python
from utils.evaluate_gan import evaluate_and_compare

results = evaluate_and_compare(
    gan_model=gan,
    deeprir_model=deeprir,
    dataloader=test_loader,
    device="cuda"
)

# Results will show MSE, LSD, ATE for both models
```

## TIMIT Database Access

### License Information

**TIMIT** (LDC93S1) is a commercial dataset requiring a license:
- **Distributor**: Linguistic Data Consortium (LDC)
- **Price**: ~$250 USD (academic), ~$2500 (commercial)
- **Website**: https://catalog.ldc.upenn.edu/LDC93S1

### Free Alternative: LibriSpeech

Since TIMIT requires a license, we provide **automatic LibriSpeech support**:

```python
from data.timit_loader import TIMITLoader

# Initialize with free alternative
loader = TIMITLoader(use_alternative=True, sample_rate=16000)

# Automatically downloads LibriSpeech (CC BY 4.0 - free to use)
utterances = loader.get_test_utterances(num_utterances=100, max_duration=5.0)

# Returns list of (waveform, sample_rate) tuples
for waveform, sr in utterances:
    # Use for dereverberation experiments
    pass
```

**LibriSpeech advantages:**
- ✅ Free and open (CC BY 4.0 license)
- ✅ 1000 hours of speech data
- ✅ Automatic download via torchaudio
- ✅ High quality read speech

### Check License Status

```python
from data.timit_loader import check_timit_license

# Prints complete license information and alternatives
check_timit_license()
```

## Model Architecture

### GAN Generator

```
Input: [batch, input_dim] geometry features
       [batch, latent_dim] random noise

Architecture:
  Linear(input_dim + latent_dim → hidden_dim)
  ↓
  6× Deconvolution blocks (hidden_dim → 64 → 4096)
  ↓
  Final Conv1d
  ↓
Output: [batch, output_dim] RIR samples
```

### GAN Discriminator

```
Input: [batch, output_dim] RIR samples
       [batch, input_dim] geometry features (conditioning)

Architecture:
  5× Convolution blocks (1 → 64 channels, 4096 → 128 samples)
  ↓
  Global Average Pooling
  ↓
  Classifier (hidden_dim → 1)
  ↓
Output: [batch, 1] real/fake probability
```

## Training Options

### Option 1: Standard GAN (BCE Loss)

```python
from utils.train_gan import train_gan

g_losses, d_losses = train_gan(
    gan=gan,
    dataloader=train_loader,
    epochs=50,
    lr_g=0.0002,
    lr_d=0.0002,
    d_steps=1,  # Discriminator updates per iteration
    g_steps=1,  # Generator updates per iteration
    label_smoothing=0.1,  # Helps stability
    device="cuda"
)
```

### Option 2: WGAN-GP (Recommended)

More stable, doesn't require careful balancing:

```python
from utils.train_gan import train_gan_wgan_gp

g_losses, d_losses = train_gan_wgan_gp(
    gan=gan,
    dataloader=train_loader,
    epochs=50,
    lr=0.0001,
    lambda_gp=10.0,  # Gradient penalty coefficient
    n_critic=5,  # Critic updates per generator update
    device="cuda"
)
```

## Evaluation Metrics

All metrics match those used in the paper for fair comparison:

- **MSE**: Mean Squared Error between predicted and target RIR
- **LSD**: Log-Spectral Distance (dB) in frequency domain
- **ATE**: Arrival Time Error (samples) for direct path detection

```python
from utils.evaluate_gan import evaluate_gan

results = evaluate_gan(model=gan, dataloader=test_loader, device="cuda")

print(f"MSE: {results['MSE']:.6f} ± {results['MSE_std']:.6f}")
print(f"LSD: {results['LSD']:.4f} ± {results['LSD_std']:.4f} dB")
print(f"ATE: {results['ATE']:.2f} ± {results['ATE_std']:.2f} samples")
```

## Integration with Existing Code

The implementation preserves the existing repository structure:

```
DeepRIRnet/
├── models/
│   ├── deep_rir_net.py       # Original DeepRIRNet
│   └── gan_baseline.py        # NEW: GAN baseline
├── data/
│   ├── dataset.py             # Original RIR dataset
│   └── timit_loader.py        # NEW: TIMIT/LibriSpeech loader
├── utils/
│   ├── losses.py              # Original loss functions
│   ├── train_gan.py           # NEW: GAN training
│   └── evaluate_gan.py        # NEW: GAN evaluation
└── examples/
    └── gan_baseline_example.py # NEW: Usage example
```

## Dereverberation Experiments

Use generated RIRs for speech dereverberation:

```python
import numpy as np
from scipy import signal

# 1. Load clean speech
loader = TIMITLoader(use_alternative=True)
clean_speech, sr = loader.get_test_utterances(num_utterances=1)[0]

# 2. Estimate RIR
geometry = torch.tensor([[L, W, H, x_s, y_s, x_r, y_r]])
h_estimated = gan.generate(geometry).cpu().numpy()

# 3. Generate reverberant speech
reverberant = signal.convolve(clean_speech, h_estimated[0], mode='same')

# 4. Apply dereverberation method
# ... your dereverberation algorithm ...

# 5. Evaluate (PESQ, STOI, etc.)
# ... compute speech quality metrics ...
```

## Comparison with DeepRIRNet

Expected performance (qualitative):

| Method | Generalization | Training Data | Convergence |
|--------|---------------|---------------|-------------|
| **GAN Baseline** | Limited | Requires large dataset | Unstable |
| **DeepRIRNet (Source-only)** | Limited | Same as GAN | Stable |
| **DeepRIRNet (Transfer)** | ✅ Excellent | Small target domain | Fast |

The GAN baseline provides a fair comparison to demonstrate the advantages of:
1. Transfer learning for domain adaptation
2. Physics-informed regularizers
3. Selective layer freezing

## Repository Changes

All changes are **minimal and non-invasive**:
- ✅ No modifications to existing files
- ✅ New files in appropriate directories
- ✅ Consistent code style and documentation
- ✅ Dependencies added to requirements.txt


## Questions?

For issues or questions about the GAN baseline implementation:
1. Check `examples/gan_baseline_example.py` for complete usage
2. Review model architecture in `models/gan_baseline.py`
3. See training utilities in `utils/train_gan.py`

---

**Note**: This implementation follows standard GAN practices for RIR generation and provides a fair baseline for comparison with the proposed transfer learning approach.
