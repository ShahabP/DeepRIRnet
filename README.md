# DeepRIRNet — geometry-aware Room Impulse Response prediction

This repository implements **DeepRIRNet**, a geometry-aware transfer-learning model for Room Impulse Response (RIR) estimation (paper attached). The model factorizes a geometry encoder and a temporal decoder (two-layer LSTM) and uses physics-informed losses (sparsity + energy decay) during training for realistic RIRs.

Paper: *A Novel Transfer Learning Approach for Room Impulse Response Estimation...* (see `paper/RIRTransfer.pdf`). :contentReference[oaicite:1]{index=1}

## Quick start

1. Create a virtual env and install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
