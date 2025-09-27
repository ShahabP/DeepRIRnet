# DeepRIRNet

PyTorch implementation of DeepRIRNet (geometry-aware RIR prediction).

Features:
- Geometry-aware input projection → temporal embedding via MLP + learnable time-basis
- Two-layer LSTM decoder
- Loss = alpha*MSE + beta*LSD + lambda1*NormalizedEarlyL1 + lambda2*DecayPenalty
- Pretraining + fine-tuning utilities

## Installation

```bash
git clone <repo_url>
cd DeepRIRNet
pip install -r requirements.txt


---

DeepRIRNet/
├── README.md
├── requirements.txt
├── setup.py
├── deep_rirnet/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   └── utils.py
└── examples/
    └── run_dummy.py



python examples/run_dummy.py

