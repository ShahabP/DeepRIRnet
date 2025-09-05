pip install torch torchaudio matplotlib numpy scipy

▶️ Usage

To train the model and run evaluation:

python main.py




---

## ⚙️ Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/yourusername/DeepRIRNet.git
cd DeepRIRNet
pip install -r requirements.txt



# DeepRIRNet

DeepRIRNet is a PyTorch implementation for generating and predicting room impulse responses (RIRs) using deep recurrent neural networks with physics-inspired regularizers.

---

DeepRIRNet/
│
├── data/
│ └── dataset.py # Dataset class
│
├── models/
│ └── deep_rir_net.py # Neural network model
│
├── utils/
│ ├── rir_generator.py # Synthetic RIR generation
│ ├── regularizers.py # Physics-inspired regularizers
│ └── losses.py # Hybrid loss function
│
├── train.py # Training loop
├── main.py # Main entry point (dataset creation, training, evaluation)
├── requirements.txt # Dependencies
└── README.md # Documentation
