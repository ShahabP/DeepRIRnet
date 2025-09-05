import torch
import torch.nn as nn

class DeepRIRNet(nn.Module):
    def __init__(self, input_dim, T, hidden_dim=512, num_lstm_layers=6, dropout=0.2):
        super().__init__()
        self.T = T
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            for _ in range(num_lstm_layers)
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_lstm_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x_proj = self.input_proj(x)
        x_proj = x_proj.unsqueeze(1).repeat(1, self.T, 1)

        h = x_proj
        for lstm, norm in zip(self.lstm_layers, self.norm_layers):
            out, _ = lstm(h)
            out = norm(out + h)
            out = self.dropout(out)
            h = out

        return self.out_linear(h).squeeze(-1)
