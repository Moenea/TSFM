"""
CNNLSTM: CNN-LSTM for Process Fault Prognosis
Reproduced from: "A deep learning model for process fault prognosis"
(Arunthavanathan et al., Process Safety and Environmental Protection, 2021)

Architecture:
- 1D CNN (no pooling) for multivariate feature extraction
- LSTM for temporal sequence forecasting
- Dense layer for output projection
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout

        # 1D CNN feature extraction (no pooling, per paper Section 2.1)
        self.cnn = nn.Sequential(
            nn.Conv1d(self.enc_in, self.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # LSTM sequential network (paper Section 2.2)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Dense layer (paper Section 2.3, Eq. 10)
        self.dense = nn.Linear(self.hidden_size, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, seq_len, enc_in]

        # CNN feature extraction: need [B, C, L] for Conv1d
        x = x_enc.permute(0, 2, 1)  # [B, enc_in, seq_len]
        x = self.cnn(x)              # [B, hidden_size, seq_len]
        x = x.permute(0, 2, 1)       # [B, seq_len, hidden_size]

        # LSTM forecasting
        out, _ = self.lstm(x)         # [B, seq_len, hidden_size]

        # Dense projection per timestep
        out = self.dense(out)         # [B, seq_len, c_out]

        # Take last pred_len timesteps as forecast
        return out[:, -self.pred_len:, :]
