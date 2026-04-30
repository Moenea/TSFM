"""
LSTMGRU: Hybrid LSTM-GRU for Early Prediction of Abnormal Conditions
Reproduced from: "A method for the early prediction of abnormal conditions
in chemical processes combined with physical knowledge and the data-driven model"
(Liu et al., Journal of Loss Prevention in the Process Industries, 2023)

Architecture (Section 4.3.2, Fig. 10):
- LSTM layer → Dropout(0.4) → GRU layer → Dropout(0.3) → Dense output
- Same hidden size for both LSTM and GRU layers (paper uses 256)
- Activation: tanh (default for LSTM/GRU)
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

        # LSTM layer (paper Fig. 10, first recurrent layer)
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(configs.dropout)  # paper default: 0.4

        # GRU layer (paper Fig. 10, second recurrent layer)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(configs.dropout)  # paper default: 0.3

        # Dense output layer
        self.dense = nn.Linear(self.hidden_size, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, seq_len, enc_in]

        # LSTM → Dropout
        x, _ = self.lstm(x_enc)       # [B, seq_len, hidden_size]
        x = self.dropout1(x)

        # GRU → Dropout
        x, _ = self.gru(x)            # [B, seq_len, hidden_size]
        x = self.dropout2(x)

        # Dense projection per timestep
        out = self.dense(x)            # [B, seq_len, c_out]

        # Take last pred_len timesteps
        return out[:, -self.pred_len:, :]
