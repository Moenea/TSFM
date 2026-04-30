"""
STAConvBiLSTM: Spatiotemporal Attention-based CNN-BiLSTM
Reproduced from: "Spatiotemporal attention mechanism-based deep network
for critical parameters prediction in chemical process"
(Yuan et al., Process Safety and Environmental Protection, 2021)

Architecture (Section 4, Fig. 4):
1. 1D-CNN: spatial feature extraction at each time step (no pooling)
2. Spatial Attention: weight CNN features by relevance to target (Eq. 17-19)
3. BiLSTM: temporal dependency modeling on attended features (Eq. 9-11, 20)
4. Temporal Attention: weight time steps by relevance to target (Eq. 21-23)
5. Dense: project context vector to prediction (Eq. 24)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        d_hidden = configs.d_model       # BiLSTM hidden size per direction
        d_cnn = 64                        # CNN feature dimension
        d_bi = 2 * d_hidden               # BiLSTM output (bidirectional)

        # 1D-CNN: spatial feature extraction (no pooling, paper Section 3.1)
        self.cnn = nn.Sequential(
            nn.Conv1d(configs.enc_in, d_cnn, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Spatial Attention (paper Section 4.3, Eq. 17-19)
        # Per-timestep: weight features by relevance to target
        self.sa_net = nn.Sequential(
            nn.Linear(d_cnn, d_cnn),
            nn.Tanh(),
            nn.Linear(d_cnn, d_cnn),
        )

        # BiLSTM (paper Section 3.2, Eq. 9-11)
        self.bilstm = nn.LSTM(
            input_size=d_cnn,
            hidden_size=d_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # Temporal Attention (paper Section 4.3, Eq. 21-23)
        # Weight time steps → context vector
        self.ta_linear = nn.Linear(d_bi, d_bi)
        self.ta_score = nn.Linear(d_bi, 1)

        # Dense output (paper Eq. 24)
        self.dense = nn.Linear(d_bi, self.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B = x_enc.shape[0]

        # 1D-CNN: [B, enc_in, seq_len] → [B, d_cnn, seq_len] → [B, seq_len, d_cnn]
        x = self.cnn(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        # Spatial Attention: weight features at each timestep (Eq. 17-19)
        sa_scores = self.sa_net(x)                       # [B, seq_len, d_cnn]
        sa_weights = F.softmax(sa_scores, dim=-1)        # softmax across features
        x = x * sa_weights                               # [B, seq_len, d_cnn]

        # BiLSTM: temporal modeling (Eq. 20)
        H, _ = self.bilstm(x)                            # [B, seq_len, 2*d_hidden]

        # Temporal Attention: weight time steps (Eq. 21-23)
        ta_scores = self.ta_score(torch.tanh(self.ta_linear(H)))  # [B, seq_len, 1]
        ta_weights = F.softmax(ta_scores, dim=1)                  # softmax across time
        context = (H * ta_weights).sum(dim=1)                     # [B, 2*d_hidden]

        # Dense: context → prediction (Eq. 24)
        out = self.dense(context)                         # [B, pred_len * c_out]
        return out.reshape(B, self.pred_len, self.c_out)
