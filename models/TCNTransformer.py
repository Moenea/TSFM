"""
TCN-Transformer Deep Network

Based on: "TCN-Transformer Deep Network with Random Forest for Prediction
of the Chemical Synthetic Ammonia Process"
(Dong et al., ACS Omega 2025, 10, 2269-2279)

Architecture:
  1. Input Embedding (DataEmbedding)
  2. Self-Attention Enhancement Block (MHA + FFN)
  3. TCN Residual Blocks (dilated causal convolutions)
  4. Transformer Encoder (MHA + FFN with residual connections)
  5. Transformer Decoder (masked self-attention + cross-attention + FFN)
  6. Linear Projection → prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from layers_mytimexer.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers_mytimexer.SelfAttention_Family import FullAttention, AttentionLayer
from layers_mytimexer.Embed import DataEmbedding


# ---------------------------------------------------------------------------
# TCN components
# ---------------------------------------------------------------------------

class TCNResidualBlock(nn.Module):
    """
    A single TCN residual block with two dilated causal convolutions,
    weight normalization, ReLU activation, and dropout.
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        )
        self.pad1 = nn.ConstantPad1d((padding, 0), 0.0)

        self.conv2 = weight_norm(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        )
        self.pad2 = nn.ConstantPad1d((padding, 0), 0.0)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, C, L]"""
        residual = x

        out = self.pad1(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network: a stack of residual blocks with
    exponentially increasing dilation (1, 2, 4, ...).
    """

    def __init__(self, channels, kernel_size=3, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                TCNResidualBlock(channels, kernel_size, dilation, dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """x: [B, C, L]  →  [B, C, L]"""
        return self.network(x)


# ---------------------------------------------------------------------------
# Self-Attention Enhancement Block
# ---------------------------------------------------------------------------

class SelfAttentionEnhancement(nn.Module):
    """
    Applies multi-head self-attention followed by a position-wise FFN
    to strengthen the information relationship between input features.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, L, d_model]  →  [B, L, d_model]"""
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """
    TCN-Transformer Deep Network.

    Follows the same forward interface as other models in this framework:
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec) → [B, pred_len, c_out]
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        d_model = configs.d_model
        d_ff = configs.d_ff
        n_heads = configs.n_heads
        e_layers = configs.e_layers
        d_layers = configs.d_layers
        dropout = configs.dropout
        activation = configs.activation

        # ---- Encoder side ----

        # Input embedding (TokenEmbedding + PositionalEmbedding + TimeFeatureEmbedding)
        self.enc_embedding = DataEmbedding(
            configs.enc_in, d_model, configs.embed, configs.freq, dropout
        )

        # Self-Attention Enhancement Block
        self.sa_enhance = SelfAttentionEnhancement(
            d_model, n_heads, d_ff, dropout
        )

        # TCN Block (3 residual blocks with dilated causal convolutions)
        self.tcn = TCN(d_model, kernel_size=3, num_layers=3, dropout=dropout)

        # Transformer Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model, n_heads,
                    ),
                    d_model, d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # ---- Decoder side ----

        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.dec_embedding = DataEmbedding(
                configs.dec_in, d_model, configs.embed, configs.freq, dropout
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(
                                True, configs.factor,
                                attention_dropout=dropout,
                                output_attention=False,
                            ),
                            d_model, n_heads,
                        ),
                        AttentionLayer(
                            FullAttention(
                                False, configs.factor,
                                attention_dropout=dropout,
                                output_attention=False,
                            ),
                            d_model, n_heads,
                        ),
                        d_model, d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for _ in range(d_layers)
                ],
                norm_layer=nn.LayerNorm(d_model),
                projection=nn.Linear(d_model, configs.c_out, bias=True),
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. Encoder embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)     # [B, seq_len, d_model]

        # 2. Self-Attention Enhancement
        enc_out = self.sa_enhance(enc_out)                   # [B, seq_len, d_model]

        # 3. TCN  (expects channel-first: [B, d_model, seq_len])
        enc_out = self.tcn(enc_out.transpose(1, 2)).transpose(1, 2)

        # 4. Transformer Encoder
        enc_out, _ = self.encoder(enc_out)                   # [B, seq_len, d_model]

        # 5. Decoder embedding + Transformer Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)      # [B, label_len+pred_len, d_model]
        dec_out = self.decoder(dec_out, enc_out)              # [B, label_len+pred_len, c_out]

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]   # [B, pred_len, c_out]
        return None
