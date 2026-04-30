"""
GTProgerV13: V11 + Recency-weighted Patch Embedding
Based on V11, adding learnable recency weights to patch embeddings:
- Each patch position has a learnable importance weight (initialized: near > far)
- Endogenous and exogenous embeddings both use recency weighting
- Closer-to-present patches are initially weighted higher → better trend sensitivity
- Weights are learnable: model can adjust if distant patches are also valuable
- Retains all V11 components: patch cross-attention, dual encoder, decomp, FFT, per-horizon gate
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_mytimexer.Embed import PositionalEmbedding


# ========================== Prediction Head ==========================

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class FreqDecomp(nn.Module):
    def __init__(self, top_k=2, shrink_lambd=0.01):
        super().__init__()
        self.top_k = top_k
        self.shrink_lambd = shrink_lambd

    def forward(self, x):
        x_freq = torch.fft.rfft(x, dim=-1)
        freq_bins = x_freq.shape[-1]
        k = min(self.top_k, freq_bins)
        trend_freq = torch.zeros_like(x_freq)
        trend_freq[..., :k] = x_freq[..., :k]
        trend_real = F.softshrink(trend_freq.real, lambd=self.shrink_lambd)
        trend_imag = F.softshrink(trend_freq.imag, lambd=self.shrink_lambd)
        trend_freq = torch.complex(trend_real, trend_imag)
        trend = torch.fft.irfft(trend_freq, n=x.shape[-1], dim=-1)
        return x, trend


# ========================== Series Decomposition ==========================

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        seasonal = x - moving_mean
        return seasonal, moving_mean


# ========================== DSAttention ==========================

class DSAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        scale = 1. / (E ** 0.5)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if tau is not None:
            scores = scores * tau.view(B, 1, 1, 1)
        if delta is not None:
            scores = scores + delta.view(B, 1, 1, 1)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class DSAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, mask_flag=False, factor=5,
                 attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.n_heads = n_heads
        d_keys = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.inner_attention = DSAttention(mask_flag, factor, attention_dropout, output_attention)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, keys.shape[1], H, -1)
        values = self.value_projection(values).view(B, values.shape[1], H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class TauDeltaLearner(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.tau_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.delta_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x):
        tau = self.tau_proj(x.std(dim=1)).exp()
        delta = self.delta_proj(x.mean(dim=1))
        return tau, delta


# ========================== Patch Embeddings (no global token) ==========================

class RecencyPatchEmbedding(nn.Module):
    """
    Patch embedding with learnable recency weights.
    Closer-to-present patches are initialized with higher weights.
    Weights are learnable and shared across all samples.
    """
    def __init__(self, d_model, patch_len, max_patches, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable per-patch recency weight, initialized: near > far
        # sigmoid(raw) maps to (0, 1); raw initialized so sigmoid gives linear ramp
        raw_init = torch.zeros(1, 1, max_patches, 1)
        for i in range(max_patches):
            # sigmoid(x) = (i+1)/max_patches => x = logit((i+1)/max_patches)
            p = (i + 1) / (max_patches + 1)  # avoid 0 and 1
            raw_init[0, 0, i, 0] = math.log(p / (1 - p))  # logit
        self.recency_raw = nn.Parameter(raw_init)

    def forward(self, x):
        """
        x: [B, n_vars, seq_len]
        Returns: [B, n_vars, patch_num, d_model]
        """
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        B, V, P, PL = x.shape
        x = x.reshape(B * V, P, PL)
        x = self.value_embedding(x) + self.position_embedding(x)
        x = x.reshape(B, V, P, -1)

        # Apply recency weighting: sigmoid ensures (0, 1) range
        recency = torch.sigmoid(self.recency_raw[:, :, :P, :])  # [1, 1, P, 1]
        x = x * recency

        return self.dropout(x)


# ========================== Patch Cross-Attention Encoder ==========================

class PatchCrossDecompEncoderLayer(nn.Module):
    """
    EncoderLayer with:
    1. DSAttention self-attention on endogenous patches
    2. Decompose → extract trend
    3. Cross-attention: each endogenous patch attends to ALL exogenous patches
    4. FFN + decompose → accumulate trend
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", decomp_kernel=3):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.decomp1 = SeriesDecomp(decomp_kernel)
        self.decomp2 = SeriesDecomp(decomp_kernel)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, ex_patches, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        x:          [B, patch_num, d_model] — endogenous patches (single var, already reshaped)
        ex_patches: [B, n_exo * patch_num, d_model] — all exogenous patches flattened
        """
        # 1. Self-attention + decompose
        x = x + self.dropout(self.self_attention(
            x, x, x, attn_mask=x_mask, tau=tau, delta=delta)[0])
        x, trend1 = self.decomp1(x)

        # 2. Cross-attention: endogenous patches → all exogenous patches
        x_cross = self.dropout(self.cross_attention(
            x, ex_patches, ex_patches, attn_mask=cross_mask, tau=None, delta=None)[0])
        x = x + x_cross
        x = self.norm1(x)

        # 3. FFN + decompose
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = x + y
        x, trend2 = self.decomp2(x)
        x = self.norm2(x)

        trend = trend1 + trend2
        return x, trend


class PatchCrossDecompEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, ex_patches, x_mask=None, cross_mask=None, tau=None, delta=None):
        trend_accum = torch.zeros_like(x)
        for layer in self.layers:
            x, trend = layer(x, ex_patches, x_mask=x_mask, cross_mask=cross_mask,
                             tau=tau, delta=delta)
            trend_accum = trend_accum + trend
        if self.norm is not None:
            x = self.norm(x)
        return x, trend_accum


def _build_patch_cross_encoder(configs, decomp_kernel=3):
    return PatchCrossDecompEncoder(
        [PatchCrossDecompEncoderLayer(
            DSAttentionLayer(configs.d_model, configs.n_heads,
                mask_flag=False, factor=configs.factor,
                attention_dropout=configs.dropout, output_attention=False),
            DSAttentionLayer(configs.d_model, configs.n_heads,
                mask_flag=False, factor=configs.factor,
                attention_dropout=configs.dropout, output_attention=False),
            configs.d_model, configs.d_ff,
            dropout=configs.dropout, activation=configs.activation,
            decomp_kernel=decomp_kernel,
        ) for _ in range(configs.e_layers)],
        norm_layer=nn.LayerNorm(configs.d_model)
    )


# ========================== Model ==========================

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_len_fine = max(configs.patch_len // 2, 1)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        self.n_exo = configs.enc_in - 1 if configs.features == 'MS' else 0

        self.patch_num_coarse = configs.seq_len // self.patch_len
        self.patch_num_fine = configs.seq_len // self.patch_len_fine

        # Endogenous patch embeddings with recency weighting
        self.en_embedding_coarse = RecencyPatchEmbedding(configs.d_model, self.patch_len, self.patch_num_coarse, configs.dropout)
        self.en_embedding_fine = RecencyPatchEmbedding(configs.d_model, self.patch_len_fine, self.patch_num_fine, configs.dropout)

        # Exogenous patch embeddings with recency weighting
        self.ex_embedding_coarse = RecencyPatchEmbedding(configs.d_model, self.patch_len, self.patch_num_coarse, configs.dropout)
        self.ex_embedding_fine = RecencyPatchEmbedding(configs.d_model, self.patch_len_fine, self.patch_num_fine, configs.dropout)

        # Tau/delta learners
        self.tau_delta_coarse = TauDeltaLearner(configs.d_model, configs.seq_len)
        self.tau_delta_fine = TauDeltaLearner(configs.d_model, configs.seq_len)

        # Encoders with patch cross-attention
        self.encoder_coarse = _build_patch_cross_encoder(configs, decomp_kernel=3)
        self.encoder_fine = _build_patch_cross_encoder(configs, decomp_kernel=3)

        self.freq_decomp = FreqDecomp(top_k=2, shrink_lambd=0.01)

        # Total tokens: coarse_patches + fine_patches (no global token)
        total_tokens = self.patch_num_coarse + self.patch_num_fine
        head_nf = configs.d_model * total_tokens
        head_vars = 1 if configs.features == 'MS' else configs.enc_in

        self.head_level = FlattenHead(head_vars, head_nf, configs.pred_len,
                                      head_dropout=configs.dropout)
        self.head_trend = FlattenHead(head_vars, head_nf, configs.pred_len,
                                      head_dropout=configs.dropout)

        self.gate = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.GELU(),
            nn.Linear(configs.d_model // 2, configs.pred_len),
            nn.Sigmoid()
        )

    def _encode(self, x_enc, x_mark_enc, multi):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-3)
            x_enc /= stdev
        else:
            means, stdev = None, None

        if multi:
            en_input = x_enc.permute(0, 2, 1)  # [B, enc_in, seq_len]
            ex_input = en_input  # all vars are both endo and exo
        else:
            en_input = x_enc[:, :, -1:].permute(0, 2, 1)  # [B, 1, seq_len]
            ex_input = x_enc[:, :, :-1].permute(0, 2, 1)  # [B, n_exo, seq_len]

        B = en_input.shape[0]

        # --- Coarse branch ---
        en_coarse = self.en_embedding_coarse(en_input)  # [B, 1, patch_num_c, d_model]
        ex_coarse = self.ex_embedding_coarse(ex_input)  # [B, n_exo, patch_num_c, d_model]

        # Flatten endogenous: [B, patch_num_c, d_model]
        en_c = en_coarse.squeeze(1)  # [B, patch_num_c, d_model] (n_vars=1 for MS)
        # Flatten exogenous: [B, n_exo * patch_num_c, d_model]
        n_exo = ex_coarse.shape[1]
        ex_c = ex_coarse.reshape(B, n_exo * self.patch_num_coarse, -1)

        tau_c, delta_c = self.tau_delta_coarse(en_c)
        seasonal_c, trend_c = self.encoder_coarse(en_c, ex_c, tau=tau_c, delta=delta_c)

        # --- Fine branch ---
        en_fine = self.en_embedding_fine(en_input)
        ex_fine = self.ex_embedding_fine(ex_input)

        en_f = en_fine.squeeze(1)
        ex_f = ex_fine.reshape(B, n_exo * self.patch_num_fine, -1)

        tau_f, delta_f = self.tau_delta_fine(en_f)
        seasonal_f, trend_f = self.encoder_fine(en_f, ex_f, tau=tau_f, delta=delta_f)

        # Concatenate branches: [B, patch_num_c + patch_num_f, d_model]
        seasonal_out = torch.cat([seasonal_c, seasonal_f], dim=1)
        trend_out = torch.cat([trend_c, trend_f], dim=1)

        # Reshape to [B, n_vars, d_model, total_patches]
        seasonal_out = seasonal_out.unsqueeze(1).permute(0, 1, 3, 2)
        trend_out = trend_out.unsqueeze(1).permute(0, 1, 3, 2)

        return seasonal_out, trend_out, means, stdev

    def _decode(self, seasonal_out, trend_out, means, stdev, multi):
        level_pred = self.head_level(seasonal_out).permute(0, 2, 1)
        trend_pred = self.head_trend(trend_out).permute(0, 2, 1)

        combined = seasonal_out + trend_out
        gate_in = combined.mean(dim=-1).mean(dim=1)
        gate = self.gate(gate_in).unsqueeze(-1)

        out = gate * level_pred + (1.0 - gate) * trend_pred

        if self.use_norm:
            if multi:
                out = out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                out = out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            else:
                out = out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
                out = out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_gate=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                seasonal, trend, means, stdev = self._encode(x_enc, x_mark_enc, multi=True)
                dec_out = self._decode(seasonal, trend, means, stdev, multi=True)
            else:
                seasonal, trend, means, stdev = self._encode(x_enc, x_mark_enc, multi=False)
                dec_out = self._decode(seasonal, trend, means, stdev, multi=False)
            return dec_out[:, -self.pred_len:, :]
        return None
