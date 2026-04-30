"""
DiPCALSTM: Dynamic-inner PCA + LSTM for Key Alarm Variable Forecasting
Reproduced from: "A dynamic-inner LSTM prediction method for key alarm
variables forecasting in chemical process"
(Bai et al., Chinese Journal of Chemical Engineering, 2023)

Architecture:
- DiPCA: extracts the most auto-correlated (dynamic) principal components
  from high-dimensional process data (computed once from the first batch)
- Two-layer LSTM with decreasing hidden sizes (paper Table 3, Model 4)
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

        # DiPCA: reduce enc_in to n_components
        # Paper: 11 vars -> 5 components (~45%). We use similar ratio.
        self.n_components = max(configs.enc_in // 2 + 1, 2)

        # Loading matrix W computed lazily from first training batch
        self.register_buffer('dipca_W', torch.zeros(configs.enc_in, self.n_components))
        self.register_buffer('dipca_fitted', torch.tensor(False))

        # Two-layer LSTM with decreasing hidden sizes (paper Table 3/5)
        # Paper: LSTM(n_components -> 64) -> LSTM(64 -> 8) -> Linear(8 -> 1)
        self.hidden1 = configs.d_model
        self.hidden2 = max(configs.d_model // 8, 4)

        self.lstm1 = nn.LSTM(
            input_size=self.n_components,
            hidden_size=self.hidden1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=self.hidden1,
            hidden_size=self.hidden2,
            batch_first=True,
        )

        # Dense output layer (paper Eq. 10)
        self.dense = nn.Linear(self.hidden2, self.c_out)

    @torch.no_grad()
    def _fit_dipca(self, x):
        """
        Compute DiPCA loading matrix from batch data.
        DiPCA extracts latent variables with maximum autocorrelation
        (i.e., the most predictable/dynamic components).

        Paper Section 2 (Eqs. 9-16): iteratively extract loading vectors
        that maximize the predictability of latent variables from their past.
        """
        B, L, C = x.shape

        # Use consecutive timesteps within each sample (avoid cross-sample boundaries)
        X_future = x[:, 1:, :].reshape(-1, C)  # [B*(L-1), C]
        X_past = x[:, :-1, :].reshape(-1, C)   # [B*(L-1), C]
        X_all = x.reshape(-1, C)                # [B*L, C]

        # Standardize
        mean = X_all.mean(dim=0)
        std = X_all.std(dim=0).clamp(min=1e-8)
        X_f = (X_future - mean) / std
        X_p = (X_past - mean) / std
        X_norm = (X_all - mean) / std

        W = []
        for _ in range(self.n_components):
            # Cross-covariance: how well can future be predicted from past?
            # M = X_future^T @ X_past / N  (paper Eq. 12 simplified for lag=1)
            M = X_f.T @ X_p / X_f.shape[0]
            M_sym = (M + M.T) / 2  # Symmetrize for real eigendecomposition

            # Loading vector = eigenvector with largest eigenvalue
            eigenvalues, eigenvectors = torch.linalg.eigh(M_sym)
            w = eigenvectors[:, -1]  # [C]
            W.append(w)

            # Deflate: remove this component from data (paper Step 3-4)
            t_f = X_f @ w               # scores for future
            t_p = X_p @ w               # scores for past
            p_f = X_f.T @ t_f / (t_f @ t_f + 1e-8)
            p_p = X_p.T @ t_p / (t_p @ t_p + 1e-8)
            X_f = X_f - torch.outer(t_f, p_f)
            X_p = X_p - torch.outer(t_p, p_p)

        self.dipca_W.copy_(torch.stack(W, dim=1))  # [enc_in, n_components]
        self.dipca_fitted.fill_(True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, seq_len, enc_in]

        # Compute DiPCA loading matrix from first batch (then frozen)
        if not self.dipca_fitted.item():
            self._fit_dipca(x_enc)

        # DiPCA projection: reduce dimensionality
        x = x_enc @ self.dipca_W  # [B, seq_len, n_components]

        # Two-layer LSTM (paper Table 5: LSTM(n->64) -> LSTM(64->8))
        x, _ = self.lstm1(x)      # [B, seq_len, hidden1]
        x, _ = self.lstm2(x)      # [B, seq_len, hidden2]

        # Dense projection per timestep
        out = self.dense(x)        # [B, seq_len, c_out]

        # Take last pred_len timesteps as forecast
        return out[:, -self.pred_len:, :]
