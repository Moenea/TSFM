import torch
import torch.nn as nn


class TailAwareMSELoss(nn.Module):
    """
    Tail-aware MSE for threshold-sensitive industrial forecasting.

    L = mean( w_t * (y_pred - y_true)^2 )

    Weight uses a sigmoid transition so that w_t increases monotonically
    as y_true approaches and exceeds the alarm threshold.
    w_t ∈ [1, 1 + alpha], with the midpoint (1 + alpha/2) at tau.

    mode='high':
        w_t = 1 + alpha * sigmoid( k * (y_true - tau_high) )
    mode='low':
        w_t = 1 + alpha * sigmoid( k * (tau_low - y_true) )
    mode='two_sided':
        w_t = 1 + max(
                  alpha * sigmoid( k * (y_true - tau_high) ),
                  alpha * sigmoid( k * (tau_low  - y_true) )
              )

    Parameters
    ----------
    alpha : float
        Maximum extra weight above 1. Upper bound of w_t is 1 + alpha.
    beta : float
        Transition steepness k = 1 / beta. Smaller beta → sharper boundary.
        For data in [0, 1] range, beta ~ 0.003–0.01 gives a smooth transition
        over ~±0.01 around tau.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 0.005,
        mode: str = 'high',
        tau_high: float | None = None,
        tau_low: float | None = None,
        reduction: str = 'mean',
    ) -> None:
        super().__init__()
        if reduction not in ('mean', 'none'):
            raise ValueError("reduction must be 'mean' or 'none'")
        if mode not in ('high', 'low', 'two_sided'):
            raise ValueError("mode must be 'high', 'low', or 'two_sided'")
        if beta <= 0:
            raise ValueError('beta must be > 0')

        self.alpha = float(alpha)
        self.k = 1.0 / float(beta)   # steepness
        self.mode = mode
        self.tau_high = None if tau_high is None else float(tau_high)
        self.tau_low = None if tau_low is None else float(tau_low)
        self.reduction = reduction

        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        if self.mode == 'high' and self.tau_high is None:
            raise ValueError('tau_high is required when mode="high"')
        if self.mode == 'low' and self.tau_low is None:
            raise ValueError('tau_low is required when mode="low"')
        if self.mode == 'two_sided':
            if self.tau_low is None or self.tau_high is None:
                raise ValueError('tau_low and tau_high are required when mode="two_sided"')

    def _weight_high(self, y_true: torch.Tensor) -> torch.Tensor:
        tau = torch.as_tensor(self.tau_high, dtype=y_true.dtype, device=y_true.device)
        return 1.0 + self.alpha * torch.sigmoid(self.k * (y_true - tau))

    def _weight_low(self, y_true: torch.Tensor) -> torch.Tensor:
        tau = torch.as_tensor(self.tau_low, dtype=y_true.dtype, device=y_true.device)
        return 1.0 + self.alpha * torch.sigmoid(self.k * (tau - y_true))

    def _weights(self, y_true: torch.Tensor) -> torch.Tensor:
        if self.mode == 'high':
            return self._weight_high(y_true)
        if self.mode == 'low':
            return self._weight_low(y_true)

        tau_l = torch.as_tensor(self.tau_low,  dtype=y_true.dtype, device=y_true.device)
        tau_h = torch.as_tensor(self.tau_high, dtype=y_true.dtype, device=y_true.device)
        w_h = self.alpha * torch.sigmoid(self.k * (y_true - tau_h))
        w_l = self.alpha * torch.sigmoid(self.k * (tau_l  - y_true))
        return 1.0 + torch.maximum(w_h, w_l)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(f'shape mismatch: y_pred{y_pred.shape} vs y_true{y_true.shape}')

        weights = self._weights(y_true)
        loss_elem = weights * (y_pred - y_true) ** 2

        if self.reduction == 'none':
            return loss_elem
        return torch.mean(loss_elem)
