"""Pure, testable alarm-aware combined-score loss for the XMEAS10 gate.
No I/O, no argparse — imported by fuse_gate_alarm.py and unit-tested standalone."""
from __future__ import annotations
import numpy as np
import torch

EPS = 1e-6


def soft_alarm(fused: torch.Tensor, low: float, high: float, tau_a: float) -> torch.Tensor:
    """Per-step differentiable +-3sigma crossing indicator, (B,H) -> (B,H) in [0,1]."""
    a = torch.sigmoid((fused - high) / tau_a) + torch.sigmoid((low - fused) / tau_a)
    return a.clamp(0.0, 1.0)


def window_soft_or(a: torch.Tensor, half_start: int) -> torch.Tensor:
    """Soft-OR over the latter-half steps -> per-window alarm prob (B,)."""
    a_half = a[:, half_start:]
    return 1.0 - torch.prod(1.0 - a_half, dim=1)


def lead_weights(true_series: np.ndarray, origins: np.ndarray, low: float, high: float,
                 horizon: int, dt: float, onset_h: float = 30.0) -> np.ndarray:
    """Per-window earliness weight in [0,1]: for windows whose forecast origin
    precedes the run's first post-onset +-3sigma crossing, weight rises the earlier
    the origin sits (normalized by the horizon length). Windows at/after the
    crossing or with no crossing get 0."""
    alarm = (true_series > high) | (true_series < low)
    idx = np.where(alarm)[0]
    idx = idx[idx * dt >= onset_h]
    if idx.size == 0:
        return np.zeros(len(origins), dtype=np.float64)
    cross = int(idx.min())
    lead = (cross - origins).astype(np.float64)           # steps ahead
    w = np.clip(lead / float(horizon), 0.0, 1.0)          # cap at one horizon
    w[origins >= cross] = 0.0
    return w


def alarm_aware_loss(fused, true, y_alarm, clean, lead_w, low, high, *,
                     tau_a, lambda_far, lambda_lead, lambda_mse, half_start):
    """fused,true: (B,H); y_alarm,clean: (B,) bool; lead_w: (B,) float in [0,1].
    Returns (scalar loss, component dict)."""
    a = soft_alarm(fused, low, high, tau_a)
    A = window_soft_or(a, half_start).clamp(EPS, 1.0 - EPS)      # (B,)
    y = y_alarm.float()
    pos = y_alarm
    neg = clean & (~y_alarm)
    elig = (pos | neg).float()
    bce = -(y * torch.log(A) + (1.0 - y) * torch.log(1.0 - A))
    bce = (bce * elig).sum() / elig.sum().clamp(min=1.0)
    negf = neg.float()
    far = (A * negf).sum() / negf.sum().clamp(min=1.0)
    posf = pos.float()
    lead = ((lead_w * A) * posf).sum() / posf.sum().clamp(min=1.0)
    mse = torch.mean((fused - true) ** 2)
    loss = bce + lambda_far * far - lambda_lead * lead + lambda_mse * mse
    return loss, {"bce": float(bce), "far": float(far), "lead": float(lead),
                  "mse": float(mse), "loss": float(loss)}
