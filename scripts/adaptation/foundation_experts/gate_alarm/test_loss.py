import numpy as np
import torch
from loss import soft_alarm, window_soft_or, lead_weights, alarm_aware_loss

LOW, HIGH = 0.18, 0.244


def test_soft_alarm_high_when_above_limit():
    fused = torch.tensor([[0.30, 0.30]])          # well above HIGH
    a = soft_alarm(fused, LOW, HIGH, tau_a=0.003)
    assert float(a.min()) > 0.9


def test_soft_alarm_low_when_inside_band():
    fused = torch.tensor([[0.21, 0.21]])          # inside band
    a = soft_alarm(fused, LOW, HIGH, tau_a=0.003)
    assert float(a.max()) < 0.1


def test_window_soft_or_fires_if_any_latter_step_alarms():
    a = torch.tensor([[0.0, 0.0, 0.0, 0.95]])     # only last step alarms
    A = window_soft_or(a, half_start=2)
    assert float(A[0]) > 0.9


def test_window_soft_or_ignores_early_steps():
    a = torch.tensor([[0.95, 0.95, 0.0, 0.0]])    # only early steps alarm
    A = window_soft_or(a, half_start=2)
    assert float(A[0]) < 0.1


def test_lead_weights_earlier_origin_gets_more_weight():
    # crossing at index 120; window origins 100 (early) and 118 (late).
    # onset_h=0.0 disables the post-onset gate so we test the earliness math alone
    # (real data crosses after t=30h; here the synthetic crossing is at t=6h).
    s = np.full(200, 0.21); s[120:] = 0.30
    w = lead_weights(s, np.array([100, 118]), LOW, HIGH, horizon=15, dt=0.05, onset_h=0.0)
    assert w[0] > w[1] >= 0.0 and w[0] <= 1.0


def test_lead_weights_zero_when_crossing_is_pre_onset():
    # with the real onset gate (30h), a t=6h crossing yields all-zero weights
    s = np.full(200, 0.21); s[120:] = 0.30
    w = lead_weights(s, np.array([100, 118]), LOW, HIGH, horizon=15, dt=0.05)
    assert (w == 0.0).all()


def test_loss_prefers_crossing_on_positives():
    # A positive window whose fused crosses -> lower loss than one that stays flat.
    y = torch.tensor([True]); clean = torch.tensor([True]); lead = torch.tensor([1.0])
    true = torch.full((1, 15), 0.30)
    cross = torch.full((1, 15), 0.30)             # fused crosses (matches true)
    flat = torch.full((1, 15), 0.21)              # fused stays inside band
    kw = dict(low=LOW, high=HIGH, tau_a=0.003, lambda_far=1.0,
              lambda_lead=1.0, lambda_mse=0.1, half_start=7)
    l_cross, _ = alarm_aware_loss(cross, true, y, clean, lead, **kw)
    l_flat, _ = alarm_aware_loss(flat, true, y, clean, lead, **kw)
    assert float(l_cross) < float(l_flat)


def test_loss_penalizes_false_alarm_on_clean_negative():
    y = torch.tensor([False]); clean = torch.tensor([True]); lead = torch.tensor([0.0])
    true = torch.full((1, 15), 0.21)              # true stays inside band
    cross = torch.full((1, 15), 0.30); flat = torch.full((1, 15), 0.21)
    kw = dict(low=LOW, high=HIGH, tau_a=0.003, lambda_far=1.0,
              lambda_lead=1.0, lambda_mse=0.0, half_start=7)
    l_cross, _ = alarm_aware_loss(cross, true, y, clean, lead, **kw)
    l_flat, _ = alarm_aware_loss(flat, true, y, clean, lead, **kw)
    assert float(l_cross) > float(l_flat)         # crossing on a clean negative is worse
