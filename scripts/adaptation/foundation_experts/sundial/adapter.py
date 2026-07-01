# scripts/adaptation/foundation_experts/sundial/adapter.py
"""Sundial expert adapter. Sundial is a generative (flow-matching) TSFM; we use
it ZERO-SHOT (consistent with the Time-MoE finding that fine-tuning these general
TSFMs overfits this small single-variate signal). predict() writes window-aligned,
original-scale forecasts in the shared on-disk contract; the point forecast is the
mean over `num_samples` generative draws."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/aicode/sherwin/TSFM")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/adaptation/foundation_experts"))
from common import expert_io as io  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

EPS = 1e-5
DEFAULT_CKPT = str(ROOT / "checkpoints/sundial_local")


def _norm(ctx: torch.Tensor):
    mean = ctx.mean(dim=-1, keepdim=True)
    std = ctx.std(dim=-1, keepdim=True) + EPS
    return (ctx - mean) / std, mean, std


def _load_model(ckpt_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(ckpt_id_or_dir, trust_remote_code=True)
    return model.to(device)


def predict(args) -> None:
    src = args.ckpt_dir if (args.ckpt_dir and not args.zero_shot) else args.ckpt_id
    model = _load_model(str(src), args.device)
    model.eval()
    contexts, trues = io.iter_infer_windows(
        args.data_root, args.split_file, args.target,
        args.seq_len, args.pred_len, args.horizon)
    preds = np.empty((contexts.shape[0], args.horizon), dtype=np.float64)
    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, len(contexts), bs):
            ctx = torch.tensor(contexts[i:i + bs], dtype=torch.float32, device=args.device)
            normed, mean, std = _norm(ctx)
            out = model.generate(normed, max_new_tokens=args.horizon,
                                 num_samples=args.num_samples)   # (b, num_samples, horizon)
            fc = out.float().mean(dim=1)[:, -args.horizon:]       # sample mean -> (b, horizon)
            fc = fc * std + mean
            preds[i:i + bs] = fc.cpu().numpy()
    io.save_result(args.out_dir, preds, trues,
                   {"model": "Sundial", "ckpt": str(args.ckpt_id),
                    "num_samples": args.num_samples, "zero_shot": True,
                    "horizon": args.horizon})
    print(f"saved {preds.shape} -> {args.out_dir}")


def fit(args) -> None:
    # Fallback: Sundial is used zero-shot (fine-tuning general TSFMs overfits this
    # small single-variate dataset — see the Time-MoE study in WORKFLOW.md).
    Path(args.ckpt_dir or DEFAULT_CKPT).mkdir(parents=True, exist_ok=True)
    print("Sundial fit: zero-shot only (no fine-tune). predict() uses the pretrained weights.",
          flush=True)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fit", "predict"], required=True)
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--ckpt-id", default=DEFAULT_CKPT)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--zero-shot", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    (predict if args.mode == "predict" else fit)(args)
