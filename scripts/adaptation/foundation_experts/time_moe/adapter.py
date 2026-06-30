# scripts/adaptation/foundation_experts/time_moe/adapter.py
"""Time-MoE expert adapter. fit() few-shot fine-tunes on the same seed-2021
subset windows as Timer-XL; predict() writes window-aligned, original-scale
forecasts in the shared on-disk contract."""
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


def _norm(ctx: torch.Tensor):
    mean = ctx.mean(dim=-1, keepdim=True)
    std = ctx.std(dim=-1, keepdim=True) + EPS
    return (ctx - mean) / std, mean, std


def _load_model(ckpt_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(ckpt_id_or_dir, trust_remote_code=True)
    return model.to(device)


def predict(args) -> None:
    model = _load_model(args.ckpt_dir if (args.ckpt_dir and not args.zero_shot) else args.ckpt_id,
                        args.device)
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
            out = model.generate(normed, max_new_tokens=args.horizon)   # PROBE-confirmed
            fc = out[:, -args.horizon:]
            fc = fc * std + mean
            preds[i:i + bs] = fc.cpu().numpy()
    io.save_result(args.out_dir, preds, trues,
                   {"model": "Time-MoE", "ckpt": args.ckpt_id,
                    "zero_shot": bool(args.zero_shot), "horizon": args.horizon})
    print(f"saved {preds.shape} -> {args.out_dir}")


def fit(args) -> None:
    files, pairs = io.select_train_pairs(args.data_root, args.split_file, args.target,
                                         args.ratio, args.seq_len, args.pred_len)
    contexts, futures = io.windows_from_pairs(args.data_root, files, pairs, args.target,
                                              args.seq_len, args.horizon)
    model = _load_model(args.ckpt_id, args.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ctx_t = torch.tensor(contexts, dtype=torch.float32, device=args.device)   # (M,96)
    fut_t = torch.tensor(futures,  dtype=torch.float32, device=args.device)   # (M,15)
    n = len(ctx_t)
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=args.device); total = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            ctx, fut = ctx_t[idx], fut_t[idx]
            mean = ctx.mean(dim=-1, keepdim=True); std = ctx.std(dim=-1, keepdim=True) + EPS
            seq = torch.cat([ctx, fut], dim=1)            # (b,111) original scale
            normed = (seq - mean) / std                   # normalize WHOLE window by CONTEXT stats
            loss = model(input_ids=normed, labels=normed).loss
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(idx)
        print(f"epoch {epoch} loss {total/n:.4f}", flush=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.ckpt_dir)
    print(f"saved fine-tuned checkpoint -> {args.ckpt_dir}", flush=True)


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
    p.add_argument("--ckpt-id", default="Maple728/TimeMoE-50M")
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--zero-shot", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    (predict if args.mode == "predict" else fit)(args)
