"""Convert an alarm-aware softmax gate into a per-step hard expert route.

The alarm-aware gate is trained as a soft convex fusion.  For a diagnostic
Pareto comparison, this script selects the expert with the largest gate weight
at each horizon step, preserving the gate's learned context-dependent routing
without averaging away threshold-crossing excursions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def target_view(a: np.ndarray) -> np.ndarray:
    return a[:, :, 0] if a.ndim == 3 else a


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--expert", action="append", required=True,
                   help="expert result directory; order must match weights")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=5)
    args = p.parse_args()

    weights = np.load(args.weights)
    experts = [target_view(np.load(Path(d) / "pred.npy")) for d in args.expert]
    stack = np.stack([x[:, :args.horizon] for x in experts], axis=-1)
    if weights.shape[:2] != stack.shape[:2] or weights.shape[2] != stack.shape[2]:
        raise ValueError(f"weights {weights.shape} vs expert stack {stack.shape}")
    route = np.argmax(weights[:, :args.horizon], axis=-1)
    hard = np.take_along_axis(stack, route[..., None], axis=-1)[..., 0]

    base = target_view(np.load(Path(args.expert[0]) / "pred.npy")).copy()
    base[:, :args.horizon] = hard
    true = target_view(np.load(Path(args.expert[0]) / "true.npy"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "pred.npy", base[:, :, None].astype(np.float32))
    np.save(args.output_dir / "true.npy", true[:, :, None].astype(np.float32))
    np.save(args.output_dir / "route.npy", route.astype(np.int8))
    (args.output_dir / "meta.json").write_text(
        '{"method": "Gate-alarm-hard-route", "horizon": %d}\n' % args.horizon,
        encoding="utf-8")
    print(f"saved {args.output_dir} route fractions={np.bincount(route.ravel(), minlength=stack.shape[2]) / route.size}")


if __name__ == "__main__":
    main()
