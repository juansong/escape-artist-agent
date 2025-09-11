#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_returns(run_dir: Path) -> np.ndarray:
    p = run_dir / "returns.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return np.load(p)

def main():
    '''
    Merge multiple `returns.npy` series (learning curves) into a single PNG.
    
    Args:
        "--runs DIR ...": each must contain "returns.npy".
        "--labels STR ...": one label per run.
        "--out PATH": output PNG (e.g., "assets/curve_*.png").
    Returns / Artifacts:
        Combined curves saved to "--out".
    '''
    ap = argparse.ArgumentParser(description="Combine learning curves from multiple runs")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run folders (each must contain returns.npy). Order = legend order.")
    ap.add_argument("--labels", nargs="+", required=True,
                    help="Legend labels (same length as --runs).")
    ap.add_argument("--out", type=str, default="assets/curve_medium_mc_mc-off_q.png",
                    help="Output PNG path.")
    args = ap.parse_args()

    if len(args.runs) != len(args.labels):
        raise SystemExit("len(--runs) must equal len(--labels)")

    plt.figure(figsize=(7.5, 4.5))
    for run, label in zip(args.runs, args.labels):
        r = load_returns(Path(run))
        xs = np.arange(1, len(r) + 1)
        plt.plot(xs, r, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.title("Learning Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
