#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from escape_artist.envs.escape_env import EscapeEnv, EnvConfig

def build_env(size: Tuple[int,int], traps_pct: float, slip: float, r_safe: int, seed: int):
    cfg = EnvConfig(
        size=size, start=(0,0), goal=(size[1]-1, size[0]-1),
        traps_pct=traps_pct, r_safe=r_safe, slip=slip,
        lethal_traps=True, max_steps=200,
        layout_mode="per_episode", seed=seed, obs_mode="pos",
    )
    return EscapeEnv(cfg)

def render_grid_image(env: EscapeEnv):
    grid = env.grid.copy()
    h, w = grid.shape
    img = np.zeros((h, w), dtype=float)
    img[grid == 1] = 0.6  # TRAP
    img[grid == 2] = 0.2  # GOAL
    return img

def main():
    ap = argparse.ArgumentParser(description="Create a montage of random layouts")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--size", type=int, nargs=2, default=[10, 10], help="H W")
    ap.add_argument("--traps_pct", type=float, default=0.10)
    ap.add_argument("--slip", type=float, default=0.1)
    ap.add_argument("--r_safe", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="assets/layout_montage.png")
    args = ap.parse_args()

    env = build_env(tuple(args.size), args.traps_pct, args.slip, args.r_safe, args.seed)

    fig, axes = plt.subplots(args.rows, args.cols, figsize=(4*args.cols, 4*args.rows))
    axes = np.atleast_2d(axes)

    for i in range(args.rows * args.cols):
        r, c = divmod(i, args.cols)
        # new episode ⇒ new layout (per_episode mode)
        env.reset()
        img = render_grid_image(env)
        ax = axes[r, c]
        ax.imshow(img, origin="upper")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Layout {i+1}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Random layouts (traps=X, goal=*)", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved montage: {out_path}")

if __name__ == "__main__":
    main()
