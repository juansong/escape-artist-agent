# escape_artist/utils/plotting.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- small utils -----------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_fig(fig: plt.Figure, out_path: Path):
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ----------------------------- core helpers -----------------------------

def compute_V_and_policy(Q: np.ndarray, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      V: (nS,) where V[s] = max_a Q[s, a]
      A: (nS,) greedy action indices with uniform tie-breaking
    """
    if rng is None:
        rng = np.random.default_rng()
    V = Q.max(axis=1)
    A = np.zeros(Q.shape[0], dtype=np.int32)
    for s, row in enumerate(Q):
        maxv = row.max()
        cand = np.flatnonzero(row == maxv)
        A[s] = rng.choice(cand)
    return V, A


# ----------------------------- public plotting API -----------------------------

def plot_learning_curve(returns: np.ndarray, out_dir: Path) -> Path:
    """
    Plot episodic returns vs. episodes.
    Saves to <out_dir>/figs/learning_curve.png and returns the path.
    """
    fig = plt.figure(figsize=(6, 4))
    xs = np.arange(1, len(returns) + 1)
    plt.plot(xs, returns)
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.title("Learning Curve")
    plt.grid(True, alpha=0.3)

    out_path = Path(out_dir) / "figs" / "learning_curve.png"
    _save_fig(fig, out_path)
    return out_path


def plot_value_heatmap_and_policy(env, Q: np.ndarray, out_dir: Path, title: str = "V(s) & Greedy Policy") -> Path:
    """
    Draw V(s) as a heatmap and overlay greedy-policy arrows over the env's current layout.

    Assumes:
      - env has attributes h, w, grid (with TRAP=1, GOAL=2)
    """
    h, w = env.h, env.w
    grid = env.grid.copy()
    V, A = compute_V_and_policy(Q)

    V_img = V.reshape(h, w)
    mask = (grid == 1)  # TRAP
    V_masked = np.ma.array(V_img, mask=mask)

    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(V_masked, origin="upper")
    plt.colorbar(im, shrink=0.8, label="V(s) = max_a Q(s,a)")

    # traps & goal markers
    ys, xs = np.where(grid == 1)
    plt.scatter(xs, ys, marker="X", s=40)  # traps
    gy, gx = np.argwhere(grid == 2)[0]
    plt.scatter([gx], [gy], marker="*", s=120)  # goal

    # greedy arrows
    # actions: 0=up,1=right,2=down,3=left
    dx = {0: 0, 1: 1, 2: 0, 3: -1}
    dy = {0: -1, 1: 0, 2: 1, 3: 0}
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 1:
                continue
            s = y * w + x
            a = int(A[s])
            plt.arrow(x, y, dx[a] * 0.3, dy[a] * 0.3, head_width=0.15, length_includes_head=True)

    plt.title(title)
    plt.xticks(range(w))
    plt.yticks(range(h))
    plt.gca().invert_yaxis()

    out_path = Path(out_dir) / "figs" / "value_heatmap_policy.png"
    _save_fig(fig, out_path)
    return out_path


def plot_greedy_rollout(env, Q: np.ndarray, greedy_rollout_fn, out_dir: Path, max_steps: Optional[int] = None) -> Path:
    """
    Plot a single greedy rollout trajectory over the env's current layout.

    Args:
      env: your EscapeEnv (needs .grid, .h, .w, .cfg.max_steps)
      Q:   tabular Q-table (nS x nA)
      greedy_rollout_fn: callable(env, Q, max_steps) -> (total_return: float, traj: [(x,y), ...])
      max_steps: optional rollout cap

    Saves to <out_dir>/figs/greedy_rollout.png and returns the path.
    """
    total, traj = greedy_rollout_fn(env, Q, max_steps=max_steps if max_steps is not None else env.cfg.max_steps)
    grid = env.grid.copy()
    h, w = grid.shape

    fig = plt.figure(figsize=(6, 6))
    bg = np.zeros_like(grid, dtype=float)
    bg[grid == 1] = 0.6  # TRAP
    bg[grid == 2] = 0.2  # GOAL
    plt.imshow(bg, origin="upper")

    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    plt.plot(xs, ys, linewidth=2)
    plt.scatter([xs[0]], [ys[0]], marker="o", s=60)  # start
    plt.scatter([xs[-1]], [ys[-1]], marker="s", s=60)  # end

    plt.title(f"Greedy Rollout (Return={total:.2f})")
    plt.xticks(range(w))
    plt.yticks(range(h))
    plt.gca().invert_yaxis()

    out_path = Path(out_dir) / "figs" / "greedy_rollout.png"
    _save_fig(fig, out_path)
    return out_path
