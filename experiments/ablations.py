#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import csv

from escape_artist.envs.escape_env import EscapeEnv, EnvConfig
from escape_artist.envs.generators import GOAL
from escape_artist.algos.mc_control import MCConfig, train_mc_control
from escape_artist.algos.q_learning import QLConfig, train_q_learning

# --- helpers -----------------------------------------------------------------

def build_env(size, traps_pct, slip, r_safe=1, seed=123, layout_mode="per_episode"):
    cfg = EnvConfig(
        size=size, start=(0, 0), goal=(size[1]-1, size[0]-1),
        traps_pct=traps_pct, r_safe=r_safe, slip=slip,
        lethal_traps=True, step_cost=-0.01, trap_penalty=-1.0, goal_reward=1.0,
        max_steps=200, layout_mode=layout_mode, seed=seed, obs_mode="pos",
    )
    return EscapeEnv(cfg)

def greedy_success_rate(env: EscapeEnv, Q: np.ndarray, episodes: int = 30) -> tuple[float, float]:
    """
    Evaluate greedy policy success across 'episodes' layouts (per_episode env).
    Returns (success_rate, avg_return_proxy).
    """
    from escape_artist.algos.mc_control import greedy_policy_rollout
    successes = 0
    returns = []
    for _ in range(episodes):
        total, traj = greedy_policy_rollout(env, Q, max_steps=env.cfg.max_steps)
        x, y = traj[-1]
        success = (env.grid[y, x] == GOAL)
        successes += int(success)
        returns.append(total)
    return successes / episodes, float(np.mean(returns))

def heatmap(values, xs, ys, title, out_path: Path):
    arr = np.array(values).reshape(len(ys), len(xs))  # rows=ys, cols=xs
    fig = plt.figure(figsize=(6, 5))
    im = plt.imshow(arr, origin="upper", aspect="auto")
    plt.colorbar(im, shrink=0.85, label="Success rate")
    plt.xticks(range(len(xs)), [str(x) for x in xs])
    plt.yticks(range(len(ys)), [str(y) for y in ys])
    plt.xlabel("traps_pct")
    plt.ylabel("slip")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Ablations: traps_pct × slip for MC (Every) and Q-learning")
    ap.add_argument("--size", type=int, nargs=2, default=[10, 10], help="H W")
    ap.add_argument("--traps", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.18])
    ap.add_argument("--slips", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    ap.add_argument("--episodes", type=int, default=4000, help="Train episodes per setting")
    ap.add_argument("--eval_eps", type=int, default=30, help="Evaluation rollouts (per setting)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_csv", type=str, default="assets/ablations_mc_q.csv")
    ap.add_argument("--out_png_prefix", type=str, default="assets/ablations_heatmap")
    args = ap.parse_args()

    size = tuple(args.size)
    combos = list(product(args.slips, args.traps))  # rows=slips, cols=traps for plotting

    rows = []
    mc_success_grid = []
    q_success_grid = []

    for slip, traps in combos:
        # Train MC (Every-Visit)
        env_mc = build_env(size, traps, slip, seed=args.seed, layout_mode="per_episode")
        mc_cfg = MCConfig(
            episodes=args.episodes, gamma=0.99, visit="every",
            epsilon_start=0.2, epsilon_end=0.02, epsilon_decay_episodes=max(1, int(0.8*args.episodes)),
            seed=args.seed,
        )
        Q_mc, _ = train_mc_control(env_mc, mc_cfg)
        sr_mc, ret_mc = greedy_success_rate(env_mc, Q_mc, episodes=args.eval_eps)

        # Train Q-Learning
        env_q = build_env(size, traps, slip, seed=args.seed, layout_mode="per_episode")
        q_cfg = QLConfig(
            episodes=args.episodes, gamma=0.99,
            alpha_start=0.5, alpha_end=0.1, alpha_decay_episodes=max(1, int(0.8*args.episodes)),
            epsilon_start=0.2, epsilon_end=0.02, epsilon_decay_episodes=max(1, int(0.8*args.episodes)),
            seed=args.seed,
        )
        Q_q, _ = train_q_learning(env_q, q_cfg)
        sr_q, ret_q = greedy_success_rate(env_q, Q_q, episodes=args.eval_eps)

        rows.append({
            "size": f"{size[0]}x{size[1]}",
            "slip": slip,
            "traps_pct": traps,
            "algo": "MC-every",
            "success_rate": sr_mc,
            "avg_return": ret_mc,
        })
        rows.append({
            "size": f"{size[0]}x{size[1]}",
            "slip": slip,
            "traps_pct": traps,
            "algo": "Q-learning",
            "success_rate": sr_q,
            "avg_return": ret_q,
        })
        mc_success_grid.append(sr_mc)
        q_success_grid.append(sr_q)
        print(f"[ablations] slip={slip:.2f} traps={traps:.2f} | MC sr={sr_mc:.2f} | Q sr={sr_q:.2f}")

    # write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {out_csv}")

    # heatmaps
    prefix = Path(args.out_png_prefix)
    heatmap(mc_success_grid, xs=args.traps, ys=args.slips, title="Success Rate (MC Every-Visit)",
            out_path=prefix.with_name(prefix.name + "_mc.png"))
    heatmap(q_success_grid, xs=args.traps, ys=args.slips, title="Success Rate (Q-Learning)",
            out_path=prefix.with_name(prefix.name + "_q.png"))
    print("Saved heatmaps next to CSV.")

if __name__ == "__main__":
    main()
