#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import yaml

from escape_artist.envs.escape_env import EscapeEnv, EnvConfig
from escape_artist.envs.generators import GOAL
from escape_artist.algos.mc_control import greedy_policy_rollout
from escape_artist.utils.plotting import plot_greedy_rollout as plot_rollout_fig


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _tuple(v, default):
    return tuple(v) if isinstance(v, (list, tuple)) else default


def build_env(cfg: Dict[str, Any], layout_mode: str, seed: int) -> EscapeEnv:
    e = cfg.get("env", {})
    size = _tuple(e.get("size", (10, 10)), (10, 10))
    start = _tuple(e.get("start", (0, 0)), (0, 0))
    goal  = _tuple(e.get("goal", (size[1]-1, size[0]-1)), (size[1]-1, size[0]-1))
    env_cfg = EnvConfig(
        size=size, start=start, goal=goal,
        traps_pct=float(e.get("traps_pct", 0.10)),
        r_safe=int(e.get("r_safe", 1)),
        slip=float(e.get("slip", 0.1)),
        lethal_traps=bool(e.get("lethal_traps", True)),
        step_cost=float(e.get("step_cost", -0.01)),
        trap_penalty=float(e.get("trap_penalty", -1.0)),
        goal_reward=float(e.get("goal_reward", 1.0)),
        max_steps=int(e.get("max_steps", 200)),
        layout_mode=layout_mode,  # "per_env" or "per_episode"
        seed=seed,
        obs_mode=str(e.get("obs_mode", "pos")),
    )
    return EscapeEnv(env_cfg)


def main():
    ap = argparse.ArgumentParser(description="Find a successful greedy rollout and save the overlay image")
    ap.add_argument("--run", required=True, help="Run dir containing Q.npy")
    ap.add_argument("--config", required=True, help="Env YAML (e.g., configs/medium.yaml)")
    ap.add_argument("--out", default="assets/figs/greedy_rollout_success.png", help="Output PNG path")
    ap.add_argument("--layout", choices=["per_env", "per_episode"], default="per_env",
                    help="Layout mode used during search (per_env is stable for a hero image)")
    ap.add_argument("--seed", type=int, default=123, help="Base seed")
    ap.add_argument("--max_tries", type=int, default=300, help="Max layouts/seeds to try")
    ap.add_argument("--max_steps", type=int, default=None, help="Optional rollout cap")
    args = ap.parse_args()

    run_dir = Path(args.run)
    Q_path = run_dir / "Q.npy"
    if not Q_path.exists():
        raise FileNotFoundError(f"Missing {Q_path}")
    Q = np.load(Q_path)

    cfg = _load_yaml(Path(args.config))

    # Try different seeds (per_env gives a clean, fixed layout image)
    for i in range(args.max_tries):
        seed_i = int(args.seed + i)
        env = build_env(cfg, layout_mode=args.layout, seed=seed_i)
        env.reset(seed=seed_i)
        total, traj = greedy_policy_rollout(env, Q, max_steps=args.max_steps or env.cfg.max_steps)
        x_last, y_last = traj[-1]
        success = (env.grid[y_last, x_last] == GOAL)
        if success:
            out_path = Path(args.out)
            # Use plotting helper (it saves to <out_dir>/figs/greedy_rollout.png)
            plot_rollout_fig(env, Q, greedy_policy_rollout, out_dir=out_path.parent, max_steps=env.cfg.max_steps)
            # Rename to the requested filename
            (out_path.parent / "figs").mkdir(parents=True, exist_ok=True)
            default_png = out_path.parent / "figs" / "greedy_rollout.png"
            if default_png.exists():
                default_png.rename(out_path)
            print(f"SUCCESS at seed={seed_i} → {out_path}")
            return
    raise SystemExit(f"No successful rollout found after {args.max_tries} tries. "
                     f"Try increasing --max_tries or training longer.")


if __name__ == "__main__":
    main()
