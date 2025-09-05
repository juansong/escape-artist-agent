# experiments/run_experiment.py
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import yaml
import matplotlib.pyplot as plt

from escape_artist.envs.escape_env import EscapeEnv, EnvConfig
from escape_artist.algos.mc_control import MCConfig, train_mc_control, greedy_policy_rollout
from escape_artist.algos.mc_offpolicy import OffMCConfig, train_mc_offpolicy
from escape_artist.algos.q_learning import QLConfig, train_q_learning


# ------------------------------- I/O utils -------------------------------

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _save_yaml(obj: Dict[str, Any], path: str | Path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy_config_to_out(config_path: Path, out_dir: Path):
    dst = out_dir / "config_used.yaml"
    try:
        shutil.copy2(config_path, dst)
    except Exception:
        with open(dst, "w") as f:
            f.write("# Failed to copy original config. Provide the same config when plotting.\n")


# ------------------------------- Plotting -------------------------------

def plot_learning_curve(returns: np.ndarray, out_dir: Path):
    fig = plt.figure(figsize=(6, 4))
    xs = np.arange(1, len(returns) + 1)
    plt.plot(xs, returns)
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.title("Learning Curve")
    plt.grid(True, alpha=0.3)
    _ensure_dir(out_dir / "figs")
    fig.savefig(out_dir / "figs" / "learning_curve.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _compute_V_and_policy(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns V(s)=max_a Q(s,a) and greedy π(s)=argmax_a Q(s,a) with uniform tie-breaking."""
    V = Q.max(axis=1)
    A = np.zeros(Q.shape[0], dtype=np.int32)
    rng = np.random.default_rng()
    for s, row in enumerate(Q):
        maxv = row.max()
        cand = np.flatnonzero(row == maxv)
        A[s] = rng.choice(cand)
    return V, A


def plot_value_heatmap_and_policy(env: EscapeEnv, Q: np.ndarray, out_dir: Path, title: str = "Value & Greedy Policy"):
    """Draw V(s) as a heatmap and overlay greedy arrows on the current layout."""
    h, w = env.h, env.w
    grid = env.grid.copy()
    V, A = _compute_V_and_policy(Q)

    V_img = V.reshape(h, w)
    mask = (grid == 1)  # TRAP
    V_masked = np.ma.array(V_img, mask=mask)

    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(V_masked, origin="upper")
    plt.colorbar(im, shrink=0.8, label="V(s) = max_a Q(s,a)")

    # traps & goal markers
    ys, xs = np.where(grid == 1)  # traps
    plt.scatter(xs, ys, marker="X", s=40)
    gy, gx = np.argwhere(grid == 2)[0]  # goal
    plt.scatter([gx], [gy], marker="*", s=120)

    # greedy arrows
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
    _ensure_dir(out_dir / "figs")
    fig.savefig(out_dir / "figs" / "value_heatmap_policy.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_greedy_rollout(env: EscapeEnv, Q: np.ndarray, out_dir: Path, max_steps: int | None = None):
    """Plot a greedy rollout path over the current layout."""
    total, traj = greedy_policy_rollout(env, Q, max_steps=max_steps)
    grid = env.grid.copy()
    h, w = grid.shape

    fig = plt.figure(figsize=(6, 6))
    bg = np.zeros_like(grid, dtype=float)
    bg[grid == 1] = 0.6
    bg[grid == 2] = 0.2
    plt.imshow(bg, origin="upper")

    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    plt.plot(xs, ys, linewidth=2)
    plt.scatter([xs[0]], [ys[0]], marker="o", s=60)
    plt.scatter([xs[-1]], [ys[-1]], marker="s", s=60)

    plt.title(f"Greedy Rollout (Return={total:.2f})")
    plt.xticks(range(w))
    plt.yticks(range(h))
    plt.gca().invert_yaxis()
    _ensure_dir(out_dir / "figs")
    fig.savefig(out_dir / "figs" / "greedy_rollout.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ------------------------------- Instantiation -------------------------------

def build_env_from_config(cfg: Dict[str, Any]) -> EscapeEnv:
    env_kwargs = cfg.get("env", {})

    def _tuple(v, default):
        return tuple(v) if isinstance(v, (list, tuple)) else default

    size = _tuple(env_kwargs.get("size", (8, 8)), (8, 8))
    start = _tuple(env_kwargs.get("start", (0, 0)), (0, 0))
    goal = _tuple(env_kwargs.get("goal", (size[1] - 1, size[0] - 1)), (size[1] - 1, size[0] - 1))

    env_config = EnvConfig(
        size=size,
        start=start,
        goal=goal,
        traps_pct=float(env_kwargs.get("traps_pct", 0.08)),
        r_safe=int(env_kwargs.get("r_safe", 1)),
        slip=float(env_kwargs.get("slip", 0.0)),
        lethal_traps=bool(env_kwargs.get("lethal_traps", True)),
        step_cost=float(env_kwargs.get("step_cost", -0.01)),
        trap_penalty=float(env_kwargs.get("trap_penalty", -1.0)),
        goal_reward=float(env_kwargs.get("goal_reward", 1.0)),
        max_steps=int(env_kwargs.get("max_steps", 200)),
        layout_mode=str(env_kwargs.get("layout_mode", "per_episode")),
        seed=env_kwargs.get("seed", 42),
        obs_mode=str(env_kwargs.get("obs_mode", "pos")),
    )
    return EscapeEnv(env_config)


def build_mc_config_from_config(cfg: Dict[str, Any], cli_overrides: Dict[str, Any]) -> MCConfig:
    algo_kwargs = cfg.get("algo", {})
    visit = cli_overrides.get("visit") or algo_kwargs.get("visit", "first")
    episodes = int(cli_overrides.get("episodes") or algo_kwargs.get("episodes", 10_000))
    gamma = float(algo_kwargs.get("gamma", 0.99))
    eps_start = float(algo_kwargs.get("epsilon_start", 0.2))
    eps_end = float(algo_kwargs.get("epsilon_end", 0.02))
    eps_decay = int(algo_kwargs.get("epsilon_decay_episodes", 8000))
    seed = cli_overrides.get("seed") if cli_overrides.get("seed") is not None else algo_kwargs.get("seed", 42)
    out_dir = cli_overrides.get("out_dir")
    return MCConfig(
        episodes=episodes,
        gamma=gamma,
        visit=visit,
        epsilon_start=eps_start,
        epsilon_end=eps_end,
        epsilon_decay_episodes=eps_decay,
        seed=seed,
        out_dir=out_dir,
    )


def build_offmc_config_from_config(cfg: Dict[str, Any], cli_overrides: Dict[str, Any]) -> OffMCConfig:
    algo_kwargs = cfg.get("algo", {})
    is_type = cli_overrides.get("is_type") or algo_kwargs.get("is_type", "weighted")
    episodes = int(cli_overrides.get("episodes") or algo_kwargs.get("episodes", 15_000))
    gamma = float(algo_kwargs.get("gamma", 0.99))
    eps_b_start = float(algo_kwargs.get("epsilon_behavior_start", 0.2))
    eps_b_end = float(algo_kwargs.get("epsilon_behavior_end", 0.02))
    eps_b_decay = int(algo_kwargs.get("epsilon_behavior_decay_episodes", 10_000))
    seed = cli_overrides.get("seed") if cli_overrides.get("seed") is not None else algo_kwargs.get("seed", 42)
    out_dir = cli_overrides.get("out_dir")
    return OffMCConfig(
        episodes=episodes,
        gamma=gamma,
        is_type=is_type,
        epsilon_behavior_start=eps_b_start,
        epsilon_behavior_end=eps_b_end,
        epsilon_behavior_decay_episodes=eps_b_decay,
        seed=seed,
        out_dir=out_dir,
    )


def build_q_config_from_config(cfg: Dict[str, Any], cli_overrides: Dict[str, Any]) -> QLConfig:
    algo_kwargs = cfg.get("algo", {})
    episodes = int(cli_overrides.get("episodes") or algo_kwargs.get("episodes", 10_000))
    gamma = float(algo_kwargs.get("gamma", 0.99))

    alpha_start = float(algo_kwargs.get("alpha_start", 0.5))
    alpha_end = float(algo_kwargs.get("alpha_end", 0.05))
    alpha_decay = int(algo_kwargs.get("alpha_decay_episodes", 8_000))

    eps_start = float(algo_kwargs.get("epsilon_start", 0.2))
    eps_end = float(algo_kwargs.get("epsilon_end", 0.02))
    eps_decay = int(algo_kwargs.get("epsilon_decay_episodes", 8_000))

    seed = cli_overrides.get("seed") if cli_overrides.get("seed") is not None else algo_kwargs.get("seed", 42)
    out_dir = cli_overrides.get("out_dir")

    return QLConfig(
        episodes=episodes,
        gamma=gamma,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        alpha_decay_episodes=alpha_decay,
        epsilon_start=eps_start,
        epsilon_end=eps_end,
        epsilon_decay_episodes=eps_decay,
        seed=seed,
        out_dir=out_dir,
    )


# ------------------------------- Main flows -------------------------------

def _common_plots(env_cfg: Dict[str, Any], Q: np.ndarray, returns: np.ndarray, out_dir: Path, title_suffix: str):
    plot_learning_curve(returns, out_dir)

    # fixed-layout env for stable overlays
    cfg_fixed = {"env": dict(env_cfg)}
    cfg_fixed["env"]["layout_mode"] = "per_env"
    fixed_env = build_env_from_config(cfg_fixed)
    fixed_env.reset(seed=fixed_env.cfg.seed)

    plot_value_heatmap_and_policy(fixed_env, Q, out_dir, title=f"V(s) & Greedy Policy {title_suffix}")
    plot_greedy_rollout(fixed_env, Q, out_dir, max_steps=fixed_env.cfg.max_steps)


def run_train(args):
    cfg = _load_yaml(args.config)
    out_dir = _ensure_dir(args.out)
    _copy_config_to_out(Path(args.config), out_dir)

    env = build_env_from_config(cfg)
    algo_type = (cfg.get("algo", {}).get("type") or args.algo or "mc").lower()

    if algo_type == "mc":
        mc_cfg = build_mc_config_from_config(cfg, {
            "visit": args.visit,
            "episodes": args.episodes,
            "seed": args.seed,
            "out_dir": str(out_dir),
        })
        print(f"Training MC ({mc_cfg.visit}-visit) for {mc_cfg.episodes} episodes…")
        Q, returns = train_mc_control(env, mc_cfg)
        _common_plots(cfg.get("env", {}), Q, np.asarray(returns, dtype=float), out_dir, title_suffix="(MC, fixed layout)")

    elif algo_type == "mc_off":
        off_cfg = build_offmc_config_from_config(cfg, {
            "is_type": args.is_type,
            "episodes": args.episodes,
            "seed": args.seed,
            "out_dir": str(out_dir),
        })
        print(f"Training MC-OFF ({off_cfg.is_type}) for {off_cfg.episodes} episodes…")
        Q, returns = train_mc_offpolicy(env, off_cfg)
        _common_plots(cfg.get("env", {}), Q, np.asarray(returns, dtype=float), out_dir, title_suffix=f"(MC-OFF/{off_cfg.is_type}, fixed layout)")

    elif algo_type == "q":
        q_cfg = build_q_config_from_config(cfg, {
            "episodes": args.episodes,
            "seed": args.seed,
            "out_dir": str(out_dir),
        })
        print(f"Training Q-Learning for {q_cfg.episodes} episodes…")
        Q, returns = train_q_learning(env, q_cfg)
        _common_plots(cfg.get("env", {}), Q, np.asarray(returns, dtype=float), out_dir, title_suffix="(Q-Learning, fixed layout)")

    else:
        raise SystemExit(f"Unsupported --algo '{algo_type}'. Use 'mc', 'mc_off', or 'q'.")

    print(f"Done. Artifacts in: {out_dir}")


def run_plot_only(args):
    run_dir = Path(args.from_dir)
    Q_path = run_dir / "Q.npy"
    ret_path = run_dir / "returns.npy"
    cfg_used = run_dir / "config_used.yaml"

    if not Q_path.exists() or not ret_path.exists():
        raise FileNotFoundError("Q.npy or returns.npy missing in the run directory.")

    Q = np.load(Q_path)
    returns = np.load(ret_path)

    # Prefer saved config_used.yaml; else require --config
    if cfg_used.exists():
        cfg = _load_yaml(cfg_used)
    elif args.config:
        cfg = _load_yaml(args.config)
    else:
        raise FileNotFoundError("Provide --config or ensure config_used.yaml exists in the run dir.")

    env = build_env_from_config({"env": cfg.get("env", {})})
    env.reset(seed=env.cfg.seed)

    print("Generating plots from saved artifacts…")
    _common_plots(cfg.get("env", {}), Q, np.asarray(returns, dtype=float), run_dir, title_suffix="(from saved)")
    print(f"Done. Figures saved under: {run_dir / 'figs'}")


# ------------------------------- CLI -------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Escape Artist Agent - Experiments CLI")

    p.add_argument("--algo", choices=["mc", "mc_off", "q"], default="mc",
                   help="Algorithm to run.")
    p.add_argument("--visit", choices=["first", "every"], default="first",
                   help="For --algo mc: first-visit vs every-visit.")
    p.add_argument("--is", dest="is_type", choices=["weighted", "ordinary"], default="weighted",
                   help="For --algo mc_off: importance sampling type.")
    p.add_argument("--episodes", type=int, default=None,
                   help="Number of training episodes (overrides config).")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed (overrides config).")
    p.add_argument("--config", type=str, required=False,
                   help="Path to YAML config (required for training unless --plot).")
    p.add_argument("--out", type=str, required=False,
                   help="Output directory to save artifacts & figures.")

    p.add_argument("--plot", action="store_true",
                   help="Plot-only mode: regenerate figures from a previous run.")
    p.add_argument("--from", dest="from_dir", type=str, default=None,
                   help="Existing run directory containing Q.npy and returns.npy. Used with --plot.")

    return p.parse_args()


def main():
    args = parse_args()

    if args.plot:
        if not args.from_dir:
            raise SystemExit("When using --plot, provide --from <run_dir>.")
        run_plot_only(args)
        return

    if not args.config:
        raise SystemExit("Please provide --config for training.")
    if not args.out:
        # default out dir name
        suffix = (
            args.is_type if args.algo == "mc_off"
            else (args.visit if args.algo == "mc" else "q")
        )
        args.out = f"runs/{args.algo}_{suffix}"
    run_train(args)


if __name__ == "__main__":
    main()
