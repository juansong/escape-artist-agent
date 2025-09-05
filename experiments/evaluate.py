#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import yaml

from escape_artist.envs.escape_env import EscapeEnv, EnvConfig
from escape_artist.algos.mc_control import greedy_policy_rollout  # for consistency


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_env_from_config(cfg: Dict[str, Any]) -> EscapeEnv:
    env_kwargs = cfg.get("env", {})
    def _tuple(v, default):
        return tuple(v) if isinstance(v, (list, tuple)) else default

    size = _tuple(env_kwargs.get("size", (10, 10)), (10, 10))
    start = _tuple(env_kwargs.get("start", (0, 0)), (0, 0))
    goal  = _tuple(env_kwargs.get("goal", (size[1]-1, size[0]-1)), (size[1]-1, size[0]-1))

    env_cfg = EnvConfig(
        size=size,
        start=start,
        goal=goal,
        traps_pct=float(env_kwargs.get("traps_pct", 0.10)),
        r_safe=int(env_kwargs.get("r_safe", 1)),
        slip=float(env_kwargs.get("slip", 0.1)),
        lethal_traps=bool(env_kwargs.get("lethal_traps", True)),
        step_cost=float(env_kwargs.get("step_cost", -0.01)),
        trap_penalty=float(env_kwargs.get("trap_penalty", -1.0)),
        goal_reward=float(env_kwargs.get("goal_reward", 1.0)),
        max_steps=int(env_kwargs.get("max_steps", 200)),
        layout_mode="per_episode",  # IMPORTANT: robustness eval
        seed=env_kwargs.get("seed", 123),
        obs_mode=str(env_kwargs.get("obs_mode", "pos")),
    )
    return EscapeEnv(env_cfg)


def greedy_action(Q_row: np.ndarray, rng: np.random.Generator) -> int:
    m = Q_row.max()
    cand = np.flatnonzero(Q_row == m)
    return int(rng.choice(cand))


def eval_policy(env: EscapeEnv, Q: np.ndarray, episodes: int, rng: np.random.Generator) -> Dict[str, float]:
    """Evaluate greedy policy w.r.t Q over N episodes with per-episode layouts."""
    h, w = env.h, env.w
    n_success = 0
    n_detect  = 0
    steps_all: List[int] = []
    steps_success: List[int] = []

    for _ in range(episodes):
        obs, info = env.reset()
        steps = 0
        detected = False
        done = False

        while not done and steps < env.cfg.max_steps:
            if "pos" in info:
                x, y = info["pos"]
            else:
                x, y = int(obs[0]), int(obs[1])
            s = y * w + x

            a = greedy_action(Q[s], rng)
            obs, r, term, trunc, info = env.step(a)
            steps += 1

            if info.get("is_trap", False):
                detected = True

            done = term or trunc

        # success = info.get("is_goal", False)
        # robust success check at episode end
        x_last, y_last = info.get("pos", (None, None))
        success = bool(info.get("is_goal", False))
        if x_last is not None and y_last is not None:
            from escape_artist.envs.generators import GOAL
            success = success or (env.grid[y_last, x_last] == GOAL)

        n_success += int(success)
        n_detect  += int(detected)
        steps_all.append(steps)
        if success:
            steps_success.append(steps)


    eps = episodes if episodes > 0 else 1
    success_rate = n_success / eps
    detection_rate = n_detect / eps
    avg_steps_success = float(np.mean(steps_success)) if steps_success else float("nan")
    avg_steps_all = float(np.mean(steps_all)) if steps_all else float("nan")
    timeout_rate = sum(1 for s in steps_all if s >= env.cfg.max_steps) / eps

    return {
        "success_rate": success_rate,
        "avg_steps_success": avg_steps_success,
        "avg_steps_all": avg_steps_all,
        "detection_rate": detection_rate,
        "timeout_rate": timeout_rate,  # NEW
    }


# def to_markdown_table(rows: List[Dict[str, Any]]) -> str:
#     headers = ["Method", "Success Rate ↑", "Avg Steps ↓", "Detection Rate ↓"]
#     sep = "| " + " | ".join(["---"] * len(headers)) + " |"
#     lines = ["| " + " | ".join(headers) + " |", sep]
#     for r in rows:
#         line = "| " + " | ".join([
#             str(r["label"]),
#             f"{r['success_rate']*100:0.1f}%",
#             f"{r['avg_steps_success']:.1f}" if np.isfinite(r['avg_steps_success']) else "—",
#             f"{r['detection_rate']*100:0.1f}%",
#         ]) + " |"
#         lines.append(line)
#     return "\n".join(lines)

def to_markdown_table(rows):
    headers = ["Method", "Success Rate ↑", "Avg Steps ↓", "Detection Rate ↓", "Timeout Rate ↓"]
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = ["| " + " | ".join(headers) + " |", sep]
    for r in rows:
        line = "| " + " | ".join([
            str(r["label"]),
            f"{r['success_rate']*100:0.1f}%",
            f"{r['avg_steps_success']:.1f}" if np.isfinite(r['avg_steps_success']) else "—",
            f"{r['detection_rate']*100:0.1f}%",
            f"{r['timeout_rate']*100:0.1f}%",
        ]) + " |"
        lines.append(line)
    return "\n".join(lines)



def main():
    ap = argparse.ArgumentParser(description="Evaluate greedy policies from run dirs and summarize metrics")
    ap.add_argument("--config", type=str, required=True, help="Env YAML to define evaluation setting")
    ap.add_argument("--runs", nargs="+", required=True, help="Run folders (each must contain Q.npy)")
    ap.add_argument("--labels", nargs="+", required=True, help="Labels for table (same length as --runs)")
    ap.add_argument("--episodes", type=int, default=200, help="Evaluation episodes (per method)")
    ap.add_argument("--seed", type=int, default=2025, help="RNG seed for tie-breaking")
    ap.add_argument("--out_csv", type=str, default="assets/eval_medium.csv")
    ap.add_argument("--out_md", type=str, default="assets/eval_medium.md")
    args = ap.parse_args()

    if len(args.runs) != len(args.labels):
        raise SystemExit("len(--runs) must equal len(--labels)")

    cfg = _load_yaml(Path(args.config))
    env = build_env_from_config(cfg)
    rng = np.random.default_rng(args.seed)

    results = []
    csv_rows = []

    for run, label in zip(args.runs, args.labels):
        q_path = Path(run) / "Q.npy"
        if not q_path.exists():
            raise FileNotFoundError(f"Missing {q_path}")
        Q = np.load(q_path)
        metrics = eval_policy(env, Q, episodes=args.episodes, rng=rng)
        row = {"label": label, **metrics}
        results.append(row)

        csv_rows.append({
            "method": label,
            "success_rate": metrics["success_rate"],
            "avg_steps_success": metrics["avg_steps_success"],
            "avg_steps_all": metrics["avg_steps_all"],
            "detection_rate": metrics["detection_rate"],
        })
        print(f"[eval] {label}: SR={metrics['success_rate']:.3f} | "
              f"Steps_succ={metrics['avg_steps_success']:.1f} | "
              f"Detect={metrics['detection_rate']:.3f}")

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved CSV: {out_csv}")

    # Write Markdown table
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md_table = to_markdown_table(results)
    out_md.write_text(md_table, encoding="utf-8")
    print(f"Saved Markdown table: {out_md}\n\n{md_table}\n")


if __name__ == "__main__":
    main()
