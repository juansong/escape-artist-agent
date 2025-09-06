#!/usr/bin/env python
# from __future__ import annotations
# import argparse
# from pathlib import Path
# import imageio.v2 as imageio

# def main():
#     ap = argparse.ArgumentParser(description="Make a hero GIF from rollout PNGs")
#     ap.add_argument("--frames", nargs="+", required=True,
#                     help="Ordered list of PNG files (e.g., easy/medium/hard rollouts).")
#     ap.add_argument("--out", type=str, default="assets/escape-artist-hero.gif",
#                     help="Output GIF path.")
#     ap.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2).")
#     args = ap.parse_args()

#     images = []
#     for f in args.frames:
#         p = Path(f)
#         if not p.exists():
#             raise FileNotFoundError(f"Missing frame: {p}")
#         images.append(imageio.imread(p))

#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     imageio.mimsave(out_path, images, fps=args.fps, loop=0)
#     print(f"Saved GIF: {out_path}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
from __future__ import annotations
import argparse, io
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import yaml

from escape_artist.envs.escape_env import EscapeEnv, EnvConfig
from escape_artist.envs.generators import GOAL, TRAP

# -------------------- utility: load YAML & build env --------------------
def _load_yaml(p: Path):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _tuple(v, default):
    return tuple(v) if isinstance(v, (list, tuple)) else default

def build_env(cfg_dict, layout_mode: str, seed: int) -> EscapeEnv:
    e = cfg_dict.get("env", {})
    size  = _tuple(e.get("size", (10,10)), (10,10))
    start = _tuple(e.get("start", (0,0)), (0,0))
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
        layout_mode=layout_mode,    # "per_env" (stable hero) or "per_episode"
        seed=seed,
        obs_mode=str(e.get("obs_mode", "pos")),
    )
    return EscapeEnv(env_cfg)

# -------------------- frame rendering for rollout -----------------------
def _render_frame(env: EscapeEnv, traj):
    grid = env.grid
    h, w = grid.shape
    bg = np.zeros_like(grid, dtype=float)
    bg[grid == TRAP] = 0.6
    bg[grid == GOAL] = 0.2

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(bg, origin="upper")
    xs = [p[0] for p in traj]; ys = [p[1] for p in traj]
    if len(xs) > 1:
        plt.plot(xs, ys, linewidth=2)
    if len(xs) >= 1:
        plt.scatter([xs[0]], [ys[0]], marker="o", s=60)  # start
        plt.scatter([xs[-1]], [ys[-1]], marker="s", s=60) # current
    plt.title("Greedy Rollout")
    plt.xticks(range(w)); plt.yticks(range(h)); plt.gca().invert_yaxis()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig); buf.seek(0)
    return imageio.imread(buf)

def _greedy_idx(qrow, rng):
    m = qrow.max()
    cand = np.flatnonzero(qrow == m)
    return int(rng.choice(cand))

def _rollout_frames(env: EscapeEnv, Q: np.ndarray, rng: np.random.Generator):
    frames = []
    obs, info = env.reset(seed=env.cfg.seed)
    traj = [info["pos"]]
    frames.append(_render_frame(env, traj))

    done = False; steps = 0; h, w = env.h, env.w
    while not done and steps < env.cfg.max_steps:
        x, y = traj[-1]; s = y*w + x
        a = _greedy_idx(Q[s], rng)
        obs, r, term, trunc, info = env.step(a)
        traj.append(info["pos"])
        frames.append(_render_frame(env, traj))
        steps += 1
        done = term or trunc

    x_last, y_last = traj[-1]
    return frames, bool(env.grid[y_last, x_last] == GOAL)

# --------------------------- main ---------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Make a GIF: either stitch PNG frames or animate a rollout from Q.npy")
    # mode A: stitch provided frames (legacy)
    ap.add_argument("--frames", nargs="+", help="Ordered PNG frames to stitch (legacy mode)")
    # mode B: animate greedy rollout from Q
    ap.add_argument("--run", help="Run dir containing Q.npy (enables rollout mode)")
    ap.add_argument("--config", help="Env YAML (used with --run to build env)")
    ap.add_argument("--want", choices=["any","success","failure"], default="any",
                    help="For rollout mode: search seeds until we find this outcome")
    ap.add_argument("--layout", choices=["per_env","per_episode"], default="per_env",
                    help="Layout mode for rollout mode ('per_env' is best for a stable hero GIF)")
    ap.add_argument("--seed", type=int, default=123, help="Base seed for rollout mode")
    ap.add_argument("--max_tries", type=int, default=300, help="Seeds to try in rollout mode")
    # output options
    ap.add_argument("--out", default="assets/escape-artist-hero.gif", help="Output GIF path")
    ap.add_argument("--fps", type=int, default=2, help="GIF frames per second")
    ap.add_argument("--hold_end", type=int, default=6, help="Duplicate last frame this many times in GIF")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -------- Mode A: stitch given frames (unchanged behavior) ----------
    if args.frames:
        imgs = []
        for f in args.frames:
            p = Path(f)
            if not p.exists():
                raise FileNotFoundError(f"Missing frame: {p}")
            imgs.append(imageio.imread(p))
        if imgs:
            imgs = imgs + [imgs[-1]] * max(0, int(args.hold_end))
        imageio.mimsave(out_path, imgs, fps=args.fps, loop=0)
        print(f"Saved GIF from {len(args.frames)} frames: {out_path}")
        return

    # -------- Mode B: animate rollout from Q.npy ------------------------
    if not (args.run and args.config):
        raise SystemExit("Either provide --frames ... OR provide both --run and --config for rollout mode.")

    q_path = Path(args.run) / "Q.npy"
    if not q_path.exists():
        raise FileNotFoundError(f"Missing {q_path}")
    Q = np.load(q_path)
    cfg_dict = _load_yaml(Path(args.config))

    rng = np.random.default_rng(0)
    for i in range(args.max_tries):
        seed_i = args.seed + i
        env = build_env(cfg_dict, layout_mode=args.layout, seed=seed_i)
        frames, success = _rollout_frames(env, Q, rng)
        ok = (args.want == "any") or (args.want == "success" and success) or (args.want == "failure" and not success)
        if ok:
            frames = frames + [frames[-1]] * max(0, int(args.hold_end))
            imageio.mimsave(out_path, frames, fps=args.fps, loop=0)
            print(f"Saved GIF ({'success' if success else 'failure'}) at seed={seed_i}: {out_path}")
            return

    raise SystemExit(f"No rollout of type '{args.want}' found after {args.max_tries} tries. "
                     f"Increase --max_tries or train longer.")

if __name__ == "__main__":
    main()
