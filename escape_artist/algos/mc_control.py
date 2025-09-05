# escape_artist/algos/mc_control.py

# supports both first-visit and every-visit Monte Carlo control with epsilon-greedy policy.
# uses `info["pos"]`` -> robust to `obs_mode`.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os
import json
import numpy as np


@dataclass
class MCConfig:
    episodes: int = 10_000
    gamma: float = 0.99
    visit: str = "first"          # "first" | "every"
    epsilon_start: float = 0.2
    epsilon_end: float = 0.02
    epsilon_decay_episodes: int = 8_000
    seed: Optional[int] = 42
    out_dir: Optional[str] = None  # if set, saves Q, returns, and params


def _epsilon_by_episode(cfg: MCConfig, ep: int) -> float:
    if cfg.epsilon_decay_episodes <= 0:
        return cfg.epsilon_end
    frac = min(1.0, ep / cfg.epsilon_decay_episodes)
    return float(cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start))


def _state_to_index(pos: Tuple[int, int], w: int) -> int:
    """Tabular index for a grid position (x, y)."""
    x, y = pos
    return y * w + x


def _epsilon_greedy(Q: np.ndarray, s_idx: int, epsilon: float, rng: np.random.Generator) -> int:
    """Choose action ε-greedily from Q[s]."""
    if rng.random() < epsilon:
        return int(rng.integers(0, Q.shape[1]))
    # break ties uniformly
    row = Q[s_idx]
    maxv = row.max()
    candidates = np.flatnonzero(row == maxv)
    return int(rng.choice(candidates))


def _ensure_out_dir(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def _save_artifacts(out_dir: str, Q: np.ndarray, returns: List[float], cfg: MCConfig):
    np.save(os.path.join(out_dir, "Q.npy"), Q)
    np.save(os.path.join(out_dir, "returns.npy"), np.array(returns, dtype=np.float32))
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(
            {
                "episodes": cfg.episodes,
                "gamma": cfg.gamma,
                "visit": cfg.visit,
                "epsilon_start": cfg.epsilon_start,
                "epsilon_end": cfg.epsilon_end,
                "epsilon_decay_episodes": cfg.epsilon_decay_episodes,
                "seed": cfg.seed,
            },
            f,
            indent=2,
        )


def _episode_returns(rewards: List[float], gamma: float) -> List[float]:
    """Return G_t for each timestep t (list, same length as rewards)."""
    G = 0.0
    out = [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        out[t] = G
    return out


def generate_episode(
    env,
    Q: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int], List[float], Dict]:
    """
    Roll out one episode using ε-greedy policy over tabular Q.
    Returns (states_idx, actions, rewards, last_info).
    Uses env.info["pos"] to read (x, y) robustly across obs modes.
    """
    obs, info = env.reset()
    h, w = env.h, env.w  # expected from EscapeEnv; falls back if present
    s_idx_list: List[int] = []
    a_list: List[int] = []
    r_list: List[float] = []

    while True:
        # Determine the tabular index of the current state.
        # Prefer info["pos"]; if absent, infer from obs (pos mode).
        if "pos" in info:
            s_idx = _state_to_index(info["pos"], w)
        else:
            # obs assumed to be [x, y] in "pos" mode
            x, y = int(obs[0]), int(obs[1])
            s_idx = _state_to_index((x, y), w)

        a = _epsilon_greedy(Q, s_idx, epsilon, rng)
        obs, r, terminated, truncated, info = env.step(a)

        s_idx_list.append(s_idx)
        a_list.append(a)
        r_list.append(float(r))

        if terminated or truncated:
            return s_idx_list, a_list, r_list, info


def train_mc_control(env, cfg: MCConfig) -> Tuple[np.ndarray, List[float]]:
    """
    Monte Carlo control (on-policy) with First-Visit or Every-Visit updates.
    Returns (Q, episodic_returns).
    """
    assert cfg.visit in ("first", "every"), "cfg.visit must be 'first' or 'every'"

    # Reproducible RNG
    rng = np.random.default_rng(cfg.seed)

    # Tabular shapes
    h, w = env.h, env.w  # EscapeEnv exposes grid size
    nS = h * w
    nA = getattr(env.action_space, "n", 4)

    # Q and counts
    Q = np.zeros((nS, nA), dtype=np.float32)
    N = np.zeros((nS, nA), dtype=np.int32)  # visit counts for incremental mean

    episodic_returns: List[float] = []

    out_dir = _ensure_out_dir(cfg.out_dir)
    if out_dir:
        with open(os.path.join(out_dir, "seed.txt"), "w") as f:
            f.write(str(cfg.seed if cfg.seed is not None else ""))

    for ep in range(cfg.episodes):
        epsilon = _epsilon_by_episode(cfg, ep)
        s_idx_list, a_list, r_list, last_info = generate_episode(env, Q, epsilon, rng)

        # Compute returns G_t for each step
        G_list = _episode_returns(r_list, cfg.gamma)

        if cfg.visit == "first":
            seen: set[Tuple[int, int]] = set()
            for t, (s_idx, a, Gt) in enumerate(zip(s_idx_list, a_list, G_list)):
                key = (s_idx, a)
                if key in seen:
                    continue
                seen.add(key)
                N[s_idx, a] += 1
                Q[s_idx, a] += (Gt - Q[s_idx, a]) / N[s_idx, a]
        else:  # every-visit
            for s_idx, a, Gt in zip(s_idx_list, a_list, G_list):
                N[s_idx, a] += 1
                Q[s_idx, a] += (Gt - Q[s_idx, a]) / N[s_idx, a]

        ep_return = float(sum(r_list))
        episodic_returns.append(ep_return)

        # Optional lightweight progress print (keeps CLI tidy)
        if (ep + 1) % max(1, cfg.episodes // 20) == 0:
            print(
                f"[MC-{cfg.visit}] ep {ep + 1}/{cfg.episodes} | "
                f"epsilon={epsilon:.3f} | G={ep_return:.3f}"
            )

    if out_dir:
        _save_artifacts(out_dir, Q, episodic_returns, cfg)

    return Q, episodic_returns


# --------- Convenience: greedy rollout for eval/visualization ---------

def greedy_policy_rollout(env, Q: np.ndarray, max_steps: Optional[int] = None) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Rollout using greedy policy w.r.t. Q and return (total_return, trajectory of positions).
    """
    obs, info = env.reset()
    h, w = env.h, env.w
    total = 0.0
    traj: List[Tuple[int, int]] = [tuple(info["pos"]) if "pos" in info else (int(obs[0]), int(obs[1]))]
    steps = 0
    limit = max_steps if max_steps is not None else env.cfg.max_steps

    while steps < limit:
        if "pos" in info:
            s_idx = _state_to_index(info["pos"], w)
        else:
            x, y = int(obs[0]), int(obs[1])
            s_idx = _state_to_index((x, y), w)

        # Greedy action (break ties uniformly)
        row = Q[s_idx]
        maxv = row.max()
        candidates = np.flatnonzero(row == maxv)
        a = int(np.random.default_rng().choice(candidates))

        obs, r, term, trunc, info = env.step(a)
        total += float(r)
        steps += 1
        pos = tuple(info["pos"]) if "pos" in info else (int(obs[0]), int(obs[1]))
        traj.append(pos)
        if term or trunc:
            break

    return total, traj
