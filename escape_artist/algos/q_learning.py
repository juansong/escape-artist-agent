# escape_artist/algos/q_learning.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os
import json
import numpy as np


@dataclass
class QLConfig:
    episodes: int = 10_000
    gamma: float = 0.99
    alpha_start: float = 0.5
    alpha_end: float = 0.05
    alpha_decay_episodes: int = 8_000
    epsilon_start: float = 0.2
    epsilon_end: float = 0.02
    epsilon_decay_episodes: int = 8_000
    seed: Optional[int] = 42
    out_dir: Optional[str] = None  # saves Q, returns, params if set


def _sched_linear(start: float, end: float, horizon: int, t: int) -> float:
    if horizon <= 0:
        return end
    frac = min(1.0, t / horizon)
    return float(start + frac * (end - start))


def _state_to_index(pos: Tuple[int, int], w: int) -> int:
    x, y = pos
    return y * w + x


def _epsilon_greedy(Q: np.ndarray, s_idx: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, Q.shape[1]))
    row = Q[s_idx]
    maxv = row.max()
    cand = np.flatnonzero(row == maxv)
    return int(rng.choice(cand))


def _ensure_out(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def _save(out_dir: str, Q: np.ndarray, returns: List[float], cfg: QLConfig):
    np.save(os.path.join(out_dir, "Q.npy"), Q)
    np.save(os.path.join(out_dir, "returns.npy"), np.asarray(returns, dtype=np.float32))
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(
            {
                "episodes": cfg.episodes,
                "gamma": cfg.gamma,
                "alpha_start": cfg.alpha_start,
                "alpha_end": cfg.alpha_end,
                "alpha_decay_episodes": cfg.alpha_decay_episodes,
                "epsilon_start": cfg.epsilon_start,
                "epsilon_end": cfg.epsilon_end,
                "epsilon_decay_episodes": cfg.epsilon_decay_episodes,
                "seed": cfg.seed,
            },
            f,
            indent=2,
        )


def train_q_learning(env, cfg: QLConfig) -> Tuple[np.ndarray, List[float]]:
    """
    Tabular Q-Learning with linear ε/α schedules and uniform tie-breaking.
    Core: TD updates `Q[s,a] += α (r + γ max_a' Q[s',a'] − Q[s,a])`.

    Args:
        env (EscapeEnv).
        cfg (QLConfig).
    Returns: 
        Q (np.ndarray): shape (H*W,4).
        returns (np.ndarray[episodes]).
    """
    rng = np.random.default_rng(cfg.seed)

    h, w = env.h, env.w
    nS, nA = h * w, getattr(env.action_space, "n", 4)
    Q = np.zeros((nS, nA), dtype=np.float32)

    episodic_returns: List[float] = []

    out_dir = _ensure_out(cfg.out_dir)
    if out_dir and cfg.seed is not None:
        with open(os.path.join(out_dir, "seed.txt"), "w") as f:
            f.write(str(cfg.seed))

    for ep in range(cfg.episodes):
        epsilon = _sched_linear(cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_episodes, ep)
        alpha = _sched_linear(cfg.alpha_start, cfg.alpha_end, cfg.alpha_decay_episodes, ep)

        obs, info = env.reset()
        total = 0.0
        while True:
            if "pos" in info:
                s = _state_to_index(info["pos"], w)
            else:
                x, y = int(obs[0]), int(obs[1])
                s = _state_to_index((x, y), w)

            a = _epsilon_greedy(Q, s, epsilon, rng)
            obs, r, term, trunc, info = env.step(a)
            total += float(r)

            # next-state greedy target
            if "pos" in info:
                s_next = _state_to_index(info["pos"], w)
            else:
                x2, y2 = int(obs[0]), int(obs[1])
                s_next = _state_to_index((x2, y2), w)

            # Q-learning update: Q[s,a] ← Q[s,a] + α (r + γ max_a' Q[s',a'] − Q[s,a])
            td_target = float(r) + (0.0 if term or trunc else cfg.gamma * Q[s_next].max())
            Q[s, a] += alpha * (td_target - Q[s, a])

            if term or trunc:
                break

        episodic_returns.append(total)

        if (ep + 1) % max(1, cfg.episodes // 20) == 0:
            print(f"[Q] ep {ep + 1}/{cfg.episodes} | eps={epsilon:.3f} α={alpha:.3f} | G={total:.3f}")

    if out_dir:
        _save(out_dir, Q, episodic_returns, cfg)

    return Q, episodic_returns
